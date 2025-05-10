import logging
import os
from datetime import datetime

import numpy as np

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, max_exposure=0.2, max_leverage=2.0):
        self.max_exposure = (
            max_exposure  # Exposition maximale par actif (20% par défaut)
        )
        self.max_leverage = max_leverage
        self.exposure_history = []  # Pour le suivi et le debugging

    def check_exposure(self, portfolio, proposed_action):
        """Vérifie et ajuste l'action pour respecter les limites d'exposition"""
        current_exposure = portfolio.current_exposure()
        action_type, amount = proposed_action

        # Enregistrement pour l'analyse post-trade
        self.exposure_history.append(
            {
                "timestamp": datetime.now(),
                "current": current_exposure,
                "proposed": proposed_action,
            }
        )

        # Calcul de l'exposition potentielle après action
        if action_type == "buy":
            potential_exposure = current_exposure + amount
        elif action_type == "sell":
            potential_exposure = current_exposure - amount
        else:
            potential_exposure = current_exposure

        # Ajustement de l'action si nécessaire
        if potential_exposure > self.max_exposure:
            allowed_addition = self.max_exposure - current_exposure
            adjusted_amount = round(max(0, allowed_addition), 4)
            return ("hold", 0) if adjusted_amount == 0 else ("buy", adjusted_amount)

        if potential_exposure < -self.max_exposure:
            allowed_reduction = abs(-self.max_exposure - current_exposure)
            return ("sell", round(min(amount, allowed_reduction), 4))

        # Correction finale pour le blocage complet
        if action_type == "buy" and current_exposure >= self.max_exposure:
            return ("hold", 0)  # Blocage inconditionnel à la limite

        return proposed_action


class TradingEnvironment:
    def __init__(self, data=None, initial_balance=10000, risk_params=None):
        """Environnement de trading avec gestion des risques intégrée

        Args:
            data (DataFrame): Données de trading
            initial_balance (float): Solde initial
            risk_params (dict): Paramètres de risque (ex: {'max_exposure': 0.15})
        """
        # Ajouter dans __init__
        self.risk_manager = RiskManager(**(risk_params or {}))
        self.data = data
        self.current_step = 0
        self.prices = self.data["close"].tolist() if data is not None else []
        self.reward = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = type(
            "Portfolio", (), {"current_exposure": lambda: 0}
        )()  # Mock par défaut

    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode"""
        self.current_step = 0
        self.balance = self.initial_balance
        state = self._get_state()
        return state, {}  # Retourne l'état initial et un dict info vide

    def get_current_price(self):
        if self.current_step < len(self.prices):
            return self.prices[self.current_step]
        return self.prices[-1]  # Retourne le dernier prix disponible

    def _get_state(self):
        """Récupère l'état actuel de l'environnement"""
        if self.data is None or self.current_step >= len(self.data):
            return np.zeros(10)  # État par défaut

        # Extraction des caractéristiques pertinentes
        window = min(10, self.current_step + 1)
        df_window = self.data.iloc[
            self.current_step - window + 1 : self.current_step + 1
        ]

        # Caractéristiques de base
        state = []
        if not df_window.empty:
            # Prix normalisés
            close_prices = df_window["close"].values
            normalized_closes = (close_prices - np.mean(close_prices)) / (
                np.std(close_prices) + 1e-8
            )
            state.extend(normalized_closes)

            # Autres indicateurs si disponibles
            if "rsi" in df_window.columns:
                rsi = df_window["rsi"].values[-1] / 100.0  # Normalisation
                state.append(rsi)

            if "macd" in df_window.columns and "signal_line" in df_window.columns:
                macd = df_window["macd"].values[-1]
                signal = df_window["signal_line"].values[-1]
                state.append(
                    (macd - signal) / (abs(macd) + abs(signal) + 1e-8)
                )  # Normalisation

            # Ajouter des variables liées au portefeuille
            current_price = self.get_current_price()
            state.append(self.balance / self.initial_balance)  # Balance normalisée

            # Padding pour assurer une taille constante
            while len(state) < 10:
                state.append(0.0)

            # Limiter à 10 caractéristiques maximum
            state = state[:10]

        return np.array(state)

    def step(self, action):
        """Exécute une action dans l'environnement et retourne le nouvel état, la récompense, etc."""
        # Sauvegarder l'état précédent pour calculer la récompense
        previous_balance = self.balance

        # Traduire l'action numérique en action de trading
        action_map = {0: "hold", 1: "buy", 2: "sell"}
        action_type = action_map.get(action, "hold")
        amount = 0.1  # Montant fixe par défaut

        # Récupérer le prix actuel
        current_price = self.get_current_price()

        # Appliquer l'action
        if action_type == "buy":
            # Vérifier les limites d'exposition
            adjusted_action = self.risk_manager.check_exposure(
                self.portfolio, (action_type, amount)
            )
            if adjusted_action[0] == "buy":
                cost = current_price * adjusted_action[1]
                if cost <= self.balance:
                    self.balance -= cost
                    # Ici, mettre à jour le portefeuille

        elif action_type == "sell":
            # Vérifier les limites d'exposition
            adjusted_action = self.risk_manager.check_exposure(
                self.portfolio, (action_type, amount)
            )
            if adjusted_action[0] == "sell":
                profit = current_price * adjusted_action[1]
                self.balance += profit
                # Ici, mettre à jour le portefeuille

        # Avancer d'un pas
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        # Calculer la récompense
        new_balance = self.balance
        reward = (
            new_balance - previous_balance
        ) / previous_balance  # Rendement en pourcentage

        # Récupérer le nouvel état
        next_state = self._get_state()

        # Informations supplémentaires
        info = {
            "balance": self.balance,
            "price": current_price,
            "step": self.current_step,
            "action": action_type,
        }

        return next_state, reward, done, info

    def add_penalty(self, penalty_type):
        """Applique des pénalités pour risque"""
        if penalty_type == "exposure_limit_violation":
            self.reward -= 0.1  # Pénalité importante pour décourager les violations

    def log_violation_details(self, original, adjusted):
        """Journalise les détails des violations"""
        pass  # Implémentation minimale pour les tests


class RLAgent:
    """Classe encapsulant les agents d'apprentissage par renforcement pour le trading."""

    def __init__(self, model_dir="info_retour/models"):
        """
        Initialise l'agent de trading.

        Args:
            model_dir (str): Chemin vers le répertoire des modèles
        """
        self.model_dir = model_dir
        self.agent = None
        self.model_path = None
        self.env = None
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Agent RL initialisé avec répertoire de modèles: {model_dir}")

    def train(self, data, total_timesteps=10000, save_path=None):
        """
        Entraîne un agent RL sur les données fournies.

        Args:
            data (DataFrame): Données d'entraînement
            total_timesteps (int): Nombre total d'étapes d'entraînement
            save_path (str, optional): Chemin pour sauvegarder le modèle entraîné

        Returns:
            dict: Métriques d'entraînement
        """
        try:
            # Créer l'environnement avec les données
            self.env = TradingEnvironment(data)

            # Définir la taille de l'état et des actions
            state_size = len(self.env._get_state())
            action_size = 3  # hold, buy, sell

            # Utiliser soit l'agent existant, soit en créer un nouveau
            if self.agent is None:
                # Importation tardive pour éviter les dépendances circulaires
                try:
                    from ai_trading.rl.dqn_agent import DQNAgent

                    # Créer un nouvel agent DQN (à remplacer par un autre type d'agent si nécessaire)
                    self.agent = DQNAgent(
                        state_size=state_size,
                        action_size=action_size,
                        learning_rate=0.001,
                        gamma=0.95,
                        epsilon=1.0,
                        epsilon_decay=0.995,
                        epsilon_min=0.01,
                        batch_size=32,
                        memory_size=10000,
                        device="cpu",  # Utiliser "cuda" si disponible
                    )
                    logger.info(
                        f"Agent DQN créé avec state_size={state_size}, action_size={action_size}"
                    )
                except ImportError:
                    logger.error(
                        "Impossible d'importer DQNAgent, utilisation d'un agent fictif"
                    )
                    # Créer un agent fictif pour les tests
                    from collections import namedtuple

                    MockAgent = namedtuple("MockAgent", ["act", "learn"])
                    self.agent = MockAgent(
                        act=lambda state, use_epsilon=True: np.random.randint(0, 3),
                        learn=lambda: {"loss": 0.0},
                    )

            # Boucle d'entraînement
            logger.info(f"Début de l'entraînement pour {total_timesteps} étapes...")
            total_reward = 0
            episode_rewards = []
            losses = []

            for t in range(total_timesteps):
                if t % 1000 == 0:
                    logger.info(f"Étape {t}/{total_timesteps}")

                # Réinitialiser l'environnement si nécessaire
                if self.env.current_step >= len(data) - 1:
                    state, _ = self.env.reset()
                    episode_reward = 0
                else:
                    state = self.env._get_state()
                    episode_reward = 0

                # Récupérer une action de l'agent
                action = self.agent.act(state)

                # Exécuter l'action dans l'environnement
                next_state, reward, done, _ = self.env.step(action)

                # Stocker l'expérience dans la mémoire de l'agent
                if hasattr(self.agent, "remember"):
                    self.agent.remember(state, action, reward, next_state, done)

                # Apprendre à partir de l'expérience
                if hasattr(self.agent, "learn"):
                    metrics = self.agent.learn()
                    if metrics and "loss" in metrics:
                        losses.append(metrics["loss"])

                # Accumuler les récompenses
                episode_reward += reward

                if done:
                    episode_rewards.append(episode_reward)
                    logger.debug(f"Épisode terminé avec récompense: {episode_reward}")
                    state, _ = self.env.reset()
                    episode_reward = 0
                else:
                    state = next_state

            # Sauvegarder le modèle si demandé
            if save_path:
                self.save(save_path)
                logger.info(f"Modèle sauvegardé dans {save_path}")

            # Retourner les métriques d'entraînement
            metrics = {
                "total_episodes": len(episode_rewards),
                "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
                "avg_loss": np.mean(losses) if losses else 0,
            }

            logger.info(f"Entraînement terminé. Métriques: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            raise

    def predict(self, state):
        """
        Prédit une action basée sur l'état actuel.

        Args:
            state (numpy.array): État actuel

        Returns:
            int: Action prédite (0: hold, 1: buy, 2: sell)
        """
        if self.agent is None:
            logger.warning("Agent non initialisé. Retour de l'action par défaut (hold)")
            return 0  # Action par défaut: hold

        return self.agent.act(state, use_epsilon=False)

    def backtest(self, data):
        """
        Exécute un backtest du modèle sur les données historiques.

        Args:
            data (DataFrame): Données de backtest

        Returns:
            dict: Résultats du backtest
        """
        try:
            # Créer un environnement pour le backtest
            env = TradingEnvironment(data)

            # Initialiser des variables pour le suivi
            initial_balance = env.balance
            balances = [initial_balance]
            trades = []
            actions_history = []

            # Exécuter le backtest
            state, _ = env.reset()
            done = False

            while not done:
                # Prédire l'action avec l'agent
                action = self.predict(state)

                # Enregistrer l'action
                actions_history.append(action)

                # Exécuter l'action dans l'environnement
                next_state, reward, done, info = env.step(action)

                # Enregistrer le solde actuel
                balances.append(env.balance)

                # Enregistrer le trade si ce n'est pas "hold"
                if action != 0:  # Si ce n'est pas "hold"
                    trade_result = (env.balance - balances[-2]) / balances[-2]
                    trades.append(trade_result)

                # Passer à l'état suivant
                state = next_state

            # Calculer les métriques de performance
            final_balance = env.balance
            profit_pct = (final_balance - initial_balance) / initial_balance * 100

            # Calculer le profit buy & hold
            first_price = data["close"].iloc[0]
            last_price = data["close"].iloc[-1]
            bh_profit_pct = (last_price - first_price) / first_price * 100

            # Calculer le ratio de Sharpe (simplifié)
            if trades:
                returns = np.array(trades)
                sharpe_ratio = (
                    np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                )  # Annualisé
            else:
                sharpe_ratio = 0

            # Résultats du backtest
            results = {
                "initial_balance": initial_balance,
                "final_balance": final_balance,
                "profit_pct": profit_pct,
                "bh_profit_pct": bh_profit_pct,
                "sharpe_ratio": sharpe_ratio,
                "num_trades": len(trades),
                "trades": trades,
            }

            logger.info(
                f"Backtest terminé. Profit: {profit_pct:.2f}%, BH: {bh_profit_pct:.2f}%, Trades: {len(trades)}"
            )
            return results

        except Exception as e:
            logger.error(f"Erreur lors du backtest: {str(e)}")
            raise

    def load(self, model_path):
        """
        Charge un modèle pré-entraîné.

        Args:
            model_path (str): Chemin vers le modèle à charger
        """
        try:
            self.model_path = model_path

            # Vérifier si le fichier existe
            if not os.path.exists(model_path):
                logger.error(f"Modèle non trouvé: {model_path}")
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

            # Déterminer le type d'agent à partir du chemin du modèle
            if "dqn" in model_path.lower():
                # Importer DQNAgent seulement si nécessaire
                try:
                    from ai_trading.rl.dqn_agent import DQNAgent

                    # Créer un nouvel agent avec des valeurs par défaut
                    self.agent = DQNAgent(
                        state_size=10,  # Taille par défaut
                        action_size=3,  # hold, buy, sell
                        device="cpu",  # À configurer selon la disponibilité
                    )

                    # Charger les poids du modèle
                    self.agent.load(model_path)
                    logger.info(f"Agent DQN chargé depuis {model_path}")
                except ImportError:
                    logger.error("Impossible d'importer DQNAgent")
                    raise
            else:
                logger.warning(f"Type de modèle non reconnu: {model_path}")
                # Pour l'instant, supposer qu'il s'agit d'un DQNAgent
                try:
                    from ai_trading.rl.dqn_agent import DQNAgent

                    self.agent = DQNAgent(state_size=10, action_size=3, device="cpu")

                    self.agent.load(model_path)
                    logger.info(
                        f"Agent chargé par défaut en tant que DQN depuis {model_path}"
                    )
                except ImportError:
                    logger.error("Impossible d'importer DQNAgent")
                    raise

            # Créer un environnement fictif pour les prédictions si nécessaire
            if self.env is None:
                self.env = TradingEnvironment(data=None)

        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    def save(self, model_path):
        """
        Sauvegarde le modèle actuel.

        Args:
            model_path (str): Chemin pour sauvegarder le modèle
        """
        try:
            if self.agent is None:
                logger.error("Aucun agent à sauvegarder")
                raise ValueError("Aucun agent à sauvegarder")

            # Créer le répertoire si nécessaire
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

            # Sauvegarder le modèle en utilisant la méthode de l'agent
            if hasattr(self.agent, "save"):
                self.agent.save(model_path)
                logger.info(f"Modèle sauvegardé dans {model_path}")
            else:
                logger.error("L'agent ne possède pas de méthode de sauvegarde")
                raise NotImplementedError(
                    "L'agent ne possède pas de méthode de sauvegarde"
                )

            self.model_path = model_path

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            raise
