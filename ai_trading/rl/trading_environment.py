import datetime
import logging
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces

# Configuration du logger
logger = logging.getLogger("TradingEnvironment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Ajouter l'import
from ai_trading.rl.adaptive_normalization import AdaptiveNormalizer

# Ajouter l'import
from ai_trading.rl.risk_manager import RiskManager

# Ajouter l'import de la classe TechnicalIndicators
from .technical_indicators import TechnicalIndicators

# Utiliser directement VISUALIZATION_DIR de config.py
from ai_trading.config import VISUALIZATION_DIR

VISUALIZATION_DIR = VISUALIZATION_DIR / "trading_env"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


class TradingEnvironment(gym.Env):
    """
    Environnement de trading pour l'apprentissage par renforcement.

    Cet environnement simule un marché de trading avec des données réelles,
    où un agent peut acheter, vendre ou conserver des actifs.
    L'objectif est de maximiser la valeur du portefeuille.
    """

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        transaction_fee=0.001,
        window_size=20,
        include_position=True,
        include_balance=True,
        include_technical_indicators=True,
        risk_management=True,
        normalize_observation=True,
        reward_function="simple",  # Options: "simple", "sharpe", "transaction_penalty", "drawdown"
        risk_aversion=0.1,  # Paramètre pour le coefficient de risque dans la fonction de récompense
        transaction_penalty=0.001,  # Pénalité fixe pour chaque transaction
        lookback_window=20,  # Fenêtre pour calculer le ratio de Sharpe
        action_type="discrete",  # Type d'action: "discrete" ou "continuous"
        n_discrete_actions=5,  # Nombre d'actions discrètes par catégorie (achat/vente)
        slippage_model="constant",  # Options: "constant", "proportional", "dynamic"
        slippage_value=0.001,  # Valeur de slippage pour le modèle constant
        execution_delay=0,  # Délai d'exécution en pas de temps
        **kwargs,
    ):
        """
        Initialise l'environnement de trading.

        Args:
            df (pd.DataFrame): DataFrame contenant les données du marché
            initial_balance (float): Solde initial du portefeuille
            transaction_fee (float): Frais de transaction en pourcentage
            window_size (int): Taille de la fenêtre d'observation
            include_position (bool): Inclure la position actuelle dans l'observation
            include_balance (bool): Inclure le solde dans l'observation
            include_technical_indicators (bool): Inclure les indicateurs techniques dans l'observation
            risk_management (bool): Activer la gestion des risques
            normalize_observation (bool): Normaliser les observations
            reward_function (str): Fonction de récompense à utiliser
            risk_aversion (float): Coefficient de risque pour la fonction de récompense
            transaction_penalty (float): Pénalité fixe pour chaque transaction
            lookback_window (int): Fenêtre pour calculer le ratio de Sharpe
            action_type (str): Type d'action ("discrete" ou "continuous")
            n_discrete_actions (int): Nombre d'actions discrètes par catégorie
            slippage_model (str): Modèle de slippage
            slippage_value (float): Valeur de slippage
            execution_delay (int): Délai d'exécution en pas de temps
        """
        super(TradingEnvironment, self).__init__()

        # Valider les paramètres
        assert (
            len(df) > window_size
        ), f"Le DataFrame doit contenir plus de {window_size} points de données"
        assert initial_balance > 0, "Le solde initial doit être positif"
        assert (
            0 <= transaction_fee < 1
        ), "Les frais de transaction doivent être entre 0 et 1"
        assert reward_function in [
            "simple",
            "sharpe",
            "transaction_penalty",
            "drawdown",
        ], "Fonction de récompense invalide"
        assert action_type in [
            "discrete",
            "continuous",
        ], "Type d'action invalide, doit être 'discrete' ou 'continuous'"

        # Stocker les paramètres
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.include_position = include_position
        self.include_balance = include_balance
        self.include_technical_indicators = include_technical_indicators
        self.risk_management = risk_management
        self.use_risk_manager = risk_management  # Alias pour compatibilité
        self.normalize_observation = normalize_observation
        self.use_adaptive_normalization = (
            normalize_observation  # Alias pour compatibilité
        )
        self.reward_function = reward_function
        self.risk_aversion = risk_aversion
        self.transaction_penalty = transaction_penalty
        self.lookback_window = lookback_window
        self.action_type = action_type  # Stocker le type d'action
        self.n_discrete_actions = (
            n_discrete_actions  # Stocker le nombre d'actions discrètes
        )

        # Calculer les indicateurs techniques si nécessaire
        if self.include_technical_indicators:
            # RSI
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
            exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
            self.df['macd'] = exp1 - exp2

            # Bandes de Bollinger
            sma = self.df['close'].rolling(window=20).mean()
            std = self.df['close'].rolling(window=20).std()
            self.df['bollinger_middle'] = sma

            # Remplacer les NaN par des 0
            self.df.fillna(0, inplace=True)

        # Définir les colonnes de caractéristiques
        self.feature_columns = ["close"]
        if self.include_technical_indicators:
            self.feature_columns.extend(["rsi", "macd", "bollinger_middle"])

        # Ajouter les paramètres de marché réalistes
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        self.execution_delay = execution_delay
        self.pending_orders = []  # Liste des ordres en attente d'exécution

        # Variables supplémentaires pour le calcul des récompenses
        self.portfolio_value_history = []
        self.returns_history = []
        self.actions_history = []
        self.transaction_count = 0
        self.last_transaction_step = -1
        self.max_portfolio_value = 0

        # Définir l'espace d'action selon le type
        if action_type == "discrete":
            # Actions: 0 (ne rien faire), 1-n (acheter x%), n+1-2n (vendre x%)
            # Exemple avec n_discrete_actions=5:
            # 0: ne rien faire
            # 1-5: acheter 20%, 40%, 60%, 80%, 100% du solde disponible
            # 6-10: vendre 20%, 40%, 60%, 80%, 100% des crypto détenues
            self.action_space = spaces.Discrete(1 + 2 * n_discrete_actions)
        elif action_type == "continuous":
            # Action continue entre -1 et 1
            # -1: vendre 100%, -0.5: vendre 50%, 0: ne rien faire, 0.5: acheter 50%, 1: acheter 100%
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            raise ValueError(f"Type d'action non supporté: {action_type}")

        # Réinitialiser l'environnement pour calculer la taille réelle de l'état
        temp_reset = self.reset()
        if isinstance(temp_reset, tuple):
            temp_state = temp_reset[
                0
            ]  # Pour la compatibilité avec les nouvelles versions de gym
        else:
            temp_state = temp_reset

        # Définir l'espace d'observation avec la taille réelle de l'état
        real_state_size = temp_state.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(real_state_size,), dtype=np.float32
        )

        logger.info(
            f"Environnement de trading initialisé avec {len(df)} points de données et espace d'action {action_type}"
        )

    def _build_observation_space(self):
        """Construit l'espace d'observation."""
        # Calcul correct du nombre de caractéristiques
        n_features = self.window_size + 1  # Historique des prix (close)
        n_features += 2  # Crypto détenue + solde

        if self.include_technical_indicators:
            n_features += self.window_size * 3  # RSI, MACD, BB

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement à l'état initial.

        Args:
            seed: Graine aléatoire pour la reproductibilité
            options: Options supplémentaires pour la réinitialisation

        Returns:
            observation: L'observation initiale
            info: Informations supplémentaires
        """
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.max_portfolio_value = self.initial_balance
        self.portfolio_value_history = [self.initial_balance]
        self.returns_history = []
        self.actions_history = []
        self.transaction_count = 0
        self.last_transaction_step = -1

        if self.risk_management:
            self.risk_manager = RiskManager()

        if self.normalize_observation and self.include_technical_indicators:
            self.normalizer = AdaptiveNormalizer()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """
        Exécute une action dans l'environnement avec délai d'exécution.

        Args:
            action: Action à exécuter (discrète ou continue selon action_type)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Vérifier que l'action est valide
        if self.action_type == "continuous":
            if isinstance(action, np.ndarray):
                # Convertir l'action en float pour l'espace continu
                action_value = float(action[0])
            else:
                action_value = float(action)

            # Créer un tableau numpy pour la vérification de l'espace d'action
            action_for_check = np.array([action_value], dtype=np.float32)

            if not self.action_space.contains(action_for_check):
                raise ValueError(
                    f"Action invalide: {action_value}, doit être entre -1 et 1"
                )
        else:
            if not self.action_space.contains(action):
                raise ValueError(f"Action invalide: {action}")

        # Initialiser le dictionnaire d'informations dès le début
        info = {"action_adjusted": False}  # Par défaut, l'action n'est pas ajustée

        # Sauvegarder l'état précédent pour calculer la récompense
        previous_portfolio_value = self.get_portfolio_value()

        # Appliquer le gestionnaire de risque si activé
        if self.use_risk_manager:
            adjusted_action = self.risk_manager.adjust_action(
                action,
                self.portfolio_value_history[-1],
                self.crypto_held,
                current_price=(
                    self.df.iloc[self.current_step]["close"] if not self.df.empty else 0
                ),
            )
            if adjusted_action != action:
                action = adjusted_action
                info["action_adjusted"] = True

        # Traiter les ordres en attente
        self._process_pending_orders()

        # Appliquer l'action avec délai si nécessaire
        if self.execution_delay > 0:
            current_price = self.df.iloc[self.current_step]["close"]
            
            if self.action_type == "discrete":
                action_value = self._get_action_value(action)
            else:
                action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
                
            # Calculer la quantité à trader
            if action_value > 0:  # Achat
                amount = (self.balance * abs(action_value)) / current_price
            else:  # Vente
                amount = self.crypto_held * abs(action_value)
                
            # Ajouter l'ordre à la liste des ordres en attente
            self.pending_orders.append({
                "action_value": action_value,
                "amount": amount,
                "delay": self.execution_delay
            })
        else:
            # Exécution immédiate sans délai
            if self.action_type == "discrete":
                self._apply_discrete_action(action)
            else:
                self._apply_continuous_action(action)

        # Passer à l'étape suivante
        self.current_step += 1

        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.df) - 1

        # Calculer la récompense
        current_portfolio_value = self.get_portfolio_value()
        reward = self._calculate_reward(action)

        # Enregistrer la valeur du portefeuille
        self.portfolio_value_history.append(current_portfolio_value)

        # Construire l'observation
        observation = self._get_observation()

        # Mettre à jour les informations à la fin
        current_price = self.df.iloc[self.current_step]["close"]
        info.update(
            {
                "portfolio_value": self.get_portfolio_value(),
                "balance": self.balance,
                "crypto_held": self.crypto_held,
                "current_price": current_price,
            }
        )

        # Vérifier les conditions de stop-loss pour les positions ouvertes
        if self.use_risk_manager and self.crypto_held > 0:
            position_id = f"position_{self.current_step}"

            # Mettre à jour le trailing stop si nécessaire
            if self.crypto_held > 0:
                self.risk_manager.update_trailing_stop(
                    position_id,
                    current_price,
                    self.df.iloc[self.current_step]["close"],
                    "long",
                )
            else:
                self.risk_manager.update_trailing_stop(
                    position_id,
                    current_price,
                    self.df.iloc[self.current_step]["close"],
                    "short",
                )

            # Vérifier si un stop est déclenché
            stop_result = self.risk_manager.check_stop_conditions(
                position_id, current_price, "long"
            )

            if stop_result["stop_triggered"]:
                # Fermer la position au prix du stop
                stop_price = stop_result["stop_price"]
                self._close_position(stop_price)
                logger.info(f"{stop_result['stop_type']} déclenché à {stop_price}")

        return observation, reward, done, False, info

    def _apply_slippage(self, price, action_value):
        """
        Applique le slippage au prix d'exécution.
        
        Args:
            price: Prix de base
            action_value: Valeur de l'action (positive pour achat, négative pour vente)
            
        Returns:
            float: Prix avec slippage
        """
        if self.slippage_model == "constant":
            slippage = self.slippage_value
        elif self.slippage_model == "proportional":
            slippage = self.slippage_value * abs(action_value)
        elif self.slippage_model == "dynamic":
            # Slippage dynamique basé sur le volume et la volatilité
            volatility = self.df.iloc[self.current_step]["volatility"] if "volatility" in self.df.columns else 0.01
            current_volume = self.df.iloc[self.current_step]["volume"]
            avg_volume = self.df["volume"].iloc[max(0, self.current_step-20):self.current_step+1].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            slippage = self.slippage_value * (1 + volatility) * volume_ratio
        else:
            slippage = 0
        
        # Appliquer le slippage dans la direction de l'action
        if action_value > 0:  # Achat
            return price * (1 + slippage)
        else:  # Vente
            return price * (1 - slippage)

    def _process_pending_orders(self):
        """
        Traite les ordres en attente d'exécution.
        """
        current_price = self.df.iloc[self.current_step]["close"]
        executed_orders = []
        
        for order in self.pending_orders:
            order["delay"] -= 1
            if order["delay"] <= 0:
                # Exécuter l'ordre
                price_with_slippage = self._apply_slippage(current_price, order["action_value"])
                
                if order["action_value"] > 0:  # Achat
                    self.balance -= order["amount"] * price_with_slippage * (1 + self.transaction_fee)
                    self.crypto_held += order["amount"]
                else:  # Vente
                    self.balance += order["amount"] * price_with_slippage * (1 - self.transaction_fee)
                    self.crypto_held -= order["amount"]
                
                executed_orders.append(order)
        
        # Retirer les ordres exécutés
        self.pending_orders = [order for order in self.pending_orders if order not in executed_orders]

    def _apply_discrete_action(self, action):
        """
        Applique une action discrète.

        Args:
            action (int): Indice de l'action à appliquer
        """
        # Obtenir le prix actuel
        current_price = self.df.iloc[self.current_step]["close"]

        if action == 0:  # Ne rien faire
            logger.debug("Action: HOLD")
            return

        # Calculer le pourcentage d'achat/vente
        if 1 <= action <= self.n_discrete_actions:  # Achat
            # Calculer le pourcentage d'achat (1/n, 2/n, ..., n/n)
            buy_percentage = action / self.n_discrete_actions

            # Limiter l'achat à 30% du portefeuille total
            portfolio_value = self.get_portfolio_value()
            max_buy_value = portfolio_value * 0.3

            # Calculer la valeur d'achat basée sur le pourcentage
            buy_value = self.balance * buy_percentage

            # Appliquer la limite de 30%
            buy_value = min(buy_value, max_buy_value)

            # Calculer la quantité de crypto à acheter
            max_crypto_to_buy = buy_value / (current_price * (1 + self.transaction_fee))

            # Acheter la quantité calculée
            self.crypto_held += max_crypto_to_buy
            self.balance -= (
                max_crypto_to_buy * current_price * (1 + self.transaction_fee)
            )

            logger.debug(
                f"Achat: {max_crypto_to_buy:.6f} unités à ${current_price:.2f} (limité à 30% du portefeuille)"
            )

        elif self.n_discrete_actions < action <= 2 * self.n_discrete_actions:  # Vente
            if self.crypto_held > 0:
                # Calculer le pourcentage de vente (1/n, 2/n, ..., n/n)
                sell_percentage = (
                    action - self.n_discrete_actions
                ) / self.n_discrete_actions
                crypto_to_sell = self.crypto_held * sell_percentage

                # Vendre la quantité calculée
                self.balance += (
                    crypto_to_sell * current_price * (1 - self.transaction_fee)
                )

                logger.debug(
                    f"Vente: {crypto_to_sell:.6f} unités ({sell_percentage*100:.0f}%) à ${current_price:.2f}"
                )

                self.crypto_held -= crypto_to_sell
            else:
                logger.debug("Tentative de vente sans crypto détenue")

    def _apply_continuous_action(self, action):
        """
        Applique une action continue.

        Args:
            action (float): Valeur de l'action entre -1 et 1
        """
        # Obtenir le prix actuel
        current_price = self.df.iloc[self.current_step]["close"]

        # Extraire la valeur scalaire de l'action numpy
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # Zone neutre autour de 0 pour éviter des micro-transactions
        if -0.05 <= action_value <= 0.05:
            logger.debug("Action: HOLD (zone neutre)")
            return

        if action_value > 0:  # Achat
            buy_percentage = action_value

            # Limiter l'achat à 30% du portefeuille total
            portfolio_value = self.get_portfolio_value()
            max_buy_value = portfolio_value * 0.3

            # Calculer la valeur d'achat basée sur le pourcentage
            buy_value = self.balance * buy_percentage

            # Appliquer la limite de 30%
            buy_value = min(buy_value, max_buy_value)

            # Calculer la quantité de crypto à acheter
            max_crypto_to_buy = buy_value / (current_price * (1 + self.transaction_fee))

            # Acheter la quantité calculée
            self.crypto_held += max_crypto_to_buy
            self.balance -= (
                max_crypto_to_buy * current_price * (1 + self.transaction_fee)
            )

            logger.debug(
                f"Achat: {max_crypto_to_buy:.6f} unités ({buy_percentage*100:.0f}%) à ${current_price:.2f} (limité à 30% du portefeuille)"
            )

        else:  # Vente (action_value < 0)
            if self.crypto_held > 0:
                sell_percentage = -action_value
                crypto_to_sell = self.crypto_held * sell_percentage

                # Vendre la quantité calculée
                self.balance += (
                    crypto_to_sell * current_price * (1 - self.transaction_fee)
                )

                logger.debug(
                    f"Vente: {crypto_to_sell:.6f} unités ({sell_percentage*100:.0f}%) à ${current_price:.2f}"
                )

                self.crypto_held -= crypto_to_sell
            else:
                logger.debug("Tentative de vente sans crypto détenue")

    def _get_observation(self):
        """Retourne l'observation actuelle de l'environnement."""
        if self.current_step < self.window_size:
            # Si on n'a pas assez de données, on remplit avec des zéros
            window_data = np.zeros((self.window_size, self.observation_space.shape[1]))
        else:
            # Récupérer les données de la fenêtre
            window_data = self.df.iloc[
                self.current_step - self.window_size : self.current_step
            ][self.feature_columns].values

        # Nettoyer les valeurs NaN
        window_data = np.nan_to_num(window_data, nan=0.0)

        # Normaliser les données si nécessaire
        if self.normalize_observation:
            # Calculer les statistiques sur la fenêtre
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0)
            # Éviter la division par zéro
            std = np.where(std == 0, 1, std)
            window_data = (window_data - mean) / std

        # Ajouter la position actuelle si nécessaire
        if self.include_position:
            position = np.full((self.window_size, 1), self.crypto_held)
            window_data = np.concatenate([window_data, position], axis=1)

        # Ajouter le solde si nécessaire
        if self.include_balance:
            balance = np.full((self.window_size, 1), self.balance)
            window_data = np.concatenate([window_data, balance], axis=1)

        # S'assurer que les données sont dans le bon format
        window_data = window_data.astype(np.float32)
        
        return window_data

    def render(self, mode="human"):
        """
        Affiche l'état actuel de l'environnement pour visualisation.
        """
        if self.current_step >= len(self.df):
            return

        fig = plt.figure(figsize=(16, 8))

        # Sous-graphique pour le prix et les actions
        price_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        action_ax = plt.subplot2grid((4, 1), (2, 0), rowspan=1, sharex=price_ax)
        portfolio_ax = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=price_ax)

        # Tracer le prix
        price_subset = self.df.iloc[
            max(0, self.current_step - 30) : self.current_step + 1
        ]
        price_ax.plot(price_subset.index, price_subset["close"], "b-")
        price_ax.set_title(f"Prix {self.df.columns[0]} - Étape {self.current_step}")

        # Tracer les indicateurs techniques si activés
        if self.include_technical_indicators and hasattr(self, "technical_indicators"):
            for indicator in self.technical_indicators:
                if indicator in self.df.columns:
                    price_ax.plot(
                        price_subset.index,
                        price_subset[indicator],
                        alpha=0.7,
                        label=indicator,
                    )
            price_ax.legend(loc="upper left")

        # Tracer les actions
        action_colors = {0: "gray", 1: "green", 2: "red"}  # Hold, Buy, Sell
        actions = self.actions_history[-30:] if len(self.actions_history) > 0 else []
        if actions:
            action_indices = price_subset.index[-len(actions) :]
            for i, action in enumerate(actions):
                if i < len(
                    action_indices
                ):  # Assurer que nous avons un indice correspondant
                    action_ax.bar(
                        action_indices[i], 1, color=action_colors.get(action, "gray")
                    )
        action_ax.set_title("Actions (Gris=Hold, Vert=Achat, Rouge=Vente)")
        action_ax.set_yticks([])

        # Tracer la valeur du portefeuille
        portfolio_values = (
            self.portfolio_value_history[-30:]
            if len(self.portfolio_value_history) > 0
            else []
        )
        if portfolio_values:
            portfolio_indices = price_subset.index[-len(portfolio_values) :]
            portfolio_ax.plot(portfolio_indices, portfolio_values, "g-")
        portfolio_ax.set_title(
            f"Valeur du portefeuille: ${self.get_portfolio_value():.2f}"
        )

        # Formater les dates si l'index est un DatetimeIndex
        if isinstance(self.df.index, pd.DatetimeIndex):
            for ax in [price_ax, action_ax, portfolio_ax]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(
                    mdates.WeekdayLocator(interval=max(1, len(price_subset) // 5))
                )

        plt.tight_layout()

        # Sauvegarder le graphique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step_{self.current_step:04d}"
        filename = f"trading_env_{step_str}_{timestamp}.png"
        output_path = VISUALIZATION_DIR / filename
        plt.savefig(output_path)

        if mode == "human":
            plt.pause(0.01)
        else:
            plt.close()

        return output_path

    def get_portfolio_value(self):
        """
        Retourne la valeur actuelle du portefeuille.

        Returns:
            float: Valeur du portefeuille
        """
        current_price = self.df.iloc[self.current_step]["close"]
        return self.balance + self.crypto_held * current_price

    def get_portfolio_history(self):
        """
        Retourne l'historique de la valeur du portefeuille.

        Returns:
            list: Historique des valeurs du portefeuille
        """
        return self.portfolio_value_history

    def get_portfolio_value_history(self):
        """
        Alias pour get_portfolio_history().
        Retourne l'historique de la valeur du portefeuille.

        Returns:
            list: Historique des valeurs du portefeuille
        """
        return self.portfolio_value_history

    def _calculate_reward(self, action=None):
        """
        Calcule la récompense basée sur la valeur actuelle du portefeuille et la politique de récompense.

        Args:
            action: L'action qui a été prise (pour les pénalités liées aux actions)

        Returns:
            float: La récompense calculée
        """
        # Calculer la valeur actuelle du portefeuille
        current_price = self.df.iloc[self.current_step]["close"]
        current_value = self.balance + self.crypto_held * current_price

        # Enregistrer la valeur du portefeuille
        self.portfolio_value_history.append(current_value)

        # Mettre à jour la valeur maximale du portefeuille pour le calcul du drawdown
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value

        # Si c'est le premier pas ou si nous n'avons pas d'historique suffisant, retourner une récompense nulle
        if len(self.portfolio_value_history) < 2:
            return 0.0

        # Calculer le retour en pourcentage
        previous_value = self.portfolio_value_history[-2]
        pct_change = (
            (current_value - previous_value) / previous_value
            if previous_value > 0
            else 0
        )
        self.returns_history.append(pct_change)

        # Choisir la fonction de récompense appropriée
        if self.reward_function == "simple":
            reward = self._simple_reward(pct_change)
        elif self.reward_function == "sharpe":
            reward = self._sharpe_reward()
        elif self.reward_function == "transaction_penalty":
            reward = self._transaction_penalty_reward(pct_change, action)
        elif self.reward_function == "drawdown":
            reward = self._drawdown_reward(pct_change)
        else:
            reward = self._simple_reward(pct_change)

        return reward

    def _simple_reward(self, pct_change):
        """
        Fonction de récompense simple basée sur le changement de valeur du portefeuille.

        Args:
            pct_change: Changement en pourcentage de la valeur du portefeuille

        Returns:
            float: La récompense calculée
        """
        # Récompense de base: changement en pourcentage de la valeur du portefeuille
        reward = pct_change

        # Ajout de bonus/malus basés sur la tendance récente
        if len(self.returns_history) >= 3:
            # Bonus pour une tendance positive constante
            if all(r > 0 for r in self.returns_history[-3:]):
                reward *= 1.1  # Bonus de 10%

            # Malus pour une tendance négative constante
            elif all(r < 0 for r in self.returns_history[-3:]):
                reward *= 0.9  # Malus de 10%

        return reward

    def _sharpe_reward(self):
        """
        Fonction de récompense basée sur le ratio de Sharpe.

        Returns:
            float: La récompense calculée basée sur le ratio de Sharpe
        """
        # Vérifier que nous avons suffisamment d'historique pour calculer le ratio de Sharpe
        if len(self.returns_history) < self.lookback_window:
            return 0.0

        # Calculer le ratio de Sharpe sur la fenêtre de lookback
        returns = np.array(self.returns_history[-self.lookback_window :])

        # Éviter les divisions par zéro
        if np.std(returns) == 0:
            if np.mean(returns) > 0:
                return 1.0  # Récompense positive si les rendements sont positifs mais constants
            elif np.mean(returns) < 0:
                return (
                    -1.0
                )  # Récompense négative si les rendements sont négatifs mais constants
            else:
                return 0.0  # Pas de récompense si les rendements sont tous nuls

        # Calculer le ratio de Sharpe (version simplifiée sans taux sans risque)
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
        )  # Annualisé (252 jours de trading)

        # Normaliser la récompense pour éviter les valeurs extrêmes
        reward = np.clip(sharpe_ratio, -10, 10)

        return reward

    def _transaction_penalty_reward(self, pct_change, action):
        """
        Fonction de récompense avec pénalité pour les transactions fréquentes.

        Args:
            pct_change: Changement en pourcentage de la valeur du portefeuille
            action: L'action qui a été prise

        Returns:
            float: La récompense calculée avec pénalité pour les transactions
        """
        # Récompense de base
        reward = pct_change

        # Appliquer une pénalité si une transaction a été effectuée (action 1=achat ou 2=vente)
        if action in [1, 2]:
            # Incrémenter le compteur de transactions
            self.transaction_count += 1

            # Calculer la pénalité basée sur la fréquence des transactions
            steps_since_last_transaction = (
                self.current_step - self.last_transaction_step
                if self.last_transaction_step > 0
                else self.window_size
            )

            # Plus la transaction est proche de la précédente, plus la pénalité est élevée
            frequency_penalty = self.transaction_penalty * (
                1.0 / max(1, steps_since_last_transaction)
            )

            # Appliquer la pénalité
            reward -= frequency_penalty

            # Mettre à jour le pas de la dernière transaction
            self.last_transaction_step = self.current_step

        return reward

    def _drawdown_reward(self, pct_change):
        """
        Fonction de récompense qui pénalise les drawdowns importants.

        Args:
            pct_change: Changement en pourcentage de la valeur du portefeuille

        Returns:
            float: La récompense calculée avec pénalité pour les drawdowns
        """
        # Récompense de base
        reward = pct_change

        # Calculer le drawdown actuel
        current_value = self.portfolio_value_history[-1]
        drawdown = (
            (self.max_portfolio_value - current_value) / self.max_portfolio_value
            if self.max_portfolio_value > 0
            else 0
        )

        # Pénaliser les drawdowns importants
        drawdown_penalty = (
            self.risk_aversion * drawdown * drawdown
        )  # Pénalité quadratique

        # Appliquer la pénalité
        reward -= drawdown_penalty

        return reward

    def visualize_technical_indicators(self, window_size=100):
        """
        Visualise les indicateurs techniques utilisés dans l'environnement.

        Args:
            window_size: Nombre de périodes à afficher
        """
        if not self.include_technical_indicators:
            logger.warning(
                "Les indicateurs techniques ne sont pas activés dans cet environnement."
            )
            return

        start_idx = max(0, self.current_step - window_size)
        end_idx = min(self.current_step + 1, len(self.df))
        subset = self.df.iloc[start_idx:end_idx]

        # Organiser les indicateurs par type
        trend_indicators = ["sma", "ema", "wma", "macd", "macd_signal", "macd_hist"]
        oscillator_indicators = ["rsi", "stoch_k", "stoch_d", "cci", "williams_r"]
        volatility_indicators = [
            "atr",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
        ]
        volume_indicators = ["obv", "volume"]

        # Créer un graphique avec sous-graphiques pour chaque type d'indicateur
        fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

        # Prix (avec quelques indicateurs de tendance superposés)
        axs[0].plot(subset.index, subset["close"], "k-", label="Prix")
        for ind in trend_indicators:
            if ind in subset.columns:
                axs[0].plot(subset.index, subset[ind], alpha=0.7, label=ind)
        axs[0].set_title("Prix et indicateurs de tendance")
        axs[0].legend(loc="upper left")

        # Bandes de Bollinger (si disponibles)
        if "bollinger_upper" in subset.columns and "bollinger_lower" in subset.columns:
            axs[1].plot(subset.index, subset["close"], "k-", label="Prix")
            axs[1].plot(
                subset.index,
                subset["bollinger_upper"],
                "r-",
                alpha=0.5,
                label="BB Upper",
            )
            axs[1].plot(
                subset.index,
                subset["bollinger_middle"],
                "g--",
                alpha=0.5,
                label="BB Middle",
            )
            axs[1].plot(
                subset.index,
                subset["bollinger_lower"],
                "b-",
                alpha=0.5,
                label="BB Lower",
            )
            axs[1].fill_between(
                subset.index,
                subset["bollinger_upper"],
                subset["bollinger_lower"],
                color="gray",
                alpha=0.2,
            )
            axs[1].set_title("Bandes de Bollinger")
            axs[1].legend(loc="upper left")

        # Oscillateurs
        osc_plotted = False
        for ind in oscillator_indicators:
            if ind in subset.columns:
                axs[2].plot(subset.index, subset[ind], label=ind)
                osc_plotted = True
        if osc_plotted:
            axs[2].set_title("Oscillateurs")
            # Ajouter des lignes horizontales pour les niveaux courants
            if "rsi" in subset.columns:
                axs[2].axhline(y=70, color="r", linestyle="-", alpha=0.3)
                axs[2].axhline(y=30, color="g", linestyle="-", alpha=0.3)
            if "stoch_k" in subset.columns:
                axs[2].axhline(y=80, color="r", linestyle="--", alpha=0.3)
                axs[2].axhline(y=20, color="g", linestyle="--", alpha=0.3)
            axs[2].legend(loc="upper left")
        else:
            axs[2].set_visible(False)

        # Indicateurs de volatilité
        vol_plotted = False
        for ind in volatility_indicators:
            if ind == "atr" and ind in subset.columns:
                axs[3].plot(subset.index, subset[ind], label=ind)
                vol_plotted = True
        if vol_plotted:
            axs[3].set_title("Indicateurs de volatilité (ATR)")
            axs[3].legend(loc="upper left")
        else:
            axs[3].set_visible(False)

        # Volume
        if "volume" in subset.columns:
            axs[4].bar(
                subset.index, subset["volume"], color="b", alpha=0.5, label="Volume"
            )
            axs[4].set_title("Volume")
        else:
            axs[4].set_visible(False)

        plt.tight_layout()

        # Sauvegarder le graphique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"technical_indicators_{timestamp}.png"
        output_path = VISUALIZATION_DIR / filename
        plt.savefig(output_path)
        plt.close()

        return output_path

    def _close_position(self, price):
        # ...

        # Supprimer les stop-loss et take-profit
        if self.use_risk_manager:
            position_id = f"position_{self.current_step}"
            self.risk_manager.clear_position(position_id)

        # ...

    def _get_action_value(self, action):
        """
        Convertit une action discrète en valeur continue.
        
        Args:
            action (int): Action discrète (0: hold, 1-n: buy x%, n+1-2n: sell x%)
            
        Returns:
            float: Valeur continue entre -1 et 1
        """
        if action == 0:  # Hold
            return 0.0
        elif 1 <= action <= self.n_discrete_actions:  # Buy
            return action / self.n_discrete_actions
        else:  # Sell
            sell_action = action - self.n_discrete_actions
            return -sell_action / self.n_discrete_actions
