"""
Module pour le système de trading basé sur l'apprentissage par renforcement.
"""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.models.temporal_transformer import FinancialTemporalTransformer
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logger = logging.getLogger("RLTradingSystem")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RLTradingSystem:
    """Système de trading basé sur l'apprentissage par renforcement."""

    def __init__(self, config=None):
        """
        Initialise le système de trading RL.

        Args:
            config (dict): Configuration du système
        """
        self.config = config or {}
        self._env = None
        self._agent = None
        self._transformer = None
        self.logger = logging.getLogger(__name__)
        self.data_integrator = RLDataIntegrator()

    def create_agent(self, agent_type="dqn", state_size=None, action_size=3, **kwargs):
        """
        Crée un agent d'apprentissage par renforcement.

        Args:
            agent_type (str): Type d'agent ('dqn', 'ppo', 'sac')
            state_size (int): Taille de l'espace d'état
            action_size (int): Taille de l'espace d'action
            **kwargs: Arguments supplémentaires pour l'agent

        Returns:
            Agent: Agent d'apprentissage par renforcement
        """
        if state_size is None and self._env is not None:
            # Déterminer la taille de l'état à partir de l'environnement
            state_size = self._env.observation_space.shape[0]

        if agent_type.lower() == "dqn":
            from ai_trading.rl.dqn_agent import DQNAgent

            agent = DQNAgent(state_size=state_size, action_size=action_size, **kwargs)
        elif agent_type.lower() == "ppo":
            # Implémentation future pour PPO
            raise NotImplementedError("Agent PPO non implémenté")
        elif agent_type.lower() == "sac":
            from ai_trading.rl.agents.sac_agent import SACAgent

            # Configurer les bornes d'action correctes si l'environnement existe
            action_bounds = kwargs.pop("action_bounds", (-1, 1))

            # Si l'environnement existe, on peut vérifier si l'espace d'action est continu
            if self._env is not None:
                if (
                    hasattr(self._env, "action_type")
                    and self._env.action_type == "continuous"
                ):
                    # Pour un espace Box, l'action_size devrait être 1 pour notre environnement
                    action_size = 1
                    logger.info(
                        f"Agent SAC créé pour un espace d'action continu de taille {action_size}"
                    )
                else:
                    logger.warning(
                        "L'agent SAC est optimisé pour les actions continues, mais l'environnement n'a pas d'espace d'action continu."
                    )

            agent = SACAgent(
                state_size=state_size,
                action_size=action_size,
                action_bounds=action_bounds,
                **kwargs,
            )
        else:
            raise ValueError(f"Type d'agent inconnu: {agent_type}")

        return agent

    def create_environment(self, data=None, **kwargs):
        """
        Crée un environnement de trading.

        Args:
            data (DataFrame): Données de marché
            **kwargs: Arguments supplémentaires pour l'environnement

        Returns:
            TradingEnvironment: Environnement de trading
        """
        if data is None:
            # Générer des données synthétiques
            data = self.data_integrator.generate_synthetic_data(
                n_samples=100, trend="bullish", volatility=0.02, with_sentiment=True
            )

        # Créer l'environnement
        self._env = TradingEnvironment(df=data, **kwargs)

        logger.info(f"Environnement de trading créé avec {len(data)} points de données")
        return self._env

    def train(self, agent=None, episodes=50, batch_size=32, save_path=None, data=None):
        """
        Entraîne l'agent sur l'environnement spécifié.

        Args:
            agent: Agent à entraîner (optionnel, utilisera self._agent si non spécifié)
            episodes (int): Nombre d'épisodes d'entraînement
            batch_size (int): Taille du batch pour l'entraînement
            save_path (str): Chemin pour sauvegarder l'agent (optionnel)
            data (pd.DataFrame): Données d'entraînement (optionnel)
        """
        # Créer l'environnement si nécessaire
        if self._env is None and data is not None:
            self.create_environment(data=data)
        elif self._env is None:
            raise ValueError(
                "Aucun environnement spécifié. Fournissez des données ou appelez create_environment() d'abord."
            )

        # Utiliser l'agent fourni ou celui déjà créé
        if agent:
            self._agent = agent
        elif not self._agent:
            raise ValueError("Aucun agent spécifié. Appelez create_agent() d'abord.")

        logger.info(f"Début de l'entraînement pour {episodes} épisodes...")

        for episode in range(episodes):
            state = self._env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Prendre seulement l'état, pas les infos
            done = False
            total_reward = 0

            while not done:
                action = self._agent.act(state)
                next_state, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                self._agent.remember(state, action, reward, next_state, done)
                self._agent.replay(batch_size)
                state = next_state
                total_reward += reward

            if episode % 10 == 0:
                self.logger.info(
                    f"Épisode {episode}/{episodes}, Récompense totale: {total_reward}"
                )

        if save_path:
            self._agent.save(save_path)
            self.logger.info(f"Agent sauvegardé dans {save_path}")

    def evaluate(
        self,
        agent=None,
        env=None,
        num_episodes=1,
        test_data=None,
        visualize=False,
        save_dir=None,
    ):
        """
        Évalue l'agent sur l'environnement.

        Args:
            agent: Agent d'apprentissage par renforcement
            env: Environnement de trading
            num_episodes (int): Nombre d'épisodes d'évaluation
            test_data (DataFrame): Données de test (si différentes de celles de l'environnement)
            visualize (bool): Afficher les visualisations
            save_dir (str): Répertoire pour sauvegarder les visualisations

        Returns:
            dict: Résultats de l'évaluation
        """
        if agent is None:
            if self._agent is None:
                raise ValueError("Aucun agent spécifié")
            agent = self._agent

        if env is None:
            if self._env is None:
                raise ValueError("Aucun environnement spécifié")
            env = self._env

        # Si des données de test sont fournies, créer un nouvel environnement
        if test_data is not None:
            env = self.create_environment(data=test_data)

        # Initialiser les résultats
        results = {
            "final_value": 0.0,
            "returns": [],
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "portfolio_history": [],
            "actions": [],
            "rewards": [],
        }

        # Évaluer l'agent
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Prendre seulement l'état, pas les infos
            done = False
            episode_reward = 0
            episode_actions = []
            episode_portfolio = []

            while not done:
                # Obtenir l'action de l'agent
                action = agent.act(state)
                episode_actions.append(action)

                # Exécuter l'action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_portfolio.append(info.get("portfolio_value", 0.0))

                # Mettre à jour l'état
                state = next_state

            # Mettre à jour les résultats
            results["rewards"].append(episode_reward)
            results["actions"].extend(episode_actions)
            results["portfolio_history"].extend(episode_portfolio)

        # Calculer les métriques finales
        if results["portfolio_history"]:
            initial_value = results["portfolio_history"][0]
            final_value = results["portfolio_history"][-1]
            results["final_value"] = final_value
            results["returns"] = [
                (p - initial_value) / initial_value
                for p in results["portfolio_history"]
            ]
            results["sharpe_ratio"] = np.mean(results["returns"]) / (
                np.std(results["returns"]) + 1e-8
            )
            results["max_drawdown"] = np.min(
                [
                    (p - max(results["portfolio_history"][: i + 1]))
                    / max(results["portfolio_history"][: i + 1])
                    for i, p in enumerate(results["portfolio_history"])
                ]
            )

        # Visualiser les résultats si demandé
        if visualize and save_dir:
            from ai_trading.rl.evaluation import PerformanceVisualizer

            visualizer = PerformanceVisualizer(save_dir=save_dir)
            visualizer.create_performance_dashboard(
                results=results,
                dates=env.df.index[-len(results["portfolio_history"]) :],
                actions=results["actions"],
            )

        return results

    def predict_action(self, state):
        """
        Prédit l'action à prendre pour un état donné.

        Args:
            state: L'état actuel de l'environnement

        Returns:
            action: L'action prédite
        """
        if not self._agent:
            # Créer un agent par défaut si nécessaire
            self.create_agent()
            logger.warning("Agent créé automatiquement avec les paramètres par défaut")

        return self._agent.predict(state)

    def save(self, path):
        """
        Sauvegarde le système de trading.

        Args:
            path (str): Chemin de sauvegarde
        """
        if self._agent is None:
            raise ValueError("Aucun agent à sauvegarder")

        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Sauvegarder l'agent
        self._agent.save(path)

        logger.info(f"Système de trading sauvegardé dans {path}")

    def load(self, path):
        """
        Charge le système de trading.

        Args:
            path (str): Chemin de chargement
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas")

        # Créer un agent temporaire pour charger le modèle
        from ai_trading.rl.dqn_agent import DQNAgent

        self._agent = DQNAgent(state_size=1, action_size=3)  # Tailles temporaires
        self._agent.load(path)

        logger.info(f"Système de trading chargé depuis {path}")

        return self._agent

    def test_random_strategy(self, num_episodes=10):
        """
        Teste une stratégie aléatoire pour établir une référence.

        Args:
            num_episodes (int): Nombre d'épisodes à exécuter.

        Returns:
            dict: Résultats de la stratégie aléatoire.
        """
        total_rewards = []
        portfolio_values = []

        for episode in range(num_episodes):
            state, _ = self._env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self._env.action_space.sample()  # Action aléatoire
                state, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # Enregistrer la valeur du portefeuille
                portfolio_values.append(info["portfolio_value"])

            total_rewards.append(episode_reward)

        # Calculer les métriques
        avg_reward = np.mean(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)

        # Calculer les métriques de trading
        final_portfolio_value = portfolio_values[-1]
        initial_portfolio_value = self._env.initial_balance
        total_return = (
            (final_portfolio_value - initial_portfolio_value)
            / initial_portfolio_value
            * 100
        )

        return {
            "average_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "total_return": total_return,
            "final_portfolio_value": final_portfolio_value,
        }

    def integrate_data(
        self,
        market_data=None,
        news_data=None,
        social_data=None,
        sentiment_data=None,
        split_ratio=0.8,
        **kwargs,
    ):
        """
        Intègre les données de marché, de nouvelles et de sentiment.

        Args:
            market_data (pd.DataFrame): Données de marché
            news_data (pd.DataFrame): Données de nouvelles
            social_data (pd.DataFrame): Données de médias sociaux
            sentiment_data (pd.DataFrame): Données de sentiment
            split_ratio (float): Ratio de division train/test
            **kwargs: Arguments supplémentaires

        Returns:
            tuple: (données d'entraînement, données de test)
        """
        # Si aucune donnée n'est fournie, générer des données synthétiques
        if market_data is None:
            market_data = self.data_integrator.generate_synthetic_data(
                n_samples=100, trend="random", volatility=0.02, with_sentiment=True
            )
            logger.info(f"Données synthétiques générées: {len(market_data)} points")

        # Prétraiter les données de marché
        processed_data = self.data_integrator.preprocess_market_data(market_data)

        # Intégrer les données de sentiment si disponibles
        if sentiment_data is not None:
            processed_data = self.data_integrator.integrate_sentiment_data(
                processed_data, sentiment_data
            )

        # Diviser les données en ensembles d'entraînement et de test
        train_size = int(len(processed_data) * split_ratio)
        train_data = processed_data[:train_size]
        test_data = processed_data[train_size:]

        logger.info(
            f"Données intégrées: {len(train_data)} points d'entraînement, {len(test_data)} points de test"
        )

        return train_data, test_data

    def create_transformer(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        output_dim: int = 1,
    ):
        """
        Crée un transformer temporel pour l'analyse des séquences de prix.

        Args:
            input_dim: Dimension des entrées (OHLCV = 5)
            d_model: Dimension du modèle
            nhead: Nombre de têtes d'attention
            num_layers: Nombre de couches Transformer
            dim_feedforward: Dimension du réseau feed-forward
            dropout: Taux de dropout
            max_seq_len: Longueur maximale de séquence
            output_dim: Dimension de sortie
        """
        self._transformer = FinancialTemporalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            output_dim=output_dim,
        )
        self.logger.info(
            f"Transformer temporel créé avec {num_layers} couches et {nhead} têtes d'attention"
        )

    def train_transformer(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        validation_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """
        Entraîne le modèle Transformer avec des hyperparamètres optimisés.

        Args:
            data: Données d'entraînement
            targets: Cibles d'entraînement
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des batchs
            learning_rate: Taux d'apprentissage initial
            weight_decay: Coefficient de régularisation L2
            warmup_steps: Nombre d'étapes de warmup
            gradient_clip: Valeur maximale pour le gradient clipping
            validation_split: Proportion des données pour la validation

        Returns:
            Dictionnaire contenant les historiques des pertes
        """
        if self._transformer is None:
            raise ValueError("Le modèle Transformer n'est pas initialisé")

        optimizer = torch.optim.AdamW(
            self._transformer.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler avec warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(step / warmup_steps, 1.0)
        )

        criterion = nn.MSELoss()

        # Historique des pertes
        history = {"train_loss": [], "val_loss": []}

        # Préparation des données
        dataset_size = len(data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        train_indices, val_indices = indices[split:], indices[:split]
        train_data = data[train_indices]
        train_targets = targets[train_indices]
        val_data = data[val_indices]
        val_targets = targets[val_indices]

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Mode entraînement
            self._transformer.train()
            train_losses = []

            # Entraînement par batch
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i : i + batch_size]
                batch_targets = train_targets[i : i + batch_size]

                optimizer.zero_grad()
                outputs, _ = self._transformer(batch_data)
                loss = criterion(outputs, batch_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._transformer.parameters(), gradient_clip
                )

                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())

            # Mode évaluation
            self._transformer.eval()
            val_losses = []

            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch_data = val_data[i : i + batch_size]
                    batch_targets = val_targets[i : i + batch_size]

                    outputs, _ = self._transformer(batch_data)
                    val_loss = criterion(outputs, batch_targets)
                    val_losses.append(val_loss.item())

            # Calcul des moyennes
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping à l'époque {epoch + 1}")
                break

            print(
                f"Époque {epoch + 1}/{epochs} - "
                f"Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        return history

    def predict_with_transformer(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Fait des prédictions avec le transformer temporel.

        Args:
            data: Données d'entrée de forme (batch_size, seq_len, input_dim)

        Returns:
            Tuple contenant:
            - Prédictions de forme (batch_size, output_dim)
            - Poids d'attention de chaque couche
        """
        if self._transformer is None:
            raise ValueError(
                "Le transformer n'est pas initialisé. Appelez create_transformer() d'abord."
            )

        self._transformer.eval()
        with torch.no_grad():
            predictions, attention_weights = self._transformer(data)
        return predictions, attention_weights

    def load_transformer(self, path: str):
        """
        Charge un transformer pré-entraîné.

        Args:
            path: Chemin vers le fichier du modèle
        """
        if self._transformer is None:
            raise ValueError(
                "Le transformer n'est pas initialisé. Appelez create_transformer() d'abord."
            )

        self._transformer.load_state_dict(torch.load(path))
        self.logger.info(f"Transformer chargé depuis {path}")
