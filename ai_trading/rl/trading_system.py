"""
Module pour le système de trading basé sur l'apprentissage par renforcement.
"""

import logging
import os
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_trading.optim.optimizers import Adam
from ai_trading.optim.schedulers import get_cosine_schedule_with_warmup
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
        # L'intégrateur initialise les analyseurs de sentiment (modèles LLM).
        # Le charger uniquement lorsqu'une intégration P1/P2 est demandée évite
        # de pénaliser les backtests OHLCV purs.
        self.data_integrator = None

        # Ajout du répertoire de modèles
        from ai_trading.config import MODELS_DIR

        self.models_dir = MODELS_DIR / "transformers"
        os.makedirs(self.models_dir, exist_ok=True)

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

        if state_size is None or state_size <= 0:
            raise ValueError("state_size doit être défini avant de créer un agent")

        agent_type = agent_type.lower()
        if agent_type == "dqn":
            if self._env is not None and self._env.action_type != "discrete":
                raise ValueError("DQN requiert un environnement à actions discrètes")
            from ai_trading.rl.dqn_agent import DQNAgent

            agent = DQNAgent(state_size=state_size, action_size=action_size, **kwargs)
        elif agent_type in {"double_dqn", "dueling_dqn", "double_dueling_dqn"}:
            if self._env is not None and self._env.action_type != "discrete":
                raise ValueError("Les variantes DQN requièrent un environnement à actions discrètes")
            from ai_trading.rl.agents.double_dueling_dqn import (
                DoubleDQNAgent,
                DoubleDuelingDQNAgent,
                DuelingDQNAgent,
            )

            agent_class = {
                "double_dqn": DoubleDQNAgent,
                "dueling_dqn": DuelingDQNAgent,
                "double_dueling_dqn": DoubleDuelingDQNAgent,
            }[agent_type]
            agent = agent_class(state_size=state_size, action_size=action_size, **kwargs)
        elif agent_type == "ppo":
            if self._env is not None and self._env.action_type != "continuous":
                raise ValueError("PPO requiert un environnement à actions continues")
            from ai_trading.rl.agents.ppo_agent import PPOAgent

            if self._env is not None:
                action_size = int(self._env.action_space.shape[0])
            agent = PPOAgent(state_dim=state_size, action_dim=action_size, **kwargs)
        elif agent_type == "sac":
            from ai_trading.rl.agents.sac_agent import SACAgent

            # Configurer les bornes d'action correctes si l'environnement existe
            action_bounds = kwargs.pop("action_bounds", (-1, 1))

            # Si l'environnement existe, on peut vérifier si l'espace d'action est continu
            if self._env is not None:
                if (
                    hasattr(self._env, "action_type")
                    and self._env.action_type == "continuous"
                ):
                    # Un environnement mono-actif a une dimension, un portefeuille
                    # multi-actifs une dimension par actif.
                    action_size = int(self._env.action_space.shape[0])
                    logger.info(
                        f"Agent SAC créé pour un espace d'action continu de taille {action_size}"
                    )
                else:
                    logger.warning(
                        "L'agent SAC est optimisé pour les actions continues, mais l'environnement n'a pas d'espace d'action continu."
                    )

            agent = SACAgent(
                state_dim=state_size,
                action_dim=action_size,
                action_bounds=action_bounds,
                **kwargs,
            )
        else:
            raise ValueError(f"Type d'agent inconnu: {agent_type}")

        self._agent = agent
        return agent

    def _get_data_integrator(self):
        if self.data_integrator is None:
            self.data_integrator = RLDataIntegrator()
        return self.data_integrator

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
            raise ValueError(
                "Des données OHLCV réelles explicites sont requises pour créer "
                "un environnement RL. Utiliser DataProcessor ou fournir un DataFrame."
            )

        # Créer l'environnement
        self._env = TradingEnvironment(df=data, **kwargs)

        logger.info(f"Environnement de trading créé avec {len(data)} points de données")
        return self._env

    def train(
        self,
        agent=None,
        episodes=50,
        batch_size=32,
        save_path=None,
        data=None,
        max_steps=None,
        max_total_steps=None,
        train_every=1,
        max_optimization_steps=None,
    ):
        """
        Entraîne l'agent sur l'environnement spécifié.

        Args:
            agent: Agent à entraîner (optionnel, utilisera self._agent si non spécifié)
            episodes (int): Nombre d'épisodes d'entraînement
            batch_size (int): Taille du batch pour l'entraînement
            save_path (str): Chemin pour sauvegarder l'agent (optionnel)
            data (pd.DataFrame): Données d'entraînement (optionnel)
        """
        if not isinstance(train_every, int) or train_every < 1:
            raise ValueError("train_every doit être un entier >= 1")
        if max_optimization_steps is not None and (
            not isinstance(max_optimization_steps, int) or max_optimization_steps < 1
        ):
            raise ValueError("max_optimization_steps doit être un entier >= 1 ou None")

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

        episode_rewards = []
        optimization_steps = 0
        total_steps = 0
        for episode in range(episodes):
            state, _ = self._env.reset()
            done = False
            total_reward = 0
            steps = 0
            ppo_trajectory = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}
            uses_ppo_contract = hasattr(self._agent, "get_action") and not hasattr(self._agent, "remember")

            while (
                not done
                and (max_steps is None or steps < max_steps)
                and (max_total_steps is None or total_steps < max_total_steps)
            ):
                action = self._select_agent_action(
                    self._agent, state, training=True, action_mask=self._env.get_action_mask()
                )
                action = self._env.project_action(action)
                next_state, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                if uses_ppo_contract:
                    ppo_trajectory["states"].append(state)
                    ppo_trajectory["actions"].append(action)
                    ppo_trajectory["rewards"].append(reward)
                    ppo_trajectory["next_states"].append(next_state)
                    ppo_trajectory["dones"].append(done)
                else:
                    remember_kwargs = {}
                    if "next_action_mask" in inspect.signature(self._agent.remember).parameters:
                        remember_kwargs["next_action_mask"] = self._env.get_action_mask()
                    self._agent.remember(state, action, reward, next_state, done, **remember_kwargs)
                    can_optimize = (
                        max_optimization_steps is None
                        or optimization_steps < max_optimization_steps
                    )
                    if (
                        can_optimize
                        and (total_steps + 1) % train_every == 0
                        and self._train_agent_step(self._agent, batch_size)
                    ):
                        optimization_steps += 1
                state = next_state
                total_reward += reward
                steps += 1
                total_steps += 1

            if uses_ppo_contract and ppo_trajectory["states"]:
                can_optimize = (
                    max_optimization_steps is None
                    or optimization_steps < max_optimization_steps
                )
                if can_optimize:
                    self._agent.update(**ppo_trajectory)
                    optimization_steps += 1
            episode_rewards.append(float(total_reward))
            if hasattr(self._agent, "end_episode"):
                self._agent.end_episode()
            if max_total_steps is not None and total_steps >= max_total_steps:
                break

            if episode % 10 == 0:
                self.logger.info(
                    f"Épisode {episode}/{episodes}, Récompense totale: {total_reward}"
                )

        if save_path:
            self._agent.save(save_path)
            self.logger.info(f"Agent sauvegardé dans {save_path}")

        return {
            "episode_rewards": episode_rewards,
            "optimization_steps": optimization_steps,
            "episodes": episodes,
            "total_steps": total_steps,
            "max_optimization_steps": max_optimization_steps,
        }

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
        Évalue les performances de l'agent.

        Args:
            agent: Agent à évaluer
            env: Environnement de trading
            num_episodes (int): Nombre d'épisodes à exécuter
            test_data (pandas.DataFrame, optional): Données de test
            visualize (bool): Si True, visualise les performances
            save_dir (str, optional): Répertoire pour sauvegarder les visualisations

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
            # Réinitialiser l'environnement et obtenir le premier état
            state, _ = env.reset()

            expected_state_size = getattr(agent, "state_size", getattr(agent, "state_dim", None))
            if expected_state_size != state.shape[0]:
                raise ValueError(
                    "Dimensions incompatibles entre l'agent et l'environnement: "
                    f"agent={expected_state_size}, environnement={state.shape[0]}"
                )

            done = False
            episode_reward = 0
            episode_actions = []
            episode_portfolio = []

            while not done:
                action = self._select_agent_action(
                    agent, state, training=False, action_mask=env.get_action_mask()
                )
                action = env.project_action(action)
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
            raise ValueError("Aucun agent chargé. Appelez create_agent() ou load() d'abord.")
        action_mask = self._env.get_action_mask() if self._env is not None else None
        action = self._select_agent_action(self._agent, state, training=False, action_mask=action_mask)
        return self._env.project_action(action) if self._env is not None else action

    def save(self, path):
        """
        Sauvegarde le système de trading.

        Args:
            path (str): Chemin de sauvegarde
        """
        if self._agent is None:
            raise ValueError("Aucun agent à sauvegarder")

        # Créer le répertoire si nécessaire
        Path(path).parent.mkdir(parents=True, exist_ok=True)

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

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("agent_type") != "dqn":
            raise ValueError("Le chargement automatique ne prend en charge que les checkpoints DQN versionnés")
        self._agent = self.create_agent(
            agent_type="dqn",
            state_size=checkpoint["state_size"],
            action_size=checkpoint["action_size"],
            hidden_size=checkpoint.get("hidden_size", 128),
            use_noisy_network=checkpoint.get("use_noisy_network", False),
            n_step=checkpoint.get("n_step", 1),
            device=checkpoint.get("device", "cpu"),
        )
        self._agent.load(path)

        logger.info(f"Système de trading chargé depuis {path}")

        return self._agent

    @staticmethod
    def _select_agent_action(agent, state: np.ndarray, training: bool, action_mask=None):
        """Sélection d'action unique pour les contrats DQN et SAC."""
        if hasattr(agent, "select_action"):
            if hasattr(agent, "state_dim"):
                kwargs = {"deterministic": not training}
            else:
                kwargs = {"training": training}
            if "action_mask" in inspect.signature(agent.select_action).parameters:
                kwargs["action_mask"] = action_mask
            return agent.select_action(state, **kwargs)
        if hasattr(agent, "get_action"):
            kwargs = {"deterministic": not training}
            if "action_mask" in inspect.signature(agent.get_action).parameters:
                kwargs["action_mask"] = action_mask
            action, _ = agent.get_action(state, **kwargs)
            return np.asarray(action, dtype=np.float32).reshape(-1)
        raise TypeError("L'agent ne fournit pas select_action")

    @staticmethod
    def _train_agent_step(agent, batch_size: int) -> bool:
        """Une optimisation, seulement si le replay contient assez de données."""
        buffer = getattr(agent, "replay_buffer", getattr(agent, "memory", None))
        if buffer is None or len(buffer) < batch_size:
            return False
        if hasattr(agent, "replay"):
            agent.replay(batch_size)
        else:
            agent.train(batch_size)
        return True

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
        # Aucune donnée fictive ne doit entrer dans une préparation de trading.
        if market_data is None:
            raise ValueError(
                "Des données OHLCV réelles explicites sont requises pour intégrer "
                "les données RL."
            )

        # Prétraiter les données de marché
        data_integrator = self._get_data_integrator()
        processed_data = data_integrator.preprocess_market_data(market_data)

        # Intégrer les données de sentiment si disponibles
        if sentiment_data is not None:
            processed_data = data_integrator.integrate_sentiment_data(
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
        data: Union[torch.Tensor, pd.DataFrame, str],
        targets: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        validation_split: float = 0.2,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        sequence_length: int = 50,
        pin_memory: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Entraîne le transformer sur des données financières avec DataLoader optimisé.

        Args:
            data: Données d'entrée (DataFrame, ndarray, path ou tensor)
            targets: Valeurs cibles (peut être None si incluses dans data)
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des batchs
            learning_rate: Taux d'apprentissage
            weight_decay: Régularisation L2
            warmup_steps: Nombre d'étapes pour le warmup du learning rate
            gradient_clip: Valeur maximale pour le gradient clipping
            validation_split: Proportion des données pour la validation
            num_workers: Nombre de workers pour le chargement des données
            prefetch_factor: Nombre de batchs à précharger par worker
            sequence_length: Longueur des séquences temporelles
            pin_memory: Si True, utilise la mémoire pin pour le transfert vers GPU

        Returns:
            Historique d'entraînement
        """
        from ai_trading.data.financial_dataset import (
            FinancialDataset,
            get_financial_dataloader,
        )

        history = {"train_loss": [], "val_loss": []}
        device = next(self._transformer.parameters()).device

        # Configurer l'optimiseur et le scheduler
        optimizer = Adam(
            self._transformer.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=epochs * 1000,  # Estimation approximative
        )
        criterion = nn.MSELoss()

        # Préparer les données selon leur type
        if isinstance(data, pd.DataFrame):
            # Diviser les données en train/val directement à partir du DataFrame
            dataset_size = len(data)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))

            # Mélanger les indices
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            # Créer les datasets d'entraînement et de validation
            feature_columns = (
                None  # Utiliser toutes les colonnes numériques disponibles
            )
            target_column = (
                None if targets is not None else "close"
            )  # Par défaut utiliser close

            train_dataset = FinancialDataset(
                data.iloc[train_indices],
                sequence_length=sequence_length,
                target_column=target_column,
                feature_columns=feature_columns,
                is_train=True,
                device=device,
                use_shared_memory=num_workers > 0,
                dtype=torch.float32,
            )

            val_dataset = FinancialDataset(
                data.iloc[val_indices],
                sequence_length=sequence_length,
                target_column=target_column,
                feature_columns=feature_columns,
                is_train=True,
                device=device,
                use_shared_memory=num_workers > 0,
                dtype=torch.float32,
            )

        elif isinstance(data, (str, Path)) and os.path.exists(data):
            # Diviser les données en entraînement/validation en utilisant des indices
            full_dataset = FinancialDataset(
                data,
                sequence_length=sequence_length,
                is_train=True,
                device=device,
                use_shared_memory=num_workers > 0,
                dtype=torch.float32,
            )

            dataset_size = len(full_dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))

            # Mélanger les indices
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            # Utiliser SubsetRandomSampler pour créer les datasets
            from torch.utils.data import SubsetRandomSampler

            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            # Nous utiliserons les samplers directement dans le DataLoader
            train_dataset = full_dataset
            val_dataset = full_dataset

            # Créer les DataLoaders avec les samplers
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else 2,
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else 2,
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

        else:  # torch.Tensor ou np.ndarray
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)

            # Diviser les données en deux ensembles
            dataset_size = len(data)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))

            # Mélanger les indices
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            # Si targets est fourni, diviser également
            if targets is not None:
                if isinstance(targets, np.ndarray):
                    targets = torch.tensor(targets, dtype=torch.float32)

                train_data = data[train_indices]
                train_targets = targets[train_indices]
                val_data = data[val_indices]
                val_targets = targets[val_indices]

                # Créer les datasets
                train_dataset = FinancialDataset(
                    train_data,
                    sequence_length=sequence_length,
                    is_train=True,
                    device=device,
                    use_shared_memory=num_workers > 0,
                    dtype=torch.float32,
                )

                val_dataset = FinancialDataset(
                    val_data,
                    sequence_length=sequence_length,
                    is_train=True,
                    device=device,
                    use_shared_memory=num_workers > 0,
                    dtype=torch.float32,
                )
            else:
                # Si targets n'est pas fourni, diviser uniquement les données
                train_dataset = FinancialDataset(
                    data[train_indices],
                    sequence_length=sequence_length,
                    is_train=True,
                    device=device,
                    use_shared_memory=num_workers > 0,
                    dtype=torch.float32,
                )

                val_dataset = FinancialDataset(
                    data[val_indices],
                    sequence_length=sequence_length,
                    is_train=True,
                    device=device,
                    use_shared_memory=num_workers > 0,
                    dtype=torch.float32,
                )

        # Créer les DataLoaders si ce n'est pas déjà fait
        if "train_loader" not in locals():
            train_loader = get_financial_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=False,
            )

            val_loader = get_financial_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=False,
            )

        self.logger.info(
            f"Début de l'entraînement sur {len(train_loader)} batchs d'entraînement et "
            f"{len(val_loader)} batchs de validation pour {epochs} époques"
        )

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Mode entraînement
            self._transformer.train()
            train_losses = []

            # Barre de progression pour l'entraînement
            train_loop = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs} [Train]")
            for batch_data in train_loop:
                # Extraire les features et targets
                features, batch_targets = batch_data
                features = features.to(device)
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                outputs, _ = self._transformer(features)

                # Adapter la forme si nécessaire
                if outputs.shape != batch_targets.shape:
                    if outputs.dim() > batch_targets.dim():
                        batch_targets = batch_targets.unsqueeze(-1)
                    elif outputs.dim() < batch_targets.dim():
                        outputs = outputs.unsqueeze(-1)

                loss = criterion(outputs, batch_targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self._transformer.parameters(), gradient_clip
                )

                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                train_loop.set_postfix(loss=loss.item())

            # Mode évaluation
            self._transformer.eval()
            val_losses = []

            # Barre de progression pour la validation
            val_loop = tqdm(val_loader, desc=f"Époque {epoch+1}/{epochs} [Val]")
            with torch.no_grad():
                for batch_data in val_loop:
                    # Extraire les features et targets
                    features, batch_targets = batch_data
                    features = features.to(device)
                    batch_targets = batch_targets.to(device)

                    outputs, _ = self._transformer(features)

                    # Adapter la forme si nécessaire
                    if outputs.shape != batch_targets.shape:
                        if outputs.dim() > batch_targets.dim():
                            batch_targets = batch_targets.unsqueeze(-1)
                        elif outputs.dim() < batch_targets.dim():
                            outputs = outputs.unsqueeze(-1)

                    val_loss = criterion(outputs, batch_targets)
                    val_losses.append(val_loss.item())
                    val_loop.set_postfix(loss=val_loss.item())

            # Calcul des moyennes
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Sauvegarder le meilleur modèle
                best_model_path = os.path.join(
                    self.models_dir, f"transformer_best_epoch_{epoch+1}.pt"
                )
                torch.save(self._transformer.state_dict(), best_model_path)
                self.logger.info(
                    f"Meilleur modèle sauvegardé avec Val Loss: {avg_val_loss:.4f}"
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping à l'époque {epoch+1}")
                break

            self.logger.info(
                f"Époque {epoch+1}/{epochs} - "
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
