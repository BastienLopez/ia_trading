import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import ray
import torch

# Ajuster les chemins d'importation
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

# Import des modules
from ai_trading.rl.agents.transformer_ppo_agent import TransformerPPOAgent
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration des chemins
BASE_DIR = root_dir
MODELS_DIR = BASE_DIR / "ai_trading" / "models"
RESULTS_DIR = BASE_DIR / "ai_trading" / "info_retour" / "visualisations" / "rl"
DATA_DIR = BASE_DIR / "ai_trading" / "info_retour" / "data" / "processed"

# Créer les répertoires nécessaires
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@ray.remote
class TransformerRolloutWorker:
    """Travailleur qui collecte des expériences en parallèle avec l'agent TransformerPPO."""

    def __init__(
        self,
        env_config: Dict,
        agent_config: Dict,
        worker_id: int,
        seed: int = None,
    ):
        """
        Initialise un travailleur de rollout.

        Args:
            env_config: Configuration de l'environnement
            agent_config: Configuration de l'agent
            worker_id: Identifiant du travailleur
            seed: Graine aléatoire pour la reproductibilité
        """
        self.worker_id = worker_id

        # Fixer la graine aléatoire
        if seed is not None:
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)

        # Créer l'environnement
        self.env = TradingEnvironment(**env_config)

        # Extraire les dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Mettre à jour la configuration de l'agent
        agent_config.update(
            {
                "state_dim": state_dim,
                "action_dim": action_dim,
            }
        )

        # Créer l'agent
        self.agent = TransformerPPOAgent(**agent_config)

        logger.info(f"Worker {worker_id} initialisé avec seed {seed}")

    def collect_rollouts(
        self, actor_weights: Dict, critic_weights: Dict, n_steps: int = 1000
    ) -> Dict:
        """
        Collecte des expériences avec les poids des réseaux fournis.

        Args:
            actor_weights: Poids du réseau Acteur
            critic_weights: Poids du réseau Critique
            n_steps: Nombre de pas à collecter

        Returns:
            Dict des expériences collectées
        """
        # Mettre à jour les poids de l'agent
        self.agent.actor.load_state_dict(actor_weights)
        self.agent.critic.load_state_dict(critic_weights)

        # Réinitialiser l'environnement et le buffer d'états
        state, _ = self.env.reset()
        self.agent.reset_state_buffer()

        # Variables pour stocker les expériences
        states, actions, rewards, next_states, dones = [], [], [], [], []

        # Collecter des expériences
        steps = 0
        episode_return = 0.0
        episode_count = 0

        while steps < n_steps:
            # Sélectionner une action
            action, _ = self.agent.get_action(state)

            # Exécuter l'action
            next_state, reward, done, truncated, _ = self.env.step(action)

            # Stocker l'expérience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)

            # Mettre à jour pour la prochaine étape
            state = next_state
            episode_return += reward
            steps += 1

            # Réinitialiser si l'épisode est terminé
            if done or truncated:
                state, _ = self.env.reset()
                self.agent.reset_state_buffer()
                episode_count += 1
                episode_return = 0.0

        # Calculer les statistiques
        metrics = {
            "episode_return": episode_return,
            "steps_collected": steps,
            "episodes_completed": episode_count,
            "sharpe_ratio": self.env.calculate_sharpe_ratio(),
            "max_drawdown": self.env.calculate_max_drawdown(),
        }

        # Retourner les expériences et métriques
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_states": np.array(next_states),
            "dones": np.array(dones),
            "metrics": metrics,
        }


class DistributedTransformerPPOTrainer:
    """Entraîneur distribué pour TransformerPPO avec actions continues."""

    def __init__(
        self,
        env_config: Dict,
        agent_config: Dict,
        n_workers: int = 4,
        steps_per_update: int = 2000,
        seed: int = 42,
    ):
        """
        Initialise l'entraîneur distribué TransformerPPO.

        Args:
            env_config: Configuration de l'environnement
            agent_config: Configuration de l'agent
            n_workers: Nombre de travailleurs en parallèle
            steps_per_update: Nombre de pas avant mise à jour
            seed: Graine aléatoire
        """
        self.env_config = env_config
        self.agent_config = agent_config
        self.n_workers = n_workers
        self.steps_per_update = steps_per_update
        self.seed = seed

        # Initialiser Ray s'il n'est pas déjà initialisé
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Créer l'environnement local pour obtenir les dimensions
        local_env = TradingEnvironment(**env_config)
        state_dim = local_env.observation_space.shape[0]
        action_dim = local_env.action_space.shape[0]

        # Mettre à jour la configuration de l'agent
        agent_config.update(
            {
                "state_dim": state_dim,
                "action_dim": action_dim,
            }
        )

        # Créer l'agent principal
        self.agent = TransformerPPOAgent(**agent_config)

        # Créer les travailleurs
        self.workers = [
            TransformerRolloutWorker.remote(
                env_config=env_config,
                agent_config=agent_config,
                worker_id=i,
                seed=seed + i,
            )
            for i in range(n_workers)
        ]

        # Historiques pour les métriques
        self.returns_history = []
        self.sharpe_ratio_history = []
        self.max_drawdown_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []

        logger.info(f"Trainer initialisé avec {n_workers} workers")

    def train(self, n_iterations: int = 100) -> Dict:
        """
        Entraîne l'agent TransformerPPO de manière distribuée.

        Args:
            n_iterations: Nombre d'itérations d'entraînement

        Returns:
            Dict des historiques de métriques
        """
        start_time = time.time()

        for iteration in range(n_iterations):
            iter_start = time.time()

            # Obtenir les poids actuels pour distribution
            actor_weights = self.agent.actor.state_dict()
            critic_weights = self.agent.critic.state_dict()

            # Collecter des expériences en parallèle
            rollouts = ray.get(
                [
                    worker.collect_rollouts.remote(
                        actor_weights,
                        critic_weights,
                        self.steps_per_update // self.n_workers,
                    )
                    for worker in self.workers
                ]
            )

            # Regrouper les expériences
            all_states = np.concatenate([r["states"] for r in rollouts])
            all_actions = np.concatenate([r["actions"] for r in rollouts])
            all_rewards = np.concatenate([r["rewards"] for r in rollouts])
            all_next_states = np.concatenate([r["next_states"] for r in rollouts])
            all_dones = np.concatenate([r["dones"] for r in rollouts])

            # Mettre à jour l'agent avec les expériences collectées
            losses = self.agent.update(
                states=all_states,
                actions=all_actions,
                rewards=all_rewards,
                next_states=all_next_states,
                dones=all_dones,
            )

            # Collecter les métriques
            episode_returns = [r["metrics"]["episode_return"] for r in rollouts]
            sharpe_ratios = [r["metrics"]["sharpe_ratio"] for r in rollouts]
            max_drawdowns = [r["metrics"]["max_drawdown"] for r in rollouts]

            # Mettre à jour les historiques
            self.returns_history.append(np.mean(episode_returns))
            self.sharpe_ratio_history.append(np.mean(sharpe_ratios))
            self.max_drawdown_history.append(np.mean(max_drawdowns))
            self.actor_loss_history.append(losses["actor_loss"])
            self.critic_loss_history.append(losses["critic_loss"])
            self.entropy_history.append(losses["entropy"])

            # Afficher les progrès
            iter_time = time.time() - iter_start
            total_time = time.time() - start_time

            if (iteration + 1) % 10 == 0 or iteration == 0:
                logger.info(
                    f"Iter {iteration + 1}/{n_iterations} "
                    f"[{iter_time:.1f}s, total: {total_time:.1f}s] - "
                    f"Return: {self.returns_history[-1]:.2f}, "
                    f"Sharpe: {self.sharpe_ratio_history[-1]:.2f}, "
                    f"Drawdown: {self.max_drawdown_history[-1]:.2%}, "
                    f"Actor Loss: {losses['actor_loss']:.4f}, "
                    f"Critic Loss: {losses['critic_loss']:.4f}, "
                    f"Entropy: {losses['entropy']:.4f}"
                )

            # Sauvegarder le modèle périodiquement
            if (iteration + 1) % 20 == 0 or iteration == n_iterations - 1:
                model_path = (
                    MODELS_DIR / f"distributed_transformer_ppo_iter{iteration+1}.pt"
                )
                self.agent.save(model_path)
                logger.info(f"Modèle intermédiaire sauvegardé à {model_path}")

        # Sauvegarder le modèle final
        model_path = MODELS_DIR / "distributed_transformer_ppo_final.pt"
        self.agent.save(model_path)
        logger.info(f"Modèle final sauvegardé à {model_path}")

        # Retourner les historiques de métriques
        return {
            "returns": self.returns_history,
            "sharpe_ratios": self.sharpe_ratio_history,
            "max_drawdowns": self.max_drawdown_history,
            "actor_losses": self.actor_loss_history,
            "critic_losses": self.critic_loss_history,
            "entropy": self.entropy_history,
            "training_time": time.time() - start_time,
        }

    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        Évalue l'agent entraîné.

        Args:
            n_episodes: Nombre d'épisodes d'évaluation

        Returns:
            Dict des métriques d'évaluation
        """
        # Créer un environnement d'évaluation
        eval_env = TradingEnvironment(**self.env_config)

        returns = []
        sharpe_ratios = []
        max_drawdowns = []

        for episode in range(n_episodes):
            state, _ = eval_env.reset()
            self.agent.reset_state_buffer()
            episode_return = 0.0
            done = False
            truncated = False

            while not (done or truncated):
                # Action déterministe pour l'évaluation
                action, _ = self.agent.get_action(state, deterministic=True)
                next_state, reward, done, truncated, _ = eval_env.step(action)

                episode_return += reward
                state = next_state

            returns.append(episode_return)
            sharpe_ratios.append(eval_env.calculate_sharpe_ratio())
            max_drawdowns.append(eval_env.calculate_max_drawdown())

        # Afficher les statistiques d'évaluation
        logger.info("\nRésultats de l'évaluation:")
        logger.info(f"Rendement moyen: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        logger.info(
            f"Sharpe moyen: {np.mean(sharpe_ratios):.2f} ± {np.std(sharpe_ratios):.2f}"
        )
        logger.info(
            f"Drawdown moyen: {np.mean(max_drawdowns):.2%} ± {np.std(max_drawdowns):.2%}"
        )

        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_sharpe": np.mean(sharpe_ratios),
            "std_sharpe": np.std(sharpe_ratios),
            "mean_drawdown": np.mean(max_drawdowns),
            "std_drawdown": np.std(max_drawdowns),
        }


def load_or_generate_data(use_synthetic: bool = False, n_samples: int = 5000):
    """Charge des données réelles ou génère des données synthétiques."""
    if use_synthetic:
        # Simuler un mouvement de prix avec une tendance et du bruit
        trend = np.linspace(0, 50, n_samples)
        noise = np.random.normal(0, 10, n_samples)
        sine1 = 20 * np.sin(np.linspace(0, 10, n_samples))
        sine2 = 10 * np.sin(np.linspace(0, 25, n_samples))

        # Simuler des régimes de marché
        regimes = np.zeros(n_samples)
        change_points = np.random.choice(range(n_samples), size=10, replace=False)
        for point in change_points:
            regimes[point:] += np.random.uniform(-15, 15)

        prices = 100 + trend + noise + sine1 + sine2 + regimes

        # Générer d'autres colonnes nécessaires
        data = {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_samples),
        }

        # Créer le DataFrame avec timestamp
        df = pd.DataFrame(data)
        df["timestamp"] = pd.date_range(
            start="2023-01-01", periods=n_samples, freq="1H"
        )
        df.set_index("timestamp", inplace=True)
    else:
        # Charger des données réelles
        data_path = DATA_DIR / "btc_usd_1h.csv"
        if not os.path.exists(data_path):
            logger.error(f"Fichier de données introuvable: {data_path}")
            return None

        df = pd.read_csv(data_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement distribué TransformerPPO pour trading"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Nombre de workers parallèles"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Nombre d'itérations d'entraînement"
    )
    parser.add_argument(
        "--steps", type=int, default=2000, help="Nombre de pas par itération"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Utiliser des données synthétiques"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=50,
        help="Longueur de la séquence temporelle",
    )
    args = parser.parse_args()

    # Charger ou générer des données
    df = load_or_generate_data(use_synthetic=args.synthetic)
    if df is None:
        return

    # Configuration de l'environnement
    env_config = {
        "df": df,
        "initial_balance": 10000.0,
        "transaction_fee": 0.001,
        "window_size": 30,
        "action_type": "continuous",
        "reward_function": "sharpe",
        "include_technical_indicators": True,
    }

    # Configuration de l'agent
    agent_config = {
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "activation": "gelu",
        "sequence_length": args.sequence_length,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "critic_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "update_epochs": 10,
        "mini_batch_size": 64,
    }

    # Initialiser l'entraîneur
    trainer = DistributedTransformerPPOTrainer(
        env_config=env_config,
        agent_config=agent_config,
        n_workers=args.workers,
        steps_per_update=args.steps,
        seed=42,
    )

    # Entraîner le modèle
    metrics = trainer.train(n_iterations=args.iterations)

    # Évaluer le modèle
    eval_metrics = trainer.evaluate(n_episodes=20)

    # Arrêter Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
