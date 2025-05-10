import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
import torch
from ray.util.multiprocessing import Pool

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "ai_trading" / "models"
RESULTS_DIR = BASE_DIR / "ai_trading" / "info_retour" / "visualisations" / "rl"
DATA_DIR = BASE_DIR / "ai_trading" / "info_retour" / "data" / "processed"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import des modules
from ai_trading.rl.agents.ppo_agent import PPOAgent
from ai_trading.rl.trading_environment import TradingEnvironment


@ray.remote
class RolloutWorker:
    """Travailleur qui collecte des expériences en parallèle."""

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
        agent_config.update({
            "state_dim": state_dim,
            "action_dim": action_dim,
        })
        
        # Créer l'agent
        self.agent = PPOAgent(**agent_config)
        
        logger.info(f"Worker {worker_id} initialisé avec seed {seed}")
    
    def collect_rollouts(self, weights: Dict, n_steps: int = 1000) -> Dict:
        """
        Collecte des expériences avec les poids du réseau fournis.

        Args:
            weights: Poids du réseau ActorCritic
            n_steps: Nombre de pas à collecter

        Returns:
            Dict des expériences collectées
        """
        # Mettre à jour les poids de l'agent
        self.agent.ac_network.load_state_dict(weights)
        
        # Réinitialiser l'environnement
        state, _ = self.env.reset()
        done = False
        truncated = False
        
        # Variables pour stocker les expériences
        states, actions, rewards, next_states, dones = [], [], [], [], []
        log_probs, values = [], []
        
        # Collecter des expériences
        steps = 0
        episode_return = 0.0
        
        while steps < n_steps:
            # Sélectionner une action
            action, log_prob = self.agent.get_action(state)
            
            # Exécuter l'action
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Stocker l'expérience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            log_probs.append(log_prob)
            
            # Obtenir la valeur d'état
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.agent.device)
                value = self.agent.ac_network.get_value(state_tensor)
                values.append(value.cpu().numpy())
            
            # Mettre à jour pour la prochaine étape
            state = next_state
            episode_return += reward
            steps += 1
            
            # Réinitialiser si l'épisode est terminé
            if done or truncated:
                state, _ = self.env.reset()
                done = False
                truncated = False
        
        # Calculer les statistiques
        metrics = {
            "episode_return": episode_return,
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
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "metrics": metrics,
        }


class DistributedPPOTrainer:
    """Entraîneur distribué pour PPO avec actions continues."""

    def __init__(
        self,
        env_config: Dict,
        agent_config: Dict,
        n_workers: int = 4,
        steps_per_update: int = 2000,
        seed: int = 42,
    ):
        """
        Initialise l'entraîneur distribué PPO.

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
        
        # Créer l'agent principal
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config
        )
        
        # Créer les travailleurs
        self.workers = [
            RolloutWorker.remote(
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
        Entraîne l'agent PPO de manière distribuée.

        Args:
            n_iterations: Nombre d'itérations d'entraînement

        Returns:
            Dict des historiques de métriques
        """
        start_time = time.time()
        
        for iteration in range(n_iterations):
            iter_start = time.time()
            
            # Obtenir les poids actuels pour distribution
            weights = self.agent.ac_network.state_dict()
            
            # Collecter des expériences en parallèle
            rollouts = ray.get([
                worker.collect_rollouts.remote(weights, self.steps_per_update // self.n_workers)
                for worker in self.workers
            ])
            
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
        
        # Sauvegarder le modèle final
        model_path = MODELS_DIR / "distributed_ppo_continuous.pth"
        self.agent.save(model_path)
        logger.info(f"Modèle sauvegardé à {model_path}")
        
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
        logger.info(f"Sharpe moyen: {np.mean(sharpe_ratios):.2f} ± {np.std(sharpe_ratios):.2f}")
        logger.info(f"Drawdown moyen: {np.mean(max_drawdowns):.2%} ± {np.std(max_drawdowns):.2%}")
        
        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_sharpe": np.mean(sharpe_ratios),
            "std_sharpe": np.std(sharpe_ratios),
            "mean_drawdown": np.mean(max_drawdowns),
            "std_drawdown": np.std(max_drawdowns),
        }


def load_or_generate_data(use_synthetic: bool = False, n_samples: int = 1000):
    """Charge des données réelles ou génère des données synthétiques."""
    if use_synthetic:
        # Simuler un mouvement de prix avec une tendance et du bruit
        trend = np.linspace(0, 30, n_samples)
        noise = np.random.normal(0, 5, n_samples)
        sine = 10 * np.sin(np.linspace(0, 5, n_samples))
        
        prices = 100 + trend + noise + sine
        
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
        df["timestamp"] = pd.date_range(start="2023-01-01", periods=n_samples, freq="1H")
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
    parser = argparse.ArgumentParser(description="Entraînement distribué PPO pour trading")
    parser.add_argument("--workers", type=int, default=4, help="Nombre de workers parallèles")
    parser.add_argument("--iterations", type=int, default=100, help="Nombre d'itérations d'entraînement")
    parser.add_argument("--steps", type=int, default=2000, help="Nombre de pas par itération")
    parser.add_argument("--synthetic", action="store_true", help="Utiliser des données synthétiques")
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
        "window_size": 20,
        "action_type": "continuous",
        "reward_function": "sharpe",
    }
    
    # Configuration de l'agent
    agent_config = {
        "hidden_size": 128,
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
    trainer = DistributedPPOTrainer(
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