#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'optimisation Ray RLlib pour scaler l'entraînement des modèles RL.
Permet de paralléliser l'entraînement sur plusieurs cœurs ou machines.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import torch
import numpy as np

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérification de Ray
HAVE_RAY = False
try:
    import ray
    from ray import tune
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.algorithms.sac import SAC
    from ray.rllib.algorithms.dqn import DQN
    from ray.rllib.env.env_context import EnvContext
    from ray.rllib.models import ModelCatalog
    from ray.rllib.policy.policy import Policy
    from ray.tune.registry import register_env
    HAVE_RAY = True
except ImportError:
    logger.warning("Ray n'est pas installé. Utilisez 'pip install ray[rllib]' pour l'installer.")
    # Classes fictives pour les tests quand Ray n'est pas installé
    class Algorithm:
        """Classe fictive Algorithm pour les tests."""
        pass
    
    class PPO(Algorithm):
        """Classe fictive PPO pour les tests."""
        pass
    
    class SAC(Algorithm):
        """Classe fictive SAC pour les tests."""
        pass
    
    class DQN(Algorithm):
        """Classe fictive DQN pour les tests."""
        pass
    
    class Policy:
        """Classe fictive Policy pour les tests."""
        pass
    
    def register_env(name, creator):
        """Fonction fictive register_env pour les tests."""
        pass


class RayRLlibOptimizer:
    """
    Optimiseur utilisant Ray RLlib pour l'entraînement distribué des modèles RL.
    Permet d'entraîner des agents sur plusieurs cœurs CPU ou GPU, voire sur un cluster.
    """
    
    def __init__(
        self,
        env_creator: Callable,
        algorithm: str = "PPO",
        num_workers: int = 2,
        num_gpus: float = 0,
        num_envs_per_worker: int = 1,
        framework: str = "torch",
        config_overrides: Optional[Dict[str, Any]] = None,
        checkpoint_freq: int = 10,
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        seed: Optional[int] = None,
        register_env_name: str = "trading_env"
    ):
        """
        Initialise l'optimiseur Ray RLlib.
        
        Args:
            env_creator: Fonction qui crée l'environnement RL.
            algorithm: Algorithme à utiliser ('PPO', 'SAC', 'DQN', etc.).
            num_workers: Nombre de workers parallèles.
            num_gpus: Nombre de GPUs à utiliser (peut être une fraction).
            num_envs_per_worker: Nombre d'environnements par worker.
            framework: Framework à utiliser ('torch' ou 'tf2').
            config_overrides: Paramètres additionnels pour surcharger la configuration.
            checkpoint_freq: Fréquence de sauvegarde des checkpoints (en épisodes).
            checkpoint_dir: Répertoire pour sauvegarder les checkpoints.
            log_dir: Répertoire pour les logs.
            seed: Graine pour la reproductibilité.
            register_env_name: Nom pour enregistrer l'environnement dans Ray.
        """
        if not HAVE_RAY:
            raise ImportError("Ray n'est pas installé. Utilisez 'pip install ray[rllib]' pour l'installer.")
        
        self.env_creator = env_creator
        self.algorithm_name = algorithm
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.framework = framework
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir or os.path.join(os.getcwd(), "ray_checkpoints")
        self.log_dir = log_dir or os.path.join(os.getcwd(), "ray_logs")
        
        # Initialiser Ray s'il n'est pas déjà initialisé
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=True,
                _temp_dir=tempfile.gettempdir(),
                logging_level=logging.INFO
            )
        
        # Enregistrer l'environnement avec Ray
        self.env_name = register_env_name
        register_env(self.env_name, self._env_creator_wrapper)
        
        # Créer la configuration de base pour l'algorithme
        self.config = self._create_base_config(
            num_workers=num_workers,
            num_gpus=num_gpus,
            num_envs_per_worker=num_envs_per_worker,
            framework=framework,
            seed=seed
        )
        
        # Appliquer les surcharges de configuration
        if config_overrides:
            self.config.update(config_overrides)
        
        # Créer l'algorithme
        self.agent = self._create_algorithm()
        
        logger.info(f"Optimiseur Ray RLlib initialisé avec l'algorithme {algorithm}")
        logger.info(f"Nombre de workers: {num_workers}")
        logger.info(f"Nombre de GPUs: {num_gpus}")
        logger.info(f"Framework: {framework}")
    
    def _env_creator_wrapper(self, env_config: Dict[str, Any] = None) -> Any:
        """
        Wrapper pour la fonction de création d'environnement.
        
        Args:
            env_config: Configuration de l'environnement.
            
        Returns:
            Instance de l'environnement.
        """
        return self.env_creator(env_config or {})
    
    def _create_base_config(
        self,
        num_workers: int,
        num_gpus: float,
        num_envs_per_worker: int,
        framework: str,
        seed: Optional[int]
    ) -> Dict[str, Any]:
        """
        Crée la configuration de base pour l'algorithme.
        
        Args:
            num_workers: Nombre de workers parallèles.
            num_gpus: Nombre de GPUs à utiliser.
            num_envs_per_worker: Nombre d'environnements par worker.
            framework: Framework à utiliser.
            seed: Graine pour la reproductibilité.
            
        Returns:
            Configuration de base.
        """
        config = {
            "env": self.env_name,
            "num_workers": num_workers,
            "num_gpus": num_gpus,
            "num_envs_per_env_runner": num_envs_per_worker,
            "framework": framework,
            "log_level": "INFO",
            "compress_observations": True,
            "rollout_fragment_length": 200,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
            "gamma": 0.99,
            "lr": 3e-4,
            "evaluation_config": {
                "explore": False
            },
            "evaluation_duration": 10,
            "evaluation_duration_unit": "episodes",
            "evaluation_interval": 1,
            "metrics_smoothing_episodes": 100,
            "seed": seed,
            "experimental": {
                "_validate_config": False
            }
        }
        
        # Optimisations spécifiques au framework
        if framework == "torch":
            config.update({
                "use_pytorch": True,
                "torch_optimizer_options": {
                    "weight_decay": 1e-5
                },
                "_tf_policy_handles_more_than_one_loss": False,
            })
        
        return config
    
    def _create_algorithm(self) -> Algorithm:
        """
        Crée l'algorithme RL avec la configuration spécifiée.
        
        Returns:
            Instance de l'algorithme.
        """
        if self.algorithm_name.upper() == "PPO":
            # Configuration spécifique pour PPO
            self.config.update({
                "use_critic": True,
                "use_gae": True,
                "lambda": 0.95,
                "kl_coeff": 0.2,
                "clip_param": 0.2,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "vf_loss_coeff": 0.5,
                "kl_target": 0.01
            })
            return PPO(config=self.config)
        
        elif self.algorithm_name.upper() == "SAC":
            # Configuration spécifique pour SAC
            self.config.update({
                "twin_q": True,
                "prioritized_replay": True,
                "target_network_update_freq": 1,
                "timesteps_per_iteration": 1000,
                "learning_starts": 1000,
                "optimization": {
                    "actor_learning_rate": 3e-4,
                    "critic_learning_rate": 3e-4,
                    "entropy_learning_rate": 3e-4,
                }
            })
            return SAC(config=self.config)
        
        elif self.algorithm_name.upper() == "DQN":
            # Configuration spécifique pour DQN
            self.config.update({
                "dueling": True,
                "double_q": True,
                "prioritized_replay": True,
                "buffer_size": 50000,
                "exploration_config": {
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.02,
                    "epsilon_timesteps": 10000,
                }
            })
            return DQN(config=self.config)
        
        else:
            # Pour les autres algorithmes, utiliser la fabrique générique
            return ray.rllib.algorithms.algorithm.Algorithm.from_preset(self.algorithm_name, self.config)
    
    def train(self, num_iterations: int, with_tune: bool = False) -> Dict[str, Any]:
        """
        Entraîne l'agent RL.
        
        Args:
            num_iterations: Nombre d'itérations d'entraînement.
            with_tune: Si True, utilise Ray Tune pour l'entraînement (utile pour l'optimisation d'hyperparamètres).
            
        Returns:
            Résultats de l'entraînement.
        """
        if with_tune:
            # Utiliser Ray Tune pour l'entraînement et l'optimisation des hyperparamètres
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            
            analysis = tune.run(
                self.algorithm_name,
                config=self.config,
                stop={"training_iteration": num_iterations},
                checkpoint_freq=self.checkpoint_freq,
                checkpoint_at_end=True,
                local_dir=self.checkpoint_dir,
                loggers=[ray.tune.logger.JsonLogger, ray.tune.logger.CSVLogger],
                max_failures=2,
                reuse_actors=True,
                verbose=1
            )
            
            # Récupérer la meilleure configuration
            best_trial = analysis.get_best_trial("episode_reward_mean", "max")
            best_config = best_trial.config
            best_checkpoint = best_trial.checkpoint
            
            # Charger le meilleur modèle
            if best_checkpoint:
                self.agent = Algorithm.from_checkpoint(best_checkpoint)
            
            return {
                "best_trial": best_trial,
                "best_config": best_config,
                "best_checkpoint": best_checkpoint,
                "analysis": analysis
            }
        else:
            # Entraînement standard avec l'algorithme
            results = {}
            for i in range(num_iterations):
                logger.info(f"Itération {i+1}/{num_iterations}")
                result = self.agent.train()
                results[f"iteration_{i+1}"] = result
                
                # Afficher les métriques
                train_rewards = result.get("episode_reward_mean", 0)
                train_length = result.get("episode_len_mean", 0)
                logger.info(f"Récompense moyenne: {train_rewards:.2f}")
                logger.info(f"Longueur d'épisode moyenne: {train_length:.2f}")
                
                # Sauvegarder le checkpoint périodiquement
                if (i + 1) % self.checkpoint_freq == 0:
                    checkpoint = self.agent.save(os.path.join(self.checkpoint_dir, f"checkpoint_{i+1}"))
                    logger.info(f"Checkpoint sauvegardé: {checkpoint}")
            
            # Sauvegarder le checkpoint final
            final_checkpoint = self.agent.save(os.path.join(self.checkpoint_dir, "checkpoint_final"))
            logger.info(f"Checkpoint final sauvegardé: {final_checkpoint}")
            
            # Ajouter le checkpoint final aux résultats
            results["final_checkpoint"] = final_checkpoint
            
            return results
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """
        Évalue l'agent sur plusieurs épisodes.
        
        Args:
            num_episodes: Nombre d'épisodes d'évaluation.
            render: Indique s'il faut afficher la visualisation.
            
        Returns:
            Résultats de l'évaluation.
        """
        logger.info(f"Début de l'évaluation sur {num_episodes} épisodes...")
        
        # Créer un environnement d'évaluation
        eval_env = self._env_creator_wrapper({})
        
        total_rewards = []
        episode_lengths = []
        
        for i in range(num_episodes):
            obs, info = eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            
            while not (done or truncated):
                # Dans les tests, self.agent peut être un mock injecté directement
                action = self.agent.compute_single_action(obs, explore=False)
                
                # Exécuter l'action dans l'environnement
                obs, reward, done, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if render:
                    eval_env.render()
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            logger.info(f"Épisode {i+1}: Récompense={episode_reward:.2f}, Longueur={episode_steps}")
        
        # Calculer les statistiques
        mean_reward = np.mean(total_rewards)
        mean_length = np.mean(episode_lengths)
        std_reward = np.std(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)
        
        logger.info(f"Évaluation terminée. Récompense moyenne: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "mean_episode_length": mean_length,
            "total_rewards": total_rewards,
            "episode_lengths": episode_lengths
        }
    
    def save_model(self, path: str) -> str:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            path: Chemin où sauvegarder le modèle.
            
        Returns:
            Chemin du checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = self.agent.save(path)
        logger.info(f"Modèle sauvegardé: {checkpoint}")
        return checkpoint
    
    def load_model(self, path: str):
        """
        Charge un modèle entraîné.
        
        Args:
            path: Chemin vers le checkpoint.
        """
        self.agent.restore(path)
        logger.info(f"Modèle chargé depuis {path}")
    
    def get_policy(self) -> Policy:
        """
        Retourne la politique de l'agent.
        
        Returns:
            Politique de l'agent.
        """
        return self.agent.get_policy()
    
    def export_model(self, export_dir: str, policy_id: str = "default_policy") -> Dict[str, Any]:
        """
        Exporte le modèle pour une utilisation en production.
        
        Args:
            export_dir: Répertoire d'exportation.
            policy_id: ID de la politique à exporter.
            
        Returns:
            Informations sur le modèle exporté.
        """
        os.makedirs(export_dir, exist_ok=True)
        
        # Exporter le modèle
        export_info = {}
        
        if self.framework == "torch":
            # Pour PyTorch
            policy = self.agent.get_policy(policy_id)
            model = policy.model
            
            # Sauvegarder le modèle PyTorch
            model_path = os.path.join(export_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            export_info["model_path"] = model_path
            
            # Sauvegarder les méta-informations
            meta_path = os.path.join(export_dir, "model_info.json")
            meta_info = {
                "algorithm": self.algorithm_name,
                "framework": self.framework,
                "input_shape": model.obs_space.shape if hasattr(model, 'obs_space') else None,
                "output_shape": model.action_space.shape if hasattr(model, 'action_space') else None,
                "num_outputs": model.num_outputs if hasattr(model, 'num_outputs') else None,
                "policy_id": policy_id
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_info, f, indent=4)
            export_info["meta_info"] = meta_info
            
        else:
            # Pour TensorFlow
            model_path = os.path.join(export_dir, "tf_model")
            policy = self.agent.get_policy(policy_id)
            policy.export_model(model_path)
            export_info["model_path"] = model_path
        
        logger.info(f"Modèle exporté vers {export_dir}")
        return export_info
    
    def cleanup(self):
        """Nettoie les ressources utilisées par l'optimiseur."""
        try:
            self.agent.stop()
        except Exception as e:
            logger.warning(f"Erreur lors de l'arrêt de l'agent: {e}")


def optimize_rl_with_ray(
    env_creator: Callable,
    algorithm: str = "PPO",
    num_workers: int = 2,
    num_gpus: float = 0,
    train_iterations: int = 100,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour optimiser un agent RL avec Ray RLlib.
    
    Args:
        env_creator: Fonction qui crée l'environnement RL.
        algorithm: Algorithme à utiliser ('PPO', 'SAC', 'DQN', etc.).
        num_workers: Nombre de workers parallèles.
        num_gpus: Nombre de GPUs à utiliser.
        train_iterations: Nombre d'itérations d'entraînement.
        config_overrides: Paramètres additionnels pour surcharger la configuration.
        
    Returns:
        Résultats de l'optimisation.
    """
    if not HAVE_RAY:
        raise ImportError("Ray n'est pas installé. Utilisez 'pip install ray[rllib]' pour l'installer.")
    
    # Créer l'optimiseur
    optimizer = RayRLlibOptimizer(
        env_creator=env_creator,
        algorithm=algorithm,
        num_workers=num_workers,
        num_gpus=num_gpus,
        config_overrides=config_overrides
    )
    
    # Entraîner l'agent
    try:
        train_results = optimizer.train(num_iterations=train_iterations)
        
        # Évaluer l'agent
        eval_results = optimizer.evaluate(num_episodes=10)
        
        # Retourner les résultats
        return {
            "train_results": train_results,
            "eval_results": eval_results,
            "optimizer": optimizer
        }
        
    finally:
        # Nettoyer les ressources
        optimizer.cleanup() 