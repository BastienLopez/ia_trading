#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'optimisation d'hyperparamètres utilisant Ray Tune.
Implémente l'optimisation distribuée des hyperparamètres pour les modèles d'IA de trading.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

# Vérification de Ray
try:
    import ray
    from ray import tune
    from ray.tune import Callback
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.basic_variant import BasicVariantGenerator
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.optuna import OptunaSearch

    HAVE_RAY = True
except ImportError:
    HAVE_RAY = False
    logging.warning(
        "Ray n'est pas installé. Utilisez 'pip install ray[tune]' pour l'installer."
    )

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Définir le chemin de base pour les sorties
BASE_INFO_RETOUR_DIR = (
    Path(__file__).parent.parent.parent / "info_retour" / "examples" / "ray_tune"
)
BASE_INFO_RETOUR_DIR.mkdir(parents=True, exist_ok=True)


class ProgressCallback(Callback):
    """Callback pour afficher la progression de l'optimisation."""

    def __init__(self):
        self.best_result = float("-inf")
        self.best_config = None

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Appelé lorsqu'un essai génère un résultat."""
        if "mean_reward" in result and result["mean_reward"] > self.best_result:
            self.best_result = result["mean_reward"]
            self.best_config = trial.config
            logger.info(f"Nouvel optimum trouvé: {self.best_result:.4f}")
            logger.info(f"Configuration: {self.best_config}")


class RayTuneOptimizer:
    """
    Optimiseur d'hyperparamètres utilisant Ray Tune pour trouver la meilleure configuration
    pour les agents d'IA de trading.
    """

    def __init__(
        self,
        env_creator: Callable[[], gym.Env],
        agent_creator: Callable[[Dict[str, Any]], Any],
        search_space: Dict[str, Any],
        metric: str = "mean_reward",
        mode: str = "max",
        num_samples: int = 10,
        max_iterations: int = 100,
        cpu_per_trial: int = 1,
        gpu_per_trial: float = 0,
        search_alg: str = "random",
        scheduler: str = "asha",
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        resume: bool = False,
    ):
        """
        Initialise l'optimiseur Ray Tune.

        Args:
            env_creator: Fonction qui crée l'environnement de trading
            agent_creator: Fonction qui crée l'agent avec une configuration donnée
            search_space: Espace de recherche des hyperparamètres
            metric: Métrique à optimiser
            mode: 'max' ou 'min' selon qu'on veut maximiser ou minimiser la métrique
            num_samples: Nombre d'échantillons à tester
            max_iterations: Nombre maximal d'itérations par essai
            cpu_per_trial: Nombre de CPUs par essai
            gpu_per_trial: Nombre de GPUs par essai (fraction possible)
            search_alg: Algorithme de recherche ('random', 'optuna', 'hyperopt', 'bayesopt')
            scheduler: Ordonnanceur ('asha', 'pbt')
            checkpoint_dir: Répertoire de sauvegarde des checkpoints
            log_dir: Répertoire des logs
            resume: Reprendre une optimisation précédente
        """
        if not HAVE_RAY:
            raise ImportError(
                "Ray n'est pas installé. Utilisez 'pip install ray[tune]' pour l'installer."
            )

        self.env_creator = env_creator
        self.agent_creator = agent_creator
        self.search_space = search_space
        self.metric = metric
        self.mode = mode
        self.num_samples = num_samples
        self.max_iterations = max_iterations
        self.cpu_per_trial = cpu_per_trial
        self.gpu_per_trial = gpu_per_trial
        # Utiliser les chemins dans info_retour
        self.checkpoint_dir = checkpoint_dir or str(BASE_INFO_RETOUR_DIR / "results")
        self.log_dir = log_dir or str(BASE_INFO_RETOUR_DIR / "logs")
        self.resume = resume

        # Initialiser Ray s'il n'est pas déjà initialisé
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True, log_to_driver=True, logging_level=logging.INFO
            )

        # Configurer le scheduler
        if scheduler.lower() == "asha":
            self.scheduler = ASHAScheduler(
                max_t=max_iterations,
                grace_period=max(1, min(10, max_iterations // 4)),
                reduction_factor=2,
                metric=metric,
                mode=mode,
            )
        elif scheduler.lower() == "pbt":
            self.scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric=metric,
                mode=mode,
                perturbation_interval=max(5, max_iterations // 20),
                hyperparam_mutations=self._get_pbt_mutations(search_space),
            )
        else:
            logger.warning(
                f"Scheduler {scheduler} non reconnu, utilisation de ASHAScheduler par défaut"
            )
            self.scheduler = ASHAScheduler(
                max_t=max_iterations,
                grace_period=min(10, max_iterations // 4),
                reduction_factor=2,
                metric=metric,
                mode=mode,
            )

        # Configurer l'algorithme de recherche
        if search_alg.lower() == "random":
            self.search_alg = BasicVariantGenerator()
        elif search_alg.lower() == "optuna":
            self.search_alg = OptunaSearch(metric=metric, mode=mode)
        elif search_alg.lower() == "hyperopt":
            self.search_alg = HyperOptSearch(metric=metric, mode=mode)
        else:
            logger.warning(
                f"Algorithme de recherche {search_alg} non reconnu, utilisation de la recherche aléatoire par défaut"
            )
            self.search_alg = BasicVariantGenerator()

    def _get_pbt_mutations(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Crée les mutations pour l'algorithme PBT basées sur l'espace de recherche."""
        mutations = {}
        for key, value in search_space.items():
            if isinstance(value, tune.search.sample.Float):
                lower, upper = value.lower, value.upper
                mutations[key] = tune.function(
                    lambda: np.random.uniform(0.8, 1.2), log=True
                )
            elif isinstance(value, tune.search.sample.Integer):
                mutations[key] = tune.function(
                    lambda: np.random.choice([0.8, 1.0, 1.2]), log=True
                )
            elif isinstance(value, tune.search.sample.Categorical):
                # Ne pas muter les variables catégorielles
                pass
        return mutations

    def _train_function(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fonction d'entraînement exécutée par Ray Tune pour chaque essai.

        Args:
            config: Configuration d'hyperparamètres

        Returns:
            Dictionnaire de résultats (métriques)
        """
        # Créer l'environnement
        env = self.env_creator()

        # Créer l'agent avec la configuration actuelle
        agent = self.agent_creator(config)

        # Entraîner l'agent
        for iteration in range(self.max_iterations):
            # Effectuer une itération d'entraînement
            train_metrics = agent.train()

            # Évaluer l'agent (peut être fait moins fréquemment)
            if iteration % max(1, self.max_iterations // 10) == 0:
                eval_metrics = self._evaluate_agent(agent, env)

                # Combiner les métriques d'entraînement et d'évaluation
                metrics = {**train_metrics, **eval_metrics}

                # Rapporter les métriques à Ray Tune
                tune.report(training_iteration=iteration, **metrics)

        return train_metrics

    def _evaluate_agent(
        self, agent: Any, env: gym.Env, num_episodes: int = 5
    ) -> Dict[str, float]:
        """
        Évalue l'agent dans l'environnement.

        Args:
            agent: Agent à évaluer
            env: Environnement de trading
            num_episodes: Nombre d'épisodes d'évaluation

        Returns:
            Dictionnaire de métriques d'évaluation
        """
        total_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            episode_length = 0

            while not (done or truncated):
                action = agent.compute_action(obs)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                episode_length += 1

            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        # Calculer les métriques
        mean_reward = np.mean(total_rewards)
        mean_length = np.mean(episode_lengths)

        return {
            "mean_reward": mean_reward,
            "mean_episode_length": mean_length,
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards),
            "std_reward": np.std(total_rewards),
        }

    def optimize(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Lance l'optimisation des hyperparamètres.

        Returns:
            Tuple de (meilleure configuration, métriques du meilleur essai)
        """
        # Créer les répertoires de sauvegarde si nécessaire
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Callback de progression
        progress_callback = ProgressCallback()

        # Lancer l'optimisation avec Ray Tune
        analysis = tune.run(
            self._train_function,
            config=self.search_space,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
            search_alg=self.search_alg,
            resources_per_trial={"cpu": self.cpu_per_trial, "gpu": self.gpu_per_trial},
            storage_path=self.checkpoint_dir,
            callbacks=[progress_callback],
            resume=self.resume,
            verbose=1,
            reuse_actors=True,
            max_failures=2,
        )

        # Récupérer la meilleure configuration
        best_trial = analysis.get_best_trial(metric=self.metric, mode=self.mode)
        best_config = best_trial.config
        best_metrics = best_trial.last_result

        # Sauvegarder les résultats
        result_path = os.path.join(self.log_dir, "ray_tune_results.json")
        with open(result_path, "w") as f:
            json.dump(
                {
                    "best_config": best_config,
                    "best_metrics": {
                        k: float(v) if isinstance(v, (float, int)) else v
                        for k, v in best_metrics.items()
                    },
                    "all_trials": [
                        {
                            "config": trial.config,
                            "metrics": {
                                k: float(v) if isinstance(v, (float, int)) else v
                                for k, v in trial.last_result.items()
                            },
                        }
                        for trial in analysis.trials
                    ],
                },
                f,
                indent=2,
            )

        logger.info(f"Optimisation terminée. Meilleure configuration : {best_config}")
        logger.info(f"Meilleures métriques : {best_metrics}")
        logger.info(f"Résultats sauvegardés dans {result_path}")

        # Visualiser les résultats
        self._visualize_results(analysis)

        return best_config, best_metrics

    def _visualize_results(self, analysis):
        """Génère des visualisations basiques des résultats d'optimisation."""
        try:
            # Créer un DataFrame à partir des résultats
            df = analysis.results_df

            # Sauvegarder le DataFrame en CSV
            csv_path = os.path.join(self.log_dir, "ray_tune_results.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Résultats exportés en CSV dans {csv_path}")

            # Générer quelques statistiques basiques
            stats = {
                "num_trials": len(analysis.trials),
                "metric_range": {
                    "min": float(df[self.metric].min()),
                    "max": float(df[self.metric].max()),
                    "mean": float(df[self.metric].mean()),
                    "median": float(df[self.metric].median()),
                    "std": float(df[self.metric].std()),
                },
            }

            # Sauvegarder les statistiques
            stats_path = os.path.join(self.log_dir, "ray_tune_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

            # Générer d'autres visualisations avec Matplotlib si possible
            # (Code omis pour la concision)

        except Exception as e:
            logger.warning(f"Erreur lors de la visualisation des résultats: {e}")


def optimize_rl_agent(
    env_creator: Callable[[], gym.Env],
    agent_type: str = "ppo",
    search_space: Optional[Dict[str, Any]] = None,
    num_samples: int = 10,
    max_iterations: int = 100,
    search_alg: str = "optuna",
    metric: str = "mean_reward",
    mode: str = "max",
    checkpoint_dir: Optional[str] = None,
    num_cpus: int = 1,
    num_gpus: float = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fonction simplifiée pour l'optimisation d'hyperparamètres d'agents RL.

    Args:
        env_creator: Fonction qui crée l'environnement
        agent_type: Type d'agent ('ppo', 'sac', 'dqn')
        search_space: Espace de recherche personnalisé (sinon, utilise l'espace par défaut)
        num_samples: Nombre d'échantillons à tester
        max_iterations: Nombre maximal d'itérations par essai
        search_alg: Algorithme de recherche ('random', 'optuna', etc.)
        metric: Métrique à optimiser
        mode: 'max' ou 'min'
        checkpoint_dir: Répertoire de sauvegarde des checkpoints
        num_cpus: Nombre de CPUs par essai
        num_gpus: Nombre de GPUs par essai

    Returns:
        Tuple (meilleure configuration, meilleures métriques)
    """
    if not HAVE_RAY:
        raise ImportError(
            "Ray n'est pas installé. Utilisez 'pip install ray[tune]' pour l'installer."
        )

    # Utiliser des chemins dans info_retour
    if checkpoint_dir is None:
        checkpoint_dir = str(BASE_INFO_RETOUR_DIR / "checkpoints")

    # Créer un espace de recherche par défaut si non spécifié
    if search_space is None:
        if agent_type == "ppo":
            search_space = {
                # Hyperparamètres de l'algorithme
                "gamma": tune.uniform(0.9, 0.999),  # Facteur d'actualisation
                "lr": tune.loguniform(1e-5, 1e-3),  # Taux d'apprentissage
                "train_batch_size": tune.choice([1000, 2000, 4000]),
                "sgd_minibatch_size": tune.choice([64, 128, 256]),
                "num_sgd_iter": tune.randint(1, 10),
                # Spécifique à PPO
                "clip_param": tune.uniform(0.1, 0.3),
                "vf_clip_param": tune.uniform(5.0, 20.0),
                "entropy_coeff": tune.loguniform(1e-5, 1e-2),
            }
        elif agent_type == "sac":
            search_space = {
                "gamma": tune.uniform(0.9, 0.999),
                "lr": tune.loguniform(1e-5, 1e-3),
                "train_batch_size": tune.choice([256, 512, 1024]),
                "target_entropy": tune.choice(["auto", -1, -5, -10]),
                "tau": tune.uniform(0.001, 0.1),
                "target_network_update_freq": tune.choice([1, 5, 10]),
            }
        elif agent_type == "dqn":
            search_space = {
                "gamma": tune.uniform(0.9, 0.999),
                "lr": tune.loguniform(1e-5, 1e-3),
                "buffer_size": tune.choice([10000, 50000, 100000]),
                "train_batch_size": tune.choice([32, 64, 128]),
                "learning_starts": tune.choice([1000, 5000, 10000]),
                "target_network_update_freq": tune.choice([100, 500, 1000]),
                "exploration_config.final_epsilon": tune.uniform(0.01, 0.1),
            }
        else:
            raise ValueError(f"Type d'agent non pris en charge: {agent_type}")

    # Initialiser Ray s'il n'est pas déjà initialisé
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Enregistrer l'environnement
    from ray.tune.registry import register_env

    register_env("trading_env", lambda config: env_creator())

    # Définir la fonction de création d'agent
    def agent_creator(config):
        if agent_type.lower() == "ppo":
            from ray.rllib.algorithms.ppo import PPO

            return PPO(config=dict(env=env_creator, framework="torch", **config))
        elif agent_type.lower() == "sac":
            from ray.rllib.algorithms.sac import SAC

            return SAC(config=dict(env=env_creator, framework="torch", **config))
        elif agent_type.lower() == "dqn":
            from ray.rllib.algorithms.dqn import DQN

            return DQN(config=dict(env=env_creator, framework="torch", **config))
        else:
            raise ValueError(f"Type d'agent non supporté: {agent_type}")

    # Créer et exécuter l'optimiseur
    optimizer = RayTuneOptimizer(
        env_creator=env_creator,
        agent_creator=agent_creator,
        search_space=search_space,
        metric=metric,
        mode=mode,
        num_samples=num_samples,
        max_iterations=max_iterations,
        cpu_per_trial=num_cpus,
        gpu_per_trial=num_gpus,
        search_alg=search_alg,
        scheduler="asha",
        checkpoint_dir=checkpoint_dir,
    )

    return optimizer.optimize()
