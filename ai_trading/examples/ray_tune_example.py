#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de Ray Tune pour l'optimisation d'hyperparamètres d'un agent RL.
Cet exemple démontre comment optimiser automatiquement les hyperparamètres d'un agent PPO
sur un environnement de trading simple.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import du module développé
from ai_trading.rl.optimization.ray_tune_optimizer import (
    RayTuneOptimizer,
    optimize_rl_agent,
    HAVE_RAY
)

if HAVE_RAY:
    from ray import tune
    from ray.rllib.algorithms.ppo import PPO
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

# Définition d'un environnement simplifié pour l'exemple
class SimpleTradingEnv(gym.Env):
    """
    Environnement de trading simplifié pour démontrer l'optimisation d'hyperparamètres.
    """
    
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        
        # Paramètres de l'environnement
        self.window_size = config.get("window_size", 10)
        self.n_steps = config.get("n_steps", 200)
        self.initial_balance = config.get("initial_balance", 10000.0)
        self.transaction_fee = config.get("transaction_fee", 0.001)
        
        # Générer des données synthétiques
        self.data = self._generate_price_data(self.n_steps)
        
        # Espaces d'observation et d'action
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size + 2,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Réinitialiser l'environnement
        self.reset()
    
    def _generate_price_data(self, n_steps):
        """Génère une série de prix synthétiques."""
        # Tendance + bruit + saisonnalité
        t = np.linspace(0, 4*np.pi, n_steps)
        trend = 0.001 * np.arange(n_steps)
        noise = 0.01 * np.random.randn(n_steps)
        seasonality = 0.02 * np.sin(t)
        
        # Combiner les composantes
        returns = trend + noise + seasonality
        # Convertir en prix
        prices = 100.0 * np.exp(np.cumsum(returns))
        
        return prices
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement."""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Construit l'observation basée sur la fenêtre des prix."""
        # Utiliser une fenêtre des prix normalisés + position et balance
        prices = self.data[self.current_step - self.window_size:self.current_step]
        normalized_prices = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        # Ajouter la position et le solde normalisés
        position_feature = np.array([self.position / self.initial_balance])
        balance_feature = np.array([self.balance / self.initial_balance])
        
        return np.concatenate([normalized_prices, position_feature, balance_feature]).astype(np.float32)
    
    def step(self, action):
        """Exécute une action et retourne le nouvel état."""
        # Obtenir le prix actuel
        current_price = self.data[self.current_step]
        
        # Convertir l'action continue [-1, 1] en changement de position
        target_position_pct = float(action[0])  # Entre -1 et 1
        
        # Calculer la nouvelle position cible en valeur absolue
        max_position_value = self.balance / current_price
        target_position = target_position_pct * max_position_value
        
        # Calculer le changement de position
        position_change = target_position - self.position
        
        # Appliquer les frais de transaction
        if abs(position_change) > 0:
            fee = abs(position_change * current_price * self.transaction_fee)
            self.balance -= fee
        
        # Mettre à jour la position
        if position_change > 0:  # Achat
            cost = position_change * current_price
            if cost <= self.balance:
                self.position += position_change
                self.balance -= cost
        elif position_change < 0:  # Vente
            revenue = abs(position_change) * current_price
            self.position += position_change  # position_change est négatif
            self.balance += revenue
        
        # Calculer la valeur nette
        portfolio_value = self.balance + self.position * current_price
        self.history.append(portfolio_value)
        
        # Passer à l'étape suivante
        self.current_step += 1
        
        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.data) - 1
        
        # Calculer la récompense (changement de valeur du portefeuille)
        if len(self.history) > 1:
            reward = (portfolio_value / self.history[-2] - 1) * 100  # Rendement en pourcentage
        else:
            reward = 0
        
        # Ajouter une petite pénalité pour les transactions trop fréquentes
        if abs(position_change) > 0:
            reward -= 0.01 * abs(position_change / max_position_value)
        
        observation = self._get_observation()
        truncated = False
        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "position": self.position
        }
        
        return observation, reward, done, truncated, info

def custom_search_space():
    """Définit un espace de recherche personnalisé pour l'exemple."""
    return {
        # Hyperparamètres de l'algorithme PPO
        "gamma": tune.uniform(0.9, 0.999),  # Facteur d'actualisation
        "lr": tune.loguniform(1e-5, 1e-3),  # Taux d'apprentissage
        "train_batch_size": tune.choice([1000, 2000, 4000]),  # Taille du batch d'entraînement
        "sgd_minibatch_size": tune.choice([64, 128, 256]),  # Taille des mini-batchs SGD
        "num_sgd_iter": tune.randint(3, 10),  # Nombre d'itérations SGD
        
        # Hyperparamètres spécifiques à PPO
        "clip_param": tune.uniform(0.1, 0.3),  # Paramètre d'écrêtage
        "vf_clip_param": tune.uniform(5.0, 20.0),  # Écrêtage de la fonction de valeur
        "entropy_coeff": tune.loguniform(1e-5, 1e-2),  # Coefficient d'entropie
        "vf_loss_coeff": tune.uniform(0.5, 1.0),  # Coefficient de perte VF
        
        # Paramètres du modèle
        "fcnet_hiddens": tune.choice([
            [64, 64],
            [128, 128],
            [64, 64, 64],
            [128, 64, 32]
        ]),
        "fcnet_activation": tune.choice(["tanh", "relu"])
    }

def main(args):
    """Fonction principale."""
    if not HAVE_RAY:
        print("Ray n'est pas installé. Utilisez 'pip install ray[tune]' pour l'installer.")
        return
    
    print("=" * 80)
    print("Exemple d'optimisation d'hyperparamètres avec Ray Tune")
    print("=" * 80)
    
    # Définir la fonction de création d'environnement
    def env_creator():
        return SimpleTradingEnv({
            "window_size": 20,
            "n_steps": 500,
            "initial_balance": 10000.0,
            "transaction_fee": 0.001
        })
    
    # Enregistrer l'environnement avec Ray
    try:
        from ray.tune.registry import register_env
        register_env("trading_env", lambda config: SimpleTradingEnv(config))
    except ImportError:
        print("Erreur lors de l'enregistrement de l'environnement.")
    
    # Configurer la recherche d'hyperparamètres selon la méthode choisie
    if args.method == "optimize_function":
        # Utiliser la fonction d'optimisation simplifiée
        print(f"Lancement de l'optimisation avec {args.algorithm} et {args.search_alg}")
        print(f"Nombre d'échantillons: {args.num_samples}, itérations: {args.iterations}")
        
        best_config, best_metrics = optimize_rl_agent(
            env_creator=env_creator,
            agent_type=args.algorithm,
            search_space=custom_search_space() if args.custom_space else None,
            num_samples=args.num_samples,
            max_iterations=args.iterations,
            search_alg=args.search_alg,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus
        )
        
        print("\nMeilleure configuration trouvée:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        
        print("\nMeilleures métriques:")
        for key, value in best_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
    elif args.method == "custom_optimizer":
        # Utiliser directement la classe RayTuneOptimizer pour plus de flexibilité
        print("Utilisation de RayTuneOptimizer personnalisé")
        
        # Définir la fonction de création d'agent
        def agent_creator(config):
            return PPO(
                config=dict(
                    env="trading_env",
                    framework="torch",
                    # Désactiver complètement la nouvelle API stack pour éviter les problèmes
                    api_stack={
                        "enable_rl_module_and_learner": False,
                        "enable_env_runner_and_connector_v2": False
                    },
                    experimental={
                        "_validate_config": False
                    },
                    **config
                )
            )
        
        # Créer l'optimiseur
        optimizer = RayTuneOptimizer(
            env_creator=env_creator,
            agent_creator=agent_creator,
            search_space=custom_search_space(),
            num_samples=args.num_samples,
            max_iterations=args.iterations,
            search_alg=args.search_alg,
            scheduler="asha",
            cpu_per_trial=args.num_cpus,
            gpu_per_trial=args.num_gpus
        )
        
        # Lancer l'optimisation
        best_config, best_metrics = optimizer.optimize()
        
        print("\nMeilleure configuration trouvée:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
    
    print("\nOptimisation terminée!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemple d'optimisation d'hyperparamètres avec Ray Tune")
    parser.add_argument("--method", type=str, default="optimize_function", 
                        choices=["optimize_function", "custom_optimizer"],
                        help="Méthode d'optimisation à utiliser")
    parser.add_argument("--algorithm", type=str, default="ppo", 
                        choices=["ppo", "sac", "dqn"],
                        help="Algorithme RL à optimiser")
    parser.add_argument("--search_alg", type=str, default="optuna", 
                        choices=["random", "optuna", "hyperopt", "bayesopt"],
                        help="Algorithme de recherche d'hyperparamètres")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Nombre d'échantillons à tester")
    parser.add_argument("--iterations", type=int, default=20, 
                        help="Nombre d'itérations par essai")
    parser.add_argument("--num_cpus", type=int, default=1, 
                        help="Nombre de CPUs par essai")
    parser.add_argument("--num_gpus", type=float, default=0, 
                        help="Nombre de GPUs par essai")
    parser.add_argument("--custom_space", action="store_true", 
                        help="Utiliser un espace de recherche personnalisé")
    
    args = parser.parse_args()
    main(args) 