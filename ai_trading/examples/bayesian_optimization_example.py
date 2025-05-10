"""
Exemple d'utilisation de l'optimisation bayésienne des hyperparamètres.

Ce script démontre comment utiliser l'optimisation bayésienne pour trouver
efficacement les hyperparamètres optimaux des agents d'apprentissage par renforcement.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.bayesian_optimizer import (
    optimize_sac_agent_bayesian,
    optimize_gru_sac_agent_bayesian,
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BayesianOptimizationExample")


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Exemple d'optimisation bayésienne des hyperparamètres")
    
    parser.add_argument(
        "--episodes", type=int, default=20, help="Nombre d'épisodes pour l'entraînement"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5, help="Nombre d'épisodes pour l'évaluation"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC", help="Symbole pour le nom des fichiers de résultats"
    )
    parser.add_argument(
        "--agent", type=str, choices=["sac", "gru_sac"], default="sac",
        help="Type d'agent à optimiser"
    )
    parser.add_argument(
        "--save", action="store_true", help="Sauvegarder les résultats"
    )
    parser.add_argument(
        "--initial-points", type=int, default=5, 
        help="Nombre de points initiaux pour l'optimisation bayésienne"
    )
    parser.add_argument(
        "--iterations", type=int, default=15,
        help="Nombre d'itérations pour l'optimisation bayésienne"
    )
    parser.add_argument(
        "--exploration", type=float, default=0.01,
        help="Poids d'exploration pour l'optimisation bayésienne"
    )
    parser.add_argument(
        "--multi-objective", action="store_true",
        help="Activer l'optimisation multi-objectifs"
    )
    
    return parser.parse_args()


def main():
    """
    Exemple d'utilisation de l'optimiseur bayésien d'hyperparamètres pour les agents SAC et GRU-SAC.
    """
    # Récupérer les arguments
    args = parse_args()
    
    logger.info(
        "Génération de données synthétiques pour l'optimisation bayésienne d'hyperparamètres..."
    )

    # Générer des données synthétiques pour l'entraînement
    train_data = generate_synthetic_market_data(
        n_points=1000, trend=0.001, volatility=0.01, start_price=100.0
    )

    # Ajouter des indicateurs techniques
    train_data["sma_10"] = train_data["close"].rolling(10).mean()
    train_data["sma_30"] = train_data["close"].rolling(30).mean()
    train_data["rsi"] = 50 + np.random.normal(0, 10, len(train_data))  # RSI simulé
    train_data["macd"] = train_data["sma_10"] - train_data["sma_30"]  # MACD simulé
    train_data = train_data.fillna(method='bfill')

    logger.info(f"Données synthétiques générées: {len(train_data)} points")
    
    # Définir les poids pour l'optimisation multi-objectifs si activée
    multi_objective_weights = None
    if args.multi_objective:
        multi_objective_weights = {
            "total_reward": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": -0.2,  # Négatif car on veut minimiser le drawdown
            "win_rate": 0.1,
        }
        logger.info(f"Optimisation multi-objectifs activée avec les poids: {multi_objective_weights}")
    
    # Répertoire de sauvegarde
    save_dir = None
    if args.save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = f"results/bayesian_opt/{args.agent}_{args.symbol}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Les résultats seront sauvegardés dans: {save_dir}")
    
    # Optimisation selon le type d'agent
    start_time = time.time()
    
    if args.agent == "sac":
        logger.info(
            f"Démarrage de l'optimisation bayésienne pour l'agent SAC standard..."
        )
        
        # Définir l'espace de paramètres pour SAC
        param_space = {
            "actor_learning_rate": (1e-5, 1e-3),
            "critic_learning_rate": (1e-5, 1e-3),
            "batch_size": (32, 256),
            "hidden_size": (64, 512),
            "entropy_regularization": (0.0, 0.2),
            "grad_clip_value": (0.5, 5.0),
        }
        
        # Optimiser les hyperparamètres avec l'optimisation bayésienne
        best_params = optimize_sac_agent_bayesian(
            train_data=train_data,
            param_space=param_space,
            n_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            save_dir=save_dir,
            n_jobs=1,  # Pour l'exemple, on utilise 1 processus
            n_initial_points=args.initial_points,
            n_iterations=args.iterations,
            exploration_weight=args.exploration,
            multi_objective_weights=multi_objective_weights,
        )
        
    else:  # gru_sac
        logger.info(
            f"Démarrage de l'optimisation bayésienne pour l'agent SAC avec GRU..."
        )
        
        # Définir l'espace de paramètres pour GRU-SAC
        param_space = {
            "actor_learning_rate": (1e-5, 1e-3),
            "critic_learning_rate": (1e-5, 1e-3),
            "batch_size": (32, 256),
            "hidden_size": (64, 512),
            "entropy_regularization": (0.0, 0.2),
            "grad_clip_value": (0.5, 5.0),
            "use_gru": [True],  # Paramètre fixe
            "sequence_length": (3, 20),
            "gru_units": (16, 256),
        }
        
        # Optimiser les hyperparamètres avec l'optimisation bayésienne
        best_params = optimize_gru_sac_agent_bayesian(
            train_data=train_data,
            param_space=param_space,
            n_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            save_dir=save_dir,
            n_jobs=1,  # Pour l'exemple, on utilise 1 processus
            n_initial_points=args.initial_points,
            n_iterations=args.iterations,
            exploration_weight=args.exploration,
            multi_objective_weights=multi_objective_weights,
        )
    
    # Calculer le temps d'exécution
    elapsed_time = time.time() - start_time
    logger.info(f"Optimisation terminée en {elapsed_time:.2f} secondes")
    
    # Afficher les meilleurs paramètres
    logger.info(f"Meilleurs hyperparamètres pour l'agent {args.agent}: {best_params}")
    
    logger.info(
        "Optimisation bayésienne terminée. Les résultats sont disponibles dans le répertoire spécifié."
    )


if __name__ == "__main__":
    main() 