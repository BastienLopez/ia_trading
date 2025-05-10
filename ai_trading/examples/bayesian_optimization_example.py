"""
Exemple d'utilisation de l'optimisation bayésienne pour le trading algorithmique.

Ce script montre comment utiliser l'optimiseur bayésien pour trouver les meilleurs 
hyperparamètres d'un agent de trading basé sur l'algorithme SAC (Soft Actor-Critic).
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajuster le chemin pour inclure le répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai_trading.rl.bayesian_optimizer import optimize_agent_bayesian
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.agents.multitask_agent import MultitaskTradingAgent
from ai_trading.utils.data_generator import generate_market_data
from ai_trading.config import INFO_RETOUR_DIR

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def generate_dataset(n_samples=1000, volatility=0.02, drift=0.001):
    """
    Génère un dataset synthétique pour le trading.
    
    Args:
        n_samples: Nombre d'échantillons
        volatility: Volatilité du prix
        drift: Tendance directionnelle
        
    Returns:
        pd.DataFrame: DataFrame avec les données générées
    """
    return generate_market_data(
        n_samples=n_samples,
        start_price=10000,
        volatility=volatility,
        trend=drift,
        seasonal_amplitude=0.1,
        seasonal_frequency=20,
    )


def optimize_sac_agent():
    """
    Optimise les hyperparamètres d'un agent SAC pour le trading.
    """
    logger.info("Génération des données synthétiques...")
    data = generate_dataset(n_samples=1000)
    
    # Définir l'espace des hyperparamètres à explorer
    param_space = {
        "learning_rate": (1e-5, 1e-2, "real"),
        "gamma": (0.9, 0.999, "real"),
        "tau": (0.001, 0.1, "real"),
        "batch_size": [32, 64, 128, 256],
        "hidden_size": [64, 128, 256, 512],
        "alpha": (0.1, 5.0, "real"),
    }
    
    # Répertoire pour sauvegarder les résultats
    save_dir = INFO_RETOUR_DIR / "bayesian_opt_sac"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("Démarrage de l'optimisation bayésienne...")
    best_params = optimize_agent_bayesian(
        train_data=data,
        agent_class=SACAgent,
        param_space=param_space,
        n_episodes=30,
        eval_episodes=10,
        n_calls=30,
        save_dir=save_dir,
    )
    
    logger.info(f"Meilleurs hyperparamètres trouvés: {best_params}")
    
    # Sauvegarder les meilleurs hyperparamètres dans un format facile à lire
    with open(save_dir / "best_params_summary.txt", "w") as f:
        f.write("Meilleurs hyperparamètres pour l'agent SAC:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    return best_params


def optimize_multitask_agent():
    """
    Optimise les hyperparamètres d'un agent multi-tâches pour le trading.
    """
    logger.info("Génération des données synthétiques...")
    data = generate_dataset(n_samples=1000)
    
    # Définir l'espace des hyperparamètres à explorer pour l'agent multi-tâches
    param_space = {
        "d_model": [128, 256, 512],
        "n_heads": [4, 8],
        "num_layers": [2, 4, 6],
        "risk_aversion": (0.1, 0.9, "real"),
        "exploration_rate": (0.05, 0.3, "real"),
        "lr": (1e-5, 1e-2, "real"),
    }
    
    # Répertoire pour sauvegarder les résultats
    save_dir = INFO_RETOUR_DIR / "bayesian_opt_multitask"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Fonction de création d'environnement adaptée à l'agent multi-tâches
    def create_multitask_env():
        from ai_trading.rl.trading_environment import TradingEnvironment
        env = TradingEnvironment(
            df=data,
            window_size=50,
            initial_balance=10000,
            commission=0.001,
            use_risk_manager=True,
        )
        
        # Adapter l'espace d'observation et d'action pour l'agent multi-tâches
        env.observation_space.shape = (env.state_dim,)
        
        return env
    
    from ai_trading.rl.bayesian_optimizer import BayesianOptimizer
    
    # Créer l'optimiseur bayésien
    optimizer = BayesianOptimizer(
        env_creator=create_multitask_env,
        agent_class=MultitaskTradingAgent,
        param_space=param_space,
        n_episodes=20,
        eval_episodes=5,
        save_dir=save_dir,
        n_calls=20,
    )
    
    logger.info("Démarrage de l'optimisation bayésienne pour l'agent multi-tâches...")
    best_params, best_score = optimizer.bayesian_search()
    
    logger.info(f"Meilleurs hyperparamètres trouvés: {best_params}")
    logger.info(f"Meilleur score: {best_score}")
    
    # Sauvegarder les meilleurs hyperparamètres dans un format facile à lire
    with open(save_dir / "best_params_summary.txt", "w") as f:
        f.write("Meilleurs hyperparamètres pour l'agent multi-tâches:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nMeilleur score: {best_score}\n")
    
    return best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimisation bayésienne des hyperparamètres")
    parser.add_argument("--agent", type=str, choices=["sac", "multitask"], default="sac",
                      help="Type d'agent à optimiser (sac, multitask)")
    
    args = parser.parse_args()
    
    if args.agent == "sac":
        logger.info("Optimisation des hyperparamètres de l'agent SAC")
        optimize_sac_agent()
    else:
        logger.info("Optimisation des hyperparamètres de l'agent multi-tâches")
        optimize_multitask_agent() 