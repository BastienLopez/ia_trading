import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.hyperparameter_optimizer import optimize_sac_agent, optimize_gru_sac_agent
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HyperparameterOptimizationExample")

def main():
    """
    Exemple d'utilisation de l'optimiseur d'hyperparamètres pour les agents SAC et GRU-SAC.
    """
    logger.info("Génération de données synthétiques pour l'optimisation d'hyperparamètres...")
    
    # Générer des données synthétiques pour l'entraînement
    train_data = generate_synthetic_market_data(
        n_points=1000,
        trend=0.001,
        volatility=0.01,
        start_price=100.0
    )
    
    # Ajouter des indicateurs techniques simples
    train_data['sma_10'] = train_data['close'].rolling(10).mean()
    train_data['sma_30'] = train_data['close'].rolling(30).mean()
    train_data['rsi'] = 50 + np.random.normal(0, 10, len(train_data))  # RSI simulé
    train_data = train_data.bfill()
    
    logger.info(f"Données synthétiques générées: {len(train_data)} points")

    # 1. Optimiser les hyperparamètres d'un agent SAC standard
    logger.info("Démarrage de l'optimisation d'hyperparamètres pour l'agent SAC standard...")
    
    # Définir une grille de paramètres très restreinte pour le test
    sac_param_grid = {
        "actor_learning_rate": [3e-4],
        "critic_learning_rate": [3e-4],
        "batch_size": [64],
        "hidden_size": [128],
        "entropy_regularization": [0.0],
        "grad_clip_value": [None]
    }
    
    # Optimiser les hyperparamètres
    sac_best_params = optimize_sac_agent(
        train_data=train_data,
        param_grid=sac_param_grid,
        n_episodes=3,  # Nombre réduit pour le test
        eval_episodes=2,
        save_dir="results/sac_optimization",
        n_jobs=1  # Utiliser 1 processus pour l'exemple
    )
    
    logger.info(f"Meilleurs hyperparamètres pour l'agent SAC standard: {sac_best_params}")
    
    # 2. Optimiser les hyperparamètres d'un agent SAC avec GRU
    logger.info("Démarrage de l'optimisation d'hyperparamètres pour l'agent SAC avec GRU...")
    
    # Définir une grille de paramètres très restreinte pour le test
    gru_sac_param_grid = {
        "actor_learning_rate": [3e-4],
        "critic_learning_rate": [3e-4],
        "batch_size": [64],
        "hidden_size": [128],
        "entropy_regularization": [0.01],
        "grad_clip_value": [1.0],
        "use_gru": [True],
        "sequence_length": [5],
        "gru_units": [32]
    }
    
    # Optimiser les hyperparamètres
    gru_sac_best_params = optimize_gru_sac_agent(
        train_data=train_data,
        param_grid=gru_sac_param_grid,
        n_episodes=20,  # Nombre réduit pour l'exemple
        eval_episodes=5,
        save_dir="results/gru_sac_optimization",
        n_jobs=1  # Utiliser 1 processus pour l'exemple
    )
    
    logger.info(f"Meilleurs hyperparamètres pour l'agent SAC avec GRU: {gru_sac_best_params}")
    
    logger.info("Optimisation d'hyperparamètres terminée. Les résultats sont disponibles dans les répertoires results/")

if __name__ == "__main__":
    main() 