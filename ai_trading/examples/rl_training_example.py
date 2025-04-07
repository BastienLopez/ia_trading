import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl_agent import RLTradingSystem

def run_training_example():
    """
    Exemple d'entraînement d'un agent RL sur des données de crypto-monnaie.
    """
    print("Démarrage de l'exemple d'entraînement RL...")
    
    # Créer l'intégrateur de données
    integrator = RLDataIntegrator()
    
    # Définir la période
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 an de données
    
    # Collecter et prétraiter les données
    print(f"Collecte et prétraitement des données pour BTC du {start_date} au {end_date}...")
    market_data = integrator.collect_market_data(
        symbol='BTC',
        start_date=start_date,
        end_date=end_date,
        interval='1d'
    )
    preprocessed_market_data = integrator.preprocess_market_data(market_data)
    
    sentiment_data = integrator.collect_sentiment_data(
        symbol='BTC',
        start_date=start_date,
        end_date=end_date
    )
    
    # Intégrer les données pour l'apprentissage par renforcement
    print("Intégration des données pour l'apprentissage par renforcement...")
    train_data, test_data = integrator.integrate_data(
        market_data=preprocessed_market_data,
        sentiment_data=sentiment_data,
        window_size=10,
        test_split=0.2
    )
    
    print(f"Données d'entraînement: {len(train_data)} points")
    print(f"Données de test: {len(test_data)} points")
    
    # Créer le système de trading RL
    rl_system = RLTradingSystem()
    
    # Créer l'environnement de trading avec les données d'entraînement
    print("\nCréation de l'environnement de trading...")
    env = rl_system.create_environment(
        data=train_data,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=10
    )
    
    # Créer l'agent DQN
    print("\nCréation de l'agent DQN...")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = rl_system.create_agent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=10000
    )
    
    # Configurer l'arrêt anticipé
    early_stopping = {
        'patience': 20,
        'min_delta': 0.01,
        'metric': 'returns'
    }
    
    # Entraîner l'agent
    print("\nDémarrage de l'entraînement...")
    history = rl_system.train(
        episodes=100,
        batch_size=32,
        update_target_every=5,
        save_path='models/dqn_trading',
        visualize=True,
        checkpoint_interval=10,
        early_stopping=early_stopping,
        max_steps_per_episode=None,
        use_tensorboard=True,
        tensorboard_log_dir='logs/tensorboard'
    )
    
    # Évaluer l'agent sur les données de test
    print("\nÉvaluation de l'agent sur les données de test...")
    test_env = rl_system.create_environment(
        data=test_data,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=10
    )
    
    # Créer un nouvel environnement pour l'évaluation
    results = rl_system.evaluate(test_data=test_data, visualize=True)
    
    # Afficher les résultats
    print("\nRésultats de l'évaluation:")
    print(f"Valeur finale du portefeuille: ${results['final_value']:.2f}")
    print(f"Rendement: {results['returns']*100:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.4f}")
    print(f"Drawdown maximum: {results['max_drawdown']*100:.2f}%")
    
    print("\nExemple d'entraînement terminé!")

if __name__ == "__main__":
    run_training_example() 