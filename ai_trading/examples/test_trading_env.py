import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Ajouter le répertoire racine au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.trading_environment import TradingEnvironment

def test_random_actions():
    """Teste l'environnement avec des actions aléatoires."""
    # Données de test complètes
    data = {
        'open': [100 + i for i in range(100)],
        'high': [101 + i for i in range(100)],
        'low': [99 + i for i in range(100)],
        'close': [100 + i for i in range(100)],
        'volume': [1000 + i*10 for i in range(100)]
    }
    df = pd.DataFrame(data)
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=5
    )
    
    # Réinitialiser l'environnement
    obs, _ = env.reset()
    
    # Exécuter des actions aléatoires
    done = False
    total_reward = 0
    
    while not done:
        # Générer une action aléatoire
        action = env.action_space.sample()
        
        # Exécuter l'action
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Afficher des informations
        print(f"Action: {action}, Reward: {reward:.4f}, Portfolio: ${info['portfolio_value']:.2f}")
    
    print(f"Episode terminé. Récompense totale: {total_reward:.4f}")
    print(f"Valeur finale du portefeuille: ${info['portfolio_value']:.2f}")
    print(f"Rendement: {((info['portfolio_value'] / env.initial_balance) - 1) * 100:.2f}%")

if __name__ == "__main__":
    test_random_actions() 