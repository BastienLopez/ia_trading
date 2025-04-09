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
    # Créer des données de test simples
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    prices = np.random.normal(loc=100, scale=10, size=100)
    prices = np.cumsum(np.random.normal(loc=0, scale=1, size=100)) + 100  # Marche aléatoire
    prices = np.abs(prices)  # Assurer que les prix sont positifs

    df = pd.DataFrame({"close": prices}, index=dates)
    
    # Créer l'environnement
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=10,
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