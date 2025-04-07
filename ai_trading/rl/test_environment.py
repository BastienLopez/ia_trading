import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_environment import TradingEnvironment

def test_random_actions():
    """
    Teste l'environnement avec des actions aléatoires.
    """
    # Créer des données de test
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.random.normal(loc=100, scale=10, size=100)
    prices = np.cumsum(np.random.normal(loc=0, scale=1, size=100)) + 100  # Marche aléatoire
    prices = np.abs(prices)  # Assurer que les prix sont positifs
    
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Créer l'environnement
    env = TradingEnvironment(df, initial_balance=10000, window_size=5)
    
    # Réinitialiser l'environnement
    obs = env.reset()
    
    # Exécuter des actions aléatoires
    done = False
    total_reward = 0
    
    while not done:
        action = np.random.randint(0, 3)  # Action aléatoire
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Afficher l'état actuel
        env.render()
    
    # Afficher les résultats
    print(f"Récompense totale: {total_reward:.4f}")
    print(f"Valeur finale du portefeuille: ${env.get_portfolio_value():.2f}")
    print(f"Rendement: {((env.get_portfolio_value() / 10000) - 1) * 100:.2f}%")
    
    # Tracer l'historique du portefeuille
    plt.figure(figsize=(10, 6))
    plt.plot(env.get_portfolio_history())
    plt.title('Évolution de la valeur du portefeuille')
    plt.xlabel('Étapes')
    plt.ylabel('Valeur ($)')
    plt.grid(True)
    plt.savefig('portfolio_history.png')
    plt.close()

if __name__ == "__main__":
    test_random_actions() 