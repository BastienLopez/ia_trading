import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utiliser des imports relatifs
from .trading_environment import TradingEnvironment
from .data_integration import RLDataIntegrator

def test_random_actions():
    """Teste l'environnement avec des actions aléatoires."""
    # Créer des données de test simples
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    prices = np.random.normal(loc=100, scale=10, size=100)
    prices = np.cumsum(np.random.normal(loc=0, scale=1, size=100)) + 100  # Marche aléatoire
    prices = np.abs(prices)  # Assurer que les prix sont positifs

    # Créer un DataFrame plus complet avec toutes les colonnes nécessaires
    df = pd.DataFrame(
        {
            "open": prices * 0.99,  # Prix d'ouverture légèrement inférieur
            "high": prices * 1.02,  # Prix maximum légèrement supérieur
            "low": prices * 0.98,   # Prix minimum légèrement inférieur
            "close": prices,        # Prix de clôture
            "volume": np.random.randint(1000, 10000, 100)  # Volume aléatoire
        }, 
        index=dates
    )
    
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


def test_sophisticated_rewards():
    """
    Teste les récompenses sophistiquées de l'environnement de trading:
    1. Ratio de Sharpe
    2. Pénalité pour transactions fréquentes
    3. Pénalité basée sur le drawdown
    """
    print("\n=== Test des récompenses sophistiquées ===")
    
    # Générer des données de marché
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    
    # Créer une tendance haussière puis baissière pour tester le drawdown
    trend = np.concatenate([
        np.linspace(0, 20, 100),  # Tendance haussière
        np.linspace(20, 10, 100)   # Tendance baissière (drawdown)
    ])
    
    # Ajouter du bruit à la tendance
    noise = np.random.normal(0, 1, 200)
    prices = 100 + trend + noise
    
    # Créer le DataFrame
    df = pd.DataFrame({"close": prices}, index=dates)
    df["high"] = df["close"] * 1.02  # Ajouter des colonnes nécessaires
    df["low"] = df["close"] * 0.98
    df["open"] = df["close"] * 0.99
    df["volume"] = np.random.randint(1000, 10000, 200)
    
    # Créer l'environnement
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=10,
    )
    
    # Réinitialiser l'environnement
    obs, _ = env.reset()
    
    # Listes pour stocker les résultats
    rewards = []
    portfolio_values = []
    actions_history = []
    
    # Scénario de test 1: Transactions fréquentes
    print("\nScénario 1: Test des pénalités pour transactions fréquentes")
    
    # Alterner entre acheter et vendre à chaque étape
    for i in range(20):
        # Alterner entre acheter (2) et vendre (0)
        action = 2 if i % 2 == 0 else 0
        
        # Exécuter l'action
        obs, reward, done, truncated, info = env.step(action)
        
        # Stocker les résultats
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])
        actions_history.append(action)
        
        # Afficher les informations
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.4f}, Portfolio: ${info['portfolio_value']:.2f}")
    
    # Réinitialiser pour le scénario 2
    env.reset()
    rewards_scenario2 = []
    
    # Scénario de test 2: Actions optimales
    print("\nScénario 2: Actions optimales pour maximiser le ratio de Sharpe")
    
    # Adopter une stratégie plus cohérente pendant la tendance haussière
    for i in range(20):
        # Acheter et conserver pendant la tendance haussière
        action = 2 if i < 5 else 1  # Acheter au début, puis tenir
        
        # Exécuter l'action
        obs, reward, done, truncated, info = env.step(action)
        
        # Stocker les résultats
        rewards_scenario2.append(reward)
        
        # Afficher les informations
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.4f}, Portfolio: ${info['portfolio_value']:.2f}")
    
    # Comparer les récompenses entre les deux scénarios
    print("\nComparaison des récompenses:")
    print(f"Récompense totale du scénario 1 (transactions fréquentes): {sum(rewards):.4f}")
    print(f"Récompense totale du scénario 2 (stratégie cohérente): {sum(rewards_scenario2):.4f}")
    
    # Visualiser les récompenses
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Scénario 1: Transactions fréquentes")
    plt.plot(rewards_scenario2, label="Scénario 2: Stratégie cohérente")
    plt.xlabel("Étapes")
    plt.ylabel("Récompense")
    plt.title("Comparaison des récompenses entre les deux stratégies")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_comparison.png")
    plt.close()
    
    print("Graphique sauvegardé dans 'reward_comparison.png'")


if __name__ == "__main__":
    # Ce script doit être exécuté comme un module Python
    # python -m ai_trading.rl.test_environment
    test_random_actions()
    test_sophisticated_rewards()
