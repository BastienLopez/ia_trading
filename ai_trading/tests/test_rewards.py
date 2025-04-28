import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Définir le chemin pour les visualisations
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations', 'rewards')
# Créer le répertoire s'il n'existe pas
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

from ai_trading.rl.advanced_rewards import (
    SharpeRatioReward,
    TransactionCostReward,
    DrawdownReward
)

def generate_synthetic_data(n_days=30, volatility=0.02, trend=0.001):
    """Génère des données synthétiques pour le test"""
    prices = [100]  # Prix initial
    for _ in range(n_days - 1):
        # Simuler un mouvement de prix avec tendance et volatilité
        price_change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Créer un DataFrame avec une colonne 'close'
    dates = pd.date_range(start='2023-01-01', periods=n_days)
    df = pd.DataFrame({'close': prices}, index=dates)
    return df

def test_sharpe_ratio_reward():
    """Teste la récompense basée sur le ratio de Sharpe"""
    print("\n=== Test du SharpeRatioReward ===")
    
    # Initialiser le calculateur de récompense
    reward_calculator = SharpeRatioReward(risk_free_rate=0.01, window_size=10)
    
    # Générer des séquences de rendements
    positive_returns = [0.01, 0.02, 0.01, 0.015, 0.005]  # Rendements positifs et stables
    negative_returns = [-0.01, -0.02, -0.01, -0.015, -0.005]  # Rendements négatifs
    volatile_returns = [0.05, -0.04, 0.06, -0.05, 0.04]  # Rendements volatils
    
    # Tester avec les différentes séquences
    rewards = []
    
    print("Rendements positifs et stables:")
    reward_calculator.reset()
    for ret in positive_returns:
        reward = reward_calculator.calculate(ret)
        print(f"  Rendement: {ret:.4f}, Récompense: {reward:.4f}")
        rewards.append(reward)
    
    print("\nRendements négatifs:")
    reward_calculator.reset()
    for ret in negative_returns:
        reward = reward_calculator.calculate(ret)
        print(f"  Rendement: {ret:.4f}, Récompense: {reward:.4f}")
        rewards.append(reward)
    
    print("\nRendements volatils:")
    reward_calculator.reset()
    for ret in volatile_returns:
        reward = reward_calculator.calculate(ret)
        print(f"  Rendement: {ret:.4f}, Récompense: {reward:.4f}")
        rewards.append(reward)

def test_transaction_cost_reward():
    """Teste la récompense avec coûts de transaction"""
    print("\n=== Test du TransactionCostReward ===")
    
    # Initialiser le calculateur de récompense
    reward_calculator = TransactionCostReward(base_cost=0.001, frequency_penalty=0.0005)
    
    # Scénarios de test
    scenarios = [
        {"pnl": 0.02, "action": 0, "position_value": 1000, "description": "Hold sans transaction"},
        {"pnl": 0.02, "action": 1, "position_value": 1000, "description": "Achat avec profit"},
        {"pnl": 0.02, "action": 2, "position_value": 1000, "description": "Vente avec profit"},
        {"pnl": -0.01, "action": 1, "position_value": 1000, "description": "Achat avec perte"},
        {"pnl": -0.01, "action": 2, "position_value": 1000, "description": "Vente avec perte"}
    ]
    
    # Tester chaque scénario
    reward_calculator.reset()
    for scenario in scenarios:
        reward = reward_calculator.calculate(
            scenario["pnl"], scenario["action"], scenario["position_value"]
        )
        print(f"{scenario['description']}:")
        print(f"  PnL: {scenario['pnl']:.4f}, Action: {scenario['action']}, "
              f"Valeur position: {scenario['position_value']:.2f}")
        print(f"  Récompense: {reward:.4f}")
    
    # Tester les transactions fréquentes
    print("\nTransactions fréquentes:")
    reward_calculator.reset()
    actions = [1, 2, 1, 2, 1]  # Alternance achat/vente
    for i, action in enumerate(actions):
        reward = reward_calculator.calculate(0.01, action, 1000)
        print(f"  Action {i+1}: {action}, Récompense: {reward:.4f}")

def test_drawdown_reward():
    """Teste la récompense basée sur le drawdown"""
    print("\n=== Test du DrawdownReward ===")
    
    # Initialiser le calculateur de récompense
    reward_calculator = DrawdownReward(penalty_factor=2.0)
    
    # Générer une série de valeurs de portefeuille avec un drawdown
    portfolio_values = [10000]  # Valeur initiale
    
    # Croissance initiale
    for _ in range(5):
        portfolio_values.append(portfolio_values[-1] * 1.02)  # +2% de croissance
    
    # Drawdown
    for _ in range(3):
        portfolio_values.append(portfolio_values[-1] * 0.95)  # -5% de baisse
    
    # Récupération
    for _ in range(4):
        portfolio_values.append(portfolio_values[-1] * 1.03)  # +3% de croissance
    
    # Tester la récompense à chaque étape
    reward_calculator.reset()
    print("Évolution de la récompense avec drawdown:")
    
    for i in range(1, len(portfolio_values)):
        pnl = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        reward = reward_calculator.calculate(pnl, portfolio_values[i])
        
        # Calculer le drawdown actuel pour l'affichage
        peak = max(portfolio_values[:i+1])
        current_drawdown = (peak - portfolio_values[i]) / peak if peak > 0 else 0
        
        print(f"  Étape {i}: Valeur: {portfolio_values[i]:.2f}, "
              f"PnL: {pnl:.4f}, Drawdown: {current_drawdown:.4f}, "
              f"Récompense: {reward:.4f}")

def plot_rewards_comparison():
    """Trace et compare les trois types de récompenses sur les mêmes données."""
    print("Comparaison des différentes fonctions de récompense...")
    
    # Générer des données synthétiques
    df = generate_synthetic_data(n_days=100, volatility=0.03, trend=0.0005)
    
    # Initialiser les objets de récompense
    sharpe_reward = SharpeRatioReward(lookback_window=5)
    transaction_reward = TransactionCostReward(transaction_cost=0.01)
    drawdown_reward = DrawdownReward(max_allowed_drawdown=0.1)
    
    # Simuler des actions aléatoires et calculer les récompenses
    actions = np.random.choice([0, 1, 2], size=len(df))  # 0: hold, 1: buy, 2: sell
    positions = np.zeros(len(df))
    cash = np.ones(len(df)) * 10000  # 10,000 de cash initial
    
    for i in range(1, len(df)):
        if actions[i] == 1 and cash[i-1] > 0:  # Acheter
            # Dépenser 50% du cash disponible
            spent = cash[i-1] * 0.5
            positions[i] = positions[i-1] + spent / df['close'][i]
            cash[i] = cash[i-1] - spent
        elif actions[i] == 2 and positions[i-1] > 0:  # Vendre
            # Vendre 50% des positions
            sold = positions[i-1] * 0.5
            positions[i] = positions[i-1] - sold
            cash[i] = cash[i-1] + sold * df['close'][i]
        else:  # Hold
            positions[i] = positions[i-1]
            cash[i] = cash[i-1]
    
    # Calculer la valeur du portefeuille
    portfolio_value = cash + positions * df['close'].values
    
    # Calculer les récompenses pour chaque pas de temps
    sharpe_rewards = []
    transaction_rewards = []
    drawdown_rewards = []
    
    for i in range(1, len(df)):
        # Info pour chaque étape
        info = {
            'portfolio_value': portfolio_value[i],
            'previous_portfolio_value': portfolio_value[i-1],
            'position': positions[i],
            'cash': cash[i],
            'price': df['close'][i],
            'previous_position': positions[i-1],
            'action': actions[i]
        }
        
        # Calculer les récompenses
        sharpe_rewards.append(sharpe_reward.calculate_reward(info))
        transaction_rewards.append(transaction_reward.calculate_reward(info))
        drawdown_rewards.append(drawdown_reward.calculate_reward(info))
    
    # Tracer les résultats
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Tracer le prix et la valeur du portefeuille
    ax1.plot(df.index, df['close'], label='Prix', color='blue')
    ax1.set_ylabel('Prix', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, portfolio_value, label='Valeur du portefeuille', color='green')
    ax1_twin.set_ylabel('Valeur du portefeuille', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1.set_title('Prix et Valeur du portefeuille')
    ax1.grid(True)
    
    # Tracer les récompenses
    ax2.plot(df.index[1:], sharpe_rewards, label='Sharpe', color='orange')
    ax2.plot(df.index[1:], transaction_rewards, label='Transaction', color='red')
    ax2.plot(df.index[1:], drawdown_rewards, label='Drawdown', color='purple')
    ax2.set_title('Comparaison des récompenses')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Récompense')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    reward_comparison_path = os.path.join(VISUALIZATION_DIR, 'reward_comparison.png')
    plt.savefig(reward_comparison_path)
    plt.close()
    
    print(f"\nGraphique de comparaison des récompenses enregistré sous '{reward_comparison_path}'")

if __name__ == "__main__":
    # Tester chaque type de récompense
    test_sharpe_ratio_reward()
    test_transaction_cost_reward()
    test_drawdown_reward()
    
    # Comparer les récompenses
    plot_rewards_comparison()
    
    print("\nTous les tests de récompense ont été exécutés avec succès.") 