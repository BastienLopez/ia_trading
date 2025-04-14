import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    """Trace un graphique comparant les différentes récompenses"""
    # Générer des données pour un scénario complet
    days = 60
    df = generate_synthetic_data(days, volatility=0.02, trend=0.0005)
    
    # Initialiser les calculateurs de récompense
    sharpe_reward = SharpeRatioReward(risk_free_rate=0.01, window_size=10)
    transaction_reward = TransactionCostReward(base_cost=0.001, frequency_penalty=0.0005)
    drawdown_reward = DrawdownReward(penalty_factor=2.0)
    
    # Calculer les récompenses pour chaque jour
    sharpe_rewards = []
    transaction_rewards = []
    drawdown_rewards = []
    
    # Actions simulées (alternance achat/vente/hold)
    actions = [0, 1, 0, 2, 0] * (days // 5 + 1)
    
    # Pour chaque jour, calculer les récompenses
    portfolio_value = 10000
    portfolio_values = [portfolio_value]
    
    for i in range(1, days):
        # Calculer le PnL basé sur le changement de prix
        price_change = (df['close'].iloc[i] / df['close'].iloc[i-1]) - 1
        pnl = price_change
        
        # Mettre à jour la valeur du portefeuille (simplifié)
        portfolio_value = portfolio_value * (1 + pnl)
        portfolio_values.append(portfolio_value)
        
        # Calculer les différentes récompenses
        action = actions[i]
        
        # Sharpe
        sr_reward = sharpe_reward.calculate(pnl)
        sharpe_rewards.append(sr_reward)
        
        # Transaction
        tc_reward = transaction_reward.calculate(pnl, action, portfolio_value * 0.5)
        transaction_rewards.append(tc_reward)
        
        # Drawdown
        dd_reward = drawdown_reward.calculate(pnl, portfolio_value)
        drawdown_rewards.append(dd_reward)
    
    # Créer un graphique pour comparer les récompenses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Tracer les prix
    ax1.plot(df.index[1:], df['close'].values[1:], label='Prix', color='blue')
    ax1.set_title('Prix et valeur du portefeuille')
    ax1.set_ylabel('Prix')
    ax1.legend(loc='upper left')
    
    # Ajouter la valeur du portefeuille sur un axe secondaire
    ax1_2 = ax1.twinx()
    ax1_2.plot(df.index[1:], portfolio_values[1:], label='Portefeuille', color='green', linestyle='--')
    ax1_2.set_ylabel('Valeur du portefeuille')
    ax1_2.legend(loc='upper right')
    
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
    plt.savefig('reward_comparison.png')
    plt.close()
    
    print("\nGraphique de comparaison des récompenses enregistré sous 'reward_comparison.png'")

if __name__ == "__main__":
    # Tester chaque type de récompense
    test_sharpe_ratio_reward()
    test_transaction_cost_reward()
    test_drawdown_reward()
    
    # Comparer les récompenses
    plot_rewards_comparison()
    
    print("\nTous les tests de récompense ont été exécutés avec succès.") 