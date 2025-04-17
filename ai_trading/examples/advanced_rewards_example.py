import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.trading_environment import TradingEnvironment
from rl.advanced_rewards import (
    AdvancedRewardCalculator, 
    SharpeRatioReward,
    TransactionCostReward, 
    DrawdownReward
)
from data.market_data import MarketDataFetcher
from data.preprocessing import EnhancedMarketDataPreprocessor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Démonstration des récompenses avancées pour RL en trading')
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbole de la crypto-monnaie (par défaut: BTC)')
    parser.add_argument('--days', type=int, default=90, help='Nombre de jours de données à récupérer (par défaut: 90)')
    parser.add_argument('--interval', type=str, default='1d', 
                        help='Intervalle de temps pour les données (par défaut: 1d)')
    parser.add_argument('--episodes', type=int, default=5, 
                        help='Nombre d\'épisodes de trading à simuler (par défaut: 5)')
    parser.add_argument('--window', type=int, default=30, 
                        help='Taille de la fenêtre pour les calculs de récompense (par défaut: 30)')
    parser.add_argument('--plot', action='store_true', help='Affiche les graphiques des métriques')
    
    return parser.parse_args()

def prepare_market_data(symbol, days, interval):
    """Prépare les données de marché pour la simulation."""
    # Récupérer les données
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    fetcher = MarketDataFetcher()
    df = fetcher.get_crypto_price_data(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval=interval
    )
    
    # Prétraiter les données
    preprocessor = EnhancedMarketDataPreprocessor()
    df = preprocessor.add_features(df)
    
    print(f"Données préparées: {len(df)} points de données pour {symbol}")
    return df

def run_simulation_with_reward(env, reward_calculator, num_episodes, reward_name):
    """Exécute une simulation avec un calculateur de récompense spécifique."""
    # Métriques à suivre
    portfolio_values = []
    rewards = []
    actions = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        episode_values = []
        episode_actions = []
        
        # Réinitialiser le calculateur de récompense
        reward_calculator.reset()
        
        while not done:
            # Action aléatoire pour simplifier (0=hold, 1=buy, 2=sell)
            action = np.random.choice([0, 1, 2])
            
            # Exécuter l'action dans l'environnement
            next_state, env_reward, done, info = env.step(action)
            
            # Calculer la récompense avec le calculateur spécifique
            if isinstance(reward_calculator, AdvancedRewardCalculator):
                portfolio_value = env.get_portfolio_value()
                position_size = env.portfolio.position_value
                reward = reward_calculator.calculate_combined_reward(
                    portfolio_value, action, position_size
                )
            elif isinstance(reward_calculator, SharpeRatioReward):
                # Calculer le rendement pour le ratio de Sharpe
                if len(episode_values) > 0:
                    current_return = (env.get_portfolio_value() - episode_values[-1]) / episode_values[-1]
                else:
                    current_return = 0
                reward = reward_calculator.calculate(current_return)
            elif isinstance(reward_calculator, TransactionCostReward):
                # Utiliser le PnL comme base et appliquer les coûts de transaction
                position_value = env.portfolio.position_value
                reward = reward_calculator.calculate(env_reward, action, position_value)
            elif isinstance(reward_calculator, DrawdownReward):
                # Utiliser le PnL et ajouter une pénalité basée sur le drawdown
                portfolio_value = env.get_portfolio_value()
                reward = reward_calculator.calculate(env_reward, portfolio_value)
            
            # Enregistrer les métriques
            episode_rewards.append(reward)
            episode_values.append(env.get_portfolio_value())
            episode_actions.append(action)
            
            # Passer à l'état suivant
            state = next_state
        
        # Ajouter les métriques de l'épisode aux listes globales
        portfolio_values.append(episode_values)
        rewards.append(episode_rewards)
        actions.append(episode_actions)
        
        print(f"Épisode {episode+1}/{num_episodes} avec {reward_name} - "
              f"Valeur finale: {episode_values[-1]:.2f}, "
              f"Récompense totale: {sum(episode_rewards):.2f}")
    
    return portfolio_values, rewards, actions

def plot_metrics(portfolio_values_dict, rewards_dict):
    """Affiche des graphiques comparant les différentes stratégies de récompense."""
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Graphique des valeurs de portefeuille
    for reward_name, portfolio_values in portfolio_values_dict.items():
        # Utiliser le dernier épisode pour le graphique
        axs[0].plot(portfolio_values[-1], label=reward_name)
    
    axs[0].set_title('Valeurs de portefeuille par stratégie de récompense')
    axs[0].set_xlabel('Étapes')
    axs[0].set_ylabel('Valeur du portefeuille')
    axs[0].legend()
    axs[0].grid(True)
    
    # Graphique des récompenses cumulées
    for reward_name, rewards in rewards_dict.items():
        # Utiliser le dernier épisode pour le graphique
        cumulative_rewards = np.cumsum(rewards[-1])
        axs[1].plot(cumulative_rewards, label=reward_name)
    
    axs[1].set_title('Récompenses cumulées par stratégie')
    axs[1].set_xlabel('Étapes')
    axs[1].set_ylabel('Récompense cumulée')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def advanced_rewards_demo():
    """Fonction principale de démonstration des récompenses avancées."""
    # Analyser les arguments
    args = parse_arguments()
    
    # Préparer les données
    df = prepare_market_data(args.symbol, args.days, args.interval)
    
    # Créer l'environnement de trading
    env = TradingEnvironment(
        data=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=10
    )
    
    # Initialiser les différents calculateurs de récompense
    advanced_calculator = AdvancedRewardCalculator(
        risk_free_rate=0.02,
        window_size=args.window,
        transaction_cost=0.001,
        frequency_penalty=0.0005,
        max_drawdown_penalty=2.0
    )
    
    sharpe_calculator = SharpeRatioReward(
        risk_free_rate=0.02,
        window_size=args.window,
        annualize=False
    )
    
    transaction_calculator = TransactionCostReward(
        base_cost=0.001,
        frequency_penalty=0.0005,
        window_size=args.window
    )
    
    drawdown_calculator = DrawdownReward(
        penalty_factor=2.0,
        window_size=args.window
    )
    
    # Exécuter des simulations avec chaque type de récompense
    print("\n=== Simulation avec récompense combinée ===")
    portfolio_values_combined, rewards_combined, actions_combined = run_simulation_with_reward(
        env, advanced_calculator, args.episodes, "Combinée"
    )
    
    print("\n=== Simulation avec récompense basée sur le ratio de Sharpe ===")
    portfolio_values_sharpe, rewards_sharpe, actions_sharpe = run_simulation_with_reward(
        env, sharpe_calculator, args.episodes, "Sharpe"
    )
    
    print("\n=== Simulation avec récompense incluant les coûts de transaction ===")
    portfolio_values_transaction, rewards_transaction, actions_transaction = run_simulation_with_reward(
        env, transaction_calculator, args.episodes, "Transaction"
    )
    
    print("\n=== Simulation avec récompense basée sur le drawdown ===")
    portfolio_values_drawdown, rewards_drawdown, actions_drawdown = run_simulation_with_reward(
        env, drawdown_calculator, args.episodes, "Drawdown"
    )
    
    # Compiler les résultats
    portfolio_values_dict = {
        "Combinée": portfolio_values_combined,
        "Sharpe": portfolio_values_sharpe,
        "Transaction": portfolio_values_transaction,
        "Drawdown": portfolio_values_drawdown
    }
    
    rewards_dict = {
        "Combinée": rewards_combined,
        "Sharpe": rewards_sharpe,
        "Transaction": rewards_transaction,
        "Drawdown": rewards_drawdown
    }
    
    # Afficher les graphiques si demandé
    if args.plot:
        plot_metrics(portfolio_values_dict, rewards_dict)
    
    # Résumé des performances
    print("\n=== Résumé des performances ===")
    for name, values in portfolio_values_dict.items():
        # Calculer la performance moyenne sur tous les épisodes
        final_values = [episode_values[-1] for episode_values in values]
        avg_final_value = np.mean(final_values)
        initial_value = values[0][0]  # Valeur initiale du premier épisode
        avg_return = (avg_final_value - initial_value) / initial_value * 100
        
        print(f"Stratégie {name}: Valeur finale moyenne = {avg_final_value:.2f}, "
              f"Rendement moyen = {avg_return:.2f}%")

if __name__ == "__main__":
    advanced_rewards_demo() 