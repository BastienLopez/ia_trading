import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

# Définir le chemin pour les visualisations
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations', 'performance')
# Créer le répertoire s'il n'existe pas
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def calculate_metrics(portfolio_values, risk_free_rate=0.01):
    """
    Calcule les métriques de performance d'une stratégie.
    
    Args:
        portfolio_values (list): Historique des valeurs du portefeuille
        risk_free_rate (float): Taux sans risque annualisé
        
    Returns:
        dict: Métriques de performance
    """
    # Convertir en array numpy
    portfolio_values = np.array(portfolio_values)
    
    # Calculer les rendements quotidiens
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Rendement total
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    
    # Rendement annualisé (supposant des données quotidiennes)
    annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    
    # Volatilité annualisée
    volatility = returns.std() * np.sqrt(252)
    
    # Ratio de Sharpe
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Drawdown maximum
    cumulative_returns = portfolio_values / portfolio_values[0]
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    max_drawdown = drawdown.max()
    
    # Ratio de Sortino (utilisant uniquement les rendements négatifs)
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Ratio de Calmar
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # Nombre de trades
    # Cette métrique nécessite des données supplémentaires sur les trades
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio
    }

def compare_strategies(df, strategies, initial_balance=10000, window_size=50):
    """
    Compare différentes stratégies de trading.
    
    Args:
        df (pd.DataFrame): Données de marché
        strategies (dict): Dictionnaire de stratégies à comparer
        initial_balance (float): Solde initial
        window_size (int): Taille de la fenêtre d'observation
        
    Returns:
        dict: Métriques de performance pour chaque stratégie
    """
    results = {}
    
    # Stratégie Buy & Hold (référence)
    buy_hold_values = []
    initial_price = df.iloc[window_size]['close']
    crypto_amount = initial_balance / initial_price
    
    for i in range(window_size, len(df)):
        current_price = df.iloc[i]['close']
        portfolio_value = crypto_amount * current_price
        buy_hold_values.append(portfolio_value)
    
    results['buy_hold'] = {
        'portfolio_values': buy_hold_values,
        'metrics': calculate_metrics(buy_hold_values)
    }
    
    # Évaluer chaque stratégie
    for name, strategy in strategies.items():
        portfolio_values = strategy['evaluate'](df, initial_balance, window_size)
        metrics = calculate_metrics(portfolio_values)
        
        results[name] = {
            'portfolio_values': portfolio_values,
            'metrics': metrics
        }
    
    # Visualiser les résultats
    plot_strategy_comparison(results)
    
    # Créer un tableau comparatif des métriques
    metrics_df = pd.DataFrame({name: result['metrics'] for name, result in results.items()})
    
    # Sauvegarder les métriques
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_df.to_csv(os.path.join(VISUALIZATION_DIR, f'metrics_comparison_{timestamp}.csv'))
    
    return results, metrics_df

def plot_strategy_comparison(results, title="Comparaison des stratégies"):
    """
    Trace la comparaison de différentes stratégies d'investissement.
    
    Args:
        results (dict): Résultats pour chaque stratégie
        title (str): Titre du graphique
    """
    plt.figure(figsize=(12, 8))
    
    # Tracer chaque stratégie
    for strategy_name, data in results.items():
        if "portfolio_values" in data:
            plt.plot(data["portfolio_values"], label=f"{strategy_name}")
    
    # Ajouter la référence (buy and hold)
    if "buy_hold" in results:
        plt.plot(results["buy_hold"]["portfolio_values"], linestyle="--", label="Buy & Hold", color="black")
    
    plt.title(title)
    plt.xlabel("Période")
    plt.ylabel("Valeur du portefeuille")
    plt.legend()
    plt.grid(True)
    
    # Ajouter un horodatage
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(VISUALIZATION_DIR, f'strategy_comparison_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

# Exemple d'utilisation
if __name__ == "__main__":
    from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
    from ai_trading.rl.trading_environment import TradingEnvironment
    from ai_trading.rl.dqn_agent import DQNAgent
    
    # Charger les données
    collector = EnhancedDataCollector()
    
    # Utiliser une approche alternative pour charger les données
    import pandas as pd
    import numpy as np

    # Créer des données synthétiques pour les tests
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
    prices = np.linspace(5000, 50000, 1000) + np.random.normal(0, 2000, 1000)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.uniform(0, 500, 1000),
            "low": prices - np.random.uniform(0, 500, 1000),
            "close": prices + np.random.normal(0, 200, 1000),
            "volume": np.random.uniform(1000, 5000, 1000),
        },
        index=dates,
    )
    
    # Définir les stratégies à comparer
    def evaluate_dqn(df, initial_balance, window_size):
        # Créer l'environnement
        env = TradingEnvironment(df=df, initial_balance=initial_balance, window_size=window_size, use_risk_manager=True)
        
        # Créer et entraîner l'agent
        state_size = len(env.reset()[0])
        action_size = env.action_space.n
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        
        # Entraîner l'agent (code simplifié)
        # ...
        
        # Évaluer l'agent
        env.reset()
        portfolio_values = []
        
        done = False
        state = np.reshape(env.reset()[0], [1, state_size])
        
        while not done:
            action = agent.act(state, use_epsilon=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, [1, state_size])
            portfolio_values.append(env.get_portfolio_value())
        
        return portfolio_values
    
    # Définir d'autres stratégies (MACD, RSI, etc.)
    # ...
    
    strategies = {
        'DQN': {
            'evaluate': evaluate_dqn
        },
        # Ajouter d'autres stratégies
    }
    
    # Comparer les stratégies
    results, metrics_df = compare_strategies(df, strategies)
    
    print(metrics_df) 