"""
Exemple d'environnement de trading multi-actifs pour l'apprentissage par renforcement.

Ce script montre comment configurer un environnement de trading multi-actifs
pour l'entraînement d'agents RL.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import gym
from gym import spaces

# Import des modules RL
from ai_trading.rl.environments.multi_asset_environment import MultiAssetTradingEnvironment
from ai_trading.rl.environments.market_simulator import MarketSimulator

def load_multi_asset_data():
    """Charge des données de marché simulées pour plusieurs actifs."""
    # Période de simulation
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Dictionnaire pour stocker les données des actifs
    asset_data = {}
    
    # Simuler différents actifs avec des corrélations variées
    
    # BTC (tendance haussière avec forte volatilité)
    btc_trend = np.concatenate([
        np.linspace(30000, 40000, n_days//3),
        np.linspace(40000, 35000, n_days//3),
        np.linspace(35000, 60000, n_days - 2*(n_days//3))
    ])
    btc_volatility = np.random.normal(0, 1500, n_days)
    btc_prices = btc_trend + btc_volatility
    
    # ETH (corrélé avec BTC mais avec plus de volatilité à certaines périodes)
    eth_base = 0.8 * btc_prices / 10  # Corrélation de base avec BTC
    eth_specific = np.concatenate([
        np.linspace(2000, 2500, n_days//3),
        np.linspace(2500, 2200, n_days//3),
        np.linspace(2200, 4000, n_days - 2*(n_days//3))
    ])
    eth_volatility = np.random.normal(0, 150, n_days)
    eth_prices = eth_base + eth_specific * 0.2 + eth_volatility
    
    # XRP (moins corrélé, parfois contre-tendance)
    xrp_base = 0.4 * btc_prices / 30000  # Faible corrélation avec BTC
    xrp_specific = np.concatenate([
        np.linspace(0.5, 0.7, n_days//3),
        np.linspace(0.7, 1.2, n_days//3),
        np.linspace(1.2, 0.8, n_days - 2*(n_days//3))
    ])
    xrp_volatility = np.random.normal(0, 0.05, n_days)
    xrp_prices = xrp_base + xrp_specific * 0.6 + xrp_volatility
    
    # DOT (nouvel actif avec forte croissance)
    dot_specific = np.concatenate([
        np.linspace(8, 15, n_days//3),
        np.linspace(15, 25, n_days//3),
        np.linspace(25, 18, n_days - 2*(n_days//3))
    ])
    dot_volatility = np.random.normal(0, 1.2, n_days)
    dot_prices = dot_specific + dot_volatility
    
    # Créer les DataFrames pour chaque actif
    assets = {
        'BTC/USD': btc_prices,
        'ETH/USD': eth_prices,
        'XRP/USD': xrp_prices,
        'DOT/USD': dot_prices
    }
    
    for symbol, prices in assets.items():
        # Normaliser pour éviter les valeurs négatives
        prices = np.maximum(prices, prices.min() * 0.1)
        
        # Créer les données OHLCV
        data = pd.DataFrame({
            'open': prices - np.random.uniform(0, prices * 0.01, n_days),
            'high': prices + np.random.uniform(prices * 0.005, prices * 0.02, n_days),
            'low': prices - np.random.uniform(prices * 0.005, prices * 0.02, n_days),
            'close': prices,
            'volume': np.random.uniform(prices * 100, prices * 1000, n_days)
        }, index=dates)
        
        asset_data[symbol] = data
    
    return asset_data

def add_correlation_features(asset_data):
    """Ajoute des caractéristiques basées sur les corrélations entre actifs."""
    # Extraire les prix de clôture de tous les actifs
    close_prices = pd.DataFrame({symbol: data['close'] for symbol, data in asset_data.items()})
    
    # Calculer les rendements
    returns = close_prices.pct_change().dropna()
    
    # Calculer les corrélations roulantes sur 20 jours
    for symbol1 in close_prices.columns:
        for symbol2 in close_prices.columns:
            if symbol1 != symbol2:
                # Nom de la colonne de corrélation
                corr_name = f'corr_{symbol1.split("/")[0]}_{symbol2.split("/")[0]}'
                
                # Calculer la corrélation roulante
                corr = returns[symbol1].rolling(window=20).corr(returns[symbol2])
                
                # Ajouter à chaque actif
                asset_data[symbol1][corr_name] = corr
                
    # Remplir les NaN avec zéro
    for symbol, data in asset_data.items():
        asset_data[symbol] = data.fillna(0)
    
    return asset_data

def add_technical_indicators(asset_data):
    """Ajoute des indicateurs techniques à chaque actif."""
    for symbol, data in asset_data.items():
        df = data.copy()
        
        # Moyennes mobiles
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bandes de Bollinger
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * std
        df['bb_lower'] = df['bb_middle'] - 2 * std
        
        # Momentum et volatilité
        df['momentum'] = df['close'].pct_change(periods=10)
        df['volatility'] = df['close'].rolling(window=10).std() / df['close'].rolling(window=10).mean()
        
        # Remplacer les NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Mettre à jour les données
        asset_data[symbol] = df
    
    return asset_data

def create_multi_asset_environment(asset_data):
    """Crée un environnement de trading multi-actifs."""
    # Extraire les caractéristiques pour chaque actif
    features = {
        symbol: [
            'sma_10', 'sma_30', 'ema_10', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower', 'momentum', 'volatility'
        ] + [col for col in data.columns if col.startswith('corr_')]
        for symbol, data in asset_data.items()
    }
    
    # Créer un simulateur de marché avec gestion de la liquidité et du slippage
    market_simulator = MarketSimulator(
        slippage_model='volume_based',
        liquidity_parameters={
            'BTC/USD': {'depth': 1000000, 'impact_factor': 0.1},
            'ETH/USD': {'depth': 500000, 'impact_factor': 0.15},
            'XRP/USD': {'depth': 200000, 'impact_factor': 0.2},
            'DOT/USD': {'depth': 100000, 'impact_factor': 0.25}
        },
        execution_delay=1  # Délai d'exécution d'un pas de temps
    )
    
    # Créer l'environnement multi-actifs
    env = MultiAssetTradingEnvironment(
        asset_data=asset_data,
        features=features,
        window_size=20,
        initial_balance=100000.0,
        commission=0.001,
        assets=['BTC/USD', 'ETH/USD', 'XRP/USD', 'DOT/USD'],
        max_position_size=0.25,  # Maximum 25% du portefeuille par actif
        reward_function='sortino',  # Utiliser le ratio de Sortino
        risk_free_rate=0.0,
        market_simulator=market_simulator,
        enable_stop_loss=True,
        stop_loss_percentage=0.05,  # 5% de stop-loss
        include_total_balance=True,  # Inclure le solde total dans l'état
        include_market_indicators=True,  # Inclure des indicateurs de marché globaux
        normalize_features=True  # Normaliser les caractéristiques
    )
    
    return env

def test_environment(env, n_steps=100):
    """Teste l'environnement avec des actions aléatoires."""
    # Réinitialiser l'environnement
    state = env.reset()
    
    # Historique des étapes
    states = [state]
    actions = []
    rewards = []
    balances = [env.total_balance]
    positions = [env.positions.copy()]
    
    # Effectuer des actions aléatoires
    for _ in range(n_steps):
        # Générer une action aléatoire
        action = env.action_space.sample()
        
        # Exécuter l'action
        next_state, reward, done, info = env.step(action)
        
        # Enregistrer les résultats
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        balances.append(info.get('total_balance', 0))
        positions.append(info.get('positions', {}).copy())
        
        # Mettre à jour l'état
        state = next_state
        
        if done:
            break
    
    return states, actions, rewards, balances, positions

def visualize_portfolio(balances, positions, asset_data):
    """Visualise l'évolution du portefeuille."""
    # Créer un DataFrame pour l'historique des positions
    position_history = pd.DataFrame(positions)
    
    # Créer un graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Tracer l'équilibre du portefeuille
    ax1.plot(balances, label='Valeur totale du portefeuille')
    ax1.set_title('Évolution de la valeur du portefeuille')
    ax1.set_xlabel('Pas de temps')
    ax1.set_ylabel('Valeur ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Tracer les allocations d'actifs
    if not position_history.empty:
        assets = [col for col in position_history.columns if '/' in col]
        for asset in assets:
            ax2.plot(position_history[asset], label=asset)
    
    ax2.set_title('Évolution des positions par actif')
    ax2.set_xlabel('Pas de temps')
    ax2.set_ylabel('Position ($)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Enregistrer le graphique
    plt.savefig("examples/rl/environments/multi_asset_portfolio.png")
    plt.close()
    
    print("Graphique du portefeuille enregistré sous 'examples/rl/environments/multi_asset_portfolio.png'")

def multi_asset_environment_example():
    """Exemple principal de l'environnement de trading multi-actifs."""
    # Charger les données multi-actifs
    print("Chargement des données multi-actifs...")
    asset_data = load_multi_asset_data()
    
    # Ajouter des caractéristiques de corrélation
    print("Ajout des caractéristiques de corrélation...")
    asset_data = add_correlation_features(asset_data)
    
    # Ajouter des indicateurs techniques
    print("Ajout des indicateurs techniques...")
    asset_data = add_technical_indicators(asset_data)
    
    # Créer l'environnement
    print("Création de l'environnement de trading multi-actifs...")
    env = create_multi_asset_environment(asset_data)
    
    # Afficher les informations sur l'environnement
    print(f"\n== Informations sur l'environnement ==")
    print(f"Espace d'observation: {env.observation_space}")
    print(f"Espace d'action: {env.action_space}")
    print(f"Nombre d'actifs: {len(env.assets)}")
    print(f"Taille de fenêtre d'observation: {env.window_size}")
    
    # Tester l'environnement
    print("\nTest de l'environnement avec des actions aléatoires...")
    states, actions, rewards, balances, positions = test_environment(env, n_steps=200)
    
    # Afficher les résultats
    print(f"\n== Résultats du test ==")
    print(f"Nombre d'étapes: {len(rewards)}")
    print(f"Récompense totale: {sum(rewards):.2f}")
    print(f"Récompense moyenne: {np.mean(rewards):.2f}")
    print(f"Valeur finale du portefeuille: ${balances[-1]:.2f}")
    
    # Analyser les rendements
    returns = np.diff(balances) / balances[:-1]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    max_drawdown = (np.min(balances) / np.max(balances) - 1) if max(balances) > 0 else 0
    
    print(f"Ratio de Sharpe: {sharpe:.2f}")
    print(f"Drawdown maximum: {max_drawdown:.2%}")
    
    # Visualiser le portefeuille
    visualize_portfolio(balances, positions, asset_data)
    
    return env, states, actions, rewards, balances, positions

if __name__ == "__main__":
    env, states, actions, rewards, balances, positions = multi_asset_environment_example() 