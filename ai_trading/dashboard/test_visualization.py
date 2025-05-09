"""
Script de test pour les visualisations 3D et l'analyse post-mortem.

Ce script permet de générer des données de test et de visualiser les résultats
des nouvelles fonctionnalités implémentées.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_trading.dashboard.visualization_3d import Visualizer3D
from ai_trading.dashboard.trade_analysis import TradeAnalyzer


def generate_test_market_data(n_days=100):
    """
    Génère des données de marché synthétiques pour les tests.
    
    Args:
        n_days: Nombre de jours de données à générer
        
    Returns:
        DataFrame avec les données de marché
    """
    # Créer les dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Générer les prix avec un mouvement brownien
    np.random.seed(42)  # Pour la reproductibilité
    returns = np.random.normal(0.0005, 0.01, n_days+1)
    price = 100 * (1 + returns).cumprod()
    
    # Générer le volume
    volume = np.random.lognormal(10, 1, n_days+1)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'close': price,
        'open': price * np.random.uniform(0.99, 1.01, n_days+1),
        'high': price * np.random.uniform(1.01, 1.03, n_days+1),
        'low': price * np.random.uniform(0.97, 0.99, n_days+1),
        'volume': volume
    })
    
    # Ajouter des indicateurs techniques
    df['rsi'] = np.random.uniform(30, 70, n_days+1)
    df['macd'] = np.random.normal(0, 1, n_days+1)
    df['ema20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['volatility'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
    df['atr'] = np.random.uniform(0.5, 3, n_days+1)
    df['momentum'] = df['close'].pct_change(periods=10)
    
    return df


def generate_test_trades_data(market_data, n_trades=50):
    """
    Génère des données de transactions synthétiques pour les tests.
    
    Args:
        market_data: DataFrame avec les données de marché
        n_trades: Nombre de transactions à générer
        
    Returns:
        DataFrame avec les données de transactions
    """
    np.random.seed(42)  # Pour la reproductibilité
    
    # Sélectionner des dates aléatoires pour les entrées
    entry_indices = np.random.choice(range(len(market_data) - 10), n_trades, replace=False)
    entry_times = market_data.iloc[entry_indices]['timestamp'].values
    
    # Générer les durées de détention (en nombre entier de jours)
    holding_periods = np.random.lognormal(3, 1, n_trades).astype(int)
    
    # Calculer les dates de sortie
    exit_times = []
    for i in range(n_trades):
        exit_times.append(entry_times[i] + pd.Timedelta(days=int(holding_periods[i])))
    
    # Générer les actifs
    symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'BNB']
    trade_symbols = np.random.choice(symbols, n_trades)
    
    # Générer les directions
    directions = np.random.choice(['buy', 'sell'], n_trades)
    
    # Générer les prix d'entrée et de sortie
    entry_prices = np.random.uniform(1000, 5000, n_trades)
    
    # Les profits dépendent de la direction
    profits = np.random.normal(50, 200, n_trades)
    
    # Ajuster les profits en fonction de la direction
    for i in range(n_trades):
        if directions[i] == 'sell':
            profits[i] = -profits[i]
    
    # Calculer les prix de sortie
    exit_prices = entry_prices + profits
    
    # Générer les quantités
    quantities = np.random.uniform(0.1, 2, n_trades)
    
    # Calculer les valeurs d'entrée et les profits en pourcentage
    entry_values = entry_prices * quantities
    profit_pct = (profits / entry_values) * 100
    
    # Statuts des trades
    statuses = ['closed'] * n_trades
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'entry_time': entry_times,
        'exit_time': exit_times,
        'symbol': trade_symbols,
        'direction': directions,
        'entry_price': entry_prices,
        'exit_price': exit_prices,
        'quantity': quantities,
        'entry_value': entry_values,
        'profit': profits,
        'profit_pct': profit_pct,
        'status': statuses,
        'holding_period': holding_periods * 24,  # Convertir en heures
        'entry_volatility': np.random.uniform(0.01, 0.05, n_trades)
    })
    
    return df


def test_visualizer_3d():
    """Teste les fonctionnalités de visualisation 3D."""
    print("Test des visualisations 3D...")
    
    # Générer des données de test
    market_data = generate_test_market_data(100)
    trades_data = generate_test_trades_data(market_data, 50)
    
    # Créer le visualiseur
    visualizer = Visualizer3D()
    
    # Tester la surface d'indicateurs
    print("Création de la surface d'indicateurs...")
    fig1 = visualizer.create_multi_indicator_surface(
        market_data, 'rsi', 'macd', 'close'
    )
    
    # Tester la trajectoire du portefeuille
    print("Création de la trajectoire du portefeuille...")
    fig2 = visualizer.create_portfolio_trajectory(
        market_data, window=30, n_components=3
    )
    
    # Tester les clusters de trades
    print("Création des clusters de trades...")
    fig3 = visualizer.create_trade_clusters_3d(
        trades_data, n_clusters=5
    )
    
    # Enregistrer les figures
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    fig1.write_html(os.path.join(output_dir, "indicator_surface_3d.html"))
    fig2.write_html(os.path.join(output_dir, "portfolio_trajectory_3d.html"))
    fig3.write_html(os.path.join(output_dir, "trade_clusters_3d.html"))
    
    print(f"Visualisations 3D sauvegardées dans {output_dir}")


def test_trade_analyzer():
    """Teste les fonctionnalités d'analyse post-mortem des trades."""
    print("Test de l'analyse post-mortem des trades...")
    
    # Générer des données de test avec plus de jours pour permettre plus de trades
    market_data = generate_test_market_data(200)
    
    # Limiter le nombre de trades à la taille de la population
    n_trades = min(100, len(market_data) - 10)
    trades_data = generate_test_trades_data(market_data, n_trades)
    
    # Créer l'analyseur
    analyzer = TradeAnalyzer()
    
    # Tester le résumé de performance
    print("Création du résumé de performance...")
    fig1 = analyzer.create_performance_summary(trades_data)
    
    # Tester l'analyse des gains et pertes
    print("Création de l'analyse des gains et pertes...")
    fig2 = analyzer.create_win_loss_analysis(trades_data)
    
    # Tester l'attribution de performance
    print("Création de l'analyse d'attribution...")
    fig3 = analyzer.create_trade_attribution(trades_data, market_data)
    
    # Enregistrer les figures
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    fig1.write_html(os.path.join(output_dir, "performance_summary.html"))
    fig2.write_html(os.path.join(output_dir, "win_loss_analysis.html"))
    fig3.write_html(os.path.join(output_dir, "trade_attribution.html"))
    
    print(f"Analyses post-mortem sauvegardées dans {output_dir}")


if __name__ == "__main__":
    test_visualizer_3d()
    test_trade_analyzer()
    print("Tests terminés avec succès!") 