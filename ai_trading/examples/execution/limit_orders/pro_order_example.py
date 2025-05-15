"""
Exemple d'exécution d'ordres limites et stop pour le trading crypto.

Ce script démontre comment utiliser le système d'ordres professionnels
pour placer et gérer des ordres limites, stop-loss et take-profit.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Import des modules d'exécution
from ai_trading.execution.order_manager import OrderManager
from ai_trading.execution.order_types import LimitOrder, StopOrder, StopLimitOrder, TrailingStopOrder
from ai_trading.execution.execution_simulator import ExecutionSimulator
from ai_trading.execution.performance_analyzer import ExecutionPerformanceAnalyzer

def load_market_data():
    """Charge des données de marché simulées pour tester les ordres."""
    # Période de simulation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Simuler des prix avec différentes phases de marché
    n = len(dates)
    
    # Créer une tendance avec volatilité
    base_price = 50000
    trend = np.concatenate([
        np.linspace(0, 0.02, n//4),           # Légère hausse
        np.linspace(0.02, -0.03, n//4),       # Baisse
        np.linspace(-0.03, -0.01, n//4),      # Rebond léger
        np.linspace(-0.01, 0.04, n - 3*(n//4)) # Forte hausse
    ])
    
    # Ajouter de la volatilité
    volatility = np.random.normal(0, 0.002, n)
    cumulative_returns = np.cumsum(trend + volatility)
    prices = base_price * np.exp(cumulative_returns)
    
    # Créer les données OHLCV
    data = pd.DataFrame({
        'open': prices * (1 - np.random.uniform(0, 0.001, n)),
        'high': prices * (1 + np.random.uniform(0.001, 0.003, n)),
        'low': prices * (1 - np.random.uniform(0.001, 0.003, n)),
        'close': prices,
        'volume': np.random.uniform(5, 20, n) * (1 + 0.5 * np.sin(np.linspace(0, 10, n)))
    }, index=dates)
    
    return data

def pro_order_example():
    """Exemple d'utilisation des ordres professionnels."""
    # Charger les données de marché
    print("Chargement des données de marché...")
    market_data = load_market_data()
    
    # Créer le dossier de résultats
    output_dir = "examples/execution/limit_orders/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser le simulateur d'exécution
    simulator = ExecutionSimulator(
        market_data=market_data,
        slippage_model='volume_based',
        fee_model={'maker': 0.0010, 'taker': 0.0020},
        liquidity_profile='medium'
    )
    
    # Initialiser le gestionnaire d'ordres
    order_manager = OrderManager(
        executor=simulator,
        position_limit=10.0,  # Maximum 10 BTC
        risk_limit_pct=2.0,   # Risque maximum de 2% du capital
        initial_capital=100000.0  # Capital initial de 100,000 USD
    )
    
    # Définir différents scénarios d'ordres
    print("\nDéfinition des scénarios d'ordres...")
    
    # Point de départ pour les scénarios
    start_idx = 100  # Commencer après quelques périodes
    
    # Scénario 1: Ordre limite simple d'achat
    print("\n=== Scénario 1: Ordre limite simple d'achat ===")
    entry_price_1 = market_data['close'].iloc[start_idx] * 0.99  # 1% en dessous du prix actuel
    limit_buy_1 = LimitOrder(
        symbol="BTC/USD",
        side="buy",
        quantity=1.0,
        price=entry_price_1,
        time_in_force="GTC",  # Good Till Canceled
        post_only=True  # Ordre maker uniquement
    )
    
    # Scénario 2: Ordre d'achat au marché avec stop-loss et take-profit
    print("\n=== Scénario 2: Ordre d'achat au marché avec SL et TP ===")
    market_buy = order_manager.create_market_order(
        symbol="BTC/USD",
        side="buy",
        quantity=0.5
    )
    
    # Ajouter stop-loss (-2%) et take-profit (+3%)
    current_price = market_data['close'].iloc[start_idx + 50]
    stop_loss = StopOrder(
        symbol="BTC/USD",
        side="sell",
        quantity=0.5,
        stop_price=current_price * 0.98,
        reduce_only=True  # Ne ferme que la position existante
    )
    
    take_profit = LimitOrder(
        symbol="BTC/USD",
        side="sell",
        quantity=0.5,
        price=current_price * 1.03,
        reduce_only=True
    )
    
    # Scénario 3: Ordre stop-limit pour entrer en position
    print("\n=== Scénario 3: Ordre stop-limit pour entrer en position ===")
    trigger_price = market_data['close'].iloc[start_idx + 100] * 1.01  # 1% au-dessus
    limit_price = trigger_price * 1.002  # 0.2% de buffer
    
    stop_limit_buy = StopLimitOrder(
        symbol="BTC/USD",
        side="buy",
        quantity=0.2,
        stop_price=trigger_price,
        limit_price=limit_price,
        time_in_force="GTC"
    )
    
    # Scénario 4: Trailing stop pour sortir de position
    print("\n=== Scénario 4: Trailing stop pour sortir de position ===")
    trailing_stop = TrailingStopOrder(
        symbol="BTC/USD",
        side="sell",
        quantity=0.2,
        activation_price=market_data['close'].iloc[start_idx + 150],
        callback_rate=0.01,  # 1% de repli
        reduce_only=True
    )
    
    # Exécuter les scénarios dans le simulateur
    print("\nExécution des scénarios...")
    
    # Réinitialiser le simulateur
    simulator.reset()
    
    # Enregistrer les ordres
    order_ids = []
    order_ids.append(order_manager.place_order(limit_buy_1, start_idx))
    order_ids.append(order_manager.place_order(market_buy, start_idx + 50))
    order_ids.append(order_manager.place_order(stop_loss, start_idx + 50))
    order_ids.append(order_manager.place_order(take_profit, start_idx + 50))
    order_ids.append(order_manager.place_order(stop_limit_buy, start_idx + 100))
    order_ids.append(order_manager.place_order(trailing_stop, start_idx + 150))
    
    # Exécuter la simulation
    simulator.run_full_simulation(order_manager)
    
    # Récupérer les résultats
    trades = simulator.get_executed_trades()
    orders = simulator.get_order_history()
    portfolio = simulator.get_portfolio_history()
    
    # Analyser les performances
    analyzer = ExecutionPerformanceAnalyzer(trades, orders, portfolio, market_data)
    performance = analyzer.calculate_performance()
    
    # Afficher les résultats
    print("\n=== Résultats de la simulation ===")
    print(f"Nombre d'ordres placés: {len(orders)}")
    print(f"Nombre de trades exécutés: {len(trades)}")
    print(f"Valeur finale du portefeuille: ${portfolio[-1]['total_value']:.2f}")
    print(f"P&L total: ${performance['pnl_total']:.2f}")
    print(f"Rendement total: {performance['return_pct']:.2%}")
    print(f"Taux de succès: {performance['win_rate']:.2%}")
    print(f"Ratio profit/perte: {performance['profit_loss_ratio']:.2f}")
    print(f"Slippage moyen: {performance['avg_slippage']:.4%}")
    print(f"Frais totaux: ${performance['total_fees']:.2f}")
    
    # Visualiser les ordres et les trades
    plt.figure(figsize=(12, 8))
    
    # Graphique des prix et trades
    plt.subplot(2, 1, 1)
    
    # Tracer les prix
    plt.plot(market_data.index, market_data['close'], label='Prix', color='blue')
    
    # Tracer les trades (exécutions)
    for trade in trades:
        color = 'green' if trade['side'] == 'buy' else 'red'
        marker = '^' if trade['side'] == 'buy' else 'v'
        plt.scatter(market_data.index[trade['timestamp']], trade['price'], 
                   color=color, marker=marker, s=100)
    
    # Tracer les ordres limites non exécutés
    for order in orders:
        if order['status'] in ['active', 'canceled'] and order['type'] == 'limit':
            plt.axhline(y=order['price'], color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Prix et Exécution des Ordres')
    plt.ylabel('Prix (USD)')
    plt.grid(True)
    
    # Graphique de la valeur du portefeuille
    plt.subplot(2, 1, 2)
    
    # Extraire l'historique de la valeur du portefeuille
    port_values = [p['total_value'] for p in portfolio]
    port_timestamps = [market_data.index[p['timestamp']] for p in portfolio]
    
    plt.plot(port_timestamps, port_values, label='Valeur du portefeuille', color='purple')
    plt.axhline(y=100000, color='gray', linestyle='--', label='Capital initial')
    
    plt.title('Évolution de la Valeur du Portefeuille')
    plt.xlabel('Date/Heure')
    plt.ylabel('Valeur (USD)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Enregistrer le graphique
    plt.savefig(f"{output_dir}/pro_order_simulation.png")
    plt.close()
    
    # Créer un graphique détaillé des ordres
    plt.figure(figsize=(14, 10))
    
    # Sous-graphique principal pour les prix et ordres
    ax1 = plt.subplot(3, 1, 1)
    
    # Tracer les chandeliers
    for i in range(len(market_data)):
        # Couleur du chandelier
        color = 'green' if market_data['close'].iloc[i] > market_data['open'].iloc[i] else 'red'
        
        # Corps du chandelier
        plt.plot([market_data.index[i], market_data.index[i]], 
                [market_data['open'].iloc[i], market_data['close'].iloc[i]], 
                color=color, linewidth=3)
        
        # Mèche haute/basse
        plt.plot([market_data.index[i], market_data.index[i]], 
                [market_data['low'].iloc[i], market_data['high'].iloc[i]], 
                color=color, linewidth=1)
    
    # Ajouter les points d'entrée/sortie
    for trade in trades:
        color = 'green' if trade['side'] == 'buy' else 'red'
        marker = '^' if trade['side'] == 'buy' else 'v'
        label = f"{trade['side'].capitalize()} {trade['quantity']} @ {trade['price']:.2f}"
        plt.scatter(market_data.index[trade['timestamp']], trade['price'], 
                   color=color, marker=marker, s=100, label=label)
    
    # Ajouter les niveaux des ordres limites et stops
    for order in orders:
        if order['type'] == 'limit':
            plt.axhline(y=order['price'], color='blue', linestyle='--', alpha=0.5)
            plt.text(market_data.index[order['created_at']], order['price'], 
                    f"Limit {order['side']} {order['quantity']}", alpha=0.7)
        elif order['type'] == 'stop' or order['type'] == 'stop_limit':
            plt.axhline(y=order['stop_price'], color='red', linestyle='--', alpha=0.5)
            plt.text(market_data.index[order['created_at']], order['stop_price'], 
                    f"Stop {order['side']} {order['quantity']}", alpha=0.7)
    
    plt.title('Ordres et Exécutions sur le Graphique de Prix')
    plt.ylabel('Prix (USD)')
    plt.grid(True)
    
    # Légende (limiter au nombre raisonnable d'éléments)
    handles, labels = ax1.get_legend_handles_labels()
    # Limiter à max 5 éléments uniques
    by_label = dict(zip(labels[:5], handles[:5]))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    # Graphique des positions
    plt.subplot(3, 1, 2)
    
    # Extraire l'historique des positions
    positions = [p.get('positions', {}).get('BTC/USD', 0) for p in portfolio]
    
    plt.plot(port_timestamps, positions, label='Position BTC', color='orange')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.title('Évolution de la Position BTC')
    plt.ylabel('Quantité (BTC)')
    plt.grid(True)
    
    # Graphique du P&L
    plt.subplot(3, 1, 3)
    
    # Calculer le P&L pour chaque point
    pnl = [(p['total_value'] - 100000) for p in portfolio]
    
    plt.plot(port_timestamps, pnl, label='P&L', color='green')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.title('Évolution du P&L')
    plt.xlabel('Date/Heure')
    plt.ylabel('P&L (USD)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Enregistrer le graphique détaillé
    plt.savefig(f"{output_dir}/pro_order_detailed.png")
    plt.close()
    
    print(f"\nGraphiques enregistrés dans {output_dir}")
    
    return {
        'trades': trades,
        'orders': orders,
        'portfolio': portfolio,
        'performance': performance
    }

if __name__ == "__main__":
    results = pro_order_example() 