import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importer le gestionnaire de risques
from .risk_manager import RiskManager
from .technical_indicators import TechnicalIndicators

# Définir le répertoire pour les visualisations
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations', 'misc')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def generate_test_data(periods=100, trend='bullish', volatility=0.02, seed=42):
    """
    Génère des données synthétiques pour les tests.
    
    Args:
        periods (int): Nombre de périodes
        trend (str): 'bullish', 'bearish', ou 'sideways'
        volatility (float): Volatilité des prix
        seed (int): Graine aléatoire pour la reproductibilité
        
    Returns:
        pd.DataFrame: DataFrame avec OHLCV
    """
    np.random.seed(seed)
    
    # Créer des dates
    dates = [datetime.now() - timedelta(days=periods-i) for i in range(periods)]
    
    # Générer un prix de base selon la tendance
    if trend == 'bullish':
        base_prices = np.linspace(100, 150, periods)  # Tendance haussière
    elif trend == 'bearish':
        base_prices = np.linspace(150, 100, periods)  # Tendance baissière
    else:  # sideways
        base_prices = np.ones(periods) * 125  # Marché stable
    
    # Ajouter du bruit
    noise = np.random.normal(0, volatility * 100, periods)
    close_prices = base_prices + noise
    
    # Générer OHLC
    daily_volatility = volatility * 0.5
    high_prices = close_prices + np.abs(np.random.normal(0, daily_volatility * 100, periods))
    low_prices = close_prices - np.abs(np.random.normal(0, daily_volatility * 100, periods))
    open_prices = (high_prices + low_prices) / 2 + np.random.normal(0, daily_volatility * 50, periods)
    
    # Volume
    volume = np.random.randint(1000, 10000, periods)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return df

def test_stop_loss_take_profit():
    """
    Teste les fonctionnalités de stop-loss et take-profit.
    """
    print("\n=== Test des stop-loss et take-profit ===")
    
    # Générer des données de test
    df = generate_test_data(periods=100, trend='bullish', volatility=0.02)
    
    # Initialiser le gestionnaire de risques
    risk_config = {
        'stop_loss_atr_factor': 2.0,
        'take_profit_atr_factor': 4.0
    }
    risk_manager = RiskManager(config=risk_config)
    
    # Calculer les stop-loss et take-profit pour une position longue
    entry_price = df['close'].iloc[-1]
    position_id = 'test_position'
    
    stop_loss = risk_manager.calculate_atr_stop_loss(
        data=df,
        period=14,
        direction='long',
        current_price=entry_price,
        position_id=position_id
    )
    
    take_profit = risk_manager.calculate_atr_take_profit(
        data=df,
        period=14,
        direction='long',
        current_price=entry_price,
        position_id=position_id
    )
    
    print(f"Prix d'entrée: {entry_price:.2f}")
    print(f"Stop-loss: {stop_loss:.2f} (distance: {(entry_price - stop_loss) / entry_price * 100:.2f}%)")
    print(f"Take-profit: {take_profit:.2f} (distance: {(take_profit - entry_price) / entry_price * 100:.2f}%)")
    
    # Simuler des mouvements de prix et vérifier les conditions de stop
    print("\nSimulation de mouvements de prix...")
    
    test_prices = [
        entry_price * 1.01,  # +1%
        entry_price * 1.02,  # +2%
        entry_price * 0.99,  # -1%
        entry_price * 0.97,  # -3%
        entry_price * 0.95,  # -5%
        entry_price * 1.03,  # +3%
        entry_price * 1.06   # +6%
    ]
    
    for i, price in enumerate(test_prices):
        result = risk_manager.check_stop_conditions(position_id, price, direction='long')
        if result['stop_triggered']:
            print(f"Étape {i+1}: Prix = {price:.2f} => {result['stop_type']} déclenché à {result['stop_price']:.2f}")
        else:
            print(f"Étape {i+1}: Prix = {price:.2f} => Aucun stop déclenché")
    
    return risk_manager, df

def test_trailing_stop():
    """
    Teste les fonctionnalités de trailing stop.
    """
    print("\n=== Test du trailing stop ===")
    
    # Générer des données de test avec une forte tendance haussière
    df = generate_test_data(periods=100, trend='bullish', volatility=0.015)
    
    # Initialiser le gestionnaire de risques
    risk_config = {
        'trailing_stop_activation': 0.02,  # 2% de profit
        'trailing_stop_distance': 0.01,    # 1% du prix
        'trailing_stop_atr_factor': 1.5    # Facteur ATR pour trailing stop
    }
    risk_manager = RiskManager(config=risk_config)
    
    # Initialiser une position
    entry_price = df['close'].iloc[-20]  # Prendre un prix d'il y a 20 jours
    position_id = 'trailing_test'
    current_price = entry_price
    
    # Calculer le stop-loss initial
    stop_loss = risk_manager.calculate_atr_stop_loss(
        data=df.iloc[:-20],  # Utiliser les données disponibles à ce moment
        period=14,
        direction='long',
        current_price=entry_price,
        position_id=position_id
    )
    
    # Simuler une évolution de prix
    prices = df['close'].iloc[-20:].values
    trailing_stops = []
    initial_stop = stop_loss
    
    print(f"Prix d'entrée: {entry_price:.2f}")
    print(f"Stop-loss initial: {initial_stop:.2f}")
    print("\nÉvolution du trailing stop:")
    
    for i, price in enumerate(prices):
        # Mise à jour du trailing stop
        trailing_stop = risk_manager.update_trailing_stop(
            position_id=position_id,
            current_price=price,
            entry_price=entry_price,
            direction='long'
        )
        
        trailing_stops.append(trailing_stop if trailing_stop is not None else initial_stop)
        
        # Vérifier si le stop est déclenché
        result = risk_manager.check_stop_conditions(position_id, price, direction='long')
        
        status = "Stop déclenché" if result['stop_triggered'] else "Position active"
        print(f"Jour {i+1}: Prix = {price:.2f}, Trailing Stop = {trailing_stops[-1]:.2f} => {status}")
        
        if result['stop_triggered']:
            break
    
    # Visualiser l'évolution du prix et du trailing stop
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(prices[:len(trailing_stops)])), prices[:len(trailing_stops)], label='Prix', color='blue')
    plt.plot(range(len(trailing_stops)), trailing_stops, label='Trailing Stop', color='red', linestyle='--')
    plt.axhline(y=entry_price, color='green', linestyle='-', label='Prix d\'entrée')
    plt.axhline(y=initial_stop, color='orange', linestyle=':', label='Stop-loss initial')
    
    plt.title('Évolution du prix et du trailing stop')
    plt.xlabel('Jours')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'trailing_stop_evolution.png'))
    plt.close()
    
    print(f"Graphique sauvegardé dans '{os.path.join(VISUALIZATION_DIR, 'trailing_stop_evolution.png')}'")
    
    return risk_manager, df

def test_position_size_calculation():
    """
    Teste le calcul de la taille de position optimale.
    """
    print("\n=== Test du calcul de taille de position ===")
    
    # Générer des données avec différentes volatilités
    low_vol_df = generate_test_data(periods=50, volatility=0.01, seed=123)
    med_vol_df = generate_test_data(periods=50, volatility=0.02, seed=123)
    high_vol_df = generate_test_data(periods=50, volatility=0.04, seed=123)
    
    # Initialiser le gestionnaire de risques
    risk_config = {
        'max_position_size': 0.2,       # 20% max du capital
        'max_risk_per_trade': 0.02,     # 2% du capital par trade
        'volatility_threshold': 0.15    # Seuil de volatilité
    }
    risk_manager = RiskManager(config=risk_config)
    
    # Paramètres communs
    capital = 10000
    entry_price = 100
    
    # Tester avec différentes distances de stop-loss et volatilités
    scenarios = [
        {"name": "Risque faible", "stop_loss_price": 98, "data": low_vol_df},
        {"name": "Risque moyen", "stop_loss_price": 95, "data": med_vol_df},
        {"name": "Risque élevé", "stop_loss_price": 90, "data": high_vol_df}
    ]
    
    for scenario in scenarios:
        # Mettre à jour les indicateurs techniques avec les données
        risk_manager.indicators = TechnicalIndicators(scenario["data"])
        
        # Calculer la taille de position
        position_size = risk_manager.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=scenario["stop_loss_price"]
        )
        
        # Calculer le risque réel
        risk_amount = (entry_price - scenario["stop_loss_price"]) * position_size
        risk_pct = risk_amount / capital * 100
        
        print(f"\n{scenario['name']}:")
        print(f"  Stop-loss à {scenario['stop_loss_price']} ({(entry_price - scenario['stop_loss_price']) / entry_price * 100:.1f}% de distance)")
        print(f"  Taille de position: {position_size:.2f} unités (valeur: ${position_size * entry_price:.2f})")
        print(f"  Risque: ${risk_amount:.2f} ({risk_pct:.2f}% du capital)")
    
    return risk_manager

def test_risk_manager_integration():
    """
    Teste l'intégration complète du gestionnaire de risques avec une simulation.
    """
    print("\n=== Test d'intégration du gestionnaire de risques ===")
    
    # Générer des données de marché
    df = generate_test_data(periods=120, trend='bullish', volatility=0.02)
    
    # Configuration complète
    risk_config = {
        'max_position_size': 0.2,
        'max_risk_per_trade': 0.02,
        'stop_loss_atr_factor': 2.5,
        'take_profit_atr_factor': 5.0,
        'trailing_stop_activation': 0.03,
        'trailing_stop_distance': 0.015,
        'volatility_lookback': 14,
        'volatility_threshold': 0.15,
        'position_adjustment_factor': 0.5
    }
    
    risk_manager = RiskManager(config=risk_config)
    
    # Simulation de trading
    capital = 10000
    trade_results = []
    
    # Initialisation des variables
    position = None
    entry_price = None
    position_id = None
    position_size = 0
    
    print("Simulation de trading avec gestion des risques...")
    
    # Diviser les données en tranches pour simuler l'écoulement du temps
    start_idx = 30  # Utiliser les 30 premiers jours comme historique initial
    
    for i in range(start_idx, len(df) - 1):
        current_data = df.iloc[:i+1]
        current_price = current_data['close'].iloc[-1]
        next_price = df['close'].iloc[i+1]  # Pour calculer le résultat
        
        # Mise à jour des indicateurs
        risk_manager.indicators = TechnicalIndicators(current_data)
        
        # Si pas de position, décider d'entrer
        if position is None:
            # Stratégie simple: entrer si la tendance est haussière
            sma_20 = risk_manager.indicators.calculate_sma(period=20)
            sma_50 = risk_manager.indicators.calculate_sma(period=50)
            
            if sma_20 is not None and sma_50 is not None and len(sma_20) > 0 and len(sma_50) > 0:
                if sma_20.iloc[-1] > sma_50.iloc[-1]:
                    # Calculer le stop-loss pour déterminer le risque
                    stop_loss = risk_manager.calculate_atr_stop_loss(
                        data=current_data,
                        period=14,
                        direction='long',
                        current_price=current_price
                    )
                    
                    if stop_loss is not None and stop_loss < current_price:
                        # Calculer la taille de position
                        position_size = risk_manager.calculate_position_size(
                            capital=capital,
                            entry_price=current_price,
                            stop_loss_price=stop_loss
                        )
                        
                        # Entrer en position
                        position = 'long'
                        entry_price = current_price
                        position_id = f'trade_{i}'
                        
                        # Calculer stops
                        stop_loss = risk_manager.calculate_atr_stop_loss(
                            data=current_data,
                            period=14,
                            direction=position,
                            current_price=entry_price,
                            position_id=position_id
                        )
                        
                        take_profit = risk_manager.calculate_atr_take_profit(
                            data=current_data,
                            period=14,
                            direction=position,
                            current_price=entry_price,
                            position_id=position_id
                        )
                        
                        print(f"Jour {i}: Ouverture position LONG à {entry_price:.2f}, "
                              f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
                              f"Taille: {position_size:.2f} unités")
        
        # Si en position, vérifier les stops et mettre à jour le trailing stop
        elif position == 'long':
            # Mettre à jour le trailing stop si nécessaire
            if (current_price - entry_price) / entry_price > risk_config['trailing_stop_activation']:
                risk_manager.update_trailing_stop(
                    position_id=position_id,
                    current_price=current_price,
                    entry_price=entry_price,
                    direction=position
                )
            
            # Vérifier les conditions de stop
            result = risk_manager.check_stop_conditions(position_id, current_price, direction=position)
            
            if result['stop_triggered']:
                # Calculer le P&L
                exit_price = result['stop_price']
                profit_loss = (exit_price - entry_price) * position_size
                profit_pct = (exit_price - entry_price) / entry_price * 100
                
                trade_results.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'profit_loss': profit_loss,
                    'profit_pct': profit_pct,
                    'stop_type': result['stop_type'],
                    'trade_duration': i - int(position_id.split('_')[1])
                })
                
                print(f"Jour {i}: Fermeture position LONG à {exit_price:.2f}, "
                      f"P&L: ${profit_loss:.2f} ({profit_pct:.2f}%), "
                      f"Raison: {result['stop_type']}")
                
                # Mettre à jour le capital
                capital += profit_loss
                
                # Réinitialiser la position
                position = None
                position_size = 0
                risk_manager.clear_position(position_id)
                
    # Afficher les résultats
    print("\nRésultats de la simulation:")
    print(f"Capital final: ${capital:.2f}")
    
    if trade_results:
        total_trades = len(trade_results)
        winning_trades = sum(1 for t in trade_results if t['profit_loss'] > 0)
        win_rate = winning_trades / total_trades * 100
        
        total_profit = sum(t['profit_loss'] for t in trade_results)
        avg_profit = total_profit / total_trades
        
        avg_win = sum(t['profit_loss'] for t in trade_results if t['profit_loss'] > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t['profit_loss'] for t in trade_results if t['profit_loss'] <= 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
        
        print(f"Nombre de trades: {total_trades}")
        print(f"Trades gagnants: {winning_trades} ({win_rate:.2f}%)")
        print(f"Profit total: ${total_profit:.2f}")
        print(f"Profit moyen par trade: ${avg_profit:.2f}")
        print(f"Gain moyen (trades gagnants): ${avg_win:.2f}")
        print(f"Perte moyenne (trades perdants): ${avg_loss:.2f}")
        
        # Visualiser l'évolution du capital
        cumulative_pnl = [0]
        for trade in trade_results:
            cumulative_pnl.append(cumulative_pnl[-1] + trade['profit_loss'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(cumulative_pnl)), [10000 + p for p in cumulative_pnl], label='Capital')
        plt.title('Évolution du capital avec gestion des risques')
        plt.xlabel('Nombre de trades')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'risk_management_capital.png'))
        plt.close()
        
        print(f"Graphique de l'évolution du capital sauvegardé dans '{os.path.join(VISUALIZATION_DIR, 'risk_management_capital.png')}'")
    
    return risk_manager, df, trade_results

if __name__ == "__main__":
    # Exécuter les tests
    test_stop_loss_take_profit()
    test_trailing_stop()
    test_position_size_calculation()
    test_risk_manager_integration() 