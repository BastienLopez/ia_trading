import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.risk_manager import RiskManager
from ai_trading.rl.technical_indicators import TechnicalIndicators

def test_atr_stop_loss(verbose=True, config=None):
    """
    Teste les stop-loss dynamiques basés sur l'ATR sur des données synthétiques.
    """
    # Créer des données synthétiques
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="1h")
    
    # Générer un prix qui suit un mouvement brownien géométrique
    np.random.seed(42)  # Pour la reproductibilité
    price = 20000  # Prix initial
    returns = np.random.normal(0.0001, 0.02, len(dates))  # Volatilité x10
    prices = price * np.cumprod(1 + returns)
    
    # Créer le DataFrame avec OHLCV
    data = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
        "high": prices * (1 + np.random.uniform(0.001, 0.01, len(dates))),
        "low": prices * (1 - np.random.uniform(0.001, 0.01, len(dates))),
        "close": prices,
        "volume": np.random.uniform(10, 100, len(dates)) * prices / 1000
    }, index=dates)
    
    # S'assurer que high est toujours le plus haut et low le plus bas
    data["high"] = np.maximum(np.maximum(data["high"], data["open"]), data["close"])
    data["low"] = np.minimum(np.minimum(data["low"], data["open"]), data["close"])
    
    print(f"Données synthétiques créées: {len(data)} points de données")
    
    # Initialiser le gestionnaire de risques
    default_config = {
        'stop_loss_atr_factor': 2.0,
        'take_profit_atr_factor': 3.0,
        'trailing_stop_atr_factor': 1.5,
        'trailing_stop_activation': 0.01,
        'breakout_threshold': 1.0,
        'breakout_multiplier': 0.6,
        'volatility_lookback': 21,          # Période de calcul de la volatilité
        'volatility_threshold': 0.20,       # 20% de volatilité annuelle
        'position_adjustment_factor': 0.75  # Facteur d'ajustement agressif
    }
    merged_config = {**default_config, **(config or {})}
    risk_manager = RiskManager(config=merged_config)
    
    # Initialiser les indicateurs techniques avec les données
    indicators = TechnicalIndicators(data)
    
    # Calculer l'ATR
    atr = indicators.calculate_atr(period=14)
    data['atr'] = np.nan
    if len(atr) > 0:
        data.loc[data.index[13:], 'atr'] = atr
    
    # Simuler des trades
    trades = []
    positions = []
    current_position = None
    
    for i in range(14, len(data) - 1):
        current_price = data.iloc[i]['close']
        next_price = data.iloc[i + 1]['close']
        
        # Stratégie simple : acheter si le prix augmente de 0.5%, vendre si le prix baisse de 0.5%
        if current_position is None:
            # Pas de position, vérifier si on doit acheter
            if next_price > current_price * 1.005:
                # Ouvrir une position longue
                position_id = f"position_{i}"
                entry_price = next_price
                
                # Calculer le stop-loss basé sur l'ATR
                historical_data = data.iloc[i-14:i+1]
                stop_loss = risk_manager.calculate_atr_stop_loss(
                    historical_data, period=14, direction='long', 
                    current_price=entry_price, position_id=position_id
                )
                
                # Calculer le take-profit basé sur l'ATR
                take_profit = risk_manager.calculate_atr_take_profit(
                    historical_data, period=14, direction='long', 
                    current_price=entry_price, position_id=position_id
                )
                
                current_position = {
                    'id': position_id,
                    'entry_price': entry_price,
                    'entry_index': i + 1,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': None
                }
                positions.append(current_position)
                print(f"Position ouverte à {entry_price:.2f}, stop-loss à {stop_loss:.2f}, take-profit à {take_profit:.2f}")
        else:
            # Position existante, vérifier si on doit la fermer
            position_id = current_position['id']
            entry_price = current_position['entry_price']
            
            # Mettre à jour les stops en fonction de la volatilité
            risk_manager.update_volatility_adjusted_stops(data.iloc[:i+1], position_id)
            
            # Mettre à jour le trailing stop dynamique
            historical_data = data.iloc[i-14:i+1]
            new_trailing_stop = risk_manager.update_atr_trailing_stop(
                data=historical_data,
                period=14,
                position_id=position_id,
                current_price=next_price,
                direction='long'
            )
            
            # Vérifier les conditions de sortie
            stop_result = risk_manager.check_stop_conditions(
                position_id=position_id,
                current_price=next_price,
                direction='long'
            )
            
            if stop_result['stop_triggered']:
                # Fermer la position
                exit_price = stop_result['stop_price']
                profit_pct = (exit_price - entry_price) / entry_price * 100
                
                trade = {
                    'entry_index': current_position['entry_index'],
                    'exit_index': i + 1,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'stop_type': stop_result['stop_type'],
                    'stop_loss': current_position['stop_loss'],
                    'trailing_stop': current_position['trailing_stop']
                }
                trades.append(trade)
                print(f"Position fermée à {exit_price:.2f} ({stop_result['stop_type']}), profit: {profit_pct:.2f}%")
                
                # Supprimer la position
                risk_manager.clear_position(position_id)
                current_position = None
    
    # Fermer la dernière position si elle existe encore
    if current_position is not None:
        position_id = current_position['id']
        entry_price = current_position['entry_price']
        exit_price = data.iloc[-1]['close']
        profit_pct = (exit_price - entry_price) / entry_price * 100
        
        trade = {
            'entry_index': current_position['entry_index'],
            'exit_index': len(data) - 1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_pct': profit_pct,
            'stop_type': 'end_of_data',
            'stop_loss': current_position['stop_loss'],
            'trailing_stop': current_position['trailing_stop']
        }
        trades.append(trade)
        print(f"Position fermée à la fin des données à {exit_price:.2f}, profit: {profit_pct:.2f}%")
        
        # Supprimer la position
        risk_manager.clear_position(position_id)
    
    # Afficher les résultats
    if len(trades) == 0:
        if verbose:
            print("Aucun trade n'a été effectué pendant la période.")
        return
    
    if verbose:
        print(f"\nRésumé des trades:")
        print(f"Nombre de trades: {len(trades)}")
    
    total_profit_pct = sum(trade['profit_pct'] for trade in trades)
    if verbose:
        print(f"Profit total: {total_profit_pct:.2f}%")
    
    win_trades = [trade for trade in trades if trade['profit_pct'] > 0]
    loss_trades = [trade for trade in trades if trade['profit_pct'] <= 0]
    
    if verbose:
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        print(f"Trades gagnants: {len(win_trades)} ({win_rate:.2f}%)")
        print(f"Trades perdants: {len(loss_trades)} ({100 - win_rate:.2f}%)")
    
    avg_win = sum(trade['profit_pct'] for trade in win_trades) / len(win_trades) if win_trades else 0
    avg_loss = sum(trade['profit_pct'] for trade in loss_trades) / len(loss_trades) if loss_trades else 0
    
    if verbose:
        print(f"Gain moyen: {avg_win:.2f}%")
        print(f"Perte moyenne: {avg_loss:.2f}%")
    
    # Compter les types de stop
    stop_types = {}
    for trade in trades:
        stop_type = trade['stop_type']
        if stop_type in stop_types:
            stop_types[stop_type] += 1
        else:
            stop_types[stop_type] = 1
    
    if verbose:
        print("\nTypes de stop:")
        for stop_type, count in stop_types.items():
            print(f"  {stop_type}: {count} ({count / len(trades) * 100:.2f}%)")
    
    # Visualiser les trades
    if verbose:
        visualize_trades(data, trades)
    
    return total_profit_pct

def visualize_trades(data, trades):
    """
    Visualise les trades avec les stop-loss et take-profit.
    
    Args:
        data (pd.DataFrame): Données de marché
        trades (list): Liste des trades
    """
    plt.figure(figsize=(15, 10))
    
    # Tracer le prix
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(data.index, data['close'], label='Prix', color='blue')
    
    # Tracer les trades
    for trade in trades:
        entry_index = trade['entry_index']
        exit_index = trade['exit_index']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        profit_pct = trade['profit_pct']
        stop_type = trade['stop_type']
        
        # Couleur en fonction du résultat
        color = 'green' if profit_pct > 0 else 'red'
        
        # Tracer l'entrée
        ax1.scatter(data.index[entry_index], entry_price, color=color, marker='^', s=100)
        
        # Tracer la sortie
        marker = 'v'  # Par défaut
        if stop_type == 'stop_loss':
            marker = 'X'
        elif stop_type == 'take_profit':
            marker = 'D'
        elif stop_type == 'trailing_stop':
            marker = 's'
        
        ax1.scatter(data.index[exit_index], exit_price, color=color, marker=marker, s=100)
        
        # Tracer la ligne entre l'entrée et la sortie
        ax1.plot([data.index[entry_index], data.index[exit_index]], 
                 [entry_price, exit_price], color=color, linestyle='--')
        
        # Ajouter une annotation
        ax1.annotate(f"{profit_pct:.2f}%", 
                     xy=(data.index[exit_index], exit_price),
                     xytext=(10, 0), textcoords='offset points',
                     color=color, fontweight='bold')
    
    # Ajouter les stops sur le graphique
    for trade in trades:
        entry_index = trade['entry_index']
        exit_index = trade['exit_index']
        
        # Tracer le stop-loss initial
        ax1.plot([data.index[entry_index], data.index[exit_index]],
                 [trade['stop_loss'], trade['stop_loss']],
                 color='orange', linestyle=':', alpha=0.5)
        
        # Tracer le trailing stop
        if trade['stop_type'] == 'trailing_stop':
            ax1.plot([data.index[entry_index], data.index[exit_index]],
                     [trade['trailing_stop'], trade['trailing_stop']],
                     color='purple', linestyle='--', alpha=0.3)
    
    # Ajouter un titre et des étiquettes
    ax1.set_title('Trades avec stop-loss dynamiques basés sur l\'ATR')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Prix')
    
    # Ajouter une légende pour les marqueurs
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Entrée (gain)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Entrée (perte)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='green', markersize=10, label='Sortie manuelle (gain)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Sortie manuelle (perte)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Stop-loss'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=10, label='Take-profit'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Trailing stop')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Graphique de la volatilité
    ax2 = plt.subplot(2, 1, 2)
    returns = np.log(data['close'] / data['close'].shift(1))
    volatility = returns.rolling(14).std() * np.sqrt(252)
    ax2.plot(data.index, volatility, label='Volatilité Annualisée', color='red')
    ax2.axhline(y=0.20, color='orange', linestyle='--', label='Seuil de Volatilité (20%)')
    ax2.set_ylabel('Volatilité')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_atr_stop_loss() 