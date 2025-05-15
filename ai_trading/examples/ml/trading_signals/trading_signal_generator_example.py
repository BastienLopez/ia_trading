"""
Exemple de génération de signaux de trading.

Ce script montre comment utiliser le générateur de signaux de trading
pour créer, filtrer et évaluer des signaux d'achat/vente.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import des modules de signaux de trading
from ai_trading.ml.trading_signals.signal_generator import SignalGenerator, Signal, SignalType
from ai_trading.ml.trading_signals.ml_model import TradingSignalModel

def load_market_data(symbol, start_date, end_date):
    """Charge des données de marché pour l'exemple."""
    # Simuler des données OHLCV pour l'exemple
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # Créer une tendance artificielle
    trend = np.concatenate([
        np.linspace(100, 120, n//3),
        np.linspace(120, 90, n//3),
        np.linspace(90, 110, n - 2*(n//3))
    ])
    
    # Ajouter du bruit
    noise = np.random.normal(0, 3, n)
    close_prices = trend + noise
    
    data = pd.DataFrame({
        'open': close_prices - np.random.uniform(0, 2, n),
        'high': close_prices + np.random.uniform(1, 3, n),
        'low': close_prices - np.random.uniform(1, 3, n),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, n)
    }, index=dates)
    
    return data

def signal_generation_example():
    """Exemple de génération et filtrage de signaux de trading."""
    # Définir la période d'analyse
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Charger les données de marché
    market_data = load_market_data('BTC/USD', start_date, end_date)
    
    # Initialiser le générateur de signaux
    signal_generator = SignalGenerator()
    
    # Générer des signaux basés sur différents indicateurs techniques
    signals = signal_generator.generate_signals(
        market_data,
        indicators=['sma', 'ema', 'macd', 'rsi', 'bollinger'],
        params={
            'sma_period': 20,
            'ema_period': 10,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'bollinger_period': 20,
            'bollinger_std': 2.0
        }
    )
    
    print(f"Nombre total de signaux générés: {len(signals)}")
    
    # Filtrer les signaux selon leur confiance
    filtered_signals = signal_generator.filter_signals(signals, min_confidence=0.7)
    print(f"Nombre de signaux après filtrage (confiance >= 0.7): {len(filtered_signals)}")
    
    # Évaluer et scorer les signaux
    scored_signals = signal_generator.score_signals(filtered_signals, market_data)
    
    # Trier les signaux par score
    top_signals = sorted(scored_signals, key=lambda s: s.metadata.get('score', 0), reverse=True)[:5]
    
    # Afficher les meilleurs signaux
    print("\n== Top 5 des signaux ==")
    for i, signal in enumerate(top_signals, 1):
        print(f"{i}. {signal.timestamp.date()} - Type: {signal.signal_type.name}, "
              f"Source: {signal.source}, Confiance: {signal.confidence:.2f}, "
              f"Score: {signal.metadata.get('score', 0):.2f}")
    
    # Utiliser un modèle ML pour générer des signaux additionnels
    model = TradingSignalModel()
    
    # Simuler un entraînement préalable du modèle
    print("\nEntraînement du modèle ML de signaux (simulation)...")
    
    # Générer des signaux avec le modèle ML
    ml_signals = model.predict(market_data)
    print(f"Nombre de signaux générés par le modèle ML: {len(ml_signals)}")
    
    # Visualiser les signaux sur le graphique des prix
    visualize_signals(market_data, top_signals)
    
    return top_signals

def visualize_signals(market_data, signals):
    """Visualise les signaux sur le graphique des prix."""
    plt.figure(figsize=(12, 6))
    
    # Tracer les prix de clôture
    plt.plot(market_data.index, market_data['close'], label='Prix', color='blue')
    
    # Ajouter les signaux
    for signal in signals:
        if signal.signal_type == SignalType.BUY:
            plt.scatter(signal.timestamp, market_data.loc[signal.timestamp, 'close'],
                       marker='^', color='green', s=100, label='Achat')
        elif signal.signal_type == SignalType.SELL:
            plt.scatter(signal.timestamp, market_data.loc[signal.timestamp, 'close'],
                       marker='v', color='red', s=100, label='Vente')
    
    plt.title('Prix et Signaux de Trading')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.grid(True)
    
    # Gérer les doublons dans la légende
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Enregistrer le graphique
    plt.savefig("examples/ml/trading_signals/trading_signals_chart.png")
    
    # Fermer la figure pour libérer la mémoire
    plt.close()
    
    print("Graphique des signaux enregistré sous 'examples/ml/trading_signals/trading_signals_chart.png'")

if __name__ == "__main__":
    signals = signal_generation_example() 