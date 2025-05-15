"""
Exemple d'utilisation du générateur de signaux multi-timeframes.
Ce module démontre comment combiner des signaux de trading provenant
de différentes échelles temporelles pour obtenir des signaux plus fiables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

from ai_trading.ml.trading_signals.multi_timeframe_signals import MultiTimeframeSignalGenerator
from ai_trading.ml.trading_signals.signal_generator import SignalType

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples=1000, trend="bullish", volatility=0.02, seed=42):
    """Génère des données OHLCV pour tester le générateur de signaux."""
    np.random.seed(seed)
    
    # Date de début
    start_date = datetime(2023, 1, 1)
    
    # Créer des dates à intervalles d'une minute
    dates = [start_date + timedelta(minutes=i) for i in range(n_samples)]
    
    # Créer une tendance de base en fonction du paramètre
    if trend == "bullish":
        base_trend = np.linspace(100, 150, n_samples)
    elif trend == "bearish":
        base_trend = np.linspace(150, 100, n_samples)
    elif trend == "volatile":
        base_trend = 125 + 25 * np.sin(np.linspace(0, 3*np.pi, n_samples))
    else:  # sideways
        base_trend = np.full(n_samples, 125)
    
    # Ajouter du bruit
    noise_level = volatility * 100  # Approximation pour le niveau de bruit
    noise = np.random.normal(0, noise_level, n_samples)
    close_prices = base_trend + noise
    
    # Créer des variations pour high, low, open basées sur close
    high_prices = close_prices + np.random.uniform(0, 5, n_samples)
    low_prices = close_prices - np.random.uniform(0, 5, n_samples)
    open_prices = close_prices - np.random.uniform(-3, 3, n_samples)
    
    # Volume avec quelques pics
    volume = np.random.uniform(1000, 5000, n_samples)
    volume[np.random.choice(range(n_samples), 10)] *= 3  # Quelques pics de volume
    
    # Assembler en DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return df


def create_multi_timeframe_data(base_data):
    """Crée des données pour différents timeframes à partir des données de base."""
    data_dict = {
        "1m": base_data,
        "5m": base_data.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        "15m": base_data.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        "1h": base_data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        "4h": base_data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        "1d": base_data.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    }
    
    return data_dict


def create_divergent_scenario(base_data):
    """Crée un scénario avec divergence entre timeframes courts et longs."""
    # Copier les données de base
    modified_data = base_data.copy()
    
    # Définir des périodes pour créer une divergence
    last_segment = modified_data.index[-200:]
    
    # Créer un mouvement baissier dans les dernières périodes
    modified_data.loc[last_segment, 'close'] = modified_data.loc[last_segment, 'close'] * np.linspace(1, 0.85, len(last_segment))
    modified_data.loc[last_segment, 'low'] = modified_data.loc[last_segment, 'low'] * np.linspace(1, 0.85, len(last_segment))
    modified_data.loc[last_segment, 'high'] = modified_data.loc[last_segment, 'high'] * np.linspace(1, 0.85, len(last_segment))
    
    # Créer les données multi-timeframes
    data_dict = create_multi_timeframe_data(modified_data)
    
    # Modifier les timeframes longs (1h, 4h, 1d) pour qu'ils restent haussiers
    for tf in ["1h", "4h", "1d"]:
        # Récupérer les dernières périodes du timeframe long
        last_tf_rows = data_dict[tf].index[-5:]
        
        # Corriger la tendance baissière pour la rendre légèrement haussière
        data_dict[tf].loc[last_tf_rows, 'close'] *= np.linspace(1, 1.08, len(last_tf_rows))
        data_dict[tf].loc[last_tf_rows, 'high'] *= np.linspace(1, 1.1, len(last_tf_rows))
        data_dict[tf].loc[last_tf_rows, 'low'] *= np.linspace(1, 1.05, len(last_tf_rows))
    
    return data_dict


def plot_signals(data_dict, signals, title="Signaux Multi-Timeframes"):
    """Visualiser les prix et les signaux générés."""
    plt.figure(figsize=(14, 8))
    
    # Tracer les prix de clôture pour différents timeframes
    timeframes = ["1m", "15m", "1h", "4h"]
    for i, tf in enumerate(timeframes):
        if tf in data_dict:
            plt.plot(data_dict[tf].index, data_dict[tf]['close'], 
                    label=f'Prix ({tf})', alpha=0.6, linewidth=1+i*0.5)
    
    # Ajouter les signaux d'achat et de vente
    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    sell_signals = [s for s in signals if s.type == SignalType.SELL]
    divergence_signals = [s for s in signals if s.source == "Divergence-TF"]
    
    for signal in buy_signals:
        plt.scatter(signal.timestamp, signal.price, color='green', s=100, marker='^', 
                   label=f'Achat ({signal.source})' if 'Achat' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    for signal in sell_signals:
        plt.scatter(signal.timestamp, signal.price, color='red', s=100, marker='v', 
                   label=f'Vente ({signal.source})' if 'Vente' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    for signal in divergence_signals:
        plt.scatter(signal.timestamp, signal.price, color='orange', s=120, marker='o', 
                   label=f'Divergence' if 'Divergence' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Ajouter des annotations pour les scores de confiance
    for signal in signals:
        plt.annotate(f'{signal.confidence:.2f}', 
                    (signal.timestamp, signal.price), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Zoomer sur la dernière partie pour mieux voir les signaux
    if signals:
        oldest_signal_time = min(signal.timestamp for signal in signals)
        plt.xlim(oldest_signal_time - pd.Timedelta(days=1), data_dict["1m"].index[-1])
    
    plt.show()


def main():
    """Fonction principale de démonstration."""
    print("=== Démonstration des Signaux Multi-Timeframes ===")
    
    # 1. Générer des données de marché
    print("\n1. Génération des données de marché...")
    base_data = generate_sample_data(n_samples=5000, trend="bullish")
    
    # 2. Créer deux scénarios: normal et divergence
    print("\n2. Création de deux scénarios: normal et divergence...")
    normal_data = create_multi_timeframe_data(base_data)
    divergent_data = create_divergent_scenario(base_data)
    
    # 3. Initialiser le générateur de signaux multi-timeframes
    print("\n3. Initialisation du générateur de signaux multi-timeframes...")
    generator = MultiTimeframeSignalGenerator()
    
    # 4. Générer des signaux pour le scénario normal
    print("\n4. Génération des signaux pour le scénario normal...")
    normal_signals = generator.generate_signals(normal_data, "BTC/USD")
    
    # Afficher les signaux générés
    print(f"\n{len(normal_signals)} signaux générés pour le scénario normal:")
    for i, signal in enumerate(normal_signals[:3]):  # Afficher les 3 premiers signaux
        print(f"Signal {i+1}: {signal}")
        if signal.metadata:
            for key, value in signal.metadata.items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
    
    # 5. Générer des signaux pour le scénario avec divergence
    print("\n5. Génération des signaux pour le scénario avec divergence...")
    divergent_signals = generator.generate_signals(divergent_data, "BTC/USD")
    
    # Afficher les signaux générés
    print(f"\n{len(divergent_signals)} signaux générés pour le scénario avec divergence:")
    for i, signal in enumerate(divergent_signals[:3]):  # Afficher les 3 premiers signaux
        print(f"Signal {i+1}: {signal}")
        if signal.metadata:
            for key, value in signal.metadata.items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
    
    # 6. Visualiser les signaux
    print("\n6. Visualisation des signaux...")
    plot_signals(normal_data, normal_signals, "Signaux Multi-Timeframes - Scénario Normal")
    plot_signals(divergent_data, divergent_signals, "Signaux Multi-Timeframes - Scénario avec Divergence")
    
    # 7. Démontrer l'impact du filtre de volatilité
    print("\n7. Démonstration de l'impact du filtre de volatilité...")
    
    # Créer des données avec haute volatilité
    volatile_data = generate_sample_data(n_samples=5000, trend="volatile", volatility=0.05)
    volatile_tf_data = create_multi_timeframe_data(volatile_data)
    
    # Générer des signaux avec filtre de volatilité activé
    generator.config["volatility_filter"]["enabled"] = True
    signals_with_filter = generator.generate_signals(volatile_tf_data, "BTC/USD")
    
    # Générer des signaux avec filtre de volatilité désactivé
    generator.config["volatility_filter"]["enabled"] = False
    signals_without_filter = generator.generate_signals(volatile_tf_data, "BTC/USD")
    
    print(f"\nComparaison des signaux avec et sans filtre de volatilité:")
    print(f"- Avec filtre: {len(signals_with_filter)} signaux générés")
    print(f"- Sans filtre: {len(signals_without_filter)} signaux générés")
    
    # Afficher les ajustements de confiance dus à la volatilité
    if signals_with_filter:
        print("\nExemple d'ajustement de confiance dû à la volatilité:")
        for signal in signals_with_filter[:2]:
            if "volatility_adjustment" in signal.metadata:
                print(f"Signal: {signal}")
                print(f"  Volatilité: {signal.metadata['volatility']:.4f}")
                print(f"  Ajustement: {signal.metadata['volatility_adjustment']}")
    
    # 8. Tracer les signaux avec et sans filtre de volatilité
    plot_signals(volatile_tf_data, signals_with_filter, "Signaux avec Filtre de Volatilité")
    plot_signals(volatile_tf_data, signals_without_filter, "Signaux sans Filtre de Volatilité")
    
    print("\nDémonstration terminée.")


if __name__ == "__main__":
    main() 