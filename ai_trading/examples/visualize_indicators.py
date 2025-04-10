import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.technical_indicators import TechnicalIndicators

def generate_synthetic_data(n_samples=500, trend='bullish', volatility=0.02):
    """
    Génère des données synthétiques pour tester les indicateurs.
    
    Args:
        n_samples (int): Nombre d'échantillons à générer.
        trend (str): Tendance des prix ('bullish', 'bearish', 'sideways', 'volatile').
        volatility (float): Niveau de volatilité des prix.
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données OHLCV.
    """
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
    
    # Générer les prix selon la tendance
    if trend == 'bullish':
        # Tendance haussière
        base_prices = np.linspace(100, 200, n_samples)
    elif trend == 'bearish':
        # Tendance baissière
        base_prices = np.linspace(200, 100, n_samples)
    elif trend == 'sideways':
        # Tendance latérale
        base_prices = np.ones(n_samples) * 150
    elif trend == 'volatile':
        # Tendance volatile
        base_prices = np.ones(n_samples) * 150
        for i in range(1, n_samples):
            if i % 50 < 25:
                base_prices[i] = base_prices[i-1] * (1 + 0.01)
            else:
                base_prices[i] = base_prices[i-1] * (1 - 0.01)
    else:
        raise ValueError(f"Tendance inconnue: {trend}")
    
    # Ajouter de la volatilité
    noise = np.random.normal(0, volatility, n_samples)
    prices = base_prices * (1 + noise)
    
    # Créer les données OHLCV
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = prices * (1 + np.random.normal(0, 0.005, n_samples))
    df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    df['volume'] = np.random.uniform(1000, 5000, n_samples) * (1 + noise)
    
    return df

def plot_indicators(data, indicators):
    """
    Trace les indicateurs techniques sur un graphique.
    
    Args:
        data (pandas.DataFrame): DataFrame contenant les données OHLCV.
        indicators (TechnicalIndicators): Instance de la classe TechnicalIndicators.
    """
    # Ajuster les données pour correspondre à la longueur des indicateurs
    start_index = indicators.window_size  # Commencer après la période de calcul
    adjusted_data = data.iloc[start_index:]
    
    # Créer la figure avec GridSpec
    fig = plt.figure(figsize=(15, 25))
    gs = GridSpec(6, 1, figure=fig)
    
    # Graphique des prix et EMA
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Prix et indicateurs de tendance')
    
    # Utiliser les données ajustées
    ax1.plot(adjusted_data.index, adjusted_data['close'], label='Prix de clôture')
    
    # Récupérer les indicateurs avec les mêmes indices
    ema9 = indicators.calculate_ema(9).iloc[start_index:]
    ema21 = indicators.calculate_ema(21).iloc[start_index:]
    
    ax1.plot(ema9.index, ema9, label='EMA 9', alpha=0.7)
    ax1.plot(ema21.index, ema21, label='EMA 21', alpha=0.7)
    
    # Ajouter les EMAs
    ema50 = indicators.calculate_ema(period=50)
    ema200 = indicators.calculate_ema(period=200)
    
    ax1.plot(adjusted_data.index, ema50, label='EMA 50', alpha=0.7)
    ax1.plot(adjusted_data.index, ema200, label='EMA 200', alpha=0.7)
    
    # Ajouter les Bandes de Bollinger
    upper_bb, middle_bb, lower_bb = indicators.calculate_bollinger_bands()
    ax1.plot(adjusted_data.index, upper_bb, 'r--', label='Bande supérieure', alpha=0.5)
    ax1.plot(adjusted_data.index, middle_bb, 'g--', label='Bande moyenne', alpha=0.5)
    ax1.plot(adjusted_data.index, lower_bb, 'r--', label='Bande inférieure', alpha=0.5)
    
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Graphique du MACD
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('MACD')
    
    macd_line, signal_line, histogram = indicators.calculate_macd()
    
    ax2.plot(adjusted_data.index, macd_line, label='MACD')
    ax2.plot(adjusted_data.index, signal_line, label='Signal')
    ax2.bar(adjusted_data.index, histogram, label='Histogramme', alpha=0.5)
    
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Graphique du RSI
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title('RSI')
    
    rsi = indicators.calculate_rsi()
    
    ax3.plot(adjusted_data.index, rsi, label='RSI')
    ax3.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    ax3.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # Graphique du Stochastique
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_title('Stochastique')
    
    stoch_k, stoch_d = indicators.calculate_stochastic()
    
    ax4.plot(adjusted_data.index, stoch_k, label='%K')
    ax4.plot(adjusted_data.index, stoch_d, label='%D')
    ax4.axhline(y=80, color='r', linestyle='-', alpha=0.3)
    ax4.axhline(y=20, color='g', linestyle='-', alpha=0.3)
    
    ax4.legend(loc='upper left')
    ax4.grid(True)
    
    # Graphique de l'ADX
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.set_title('ADX')
    
    adx, plus_di, minus_di = indicators.calculate_adx()
    
    ax5.plot(adjusted_data.index, adx, label='ADX')
    ax5.plot(adjusted_data.index, plus_di, label='+DI')
    ax5.plot(adjusted_data.index, minus_di, label='-DI')
    ax5.axhline(y=25, color='r', linestyle='-', alpha=0.3)
    
    ax5.legend(loc='upper left')
    ax5.grid(True)
    
    # Graphique du Volume et OBV
    ax6 = fig.add_subplot(gs[5, 0])
    ax6.set_title('Volume et OBV')
    
    obv = indicators.calculate_obv()
    volume_avg = indicators.calculate_volume_average()
    
    ax6.bar(adjusted_data.index, adjusted_data['volume'], label='Volume', alpha=0.3)
    ax6.plot(adjusted_data.index, volume_avg, label='Volume moyen', color='r')
    
    ax6_twin = ax6.twinx()
    ax6_twin.plot(adjusted_data.index, obv, label='OBV', color='g')
    
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('indicators_visualization.png')
    plt.show()

def main():
    """Fonction principale."""
    # Générer des données synthétiques
    data = generate_synthetic_data(n_samples=200, trend='bullish', volatility=0.02)
    
    # Initialiser la classe d'indicateurs
    indicators = TechnicalIndicators(data)
    
    # Tracer les indicateurs
    plot_indicators(data, indicators)
    
    print("Visualisation des indicateurs terminée. Image sauvegardée sous 'indicators_visualization.png'.")

if __name__ == "__main__":
    main() 