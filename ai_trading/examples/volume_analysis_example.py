"""
Exemple d'utilisation du module d'analyse volumétrique.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ai_trading.indicators.volume_analysis import (
    VolumeAnalyzer,
    VolumeProfileType,
    volume_delta,
    on_balance_volume,
    accelerating_volume
)


def create_example_data(n_samples=200):
    """Crée des données OHLCV simulées pour la démonstration."""
    np.random.seed(42)
    
    # Créer des prix avec tendance, cycles et bruit
    t = np.linspace(0, 4 * np.pi, n_samples)
    trend = np.linspace(0, 15, n_samples)
    cycle1 = 5 * np.sin(t)
    cycle2 = 2 * np.sin(2.5 * t)
    noise = np.random.normal(0, 1, n_samples)
    
    # Créer la série de prix de clôture
    close = 100 + trend + cycle1 + cycle2 + noise
    
    # Faire varier le volume en fonction des mouvements de prix
    base_volume = 1000 + 200 * np.random.random(n_samples)
    price_changes = np.abs(np.diff(close, prepend=close[0]))
    
    # Plus de volume sur les mouvements importants
    volume = base_volume + 500 * price_changes
    
    # Ajouter quelques pics de volume spécifiques
    volume[40] = volume[40] * 3  # Grand pic de volume
    volume[80] = volume[80] * 4  # Grand pic de volume
    volume[120] = volume[120] * 3.5  # Grand pic de volume
    volume[160] = volume[160] * 2.5  # Grand pic de volume
    
    # Générer les prix d'ouverture, haut et bas
    price_volatility = np.random.uniform(0.2, 0.8, n_samples)
    open_price = close - 0.5 * price_volatility * (close - np.roll(close, 1))
    
    candle_size = 1.5 * price_volatility * np.sqrt(volume) / 30
    high = np.maximum(close, open_price) + candle_size
    low = np.minimum(close, open_price) - candle_size
    
    # Créer un DataFrame avec les données
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Ajouter un index de date
    data.index = pd.date_range(start='2023-01-01', periods=n_samples, freq='1D')
    
    return data


def main():
    """Fonction principale de démonstration."""
    # Créer des données simulées
    data = create_example_data(200)
    
    # Initialiser l'analyseur de volume
    analyzer = VolumeAnalyzer(data)
    
    # 1. Analyse des profils de volume et points de contrôle
    plt.figure(figsize=(15, 12))
    
    # 1.1 Profil de volume par prix
    ax1 = plt.subplot(3, 2, 1)
    analyzer.plot_volume_profile(
        start_idx=50,
        end_idx=150,
        profile_type=VolumeProfileType.PRICE,
        ax=ax1,
        horizontal=True
    )
    ax1.set_title("Profil de Volume par Prix")
    
    # 1.2 VWAP avec bandes
    ax2 = plt.subplot(3, 2, 2)
    analyzer.plot_volume_profile(
        start_idx=50,
        end_idx=150,
        profile_type=VolumeProfileType.VWAP,
        ax=ax2
    )
    ax2.set_title("VWAP avec Bandes")
    
    # 2. Détection des points de contrôle (niveaux de prix importants)
    control_points = analyzer.find_control_points(
        lookback_periods=30,
        min_volume_percentile=0.75
    )
    
    # 2.1 Tracer les prix avec les points de contrôle
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(data.index, data['close'], label='Prix de clôture')
    
    # Tracer des lignes horizontales pour les points de contrôle
    for cp in control_points[:5]:  # 5 points les plus importants
        ax3.axhline(cp['price'], color='r', linestyle='--', alpha=0.3 + 0.5 * cp['importance'])
        ax3.text(data.index[0], cp['price'], f"POC: {cp['price']:.2f}", verticalalignment='center')
    
    ax3.set_title("Points de Contrôle (Niveaux de Prix Importants)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Prix")
    
    # 3. Détection des anomalies de volume
    volume_anomalies = analyzer.detect_volume_anomalies(
        window_size=20,
        threshold_sigma=2.0
    )
    
    # 3.1 Tracer les volumes avec anomalies mises en évidence
    ax4 = plt.subplot(3, 2, 4)
    ax4.bar(data.index, data['volume'], alpha=0.5, label='Volume normal')
    
    # Mettre en évidence les anomalies
    if len(volume_anomalies) > 0:
        high_vol_anomalies = volume_anomalies[volume_anomalies['anomaly_type'] == 'high_volume']
        low_vol_anomalies = volume_anomalies[volume_anomalies['anomaly_type'] == 'low_volume']
        
        if len(high_vol_anomalies) > 0:
            ax4.bar(high_vol_anomalies.index, high_vol_anomalies['volume'], 
                   color='red', label='Volume anormalement élevé')
        
        if len(low_vol_anomalies) > 0:
            ax4.bar(low_vol_anomalies.index, low_vol_anomalies['volume'], 
                   color='blue', label='Volume anormalement bas')
    
    ax4.set_title("Anomalies de Volume")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Volume")
    ax4.legend()
    
    # 4. Corrélation Volume-Prix
    correlation = analyzer.calculate_volume_price_correlation(window_size=20)
    
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(data.index, correlation, label='Corrélation Volume-Prix')
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    ax5.axhline(y=-0.5, color='r', linestyle='--', alpha=0.3)
    ax5.set_title("Corrélation Volume-Prix")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Coefficient de Corrélation")
    
    # 5. Analyse du Delta de Volume
    delta_analysis = volume_delta(data, window_size=14)
    
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(delta_analysis.index, delta_analysis['cumulative_delta'], label='Delta Cumulatif')
    ax6.plot(delta_analysis.index, delta_analysis['volume_delta'], label='Delta de Volume', alpha=0.5)
    ax6.set_title("Delta de Volume (Pression Acheteur vs Vendeur)")
    ax6.set_xlabel("Date")
    ax6.set_ylabel("Delta")
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('volume_analysis_example.png', dpi=300)
    plt.close()
    
    # 6. Validation des signaux
    print("\nValidation des signaux aux anomalies de volume:")
    
    # Valider les signaux aux points d'anomalie de volume
    for idx in [40, 80, 120, 160]:
        validation = analyzer.validate_signal(idx, lookback=10)
        print(f"\nSignal à l'indice {idx} (date {data.index[idx].date()}):")
        print(f"  Prix: {data['close'].iloc[idx]:.2f}")
        print(f"  Volume: {data['volume'].iloc[idx]:.0f}")
        print(f"  Validation: {'VALIDE' if validation['is_valid'] else 'INVALIDE'}")
        print(f"  Force du signal: {validation['strength']}")
        print(f"  Rapport volume: {validation['volume_ratio']:.2f}x")
        print(f"  Corrélation volume-prix: {validation['volume_price_correlation']:.2f}")
        print(f"  Recommandation: {validation['recommendation']}")
    
    print("\nDémonstration terminée. Voir l'image 'volume_analysis_example.png' pour les graphiques.")


if __name__ == "__main__":
    main() 