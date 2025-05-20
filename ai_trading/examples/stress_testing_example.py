"""
Exemple d'utilisation du module de stress testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ai_trading.ml.backtesting.stress_testing import StressTester, ScenarioType, StressScenario

def create_sample_data(start_date='2023-01-01', end_date='2023-12-31', freq='1D'):
    """Crée un jeu de données OHLCV pour l'exemple."""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Assurer que high > open > close > low
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(2, 0.5, len(dates)))
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(2, 0.5, len(dates)))
    
    return data

def plot_scenario_comparison(original_data, stressed_data, title):
    """Affiche une comparaison entre les données originales et stressées."""
    plt.figure(figsize=(15, 10))
    
    # Prix de clôture
    plt.subplot(2, 1, 1)
    plt.plot(original_data.index, original_data['close'], label='Original', color='blue')
    plt.plot(stressed_data.index, stressed_data['close'], label='Stressed', color='red')
    plt.title(f'{title} - Prix de clôture')
    plt.legend()
    plt.grid(True)
    
    # Volume
    plt.subplot(2, 1, 2)
    plt.plot(original_data.index, original_data['volume'], label='Original', color='blue')
    plt.plot(stressed_data.index, stressed_data['volume'], label='Stressed', color='red')
    plt.title(f'{title} - Volume')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Fonction principale de démonstration."""
    # Créer des données de test
    data = create_sample_data()
    
    # Initialiser le testeur de stress
    tester = StressTester()
    
    # 1. Scénario de crash
    print("\n1. Test du scénario de crash")
    stressed_crash = tester.generate_stress_scenario(data, ScenarioType.CRASH)
    plot_scenario_comparison(data, stressed_crash, "Scénario de crash")
    
    # 2. Scénario de pic de volatilité
    print("\n2. Test du scénario de pic de volatilité")
    stressed_vol = tester.generate_stress_scenario(data, ScenarioType.VOLATILITY_SPIKE)
    plot_scenario_comparison(data, stressed_vol, "Pic de volatilité")
    
    # 3. Scénario de crise de liquidité
    print("\n3. Test du scénario de crise de liquidité")
    stressed_liq = tester.generate_stress_scenario(data, ScenarioType.LIQUIDITY_CRISIS)
    plot_scenario_comparison(data, stressed_liq, "Crise de liquidité")
    
    # 4. Scénario de crash éclair
    print("\n4. Test du scénario de crash éclair")
    stressed_flash = tester.generate_stress_scenario(data, ScenarioType.FLASH_CRASH)
    plot_scenario_comparison(data, stressed_flash, "Crash éclair")
    
    # 5. Utilisation de la classe StressScenario
    print("\n5. Test de la classe StressScenario")
    scenario = StressScenario(
        name="Custom Stress",
        price_shock=-0.3,
        volatility_multiplier=2.0,
        volume_multiplier=0.5,
        duration=timedelta(days=10),
        start_date=data.index[50]
    )
    
    market_data = {'TEST': data}
    stressed_custom = scenario.apply_to_data(market_data)['TEST']
    plot_scenario_comparison(data, stressed_custom, "Scénario personnalisé")
    
    # Afficher quelques statistiques
    print("\nStatistiques des scénarios:")
    scenarios = {
        'Original': data,
        'Crash': stressed_crash,
        'Volatility Spike': stressed_vol,
        'Liquidity Crisis': stressed_liq,
        'Flash Crash': stressed_flash,
        'Custom': stressed_custom
    }
    
    stats = pd.DataFrame({
        name: {
            'Rendement moyen': df['close'].pct_change().mean(),
            'Volatilité': df['close'].pct_change().std(),
            'Volume moyen': df['volume'].mean(),
            'Max drawdown': (df['close'] / df['close'].expanding().max() - 1).min()
        }
        for name, df in scenarios.items()
    })
    
    print("\n", stats.round(4))

if __name__ == "__main__":
    main() 