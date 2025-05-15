"""
Exemple de correction du biais de survivance dans le backtesting.

Ce script montre comment corriger le biais de survivance
lors de l'évaluation de stratégies de trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import des modules de backtesting
from ai_trading.ml.backtesting.survivorship_bias import (
    SurvivorshipBiasCorrector,
    MarketRegimeDetector,
    TimeSeriesCV,
    BootstrapAnalyzer,
    StatisticalTester,
    MarketRegime
)

def load_historical_data():
    """Charge des données historiques simulées avec actifs survivants et délistés."""
    # Période de simulation
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Dictionnaire pour stocker les données
    market_data = {}
    
    # Créer des actifs survivants
    for i in range(3):
        symbol = f"ASSET_{i+1}"
        
        # Simuler une tendance de fond avec du bruit
        trend = np.linspace(100, 120, len(dates)) if i % 2 == 0 else np.linspace(100, 80, len(dates))
        noise = np.random.normal(0, 5, len(dates))
        close_prices = trend + noise
        
        data = pd.DataFrame({
            'open': close_prices - np.random.uniform(0, 2, len(dates)),
            'high': close_prices + np.random.uniform(1, 3, len(dates)),
            'low': close_prices - np.random.uniform(1, 3, len(dates)),
            'close': close_prices,
            'volume': np.random.uniform(1000, 5000, len(dates))
        }, index=dates)
        
        market_data[symbol] = data
    
    # Créer des actifs délistés
    delisted_info = {}
    
    # Premier actif délisté (faillite à mi-chemin)
    delisting_date1 = start_date + (end_date - start_date) // 2
    mid_point = len(dates) // 2
    
    # Simuler un effondrement avant la délisting
    trend1 = np.concatenate([
        np.linspace(100, 80, mid_point - 30),
        np.linspace(80, 20, 30)
    ])
    noise1 = np.random.normal(0, 3, mid_point)
    close_prices1 = trend1 + noise1
    
    delisted_data1 = pd.DataFrame({
        'open': close_prices1 - np.random.uniform(0, 2, mid_point),
        'high': close_prices1 + np.random.uniform(1, 3, mid_point),
        'low': close_prices1 - np.random.uniform(1, 3, mid_point),
        'close': close_prices1,
        'volume': np.random.uniform(1000, 5000, mid_point)
    }, index=dates[:mid_point])
    
    market_data["DELISTED_1"] = delisted_data1
    delisted_info["DELISTED_1"] = {
        "delisting_date": delisting_date1,
        "reason": "bankruptcy",
        "price_impact": -0.9  # Chute de 90%
    }
    
    # Second actif délisté (fusion vers la fin)
    delisting_date2 = end_date - timedelta(days=30)
    end_point = len(dates) - 30
    
    # Simuler une hausse avant la fusion
    trend2 = np.concatenate([
        np.linspace(100, 90, end_point - 30),
        np.linspace(90, 110, 30)
    ])
    noise2 = np.random.normal(0, 4, end_point)
    close_prices2 = trend2 + noise2
    
    delisted_data2 = pd.DataFrame({
        'open': close_prices2 - np.random.uniform(0, 2, end_point),
        'high': close_prices2 + np.random.uniform(1, 3, end_point),
        'low': close_prices2 - np.random.uniform(1, 3, end_point),
        'close': close_prices2,
        'volume': np.random.uniform(1000, 5000, end_point)
    }, index=dates[:end_point])
    
    market_data["DELISTED_2"] = delisted_data2
    delisted_info["DELISTED_2"] = {
        "delisting_date": delisting_date2,
        "reason": "merger",
        "price_impact": 0.15  # Prime de fusion de 15%
    }
    
    return market_data, delisted_info, dates

def run_backtest(market_data, is_corrected=False):
    """Simule un backtest avec et sans correction du biais de survivance."""
    # Simuler une stratégie simple (achat et conservation)
    portfolio_values = {}
    initial_capital = 100000
    
    for symbol, data in market_data.items():
        # Acheter une quantité égale de chaque actif au début
        allocation = initial_capital / len(market_data)
        shares = allocation / data['close'].iloc[0]
        
        # Calculer la valeur du portefeuille pour cet actif
        portfolio_values[symbol] = data['close'] * shares
    
    # Calculer la valeur totale du portefeuille par jour
    total_value = pd.DataFrame(portfolio_values).sum(axis=1)
    
    # Calculer les rendements
    returns = total_value.pct_change().dropna()
    
    # Calculer les métriques de performance
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualisé
    total_return = (total_value.iloc[-1] / total_value.iloc[0]) - 1
    max_drawdown = (total_value / total_value.cummax() - 1).min()
    
    result = {
        'equity_curve': total_value,
        'returns': returns,
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'max_drawdown': max_drawdown
    }
    
    # Afficher les résultats
    print(f"\n== Résultats du backtest {'avec correction' if is_corrected else 'sans correction'} ==")
    print(f"Rendement total: {total_return:.2%}")
    print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
    print(f"Drawdown maximum: {max_drawdown:.2%}")
    
    return result

def backtest_with_market_regimes(market_data, dates):
    """Effectue un backtest avec détection des régimes de marché."""
    detector = MarketRegimeDetector()
    
    # Détecter les régimes de marché pour un actif représentatif
    regimes_data = detector.detect_regimes(market_data, "ASSET_1")
    regimes = regimes_data["ASSET_1"]["regime"]
    
    # Créer un validateur croisé avec prise en compte des régimes
    cv = TimeSeriesCV(config={"n_splits": 3, "detect_regime": True})
    windows = cv.split(market_data, "ASSET_1")
    
    # Afficher les fenêtres de test et les régimes associés
    print("\n== Fenêtres de test par régime de marché ==")
    for i, window in enumerate(windows):
        print(f"Fenêtre {i+1}:")
        print(f"  - Période d'entraînement: {window.train_start.date()} à {window.train_end.date()}")
        print(f"  - Période de test: {window.test_start.date()} à {window.test_end.date()}")
        print(f"  - Régime dominant: {window.regime.name}")
    
    # Visualiser les régimes de marché
    plt.figure(figsize=(12, 8))
    
    # Tracer le prix de l'actif
    asset_data = market_data["ASSET_1"]
    plt.subplot(2, 1, 1)
    plt.plot(asset_data.index, asset_data['close'], label='Prix ASSET_1')
    plt.title('Prix et Régimes de Marché')
    plt.ylabel('Prix')
    plt.grid(True)
    
    # Tracer les régimes de marché
    plt.subplot(2, 1, 2)
    for regime in MarketRegime:
        mask = regimes == regime.value
        if mask.any():
            plt.fill_between(dates[mask], 0, 1, label=regime.name, alpha=0.7)
    
    plt.xlabel('Date')
    plt.ylabel('Régime')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    
    # Enregistrer le graphique
    plt.savefig("examples/ml/backtesting/market_regimes.png")
    plt.close()
    
    print("Graphique des régimes de marché enregistré sous 'examples/ml/backtesting/market_regimes.png'")
    
    return regimes

def survivorship_bias_example():
    """Exemple principal de correction du biais de survivance."""
    # Charger les données
    market_data, delisted_info, dates = load_historical_data()
    
    # Effectuer un backtest sans correction
    uncorrected_result = run_backtest(market_data, is_corrected=False)
    
    # Créer un correcteur de biais de survivance
    corrector = SurvivorshipBiasCorrector()
    
    # Corriger les données
    corrected_data = corrector.correct_dataset(market_data, delisted_info)
    
    # Effectuer un backtest avec correction
    corrected_result = run_backtest(corrected_data, is_corrected=True)
    
    # Comparer les résultats
    print("\n== Comparaison des performances ==")
    print(f"Différence de rendement: {corrected_result['total_return'] - uncorrected_result['total_return']:.2%}")
    print(f"Différence de Sharpe: {corrected_result['sharpe_ratio'] - uncorrected_result['sharpe_ratio']:.2f}")
    
    # Visualiser les courbes d'équité
    plt.figure(figsize=(12, 6))
    plt.plot(uncorrected_result['equity_curve'].index, uncorrected_result['equity_curve'], 
             label='Sans correction', color='blue')
    plt.plot(corrected_result['equity_curve'].index, corrected_result['equity_curve'], 
             label='Avec correction', color='red')
    
    plt.title('Impact du Biais de Survivance sur la Performance')
    plt.xlabel('Date')
    plt.ylabel('Valeur du Portefeuille')
    plt.grid(True)
    plt.legend()
    
    # Enregistrer le graphique
    plt.savefig("examples/ml/backtesting/survivorship_bias_impact.png")
    plt.close()
    
    print("Graphique de l'impact du biais de survivance enregistré sous 'examples/ml/backtesting/survivorship_bias_impact.png'")
    
    # Effectuer une analyse avec régimes de marché
    regimes = backtest_with_market_regimes(market_data, dates)
    
    # Effectuer des tests statistiques
    tester = StatisticalTester()
    test_results = tester.test_strategy(corrected_result, corrected_data)
    
    # Afficher le résumé des tests
    summary = tester.summarize_tests(test_results)
    print("\n" + summary)
    
    return corrected_result, uncorrected_result, regimes

if __name__ == "__main__":
    corrected_result, uncorrected_result, regimes = survivorship_bias_example() 