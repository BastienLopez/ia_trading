"""
Exemple d'utilisation du module de backtesting avancé.

Ce script démontre comment utiliser les fonctionnalités avancées du module de backtesting,
notamment:
- L'exécution réaliste des ordres avec slippage
- La modélisation des coûts de transaction
- Le stress testing avec différents scénarios
- L'analyse de sensibilité des paramètres

Il utilise une stratégie simple de croisement de moyennes mobiles à titre d'exemple.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import json
import os

from ai_trading.ml.backtesting.backtest_engine import BacktestEngine, Order, OrderType, OrderSide
from ai_trading.ml.backtesting.stress_testing import StressTester, ScenarioType
from ai_trading.ml.backtesting.sensitivity_analysis import SensitivityAnalyzer, ParameterRange

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=1000, start_date=None, volatility=0.02):
    """Génère des données OHLCV avec différentes phases de marché."""
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    
    # Créer un index temporel
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Générer les prix avec différentes phases
    prices = []
    
    # Phase haussière initiale (30% des données)
    uptrend_length = int(n_samples * 0.3)
    uptrend = np.linspace(100, 140, uptrend_length) + np.random.normal(0, 100 * volatility, uptrend_length)
    prices.extend(uptrend)
    
    # Phase latérale (20% des données)
    sideways_length = int(n_samples * 0.2)
    sideways = np.full(sideways_length, 140) + np.random.normal(0, 140 * volatility * 1.5, sideways_length)
    prices.extend(sideways)
    
    # Phase baissière (30% des données)
    downtrend_length = int(n_samples * 0.3)
    downtrend = np.linspace(140, 90, downtrend_length) + np.random.normal(0, 90 * volatility, downtrend_length)
    prices.extend(downtrend)
    
    # Phase de reprise (20% des données)
    recovery_length = n_samples - len(prices)
    recovery = np.linspace(90, 120, recovery_length) + np.random.normal(0, 100 * volatility, recovery_length)
    prices.extend(recovery)
    
    # Créer les autres colonnes
    high_prices = [p + np.random.uniform(0, p * 0.01) for p in prices]
    low_prices = [p - np.random.uniform(0, p * 0.01) for p in prices]
    open_prices = [p + np.random.normal(0, p * 0.005) for p in prices]
    
    # Générer des volumes avec des pics occasionnels
    volumes = np.random.uniform(1000, 3000, n_samples)
    volume_spikes = np.random.choice(range(n_samples), size=int(n_samples * 0.05), replace=False)
    volumes[volume_spikes] *= 3
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return data

def moving_average_crossover_strategy(data, params):
    """
    Stratégie simple basée sur le croisement de moyennes mobiles.
    
    Args:
        data: Données de marché
        params: Paramètres de la stratégie
        
    Returns:
        DataFrame avec les signaux
    """
    df = data.copy()
    
    # Paramètres
    ma_short = params.get("ma_short", 10)
    ma_long = params.get("ma_long", 30)
    
    # Calcul des moyennes mobiles
    df["ma_short"] = df["close"].rolling(window=ma_short).mean()
    df["ma_long"] = df["close"].rolling(window=ma_long).mean()
    
    # Signaux
    df["signal"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1
    df.loc[df["ma_short"] < df["ma_long"], "signal"] = -1
    
    # Stop loss et take profit
    stop_loss_pct = params.get("stop_loss_pct", 0.02)
    take_profit_pct = params.get("take_profit_pct", 0.04)
    position_size = params.get("position_size", 0.1)
    
    df["stop_loss"] = df["close"] * (1 - stop_loss_pct)
    df["take_profit"] = df["close"] * (1 + take_profit_pct)
    df["position_size"] = position_size
    
    return df

def rsi_strategy(data, params):
    """
    Stratégie basée sur l'indicateur RSI (Relative Strength Index).
    
    Args:
        data: Données de marché
        params: Paramètres de la stratégie
        
    Returns:
        DataFrame avec les signaux
    """
    df = data.copy()
    
    # Paramètres
    lookback_window = params.get("lookback_window", 14)
    rsi_upper = params.get("rsi_upper", 70)
    rsi_lower = params.get("rsi_lower", 30)
    profit_target_pct = params.get("profit_target_pct", 0.03)
    stop_loss_pct = params.get("stop_loss_pct", 0.02)
    trade_size = params.get("trade_size", 1000.0)
    
    # Calcul du RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=lookback_window).mean()
    avg_loss = loss.rolling(window=lookback_window).mean()
    
    rs = avg_gain / avg_loss.replace(0, 0.001)  # Éviter la division par zéro
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Signaux basés sur le RSI
    df["signal"] = 0
    df.loc[df["rsi"] < rsi_lower, "signal"] = 1  # Surachat (signal d'achat)
    df.loc[df["rsi"] > rsi_upper, "signal"] = -1  # Survente (signal de vente)
    
    # Stop loss et take profit
    df["stop_loss"] = df["close"] * (1 - stop_loss_pct)
    df["take_profit"] = df["close"] * (1 + profit_target_pct)
    df["position_size"] = trade_size / df["close"]
    
    return df

def run_basic_backtest(market_data, strategy_params=None):
    """
    Exécute un backtest de base.
    
    Args:
        market_data: Données de marché
        strategy_params: Paramètres de la stratégie
        
    Returns:
        Résultats du backtest
    """
    logger.info("Exécution d'un backtest de base")
    
    # Paramètres par défaut si non fournis
    if strategy_params is None:
        strategy_params = {
            "lookback_window": 14,
            "rsi_upper": 70,
            "rsi_lower": 30,
            "profit_target_pct": 0.03,
            "stop_loss_pct": 0.02,
            "trade_size": 1000.0
        }
    
    # Créer le moteur de backtest
    engine = BacktestEngine(
        config={
            "initial_capital": 50000,
            "fee_model": {
                "maker_fee": 0.001,
                "taker_fee": 0.002,
                "slippage": 0.001
            }
        }
    )
    
    # Exécuter le backtest
    results = engine.run(market_data, rsi_strategy, strategy_params)
    
    # Afficher les statistiques
    logger.info(f"Nombre de trades: {len(results.trades)}")
    logger.info(f"P&L total: ${results.statistics['total_pnl']:.2f}")
    logger.info(f"Taux de réussite: {results.statistics['win_rate']:.1f}%")
    logger.info(f"Drawdown maximum: {results.statistics['max_drawdown']:.1f}%")
    
    return results

def run_stress_tests(market_data, base_strategy_params=None):
    """
    Exécute des tests de stress sur différents scénarios.
    
    Args:
        market_data: Données de marché
        base_strategy_params: Paramètres de base pour la stratégie
        
    Returns:
        Résultats des tests de stress
    """
    logger.info("Exécution des tests de stress")
    
    # Paramètres par défaut si non fournis
    if base_strategy_params is None:
        base_strategy_params = {
            "lookback_window": 14,
            "rsi_upper": 70,
            "rsi_lower": 30,
            "profit_target_pct": 0.03,
            "stop_loss_pct": 0.02,
            "trade_size": 1000.0
        }
    
    # Créer le générateur de tests de stress
    stress_tester = StressTester()
    
    # Résultats par scénario
    results = {}
    
    # Scénario 1: Crash de marché
    logger.info("Test du scénario: Scénario de crash")
    crash_data = stress_tester.generate_crash_scenario(market_data)
    crash_result = run_basic_backtest(crash_data, base_strategy_params)
    results["Scénario de crash"] = {
        "total_pnl": crash_result.statistics["total_pnl"],
        "win_rate": crash_result.statistics["win_rate"],
        "max_drawdown": crash_result.statistics["max_drawdown"]
    }
    
    # Scénario 2: Pic de volatilité
    logger.info("Test du scénario: Pic de volatilité")
    volatility_data = stress_tester.generate_volatility_spike_scenario(market_data)
    volatility_result = run_basic_backtest(volatility_data, base_strategy_params)
    results["Pic de volatilité"] = {
        "total_pnl": volatility_result.statistics["total_pnl"],
        "win_rate": volatility_result.statistics["win_rate"],
        "max_drawdown": volatility_result.statistics["max_drawdown"]
    }
    
    # Scénario 3: Crise de liquidité
    logger.info("Test du scénario: Crise de liquidité")
    liquidity_data = stress_tester.generate_liquidity_crisis_scenario(market_data)
    liquidity_result = run_basic_backtest(liquidity_data, base_strategy_params)
    results["Crise de liquidité"] = {
        "total_pnl": liquidity_result.statistics["total_pnl"],
        "win_rate": liquidity_result.statistics["win_rate"],
        "max_drawdown": liquidity_result.statistics["max_drawdown"]
    }
    
    # Scénario 4: Flash crash
    logger.info("Test du scénario: Flash crash")
    flash_crash_data = stress_tester.generate_flash_crash_scenario(market_data)
    flash_crash_result = run_basic_backtest(flash_crash_data, base_strategy_params)
    results["Flash crash"] = {
        "total_pnl": flash_crash_result.statistics["total_pnl"],
        "win_rate": flash_crash_result.statistics["win_rate"],
        "max_drawdown": flash_crash_result.statistics["max_drawdown"]
    }
    
    # Afficher les résultats comparatifs
    logger.info("Résultats comparatifs des tests de stress:")
    logger.info(json.dumps(results, indent=2))
    
    return results

def run_sensitivity_analysis(market_data):
    """
    Exécute une analyse de sensibilité pour identifier les paramètres les plus importants.
    
    Args:
        market_data: Dictionnaire des données de marché par symbole
        
    Returns:
        Résultats de l'analyse de sensibilité
    """
    logger.info("Exécution de l'analyse de sensibilité...")
    
    # Définir les plages de paramètres à tester
    parameter_ranges = [
        ParameterRange(
            name="lookback_window",
            values=[5, 10, 15, 20, 25, 30],
            display_name="Période de lookback",
            description="Nombre de barres pour le calcul des indicateurs"
        ),
        ParameterRange(
            name="rsi_upper",
            values=[70, 75, 80, 85],
            display_name="Seuil RSI haut",
            description="Seuil supérieur pour les signaux de survente"
        ),
        ParameterRange(
            name="rsi_lower",
            values=[15, 20, 25, 30],
            display_name="Seuil RSI bas",
            description="Seuil inférieur pour les signaux de surachat"
        ),
        ParameterRange(
            name="profit_target_pct",
            values=[0.01, 0.02, 0.03, 0.04, 0.05],
            display_name="Target de profit",
            description="Pourcentage de profit cible"
        ),
        ParameterRange(
            name="stop_loss_pct",
            values=[0.01, 0.015, 0.02, 0.025, 0.03],
            display_name="Stop loss",
            description="Pourcentage de stop loss"
        )
    ]
    
    # Paramètres de base (non variables)
    base_parameters = {
        "trade_size": 1000.0,
        "max_positions": 3,
        "position_sizing": "equal"
    }
    
    # Configurer l'analyse de sensibilité
    analyzer = SensitivityAnalyzer(
        config={
            "metrics": ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"],
            "optimization_target": "sharpe_ratio",
            "parallel": {
                "enabled": True,
                "max_workers": os.cpu_count() - 1 if os.cpu_count() > 1 else 1
            }
        }
    )
    
    # Exécuter l'analyse
    results = analyzer.analyze(
        market_data=market_data,
        strategy_fn=rsi_strategy,
        parameter_ranges=parameter_ranges,
        base_parameters=base_parameters
    )
    
    # Gérer le cas où l'analyse ne produit pas de résultat valide
    if results is None:
        logger.error("L'analyse de sensibilité n'a pas produit de résultat valide.")
        return None
    
    # Afficher les résultats
    logger.info("Analyse de sensibilité terminée")
    logger.info(f"Meilleurs paramètres : {results.best_parameters}")
    logger.info(f"Nombre de combinaisons testées: {len(results.metrics)}")
    
    # Sauvegarder les résultats
    results.save("output/sensitivity_results.json")
    
    # Visualisations
    if not results.metrics.empty:
        # Créer un graphique de sensibilité
        plt.figure(figsize=(12, 8))
        fig = results.plot_sensitivity(metric="sharpe_ratio", top_n=3)
        if fig:
            plt.savefig("output/sensitivity_chart.png")
        
        # Créer un heatmap
        if "lookback_window" in results.metrics.columns and "profit_target_pct" in results.metrics.columns:
            plt.figure(figsize=(12, 8))
            fig = results.plot_heatmap("lookback_window", "profit_target_pct", metric="sharpe_ratio")
            if fig:
                plt.savefig("output/heatmap.png")
    
    return results

def main():
    """Fonction principale."""
    logger.info("Démarrage de l'exemple de backtesting avancé")
    
    # 1. Générer des données de marché
    logger.info("Génération des données de marché...")
    btc_data = generate_sample_data(n_samples=2000, volatility=0.03)
    eth_data = generate_sample_data(n_samples=2000, volatility=0.04)
    market_data = {
        "BTC/USD": btc_data,
        "ETH/USD": eth_data
    }
    
    # 2. Exécuter un backtest de base pour référence
    logger.info("Exécution du backtest de base...")
    base_result = run_basic_backtest(market_data)
    
    # 3. Exécuter des tests de stress
    logger.info("Exécution des tests de stress...")
    stress_results = run_stress_tests(market_data)
    
    # 4. Exécuter une analyse de sensibilité
    logger.info("Exécution de l'analyse de sensibilité...")
    sensitivity_results = run_sensitivity_analysis(market_data)
    
    # 5. Exécuter un backtest final avec les paramètres optimisés
    logger.info("Exécution du backtest final avec paramètres optimisés...")
    final_result = run_basic_backtest(market_data, sensitivity_results.best_parameters)
    
    logger.info("Exemple de backtesting avancé terminé")

if __name__ == "__main__":
    main() 