"""
Module de backtesting avancé qui fournit des outils pour simuler des stratégies
de trading avec un haut niveau de réalisme, incluant des modèles d'exécution d'ordres,
des coûts de transaction réalistes et des outils d'analyse de performance.
"""

from ai_trading.ml.backtesting.backtest_engine import BacktestEngine
from ai_trading.ml.backtesting.execution_model import ExecutionModel
from ai_trading.ml.backtesting.transaction_costs import TransactionCostModel
from ai_trading.ml.backtesting.stress_testing import StressTester
from ai_trading.ml.backtesting.sensitivity_analysis import SensitivityAnalyzer
from ai_trading.ml.backtesting.survivorship_bias import (
    SurvivorshipBiasCorrector,
    MarketRegimeDetector,
    TimeSeriesCV,
    BootstrapAnalyzer,
    StatisticalTester,
    MarketRegime
) 