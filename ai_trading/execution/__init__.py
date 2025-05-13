"""
Module d'exécution pour les stratégies de trading optimisées.

Ce module fournit différentes stratégies d'exécution pour les ordres de trading
afin d'optimiser l'exécution en fonction des conditions de marché.
"""

from .adaptive_execution import (
    AdaptiveExecutionStrategy,
    AdaptiveExecutor,
    AggressiveExecutionStrategy,
    ExecutionMode,
    ExecutionStrategy,
    NormalExecutionStrategy,
    PassiveExecutionStrategy,
)
from .risk_manager import (
    DrawdownControl,
    DynamicStopLoss,
    ExposureManager,
    RiskConfig,
    RiskLevel,
    RiskManager,
    StopLossConfig,
    StopType,
)
from .smart_routing import (
    BestLiquidityStrategy,
    ExchangeInfo,
    ExchangePriority,
    ExchangeVenue,
    FastestExecutionStrategy,
    LowestFeeStrategy,
    OrderRoutingStrategy,
    SmartRouter,
    SmartRoutingStrategy,
)

__all__ = [
    # Exécution adaptative
    "ExecutionStrategy",
    "ExecutionMode",
    "AdaptiveExecutor",
    "PassiveExecutionStrategy",
    "NormalExecutionStrategy",
    "AggressiveExecutionStrategy",
    "AdaptiveExecutionStrategy",
    # Routage intelligent
    "ExchangeVenue",
    "ExchangePriority",
    "ExchangeInfo",
    "OrderRoutingStrategy",
    "SmartRouter",
    "LowestFeeStrategy",
    "BestLiquidityStrategy",
    "FastestExecutionStrategy",
    "SmartRoutingStrategy",
    # Gestion avancée du risque
    "StopType",
    "RiskLevel",
    "StopLossConfig",
    "RiskConfig",
    "DynamicStopLoss",
    "DrawdownControl",
    "ExposureManager",
    "RiskManager",
]

# Ces modules seront implémentés ultérieurement
# from ai_trading.execution.market_impact import (
#     MarketImpactEstimator,
#     SlippageModel
# )

# from ai_trading.execution.order_splitter import (
#     OrderSplitter,
#     LiquidityBasedSplitter
# )
