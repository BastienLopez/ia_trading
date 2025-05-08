# -*- coding: utf-8 -*-
"""
Package contenant les modules de données pour l'IA Trading.
"""

from ai_trading.data.enhanced_market_data import EnhancedMarketDataFetcher
from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader
from ai_trading.data.market_data import MarketDataFetcher
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.data.technical_indicators import TechnicalIndicators

try:
    HAVE_OPTIMIZERS = True
except ImportError:
    HAVE_OPTIMIZERS = False

__all__ = [
    "MarketDataFetcher",
    "EnhancedMarketDataFetcher",
    "TechnicalIndicators",
    "generate_synthetic_market_data",
    "FinancialDataset",
    "get_financial_dataloader",
]

if HAVE_OPTIMIZERS:
    __all__.extend(
        [
            "convert_to_parquet",
            "convert_to_hdf5",
            "optimize_dataset_storage",
        ]
    )
