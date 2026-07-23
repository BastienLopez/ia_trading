"""Compatibilité avec l'ancien chemin d'import multi-actifs.

L'environnement complet (indicateurs, contraintes, délais et récompenses) est
centralisé dans :mod:`ai_trading.rl.multi_asset_trading_environment`.
"""

from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment

__all__ = ["MultiAssetTradingEnvironment"]
