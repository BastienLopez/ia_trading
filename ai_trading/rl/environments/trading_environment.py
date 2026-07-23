"""Compatibilité avec l'ancien chemin d'import de l'environnement RL.

L'implémentation unique est :mod:`ai_trading.rl.trading_environment`.
Conserver cet alias évite de casser les scripts externes tout en empêchant
le retour d'un environnement historique incomplet et divergent.
"""

from ai_trading.rl.trading_environment import TradingEnvironment

__all__ = ["TradingEnvironment"]
