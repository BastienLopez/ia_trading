"""Point d'entrée DQN historique.

L'implémentation maintenue vit dans :mod:`ai_trading.rl.agents.dqn_agent`.
Ce module conserve les imports existants sans maintenir une seconde version de
l'algorithme, source d'écarts entre entraînement et API.
"""

from ai_trading.rl.agents.dqn_agent import DQNAgent

__all__ = ["DQNAgent"]
