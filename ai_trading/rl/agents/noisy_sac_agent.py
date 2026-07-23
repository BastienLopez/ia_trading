"""Point de compatibilité pour l'ancien import ``NoisySACAgent``.

Les réseaux bruités sont implémentés et testés sur le DQN, où ils apportent
une exploration paramétrique mesurable. Le SAC utilise son exploration
stochastique native ; cette classe ne prétend donc plus fournir une seconde
implémentation NoisyLinear cassée.
"""

import warnings

from ai_trading.rl.agents.sac_agent import OptimizedSACAgent


class NoisySACAgent(OptimizedSACAgent):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "NoisySACAgent est déprécié : utilisez OptimizedSACAgent. "
            "L'exploration SAC est stochastique par conception.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
