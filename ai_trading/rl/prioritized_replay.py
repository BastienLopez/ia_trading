"""
Module d'adaptation pour la compatibilité avec l'agent TransformerSAC.
Ce module réexporte PrioritizedReplayMemory sous le nom PrioritizedReplayBuffer.
"""

from ai_trading.rl.prioritized_replay_memory import PrioritizedReplayMemory

# Réexporter PrioritizedReplayMemory sous le nom PrioritizedReplayBuffer
# pour maintenir la compatibilité avec TransformerSACAgent
PrioritizedReplayBuffer = PrioritizedReplayMemory 