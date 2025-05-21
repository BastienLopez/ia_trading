"""
Exemples pour l'agent DQN (base, UCB, replay priorisé).
"""

import numpy as np
import torch
from ai_trading.rl.agents.dqn_agent import DQNAgent, PrioritizedReplayBuffer
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_dqn():
    """Test de l'agent DQN de base."""
    logger.info("Test de l'agent DQN de base")
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size, use_ucb=False, use_prioritized_replay=False, device="cpu")

    # Test de l'initialisation
    assert agent.state_size == state_size
    assert agent.action_size == action_size
    assert agent.epsilon == 1.0
    assert len(agent.memory) == 0
    assert agent.steps_done == 0

    # Test de la sélection d'action
    state = np.random.randn(state_size)
    action = agent.select_action(state, training=True)
    assert 0 <= action < action_size
    assert agent.steps_done == 1

    # Test de l'apprentissage
    for _ in range(agent.batch_size):
        next_state = np.random.randn(state_size)
        reward = np.random.rand()
        done = np.random.choice([False, True])
        agent.memory.append((
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done])
        ))

    loss = agent.optimize_model()
    assert loss is not None
    logger.info("Test DQN de base réussi")

def test_ucb_exploration():
    """Test de l'exploration UCB."""
    logger.info("Test de l'exploration UCB")
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size, use_ucb=True, use_prioritized_replay=False, device="cpu")

    # Test de l'exploration initiale
    state = np.random.randn(state_size)
    for i in range(action_size):
        action = agent.select_action(state)
        assert action == i  # Les premières actions doivent être explorées séquentiellement
        assert agent.action_counts[action] == 1

    # Test de la mise à jour UCB
    action = agent.select_action(state)
    agent.update_ucb(action, 1.0)
    assert agent.action_counts[action] > 0
    assert agent.action_values[action] > 0  # On vérifie uniquement l'action mise à jour
    logger.info("Test UCB réussi")

def test_prioritized_replay():
    """Test du replay priorisé."""
    logger.info("Test du replay priorisé")
    state_size = 4
    action_size = 2
    buffer_size = 100
    agent = DQNAgent(state_size, action_size, use_ucb=False, use_prioritized_replay=True, device="cpu", buffer_size=buffer_size)

    # Remplir le buffer avec des priorités connues
    for _ in range(buffer_size):
        state = np.random.randn(state_size)
        action = agent.select_action(state)
        next_state = np.random.randn(state_size)
        reward = np.random.rand()
        done = np.random.choice([False, True])
        agent.memory.push(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done])
        )
    # Initialiser toutes les priorités à 1.0
    agent.memory.priorities = [1.0 for _ in range(buffer_size)]

    # Test de l'échantillonnage priorisé
    batch, indices, weights = agent.memory.sample(agent.batch_size)
    assert len(batch.state) == agent.batch_size
    assert len(indices) == agent.batch_size
    assert len(weights) == agent.batch_size

    # Test de la mise à jour des priorités
    new_priorities = np.random.rand(agent.batch_size)
    agent.memory.update_priorities(indices, new_priorities)

    # Vérification des priorités mises à jour uniquement pour les indices concernés
    for i, (idx, priority) in enumerate(zip(indices, new_priorities)):
        stored_priority = agent.memory.priorities[idx]
        if np.isclose(stored_priority, 1.0, rtol=1e-3):
            print(f"Index {i}: La priorité n'a pas été modifiée pour l'indice {idx}")
        assert not np.isclose(stored_priority, 1.0, rtol=1e-3), f"La priorité n'a pas été modifiée pour l'indice {idx}"

    # Test de l'optimisation avec replay priorisé
    loss = agent.optimize_model()
    assert loss is not None
    logger.info("Test replay priorisé réussi")

def test_model_save_load():
    """Test de la sauvegarde et du chargement du modèle."""
    logger.info("Test de sauvegarde/chargement du modèle")
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size, device="cpu")

    # Entraînement minimal
    state = np.random.randn(state_size)
    action = agent.select_action(state)
    next_state = np.random.randn(state_size)
    reward = np.random.rand()
    done = False
    if hasattr(agent.memory, "push"):
        agent.memory.push(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done])
        )
    else:
        agent.memory.append((
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done])
        ))
    agent.optimize_model()

    # Sauvegarde
    save_path = "dqn_model_test.pt"
    agent.save(save_path)

    # Chargement
    new_agent = DQNAgent(state_size, action_size, device="cpu")
    new_agent.load(save_path)

    # Vérification
    assert new_agent.steps_done == agent.steps_done
    assert new_agent.epsilon == agent.epsilon
    assert np.array_equal(new_agent.action_counts, agent.action_counts)
    assert np.array_equal(new_agent.action_values, agent.action_values)
    logger.info("Test sauvegarde/chargement réussi")

if __name__ == "__main__":
    # Exécution des tests
    test_basic_dqn()
    test_ucb_exploration()
    test_prioritized_replay()
    test_model_save_load()
    logger.info("Tous les tests ont été exécutés avec succès") 