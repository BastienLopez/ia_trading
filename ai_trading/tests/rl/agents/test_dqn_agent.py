"""
Tests pour l'agent DQN avec UCB et replay priorisé.
"""

import pytest
import numpy as np
import torch
from ai_trading.rl.agents.dqn_agent import DQNAgent, PrioritizedReplayBuffer

@pytest.fixture
def dqn_agent():
    """Fixture pour créer un agent DQN de test."""
    return DQNAgent(
        state_size=4,
        action_size=2,
        hidden_size=64,
        learning_rate=0.001,
        buffer_size=1000,
        batch_size=32
    )

@pytest.fixture
def replay_buffer():
    """Fixture pour créer un buffer de replay priorisé de test."""
    return PrioritizedReplayBuffer(capacity=1000)

def test_dqn_initialization(dqn_agent):
    """Test l'initialisation de l'agent DQN."""
    assert dqn_agent.state_size == 4
    assert dqn_agent.action_size == 2
    assert dqn_agent.hidden_size == 64
    assert dqn_agent.learning_rate == 0.001
    assert dqn_agent.epsilon == 1.0
    assert len(dqn_agent.memory) == 0
    assert dqn_agent.steps_done == 0

def test_action_selection(dqn_agent):
    """Test la sélection d'action."""
    state = np.random.randn(4)
    
    # Test en mode entraînement
    action = dqn_agent.select_action(state, training=True)
    assert action in [0, 1]
    assert dqn_agent.steps_done == 1
    
    # Test en mode évaluation
    action = dqn_agent.select_action(state, training=False)
    assert action in [0, 1]

def test_ucb_exploration(dqn_agent):
    """Test l'exploration UCB."""
    state = np.random.randn(4)
    
    # Test initial (premières actions)
    for i in range(dqn_agent.action_size):
        action = dqn_agent.select_action(state)
        assert action == i
        assert dqn_agent.action_counts[action] == 1
    
    # Test après quelques actions
    for _ in range(10):
        action = dqn_agent.select_action(state)
        dqn_agent.update_ucb(action, 1.0)
    
    assert np.all(dqn_agent.action_counts > 0)
    assert np.all(dqn_agent.action_values > 0)

def test_prioritized_replay(replay_buffer):
    """Test le buffer de replay priorisé."""
    # Ajout de transitions
    for _ in range(5):
        state = torch.randn(4)
        action = torch.tensor([0])
        next_state = torch.randn(4)
        reward = torch.tensor([1.0])
        done = torch.tensor([False])
        replay_buffer.push(state, action, next_state, reward, done)
    
    assert len(replay_buffer.memory) == 5
    assert len(replay_buffer.priorities) == 5
    
    # Test de l'échantillonnage
    batch, indices, weights = replay_buffer.sample(3)
    assert len(batch.state) == 3
    assert len(indices) == 3
    assert len(weights) == 3
    
    # Test de la mise à jour des priorités
    new_priorities = [0.5, 0.7, 0.3]
    replay_buffer.update_priorities(indices, new_priorities)
    
    # Vérification des priorités mises à jour
    for i, idx in enumerate(indices):
        assert abs(replay_buffer.priorities[idx] - new_priorities[i]) < 1e-6, f"Priorité incorrecte à l'index {i}: attendu {new_priorities[i]}, obtenu {replay_buffer.priorities[idx]}"

def test_model_optimization(dqn_agent):
    """Test l'optimisation du modèle."""
    device = dqn_agent.device
    # Remplissage du buffer
    for _ in range(dqn_agent.batch_size):
        state = torch.randn(1, 4, device=device)
        action = torch.tensor([[0]], device=device)
        next_state = torch.randn(1, 4, device=device)
        reward = torch.tensor([[1.0]], device=device)
        done = torch.tensor([[False]], device=device)
        dqn_agent.memory.push(state, action, next_state, reward, done)
    
    # Test de l'optimisation
    loss = dqn_agent.optimize_model()
    assert loss is not None
    assert isinstance(loss, float)
    assert loss >= 0

def test_model_save_load(dqn_agent, tmp_path):
    """Test la sauvegarde et le chargement du modèle."""
    # Sauvegarde
    save_path = tmp_path / "dqn_model.pt"
    dqn_agent.save(str(save_path))
    assert save_path.exists()
    
    # Chargement
    new_agent = DQNAgent(
        state_size=4,
        action_size=2,
        hidden_size=64
    )
    new_agent.load(str(save_path))
    
    # Vérification des paramètres
    assert new_agent.steps_done == dqn_agent.steps_done
    assert new_agent.epsilon == dqn_agent.epsilon
    assert np.array_equal(new_agent.action_counts, dqn_agent.action_counts)
    assert np.array_equal(new_agent.action_values, dqn_agent.action_values) 