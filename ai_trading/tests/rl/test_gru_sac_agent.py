import logging
import os
import sys
import unittest

import numpy as np
import tensorflow as tf
import torch
import pytest

# Configurer le niveau de log pour réduire les sorties de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.agents.sac_agent import OptimizedSACAgent
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestGRUSACAgent")

@pytest.fixture
def sac_agent():
    state_dim = 10
    action_dim = 2
    sequence_length = 5
    return OptimizedSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        sequence_length=sequence_length,
        d_model=64,  # Taille réduite pour les tests
        n_heads=2,   # Moins de têtes pour les tests
        num_layers=2 # Moins de couches pour les tests
    )

def test_sequence_handling(sac_agent):
    # Test avec une séquence d'états
    sequence = np.random.randn(5, 10)  # (seq_len, state_dim)
    action = sac_agent.select_action(sequence)
    assert action.shape == (2,)
    
    # Test avec un état unique (devrait être converti en séquence)
    single_state = np.random.randn(10)
    action = sac_agent.select_action(single_state)
    assert action.shape == (2,)

def test_training_with_sequences(sac_agent):
    # Créer des données de test avec séquences
    state_seq = np.random.randn(5, 10)  # (seq_len, state_dim)
    action = np.random.randn(2)
    reward = 1.0
    next_state_seq = np.random.randn(5, 10)
    done = False

    # Ajouter l'expérience au buffer
    sac_agent.remember(state_seq, action, reward, next_state_seq, done)

    # Entraîner l'agent
    metrics = sac_agent.train()
    
    assert "critic1_loss" in metrics
    assert "critic2_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha_loss" in metrics

def test_transformer_architecture(sac_agent):
    # Vérifier que l'architecture Transformer est correctement configurée
    assert hasattr(sac_agent.actor, 'transformer')
    assert hasattr(sac_agent.critic1, 'transformer')
    assert hasattr(sac_agent.critic2, 'transformer')
    
    # Vérifier les dimensions
    assert sac_agent.actor.transformer.layers[0].self_attn.num_heads == 2
    assert sac_agent.actor.transformer.layers[0].linear1.out_features == 256

def test_sequence_padding(sac_agent):
    # Test avec une séquence plus courte que sequence_length
    short_seq = np.random.randn(3, 10)  # (3, state_dim)
    action = sac_agent.select_action(short_seq)
    assert action.shape == (2,)
    
    # Test avec une séquence plus longue que sequence_length
    long_seq = np.random.randn(7, 10)  # (7, state_dim)
    action = sac_agent.select_action(long_seq)
    assert action.shape == (2,)

def test_memory_management(sac_agent):
    # Vérifier que le buffer gère correctement les séquences
    for _ in range(10):
        state_seq = np.random.randn(5, 10)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state_seq = np.random.randn(5, 10)
        done = False
        
        sac_agent.remember(state_seq, action, reward, next_state_seq, done)
    
    # Vérifier que l'entraînement fonctionne avec les séquences
    metrics = sac_agent.train()
    assert all(key in metrics for key in ["critic1_loss", "critic2_loss", "actor_loss"])

# class TestGRUSACAgent(unittest.TestCase):
#     ... (tout le contenu de la classe est commenté)

if __name__ == "__main__":
    unittest.main()
