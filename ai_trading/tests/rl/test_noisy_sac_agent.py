#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests pour la classe NoisySACAgent."""

import logging
import os
import tempfile
import unittest
import warnings

import numpy as np
import torch
import pytest
from ai_trading.rl.agents.sac_agent import OptimizedSACAgent

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mise à jour des filtres d'avertissement pour utiliser des approches plus modernes
# Ces filtres sont plus précis et évitent de masquer tous les avertissements TF
warnings.filterwarnings("ignore", message=".*jax.xla_computation is deprecated.*")
warnings.filterwarnings("ignore", message=".*tensorflow.*deprecated.*")
warnings.filterwarnings("ignore", message=".*tensorflow.*removed in a future version.*")
# Ignorer l'avertissement concernant distutils.version.LooseVersion dans tensorflow_probability
warnings.filterwarnings(
    "ignore", message=".*distutils Version classes are deprecated.*"
)
warnings.filterwarnings("ignore", message=".*'imghdr' is deprecated.*")


@pytest.fixture
def sac_agent():
    state_dim = 10
    action_dim = 2
    sequence_length = 5
    return OptimizedSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        sequence_length=sequence_length
    )

# class TestNoisySACAgent(unittest.TestCase):
#     ... (tout le contenu de la classe est commenté)

def test_noise_injection(sac_agent):
    # Test avec exploration (stochastic=True)
    state = np.random.randn(5, 10)  # (seq_len, state_dim)
    action = sac_agent.select_action(state, deterministic=False)
    assert action.shape == (2,)
    
    # Vérifier que l'action est dans les limites [-1, 1]
    assert np.all(action >= -1) and np.all(action <= 1)

def test_noise_scale(sac_agent):
    # Vérifier que le bruit est correctement appliqué
    state = np.random.randn(5, 10)
    
    # Obtenir plusieurs actions pour le même état
    actions = [sac_agent.select_action(state, deterministic=False) for _ in range(10)]
    actions = np.array(actions)
    
    # Vérifier que les actions sont différentes (à cause du bruit)
    assert not np.allclose(actions[0], actions[1])
    
    # Vérifier que la variance est raisonnable
    action_std = np.std(actions, axis=0)
    assert np.all(action_std > 0)  # Devrait y avoir de la variance
    assert np.all(action_std < 0.5)  # Mais pas trop de variance

def test_deterministic_actions(sac_agent):
    # Test sans exploration (stochastic=False)
    state = np.random.randn(5, 10)
    action1 = sac_agent.select_action(state, deterministic=True)
    action2 = sac_agent.select_action(state, deterministic=True)
    
    # Les actions devraient être identiques en mode déterministe
    assert np.allclose(action1, action2)

def test_noise_training(sac_agent):
    # Test de l'entraînement avec le bruit
    state_seq = np.random.randn(5, 10)
    action = np.random.randn(2)
    reward = 1.0
    next_state_seq = np.random.randn(5, 10)
    done = False

    # Ajouter plusieurs expériences
    for _ in range(10):
        sac_agent.remember(state_seq, action, reward, next_state_seq, done)

    # Entraîner l'agent
    metrics = sac_agent.train()
    
    assert "critic1_loss" in metrics
    assert "critic2_loss" in metrics
    assert "actor_loss" in metrics

def test_noise_consistency(sac_agent):
    # Vérifier que le bruit est cohérent entre les appels
    state = np.random.randn(5, 10)
    
    # Obtenir des actions en mode stochastique
    actions_stochastic = [sac_agent.select_action(state, deterministic=False) for _ in range(5)]
    
    # Obtenir des actions en mode déterministe
    actions_deterministic = [sac_agent.select_action(state, deterministic=True) for _ in range(5)]
    
    # Vérifier que les actions stochastiques sont différentes
    assert not np.allclose(actions_stochastic[0], actions_stochastic[1])
    
    # Vérifier que les actions déterministes sont identiques
    assert np.allclose(actions_deterministic[0], actions_deterministic[1])


if __name__ == "__main__":
    unittest.main()
