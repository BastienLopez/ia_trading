# Avertissements de dépréciation connus :
# 1. PyTorch: torch.utils._pytree._register_pytree_node est déprécié
#    Solution: Utiliser torch.utils._pytree.register_pytree_node à la place
# 2. TensorFlow Probability: distutils.version.LooseVersion est déprécié
#    Solution: Utiliser packaging.version à la place
# Ces avertissements n'affectent pas le fonctionnement du code mais seront corrigés dans les futures versions

import logging
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from gymnasium import spaces

# Filtres pour ignorer les avertissements de dépréciation connus
warnings.filterwarnings(
    "ignore", message=".*distutils Version classes are deprecated.*"
)
warnings.filterwarnings("ignore", message=".*'imghdr' is deprecated.*")
warnings.filterwarnings("ignore", message=".*tensorflow.*deprecated.*")

# Configurer le logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.agents.sac_agent import TransformerSACAgent
from ai_trading.rl.trading_environment import TradingEnvironment


class SimpleEnv(gym.Env):
    """Environnement simplifié pour tester l'agent."""

    def __init__(self, state_dim=10, action_dim=2, sequence_length=5):
        super(SimpleEnv, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.current_step = 0
        self.max_steps = 100

        # Définir les espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Générer un état aléatoire
        state = np.random.random(self.state_dim).astype(np.float32)
        return state, {}

    def step(self, action):
        # Incrémenter le compteur d'étapes
        self.current_step += 1

        # Générer un nouvel état
        next_state = np.random.random(self.state_dim).astype(np.float32)

        # Calculer une récompense simple basée sur l'action
        reward = float(np.sum(action)) / self.action_dim

        # Vérifier si l'épisode est terminé
        done = self.current_step >= self.max_steps

        return next_state, reward, done, False, {}


class TestSACAgent(unittest.TestCase):
    """Tests pour l'agent SAC avec architecture Transformer."""

    def setUp(self):
        """Configuration pour chaque test."""
        # Paramètres pour l'agent et l'environnement
        self.state_dim = 10
        self.action_dim = 2
        self.sequence_length = 5

        # Créer l'environnement de test
        self.env = SimpleEnv(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            sequence_length=self.sequence_length,
        )

        # Créer l'agent
        self.agent = TransformerSACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            d_model=32,
            n_heads=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.1,
            activation="gelu",
            max_seq_len=20,
            sequence_length=self.sequence_length,
            hidden_dim=32,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            buffer_size=100,
            batch_size=4,
            device="cpu",
            action_bounds=(-1.0, 1.0),
        )

        # Créer un environnement de trading pour les tests d'intégration
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range(start="2023-01-01", periods=n_samples)
        prices = np.linspace(100, 200, n_samples) + np.random.normal(0, 5, n_samples)
        volumes = np.random.normal(1000, 200, n_samples)

        self.df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices - np.random.uniform(0, 2, n_samples),
                "high": prices + np.random.uniform(0, 2, n_samples),
                "low": prices - np.random.uniform(0, 2, n_samples),
                "close": prices,
                "volume": volumes,
                "market_cap": prices * volumes,
            }
        )

        self.trading_env = TradingEnvironment(
            df=self.df,
            initial_balance=10000.0,
            transaction_fee=0.001,
            window_size=10,
            action_type="continuous",
        )

    def test_initialization(self):
        """Teste l'initialisation de l'agent."""
        # Vérifier que les attributs principaux sont correctement initialisés
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.sequence_length, self.sequence_length)

        # Vérifier que les réseaux sont initialisés
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic1)
        self.assertIsNotNone(self.agent.critic2)
        self.assertIsNotNone(self.agent.target_critic1)
        self.assertIsNotNone(self.agent.target_critic2)

    def test_action_selection(self):
        """Teste la sélection d'actions."""
        state = np.random.random(self.state_dim).astype(np.float32)

        # Test mode stochastique
        action = self.agent.select_action(state)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

        # Test mode déterministe
        action_det = self.agent.select_action(state, deterministic=True)
        self.assertIsInstance(action_det, np.ndarray)
        self.assertEqual(action_det.shape, (self.action_dim,))
        self.assertTrue(np.all(action_det >= -1.0) and np.all(action_det <= 1.0))

    def test_training(self):
        """Teste l'entraînement de l'agent."""
        # Ajouter quelques expériences au buffer
        for _ in range(10):
            state = np.random.random(self.state_dim).astype(np.float32)
            action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
            reward = np.random.random()
            next_state = np.random.random(self.state_dim).astype(np.float32)
            done = False
            self.agent.replay_buffer.add(state, action, reward, next_state, done)

        # Entraîner l'agent
        metrics = self.agent.train()

        # Vérifier les métriques
        self.assertIn("critic1_loss", metrics)
        self.assertIn("critic2_loss", metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("alpha_loss", metrics)
        self.assertIn("alpha", metrics)

    def test_save_load(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Sauvegarder le modèle
            save_path = os.path.join(temp_dir, "model.pt")
            self.agent.save(save_path)

            # Vérifier que le fichier existe
            self.assertTrue(os.path.exists(save_path))

            # Créer un nouvel agent
            new_agent = TransformerSACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                d_model=32,
                n_heads=4,
                num_layers=2,
                dim_feedforward=64,
                dropout=0.1,
                activation="gelu",
                max_seq_len=20,
                sequence_length=self.sequence_length,
                hidden_dim=32,
                learning_rate=3e-4,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                buffer_size=100,
                batch_size=4,
                device="cpu",
                action_bounds=(-1.0, 1.0),
            )

            # Charger le modèle
            new_agent.load(save_path)

            # Vérifier que les poids sont identiques
            for p1, p2 in zip(self.agent.actor.parameters(), new_agent.actor.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_trading_integration(self):
        """Teste l'intégration avec l'environnement de trading."""
        # Réinitialiser l'environnement
        state, _ = self.trading_env.reset()

        # Exécuter quelques étapes
        for _ in range(10):
            # Sélectionner une action
            action = self.agent.select_action(state)

            # Exécuter l'action
            next_state, reward, terminated, truncated, _ = self.trading_env.step(action)

            # Mémoriser l'expérience
            self.agent.replay_buffer.add(state, action, reward, next_state, terminated)

            # Mettre à jour l'état
            state = next_state

            # Entraîner l'agent si suffisamment d'expériences
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                metrics = self.agent.train()
                self.assertIsInstance(metrics, dict)

    def test_transformer_architecture(self):
        """Teste spécifiquement l'architecture Transformer."""
        # Créer une séquence d'états
        batch_size = 4
        states = np.random.random((batch_size, self.sequence_length, self.state_dim)).astype(np.float32)
        states_tensor = torch.FloatTensor(states)

        # Tester le forward pass de l'acteur
        mean, log_std = self.agent.actor(states_tensor)
        self.assertEqual(mean.shape, (batch_size, self.action_dim))
        self.assertEqual(log_std.shape, (batch_size, self.action_dim))

        # Tester le forward pass des critiques
        actions = np.random.uniform(-1.0, 1.0, size=(batch_size, self.action_dim)).astype(np.float32)
        actions_tensor = torch.FloatTensor(actions)

        q1 = self.agent.critic1(states_tensor, actions_tensor)
        q2 = self.agent.critic2(states_tensor, actions_tensor)

        self.assertEqual(q1.shape, (batch_size, 1))
        self.assertEqual(q2.shape, (batch_size, 1))


if __name__ == "__main__":
    unittest.main()
