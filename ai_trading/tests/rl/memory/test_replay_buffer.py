"""
Tests unitaires pour les différentes implémentations de tampons de replay.

Ce module contient les tests pour :
- ReplayBuffer : Tampon de replay standard
- PrioritizedReplayBuffer : Tampon de replay prioritaire
- NStepReplayBuffer : Tampon de replay avec retours multi-étapes
- DiskReplayBuffer : Tampon de replay sur disque
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_trading.rl.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    NStepReplayBuffer,
    DiskReplayBuffer,
)


class TestReplayBuffer(unittest.TestCase):
    """Tests pour le tampon de replay standard."""

    def setUp(self):
        """Initialise les variables de test."""
        self.buffer_size = 1000
        self.batch_size = 32
        self.state_dim = 10
        self.action_dim = 4
        self.buffer = ReplayBuffer(self.buffer_size)

    def test_add_and_sample(self):
        """Test l'ajout et l'échantillonnage de transitions."""
        # Créer des données de test
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False

        # Ajouter plusieurs transitions
        for _ in range(self.batch_size):
            self.buffer.add(state, action, reward, next_state, done)

        # Échantillonner un batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Vérifier les dimensions
        self.assertEqual(states.shape, (self.batch_size, self.state_dim))
        self.assertEqual(actions.shape, (self.batch_size, self.action_dim))
        self.assertEqual(rewards.shape, (self.batch_size, 1))
        self.assertEqual(next_states.shape, (self.batch_size, self.state_dim))
        self.assertEqual(dones.shape, (self.batch_size, 1))

    def test_buffer_size_limit(self):
        """Test que le tampon ne dépasse pas sa taille maximale."""
        # Ajouter plus d'éléments que la taille du tampon
        for _ in range(self.buffer_size + 100):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Vérifier que la taille est correcte
        self.assertEqual(len(self.buffer), self.buffer_size)

    def test_clear(self):
        """Test la méthode clear."""
        # Ajouter quelques éléments
        for _ in range(10):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Vider le tampon
        self.buffer.clear()

        # Vérifier que le tampon est vide
        self.assertEqual(len(self.buffer), 0)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Tests pour le tampon de replay prioritaire."""

    def setUp(self):
        """Initialise les variables de test."""
        self.buffer_size = 1000
        self.batch_size = 32
        self.state_dim = 10
        self.action_dim = 4
        self.buffer = PrioritizedReplayBuffer(
            self.buffer_size,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )

    def test_add_and_sample(self):
        """Test l'ajout et l'échantillonnage de transitions avec priorités."""
        # Créer des données de test
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False

        # Ajouter plusieurs transitions
        for _ in range(self.batch_size):
            self.buffer.add(state, action, reward, next_state, done)

        # Échantillonner un batch
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(
            self.batch_size
        )

        # Vérifier les dimensions
        self.assertEqual(states.shape, (self.batch_size, self.state_dim))
        self.assertEqual(actions.shape, (self.batch_size, self.action_dim))
        self.assertEqual(rewards.shape, (self.batch_size, 1))
        self.assertEqual(next_states.shape, (self.batch_size, self.state_dim))
        self.assertEqual(dones.shape, (self.batch_size, 1))
        self.assertEqual(indices.shape, (self.batch_size,))
        self.assertEqual(weights.shape, (self.batch_size,))

    def test_update_priorities(self):
        """Test la mise à jour des priorités."""
        # Ajouter quelques transitions
        for _ in range(self.batch_size):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Échantillonner un batch
        _, _, _, _, _, indices, _ = self.buffer.sample(self.batch_size)

        # Mettre à jour les priorités
        td_errors = np.random.randn(self.batch_size)
        self.buffer.update_priorities(indices, td_errors)

        # Vérifier que les priorités ont été mises à jour
        self.assertTrue(np.all(self.buffer.priorities[:len(self.buffer)] > 0))

    def test_beta_increment(self):
        """Test l'incrémentation de beta."""
        initial_beta = self.buffer.beta

        # Ajouter suffisamment de transitions
        for _ in range(self.batch_size):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Faire plusieurs échantillonnages
        for _ in range(10):
            self.buffer.sample(self.batch_size)

        # Vérifier que beta a augmenté
        self.assertGreater(self.buffer.beta, initial_beta)
        self.assertLessEqual(self.buffer.beta, 1.0)


class TestNStepReplayBuffer(unittest.TestCase):
    """Tests pour le tampon de replay avec retours multi-étapes."""

    def setUp(self):
        """Initialise les variables de test."""
        self.buffer_size = 1000
        self.batch_size = 32
        self.state_dim = 10
        self.action_dim = 4
        self.n_steps = 3
        self.gamma = 0.99
        self.buffer = NStepReplayBuffer(
            self.buffer_size,
            n_steps=self.n_steps,
            gamma=self.gamma,
        )

    def test_n_step_returns(self):
        """Test le calcul des retours sur n étapes."""
        # Créer une séquence de transitions
        states = [np.random.randn(self.state_dim) for _ in range(self.n_steps + 1)]
        actions = [np.random.randn(self.action_dim) for _ in range(self.n_steps)]
        rewards = [1.0 for _ in range(self.n_steps)]
        dones = [False for _ in range(self.n_steps)]

        # Ajouter les transitions
        for i in range(self.n_steps):
            self.buffer.add(
                states[i],
                actions[i],
                rewards[i],
                states[i + 1],
                dones[i],
            )

        # Vérifier que le tampon contient une transition
        self.assertEqual(len(self.buffer), 1)

        # Échantillonner et vérifier la récompense
        states, actions, rewards, next_states, dones = self.buffer.sample(1)
        expected_reward = sum(self.gamma**i * r for i, r in enumerate(rewards))
        self.assertAlmostEqual(rewards[0][0], expected_reward)

    def test_handle_episode_end(self):
        """Test la gestion de la fin d'épisode."""
        # Ajouter quelques transitions
        for _ in range(self.n_steps - 1):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Gérer la fin d'épisode
        self.buffer.handle_episode_end()

        # Vérifier que le tampon temporaire est vide
        self.assertEqual(len(self.buffer.n_step_buffer), 0)


class TestDiskReplayBuffer(unittest.TestCase):
    """Tests pour le tampon de replay sur disque."""

    def setUp(self):
        """Initialise les variables de test."""
        self.buffer_size = 1000
        self.batch_size = 32
        self.state_dim = 10
        self.action_dim = 4
        self.cache_size = 100

        # Créer un répertoire temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        self.buffer = DiskReplayBuffer(
            self.buffer_size,
            self.state_dim,
            self.action_dim,
            storage_path=self.test_dir,
            cache_size=self.cache_size,
        )

    def tearDown(self):
        """Nettoie les fichiers temporaires."""
        shutil.rmtree(self.test_dir)

    def test_add_and_sample(self):
        """Test l'ajout et l'échantillonnage de transitions."""
        # Créer des données de test
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False

        # Ajouter plusieurs transitions
        for _ in range(self.batch_size):
            self.buffer.add(state, action, reward, next_state, done)

        # Forcer l'écriture du cache
        self.buffer._flush_cache(force=True)

        # Échantillonner un batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Vérifier les dimensions
        self.assertEqual(states.shape, (self.batch_size, self.state_dim))
        self.assertEqual(actions.shape, (self.batch_size, self.action_dim))
        self.assertEqual(rewards.shape, (self.batch_size, 1))
        self.assertEqual(next_states.shape, (self.batch_size, self.state_dim))
        self.assertEqual(dones.shape, (self.batch_size, 1))

    def test_buffer_size_limit(self):
        """Test que le tampon ne dépasse pas sa taille maximale."""
        # Ajouter plus d'éléments que la taille du tampon
        for _ in range(self.buffer_size + 100):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Forcer l'écriture du cache
        self.buffer._flush_cache(force=True)

        # Vérifier que la taille est correcte
        self.assertEqual(len(self.buffer), self.buffer_size)

    def test_save_and_load(self):
        """Test la sauvegarde et le chargement des métadonnées."""
        # Ajouter quelques transitions
        for _ in range(10):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Sauvegarder les métadonnées
        metadata_path = os.path.join(self.test_dir, "metadata.pkl")
        self.buffer.save_metadata(metadata_path)

        # Charger un nouveau tampon
        new_buffer = DiskReplayBuffer.load(self.test_dir, metadata_path)

        # Vérifier que les paramètres sont identiques
        self.assertEqual(new_buffer.buffer_size, self.buffer.buffer_size)
        self.assertEqual(new_buffer.state_dim, self.buffer.state_dim)
        self.assertEqual(new_buffer.action_dim, self.buffer.action_dim)
        self.assertEqual(new_buffer.cache_size, self.buffer.cache_size)

    def test_performance_metrics(self):
        """Test la collecte des métriques de performance."""
        # Ajouter suffisamment de transitions
        for _ in range(self.batch_size):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            self.buffer.add(state, action, 1.0, state, False)

        # Forcer l'écriture du cache
        self.buffer._flush_cache(force=True)

        # Échantillonner quelques batches
        for _ in range(5):
            self.buffer.sample(self.batch_size)

        # Récupérer les métriques
        metrics = self.buffer.get_performance_metrics()

        # Vérifier que les métriques sont présentes
        self.assertIn("write_time", metrics)
        self.assertIn("read_time", metrics)
        self.assertIn("total_writes", metrics)
        self.assertIn("total_reads", metrics)
        self.assertIn("avg_write_time", metrics)
        self.assertIn("avg_read_time", metrics)


if __name__ == "__main__":
    unittest.main() 