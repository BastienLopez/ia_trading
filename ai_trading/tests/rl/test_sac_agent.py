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
import pytest

# Filtres pour ignorer les avertissements de dépréciation connus
warnings.filterwarnings(
    "ignore", message=".*distutils Version classes are deprecated.*"
)
warnings.filterwarnings("ignore", message=".*'imghdr' is deprecated.*")
warnings.filterwarnings("ignore", message=".*tensorflow.*deprecated.*")

# Configurer le logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurer le niveau de log pour réduire les sorties de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.agents.sac_agent import TransformerSACAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data


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


@pytest.fixture
def sac_agent():
    state_dim = 10
    action_dim = 2
    sequence_length = 5
    return TransformerSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=64,  # Taille réduite pour les tests
        n_heads=2,   # Moins de têtes pour les tests
        num_layers=2, # Moins de couches pour les tests
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        max_seq_len=20,
        sequence_length=sequence_length,
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
    sac_agent.replay_buffer.add(state_seq, action, reward, next_state_seq, done)

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

def test_entropy_regularization(sac_agent):
    # Vérifier que la régularisation d'entropie fonctionne
    state = np.random.randn(5, 10)
    
    # Test action déterministe
    action_det = sac_agent.select_action(state, deterministic=True)
    assert action_det.shape == (2,)
    
    # Test action stochastique
    action_stoch = sac_agent.select_action(state, deterministic=False)
    assert action_stoch.shape == (2,)
    
    # Les actions devraient être différentes
    assert not np.array_equal(action_det, action_stoch)

def test_save_load(sac_agent, tmp_path):
    # Test sauvegarde et chargement du modèle
    save_path = os.path.join(tmp_path, "sac_agent.pt")
    
    # Sauvegarder le modèle
    sac_agent.save(save_path)
    assert os.path.exists(save_path)
    
    # Créer un nouvel agent
    new_agent = TransformerSACAgent(
        state_dim=10,
        action_dim=2,
        d_model=64,
        n_heads=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        max_seq_len=20,
        sequence_length=5,
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
    state = np.random.randn(5, 10)
    action1 = sac_agent.select_action(state, deterministic=True)
    action2 = new_agent.select_action(state, deterministic=True)
    assert np.allclose(action1, action2)

def test_replay_buffer(sac_agent):
    # Test du buffer de replay
    for _ in range(10):
        state = np.random.randn(5, 10)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state = np.random.randn(5, 10)
        done = False
        
        sac_agent.replay_buffer.add(state, action, reward, next_state, done)
    
    # Vérifier que l'entraînement fonctionne avec le buffer
    metrics = sac_agent.train()
    assert all(key in metrics for key in ["critic1_loss", "critic2_loss", "actor_loss"])


if __name__ == "__main__":
    unittest.main()
