import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
import pytest
import torch

# Configurer le logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.agents.sac_agent import OptimizedSACAgent
from ai_trading.rl.entropy_regularization import AdaptiveEntropyRegularization
from ai_trading.rl.trading_environment import TradingEnvironment


class TestEntropyRegularization(unittest.TestCase):
    """Tests pour le mécanisme de régularisation d'entropie adaptative."""

    def _generate_test_data(self):
        """Génère des données de test pour les tests d'entropie."""
        # Générer des données de marché aléatoires
        n_samples = 1000
        dates = pd.date_range(start="2023-01-01", periods=n_samples)
        prices = np.linspace(100, 200, n_samples) + np.random.normal(0, 5, n_samples)
        volumes = np.random.normal(1000, 200, n_samples)

        # Créer un DataFrame avec les données
        df = pd.DataFrame(
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

        return df

    def setUp(self):
        """Initialisation avant chaque test."""
        # Créer un environnement de trading simple
        self.df = self._generate_test_data()

        self.env = TradingEnvironment(
            df=self.df,
            initial_balance=10000.0,
            window_size=10,
            transaction_fee=0.001,
            action_type="continuous",
        )

        # Configurer l'état pour les tests
        state, _ = self.env.reset()
        self.state_size = state.shape[0]
        self.action_size = 1  # Action continue

        # Paramètres de test pour la régularisation d'entropie
        self.initial_alpha = 0.1
        self.update_interval = 5
        self.reward_scaling = 5.0
        self.target_entropy_ratio = 0.5

    def _calculate_action_entropy(self, agent, states, n_samples=50):
        """Calcule l'entropie moyenne des actions de l'agent sur un ensemble d'états."""
        total_std = 0.0
        n_states = min(len(states), 100)  # Limiter le nombre d'états à évaluer

        # Sélectionner un sous-ensemble d'états pour l'évaluation
        sample_indices = np.random.choice(len(states), size=n_states, replace=False)
        sampled_states = states[sample_indices]

        for state in sampled_states:
            # Obtenir plusieurs échantillons d'actions pour le même état
            actions = []
            for _ in range(n_samples):
                # S'assurer que l'état est correctement formaté
                if not isinstance(state, np.ndarray):
                    state = np.array(state, dtype=np.float32)

                # Utiliser evaluate=False pour avoir de l'exploration
                action = agent.select_action(state, deterministic=False)
                actions.append(action)

            # Calculer l'écart-type des actions pour cet état
            actions = np.array(actions)
            std = np.std(actions, axis=0).mean()
            total_std += std

        # Retourner l'écart-type moyen
        return total_std / n_states

    def test_initialization(self):
        """Teste l'initialisation de la régularisation d'entropie adaptative."""
        # Créer un objet de régularisation d'entropie
        entropy_reg = AdaptiveEntropyRegularization(
            action_size=self.action_size,
            initial_alpha=self.initial_alpha,
            update_interval=self.update_interval,
            reward_scaling=self.reward_scaling,
            target_entropy_ratio=self.target_entropy_ratio,
        )

        # Vérifier les attributs
        self.assertEqual(entropy_reg.action_size, self.action_size)
        self.assertEqual(entropy_reg.initial_alpha, self.initial_alpha)
        self.assertEqual(entropy_reg.update_interval, self.update_interval)
        self.assertEqual(entropy_reg.reward_scaling, self.reward_scaling)
        self.assertEqual(entropy_reg.target_entropy_ratio, self.target_entropy_ratio)

        # Vérifier que les attributs calculés sont corrects
        self.assertEqual(
            entropy_reg.target_entropy, -self.action_size * self.target_entropy_ratio
        )
        self.assertAlmostEqual(
            entropy_reg.log_alpha.numpy(), np.log(self.initial_alpha), places=5
        )
        self.assertEqual(entropy_reg.steps_counter, 0)

    def test_get_alpha(self):
        """Teste la récupération de la valeur d'alpha."""
        entropy_reg = AdaptiveEntropyRegularization(
            action_size=self.action_size, initial_alpha=self.initial_alpha
        )

        # Vérifier que la méthode get_alpha retourne la bonne valeur
        alpha = entropy_reg.get_alpha()
        self.assertAlmostEqual(alpha.numpy(), self.initial_alpha, places=5)

        # Modifier log_alpha et vérifier que get_alpha retourne la nouvelle valeur
        new_log_alpha = tf.Variable(np.log(0.5), dtype=tf.float32)
        entropy_reg.log_alpha = new_log_alpha

        alpha = entropy_reg.get_alpha()
        self.assertAlmostEqual(alpha.numpy(), 0.5, places=5)

    def test_entropy_regularization(self):
        """Vérifie que l'agent avec régularisation d'entropie a une entropie d'action plus élevée."""
        # Créer deux agents: un avec régularisation d'entropie, un sans
        agent_with_entropy = OptimizedSACAgent(
            state_dim=self.state_size,
            action_dim=self.action_size,
            entropy_regularization=2.0,  # Valeur élevée pour maximiser l'entropie
        )

        agent_without_entropy = OptimizedSACAgent(
            state_dim=self.state_size,
            action_dim=self.action_size,
            entropy_regularization=0.0,  # Pas de régularisation d'entropie
        )

        # Collecter des expériences dans l'environnement
        state, _ = self.env.reset()
        for _ in range(100):
            action_with = agent_with_entropy.select_action(state)
            next_state, reward, done, _, info = self.env.step(action_with)
            agent_with_entropy.remember(state, action_with, reward, next_state, done)

            action_without = agent_without_entropy.select_action(state)
            next_state, reward, done, _, info = self.env.step(action_without)
            agent_without_entropy.remember(
                state, action_without, reward, next_state, done
            )

            if done:
                state, _ = self.env.reset()
            else:
                state = next_state

        # Entraîner les agents pendant plusieurs épisodes
        for _ in range(50):
            if len(agent_with_entropy.replay_buffer) >= agent_with_entropy.batch_size:
                agent_with_entropy.train()

            if (
                len(agent_without_entropy.replay_buffer)
                >= agent_without_entropy.batch_size
            ):
                agent_without_entropy.train()

        # Générer des états pour tester l'entropie
        test_states = []
        state, _ = self.env.reset()
        for _ in range(100):
            test_states.append(state)
            action = np.random.uniform(-1, 1, (1,))
            next_state, _, done, _, _ = self.env.step(action)
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state

        test_states = np.array(test_states)

        # Calculer l'entropie des actions pour chaque agent
        entropy_with = self._calculate_action_entropy(agent_with_entropy, test_states)
        entropy_without = self._calculate_action_entropy(
            agent_without_entropy, test_states
        )

        print(f"Entropie avec régularisation: {entropy_with}")
        print(f"Entropie sans régularisation: {entropy_without}")

        # Vérifier que l'entropie est plus élevée avec la régularisation
        # Dans SAC, la régularisation d'entropie encourage l'exploration
        # en maximisant l'entropie de la politique
        self.assertGreater(
            entropy_with,
            entropy_without * 1.05,
            "L'entropie avec régularisation devrait être significativement plus élevée",
        )


@pytest.fixture
def sac_agent():
    state_dim = 10
    action_dim = 2
    return OptimizedSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        entropy_regularization=0.2
    )

def test_entropy_regularization_initialization(sac_agent):
    assert sac_agent.entropy_regularization == 0.2
    assert isinstance(sac_agent.alpha, float)
    assert sac_agent.alpha > 0

def test_entropy_regularization_training(sac_agent):
    # Créer des données de test
    state = np.random.randn(10)
    action = np.random.randn(2)
    reward = 1.0
    next_state = np.random.randn(10)
    done = False

    # Ajouter l'expérience au buffer
    sac_agent.remember(state, action, reward, next_state, done)

    # Entraîner l'agent
    metrics = sac_agent.train()

    # Vérifier que l'entropie est correctement régularisée
    assert "alpha" in metrics
    assert metrics["alpha"] > 0
    assert "actor_loss" in metrics
    assert "alpha_loss" in metrics

def test_entropy_regularization_action_selection(sac_agent):
    state = np.random.randn(10)
    
    # Test action déterministe
    action_det = sac_agent.select_action(state, deterministic=True)
    assert action_det.shape == (2,)
    
    # Test action stochastique
    action_stoch = sac_agent.select_action(state, deterministic=False)
    assert action_stoch.shape == (2,)
    
    # Les actions devraient être différentes
    assert not np.array_equal(action_det, action_stoch)

def test_entropy_regularization_alpha_update():
    # On force l'alpha automatique
    sac_agent = OptimizedSACAgent(state_dim=10, action_dim=2, entropy_regularization=0)
    initial_alpha = sac_agent.alpha

    # Simuler plusieurs étapes d'entraînement avec séquences
    for _ in range(10):
        state = np.random.randn(5, 10)  # (sequence_length, state_dim)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state = np.random.randn(5, 10)
        done = False

        sac_agent.remember(state, action, reward, next_state, done)
        metrics = sac_agent.train()

    # Vérifier que alpha a été mis à jour
    assert sac_agent.alpha != initial_alpha


if __name__ == "__main__":
    unittest.main()
