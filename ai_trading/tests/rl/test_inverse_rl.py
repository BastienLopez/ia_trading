"""
Tests unitaires pour le module d'apprentissage inverse par renforcement (inverse_rl.py).

Ces tests valident:
- La construction correcte des réseaux de récompense
- Le chargement des démonstrations d'experts
- Les calculs de fréquence de visite d'états
- Les prédictions de récompense
- La fonctionnalité de sauvegarde et chargement des modèles
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

# Ajout du répertoire parent au path pour les imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ai_trading.rl.inverse_rl import (
    ApprenticeshipLearning,
    MaximumEntropyIRL,
    RewardNetwork,
)


class TestRewardNetwork(unittest.TestCase):
    """Tests pour la classe RewardNetwork."""

    def setUp(self):
        """Initialisation pour chaque test."""
        self.state_dim = 10
        self.action_dim = 3
        self.batch_size = 5
        self.network = RewardNetwork(self.state_dim, self.action_dim)

    def test_initialization(self):
        """Teste l'initialisation correcte du réseau."""
        # Vérifier les attributs
        self.assertEqual(self.network.state_dim, self.state_dim)
        self.assertEqual(self.network.action_dim, self.action_dim)

        # Vérifier la structure du modèle
        self.assertIsNotNone(self.network.model)

        # Vérifier que le premier module est une couche linéaire avec la bonne dimension
        first_layer = list(self.network.model.children())[0]
        self.assertIsInstance(first_layer, torch.nn.Linear)
        self.assertEqual(first_layer.in_features, self.state_dim + self.action_dim)

    def test_forward_discrete_action(self):
        """Teste le passage avant avec des actions discrètes."""
        # Créer des données d'entrée
        states = torch.randn(self.batch_size, self.state_dim)
        actions = torch.randint(0, self.action_dim, (self.batch_size,))

        # Passer à travers le réseau
        rewards = self.network(states, actions)

        # Vérifier les dimensions de sortie
        self.assertEqual(rewards.shape, (self.batch_size, 1))

    def test_forward_continuous_action(self):
        """Teste le passage avant avec des actions continues."""
        # Créer des données d'entrée
        states = torch.randn(self.batch_size, self.state_dim)
        actions = torch.randn(self.batch_size, self.action_dim)

        # Passer à travers le réseau
        rewards = self.network(states, actions)

        # Vérifier les dimensions de sortie
        self.assertEqual(rewards.shape, (self.batch_size, 1))


class TestMaximumEntropyIRL(unittest.TestCase):
    """Tests pour la classe MaximumEntropyIRL."""

    def setUp(self):
        """Initialisation pour chaque test."""
        self.state_dim = 5
        self.action_dim = 3
        self.maxent_irl = MaximumEntropyIRL(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=0.99,
            learning_rate=0.001,
            regularization=0.01,
            device="cpu",
        )

        # Créer des démonstrations d'experts simulées
        self.demos = []
        for _ in range(3):  # 3 trajectoires
            trajectory = []
            for _ in range(10):  # 10 transitions par trajectoire
                state = np.random.rand(self.state_dim)
                action = np.random.randint(0, self.action_dim)
                trajectory.append((state, action))
            self.demos.append(trajectory)

    def test_initialization(self):
        """Teste l'initialisation correcte de l'algorithme."""
        self.assertEqual(self.maxent_irl.state_dim, self.state_dim)
        self.assertEqual(self.maxent_irl.action_dim, self.action_dim)
        self.assertEqual(self.maxent_irl.gamma, 0.99)
        self.assertEqual(self.maxent_irl.learning_rate, 0.001)
        self.assertEqual(self.maxent_irl.regularization, 0.01)
        self.assertEqual(self.maxent_irl.device, "cpu")

        # Vérifier l'initialisation du réseau de récompense
        self.assertIsInstance(self.maxent_irl.reward_network, RewardNetwork)

        # Vérifier l'initialisation de l'optimiseur
        self.assertIsNotNone(self.maxent_irl.optimizer)

    def test_load_expert_demonstrations(self):
        """Teste le chargement des démonstrations d'experts."""
        self.maxent_irl.load_expert_demonstrations(self.demos)

        # Vérifier que les démos sont correctement stockées
        self.assertEqual(self.maxent_irl.expert_demos, self.demos)

        # Vérifier que les états et actions sont convertis en tensors
        self.assertIsInstance(self.maxent_irl.expert_states, torch.Tensor)
        self.assertIsInstance(self.maxent_irl.expert_actions, torch.Tensor)

        # Vérifier les dimensions
        total_transitions = sum(len(trajectory) for trajectory in self.demos)
        self.assertEqual(self.maxent_irl.expert_states.shape[0], total_transitions)
        self.assertEqual(self.maxent_irl.expert_states.shape[1], self.state_dim)
        self.assertEqual(self.maxent_irl.expert_actions.shape[0], total_transitions)

    def test_compute_state_visitation_freq(self):
        """Teste le calcul des fréquences de visite d'états."""
        # Créer des trajectoires simples avec des états répétés
        trajectories = [
            [(np.array([1, 2, 3, 4, 5]), 0), (np.array([1, 2, 3, 4, 5]), 1)],
            [(np.array([5, 4, 3, 2, 1]), 0), (np.array([1, 2, 3, 4, 5]), 2)],
        ]

        freq = self.maxent_irl.compute_state_visitation_freq(trajectories)

        # Vérifier que c'est un dictionnaire
        self.assertIsInstance(freq, dict)

        # Vérifier que la somme des fréquences est 1
        self.assertAlmostEqual(sum(freq.values()), 1.0)

        # Vérifier que l'état répété a une fréquence plus élevée
        state1 = tuple([1, 2, 3, 4, 5])
        state2 = tuple([5, 4, 3, 2, 1])
        self.assertGreater(freq[state1], freq[state2])

    def test_predict_reward(self):
        """Teste la prédiction de récompense."""
        # Créer un état et une action
        state = torch.randn(1, self.state_dim)
        action = torch.tensor([1])

        # Prédire la récompense
        reward = self.maxent_irl.predict_reward(state, action)

        # Vérifier le type et la dimension
        self.assertIsInstance(reward, torch.Tensor)
        self.assertEqual(reward.shape, (1, 1))

    def test_save_load(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Créer un dossier temporaire pour les tests
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "maxent_irl_model.pt")

            # Sauvegarder le modèle
            self.maxent_irl.save(save_path)

            # Vérifier que le fichier existe
            self.assertTrue(os.path.exists(save_path))

            # Créer une nouvelle instance
            new_irl = MaximumEntropyIRL(
                state_dim=10,  # Différent de l'original
                action_dim=5,  # Différent de l'original
                device="cpu",
            )

            # Charger le modèle
            new_irl.load(save_path)

            # Vérifier que les attributs sont correctement chargés
            self.assertEqual(new_irl.state_dim, self.state_dim)
            self.assertEqual(new_irl.action_dim, self.action_dim)
            self.assertEqual(new_irl.gamma, 0.99)
            self.assertEqual(new_irl.regularization, 0.01)


class TestApprenticeshipLearning(unittest.TestCase):
    """Tests pour la classe ApprenticeshipLearning."""

    def setUp(self):
        """Initialisation pour chaque test."""
        self.state_dim = 5
        self.action_dim = 3
        self.al = ApprenticeshipLearning(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=0.99,
            device="cpu",
        )

        # Créer des démonstrations d'experts simulées
        self.demos = []
        for _ in range(3):  # 3 trajectoires
            trajectory = []
            for _ in range(10):  # 10 transitions par trajectoire
                state = np.random.rand(self.state_dim)
                action = np.random.randint(0, self.action_dim)
                trajectory.append((state, action))
            self.demos.append(trajectory)

    def test_initialization(self):
        """Teste l'initialisation correcte de l'algorithme."""
        self.assertEqual(self.al.state_dim, self.state_dim)
        self.assertEqual(self.al.action_dim, self.action_dim)
        self.assertEqual(
            self.al.feature_dim, self.state_dim
        )  # Par défaut, feature_dim = state_dim
        self.assertEqual(self.al.gamma, 0.99)
        self.assertEqual(self.al.device, "cpu")

        # Vérifier les poids de récompense
        self.assertIsInstance(self.al.reward_weights, torch.Tensor)
        self.assertEqual(self.al.reward_weights.shape, (self.state_dim,))

    def test_extract_features_default(self):
        """Teste l'extraction de caractéristiques par défaut (identité)."""
        state = np.random.rand(self.state_dim)
        features = self.al.extract_features(state)

        # Vérifier que les caractéristiques sont identiques à l'état
        np.testing.assert_array_equal(features, state)

    def test_extract_features_custom(self):
        """Teste l'extraction de caractéristiques avec un extracteur personnalisé."""

        # Définir un extracteur de caractéristiques personnalisé
        def custom_extractor(state):
            return state * 2

        # Créer une instance avec l'extracteur personnalisé
        al_custom = ApprenticeshipLearning(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            feature_extractor=custom_extractor,
            feature_dim=self.state_dim,
        )

        state = np.random.rand(self.state_dim)
        features = al_custom.extract_features(state)

        # Vérifier que les caractéristiques sont transformées
        np.testing.assert_array_equal(features, state * 2)

    def test_compute_expected_feature_counts(self):
        """Teste le calcul des comptes de caractéristiques attendus."""
        # Créer des trajectoires simples
        trajectories = [
            [(np.ones(self.state_dim), 0), (np.ones(self.state_dim) * 2, 1)],
            [(np.ones(self.state_dim) * 3, 0), (np.ones(self.state_dim) * 4, 2)],
        ]

        feature_counts = self.al.compute_expected_feature_counts(trajectories)

        # Vérifier les dimensions
        self.assertEqual(feature_counts.shape, (self.state_dim,))

        # Vérifier que les valeurs sont raisonnables
        self.assertTrue(np.all(feature_counts > 0))

    def test_load_expert_demonstrations(self):
        """Teste le chargement des démonstrations d'experts."""
        self.al.load_expert_demonstrations(self.demos)

        # Vérifier que les démos sont correctement stockées
        self.assertEqual(self.al.expert_demos, self.demos)

        # Vérifier que les comptes de caractéristiques sont calculés
        self.assertIsInstance(self.al.expert_feature_counts, np.ndarray)
        self.assertEqual(self.al.expert_feature_counts.shape, (self.state_dim,))

    def test_compute_reward(self):
        """Teste le calcul de récompense."""
        # Définir des poids de récompense
        self.al.reward_weights = torch.ones(self.state_dim)

        # Calculer la récompense pour un état
        state = np.ones(self.state_dim)
        reward = self.al.compute_reward(state)

        # Vérifier la valeur de la récompense
        self.assertEqual(reward, self.state_dim)  # sum(1 * 1) pour chaque dimension

    def test_save_load(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Définir des poids non nuls
        self.al.reward_weights = torch.ones(self.state_dim) * 0.5

        # Créer un dossier temporaire pour les tests
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "al_model.pkl")

            # Sauvegarder le modèle
            self.al.save(save_path)

            # Vérifier que le fichier existe
            self.assertTrue(os.path.exists(save_path))

            # Créer une nouvelle instance
            new_al = ApprenticeshipLearning(
                state_dim=10,  # Différent de l'original
                action_dim=5,  # Différent de l'original
                device="cpu",
            )

            # Charger le modèle
            new_al.load(save_path)

            # Vérifier que les attributs sont correctement chargés
            self.assertEqual(new_al.state_dim, self.state_dim)
            self.assertEqual(new_al.action_dim, self.action_dim)
            self.assertEqual(new_al.feature_dim, self.state_dim)
            self.assertEqual(new_al.gamma, 0.99)

            # Vérifier que les poids sont correctement chargés
            self.assertIsInstance(new_al.reward_weights, torch.Tensor)
            self.assertEqual(new_al.reward_weights.shape, (self.state_dim,))
            self.assertTrue(torch.all(new_al.reward_weights == 0.5))


if __name__ == "__main__":
    unittest.main()
