import os
import sys
import tempfile
import unittest

import numpy as np

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.dqn_agent import DQNAgent


class TestDQNAgent(unittest.TestCase):
    """Tests pour l'agent DQN."""

    def setUp(self):
        """Prépare l'agent pour les tests."""
        # Paramètres de l'agent
        self.state_size = 20
        self.action_size = 3
        self.batch_size = 32

        # Créer l'agent
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=self.batch_size,
            memory_size=1000,
        )

    def test_build_model(self):
        """Teste la construction du modèle."""
        # Vérifier que les modèles sont créés
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.target_model)

        # Vérifier que les modèles ont la bonne architecture
        # Vérifier la première couche (input)
        self.assertEqual(self.agent.model.fc1.in_features, self.state_size)
        # Vérifier la dernière couche (output)
        self.assertEqual(self.agent.model.fc4.out_features, self.action_size)

    def test_act(self):
        """Teste la méthode d'action."""
        # Créer un état de test
        state = np.random.random((1, self.state_size))

        # Tester avec epsilon = 1 (exploration)
        self.agent.epsilon = 1.0
        actions = [self.agent.act(state) for _ in range(100)]

        # Vérifier que toutes les actions sont possibles
        self.assertTrue(all(0 <= a < self.action_size for a in actions))

        # Vérifier qu'il y a de l'exploration (actions aléatoires)
        self.assertGreater(len(set(actions)), 1)

        # Tester avec epsilon = 0 (exploitation)
        self.agent.epsilon = 0.0
        action = self.agent.act(state)

        # Vérifier que l'action est valide
        self.assertTrue(0 <= action < self.action_size)

        # Vérifier que l'action est déterministe
        self.assertEqual(action, self.agent.act(state))

    def test_remember(self):
        """Teste la méthode de mémorisation."""
        # Créer une expérience de test
        state = np.random.random((1, self.state_size))
        action = 1
        reward = 0.5
        next_state = np.random.random((1, self.state_size))
        done = False

        # Mémoriser l'expérience
        initial_memory_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)

        # Vérifier que la mémoire a augmenté
        self.assertEqual(len(self.agent.memory), initial_memory_len + 1)

    def test_replay(self):
        """Teste la méthode de replay."""
        # Remplir la mémoire avec des expériences aléatoires
        for _ in range(self.batch_size * 2):
            state = np.random.random((1, self.state_size))
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            next_state = np.random.random((1, self.state_size))
            done = np.random.choice([True, False])

            self.agent.remember(state, action, reward, next_state, done)

        # Effectuer un replay
        loss = self.agent.replay()

        # Vérifier que la perte est un nombre
        self.assertIsInstance(loss, float)

    def test_update_target_model(self):
        """Teste la mise à jour du modèle cible."""
        # Modifier les poids du modèle principal
        weights = self.agent.model.get_weights()
        modified_weights = [w + 0.1 for w in weights]
        self.agent.model.set_weights(modified_weights)

        # Vérifier que les modèles sont différents
        model_weights = self.agent.model.get_weights()
        target_weights = self.agent.target_model.get_weights()

        # Au moins un poids devrait être différent
        self.assertTrue(
            any(
                not np.array_equal(w1, w2)
                for w1, w2 in zip(model_weights, target_weights)
            )
        )

        # Mettre à jour le modèle cible
        self.agent.update_target_model()

        # Vérifier que les modèles sont maintenant identiques
        model_weights = self.agent.model.get_weights()
        target_weights = self.agent.target_model.get_weights()

        for w1, w2 in zip(model_weights, target_weights):
            np.testing.assert_array_equal(w1, w2)

    def test_save_load(self):
        """Teste les méthodes de sauvegarde et de chargement."""
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            model_path = temp_file.name

        try:
            # Sauvegarder le modèle
            self.agent.save(model_path)

            # Vérifier que le fichier existe
            self.assertTrue(os.path.exists(model_path))

            # Modifier les poids du modèle
            weights = self.agent.model.get_weights()
            modified_weights = [w + 0.1 for w in weights]
            self.agent.model.set_weights(modified_weights)

            # Charger le modèle
            self.agent.load(model_path)

            # Vérifier que les poids sont restaurés
            loaded_weights = self.agent.model.get_weights()

            for w1, w2 in zip(weights, loaded_weights):
                np.testing.assert_array_almost_equal(w1, w2)

        finally:
            # Nettoyer
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_epsilon_decay(self):
        """Teste la décroissance d'epsilon."""
        initial_epsilon = self.agent.epsilon

        # Appliquer la décroissance
        self.agent.decay_epsilon()

        # Vérifier que epsilon a diminué
        self.assertLess(self.agent.epsilon, initial_epsilon)

        # Vérifier que epsilon ne descend pas en dessous du minimum
        for _ in range(1000):  # Beaucoup d'itérations pour atteindre epsilon_min
            self.agent.decay_epsilon()

        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)


if __name__ == "__main__":
    unittest.main()
