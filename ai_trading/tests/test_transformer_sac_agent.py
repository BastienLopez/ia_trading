import os
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Ajouter le répertoire parent au chemin pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.transformer_sac_agent import TransformerSACAgent


# Environnement de test simplifié
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


class TestTransformerSACAgent(unittest.TestCase):
    """Tests pour l'agent TransformerSAC."""

    def setUp(self):
        """Initialise l'environnement et l'agent pour les tests."""
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
            action_bounds=(-1.0, 1.0),
            sequence_length=self.sequence_length,
            batch_size=4,  # Petite taille de batch pour les tests
            buffer_size=100,  # Petite taille de tampon pour les tests
            embed_dim=32,
            num_heads=4,
            ff_dim=128,
            num_transformer_blocks=2,
            rnn_units=32,
            model_type="gru",  # Utiliser GRU pour les tests
            device="cpu",  # Utiliser CPU pour les tests
        )

    def test_initialization(self):
        """Teste l'initialisation de l'agent."""
        # Vérifier que les attributs principaux sont correctement initialisés
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.sequence_length, self.sequence_length)

        # Vérifier que les réseaux sont initialisés
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic_1)
        self.assertIsNotNone(self.agent.critic_2)
        self.assertIsNotNone(self.agent.critic_1_target)
        self.assertIsNotNone(self.agent.critic_2_target)

        # Vérifier que les historiques de perte sont initialisés
        self.assertEqual(len(self.agent.actor_loss_history), 0)
        self.assertEqual(len(self.agent.critic_loss_history), 0)
        self.assertEqual(len(self.agent.alpha_loss_history), 0)

    def test_state_buffer(self):
        """Teste le tampon d'état de l'agent."""
        # Vérifier que le tampon est initialement vide
        self.assertEqual(len(self.agent.state_buffer), 0)

        # Ajouter un état
        state = np.random.random(self.state_dim)
        self.agent.update_state_buffer(state)

        # Vérifier que l'état a été ajouté
        self.assertEqual(len(self.agent.state_buffer), 1)
        np.testing.assert_array_equal(self.agent.state_buffer[0], state)

        # Ajouter plus d'états que la longueur de séquence
        for _ in range(self.sequence_length + 5):
            state = np.random.random(self.state_dim)
            self.agent.update_state_buffer(state)

        # Vérifier que le tampon a la bonne taille
        self.assertEqual(len(self.agent.state_buffer), self.sequence_length)

        # Réinitialiser le tampon
        self.agent.reset_state_buffer()

        # Vérifier que le tampon est vide
        self.assertEqual(len(self.agent.state_buffer), 0)

    def test_get_sequence_state(self):
        """Teste la récupération de la séquence d'états."""
        # Ajouter quelques états au tampon
        for i in range(3):
            state = np.ones(self.state_dim) * i
            self.agent.update_state_buffer(state)

        # Obtenir la séquence d'états
        sequence = self.agent.get_sequence_state()

        # Vérifier la forme de la séquence
        self.assertEqual(sequence.shape, (self.sequence_length, self.state_dim))

        # Vérifier que la séquence contient les états (avec padding)
        for i in range(self.sequence_length - 3):
            np.testing.assert_array_equal(sequence[i], np.zeros(self.state_dim))

        for i in range(3):
            np.testing.assert_array_equal(
                sequence[self.sequence_length - 3 + i], np.ones(self.state_dim) * i
            )

    def test_sample_action(self):
        """Teste l'échantillonnage d'actions."""
        # Réinitialiser le tampon d'état
        self.agent.reset_state_buffer()

        # Ajouter un état au tampon
        state = np.random.random(self.state_dim)
        self.agent.update_state_buffer(state)

        # Échantillonner une action en mode normal
        action = self.agent.sample_action(state)

        # Vérifier la forme de l'action
        self.assertEqual(action.shape, (self.action_dim,))

        # Vérifier que l'action est dans les limites
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

        # Échantillonner une action en mode évaluation
        eval_action = self.agent.sample_action(state, evaluate=True)

        # Vérifier la forme de l'action d'évaluation
        self.assertEqual(eval_action.shape, (self.action_dim,))

    def test_remember(self):
        """Teste la mémorisation des transitions."""
        # Vérifier que le tampon de replay est initialement vide
        self.assertEqual(len(self.agent.replay_buffer), 0)

        # Créer une transition
        state = np.random.random(self.state_dim)
        action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        reward = 1.0
        next_state = np.random.random(self.state_dim)
        done = False

        # Mémoriser la transition
        self.agent.remember(state, action, reward, next_state, done)

        # Vérifier que la transition a été mémorisée
        self.assertEqual(len(self.agent.replay_buffer), 1)

    def test_update_target_networks(self):
        """Teste la mise à jour des réseaux cibles."""
        # Sauvegarder les poids initiaux
        critic_1_weights = [p.clone() for p in self.agent.critic_1.parameters()]
        critic_2_weights = [p.clone() for p in self.agent.critic_2.parameters()]

        # Modifier les poids des réseaux principaux
        for param in self.agent.critic_1.parameters():
            param.data += torch.randn_like(param.data) * 0.1
        for param in self.agent.critic_2.parameters():
            param.data += torch.randn_like(param.data) * 0.1

        # Mettre à jour les réseaux cibles
        self.agent.update_target_networks()

        # Vérifier que les poids ont été mis à jour et sont proches mais pas identiques
        for target_param, param in zip(
            self.agent.critic_1_target.parameters(), self.agent.critic_1.parameters()
        ):
            # Les paramètres ne devraient pas être exactement égaux
            self.assertFalse(torch.equal(target_param, param))
            # Vérifier que les poids ont été mis à jour (ne sont pas restés les mêmes)
            self.assertFalse(torch.equal(target_param.data, critic_1_weights[0].data))
            # Vérifier que les poids sont dans l'intervalle attendu
            diff = torch.abs(target_param.data - param.data)
            self.assertTrue(torch.all(diff <= 1.0))

        for target_param, param in zip(
            self.agent.critic_2_target.parameters(), self.agent.critic_2.parameters()
        ):
            self.assertFalse(torch.equal(target_param, param))
            self.assertFalse(torch.equal(target_param.data, critic_2_weights[0].data))
            diff = torch.abs(target_param.data - param.data)
            self.assertTrue(torch.all(diff <= 1.0))

    def test_save_load_models(self):
        """Teste la sauvegarde et le chargement des modèles."""
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configurer le répertoire de checkpoints de l'agent
            self.agent.checkpoints_dir = Path(temp_dir)

            # Sauvegarder les modèles
            self.agent.save_models(suffix="_test")

            # Vérifier que les fichiers ont été créés
            checkpoint_files = os.listdir(temp_dir)
            self.assertTrue(any("checkpoint" in f for f in checkpoint_files))

            # Trouver le dernier checkpoint
            checkpoint_dir = sorted([d for d in checkpoint_files if "checkpoint" in d])[
                -1
            ]
            model_path = os.path.join(temp_dir, checkpoint_dir, "models.pt")

            # Charger les modèles
            self.agent.load_models(model_path)

            # Vérifier que les modèles ont été chargés
            self.assertIsNotNone(self.agent.actor)
            self.assertIsNotNone(self.agent.critic_1)
            self.assertIsNotNone(self.agent.critic_2)
            self.assertIsNotNone(self.agent.critic_1_target)
            self.assertIsNotNone(self.agent.critic_2_target)

    def test_integration(self):
        """Teste l'intégration complète de l'agent avec l'environnement."""
        # Réinitialiser l'environnement et le tampon d'état
        state, _ = self.env.reset()
        self.agent.reset_state_buffer()

        # Remplir le tampon d'état avec l'état initial
        for _ in range(self.sequence_length):
            self.agent.update_state_buffer(state)

        # Exécuter quelques étapes
        for step in range(10):
            # Échantillonner une action
            action = self.agent.sample_action(state)

            # Exécuter l'action dans l'environnement
            next_state, reward, done, _, _ = self.env.step(action)

            # Mémoriser la transition
            sequence_state = self.agent.get_sequence_state()
            next_sequence_state = np.copy(sequence_state)
            next_sequence_state[:-1] = sequence_state[1:]
            next_sequence_state[-1] = next_state

            self.agent.remember(
                sequence_state, action, reward, next_sequence_state, done
            )

            # Entraîner l'agent
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                critic_loss, actor_loss, alpha_loss = self.agent.train()

                # Vérifier que les pertes sont des nombres
                self.assertIsInstance(critic_loss, float)
                self.assertIsInstance(actor_loss, float)
                if alpha_loss is not None:
                    self.assertIsInstance(alpha_loss, float)

            # Mettre à jour l'état et le tampon d'état
            state = next_state
            self.agent.update_state_buffer(state)

            # Vérifier que le tampon d'état a la bonne taille
            self.assertEqual(len(self.agent.state_buffer), self.sequence_length)


if __name__ == "__main__":
    unittest.main()
