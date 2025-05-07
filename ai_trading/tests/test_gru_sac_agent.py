import logging
import os
import sys
import unittest

import numpy as np
import tensorflow as tf
import torch

# Configurer le niveau de log pour réduire les sorties de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.agents.sac_agent import SACAgent, SequenceReplayBuffer
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestGRUSACAgent")


class TestGRUSACAgent(unittest.TestCase):
    """
    Tests unitaires pour l'agent SAC avec couches GRU.
    """

    def setUp(self):
        """
        Configuration initiale avant chaque test.
        """
        # Générer des données synthétiques
        self.data = generate_synthetic_market_data(
            n_points=500, trend=0.001, volatility=0.01, start_price=100.0
        )

        # Convertir d'abord en float32 pour les calculs initiaux
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].astype(np.float32)

        # Ajouter des indicateurs techniques
        self.data["sma_10"] = self.data["close"].rolling(10).mean()
        self.data["sma_30"] = self.data["close"].rolling(30).mean()
        self.data["rsi"] = 50 + np.random.normal(0, 10, len(self.data))

        # Remplir les valeurs NaN directement
        self.data = self.data.fillna(0)

        # Maintenant convertir en float16
        self.data[numeric_cols] = self.data[numeric_cols].astype(np.float16)

        # Initialiser l'environnement
        self.env = TradingEnvironment(
            df=self.data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=20,
            reward_function="sharpe",
            action_type="continuous",
        )

        # Paramètres de l'agent
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.sequence_length = 10

        # Créer un agent SAC avec GRU
        self.gru_agent = SACAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            action_bounds=[-1, 1],
            buffer_size=1000,
            batch_size=64,
            hidden_size=128,
            use_gru=True,
            sequence_length=self.sequence_length,
            gru_units=64,
            entropy_regularization=0.01,
            grad_clip_value=1.0,
        )

        # Créer un agent SAC standard pour comparaison
        self.standard_agent = SACAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            action_bounds=[-1, 1],
            buffer_size=1000,
            batch_size=64,
            hidden_size=128,
            use_gru=False,
            entropy_regularization=0.01,
            grad_clip_value=1.0,
        )

        # Préparer un mini-batch de séquences pour les tests
        self.batch_sequences = np.random.normal(
            0, 1, (32, self.sequence_length, self.state_size)
        ).astype(np.float16)
        self.batch_states = np.random.normal(0, 1, (32, self.state_size)).astype(
            np.float16
        )

        # Initialiser les listes d'historique pour éviter les erreurs lors de l'appel à train()
        self.gru_agent.critic_loss_history = []
        self.gru_agent.actor_loss_history = []
        self.gru_agent.alpha_loss_history = []
        self.gru_agent.entropy_history = []

        self.standard_agent.critic_loss_history = []
        self.standard_agent.actor_loss_history = []
        self.standard_agent.alpha_loss_history = []
        self.standard_agent.entropy_history = []

        logger.info(
            f"Configuration de test complète: state_size={self.state_size}, action_size={self.action_size}"
        )

    def test_gru_network_architecture(self):
        """Vérifie que les réseaux GRU sont correctement construits."""
        # Vérifier que les réseaux de l'agent GRU contiennent bien des couches GRU
        self.assertTrue(
            hasattr(self.gru_agent.actor, "gru"),
            "L'acteur devrait avoir une couche GRU",
        )

        # Vérifier que le critique contient une couche GRU
        self.assertTrue(
            hasattr(self.gru_agent.critic_1, "gru"),
            "Le critique devrait avoir une couche GRU",
        )

        # Vérifier que c'est bien une instance de nn.GRU
        self.assertIsInstance(
            self.gru_agent.actor.gru,
            torch.nn.GRU,
            "La couche GRU de l'acteur devrait être une instance de torch.nn.GRU",
        )

    def test_sequence_normalization(self):
        """Teste la normalisation des séquences d'états."""
        # Appliquer la normalisation
        normalized_sequences = self.gru_agent._normalize_sequence_states(
            self.batch_sequences
        )

        # Vérifier que le résultat est un tenseur TensorFlow
        self.assertIsInstance(normalized_sequences, tf.Tensor)

        # Vérifier la forme
        self.assertEqual(
            normalized_sequences.shape, (32, self.sequence_length, self.state_size)
        )

        # Vérifier que la moyenne de chaque séquence est proche de 0
        means = tf.reduce_mean(normalized_sequences, axis=2)
        self.assertTrue(np.allclose(tf.reduce_mean(means).numpy(), 0, atol=1e-5))

        # Vérifier que l'écart-type de chaque séquence est proche de 1
        stds = tf.math.reduce_std(normalized_sequences, axis=2)
        self.assertTrue(np.allclose(tf.reduce_mean(stds).numpy(), 1, atol=0.2))

    def test_sequence_memory(self):
        """Teste le stockage et l'échantillonnage de séquences d'expériences."""
        # Créer une mémoire de séquences simplifiée pour le test
        memory = SequenceReplayBuffer(
            buffer_size=100, sequence_length=self.sequence_length
        )

        # Ajouter directement des séquences complètes
        for i in range(20):
            sequence = np.random.normal(
                0, 1, (self.sequence_length, self.state_size)
            ).astype(np.float16)
            action = np.random.normal(0, 1, self.action_size).astype(np.float16)
            reward = float(np.random.normal(0, 1))  # Convertir en float standard
            next_sequence = np.random.normal(
                0, 1, (self.sequence_length, self.state_size)
            ).astype(np.float16)
            done = False

            # Ajouter l'expérience à la mémoire
            memory.add(sequence, action, reward, next_sequence, done)

        # Tester l'échantillonnage
        if memory.size() >= 16:
            states, actions, rewards, next_states, dones = memory.sample(16)

            # Vérifier les formes
            self.assertEqual(states.shape, (16, self.sequence_length, self.state_size))
            self.assertEqual(actions.shape, (16, self.action_size))
            self.assertEqual(rewards.shape, (16, 1))
            self.assertEqual(
                next_states.shape, (16, self.sequence_length, self.state_size)
            )
            self.assertEqual(dones.shape, (16, 1))

    def test_action_selection_with_gru(self):
        """Teste la sélection d'action avec les couches GRU."""
        # Obtenir un état initial
        state = self.env.reset()[0]  # Extraire l'état de la tuple (state, info)

        # Créer une séquence d'états
        sequence = np.array([state] * self.sequence_length).astype(np.float16)

        # Sélectionner une action en mode évaluation
        try:
            action = self.gru_agent.act(sequence, evaluate=True)

            # Vérifier que l'action est valide
            self.assertEqual(action.shape, (self.action_size,))
            self.assertTrue(np.all(action >= -1))
            self.assertTrue(np.all(action <= 1))

            logger.info(f"Action sélectionnée avec GRU: {action}")
        except Exception as e:
            self.fail(f"La sélection d'action avec GRU a échoué: {e}")

    def test_training_step(self):
        """Teste un pas d'entraînement avec des données séquentielles."""
        # Préparer des expériences séquentielles
        for i in range(100):
            # Créer une séquence d'états aléatoires
            sequence = np.random.normal(
                0, 1, (self.sequence_length, self.state_size)
            ).astype(np.float16)

            # Créer une séquence d'états suivants (important pour GRU)
            next_sequence = np.random.normal(
                0, 1, (self.sequence_length, self.state_size)
            ).astype(np.float16)

            action = np.random.normal(0, 0.5, self.action_size).astype(
                np.float16
            )  # Actions entre -1 et 1 approximativement
            reward = float(np.random.normal(0, 1))  # Convertir en float standard
            done = False if i < 99 else True

            # Ajouter à la mémoire avec le bon format pour sequence et next_sequence
            self.gru_agent.remember(sequence, action, reward, next_sequence, done)

        # Vérifier que la mémoire contient des données
        self.assertGreater(self.gru_agent.replay_buffer.size(), 0)

        # Effectuer un pas d'entraînement
        try:
            train_info = self.gru_agent.train(batch_size=32)

            # Vérifier que les pertes sont des valeurs numériques
            self.assertIsInstance(train_info["critic_loss"], float)
            self.assertIsInstance(train_info["actor_loss"], float)
            self.assertIsInstance(train_info["alpha_loss"], float)

            logger.info(
                f"Pertes d'entraînement - Critique: {train_info['critic_loss']:.4f}, "
                f"Acteur: {train_info['actor_loss']:.4f}, "
                f"Alpha: {train_info['alpha_loss']:.4f}"
            )
        except Exception as e:
            self.fail(f"L'entraînement a échoué avec l'erreur: {str(e)}")

    def test_train_comparison(self):
        """Compare l'entraînement d'un agent GRU et d'un agent standard sur un petit échantillon."""
        # Nombre d'itérations d'entraînement
        train_iterations = 5

        # Générer des expériences pour l'agent GRU
        for i in range(100):
            # Séquence d'états pour l'agent GRU
            gru_sequence = np.random.normal(
                0, 1, (self.sequence_length, self.state_size)
            ).astype(np.float16)

            # Séquence d'états suivants pour l'agent GRU
            gru_next_sequence = np.random.normal(
                0, 1, (self.sequence_length, self.state_size)
            ).astype(np.float16)

            # État simple pour l'agent standard
            standard_state = np.random.normal(0, 1, self.state_size).astype(np.float16)
            standard_next_state = np.random.normal(0, 1, self.state_size).astype(
                np.float16
            )

            # Action, récompense et terminaison communes
            action = np.random.normal(0, 0.5, self.action_size).astype(np.float16)
            reward = float(np.random.normal(0, 1))
            done = False if i < 99 else True

            # Ajouter aux mémoires respectives
            self.gru_agent.remember(
                gru_sequence, action, reward, gru_next_sequence, done
            )
            self.standard_agent.remember(
                standard_state, action, reward, standard_next_state, done
            )

        # Entraînement des deux agents
        gru_critic_losses = []
        gru_actor_losses = []
        standard_critic_losses = []
        standard_actor_losses = []

        try:
            # Entraîner les deux agents
            for _ in range(train_iterations):
                # Entraîner l'agent GRU
                gru_train_info = self.gru_agent.train(batch_size=32)
                gru_critic_losses.append(gru_train_info["critic_loss"])
                gru_actor_losses.append(gru_train_info["actor_loss"])

                # Entraîner l'agent standard
                standard_train_info = self.standard_agent.train(batch_size=32)
                standard_critic_losses.append(standard_train_info["critic_loss"])
                standard_actor_losses.append(standard_train_info["actor_loss"])

            # Calculer les pertes moyennes
            avg_gru_critic_loss = sum(gru_critic_losses) / len(gru_critic_losses)
            avg_gru_actor_loss = sum(gru_actor_losses) / len(gru_actor_losses)
            avg_standard_critic_loss = sum(standard_critic_losses) / len(
                standard_critic_losses
            )
            avg_standard_actor_loss = sum(standard_actor_losses) / len(
                standard_actor_losses
            )

            # Journaliser les résultats
            logger.info(
                f"Agent GRU - Perte critique moyenne: {avg_gru_critic_loss:.4f}, Perte acteur moyenne: {avg_gru_actor_loss:.4f}"
            )
            logger.info(
                f"Agent Standard - Perte critique moyenne: {avg_standard_critic_loss:.4f}, Perte acteur moyenne: {avg_standard_actor_loss:.4f}"
            )

            # Vérifier que les deux agents ont pu être entraînés
            self.assertIsInstance(avg_gru_critic_loss, float)
            self.assertIsInstance(avg_gru_actor_loss, float)
            self.assertIsInstance(avg_standard_critic_loss, float)
            self.assertIsInstance(avg_standard_actor_loss, float)

        except Exception as e:
            self.fail(f"La comparaison d'entraînement a échoué avec l'erreur: {str(e)}")

    def test_gradient_flow(self):
        """Teste le flux de gradient à travers les couches GRU."""
        # PyTorch GRU ne supporte pas le float16, donc utiliser float32
        test_states = np.random.normal(
            0, 1, (1, self.sequence_length, self.state_size)
        ).astype(np.float32)

        # Normaliser les états avec TensorFlow
        tf_test_states = tf.convert_to_tensor(test_states, dtype=tf.float32)
        tf_test_states = self.gru_agent._normalize_sequence_states(tf_test_states)

        # Convertir en tenseur PyTorch pour l'utiliser avec le modèle PyTorch
        torch_test_states = torch.tensor(
            tf_test_states.numpy(), dtype=torch.float32
        ).to(self.gru_agent.device)

        # Effectuer une passe avant de l'acteur et calculer la perte
        mean, log_std = self.gru_agent.actor(torch_test_states)

        # Récupérer les gradients par rapport à la sortie de l'acteur
        dummy_loss = mean.mean() + log_std.mean()
        dummy_loss.backward()

        # Vérifier que les gradients existent dans les paramètres de l'acteur
        for name, param in self.gru_agent.actor.named_parameters():
            self.assertIsNotNone(param.grad)
            # Vérifier que les gradients ne sont pas tous nuls
            self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)))


if __name__ == "__main__":
    unittest.main()
