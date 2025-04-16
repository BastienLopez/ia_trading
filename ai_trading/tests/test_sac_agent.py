import unittest
import numpy as np
import tensorflow as tf
import sys
import os
import logging
import pandas as pd
import tempfile

# Configurer le logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment


class TestSACAgent(unittest.TestCase):
    """Tests unitaires pour l'agent SAC (Soft Actor-Critic)"""

    def setUp(self):
        """Configuration pour chaque test"""
        # Créer un petit environnement de test avec des données synthétiques
        np.random.seed(42)  # Pour la reproductibilité
        tf.random.set_seed(42)
        
        # Générer des données synthétiques pour l'environnement
        n_samples = 100
        dates = pd.date_range(start='2023-01-01', periods=n_samples)
        prices = np.linspace(100, 200, n_samples) + np.random.normal(0, 5, n_samples)
        volumes = np.random.normal(1000, 200, n_samples)
        
        # Créer un DataFrame avec les données
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': prices - np.random.uniform(0, 2, n_samples),
            'high': prices + np.random.uniform(0, 2, n_samples),
            'low': prices - np.random.uniform(0, 2, n_samples),
            'close': prices,
            'volume': volumes,
            'market_cap': prices * volumes
        })
        
        # Créer l'environnement avec action_type="continuous"
        self.env = TradingEnvironment(
            df=self.df,
            initial_balance=10000.0,
            transaction_fee=0.001,
            window_size=10,
            action_type="continuous"  # Important pour tester un agent avec actions continues
        )
        
        # Créer un état pour déterminer sa taille réelle
        state, _ = self.env.reset()
        # Dans l'implémentation actuelle, la taille réelle de l'état diffère 
        # de celle déclarée dans observation_space
        self.state_size = state.shape[0]  # Utiliser la taille réelle, pas la taille déclarée
        
        # Créer l'agent SAC avec la bonne taille d'état
        self.agent = SACAgent(
            state_size=self.state_size,
            action_size=1,  # L'environnement a un espace d'action de dimension 1
            action_bounds=(-1, 1),  # Les actions sont normalisées entre -1 et 1
            batch_size=32,
            buffer_size=5000,
            hidden_size=64
        )
        
        # Collecter quelques expériences pour le tampon de replay
        state, _ = self.env.reset()
        for _ in range(50):
            action = np.random.uniform(-1, 1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.agent.remember(state, action, reward, next_state, terminated)
            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state

    def test_agent_initialization(self):
        """Teste que l'agent est correctement initialisé"""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, 1)
        self.assertEqual(self.agent.action_low, -1)
        self.assertEqual(self.agent.action_high, 1)
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic_1)
        self.assertIsNotNone(self.agent.critic_2)
        self.assertIsNotNone(self.agent.critic_1_target)
        self.assertIsNotNone(self.agent.critic_2_target)

    def test_action_selection(self):
        """Teste que l'agent peut sélectionner des actions correctement"""
        state, _ = self.env.reset()
        
        # Test mode stochastique
        action = self.agent.act(state)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(-1 <= action[0] <= 1)
        
        # Test mode déterministe
        action_det = self.agent.act(state, deterministic=True)
        self.assertIsInstance(action_det, np.ndarray)
        self.assertEqual(action_det.shape, (1,))
        self.assertTrue(-1 <= action_det[0] <= 1)

    def test_training(self):
        """Teste que l'agent peut être entraîné"""
        # Entraîner l'agent sur un lot
        metrics = self.agent.train()
        
        # Vérifier que les métriques existent
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("alpha_loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("alpha", metrics)
        
        # Vérifier que les historiques sont mis à jour
        self.assertEqual(len(self.agent.critic_loss_history), 1)
        self.assertEqual(len(self.agent.actor_loss_history), 1)
        self.assertEqual(len(self.agent.alpha_loss_history), 1)
        self.assertEqual(len(self.agent.entropy_history), 1)

    def test_save_load(self):
        """Teste les fonctionnalités de sauvegarde et chargement"""
        import tempfile
        import shutil
        
        # Créer un répertoire temporaire
        temp_dir = tempfile.mkdtemp()
        try:
            # Sauvegarder l'agent
            self.agent.save(temp_dir)
            
            # Vérifier que les fichiers existent
            self.assertTrue(os.path.exists(f"{temp_dir}/actor.h5"))
            self.assertTrue(os.path.exists(f"{temp_dir}/critic_1.h5"))
            self.assertTrue(os.path.exists(f"{temp_dir}/critic_2.h5"))
            self.assertTrue(os.path.exists(f"{temp_dir}/log_alpha.npy"))
            
            # Créer un nouvel agent avec les mêmes paramètres
            new_agent = SACAgent(
                state_size=self.state_size,
                action_size=1,
                action_bounds=(-1, 1),
                hidden_size=64  # Utiliser le même hidden_size que l'agent original
            )
            
            # Charger les poids
            new_agent.load(temp_dir)
            
            # Vérifier que les poids sont chargés (comparer un poids de chaque réseau)
            for model1, model2 in [
                (self.agent.actor, new_agent.actor),
                (self.agent.critic_1, new_agent.critic_1)
            ]:
                # Vérifier que les poids d'au moins une couche sont égaux
                original_weights = model1.get_weights()[0]
                loaded_weights = model2.get_weights()[0]
                np.testing.assert_array_equal(original_weights, loaded_weights)
                
        finally:
            # Nettoyer
            shutil.rmtree(temp_dir)

    def test_replay_buffer(self):
        """Teste le fonctionnement du tampon de replay"""
        # Vérifier la taille actuelle
        initial_size = len(self.agent.replay_buffer)
        
        # Ajouter une expérience
        state = np.random.random(self.state_size)
        action = np.array([0.5])
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False
        
        self.agent.remember(state, action, reward, next_state, done)
        
        # Vérifier que la taille a augmenté
        self.assertEqual(len(self.agent.replay_buffer), initial_size + 1)
        
        # Échantillonner un lot
        batch_size = 10
        if len(self.agent.replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = self.agent.replay_buffer.sample(batch_size)
            
            # Vérifier les dimensions
            self.assertEqual(states.shape[0], batch_size)
            self.assertEqual(actions.shape[0], batch_size)
            self.assertEqual(rewards.shape[0], batch_size)
            self.assertEqual(next_states.shape[0], batch_size)
            self.assertEqual(dones.shape[0], batch_size)

    def test_scale_unscale_actions(self):
        """Teste les fonctions de mise à l'échelle et déséchelonnage des actions"""
        # Initialiser un agent avec des limites d'action personnalisées
        agent = SACAgent(
            state_size=self.state_size,
            action_size=1,
            action_bounds=(-2, 3)  # Limites personnalisées
        )
        
        # Tester le dimensionnement
        normalized_actions = np.array([[-1.0], [0.0], [1.0]])
        scaled_actions = agent._scale_action(normalized_actions)
        
        # Vérifier les valeurs
        np.testing.assert_array_almost_equal(scaled_actions, np.array([[-2.0], [0.5], [3.0]]))
        
        # Tester le dédimensionnement
        unscaled_actions = agent._unscale_action(scaled_actions)
        np.testing.assert_array_almost_equal(unscaled_actions, normalized_actions)


if __name__ == '__main__':
    unittest.main() 