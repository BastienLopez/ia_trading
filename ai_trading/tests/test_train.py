import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.train import train_agent, TrainingMonitor
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.data_integration import RLDataIntegrator

class TestTrain(unittest.TestCase):
    """Tests pour la boucle d'entraînement."""
    
    def setUp(self):
        """Prépare l'environnement, l'agent et les données pour les tests."""
        # Créer un intégrateur de données
        integrator = RLDataIntegrator()
        
        # Générer des données synthétiques
        self.test_data = integrator.generate_synthetic_data(
            n_samples=100,
            trend='bullish',
            volatility=0.02,
            with_sentiment=True
        )
        
        # Créer l'environnement
        self.env = TradingEnvironment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
        
        # Créer l'agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000
        )
    
    def test_training_monitor(self):
        """Teste la classe TrainingMonitor."""
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer un moniteur d'entraînement
            monitor = TrainingMonitor(save_dir=temp_dir)
            
            # Mettre à jour le moniteur avec des données fictives
            for i in range(10):
                monitor.update(
                    episode=i,
                    reward=i * 10,
                    portfolio_value=10000 + i * 100,
                    returns=i * 0.01,
                    loss=1.0 / (i + 1),
                    epsilon=1.0 - i * 0.1
                )
            
            # Sauvegarder les graphiques
            monitor.save_plots()
            
            # Vérifier que les fichiers sont créés
            expected_files = ['rewards.png', 'portfolio_values.png', 'returns.png', 'losses.png', 'epsilon.png']
            for file in expected_files:
                self.assertTrue(os.path.exists(os.path.join(temp_dir, file)))
            
            # Récupérer l'historique
            history = monitor.get_history()
            
            # Vérifier que l'historique contient les bonnes clés
            expected_keys = ['episode_rewards', 'episode_portfolio_values', 'episode_returns', 'losses', 'epsilon']
            for key in expected_keys:
                self.assertIn(key, history)
                self.assertEqual(len(history[key]), 10)
    
    def test_train_agent(self):
        """Teste la fonction d'entraînement de l'agent."""
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Entraîner l'agent pour quelques épisodes
            history = train_agent(
                agent=self.agent,
                env=self.env,
                episodes=5,
                batch_size=32,
                update_target_every=2,
                save_path=os.path.join(temp_dir, 'test_model'),
                visualize=True,
                checkpoint_interval=2,
                early_stopping=None,
                max_steps_per_episode=None,
                use_tensorboard=False
            )
            
            # Vérifier que l'historique est retourné
            self.assertIsInstance(history, dict)
            
            # Vérifier que l'historique contient les bonnes clés
            expected_keys = ['episode_rewards', 'episode_portfolio_values', 'episode_returns', 'losses', 'epsilon']
            for key in expected_keys:
                self.assertIn(key, history)
                self.assertEqual(len(history[key]), 5)
            
            # Vérifier que le modèle final est sauvegardé
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'test_model_final.h5')))
    
    def test_early_stopping(self):
        """Teste la fonctionnalité d'arrêt anticipé."""
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Entraîner l'agent avec arrêt anticipé
            history = train_agent(
                agent=self.agent,
                env=self.env,
                episodes=20,  # Plus d'épisodes que nécessaire
                batch_size=32,
                update_target_every=2,
                save_path=os.path.join(temp_dir, 'test_model'),
                visualize=False,
                checkpoint_interval=5,
                early_stopping={
                    'patience': 2,
                    'min_delta': 0.0,
                    'metric': 'reward'
                },
                max_steps_per_episode=None,
                use_tensorboard=False
            )
            
            # Vérifier que l'entraînement s'est arrêté avant 20 épisodes
            self.assertLess(len(history['episode_rewards']), 20)
            
            # Vérifier que le meilleur modèle est sauvegardé
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'test_model_best.h5')))

if __name__ == '__main__':
    unittest.main() 