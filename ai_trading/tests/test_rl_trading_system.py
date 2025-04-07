import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl_agent import RLTradingSystem
from ai_trading.rl.data_integration import RLDataIntegrator

class TestRLTradingSystem(unittest.TestCase):
    """Tests pour le système complet de trading RL."""
    
    def setUp(self):
        """Prépare le système et les données pour les tests."""
        # Créer un intégrateur de données
        self.integrator = RLDataIntegrator()
        
        # Générer des données synthétiques
        self.test_data = self.integrator.generate_synthetic_data(
            n_samples=100,
            trend='bullish',
            volatility=0.02,
            with_sentiment=True
        )
        
        # Créer le système de trading
        self.system = RLTradingSystem()
    
    def test_create_environment(self):
        """Teste la création de l'environnement."""
        # Créer l'environnement
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
        
        # Vérifier que l'environnement est créé
        self.assertIsNotNone(env)
        self.assertEqual(env.initial_balance, 10000)
        self.assertEqual(env.transaction_fee, 0.001)
        self.assertEqual(env.window_size, 10)
    
    def test_create_agent(self):
        """Teste la création de l'agent."""
        # Créer l'environnement d'abord
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
        
        # Créer l'agent
        agent = self.system.create_agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000
        )
        
        # Vérifier que l'agent est créé
        self.assertIsNotNone(agent)
        self.assertEqual(agent.state_size, env.observation_space.shape[0])
        self.assertEqual(agent.action_size, env.action_space.n)
    
    def test_train(self):
        """Teste l'entraînement du système."""
        # Créer l'environnement
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
        
        # Créer l'agent
        agent = self.system.create_agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000
        )
        
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Entraîner le système pour quelques épisodes
            history = self.system.train(
                episodes=3,
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
                self.assertEqual(len(history[key]), 3)
    
    def test_evaluate(self):
        """Teste l'évaluation du système."""
        # Créer l'environnement
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
        
        # Créer l'agent
        agent = self.system.create_agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=0.0,  # Pas d'exploration pour l'évaluation
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000
        )
        
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Évaluer le système
            results = self.system.evaluate(
                test_data=self.test_data,
                num_episodes=1,
                visualize=True,
                save_dir=temp_dir
            )
            
            # Vérifier que les résultats sont retournés
            self.assertIsInstance(results, dict)
            
            # Vérifier que les résultats contiennent les bonnes clés
            expected_keys = ['final_value', 'returns', 'sharpe_ratio', 'max_drawdown', 
                            'portfolio_history', 'actions', 'rewards']
            
            for key in expected_keys:
                self.assertIn(key, results)
    
    def test_integrate_data(self):
        """Teste l'intégration des données."""
        # Créer des données de marché et de sentiment
        market_data = pd.DataFrame({
            'open': np.random.random(100) * 100 + 100,
            'high': np.random.random(100) * 100 + 110,
            'low': np.random.random(100) * 100 + 90,
            'close': np.random.random(100) * 100 + 105,
            'volume': np.random.random(100) * 1000 + 1000
        })
        
        sentiment_data = pd.DataFrame({
            'compound_score': np.random.uniform(-1, 1, 100)
        })
        
        # Intégrer les données
        train_data, test_data = self.system.integrate_data(
            market_data=market_data,
            sentiment_data=sentiment_data,
            window_size=10,
            test_split=0.2
        )
        
        # Vérifier que les données sont divisées correctement
        expected_train_size = int(len(market_data) * 0.8)
        expected_test_size = len(market_data) - expected_train_size
        
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), expected_test_size)
    
    def test_test_random_strategy(self):
        """Teste la stratégie aléatoire."""
        # Créer l'environnement
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
        
        # Tester la stratégie aléatoire
        results = self.system.test_random_strategy(num_episodes=2)
        
        # Vérifier que les résultats sont retournés
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        # Vérifier que chaque résultat contient les bonnes clés
        expected_keys = ['episode', 'final_value', 'returns', 'avg_reward', 'portfolio_history']
        for result in results:
            for key in expected_keys:
                self.assertIn(key, result)

if __name__ == '__main__':
    unittest.main() 