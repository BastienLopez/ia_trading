import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.evaluation import evaluate_agent, PerformanceMetrics, PerformanceVisualizer
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.data_integration import RLDataIntegrator

class TestEvaluation(unittest.TestCase):
    """Tests pour l'évaluation et la visualisation."""
    
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
            epsilon=0.0,  # Pas d'exploration pour les tests
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000
        )
    
    def test_performance_metrics(self):
        """Teste les métriques de performance."""
        # Créer des données de test
        portfolio_values = np.array([10000, 10100, 10200, 10150, 10300, 10400])
        benchmark_values = np.array([10000, 10050, 10100, 10150, 10200, 10250])
        
        # Calculer les métriques
        metrics = PerformanceMetrics.calculate_all_metrics(
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            risk_free_rate=0.01
        )
        
        # Vérifier que les métriques sont calculées
        expected_metrics = ['total_return', 'annualized_return', 'volatility', 
                           'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                           'calmar_ratio', 'omega_ratio', 'beta', 'alpha', 
                           'tracking_error', 'information_ratio']
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Vérifier que les valeurs sont cohérentes
        self.assertGreater(metrics['total_return'], 0)
        self.assertGreater(metrics['sharpe_ratio'], 0)
        self.assertLessEqual(metrics['max_drawdown'], 0)
    
    def test_performance_visualizer(self):
        """Teste le visualiseur de performances."""
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer un visualiseur
            visualizer = PerformanceVisualizer(save_dir=temp_dir)
            
            # Créer des données de test
            portfolio_values = np.array([10000, 10100, 10200, 10150, 10300, 10400])
            benchmark_values = np.array([10000, 10050, 10100, 10150, 10200, 10250])
            dates = pd.date_range(start='2023-01-01', periods=6, freq='D')
            
            # Tracer la performance du portefeuille
            visualizer.plot_portfolio_performance(
                portfolio_values=portfolio_values,
                benchmark_values=benchmark_values,
                dates=dates
            )
            
            # Vérifier que le fichier est créé
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'portfolio_performance.png')))
            
            # Tracer la distribution des rendements
            returns = PerformanceMetrics.calculate_returns(portfolio_values)
            visualizer.plot_returns_distribution(returns)
            
            # Vérifier que le fichier est créé
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'returns_distribution.png')))
    
    def test_evaluate_agent(self):
        """Teste la fonction d'évaluation de l'agent."""
        # Évaluer l'agent
        results = evaluate_agent(
            agent=self.agent,
            env=self.env,
            num_episodes=2
        )
        
        # Vérifier que les résultats sont retournés
        self.assertIsInstance(results, dict)
        
        # Vérifier que les résultats contiennent les bonnes clés
        expected_keys = ['final_value', 'returns', 'sharpe_ratio', 'max_drawdown', 
                        'portfolio_history', 'actions', 'rewards']
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Vérifier que les valeurs sont cohérentes
        self.assertGreater(results['final_value'], 0)
        self.assertIsInstance(results['returns'], float)
        self.assertIsInstance(results['sharpe_ratio'], float)
        self.assertLessEqual(results['max_drawdown'], 0)
    
    def test_create_performance_dashboard(self):
        """Teste la création d'un tableau de bord de performance."""
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer un visualiseur
            visualizer = PerformanceVisualizer(save_dir=temp_dir)
            
            # Évaluer l'agent
            results = evaluate_agent(
                agent=self.agent,
                env=self.env,
                num_episodes=1
            )
            
            # Créer un tableau de bord
            visualizer.create_performance_dashboard(
                results=results,
                dates=self.test_data.index,
                actions=results['actions']
            )
            
            # Vérifier que les fichiers sont créés
            expected_files = ['portfolio_performance.png', 'returns_distribution.png', 
                             'drawdown.png', 'actions_distribution.png']
            
            for file in expected_files:
                self.assertTrue(os.path.exists(os.path.join(temp_dir, file)))

if __name__ == '__main__':
    unittest.main() 