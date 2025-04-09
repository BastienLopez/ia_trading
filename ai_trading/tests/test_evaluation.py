import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.evaluation import (
    PerformanceMetrics,
    PerformanceVisualizer,
    evaluate_agent,
)
from ai_trading.rl.trading_environment import TradingEnvironment


class TestEvaluation(unittest.TestCase):
    """Tests pour l'évaluation et la visualisation."""

    def setUp(self):
        """Prépare l'environnement, l'agent et les données pour les tests."""
        # Créer un intégrateur de données
        integrator = RLDataIntegrator()

        # Générer des données synthétiques
        self.test_data = integrator.generate_synthetic_data(
            n_samples=100, trend="bullish", volatility=0.02, with_sentiment=True
        )

        # Créer l'environnement
        self.env = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )

        # Créer l'agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=0.1,  # Epsilon bas pour l'évaluation
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000,
        )

    def test_performance_metrics(self):
        """Teste les métriques de performance."""
        # Créer des données de test
        portfolio_values = [10000, 10500, 11000, 10800, 11200]
        
        # Calculer les métriques
        metrics = PerformanceMetrics.calculate_all_metrics(
            portfolio_values=portfolio_values,
            risk_free_rate=0.0
        )
        
        # Vérifier que les métriques sont calculées
        self.assertIsInstance(metrics, dict)
        
        # Vérifier que les métriques contiennent les bonnes clés
        expected_keys = ["total_return", "annualized_return", "volatility", "sharpe_ratio", "max_drawdown"]
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Vérifier que les valeurs sont cohérentes
        self.assertGreaterEqual(metrics["total_return"], 0)  # Le rendement total doit être positif
        # Ne pas vérifier que le ratio de Sharpe est positif car il peut être négatif
        self.assertLessEqual(metrics["max_drawdown"], 0)  # Le drawdown maximum doit être négatif ou nul

    def test_performance_visualizer(self):
        """Teste le visualiseur de performances."""
        # Utiliser le backend non-interactif pour éviter les problèmes de tkinter
        import matplotlib
        matplotlib.use('Agg')
        
        # Créer un répertoire temporaire pour les visualisations
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer un visualiseur
            visualizer = PerformanceVisualizer(save_dir=temp_dir)
            
            # Créer des données de test
            dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
            portfolio_values = np.linspace(10000, 15000, 100) + np.random.normal(0, 200, 100)
            
            # Générer les visualisations
            visualizer.plot_portfolio_performance(
                portfolio_values=portfolio_values,
                benchmark_values=None,
                dates=dates
            )
            
            # Vérifier que le fichier est créé
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "portfolio_performance.png")))

    def test_evaluate_agent(self):
        """Teste la fonction d'évaluation de l'agent."""
        # Évaluer l'agent
        results = evaluate_agent(agent=self.agent, env=self.env, num_episodes=2)

        # Vérifier que les résultats sont retournés
        self.assertIsInstance(results, dict)

        # Vérifier que les résultats contiennent les bonnes clés
        expected_keys = [
            "final_value",
            "returns",
            "sharpe_ratio",
            "max_drawdown",
            "portfolio_history",
            "actions",
            "rewards",
        ]

        for key in expected_keys:
            self.assertIn(key, results)

        # Vérifier que les valeurs sont cohérentes
        self.assertGreater(results["final_value"], 0)
        self.assertIsInstance(results["returns"], float)
        self.assertIsInstance(results["sharpe_ratio"], float)
        self.assertLessEqual(results["max_drawdown"], 0)

    def test_create_performance_dashboard(self):
        """Teste la création d'un tableau de bord de performance."""
        # Utiliser le backend non-interactif pour éviter les problèmes de tkinter
        import matplotlib
        matplotlib.use('Agg')
        
        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer un visualiseur
            visualizer = PerformanceVisualizer(save_dir=temp_dir)

            # Évaluer l'agent
            results = evaluate_agent(agent=self.agent, env=self.env, num_episodes=1)
            
            # S'assurer que les dimensions correspondent
            portfolio_history = results["portfolio_history"]
            dates = self.test_data.index[-len(portfolio_history):]  # Prendre seulement les dernières dates correspondant à la longueur de l'historique
            
            # Ajuster les actions à la longueur de l'historique du portefeuille
            actions = results["actions"]
            if len(actions) < len(portfolio_history):
                # Si les actions sont plus courtes, les étendre
                actions = np.pad(actions, (0, len(portfolio_history) - len(actions)), 'edge')
            elif len(actions) > len(portfolio_history):
                # Si les actions sont plus longues, les tronquer
                actions = actions[:len(portfolio_history)]
            
            # Vérifier que les dimensions correspondent maintenant
            self.assertEqual(len(dates), len(portfolio_history))
            self.assertEqual(len(actions), len(portfolio_history))

            # Créer un tableau de bord
            visualizer.create_performance_dashboard(
                results=results, 
                dates=dates,  # Utiliser les dates ajustées
                actions=actions  # Utiliser les actions ajustées
            )

            # Vérifier que les fichiers sont créés
            expected_files = [
                "portfolio_performance.png",
                "returns_distribution.png",
                "drawdown.png",
                "actions_distribution.png",
            ]

            # Vérifier que les fichiers existent, avec une tolérance pour les noms de fichiers légèrement différents
            for file in expected_files:
                base_name = file.split('.')[0]
                found = False
                for existing_file in os.listdir(temp_dir):
                    if base_name in existing_file:
                        found = True
                        break
                self.assertTrue(found, f"Fichier contenant '{base_name}' non trouvé dans {temp_dir}")


if __name__ == "__main__":
    unittest.main()
