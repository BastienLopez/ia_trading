"""
Test d'intégration pour l'optimisation bayésienne des hyperparamètres.

Ce test vérifie le fonctionnement complet de l'optimisation bayésienne
avec un environnement de trading simplifié et un petit ensemble de données.
"""

import os
import shutil
import unittest
import tempfile
import logging
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.bayesian_optimizer import (
    BayesianOptimizer,
    optimize_sac_agent_bayesian,
)
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment


# Désactiver les logs pour les tests
logging.getLogger("ai_trading.rl.bayesian_optimizer").setLevel(logging.ERROR)


class TestBayesianOptimizationIntegration(unittest.TestCase):
    """Test d'intégration pour l'optimisation bayésienne."""

    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        # Créer un répertoire temporaire pour les résultats
        cls.test_dir = tempfile.mkdtemp(prefix="bayesian_opt_integration_")

        # Générer des données synthétiques pour les tests
        cls.test_data = generate_synthetic_market_data(
            n_points=100,  # Taille réduite pour les tests
            trend=0.001,
            volatility=0.01,
            start_price=100.0,
        )

        # Ajouter des indicateurs techniques
        cls.test_data["sma_10"] = cls.test_data["close"].rolling(10).mean()
        cls.test_data["sma_30"] = cls.test_data["close"].rolling(30).mean()
        cls.test_data["rsi"] = 50 + np.random.normal(0, 10, len(cls.test_data))
        cls.test_data = cls.test_data.fillna(0)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def create_test_environment(self):
        """Créer un environnement de trading pour les tests."""
        return TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            reward_function="simple",
            action_type="continuous",
        )

    def test_bayesian_optimization_integration(self):
        """Test d'intégration complet de l'optimisation bayésienne."""
        # Définir un espace de paramètres très restreint pour accélérer le test
        param_space = {
            "learning_rate": (1e-4, 5e-4),
            "batch_size": (32, 64),
            "hidden_size": (64, 128),
        }

        # Patcher les méthodes qui utilisent matplotlib et l'évaluation des paramètres
        with patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._generate_convergence_plot"), \
             patch("ai_trading.rl.hyperparameter_optimizer.HyperparameterOptimizer._generate_plots"), \
             patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._evaluate_params") as mock_evaluate:
            
            # Configurer le mock pour retourner des résultats valides
            mock_evaluate.side_effect = [
                (0.5, {}, {"total_reward": 100}),
                (0.7, {}, {"total_reward": 120}),
                (0.8, {}, {"total_reward": 150}),
            ]
            
            # Créer et exécuter l'optimiseur bayésien avec un nombre minimal d'itérations
            optimizer = BayesianOptimizer(
                env_creator=self.create_test_environment,
                agent_class=SACAgent,
                param_space=param_space,
                n_episodes=2,  # Très peu d'épisodes pour le test
                eval_episodes=1,
                save_dir=self.test_dir,
                n_initial_points=2,  # Minimum pour ajuster le GP
                n_iterations=1,      # Une seule itération bayésienne
                verbose=0,           # Pas de logs
            )

            # Exécuter l'optimisation
            best_params, best_score = optimizer.bayesian_optimization()

        # Vérifier que nous avons obtenu des résultats
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_score)
        
        # Vérifier que le mock a été appelé le bon nombre de fois
        self.assertEqual(mock_evaluate.call_count, 3)  # 2 points initiaux + 1 itération
        
        # Vérifier que le meilleur score est correct
        self.assertEqual(best_score, 0.8)
        
        # Vérifier que des fichiers de résultats ont été créés
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith("bayesian_optimization_") for f in files))

    def test_helper_function_integration(self):
        """Test d'intégration de la fonction helper optimize_sac_agent_bayesian."""
        # Définir un espace de paramètres très restreint
        param_space = {
            "learning_rate": (1e-4, 5e-4),
            "batch_size": (32, 64),
            "hidden_size": (64, 128),
        }

        # Mock complet pour optimize_sac_agent_bayesian
        with patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer") as mock_optimizer_class:
            # Configurer le mock pour retourner un résultat valide
            mock_optimizer = MagicMock()
            mock_optimizer.bayesian_optimization.return_value = (
                {"learning_rate": 3e-4, "batch_size": 64, "hidden_size": 128},
                0.8
            )
            mock_optimizer_class.return_value = mock_optimizer
            
            # Exécuter l'optimisation avec la fonction helper
            best_params = optimize_sac_agent_bayesian(
                train_data=self.test_data,
                param_space=param_space,
                n_episodes=2,
                eval_episodes=1,
                save_dir=self.test_dir,
                n_initial_points=2,
                n_iterations=1,
            )

        # Vérifier que nous avons obtenu des résultats
        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
        
        # Vérifier que le mock a été appelé correctement
        mock_optimizer_class.assert_called_once()
        mock_optimizer.bayesian_optimization.assert_called_once()


if __name__ == "__main__":
    unittest.main() 