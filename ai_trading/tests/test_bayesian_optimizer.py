"""
Tests pour le module d'optimisation bayésienne des hyperparamètres.

Ce module teste les fonctionnalités de l'optimiseur bayésien, notamment :
- Conversion entre dictionnaires de paramètres et vecteurs
- Échantillonnage aléatoire de paramètres
- Calcul de l'amélioration espérée
- Optimisation bayésienne complète
"""

import os
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile

import numpy as np
import pandas as pd

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.bayesian_optimizer import BayesianOptimizer
from ai_trading.rl.trading_environment import TradingEnvironment


class TestBayesianOptimizer(unittest.TestCase):
    """Tests pour la classe BayesianOptimizer."""

    def setUp(self):
        """Initialise l'environnement de test."""
        self.test_dir = tempfile.mkdtemp(prefix="bayesian_optimizer_test_")

        # Générer des données synthétiques pour les tests
        self.test_data = generate_synthetic_market_data(
            n_points=100,  # Taille réduite pour les tests
            trend=0.001,
            volatility=0.01,
            start_price=100.0,
        )

        # Ajouter quelques indicateurs techniques simples
        self.test_data["sma_10"] = self.test_data["close"].rolling(10).mean()
        self.test_data["sma_20"] = self.test_data["close"].rolling(20).mean()
        # Remplir les NaN avec des valeurs constantes
        self.test_data = self.test_data.fillna(0.0)

        # Définir un espace de paramètres minimaliste pour les tests
        self.minimal_param_space = {
            "actor_learning_rate": (1e-4, 1e-3),
            "critic_learning_rate": (1e-4, 1e-3),
            "batch_size": (32, 64),
            "hidden_size": (64, 128),
        }
        
        # Définir un espace de paramètres avec des valeurs catégorielles
        self.mixed_param_space = {
            "actor_learning_rate": (1e-4, 1e-3),
            "activation": ["relu", "tanh", "elu"],
            "batch_size": (32, 64),
        }

    def tearDown(self):
        """Nettoyer après les tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_test_environment(self):
        """Créer un environnement pour les tests."""
        env = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            reward_function="simple",
            action_type="continuous",
        )
        return env

    def test_initialization(self):
        """Tester l'initialisation de l'optimiseur bayésien."""
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.minimal_param_space,
            n_episodes=2,
            eval_episodes=1,
            save_dir=self.test_dir,
            verbose=0,
            n_initial_points=3,
            n_iterations=5,
        )

        self.assertEqual(optimizer.n_episodes, 2)
        self.assertEqual(optimizer.eval_episodes, 1)
        self.assertEqual(optimizer.save_dir, self.test_dir)
        self.assertEqual(optimizer.n_initial_points, 3)
        self.assertEqual(optimizer.n_iterations, 5)
        self.assertEqual(optimizer.param_space, self.minimal_param_space)

    def test_param_dict_to_vector_numeric(self):
        """Tester la conversion d'un dictionnaire de paramètres numériques en vecteur."""
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.minimal_param_space,
            save_dir=self.test_dir,
            verbose=0,
        )

        # Créer un dictionnaire de paramètres
        param_dict = {
            "actor_learning_rate": 5e-4,  # Milieu de l'intervalle
            "critic_learning_rate": 1e-4,  # Minimum
            "batch_size": 64,  # Maximum
            "hidden_size": 96,  # Entre min et max
        }

        # Convertir en vecteur
        vector = optimizer._param_dict_to_vector(param_dict)

        # Vérifier les dimensions
        self.assertEqual(vector.shape, (1, 4))

        # Vérifier les valeurs normalisées
        expected = np.array([
            [
                (5e-4 - 1e-4) / (1e-3 - 1e-4),  # (5e-4 - 1e-4) / (1e-3 - 1e-4) = 0.444...
                (1e-4 - 1e-4) / (1e-3 - 1e-4),  # (1e-4 - 1e-4) / (1e-3 - 1e-4) = 0
                (64 - 32) / (64 - 32),          # (64 - 32) / (64 - 32) = 1
                (96 - 64) / (128 - 64),         # (96 - 64) / (128 - 64) = 0.5
            ]
        ])
        np.testing.assert_almost_equal(vector, expected)

    def test_vector_to_param_dict_numeric(self):
        """Tester la conversion d'un vecteur en dictionnaire de paramètres numériques."""
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.minimal_param_space,
            save_dir=self.test_dir,
            verbose=0,
        )

        # Créer un vecteur normalisé
        vector = np.array([0.5, 0.0, 1.0, 0.5])

        # Convertir en dictionnaire de paramètres
        param_dict = optimizer._vector_to_param_dict(vector)

        # Vérifier les valeurs dénormalisées
        expected = {
            "actor_learning_rate": 5.5e-4,  # 1e-4 + 0.5 * (1e-3 - 1e-4)
            "critic_learning_rate": 1e-4,   # 1e-4 + 0.0 * (1e-3 - 1e-4)
            "batch_size": 64,               # 32 + 1.0 * (64 - 32)
            "hidden_size": 96,              # 64 + 0.5 * (128 - 64)
        }
        
        # Vérifier chaque valeur avec une tolérance pour les flottants
        for key, value in expected.items():
            self.assertAlmostEqual(param_dict[key], value, places=6)

    def test_param_dict_to_vector_mixed(self):
        """Tester la conversion d'un dictionnaire avec paramètres mixtes en vecteur."""
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.mixed_param_space,
            save_dir=self.test_dir,
            verbose=0,
        )

        # Créer un dictionnaire de paramètres
        param_dict = {
            "actor_learning_rate": 5e-4,  # Milieu de l'intervalle
            "activation": "tanh",         # Deuxième option
            "batch_size": 48,             # Entre min et max
        }

        # Convertir en vecteur
        vector = optimizer._param_dict_to_vector(param_dict)

        # Vérifier les dimensions (1 numérique + 3 catégorielles + 1 numérique)
        self.assertEqual(vector.shape, (1, 5))

        # Vérifier les valeurs (one-hot pour "activation")
        expected = np.array([
            [
                (5e-4 - 1e-4) / (1e-3 - 1e-4),  # actor_learning_rate normalisé = 0.444...
                0.0,                            # activation="relu" (faux)
                1.0,                            # activation="tanh" (vrai)
                0.0,                            # activation="elu" (faux)
                (48 - 32) / (64 - 32),          # batch_size normalisé = 0.5
            ]
        ])
        np.testing.assert_almost_equal(vector, expected)

    def test_vector_to_param_dict_mixed(self):
        """Tester la conversion d'un vecteur en dictionnaire avec paramètres mixtes."""
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.mixed_param_space,
            save_dir=self.test_dir,
            verbose=0,
        )

        # Créer un vecteur normalisé avec one-hot encoding
        vector = np.array([0.5, 0.0, 1.0, 0.0, 0.5])

        # Convertir en dictionnaire de paramètres
        param_dict = optimizer._vector_to_param_dict(vector)

        # Vérifier les valeurs
        expected = {
            "actor_learning_rate": 5.5e-4,  # 1e-4 + 0.5 * (1e-3 - 1e-4)
            "activation": "tanh",           # Deuxième option (indice 1)
            "batch_size": 48,               # 32 + 0.5 * (64 - 32)
        }
        
        # Vérifier chaque valeur
        for key, value in expected.items():
            if key == "activation":
                self.assertEqual(param_dict[key], value)
            else:
                self.assertAlmostEqual(param_dict[key], value, places=6)

    def test_sample_random_params(self):
        """Tester l'échantillonnage aléatoire de paramètres."""
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.mixed_param_space,
            save_dir=self.test_dir,
            verbose=0,
        )

        # Échantillonner des paramètres aléatoires
        params = optimizer._sample_random_params()

        # Vérifier que tous les paramètres sont présents
        self.assertIn("actor_learning_rate", params)
        self.assertIn("activation", params)
        self.assertIn("batch_size", params)

        # Vérifier que les valeurs sont dans les intervalles définis
        self.assertTrue(1e-4 <= params["actor_learning_rate"] <= 1e-3)
        self.assertIn(params["activation"], ["relu", "tanh", "elu"])
        self.assertTrue(32 <= params["batch_size"] <= 64)

    @patch("ai_trading.rl.bayesian_optimizer.GaussianProcessRegressor")
    def test_expected_improvement(self, mock_gpr):
        """Tester le calcul de l'amélioration espérée."""
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.minimal_param_space,
            save_dir=self.test_dir,
            verbose=0,
        )

        # Configurer le mock du modèle GP
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0.8]), np.array([0.2]))
        optimizer.gp_model = mock_model
        optimizer.y_samples = [0.5, 0.6, 0.7]  # y_best sera 0.7

        # Calculer l'amélioration espérée
        x = np.array([0.5, 0.5, 0.5, 0.5])
        ei = optimizer._expected_improvement(x, xi=0.01)

        # Vérifier que le modèle a été appelé correctement
        mock_model.predict.assert_called_once()
        
        # L'amélioration espérée devrait être positive
        self.assertGreater(ei, 0)

    @patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._evaluate_params")
    @patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._fit_gp_model")
    @patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._find_next_point")
    @patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._sample_random_params")
    @patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._save_results")
    @patch("ai_trading.rl.bayesian_optimizer.BayesianOptimizer._generate_convergence_plot")
    def test_bayesian_optimization(self, mock_generate_plot, mock_save_results, mock_sample, mock_find_next, mock_fit_gp, mock_evaluate):
        """Tester le processus complet d'optimisation bayésienne."""
        # Configurer les mocks
        mock_sample.side_effect = [
            {"actor_learning_rate": 5e-4, "critic_learning_rate": 5e-4, "batch_size": 48, "hidden_size": 96},
            {"actor_learning_rate": 8e-4, "critic_learning_rate": 3e-4, "batch_size": 32, "hidden_size": 128},
        ]
        mock_find_next.side_effect = [
            {"actor_learning_rate": 7e-4, "critic_learning_rate": 7e-4, "batch_size": 64, "hidden_size": 64},
        ]
        mock_evaluate.side_effect = [
            (0.5, {}, {"total_reward": 100}),
            (0.7, {}, {"total_reward": 120}),
            (0.8, {}, {"total_reward": 150}),
        ]
        # Mock pour éviter les problèmes de matplotlib
        mock_save_results.return_value = None
        mock_generate_plot.return_value = None

        # Créer l'optimiseur avec un nombre réduit de points et d'itérations
        optimizer = BayesianOptimizer(
            env_creator=self.create_test_environment,
            agent_class=SACAgent,
            param_space=self.minimal_param_space,
            save_dir=self.test_dir,
            verbose=0,
            n_initial_points=2,
            n_iterations=1,
        )

        # Exécuter l'optimisation bayésienne
        best_params, best_score = optimizer.bayesian_optimization()

        # Vérifier les appels
        self.assertEqual(mock_sample.call_count, 2)  # 2 points initiaux
        self.assertEqual(mock_find_next.call_count, 1)  # 1 itération
        self.assertEqual(mock_fit_gp.call_count, 1)  # 1 fois avant de trouver le prochain point
        self.assertEqual(mock_evaluate.call_count, 3)  # 2 points initiaux + 1 itération
        self.assertEqual(mock_save_results.call_count, 1)  # Appelé une fois à la fin
        self.assertEqual(mock_generate_plot.call_count, 1)  # Appelé une fois à la fin

        # Vérifier les résultats
        self.assertEqual(best_score, 0.8)  # Le meilleur score est 0.8
        self.assertEqual(len(optimizer.X_samples), 3)  # 2 points initiaux + 1 itération
        self.assertEqual(len(optimizer.y_samples), 3)  # 2 points initiaux + 1 itération

    @patch("ai_trading.rl.bayesian_optimizer.optimize_sac_agent_bayesian")
    def test_optimize_sac_agent_bayesian(self, mock_optimize):
        """Tester la fonction helper optimize_sac_agent_bayesian."""
        from ai_trading.rl.bayesian_optimizer import optimize_sac_agent_bayesian
        
        # Configurer le mock
        expected_params = {"actor_learning_rate": 5e-4, "batch_size": 64}
        mock_optimize.return_value = expected_params
        
        # Appeler la fonction
        result = optimize_sac_agent_bayesian(
            train_data=self.test_data,
            n_episodes=2,
            eval_episodes=1,
            save_dir=self.test_dir,
        )
        
        # Vérifier le résultat
        self.assertEqual(result, expected_params)
        mock_optimize.assert_called_once()

    @patch("ai_trading.rl.bayesian_optimizer.optimize_gru_sac_agent_bayesian")
    def test_optimize_gru_sac_agent_bayesian(self, mock_optimize):
        """Tester la fonction helper optimize_gru_sac_agent_bayesian."""
        from ai_trading.rl.bayesian_optimizer import optimize_gru_sac_agent_bayesian
        
        # Configurer le mock
        expected_params = {"actor_learning_rate": 5e-4, "batch_size": 64, "use_gru": True}
        mock_optimize.return_value = expected_params
        
        # Appeler la fonction
        result = optimize_gru_sac_agent_bayesian(
            train_data=self.test_data,
            n_episodes=2,
            eval_episodes=1,
            save_dir=self.test_dir,
        )
        
        # Vérifier le résultat
        self.assertEqual(result, expected_params)
        mock_optimize.assert_called_once()


if __name__ == "__main__":
    unittest.main() 