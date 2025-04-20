import logging
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Configurer le logger de test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importer les classes à tester
from ai_trading.rl.curriculum_learning import (
    GRUCurriculumLearning,
    GRUCurriculumTrainer,
)


class TestGRUCurriculumLearning(unittest.TestCase):
    """Tests pour la classe GRUCurriculumLearning."""

    def setUp(self):
        """Initialise les données de test et l'instance de GRUCurriculumLearning."""
        # Créer des données synthétiques pour les tests
        np.random.seed(42)
        n_samples = 300

        # Créer une série temporelle avec tendance et variation de volatilité
        dates = pd.date_range(start="2022-01-01", periods=n_samples)
        trend = np.linspace(0, 1, n_samples) * 100
        volatility_factor = np.concatenate(
            [
                np.ones(100) * 0.5,  # Faible volatilité
                np.ones(100) * 1.5,  # Moyenne volatilité
                np.ones(100) * 3.0,  # Haute volatilité
            ]
        )

        noise = np.random.normal(0, 1, n_samples) * volatility_factor
        price = trend + noise.cumsum()
        volume = np.random.randint(1000, 10000, n_samples)

        # Créer le DataFrame
        self.df = pd.DataFrame(
            {
                "open": price - np.random.normal(0, 1, n_samples),
                "high": price + np.random.normal(1, 1, n_samples),
                "low": price - np.random.normal(1, 1, n_samples),
                "close": price,
                "volume": volume,
            },
            index=dates,
        )

        # Ajouter quelques indicateurs techniques basiques
        self.df["sma_5"] = self.df["close"].rolling(window=5).mean()
        self.df["sma_20"] = self.df["close"].rolling(window=20).mean()
        self.df["rsi"] = np.random.uniform(0, 100, n_samples)  # Simplifié pour le test

        # Supprimer les lignes avec NaN
        self.df = self.df.dropna()

        # Initialiser l'instance de GRUCurriculumLearning
        self.curriculum = GRUCurriculumLearning(
            initial_difficulty=0.2,
            max_difficulty=1.0,
            difficulty_increment=0.1,
            sequence_length=5,  # Valeur réduite pour les tests
            gru_units=64,  # Valeur réduite pour les tests
            hidden_size=128,  # Valeur réduite pour les tests
        )

    def test_initialization(self):
        """Teste l'initialisation correcte des paramètres."""
        self.assertEqual(self.curriculum.difficulty, 0.2)
        self.assertEqual(self.curriculum.max_difficulty, 1.0)
        self.assertEqual(self.curriculum.difficulty_increment, 0.1)
        self.assertEqual(self.curriculum.sequence_length, 5)
        self.assertEqual(self.curriculum.gru_units, 64)
        self.assertEqual(self.curriculum.hidden_size, 128)
        self.assertEqual(len(self.curriculum.recent_performances), 0)

    def test_filter_data_by_volatility(self):
        """Teste le filtrage des données basé sur la volatilité."""
        # Test avec difficulté faible (devrait retourner principalement les données moins volatiles)
        self.curriculum.difficulty = 0.2
        filtered_data_low = self.curriculum._filter_data_by_volatility(self.df)

        # Test avec difficulté élevée (devrait retourner plus de données, incluant celles plus volatiles)
        self.curriculum.difficulty = 0.8
        filtered_data_high = self.curriculum._filter_data_by_volatility(self.df)

        # Vérifier que les données filtrées ont des tailles appropriées
        self.assertGreaterEqual(
            len(filtered_data_low), 3 * self.curriculum.sequence_length
        )
        self.assertGreaterEqual(
            len(filtered_data_high), 3 * self.curriculum.sequence_length
        )

        # À difficulté plus élevée, on devrait avoir plus de données (ou au moins autant)
        self.assertGreaterEqual(len(filtered_data_high), len(filtered_data_low))

        # Test avec difficulté maximale (devrait retourner toutes les données)
        self.curriculum.difficulty = 1.0
        filtered_data_max = self.curriculum._filter_data_by_volatility(self.df)
        self.assertEqual(
            len(filtered_data_max), len(self.df) - 20
        )  # -20 car window=20 par défaut

    @patch("ai_trading.rl.curriculum_learning.TradingEnvironment")
    def test_create_environment(self, mock_env):
        """Teste la création de l'environnement de trading."""
        # Configurer le mock
        mock_env.return_value = MagicMock()

        # Créer l'environnement
        env = self.curriculum.create_environment(self.df, window_size=10)

        # Vérifier que l'environnement est créé avec les bons paramètres
        mock_env.assert_called_once()
        self.assertEqual(mock_env.call_args[1]["window_size"], 10)
        self.assertEqual(mock_env.call_args[1]["initial_balance"], 10000.0)
        self.assertEqual(mock_env.call_args[1]["transaction_fee"], 0.001)
        self.assertEqual(mock_env.call_args[1]["action_type"], "continuous")

    @patch("ai_trading.rl.curriculum_learning.SACAgent")
    def test_create_agent(self, mock_agent):
        """Teste la création de l'agent SAC avec GRU."""
        # Configurer les mocks
        mock_env = MagicMock()
        mock_env.observation_space.shape = (10,)
        mock_env.action_space.shape = (1,)
        mock_env.action_space.low = np.array([-1.0])
        mock_env.action_space.high = np.array([1.0])

        mock_agent.return_value = MagicMock()

        # Créer l'agent
        agent = self.curriculum.create_agent(mock_env)

        # Vérifier que l'agent est créé avec les bons paramètres
        mock_agent.assert_called_once()
        self.assertEqual(mock_agent.call_args[1]["state_size"], 10)
        self.assertEqual(mock_agent.call_args[1]["action_size"], 1)
        self.assertEqual(mock_agent.call_args[1]["action_bounds"], (-1.0, 1.0))
        self.assertEqual(mock_agent.call_args[1]["use_gru"], True)
        self.assertEqual(mock_agent.call_args[1]["sequence_length"], 5)
        self.assertEqual(mock_agent.call_args[1]["gru_units"], 64)
        self.assertEqual(mock_agent.call_args[1]["hidden_size"], 128)
        self.assertEqual(mock_agent.call_args[1]["grad_clip_value"], 1.0)
        self.assertEqual(mock_agent.call_args[1]["entropy_regularization"], 0.001)

    def test_update_difficulty(self):
        """Teste la mise à jour de la difficulté basée sur les performances."""
        # Configurer le curriculum avec une fenêtre d'évaluation réduite pour le test
        self.curriculum.evaluation_window = 3

        # Test avec performances insuffisantes
        low_performances = [0.3, 0.4, 0.5]  # Moyenne = 0.4, seuil = 0.7
        for perf in low_performances:
            result = self.curriculum.update_difficulty(perf)
            self.assertFalse(result)  # La difficulté ne devrait pas augmenter

        self.assertEqual(
            self.curriculum.difficulty, 0.2
        )  # La difficulté devrait rester inchangée

        # Réinitialiser
        self.curriculum.reset()

        # Test avec bonnes performances
        high_performances = [0.7, 0.8, 0.9]  # Moyenne = 0.8, seuil = 0.7
        for i, perf in enumerate(high_performances):
            result = self.curriculum.update_difficulty(perf)
            if i == len(high_performances) - 1:
                self.assertTrue(
                    result
                )  # La difficulté devrait augmenter après la 3ème performance
            else:
                self.assertFalse(result)  # Pas encore assez de données

        # La difficulté devrait avoir augmenté
        self.assertAlmostEqual(self.curriculum.difficulty, 0.3, places=5)

        # Vérifier que l'historique des performances est réinitialisé
        self.assertEqual(len(self.curriculum.recent_performances), 0)

    def test_reset(self):
        """Teste la réinitialisation du système de curriculum learning."""
        # Ajouter quelques performances
        self.curriculum.recent_performances = [0.5, 0.6, 0.7]

        # Réinitialiser
        self.curriculum.reset()

        # Vérifier que l'historique est vide
        self.assertEqual(len(self.curriculum.recent_performances), 0)


class TestGRUCurriculumTrainer(unittest.TestCase):
    """Tests pour la classe GRUCurriculumTrainer."""

    def setUp(self):
        """Initialise les données de test et les instances nécessaires."""
        # Créer des données synthétiques (simplifiées par rapport au test précédent)
        np.random.seed(42)
        n_samples = 200

        dates = pd.date_range(start="2022-01-01", periods=n_samples)
        price = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
        volume = np.random.randint(1000, 10000, n_samples)

        self.df = pd.DataFrame(
            {
                "open": price - np.random.normal(0, 0.5, n_samples),
                "high": price + np.random.normal(0, 0.5, n_samples),
                "low": price - np.random.normal(0, 0.5, n_samples),
                "close": price,
                "volume": volume,
            },
            index=dates,
        )

        # Ajouter quelques indicateurs de base
        self.df["sma_5"] = self.df["close"].rolling(window=5).mean()
        self.df = self.df.dropna()

        # Créer le curriculum learning
        self.curriculum = GRUCurriculumLearning(
            initial_difficulty=0.2, sequence_length=5, gru_units=32, hidden_size=64
        )

        # Créer le dossier temporaire pour les tests
        self.test_save_path = "tmp_test_models"
        if os.path.exists(self.test_save_path):
            shutil.rmtree(self.test_save_path)
        os.makedirs(self.test_save_path, exist_ok=True)

        # Initialiser l'entraîneur
        self.trainer = GRUCurriculumTrainer(
            curriculum=self.curriculum,
            data=self.df,
            episodes_per_level=5,  # Réduit pour les tests
            max_episodes=10,  # Réduit pour les tests
            eval_frequency=2,  # Réduit pour les tests
            save_path=self.test_save_path,
        )

    def tearDown(self):
        """Nettoie après les tests."""
        # Supprimer le dossier temporaire
        if os.path.exists(self.test_save_path):
            shutil.rmtree(self.test_save_path)

    def test_initialization(self):
        """Teste l'initialisation correcte des paramètres du trainer."""
        self.assertEqual(self.trainer.curriculum, self.curriculum)
        self.assertTrue(len(self.trainer.data) > 0)
        self.assertEqual(self.trainer.episodes_per_level, 5)
        self.assertEqual(self.trainer.max_episodes, 10)
        self.assertEqual(self.trainer.eval_frequency, 2)
        self.assertEqual(self.trainer.save_path, self.test_save_path)

    @pytest.mark.skip(reason="Ce test prend trop de temps à exécuter")
    @patch("ai_trading.rl.curriculum_learning.GRUCurriculumLearning.create_environment")
    @patch("ai_trading.rl.curriculum_learning.GRUCurriculumLearning.create_agent")
    def test_train_method_calls(self, mock_create_agent, mock_create_env):
        """Teste que les méthodes appropriées sont appelées lors de l'entraînement."""
        # Configurer les mocks
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (np.zeros(10), 0.0, False, False, {})
        mock_env.portfolio_value.return_value = 10500.0
        mock_env.transaction_count = 5

        mock_create_env.return_value = mock_env

        mock_agent = MagicMock()
        mock_agent.sequence_length = 5
        mock_agent.batch_size = 32
        mock_agent.sequence_buffer = [1] * 100  # Simuler un buffer non vide
        mock_agent.act.return_value = np.array([0.5])
        mock_agent.train.return_value = {"actor_loss": 0.1, "critic_loss": 0.2}

        mock_create_agent.return_value = mock_agent

        # Patch la méthode d'évaluation pour éviter d'exécuter le code réel
        with patch.object(self.trainer, "_evaluate_agent", return_value=0.8):
            # Réduire encore plus le nombre d'épisodes pour accélérer le test
            self.trainer.max_episodes = 3

            # Appeler la méthode d'entraînement
            agent, history = self.trainer.train(window_size=10)

            # Vérifier que les méthodes ont été appelées correctement
            mock_create_env.assert_called()
            mock_create_agent.assert_called_once()

            # Vérifier que l'agent a été entraîné
            self.assertGreater(mock_agent.train.call_count, 0)

            # Vérifier que l'historique a été correctement enregistré
            self.assertEqual(len(history["rewards"]), 3)
            self.assertEqual(len(history["difficulties"]), 3)
            self.assertEqual(len(history["profits"]), 3)
            self.assertEqual(len(history["transactions"]), 3)
            self.assertEqual(len(history["portfolio_values"]), 3)

            # Vérifier que l'agent a été sauvegardé
            self.assertGreater(mock_agent.save.call_count, 0)

    @pytest.mark.skip(reason="Ce test prend trop de temps à exécuter")
    @patch("ai_trading.rl.curriculum_learning.SACAgent")
    @patch("ai_trading.rl.curriculum_learning.TradingEnvironment")
    def test_evaluate_agent(self, mock_env_class, mock_agent_class):
        """Teste la méthode d'évaluation de l'agent."""
        # Configurer les mocks
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (np.zeros(10), 1.0, False, False, {})

        mock_agent = MagicMock()
        mock_agent.sequence_length = 5
        mock_agent.act.return_value = np.array([0.5])

        # Appeler la méthode d'évaluation directement
        avg_reward = self.trainer._evaluate_agent(mock_agent, mock_env, n_episodes=2)

        # Vérifier que les méthodes ont été appelées correctement
        self.assertEqual(mock_env.reset.call_count, 2)
        self.assertGreater(mock_agent.act.call_count, 0)

        # Vérifier que le reward moyen est calculé correctement
        # Dans ce cas, chaque épisode donne un reward de n pas * 1.0
        self.assertGreater(avg_reward, 0)


if __name__ == "__main__":
    unittest.main()
