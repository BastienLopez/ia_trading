#!/usr/bin/env python
"""
Tests unitaires pour la classe MultiPeriodTrainer
"""

import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.rl.multi_period_trainer import MultiPeriodTrainer


# Mock pour EnhancedSentimentCollector
class EnhancedSentimentCollector:
    def collect_data(self, coins=None, days=None):
        return pd.DataFrame()


class TestMultiPeriodTrainer(unittest.TestCase):
    """Tests unitaires pour la classe MultiPeriodTrainer"""

    def setUp(self):
        """Initialise l'environnement de test."""
        self.test_save_dir = INFO_RETOUR_DIR / "test" / "multi_period_trainer"
        os.makedirs(self.test_save_dir, exist_ok=True)

        # Configuration de base pour les tests
        self.symbol = "BTC"
        self.days = 30
        self.periods = [60, 240, 1440]  # 1h, 4h, 1d en minutes
        self.agent_type = "sac"
        self.use_gru = True
        self.initial_balance = 10000
        self.use_curriculum = True
        self.epochs_per_period = 2
        self.episodes_per_epoch = 2
        self.validation_ratio = 0.2
        self.include_sentiment = False

        # Sample data for mocking
        self.market_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1H"),
                "open": np.random.uniform(40000, 50000, 100),
                "high": np.random.uniform(40000, 50000, 100),
                "low": np.random.uniform(40000, 50000, 100),
                "close": np.random.uniform(40000, 50000, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )
        self.market_data.set_index("timestamp", inplace=True)

        self.sentiment_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=30, freq="1D"),
                "sentiment_score": np.random.uniform(-1, 1, 30),
                "volume": np.random.uniform(1000, 5000, 30),
            }
        )
        self.sentiment_data.set_index("timestamp", inplace=True)

        # Créer un mock pour collector et integrator qui seront utilisés dans les tests
        self.mock_collector_patcher = patch(
            "ai_trading.rl.multi_period_trainer.EnhancedMarketDataCollector"
        )
        self.mock_collector = self.mock_collector_patcher.start()
        self.mock_collector_instance = MagicMock()
        self.mock_collector.return_value = self.mock_collector_instance

        self.mock_integrator_patcher = patch(
            "ai_trading.rl.multi_period_trainer.EnhancedSentimentCollector"
        )
        self.mock_integrator = self.mock_integrator_patcher.start()
        self.mock_integrator_instance = MagicMock()
        self.mock_integrator.return_value = self.mock_integrator_instance

        # Créer un mock pour les agents et environnements
        self.mock_agent_patcher = patch("ai_trading.rl.multi_period_trainer.SACAgent")
        self.mock_agent = self.mock_agent_patcher.start()
        self.mock_agent_instance = MagicMock()
        self.mock_agent.return_value = self.mock_agent_instance

        self.mock_env_patcher = patch(
            "ai_trading.rl.multi_period_trainer.TradingEnvironment"
        )
        self.mock_env = self.mock_env_patcher.start()
        self.mock_env_instance = MagicMock()
        self.mock_env.return_value = self.mock_env_instance

        # Mock pour MultiEnvTrading
        self.mock_multi_env_patcher = patch(
            "ai_trading.rl.multi_period_trainer.MultiEnvTrading"
        )
        self.mock_multi_env = self.mock_multi_env_patcher.start()
        self.mock_multi_env_instance = MagicMock()
        self.mock_multi_env.return_value = self.mock_multi_env_instance

        # Initialiser le trainer
        self.trainer = MultiPeriodTrainer(
            symbol=self.symbol,
            days=self.days,
            periods=self.periods,
            agent_type=self.agent_type,
            use_gru=self.use_gru,
            initial_balance=self.initial_balance,
            save_dir=self.test_save_dir,
            use_curriculum=self.use_curriculum,
            epochs_per_period=self.epochs_per_period,
            episodes_per_epoch=self.episodes_per_epoch,
            validation_ratio=self.validation_ratio,
            include_sentiment=self.include_sentiment,
        )

    def tearDown(self):
        """Nettoyage après chaque test"""
        # Arrêter tous les patchers
        self.mock_collector_patcher.stop()
        self.mock_integrator_patcher.stop()
        self.mock_agent_patcher.stop()
        self.mock_env_patcher.stop()
        self.mock_multi_env_patcher.stop()

        # Supprimer le répertoire temporaire
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir)

    @patch("ai_trading.rl.multi_period_trainer.EnhancedMarketDataCollector")
    @patch("ai_trading.rl.multi_period_trainer.EnhancedSentimentCollector")
    @patch("ai_trading.rl.multi_period_trainer.TradingEnvironment")
    @patch("ai_trading.rl.multi_period_trainer.SACAgent")
    @patch("ai_trading.rl.multi_period_trainer.GRUSACAgent")
    def test_initialization(
        self, mock_gru_agent, mock_sac_agent, mock_env, mock_sentiment, mock_market
    ):
        # Configure mocks
        mock_market_instance = mock_market.return_value
        mock_market_instance.collect_historical_data.return_value = self.market_data

        mock_sentiment_instance = mock_sentiment.return_value
        mock_sentiment_instance.collect_sentiment_data.return_value = (
            self.sentiment_data
        )

        # Create trainer instance
        trainer = MultiPeriodTrainer(
            symbol=self.symbol,
            days=self.days,
            periods=self.periods,
            agent_type=self.agent_type,
            use_gru=self.use_gru,
            initial_balance=self.initial_balance,
            save_dir=self.test_save_dir,
            use_curriculum=self.use_curriculum,
            epochs_per_period=self.epochs_per_period,
            episodes_per_epoch=self.episodes_per_epoch,
            validation_ratio=self.validation_ratio,
            include_sentiment=self.include_sentiment,
        )

        # Assert initialization properties
        self.assertEqual(trainer.symbol, self.symbol)
        self.assertEqual(trainer.days, self.days)
        self.assertEqual(trainer.periods, sorted(self.periods, reverse=True))
        self.assertEqual(trainer.agent_type, self.agent_type)
        self.assertEqual(trainer.use_gru, self.use_gru)
        self.assertEqual(trainer.initial_balance, self.initial_balance)
        self.assertEqual(trainer.save_dir, self.test_save_dir)
        self.assertEqual(trainer.use_curriculum, self.use_curriculum)
        self.assertEqual(trainer.epochs_per_period, self.epochs_per_period)
        self.assertEqual(trainer.episodes_per_epoch, self.episodes_per_epoch)
        self.assertEqual(trainer.validation_ratio, self.validation_ratio)
        self.assertEqual(trainer.include_sentiment, self.include_sentiment)

    @patch("ai_trading.rl.multi_period_trainer.EnhancedMarketDataCollector")
    @patch("ai_trading.rl.multi_period_trainer.EnhancedSentimentCollector")
    def test_collect_data(self, mock_sentiment_collector, mock_market_collector):
        """Test collect_data method."""
        # Create mock data
        mock_market_data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "open": [99, 100, 101],
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "volume": [1000, 1100, 1200],
            }
        )

        mock_sentiment_data = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7], "sentiment_magnitude": [0.8, 0.9, 1.0]}
        )

        # Configure mocks
        mock_market_collector.return_value.collect_data.return_value = mock_market_data
        mock_sentiment_collector.return_value.collect_data.return_value = (
            mock_sentiment_data
        )

        # Initialize trainer
        trainer = MultiPeriodTrainer(
            symbol=self.symbol,
            days=self.days,
            periods=self.periods,
            agent_type="sac",
            use_gru=True,
        )

        # Call collect_data
        market_data, sentiment_data = trainer.collect_data()

        # Assert data is collected correctly
        self.assertIsInstance(market_data, pd.DataFrame)
        self.assertIsInstance(sentiment_data, pd.DataFrame)
        self.assertEqual(len(market_data), len(mock_market_data))
        self.assertEqual(len(sentiment_data), len(mock_sentiment_data))

    @patch("ai_trading.rl.multi_period_trainer.EnhancedMarketDataCollector")
    @patch("ai_trading.rl.multi_period_trainer.EnhancedSentimentCollector")
    @patch("ai_trading.rl.multi_period_trainer.MultiEnvTrading")
    def test_create_env(
        self, mock_env, mock_sentiment_collector, mock_market_collector
    ):
        """Test create_env method."""
        # Mock data
        mock_market_data = pd.DataFrame()
        mock_sentiment_data = pd.DataFrame()

        # Configure mocks
        mock_market_collector.return_value.collect_data.return_value = mock_market_data
        mock_sentiment_collector.return_value.collect_data.return_value = (
            mock_sentiment_data
        )
        mock_env.return_value = MagicMock()

        # Initialize trainer
        trainer = MultiPeriodTrainer(
            symbol=self.symbol,
            days=self.days,
            periods=self.periods,
            agent_type="sac",
            use_gru=True,
        )

        # Call create_env
        env = trainer.create_env(mock_market_data, mock_sentiment_data)

        # Assert env is created
        self.assertIsNotNone(env)
        mock_env.assert_called_once()

    @patch("ai_trading.rl.multi_period_trainer.EnhancedMarketDataCollector")
    @patch("ai_trading.rl.multi_period_trainer.EnhancedSentimentCollector")
    @patch("ai_trading.rl.multi_period_trainer.MultiEnvTrading")
    @pytest.mark.skip(
        reason="Paramètres incompatibles entre les agents SACAgent et NStepSACAgent"
    )
    def test_create_agent(
        self, mock_env, mock_sentiment_collector, mock_market_collector
    ):
        """Test create_agent method."""
        # Mock data and environment
        mock_market_data = pd.DataFrame()
        mock_sentiment_data = pd.DataFrame()
        mock_env_instance = MagicMock()
        mock_env.return_value = mock_env_instance

        # Configure observation space
        mock_env_instance.observation_space.shape = (10,)
        mock_env_instance.action_space.shape = (1,)

        # Configure mocks
        mock_market_collector.return_value.collect_data.return_value = mock_market_data
        mock_sentiment_collector.return_value.collect_data.return_value = (
            mock_sentiment_data
        )

        # Test with SAC agent
        trainer_sac = MultiPeriodTrainer(
            symbol=self.symbol,
            days=self.days,
            periods=self.periods,
            agent_type="sac",
            use_gru=False,
        )

        agent_sac = trainer_sac.create_agent(mock_env_instance)
        self.assertIsNotNone(agent_sac)

        # Test with n-step SAC agent
        trainer_nstep = MultiPeriodTrainer(
            symbol=self.symbol,
            days=self.days,
            periods=self.periods,
            agent_type="n_step_sac",
            use_gru=False,
        )

        agent_nstep = trainer_nstep.create_agent(mock_env_instance)
        self.assertIsNotNone(agent_nstep)

    @patch("ai_trading.rl.multi_period_trainer.EnhancedMarketDataCollector")
    @patch("ai_trading.rl.multi_period_trainer.EnhancedSentimentCollector")
    def test_prepare_datasets(self, mock_sentiment_collector, mock_market_collector):
        """Test prepare_datasets method."""
        # Create mock data with a specific length
        data_length = 100
        mock_market_data = pd.DataFrame(
            {
                "close": np.random.rand(data_length),
                "date": pd.date_range(start="2022-01-01", periods=data_length),
            }
        )
        mock_sentiment_data = pd.DataFrame(
            {
                "sentiment_score": np.random.rand(data_length),
                "date": pd.date_range(start="2022-01-01", periods=data_length),
            }
        )

        # Configure mocks
        mock_market_collector.return_value.collect_data.return_value = mock_market_data
        mock_sentiment_collector.return_value.collect_data.return_value = (
            mock_sentiment_data
        )

        # Initialize trainer
        trainer = MultiPeriodTrainer(
            symbol=self.symbol,
            days=self.days,
            periods=self.periods,
            agent_type="sac",
            use_gru=True,
        )

        # Set validation ratio
        validation_ratio = 0.2

        # Call prepare_datasets
        train_market, train_sentiment, val_market, val_sentiment = (
            trainer.prepare_datasets(
                mock_market_data, mock_sentiment_data, validation_ratio
            )
        )

        # Expected lengths
        expected_train_length = int(data_length * (1 - validation_ratio))
        expected_val_length = data_length - expected_train_length

        # Assert dataset lengths
        self.assertEqual(len(train_market), expected_train_length)
        self.assertEqual(len(train_sentiment), expected_train_length)
        self.assertEqual(len(val_market), expected_val_length)
        self.assertEqual(len(val_sentiment), expected_val_length)

    @patch("ai_trading.rl.multi_period_trainer.TradingEnvironment")
    @patch("ai_trading.rl.multi_period_trainer.os.path.exists")
    @pytest.mark.skip(reason="Méthode train_period non complètement implémentée")
    def test_train_period(self, mock_exists, mock_env_class):
        """Tester l'entraînement sur une période spécifique"""
        # Configurer les mocks
        period = "1d"
        dataset = pd.DataFrame(
            {
                "open": np.random.rand(100),
                "high": np.random.rand(100),
                "low": np.random.rand(100),
                "close": np.random.rand(100),
                "volume": np.random.rand(100),
                "sentiment_score": np.random.rand(100),
            }
        )
        self.trainer.datasets = {
            period: {"train": dataset, "validation": dataset.copy()}
        }

        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.reset.return_value = (np.zeros(20), {})  # état initial factice
        mock_env.step.return_value = (
            np.zeros(20),
            1.0,
            False,
            False,
            {},
        )  # résultat de step factice

        mock_exists.return_value = False  # Pas de poids précédents

        # Mock pour la méthode create_agent
        self.trainer.create_agent = MagicMock(return_value=self.mock_agent_instance)
        self.trainer.create_environment = MagicMock(return_value=mock_env)

        # Mock pour load_weights
        self.mock_agent_instance.load_weights = MagicMock()

        # Exécuter la méthode à tester avec un nombre réduit d'époques pour le test
        self.trainer.epochs_per_period = 2
        self.trainer.episodes_per_epoch = 2
        result = self.trainer.train_period(period)

        # Vérifier que l'agent a été créé et entraîné
        self.trainer.create_agent.assert_called_once()
        self.trainer.create_environment.assert_called()

        # Vérifier que l'entraînement a été effectué (reset et step ont été appelés)
        self.assertTrue(mock_env.reset.called)
        self.assertTrue(mock_env.step.called)

        # Vérifier que les performances ont été enregistrées
        self.assertIn("train_rewards", result)
        self.assertIn("val_rewards", result)
        self.assertEqual(len(result["train_rewards"]), 2)  # 2 époques

    @pytest.mark.skip(reason="Test requires multi_period_trainer implementation")
    def test_train_all_periods(self):
        """Tester l'entraînement sur toutes les périodes"""
        # Mock pour train_period
        self.trainer.train_period = MagicMock(
            return_value={"train_rewards": [0.1, 0.2], "val_rewards": [0.05, 0.15]}
        )

        # Exécuter la méthode à tester
        result = self.trainer.train_all_periods()

        # Vérifier que train_period a été appelé pour chaque période
        self.assertEqual(self.trainer.train_period.call_count, len(self.periods))

        # Vérifier que les performances ont été enregistrées pour chaque période
        for period in self.periods:
            self.assertIn(period, result)
            self.assertIn("train_rewards", result[period])
            self.assertIn("val_rewards", result[period])

    @pytest.mark.skip(reason="Test requires multi_period_trainer implementation")
    def test_evaluate(self):
        """Tester l'évaluation de l'agent sur toutes les périodes"""
        # Configurer les mocks pour les données et l'agent
        for period in self.periods:
            self.trainer.data[period] = {
                "market": pd.DataFrame(
                    {
                        "open": np.random.rand(100),
                        "high": np.random.rand(100),
                        "low": np.random.rand(100),
                        "close": np.random.rand(100),
                        "volume": np.random.rand(100),
                    }
                ),
                "sentiment": pd.DataFrame({"sentiment_score": np.random.rand(100)}),
            }

        # Mock pour la création d'agent et d'environnement
        self.trainer.create_agent = MagicMock(return_value=self.mock_agent_instance)
        self.trainer.create_environment = MagicMock(return_value=self.mock_env_instance)

        # Configurer le comportement de l'environnement
        self.mock_env_instance.reset.return_value = (
            np.zeros(20),
            {},
        )  # état initial factice
        self.mock_env_instance.step.return_value = (
            np.zeros(20),
            1.0,
            False,
            False,
            {},
        )  # résultat de step factice

        # Exécuter la méthode à tester
        result = self.trainer.evaluate()

        # Vérifier que l'évaluation a été effectuée pour chaque période
        self.assertEqual(len(result), len(self.periods))
        for period in self.periods:
            self.assertIn(period, result)
            self.assertIn("avg_reward", result[period])
            self.assertIn("avg_return", result[period])
            self.assertIn("std_return", result[period])

    def test_save_load_agent(self):
        """Tester la sauvegarde et le chargement de l'agent"""
        # Configurer les mocks
        self.trainer.current_agent = self.mock_agent_instance
        self.mock_agent_instance.save_weights = MagicMock()
        self.mock_agent_instance.load_weights = MagicMock()

        # Tester la sauvegarde
        path = self.trainer.save_current_agent(custom_name="test_agent")
        self.mock_agent_instance.save_weights.assert_called_once()
        self.assertTrue("test_agent" in path)

        # Tester le chargement
        self.trainer.load_agent(path)
        self.mock_agent_instance.load_weights.assert_called_once_with(path)


if __name__ == "__main__":
    unittest.main()
