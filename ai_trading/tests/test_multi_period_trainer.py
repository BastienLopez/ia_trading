import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from ai_trading.rl.multi_period_trainer import MultiPeriodTrainer
from ai_trading.utils.constants import INFO_RETOUR_DIR

class TestMultiPeriodTrainer(unittest.TestCase):
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
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1h"),
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
    
        # Créer un mock pour collector et text_preprocessor qui seront utilisés dans les tests
        self.mock_collector_patcher = patch(
            "ai_trading.rl.multi_period_trainer.EnhancedDataCollector"
        )
        self.mock_collector = self.mock_collector_patcher.start()
        self.mock_collector_instance = MagicMock()
        self.mock_collector.return_value = self.mock_collector_instance
    
        self.mock_text_preprocessor_patcher = patch(
            "ai_trading.rl.multi_period_trainer.EnhancedTextDataPreprocessor"
        )
        self.mock_text_preprocessor = self.mock_text_preprocessor_patcher.start()
        self.mock_text_preprocessor_instance = MagicMock()
        self.mock_text_preprocessor.return_value = self.mock_text_preprocessor_instance
    
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
        """Nettoyage après les tests."""
        self.mock_collector_patcher.stop()
        self.mock_text_preprocessor_patcher.stop()
        self.mock_agent_patcher.stop()
        self.mock_env_patcher.stop()

    def test_initialization(self):
        """Test l'initialisation du MultiPeriodTrainer."""
        self.assertEqual(self.trainer.symbol, self.symbol)
        self.assertEqual(self.trainer.days, self.days)
        self.assertEqual(self.trainer.periods, self.periods)
        self.assertEqual(self.trainer.agent_type, self.agent_type)
        self.assertEqual(self.trainer.use_gru, self.use_gru)
        self.assertEqual(self.trainer.initial_balance, self.initial_balance)
        self.assertEqual(self.trainer.save_dir, self.test_save_dir)
        self.assertEqual(self.trainer.use_curriculum, self.use_curriculum)
        self.assertEqual(self.trainer.epochs_per_period, self.epochs_per_period)
        self.assertEqual(self.trainer.episodes_per_epoch, self.episodes_per_epoch)
        self.assertEqual(self.trainer.validation_ratio, self.validation_ratio)
        self.assertEqual(self.trainer.include_sentiment, self.include_sentiment)

    def test_collect_data(self):
        """Test la collecte des données."""
        # Configurer les mocks
        self.mock_collector_instance.get_merged_price_data.return_value = self.market_data
        self.mock_collector_instance.get_social_data.return_value = []
        self.mock_text_preprocessor_instance.preprocess_social_data.return_value = self.sentiment_data

        # Test sans sentiment
        market_data, sentiment_data = self.trainer.collect_data()
        self.assertIsNotNone(market_data)
        self.assertIsNone(sentiment_data)  # Doit être None car include_sentiment est False
        self.mock_collector_instance.get_merged_price_data.assert_called_once()
        self.mock_collector_instance.get_social_data.assert_not_called()
        self.mock_text_preprocessor_instance.preprocess_social_data.assert_not_called()

        # Test avec sentiment
        self.trainer.include_sentiment = True
        market_data, sentiment_data = self.trainer.collect_data()
        self.assertIsNotNone(market_data)
        self.assertIsNotNone(sentiment_data)
        self.mock_collector_instance.get_social_data.assert_called_once()
        self.mock_text_preprocessor_instance.preprocess_social_data.assert_called_once()

    def test_create_env(self):
        """Test la création de l'environnement."""
        # Configurer les mocks
        self.mock_env_instance.reset.return_value = np.zeros(10)
        self.mock_env_instance.step.return_value = (np.zeros(10), 0, False, {})

        # Appeler la méthode
        env = self.trainer.create_env(self.market_data, self.sentiment_data)

        # Vérifier les résultats
        self.assertIsNotNone(env)
        self.mock_env.assert_called_once()

    def test_prepare_datasets(self):
        """Test la préparation des datasets."""
        # Appeler la méthode
        train_data, val_data = self.trainer.prepare_datasets(self.market_data)

        # Vérifier les résultats
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(val_data)
        self.assertLess(len(val_data), len(train_data))

    def test_save_load_agent(self):
        """Test la sauvegarde et le chargement de l'agent."""
        # Configurer les mocks
        self.mock_agent_instance.save.return_value = None
        self.mock_agent_instance.load.return_value = None

        # Appeler les méthodes
        self.trainer.save_agent(self.mock_agent_instance, "test_agent")
        loaded_agent = self.trainer.load_agent("test_agent")

        # Vérifier les résultats
        self.mock_agent_instance.save.assert_called_once()
        self.mock_agent_instance.load.assert_called_once()
        self.assertIsNotNone(loaded_agent)

if __name__ == "__main__":
    unittest.main() 