import logging
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from web3 import Web3

from ..rl.multi_asset_trading_environment import MultiAssetTradingEnvironment
from ..utils.alternative_data_collector import AlternativeDataCollector

logger = logging.getLogger(__name__)


class TestAlternativeDataCollector(unittest.TestCase):
    def setUp(self):
        """Prépare l'environnement de test."""
        self.eth_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        self.keywords = ["bitcoin", "ethereum"]

        # Mock des réponses Twitter
        self.mock_tweet = Mock()
        self.mock_tweet.text = "Bitcoin is looking very bullish today! #crypto"

        # Mock des réponses Reddit
        self.mock_post = Mock()
        self.mock_post.title = "Bitcoin Analysis"
        self.mock_post.selftext = "The market is showing strong bullish signals"

        # Mock des transactions Ethereum
        self.mock_transaction = {
            "from": self.eth_address.lower(),
            "to": "0x1234567890123456789012345678901234567890",
            "value": 1000000000000000000,  # 1 ETH
            "hash": "0x123456789abcdef",
            "blockNumber": 1000,
        }

    def _setup_web3_mock(self, mock_web3_class):
        """Configure le mock Web3 pour les tests."""
        mock_w3 = MagicMock()
        mock_w3.is_connected.return_value = True
        mock_w3.is_address.return_value = True

        # Configuration du provider
        mock_provider = MagicMock()
        mock_web3_class.HTTPProvider = Mock(return_value=mock_provider)
        mock_web3_class.return_value = mock_w3

        # Configuration de eth
        mock_eth = MagicMock()
        mock_eth.get_balance.return_value = 2000000000000000000  # 2 ETH
        mock_eth.get_transaction_count.return_value = 100
        mock_eth.block_number = 1000

        mock_block = MagicMock()
        mock_block.transactions = [self.mock_transaction]
        mock_eth.get_block.return_value = mock_block

        # Configuration des méthodes de conversion
        mock_w3.from_wei.return_value = 2.0  # 2 ETH
        mock_w3.to_wei = Web3.to_wei
        mock_w3.eth = mock_eth

        return mock_w3

    @patch("web3.Web3")
    def test_get_onchain_metrics(self, mock_web3_class):
        """Teste la récupération des métriques on-chain."""
        # Configuration du mock Web3
        mock_w3 = self._setup_web3_mock(mock_web3_class)

        # Test
        collector = AlternativeDataCollector()
        collector.w3 = mock_w3  # Définir directement le mock
        metrics = collector.get_onchain_metrics(self.eth_address)

        # Vérifications
        self.assertIn("balance", metrics)
        self.assertIn("transaction_count", metrics)
        self.assertIn("average_volume", metrics)
        self.assertEqual(metrics["transaction_count"], 100)
        self.assertEqual(metrics["balance"], 2.0)  # 2 ETH

    @patch("tweepy.API")
    @patch("praw.Reddit")
    def test_analyze_social_sentiment(self, mock_reddit, mock_twitter_api):
        """Teste l'analyse de sentiment des réseaux sociaux."""
        # Configuration des mocks Twitter
        mock_twitter = Mock()
        mock_twitter.search_tweets.return_value = [self.mock_tweet]
        mock_twitter_api.return_value = mock_twitter

        # Configuration des mocks Reddit
        mock_subreddit = Mock()
        mock_subreddit.search.return_value = [self.mock_post]
        mock_reddit.return_value.subreddit.return_value = mock_subreddit

        # Test
        collector = AlternativeDataCollector(
            twitter_api_key="test",
            twitter_api_secret="test",
            reddit_client_id="test",
            reddit_client_secret="test",
        )

        sentiments = collector.analyze_social_sentiment(self.keywords[0])

        # Vérifications
        self.assertIn("twitter_avg_polarity", sentiments)
        self.assertIn("reddit_avg_polarity", sentiments)
        self.assertIn("twitter_volume", sentiments)
        self.assertIn("reddit_volume", sentiments)

    @patch("tweepy.API")
    @patch("praw.Reddit")
    @patch("web3.Web3")
    def test_collect_alternative_data(self, mock_web3_class, mock_reddit, mock_twitter):
        """Teste la collecte complète des données alternatives."""
        try:
            # Configuration des mocks pour Web3
            mock_w3 = self._setup_web3_mock(mock_web3_class)

            # Configuration des mocks pour Twitter et Reddit
            mock_twitter_api = Mock()
            mock_twitter_api.search_tweets.return_value = [self.mock_tweet]
            mock_twitter.return_value = mock_twitter_api

            mock_subreddit = Mock()
            mock_subreddit.search.return_value = [self.mock_post]
            mock_reddit.return_value.subreddit.return_value = mock_subreddit

            # Test avec une durée très courte
            collector = AlternativeDataCollector(
                twitter_api_key="test",
                twitter_api_secret="test",
                reddit_client_id="test",
                reddit_client_secret="test",
            )
            collector.w3 = mock_w3  # Définir directement le mock

            logger.info("Début de la collecte des données de test...")
            data = collector.collect_alternative_data(
                addresses=[self.eth_address],
                keywords=self.keywords,
                duration_minutes=0.1,  # 6 secondes
                interval_seconds=1,
            )
            logger.info("Fin de la collecte des données de test")

            # Vérifications
            self.assertIsInstance(data, pd.DataFrame)
            self.assertTrue(len(data) > 0)
            logger.info(f"DataFrame créé avec {len(data)} lignes")

        except Exception as e:
            logger.error(f"Erreur dans test_collect_alternative_data: {e}")
            raise


class TestDiversificationReward(unittest.TestCase):
    def setUp(self):
        """Prépare l'environnement de test pour la récompense de diversification."""
        # Créer des données de test
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        self.data_dict = {
            "BTC": pd.DataFrame(
                {
                    "close": np.random.normal(50000, 1000, 100),
                    "volume": np.random.normal(1000, 100, 100),
                },
                index=dates,
            ),
            "ETH": pd.DataFrame(
                {
                    "close": np.random.normal(3000, 100, 100),
                    "volume": np.random.normal(500, 50, 100),
                },
                index=dates,
            ),
            "XRP": pd.DataFrame(
                {
                    "close": np.random.normal(1, 0.1, 100),
                    "volume": np.random.normal(10000, 1000, 100),
                },
                index=dates,
            ),
        }

        # Initialiser l'environnement
        self.env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            initial_balance=10000.0,
            window_size=10,
            reward_function="diversification",
        )

        # Configurer les paramètres de diversification
        self.env.diversification_weight = 0.5
        self.env.min_diversification_factor = 0.2
        self.env.max_diversification_factor = 2.0
        self.env.correlation_penalty_weight = 1.0

    def test_diversification_reward_equal_weights(self):
        """Teste la récompense avec une allocation équilibrée."""
        # Simuler une allocation équilibrée
        self.env.crypto_holdings = {
            "BTC": 0.1,  # Valeurs similaires en USD
            "ETH": 1.67,
            "XRP": 5000,
        }

        reward = self.env._diversification_reward(1.0)
        metrics = self.env.last_diversification_metrics

        # Vérifications
        self.assertGreater(reward, 1.0)  # La récompense devrait être amplifiée
        self.assertGreater(
            metrics["diversification_index"], 0.6
        )  # Bonne diversification
        self.assertLess(metrics["correlation_penalty"], 1.0)

    def test_diversification_reward_concentrated(self):
        """Teste la récompense avec une allocation concentrée."""
        # Simuler une allocation concentrée sur un actif
        self.env.crypto_holdings = {"BTC": 0.2, "ETH": 0.0, "XRP": 0.0}

        reward = self.env._diversification_reward(1.0)
        metrics = self.env.last_diversification_metrics

        # Vérifications
        self.assertLess(metrics["diversification_index"], 0.5)  # Faible diversification
        self.assertGreaterEqual(reward, 1.0 * (1 + self.env.min_diversification_factor))
        self.assertLessEqual(reward, 1.0 * (1 + self.env.max_diversification_factor))

    def test_diversification_reward_empty_portfolio(self):
        """Teste la récompense avec un portefeuille vide."""
        self.env.crypto_holdings = {"BTC": 0.0, "ETH": 0.0, "XRP": 0.0}

        base_reward = 1.0
        reward = self.env._diversification_reward(base_reward)

        # La récompense devrait être égale à la récompense de base
        self.assertEqual(reward, base_reward)

    def test_diversification_reward_correlation_penalty(self):
        """Teste l'impact de la pénalité de corrélation."""
        # Simuler des données fortement corrélées
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        base_prices = np.random.normal(100, 10, 100)

        self.env.data_dict = {
            "BTC": pd.DataFrame(
                {
                    "close": base_prices * 1.0,
                    "volume": np.random.normal(1000, 100, 100),
                },
                index=dates,
            ),
            "ETH": pd.DataFrame(
                {
                    "close": base_prices * 1.1,  # Forte corrélation avec BTC
                    "volume": np.random.normal(500, 50, 100),
                },
                index=dates,
            ),
            "XRP": pd.DataFrame(
                {
                    "close": base_prices * 0.9,  # Forte corrélation avec BTC
                    "volume": np.random.normal(10000, 1000, 100),
                },
                index=dates,
            ),
        }

        # Allocation équilibrée
        self.env.crypto_holdings = {"BTC": 0.1, "ETH": 1.67, "XRP": 5000}

        reward = self.env._diversification_reward(1.0)
        metrics = self.env.last_diversification_metrics

        # La pénalité de corrélation devrait être élevée
        self.assertGreater(metrics["correlation_penalty"], 0.5)
        self.assertLess(reward, 1.0 * (1 + self.env.max_diversification_factor))

    def test_diversification_reward_parameter_bounds(self):
        """Teste les limites des paramètres de diversification."""
        # Tester avec différentes valeurs de poids
        test_weights = [0.1, 0.5, 1.0, 2.0]

        for weight in test_weights:
            self.env.diversification_weight = weight
            self.env.crypto_holdings = {"BTC": 0.1, "ETH": 1.67, "XRP": 5000}

            reward = self.env._diversification_reward(1.0)
            metrics = self.env.last_diversification_metrics

            # Vérifier que le facteur reste dans les limites
            self.assertGreaterEqual(
                metrics["diversification_factor"], self.env.min_diversification_factor
            )
            self.assertLessEqual(
                metrics["diversification_factor"], self.env.max_diversification_factor
            )


if __name__ == "__main__":
    unittest.main()
