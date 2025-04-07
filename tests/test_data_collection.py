"""
Tests unitaires pour les modules de collecte de données.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from ai_trading.utils.data_collector import CryptoDataCollector, NewsCollector, SocialMediaCollector

class TestCryptoDataCollector(unittest.TestCase):
    """Tests pour la classe CryptoDataCollector."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.collector = CryptoDataCollector()
    
    @patch('ai_trading.utils.data_collector.Client')
    def test_get_historical_klines_binance(self, mock_client):
        """Test de la récupération des données historiques Binance."""
        # Mock des données
        mock_klines = [
            [
                1499040000000,      # Timestamp
                "8100.0",           # Open
                "8200.0",           # High
                "8000.0",           # Low
                "8150.0",           # Close
                "100.0",            # Volume
                1499644799999,      # Close time
                "1000.0",           # Quote volume
                100,                # Trades
                "50.0",             # Taker buy base
                "400.0",            # Taker buy quote
                "0"                 # Ignore
            ]
        ]
        
        mock_client.return_value.get_historical_klines.return_value = mock_klines
        
        # Test
        df = self.collector.get_historical_klines(
            'BTCUSDT',
            '1d',
            '2024-01-01',
            source='binance'
        )
        
        # Vérifications
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['close'].iloc[0], 8150.0)
        self.assertTrue('volume' in df.columns)
    
    @patch('ai_trading.utils.data_collector.CoinGeckoAPI')
    def test_get_historical_klines_coingecko(self, mock_coingecko):
        """Test de la récupération des données historiques CoinGecko."""
        # Mock des données
        mock_data = {
            'prices': [[1499040000000, 8150.0]],
            'total_volumes': [[1499040000000, 1000.0]],
            'market_caps': [[1499040000000, 1000000.0]]
        }
        
        mock_coingecko.return_value.get_coin_market_chart_range_by_id.return_value = mock_data
        
        # Test
        df = self.collector.get_historical_klines(
            'bitcoin',
            '1d',
            '2024-01-01',
            source='coingecko'
        )
        
        # Vérifications
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertTrue('price' in df.columns)
        self.assertTrue('volume' in df.columns)
        self.assertTrue('market_cap' in df.columns)
    
    def test_get_current_price_all_sources(self):
        """Test de la récupération des prix actuels depuis toutes les sources."""
        with patch.multiple(
            'ai_trading.utils.data_collector',
            Client=Mock(),
            CoinGeckoAPI=Mock()
        ):
            # Configuration des mocks
            self.collector.binance_client.get_symbol_ticker.return_value = {'price': '8150.0'}
            self.collector.coingecko_client.get_price.return_value = {'bitcoin': {'usd': 8150.0}}
            
            # Test
            prices = self.collector.get_current_price('BTCUSDT', source='all')
            
            # Vérifications
            self.assertIsInstance(prices, dict)
            self.assertTrue('binance' in prices)
            self.assertTrue('coingecko' in prices)
            self.assertEqual(prices['binance'], 8150.0)

class TestNewsCollector(unittest.TestCase):
    """Tests pour la classe NewsCollector."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.collector = NewsCollector(api_key='test_key')
    
    @patch('ai_trading.utils.data_collector.NewsApiClient')
    def test_get_crypto_news(self, mock_news_api):
        """Test de la récupération des actualités crypto."""
        # Mock des données
        mock_articles = {
            'articles': [
                {
                    'title': 'Bitcoin News',
                    'description': 'Test description',
                    'url': 'http://test.com',
                    'publishedAt': '2024-01-01T00:00:00Z'
                }
            ]
        }
        
        mock_news_api.return_value.get_everything.return_value = mock_articles
        
        # Test
        articles = self.collector.get_crypto_news(
            query='bitcoin',
            language='en'
        )
        
        # Vérifications
        self.assertIsInstance(articles, list)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Bitcoin News')

class TestSocialMediaCollector(unittest.TestCase):
    """Tests pour la classe SocialMediaCollector."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.collector = SocialMediaCollector(
            api_key='test_key',
            api_secret='test_secret',
            access_token='test_token',
            access_token_secret='test_token_secret'
        )
    
    @patch('ai_trading.utils.data_collector.tweepy.API')
    def test_get_crypto_tweets(self, mock_api):
        """Test de la récupération des tweets crypto."""
        # Mock des données
        mock_tweet = Mock()
        mock_tweet.id = 123
        mock_tweet.created_at = datetime.now()
        mock_tweet.full_text = 'Test tweet about #Bitcoin'
        mock_tweet.user.screen_name = 'test_user'
        mock_tweet.retweet_count = 10
        mock_tweet.favorite_count = 20
        
        mock_api.return_value.search_tweets.return_value = [mock_tweet]
        
        # Test
        tweets = self.collector.get_crypto_tweets(
            query='bitcoin',
            count=1
        )
        
        # Vérifications
        self.assertIsInstance(tweets, list)
        self.assertEqual(len(tweets), 1)
        self.assertEqual(tweets[0]['text'], 'Test tweet about #Bitcoin')
        self.assertEqual(tweets[0]['retweets'], 10)
        self.assertEqual(tweets[0]['likes'], 20)

if __name__ == '__main__':
    unittest.main() 