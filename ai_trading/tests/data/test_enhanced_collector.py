"""
Tests unitaires pour le module de collecte de données amélioré.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pandas as pd

if "dotenv" not in sys.modules:
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv

if "pycoingecko" not in sys.modules:
    pycoingecko = ModuleType("pycoingecko")
    pycoingecko.CoinGeckoAPI = object
    sys.modules["pycoingecko"] = pycoingecko

if "requests" not in sys.modules:
    sys.modules["requests"] = ModuleType("requests")

from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
from ai_trading.utils.resilient_requester import ResilientRequester


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeCoinGecko:
    def get_coin_market_chart_by_id(self, **_kwargs):
        return {
            "prices": [[1_700_000_000_000, 100.0]],
            "total_volumes": [[1_700_000_000_000, 10.0]],
            "market_caps": [[1_700_000_000_000, 1_000.0]],
        }

    def get_global(self):
        return {"data": {"active_cryptocurrencies": 1}}

    def get_search_trending(self):
        return {"coins": [{"item": {"id": "bitcoin"}}]}


class TestEnhancedDataCollector(unittest.TestCase):
    """Tests pour la classe EnhancedDataCollector."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.temp_dir = tempfile.TemporaryDirectory()

        def http_get(url, *, params, timeout):
            if "coincap" in url:
                return FakeResponse({"data": [{"time": 1_700_000_000_000, "priceUsd": "100"}]})
            if "cryptocompare" in url:
                return FakeResponse({"Data": {"Data": [{"time": 1_700_000_000, "close": 100, "volumefrom": 10, "open": 99, "high": 101, "low": 98}]}})
            if "alternative.me" in url:
                return FakeResponse({"data": [{"value": "50", "value_classification": "Neutral", "timestamp": "1700000000"}]})
            if "cryptopanic" in url:
                return FakeResponse({"results": [{"title": "Fixture news"}]})
            raise AssertionError(f"URL inattendue: {url}")

        self.collector = EnhancedDataCollector(
            coingecko_client=FakeCoinGecko(),
            http_get=http_get,
            requester=ResilientRequester(max_retries=0, timeout=1, jitter=0),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Teste l'initialisation du collecteur."""
        self.assertIsNotNone(self.collector.coingecko)
        self.assertEqual(self.collector.coincap_base_url, "https://api.coincap.io/v2")
        self.assertEqual(
            self.collector.cryptocompare_base_url,
            "https://min-api.cryptocompare.com/data",
        )

    def test_get_crypto_prices_coingecko(self):
        """Teste la récupération des prix via CoinGecko."""
        df = self.collector.get_crypto_prices_coingecko(coin_id="bitcoin", days=3)

        # Vérification que le DataFrame n'est pas vide
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

        # Vérification des colonnes
        self.assertIn("price", df.columns)
        self.assertIn("volume", df.columns)
        self.assertIn("market_cap", df.columns)

    def test_get_crypto_prices_coincap(self):
        """Teste la récupération des prix via CoinCap."""
        df = self.collector.get_crypto_prices_coincap(coin_id="bitcoin", days=3)

        self.assertGreater(len(df), 0)
        self.assertIn("price", df.columns)

    def test_get_crypto_prices_cryptocompare(self):
        """Teste la récupération des prix via CryptoCompare."""
        df = self.collector.get_crypto_prices_cryptocompare(coin_symbol="BTC", days=3)

        self.assertGreater(len(df), 0)
        self.assertTrue("close" in df.columns or "price" in df.columns)

    def test_get_merged_price_data(self):
        """Teste la récupération et fusion des données de prix."""
        df = self.collector.get_merged_price_data(
            coin_id="bitcoin", days=3, include_fear_greed=False, mock_data=False
        )

        self.assertGreater(len(df), 0)
        self.assertTrue("close" in df.columns or "price" in df.columns)
        self.assertIn("source", df.columns)

    def test_get_fear_greed_index(self):
        """Teste la récupération de l'indice Fear & Greed."""
        df = self.collector.get_fear_greed_index(days=7)

        self.assertGreater(len(df), 0)
        self.assertIn("value", df.columns)
        self.assertTrue("classification" in df.columns or "value_classification" in df.columns)

    def test_get_global_crypto_data(self):
        """Teste la récupération des données globales du marché crypto."""
        data = self.collector.get_global_crypto_data()

        self.assertIn("data", data)
        self.assertIn("active_cryptocurrencies", data["data"])

    def test_get_trending_coins(self):
        """Teste la récupération des cryptomonnaies tendance."""
        trending = self.collector.get_trending_coins()

        self.assertIsInstance(trending, list)
        self.assertIn("item", trending[0])

    def test_get_crypto_news(self):
        """Teste la récupération des actualités crypto."""
        news = self.collector.get_crypto_news(limit=3)

        self.assertIsInstance(news, list)
        self.assertIn("title", news[0])

    def test_save_data(self):
        """Teste la sauvegarde des données."""
        # Création d'un petit DataFrame de test
        test_data = pd.DataFrame(
            {"price": [100, 101, 102], "volume": [1000, 1100, 1200]},
            index=pd.date_range(start="2023-01-01", periods=3),
        )

        with patch(
            "ai_trading.utils.enhanced_data_collector.INFO_RETOUR_DIR",
            Path(self.temp_dir.name),
        ):
            self.collector.save_data(test_data, "test_save_enhanced.csv")
        saved = Path(self.temp_dir.name) / "data" / "test_save_enhanced.csv"
        self.assertTrue(saved.exists())
        self.assertEqual(len(pd.read_csv(saved)), len(test_data))


if __name__ == "__main__":
    unittest.main()
