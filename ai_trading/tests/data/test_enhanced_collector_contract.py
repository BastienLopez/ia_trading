"""Contrats hors reseau du collecteur multi-sources historique."""

import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd


if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: False
    sys.modules["dotenv"] = dotenv
if "pycoingecko" not in sys.modules:
    pycoingecko = types.ModuleType("pycoingecko")
    pycoingecko.CoinGeckoAPI = object
    sys.modules["pycoingecko"] = pycoingecko
if "requests" not in sys.modules:
    requests = types.ModuleType("requests")
    requests.get = None
    sys.modules["requests"] = requests

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

    def get_search_trending(self):
        return {"coins": [{"item": {"id": "bitcoin"}}]}

    def get_global(self):
        return {"data": {"active_cryptocurrencies": 1}}


class EnhancedCollectorContractTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def http_get(url, *, params, timeout):
            self.calls.append((url, params, timeout))
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
            request_timeout=7,
        )

    def test_all_price_sources_return_the_documented_schema(self):
        sources = [
            self.collector.get_crypto_prices_coingecko(),
            self.collector.get_crypto_prices_coincap(),
            self.collector.get_crypto_prices_cryptocompare(),
        ]
        for frame in sources:
            self.assertIsInstance(frame, pd.DataFrame)
            self.assertFalse(frame.empty)
            self.assertIn("price", frame.columns)
            self.assertIn("volume", frame.columns)
            self.assertIn("market_cap", frame.columns)
            self.assertIsInstance(frame.index, pd.DatetimeIndex)

    def test_http_sources_use_the_injected_client_with_timeout(self):
        self.collector.get_crypto_prices_coincap()
        self.collector.get_crypto_prices_cryptocompare()
        self.collector.get_fear_greed_index()
        self.collector.get_crypto_news(limit=1)
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(all(timeout == 7 for _, _, timeout in self.calls))

    def test_coingecko_global_and_trending_outputs_are_preserved(self):
        self.assertEqual(self.collector.get_global_crypto_data()["data"]["active_cryptocurrencies"], 1)
        self.assertEqual(self.collector.get_trending_coins()[0]["item"]["id"], "bitcoin")

    def test_http_collection_retries_through_the_shared_requester(self):
        attempts = 0

        def transient_get(url, *, params, timeout):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("temporary provider failure")
            return FakeResponse({"data": [{"time": 1, "priceUsd": "12.5"}]})

        collector = EnhancedDataCollector(
            coingecko_client=FakeCoinGecko(),
            http_get=transient_get,
            requester=ResilientRequester(
                max_retries=1,
                base_delay=0,
                max_delay=0,
                timeout=1,
                jitter=0,
            ),
        )

        result = collector.get_crypto_prices_coincap("bitcoin")

        self.assertEqual(attempts, 2)
        self.assertEqual(len(result), 1)

    def test_price_fallback_is_prioritized_and_observable(self):
        """Une source vide déclenche réellement CoinCap, sans donnée fictive."""
        expected = pd.DataFrame(
            {"price": [100.0], "volume": [1.0], "market_cap": [10.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        with patch.object(
            self.collector, "get_crypto_prices_coingecko", return_value=pd.DataFrame()
        ), patch.object(
            self.collector, "get_crypto_prices_coincap", return_value=expected
        ):
            frame, source = self.collector.get_best_available_price_data("bitcoin")

        self.assertEqual(source, "coincap")
        self.assertTrue(frame.equals(expected))
        metrics = self.collector.price_requester.get_metrics()
        self.assertEqual(metrics["coingecko"]["failure_count"], 1)
        self.assertEqual(metrics["coincap"]["success_count"], 1)


if __name__ == "__main__":
    unittest.main()
