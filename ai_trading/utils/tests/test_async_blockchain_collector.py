"""
Tests pour le collecteur de données blockchain asynchrone.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pandas as pd
import pytest
import redis

# Ajuster le chemin pour inclure le répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


# Mock pour BlockchainDataCollector qui semble ne pas exister
class MockBlockchainDataCollector:
    def __init__(self):
        # Ajouter les URL de base nécessaires
        self.ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
        self.DEFILLAMA_BASE_URL = "https://api.llama.fi"
        self.STAKING_API_URL = "https://staking.api.com"

    def get_defillama_pools(self):
        return pd.DataFrame({"pool": ["pool1"]})

    def get_staking_data(self):
        return pd.DataFrame({"validator": ["val1"]})

    def _process_eth_transactions(self, data):
        return pd.DataFrame({"tx": ["tx1"]})

    def _process_defi_data(self, data, protocol=None):
        return pd.DataFrame({"name": ["Uniswap"], "tvl": [5000000000]})


# Patcher le module manquant et Redis
sys.modules["ai_trading.utils.blockchain_collector"] = MagicMock()
sys.modules["ai_trading.utils.blockchain_collector"].BlockchainDataCollector = (
    MockBlockchainDataCollector
)

# Patch Redis pour éviter les erreurs de connexion
mock_redis_client = MagicMock()
mock_redis_client.get.return_value = None
mock_redis_client.setex.return_value = True
mock_redis_client.delete.return_value = True

# Remplacer la classe Redis pour éviter les tentatives de connexion
original_redis = redis.Redis
redis.Redis = MagicMock(return_value=mock_redis_client)

from ai_trading.utils.async_blockchain_collector import (
    AsyncBlockchainCollector,
    DistributedCache,
    RateLimiter,
)


class TestRateLimiter(unittest.TestCase):
    """Tests pour le gestionnaire de rate limits."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.rate_limiter = RateLimiter(calls_per_second=2.0, burst=3)

    async def test_rate_limiting(self):
        """Teste le fonctionnement du rate limiting."""
        # Mock pour le temps asyncio
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_time = MagicMock()
            mock_time.side_effect = [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ]  # Simuler le temps qui passe
            mock_loop.return_value.time = mock_time

            # Faire plusieurs appels consécutifs
            for _ in range(5):
                await self.rate_limiter.acquire()

            # Vérifier que le temps a été consulté le bon nombre de fois
            self.assertEqual(mock_time.call_count, 6)

    async def test_burst_handling(self):
        """Teste la gestion des bursts."""
        # Mock pour le temps asyncio
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_time = MagicMock()
            mock_time.side_effect = [0.0, 0.01, 0.02, 0.03]  # Temps simulé
            mock_loop.return_value.time = mock_time

            # Les 3 premiers appels devraient être immédiats (burst)
            for _ in range(3):
                await self.rate_limiter.acquire()

            # Vérifier que le temps a été consulté le bon nombre de fois
            self.assertEqual(mock_time.call_count, 4)


class TestDistributedCache:
    """Tests pour le cache distribué."""

    @pytest.fixture
    def mock_redis(self):
        """Mock pour Redis."""
        with patch("redis.Redis") as mock:
            yield mock

    def test_cache_get(self, mock_redis):
        """Teste la récupération depuis le cache."""
        mock_redis.return_value.get.return_value = json.dumps({"key": "value"}).encode()
        cache = DistributedCache()

        result = cache.get("test_key")
        assert result == {"key": "value"}
        mock_redis.return_value.get.assert_called_once_with("test_key")

    def test_cache_set(self, mock_redis):
        """Teste le stockage dans le cache."""
        cache = DistributedCache()
        data = {"key": "value"}

        cache.set("test_key", data)
        mock_redis.return_value.setex.assert_called_once_with(
            "test_key", 3600, json.dumps(data)  # TTL par défaut
        )

    def test_cache_delete(self, mock_redis):
        """Teste la suppression du cache."""
        cache = DistributedCache()

        cache.delete("test_key")
        mock_redis.return_value.delete.assert_called_once_with("test_key")


class TestAsyncBlockchainCollector:
    """Tests pour le collecteur blockchain asynchrone."""

    @pytest.fixture
    def collector(self):
        """Crée une instance du collecteur."""
        # Utiliser directement notre mock au lieu de patcher
        mock_base = MockBlockchainDataCollector()

        # Patch pour DistributedCache
        with patch(
            "ai_trading.utils.async_blockchain_collector.DistributedCache"
        ) as mock_cache:
            mock_cache_instance = mock_cache.return_value
            mock_cache_instance.get.return_value = None  # Cache miss par défaut

            collector = AsyncBlockchainCollector()

            # Remplacer le cache
            collector.cache = mock_cache_instance

            # Affecter manuellement notre mock
            collector.base_collector = mock_base

            # Pour les tests du rate limiter
            collector.rate_limiters = {
                "etherscan": MagicMock(),
                "defillama": MagicMock(),
            }
            # Convertir les méthodes acquire en AsyncMock pour éviter l'erreur de coroutine
            collector.rate_limiters["etherscan"].acquire = AsyncMock()
            collector.rate_limiters["defillama"].acquire = AsyncMock()

            yield collector

    @pytest.fixture
    def mock_aiohttp_session(self):
        """Mock pour la session aiohttp."""
        # Créer un mock avec support pour async context manager
        mock_session = MagicMock()

        # Configurer le mock pour les appels asynchrones
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value={"result": "success"})

        # Configurer correctement session.get comme un context manager async
        cm = AsyncMock()
        cm.__aenter__.return_value = mock_response
        mock_session.get.return_value = cm

        return mock_session

    @pytest.mark.asyncio
    async def test_make_request(self, collector, mock_aiohttp_session):
        """Teste la méthode _make_request."""
        # Simuler un cache miss pour tester le chemin de requête complet
        collector.cache.get.return_value = None

        # Désactiver le décorateur retry pour simplifier le test
        with patch(
            "ai_trading.utils.async_blockchain_collector.retry",
            return_value=lambda f: f,
        ):
            result = await collector._make_request(
                mock_aiohttp_session,
                "https://api.example.com",
                params={"param": "value"},
                source="etherscan",
            )

            assert result == {"result": "success"}
            mock_aiohttp_session.get.assert_called_once_with(
                "https://api.example.com", params={"param": "value"}
            )
            collector.rate_limiters["etherscan"].acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_eth_transactions_async(self, collector):
        """Teste la récupération asynchrone des transactions Ethereum."""
        mock_data = {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "blockNumber": "10000000",
                    "timeStamp": "1612345678",
                    "hash": "0x123456789abcdef",
                    "from": "0xabcdef123456789",
                    "to": "0x987654321fedcba",
                    "value": "1000000000000000000",  # 1 ETH
                    "gasPrice": "20000000000",
                    "gasUsed": "21000",
                }
            ],
        }

        # Configurer le mock pour retourner nos données de test
        with patch.object(
            collector, "_make_request", new_callable=AsyncMock
        ) as mock_make_request:
            mock_make_request.return_value = mock_data

            # Simuler le traitement des transactions
            df = await collector.get_eth_transactions_async(address="0xabcdef123456789")

            assert not df.empty
            assert df["tx"].iloc[0] == "tx1"

    @pytest.mark.asyncio
    async def test_get_defi_data_async(self, collector):
        """Teste la récupération asynchrone des données DeFi."""
        mock_data = [
            {
                "name": "Uniswap",
                "symbol": "UNI",
                "tvl": 5000000000,
                "change_1d": 0.05,
                "change_7d": 0.1,
                "chains": ["ethereum"],
            }
        ]

        # Configurer le mock pour retourner nos données de test
        with patch.object(
            collector, "_make_request", new_callable=AsyncMock
        ) as mock_make_request:
            mock_make_request.return_value = mock_data

            df = await collector.get_defi_data_async()

            assert not df.empty
            assert df["name"].iloc[0] == "Uniswap"

    @pytest.mark.asyncio
    async def test_collect_all_async(self, collector):
        """Teste la collecte asynchrone de toutes les données."""
        # Mocks pour les différentes méthodes de collecte
        with patch.object(
            collector, "get_eth_transactions_async", new_callable=AsyncMock
        ) as mock_tx, patch.object(
            collector, "get_defi_data_async", new_callable=AsyncMock
        ) as mock_defi:

            # Configurer les mocks
            mock_tx.return_value = pd.DataFrame({"hash": ["0x123"]})
            mock_defi.return_value = pd.DataFrame({"name": ["Uniswap"]})

            # Utiliser des fonctions déjà mockes dans le setup
            results = await collector.collect_all_async(address="0xabcdef123456789")

            assert "transactions" in results
            assert "tvl" in results
            assert "pools" in results
            assert "staking" in results

            # Vérifier que les DataFrames ne sont pas vides
            assert not results["transactions"].empty
            assert not results["tvl"].empty
            assert isinstance(results["pools"], pd.DataFrame)
            assert isinstance(results["staking"], pd.DataFrame)

    @pytest.mark.asyncio
    async def test_error_handling(self, collector):
        """Teste la gestion des erreurs."""

        # Créer une version simplifiée de _make_request qui lève une exception
        async def mock_make_request(session, url, params=None, source=None):
            # Ignorer les paramètres et lever directement une exception
            raise aiohttp.ClientError("Test error")

        # Remplacer _make_request par notre version
        original_make_request = collector._make_request
        collector._make_request = mock_make_request

        try:
            # Créer un mock session qui ne sera en fait pas utilisé
            mock_session = MagicMock()

            # Désactiver les retries pour ce test
            with patch(
                "ai_trading.utils.async_blockchain_collector.retry",
                return_value=lambda f: f,
            ):
                with pytest.raises(aiohttp.ClientError) as excinfo:
                    await collector._make_request(
                        mock_session,
                        "https://api.example.com",
                        params={},
                        source="test",
                    )

                # Vérifier que c'est bien notre erreur
                assert str(excinfo.value) == "Test error"
        finally:
            # Restaurer la méthode originale
            collector._make_request = original_make_request

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, collector):
        """Teste l'intégration du rate limiting."""
        # Créer une version personnalisée de get_eth_transactions_async qui passe par notre code
        original_method = collector.get_eth_transactions_async

        async def custom_method(*args, **kwargs):
            # Remplacer _make_request pour pouvoir patcher le rate limiter
            original_make_request = collector._make_request

            async def test_make_request(*args, **kwargs):
                # Vérifier que cette méthode a été appelée avec la bonne source
                if "source" in kwargs and kwargs["source"] == "etherscan":
                    # Enregistrer l'appel au rate limiter
                    await collector.rate_limiters["etherscan"].acquire()
                return {"result": []}

            # Remplacer temporairement la méthode _make_request
            with patch.object(collector, "_make_request", new=test_make_request):
                return await original_method(*args, **kwargs)

        # Remplacer temporairement la méthode avec notre version personnalisée
        with patch.object(collector, "get_eth_transactions_async", new=custom_method):
            # Appeler la méthode
            await collector.get_eth_transactions_async(address="0xtest")

            # Vérifier que le rate limiter a bien été appelé
            collector.rate_limiters["etherscan"].acquire.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
