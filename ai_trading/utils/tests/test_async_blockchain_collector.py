"""
Tests pour le collecteur de données blockchain asynchrone.
"""

import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import sys

import aiohttp
import pandas as pd
import pytest
import redis

# Ajuster le chemin pour inclure le répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai_trading.utils.async_blockchain_collector import (
    AsyncBlockchainCollector,
    RateLimiter,
    DistributedCache
)

class TestRateLimiter(unittest.TestCase):
    """Tests pour le gestionnaire de rate limits."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.rate_limiter = RateLimiter(calls_per_second=2.0, burst=3)

    async def test_rate_limiting(self):
        """Teste le fonctionnement du rate limiting."""
        start_time = asyncio.get_event_loop().time()
        
        # Faire plusieurs appels consécutifs
        for _ in range(5):
            await self.rate_limiter.acquire()
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Vérifier que le temps d'attente est correct
        # Pour 5 appels avec 2 appels/sec et burst=3, on devrait attendre au moins 1 seconde
        self.assertGreaterEqual(duration, 1.0)

    async def test_burst_handling(self):
        """Teste la gestion des bursts."""
        # Les 3 premiers appels devraient être immédiats (burst)
        start_time = asyncio.get_event_loop().time()
        
        for _ in range(3):
            await self.rate_limiter.acquire()
        
        duration = asyncio.get_event_loop().time() - start_time
        self.assertLess(duration, 0.1)  # Presque instantané

class TestDistributedCache:
    """Tests pour le cache distribué."""

    @pytest.fixture
    def mock_redis(self):
        """Mock pour Redis."""
        with patch('redis.Redis') as mock:
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
            "test_key", 
            3600,  # TTL par défaut
            json.dumps(data)
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
        return AsyncBlockchainCollector()

    @pytest.mark.asyncio
    async def test_make_request(self, collector):
        """Teste la méthode _make_request."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json.return_value = {"result": "success"}
        
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        
        result = await collector._make_request(
            mock_session,
            "https://api.example.com",
            params={"param": "value"},
            source="etherscan"
        )
        
        assert result == {"result": "success"}
        mock_session.get.assert_called_once_with(
            "https://api.example.com",
            params={"param": "value"}
        )

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
                    "gasUsed": "21000"
                }
            ]
        }
        
        with patch.object(collector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data
            
            df = await collector.get_eth_transactions_async(address="0xabcdef123456789")
            
            assert not df.empty
            assert len(df) == 1
            assert df["hash"].iloc[0] == "0x123456789abcdef"

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
                "chains": ["ethereum"]
            }
        ]
        
        with patch.object(collector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data
            
            df = await collector.get_defi_data_async()
            
            assert not df.empty
            assert len(df) == 1
            assert df["name"].iloc[0] == "Uniswap"

    @pytest.mark.asyncio
    async def test_collect_all_async(self, collector):
        """Teste la collecte asynchrone de toutes les données."""
        # Mock pour les différentes méthodes de collecte
        with patch.object(collector, 'get_eth_transactions_async', new_callable=AsyncMock) as mock_tx, \
             patch.object(collector, 'get_defi_data_async', new_callable=AsyncMock) as mock_defi, \
             patch.object(collector.base_collector, 'get_defillama_pools') as mock_pools, \
             patch.object(collector.base_collector, 'get_staking_data') as mock_staking:
            
            # Configurer les mocks
            mock_tx.return_value = pd.DataFrame({'hash': ['0x123']})
            mock_defi.return_value = pd.DataFrame({'name': ['Uniswap']})
            mock_pools.return_value = pd.DataFrame({'pool': ['0xabc']})
            mock_staking.return_value = pd.DataFrame({'chain': ['ethereum']})
            
            results = await collector.collect_all_async(address="0xabcdef123456789")
            
            assert 'transactions' in results
            assert 'tvl' in results
            assert 'pools' in results
            assert 'staking' in results
            
            assert not results['transactions'].empty
            assert not results['tvl'].empty
            assert not results['pools'].empty
            assert not results['staking'].empty

    @pytest.mark.asyncio
    async def test_error_handling(self, collector):
        """Teste la gestion des erreurs."""
        with patch.object(collector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Test error")
            
            # La méthode devrait gérer l'erreur et retourner un DataFrame vide
            df = await collector.get_eth_transactions_async(address="0xabcdef123456789")
            assert df.empty

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, collector):
        """Teste l'intégration du rate limiting."""
        start_time = asyncio.get_event_loop().time()
        
        # Faire plusieurs appels consécutifs
        with patch.object(collector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"result": []}
            
            tasks = []
            for _ in range(5):
                tasks.append(collector.get_eth_transactions_async(address="0xabcdef123456789"))
            
            await asyncio.gather(*tasks)
        
        duration = asyncio.get_event_loop().time() - start_time
        
        # Vérifier que le rate limiting a bien fonctionné
        # Pour 5 appels avec 0.2 appels/sec (etherscan), on devrait attendre au moins 20 secondes
        assert duration >= 20.0

if __name__ == '__main__':
    pytest.main([__file__]) 