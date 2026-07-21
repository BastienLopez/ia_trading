"""
Collecteur de données blockchain asynchrone avec gestion des rate limits et cache distribué.

Ce module étend le collecteur de données blockchain de base avec :
- Collecte asynchrone multi-sources
- Gestion intelligente des rate limits
- Priorisation des sources
- Cache distribué avec Redis
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .blockchain_data_collector import (
    DEFILLAMA_BASE_URL,
    ETHERSCAN_BASE_URL,
    BlockchainDataCollector,
)
from .enhanced_cache import EnhancedDistributedCache

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AsyncBlockchainCollector")


class RateLimiter:
    """Gestionnaire de rate limits avec fenêtre glissante."""

    def __init__(self, calls_per_second: float = 1.0, burst: int = 3):
        """
        Initialise le rate limiter.

        Args:
            calls_per_second: Nombre d'appels autorisés par seconde
            burst: Nombre d'appels consécutifs autorisés
        """
        self.calls_per_second = calls_per_second
        self.burst = burst
        self.calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Attend si nécessaire pour respecter les rate limits."""
        async with self._lock:
            now = time.time()
            # Nettoyer les appels trop anciens
            self.calls = [
                t for t in self.calls if now - t < 1.0 / self.calls_per_second
            ]

            if len(self.calls) >= self.burst:
                # Attendre jusqu'à ce qu'un slot soit disponible
                wait_time = self.calls[0] + 1.0 / self.calls_per_second - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.calls = self.calls[1:]

            self.calls.append(now)


class DistributedCache(EnhancedDistributedCache):
    """Alias de compatibilité du cache commun pour le collecteur asynchrone."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialise le cache distribué.

        Args:
            host: Hôte Redis
            port: Port Redis
            db: Base de données Redis
        """
        super().__init__(host=host, port=port, db=db, prefetch_enabled=True)
        # Conservé pour les intégrations historiques qui accèdent à ``.redis``.
        self.redis = self.client


class AsyncBlockchainCollector:
    """
    Collecteur de données blockchain asynchrone avec gestion avancée des ressources.
    """

    def __init__(
        self,
        cache_host: Optional[str] = None,
        cache_port: int = 6379,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ):
        """
        Initialise le collecteur asynchrone.

        Args:
            cache_host: Hôte du cache Redis
            cache_port: Port du cache Redis
        """
        self.base_collector = BlockchainDataCollector()
        self.cache = DistributedCache(
            host=cache_host or os.getenv("REDIS_HOST", "redis"), port=cache_port
        )
        self.max_retries = max(0, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)

        # Rate limiters par API
        self.rate_limiters = {
            "etherscan": RateLimiter(calls_per_second=0.2),
            "defillama": RateLimiter(calls_per_second=0.5),
            "blockchair": RateLimiter(calls_per_second=0.1),
        }

        # Priorités des sources (1 = plus haute priorité)
        self.source_priorities = {
            "transactions": 1,
            "tvl": 2,
            "pools": 3,
            "staking": 4,
        }

    async def _make_request(
        self,
        session: Any,
        url: str,
        params: Dict = None,
        source: str = None,
    ) -> Dict:
        """
        Effectue une requête HTTP asynchrone avec retry et rate limiting.

        Args:
            session: Session aiohttp
            url: URL de la requête
            params: Paramètres de la requête
            source: Source de données pour le rate limiting

        Returns:
            Dict: Réponse JSON
        """
        # Vérifier le cache
        if aiohttp is None:
            raise RuntimeError("aiohttp est requis pour les requetes HTTP directes")
        cache_key = f"{url}_{str(params)}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data

        # Appliquer le rate limiting
        if source and source in self.rate_limiters:
            await self.rate_limiters[source].acquire()

        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    self.cache.set(cache_key, data)
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                if attempt >= self.max_retries:
                    logger.error("Requête échouée après %s tentative(s): %s", attempt + 1, error)
                    raise
                await asyncio.sleep(self.retry_backoff * (2**attempt))

        raise RuntimeError("Boucle de retry terminée sans réponse")

    async def get_eth_transactions_async(
        self, address: str = None, block: Union[int, str] = None
    ) -> pd.DataFrame:
        """Récupère et transforme les transactions Ethereum sans bloquer la boucle."""
        if not address and not block:
            return pd.DataFrame()

        params: Dict[str, Any]
        if address:
            params = {
                "module": "account", "action": "txlist", "address": address,
                "startblock": 0, "endblock": 99999999, "sort": "desc",
            }
        else:
            params = {
                "module": "proxy", "action": "eth_getBlockByNumber",
                "tag": block, "boolean": "true",
            }

        async with aiohttp.ClientSession() as session:
            data = await self._make_request(
                session, ETHERSCAN_BASE_URL, params=params, source="etherscan"
            )

        processor = getattr(self.base_collector, "_process_eth_transactions", None)
        if processor:
            return processor(data)
        return self._transactions_dataframe(data, address=address)

    async def get_defi_data_async(self, protocol: str = None) -> pd.DataFrame:
        """Récupère les données TVL DefiLlama avec la couche HTTP asynchrone."""
        url = (
            f"{DEFILLAMA_BASE_URL}/protocol/{protocol}"
            if protocol
            else f"{DEFILLAMA_BASE_URL}/protocols"
        )
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, url, source="defillama")

        processor = getattr(self.base_collector, "_process_defi_data", None)
        if processor:
            return processor(data, protocol=protocol)
        if protocol:
            frame = pd.DataFrame(data.get("tvl", [])) if isinstance(data, dict) else pd.DataFrame()
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], unit="s", errors="coerce")
            return frame
        frame = pd.DataFrame(data)
        if "lastFullyUpdated" in frame.columns:
            frame["lastUpdated"] = pd.to_datetime(
                frame["lastFullyUpdated"], unit="s", errors="coerce"
            )
        return frame

    @staticmethod
    def _transactions_dataframe(data: Dict[str, Any], address: Optional[str]) -> pd.DataFrame:
        result = data.get("result") if isinstance(data, dict) else None
        if not result:
            return pd.DataFrame()
        rows = result if address else result.get("transactions", [])
        frame = pd.DataFrame(rows)
        if "timeStamp" in frame.columns:
            frame["timeStamp"] = pd.to_datetime(
                pd.to_numeric(frame["timeStamp"], errors="coerce"), unit="s", errors="coerce"
            )
        if "value" in frame.columns:
            values = pd.to_numeric(frame["value"], errors="coerce")
            frame["ether_value"] = values / 1e18
        return frame

    async def collect_all_async(self, address: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collecte toutes les données blockchain de manière asynchrone.

        Args:
            address: Adresse Ethereum optionnelle

        Returns:
            Dict[str, pd.DataFrame]: Données collectées par type
        """
        tasks = [("tvl", self.get_defi_data_async())]
        if address:
            tasks.append(("transactions", self.get_eth_transactions_async(address)))
        tasks.extend(
            [
                ("pools", asyncio.to_thread(self.base_collector.get_defillama_pools)),
                ("staking", asyncio.to_thread(self.base_collector.get_staking_data)),
            ]
        )
        tasks.sort(key=lambda item: self.source_priorities[item[0]])
        values = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
        results = {}
        for (name, _), value in zip(tasks, values):
            if isinstance(value, Exception):
                logger.error(f"Erreur lors de la collecte de {name}: {value}")
                results[name] = pd.DataFrame()
            else:
                results[name] = value
        return results


# Exemple d'utilisation
async def main():
    collector = AsyncBlockchainCollector()

    # Exemple avec une adresse Ethereum
    address = "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"
    results = await collector.collect_all_async(address)

    for data_type, df in results.items():
        print(f"\nDonnées {data_type}:")
        if not df.empty:
            print(df.head())
        else:
            print("Aucune donnée")


if __name__ == "__main__":
    asyncio.run(main())
