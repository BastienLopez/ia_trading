"""
Collecteur de données blockchain asynchrone avec gestion des rate limits et cache distribué.

Ce module étend le collecteur de données blockchain de base avec :
- Collecte asynchrone multi-sources
- Gestion intelligente des rate limits
- Priorisation des sources
- Cache distribué avec Redis
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from functools import wraps

import aiohttp
import pandas as pd
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

from .blockchain_data_collector import BlockchainDataCollector

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
            self.calls = [t for t in self.calls if now - t < 1.0/self.calls_per_second]
            
            if len(self.calls) >= self.burst:
                # Attendre jusqu'à ce qu'un slot soit disponible
                wait_time = self.calls[0] + 1.0/self.calls_per_second - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.calls = self.calls[1:]
            
            self.calls.append(now)

class DistributedCache:
    """Cache distribué utilisant Redis."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """
        Initialise le cache distribué.
        
        Args:
            host: Hôte Redis
            port: Port Redis
            db: Base de données Redis
        """
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.default_ttl = 3600  # 1 heure
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Durée de vie en secondes
        """
        ttl = ttl or self.default_ttl
        self.redis.setex(key, ttl, json.dumps(value))
    
    def delete(self, key: str) -> None:
        """Supprime une valeur du cache."""
        self.redis.delete(key)

class AsyncBlockchainCollector:
    """
    Collecteur de données blockchain asynchrone avec gestion avancée des ressources.
    """
    
    def __init__(self, cache_host: str = 'localhost', cache_port: int = 6379):
        """
        Initialise le collecteur asynchrone.
        
        Args:
            cache_host: Hôte du cache Redis
            cache_port: Port du cache Redis
        """
        self.base_collector = BlockchainDataCollector()
        self.cache = DistributedCache(host=cache_host, port=cache_port)
        
        # Rate limiters par API
        self.rate_limiters = {
            'etherscan': RateLimiter(calls_per_second=0.2),  # 5 appels/sec
            'defillama': RateLimiter(calls_per_second=0.5),  # 2 appels/sec
            'blockchair': RateLimiter(calls_per_second=0.1),  # 10 appels/sec
        }
        
        # Priorités des sources (1 = plus haute priorité)
        self.source_priorities = {
            'etherscan': 1,
            'defillama': 2,
            'blockchair': 3,
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, session: aiohttp.ClientSession, url: str, 
                          params: Dict = None, source: str = None) -> Dict:
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
        cache_key = f"{url}_{str(params)}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # Appliquer le rate limiting
        if source and source in self.rate_limiters:
            await self.rate_limiters[source].acquire()
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Mettre en cache
                self.cache.set(cache_key, data)
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"Erreur lors de la requête {url}: {e}")
            raise
    
    async def get_eth_transactions_async(self, address: str = None, block: Union[int, str] = None) -> pd.DataFrame:
        """Version asynchrone de get_eth_transactions."""
        async with aiohttp.ClientSession() as session:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': address,
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'desc',
            } if address else {
                'module': 'proxy',
                'action': 'eth_getBlockByNumber',
                'tag': block if block != 'latest' else 'latest',
                'boolean': 'true',
            }
            
            data = await self._make_request(session, self.base_collector.ETHERSCAN_BASE_URL, 
                                          params=params, source='etherscan')
            
            return self.base_collector._process_eth_transactions(data)
    
    async def get_defi_data_async(self, protocol: str = None) -> pd.DataFrame:
        """Version asynchrone de get_defillama_tvl."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_collector.DEFILLAMA_BASE_URL}/{'protocol/' + protocol if protocol else 'protocols'}"
            data = await self._make_request(session, url, source='defillama')
            
            return self.base_collector._process_defi_data(data, protocol)
    
    async def collect_all_async(self, address: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collecte toutes les données blockchain de manière asynchrone.
        
        Args:
            address: Adresse Ethereum optionnelle
            
        Returns:
            Dict[str, pd.DataFrame]: Données collectées par type
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Prioriser les tâches selon leur importance
            if address:
                tasks.append(('transactions', self.get_eth_transactions_async(address)))
            
            tasks.extend([
                ('tvl', self.get_defi_data_async()),
                ('pools', self.base_collector.get_defillama_pools()),
                ('staking', self.base_collector.get_staking_data()),
            ])
            
            # Trier les tâches par priorité
            tasks.sort(key=lambda x: self.source_priorities.get(x[0], 999))
            
            # Exécuter les tâches en parallèle
            results = {}
            for name, task in tasks:
                try:
                    results[name] = await task
                except Exception as e:
                    logger.error(f"Erreur lors de la collecte de {name}: {e}")
                    results[name] = pd.DataFrame()
            
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