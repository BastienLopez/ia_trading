"""
Collecteur blockchain amélioré avec cache distribué optimisé.

Ce module étend AsyncBlockchainCollector avec:
- Support de Redis Cluster pour une mise à l'échelle horizontale
- Préchargement intelligent basé sur les modèles d'utilisation
- Optimisations pour les environnements multiserveurs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from .async_blockchain_collector import AsyncBlockchainCollector, RateLimiter
from .enhanced_cache import EnhancedDistributedCache

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedBlockchainCollector")

class EnhancedBlockchainCollector(AsyncBlockchainCollector):
    """
    Collecteur blockchain amélioré avec support Redis Cluster 
    et préchargement intelligent.
    """
    
    def __init__(self, 
                 redis_nodes: List[Dict[str, Union[str, int]]] = None,
                 redis_host: str = 'localhost', 
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 use_cluster: bool = False,
                 prefetch_enabled: bool = True,
                 max_prefetch: int = 5):
        """
        Initialise le collecteur blockchain amélioré.
        
        Args:
            redis_nodes: Liste des nœuds Redis Cluster (format: [{"host": "127.0.0.1", "port": 7000}])
            redis_host: Hôte Redis (pour mode non-cluster)
            redis_port: Port Redis (pour mode non-cluster)
            redis_db: Base de données Redis (pour mode non-cluster)
            use_cluster: Utiliser Redis Cluster
            prefetch_enabled: Activer le préchargement intelligent
            max_prefetch: Nombre maximum d'éléments à précharger
        """
        # Initialiser le collecteur de base sans Redis standard
        super().__init__()
        
        # Remplacer le cache standard par le cache amélioré
        self.cache = EnhancedDistributedCache(
            startup_nodes=redis_nodes,
            host=redis_host,
            port=redis_port,
            db=redis_db,
            use_cluster=use_cluster,
            prefetch_enabled=prefetch_enabled,
            max_prefetch=max_prefetch
        )
        
        # Mappage des types de données aux clés de cache
        # Utilisé pour le préchargement basé sur les relations logiques
        self.data_relationships = {
            'address': ['transactions', 'balances'],
            'transactions': ['address', 'block'],
            'block': ['transactions'],
            'protocol': ['tvl', 'pools'],
            'tvl': ['protocol'],
            'pools': ['protocol', 'staking'],
            'staking': ['pools', 'address'],
        }
        
        logger.info(f"Collecteur blockchain amélioré initialisé avec Redis {'Cluster' if use_cluster else 'Standard'}")
    
    async def _make_request(self, session, url, params=None, source=None):
        """
        Surcharge la méthode _make_request pour ajouter des fonctionnalités supplémentaires.
        
        Args:
            session: Session aiohttp
            url: URL de la requête
            params: Paramètres de la requête
            source: Source de données pour le rate limiting
            
        Returns:
            Dict: Réponse JSON
        """
        # La clé de cache est la même que dans la version de base
        cache_key = f"{url}_{str(params)}"
        
        # Vérifier si la clé est dans la liste de préchargement
        # Si oui, on lui donne une priorité plus élevée
        prefetching = cache_key in self.cache.prefetching_keys
        
        # Mesure du temps d'accès pour les statistiques
        data = None
        
        # Logique principale héritée du collecteur de base mais utilisant le cache amélioré
        data = self.cache.get(cache_key)
        if data:
            if prefetching:
                logger.debug(f"Hit sur préchargement pour {cache_key}")
            return data
        
        # Appliquer le rate limiting comme avant
        if source and source in self.rate_limiters:
            await self.rate_limiters[source].acquire()
        
        # Faire la requête comme dans la version de base
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Utiliser le cache amélioré pour stocker les données
                # On utilise TTL variable selon la source
                ttl = self._get_ttl_for_source(source)
                self.cache.set(cache_key, data, ttl=ttl)
                
                # Précharger les données liées si la source est connue
                await self._prefetch_related_data(session, source, data, params)
                
                return data
                
        except Exception as e:
            logger.error(f"Erreur lors de la requête {url}: {e}")
            raise
    
    def _get_ttl_for_source(self, source: str) -> int:
        """
        Détermine la durée de vie en cache appropriée selon la source.
        
        Args:
            source: Source de données
            
        Returns:
            int: Durée de vie en secondes
        """
        # TTL personnalisé selon la source et sa fréquence de mise à jour
        ttl_map = {
            'etherscan': 300,      # 5 minutes pour les données de transactions (changent souvent)
            'defillama': 1800,     # 30 minutes pour les données DeFi (mises à jour moins fréquentes)
            'blockchair': 900,     # 15 minutes
            'coingecko': 600,      # 10 minutes pour les prix
        }
        return ttl_map.get(source, 3600)  # 1 heure par défaut
    
    async def _prefetch_related_data(self, session, source: str, data: Dict, params: Dict):
        """
        Précharge les données connexes en fonction de la requête actuelle.
        
        Args:
            session: Session aiohttp
            source: Source de données
            data: Données récupérées
            params: Paramètres de la requête originale
        """
        # Ne rien faire si le préchargement est désactivé
        if not self.cache.prefetch_enabled:
            return
        
        # Identifier le type de données à partir des paramètres
        data_type = self._identify_data_type(source, params)
        if not data_type or data_type not in self.data_relationships:
            return
        
        # Récupérer les types de données liés à précharger
        related_types = self.data_relationships[data_type]
        
        # Planifier le préchargement des données liées
        prefetch_tasks = []
        for related_type in related_types:
            try:
                # Construire les paramètres pour la requête liée
                related_params = self._build_related_params(data_type, related_type, params, data)
                if related_params:
                    # Créer une tâche pour la requête liée, mais ne pas l'exécuter immédiatement
                    # La clé est marquée pour préchargement mais sera accédée lors d'une demande future
                    related_url = self._get_url_for_type(related_type, source)
                    if related_url:
                        cache_key = f"{related_url}_{str(related_params)}"
                        self.cache.prefetching_keys.add(cache_key)
            except Exception as e:
                logger.error(f"Erreur lors du préchargement de {related_type}: {e}")
    
    def _identify_data_type(self, source: str, params: Dict) -> Optional[str]:
        """
        Identifie le type de données à partir de la source et des paramètres.
        
        Args:
            source: Source de données
            params: Paramètres de la requête
            
        Returns:
            Optional[str]: Type de données ou None
        """
        if source == 'etherscan' and params.get('action') == 'txlist':
            return 'address'
        elif source == 'etherscan' and params.get('action') == 'eth_getBlockByNumber':
            return 'block'
        elif source == 'defillama' and 'protocol' in str(params):
            return 'protocol'
        elif source == 'defillama' and 'pools' in str(params):
            return 'pools'
        
        return None
    
    def _build_related_params(self, data_type: str, related_type: str, 
                             original_params: Dict, data: Dict) -> Optional[Dict]:
        """
        Construit les paramètres pour précharger des données liées.
        
        Args:
            data_type: Type de données d'origine
            related_type: Type de données liées à précharger
            original_params: Paramètres de la requête d'origine
            data: Données récupérées
            
        Returns:
            Optional[Dict]: Paramètres pour la requête liée ou None
        """
        # Logique de construction des paramètres selon les relations
        if data_type == 'address' and related_type == 'transactions':
            # Si on a des données sur une adresse, précharger ses transactions
            address = original_params.get('address')
            if address:
                return {
                    'module': 'account',
                    'action': 'txlist',
                    'address': address,
                    'startblock': 0,
                    'endblock': 99999999,
                    'sort': 'desc',
                }
        
        elif data_type == 'protocol' and related_type == 'tvl':
            # Si on a des données sur un protocole, précharger son TVL
            protocol = original_params.get('protocol')
            if protocol:
                return {'protocol': protocol}
        
        # Autres relations...
        
        return None
    
    def _get_url_for_type(self, data_type: str, source: str) -> Optional[str]:
        """
        Récupère l'URL appropriée pour un type de données.
        
        Args:
            data_type: Type de données
            source: Source préférée
            
        Returns:
            Optional[str]: URL pour la requête
        """
        if data_type in ['address', 'transactions', 'block']:
            return self.base_collector.ETHERSCAN_BASE_URL
        elif data_type in ['protocol', 'tvl', 'pools']:
            return self.base_collector.DEFILLAMA_BASE_URL
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache.
        
        Returns:
            Dict[str, Any]: Statistiques détaillées du cache
        """
        stats = self.cache.get_stats()
        
        # Enrichir avec des statistiques spécifiques au collecteur
        stats['data_relationships'] = self.data_relationships
        
        return stats
    
    async def collect_with_stats(self, address: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Collecte les données et retourne aussi les statistiques du cache.
        
        Args:
            address: Adresse Ethereum optionnelle
            
        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]: Données collectées et statistiques
        """
        data = await self.collect_all_async(address)
        stats = self.get_cache_stats()
        
        return data, stats


# Exemple d'utilisation du collecteur amélioré
async def main():
    # Configuration standard (sans cluster pour l'exemple)
    collector = EnhancedBlockchainCollector(
        redis_host='localhost',
        redis_port=6379,
        use_cluster=False,
        prefetch_enabled=True
    )
    
    # Exemple avec une adresse Ethereum
    address = "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"
    
    # Collecter les données et les statistiques
    data, stats = await collector.collect_with_stats(address)
    
    # Afficher les résultats
    for data_type, df in data.items():
        print(f"\nDonnées {data_type}:")
        if not df.empty:
            print(df.head())
        else:
            print("Aucune donnée")
    
    # Afficher les statistiques du cache
    print("\nStatistiques du cache:")
    for stat_name, stat_value in stats.items():
        if isinstance(stat_value, (int, float, str, bool)):
            print(f"  {stat_name}: {stat_value}")

if __name__ == "__main__":
    asyncio.run(main()) 