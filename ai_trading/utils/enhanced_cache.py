"""
Cache distribué amélioré avec Redis Cluster et préchargement intelligent.

Ce module étend le cache distribué standard avec:
- Support de Redis Cluster pour la mise à l'échelle horizontale
- Stratégie de préchargement intelligente basée sur les modèles d'utilisation
- Statistiques d'utilisation pour optimiser le cache
"""

import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import redis

from .smart_cache import SmartCache

try:
    from rediscluster import RedisCluster
except ImportError:
    try:
        from redis.cluster import RedisCluster
    except ImportError:
        RedisCluster = None

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedCache")


class AccessPattern:
    """Analyse les modèles d'accès au cache pour optimiser le préchargement."""

    def __init__(self, max_history: int = 1000, time_window: int = 3600):
        """
        Initialise l'analyseur de modèles d'accès.

        Args:
            max_history: Nombre maximum d'accès à conserver dans l'historique
            time_window: Fenêtre de temps en secondes pour l'analyse des modèles
        """
        self.access_history = []  # Liste de tuples (timestamp, clé)
        self.max_history = max_history
        self.time_window = time_window
        self.sequence_patterns = defaultdict(
            Counter
        )  # {clé_précédente: {clé_suivante: count}}

    def record_access(self, key: str) -> None:
        """
        Enregistre un accès au cache.

        Args:
            key: Clé accédée
        """
        now = time.time()

        # Enregistrer l'accès
        self.access_history.append((now, key))

        # Limiter la taille de l'historique
        if len(self.access_history) > self.max_history:
            self.access_history.pop(0)

        # Mettre à jour les modèles de séquence
        if len(self.access_history) >= 2:
            prev_key = self.access_history[-2][1]
            self.sequence_patterns[prev_key][key] += 1

    def get_frequent_patterns(
        self, min_count: int = 2
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Récupère les modèles d'accès fréquents.

        Args:
            min_count: Nombre minimum d'occurrences pour considérer un modèle

        Returns:
            Dict[str, List[Tuple[str, int]]]: Pour chaque clé, liste des clés suivantes fréquentes
        """
        result = {}
        for prev_key, counters in self.sequence_patterns.items():
            patterns = [
                (next_key, count)
                for next_key, count in counters.items()
                if count >= min_count
            ]
            if patterns:
                # Trier par nombre d'occurrences décroissant
                patterns.sort(key=lambda x: x[1], reverse=True)
                result[prev_key] = patterns
        return result

    def predict_next_accesses(self, current_key: str, limit: int = 5) -> List[str]:
        """
        Prédit les prochaines clés qui seront probablement accédées.

        Args:
            current_key: Clé actuellement accédée
            limit: Nombre maximum de prédictions

        Returns:
            List[str]: Liste des clés prédites
        """
        if current_key not in self.sequence_patterns:
            return []

        counters = self.sequence_patterns[current_key]
        # Trier les clés par fréquence d'accès décroissante
        predictions = [key for key, _ in counters.most_common(limit)]
        return predictions

    def get_hot_keys(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Récupère les clés les plus fréquemment accédées.

        Args:
            limit: Nombre maximum de clés à retourner

        Returns:
            List[Tuple[str, int]]: Liste des clés les plus accédées avec leur compte
        """
        now = time.time()
        # Ne considérer que les accès récents (dans la fenêtre de temps)
        recent_accesses = [
            key for ts, key in self.access_history if now - ts <= self.time_window
        ]

        if not recent_accesses:
            return []

        # Compter les occurrences
        counter = Counter(recent_accesses)
        return counter.most_common(limit)


class EnhancedDistributedCache:
    """Cache à deux niveaux : LRU local résilient puis Redis/Redis Cluster.

    ``SmartCache`` est le niveau L1 commun : il apporte l'éviction LRU, la
    compression et la cohérence en mémoire. Redis reste le niveau L2 partagé
    entre les conteneurs. Une indisponibilité temporaire de Redis ne doit donc
    jamais interrompre la collecte.
    """

    def __init__(
        self,
        startup_nodes: List[Dict[str, Union[str, int]]] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        use_cluster: bool = False,
        prefetch_enabled: bool = True,
        max_prefetch: int = 5,
        stats_ttl: int = 86400,  # 24 heures
        local_max_size: int = 1_000,
        compression_level: int = 3,
    ):
        """
        Initialise le cache distribué amélioré.

        Args:
            startup_nodes: Liste des nœuds Redis Cluster (format: [{"host": "127.0.0.1", "port": 7000}])
            host: Hôte Redis (pour mode non-cluster)
            port: Port Redis (pour mode non-cluster)
            db: Base de données Redis (pour mode non-cluster)
            use_cluster: Utiliser Redis Cluster
            prefetch_enabled: Activer le préchargement intelligent
            max_prefetch: Nombre maximum d'éléments à précharger
            stats_ttl: Durée de vie des statistiques en secondes
            local_max_size: Taille maximale du cache LRU local
            compression_level: Niveau de compression du cache LRU local
        """
        self.use_cluster = use_cluster
        self.prefetch_enabled = prefetch_enabled
        self.max_prefetch = max_prefetch
        self.default_ttl = 3600  # 1 heure
        self.stats_ttl = stats_ttl
        self.local_cache = SmartCache(
            max_size=local_max_size,
            ttl=self.default_ttl,
            compression_level=compression_level,
            persist=False,
        )

        # Initialiser le client Redis
        if use_cluster:
            startup_nodes = startup_nodes or [{"host": host, "port": port}]
            try:
                if RedisCluster is None:
                    raise RuntimeError("Le support Redis Cluster n'est pas installe")
                self.client = RedisCluster(
                    startup_nodes=startup_nodes, decode_responses=False
                )
                logger.info(f"Connecté à Redis Cluster avec {len(startup_nodes)} nœuds")
            except Exception as e:
                logger.error(f"Erreur lors de la connexion à Redis Cluster: {e}")
                # Fallback en mode non-cluster
                self.client = redis.Redis(host=host, port=port, db=db)
                self.use_cluster = False
                logger.warning("Fallback vers Redis standard")
        else:
            self.client = redis.Redis(host=host, port=port, db=db)
            logger.info(f"Connecté à Redis standard à {host}:{port}/{db}")

        # Initialiser l'analyseur de modèles d'accès
        self.access_pattern = AccessPattern()

        # Métriques du cache
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0

        # Ensemble des clés en cours de préchargement
        self.prefetching_keys = set()

    def _get_stat_key(self, key: str) -> str:
        """Génère une clé pour les statistiques."""
        return f"stats:{key}"

    def _update_stats(self, key: str, hit: bool) -> None:
        """
        Met à jour les statistiques d'accès.

        Args:
            key: Clé accédée
            hit: Si l'accès a généré un hit ou un miss
        """
        if hit:
            self.hits += 1
        else:
            self.misses += 1

        # Enregistrer l'accès dans l'analyseur de modèles
        self.access_pattern.record_access(key)

        # Mettre à jour les statistiques dans Redis
        stat_key = self._get_stat_key(key)
        try:
            stats = self.client.get(stat_key)
        except (redis.RedisError, OSError) as error:
            logger.warning("Redis indisponible, statistiques locales uniquement: %s", error)
            return

        now = datetime.now().timestamp()
        if stats:
            try:
                stats = json.loads(stats)
            except (TypeError, ValueError):
                stats = None
        if isinstance(stats, dict) and isinstance(stats.get("accesses"), list):
            stats["accesses"].append(now)
            # Garder seulement les 100 derniers accès
            if len(stats["accesses"]) > 100:
                stats["accesses"] = stats["accesses"][-100:]
            stats["hits"] = stats.get("hits", 0) + (1 if hit else 0)
            stats["misses"] = stats.get("misses", 0) + (0 if hit else 1)
        else:
            stats = {
                "accesses": [now],
                "hits": 1 if hit else 0,
                "misses": 0 if hit else 1,
                "created_at": now,
            }

        # Sauvegarder les statistiques
        try:
            self.client.setex(stat_key, self.stats_ttl, json.dumps(stats))
        except (redis.RedisError, OSError) as error:
            logger.warning("Redis indisponible, statistiques non persistées: %s", error)

    @staticmethod
    def _decode(value: Any) -> Any:
        """Décode de façon uniforme les valeurs JSON lues depuis Redis."""
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def _read_remote(self, key: str) -> Optional[Any]:
        """Lit Redis sans faire échouer le chemin de collecte en cas de panne."""
        try:
            value = self.client.get(key)
        except (redis.RedisError, OSError) as error:
            logger.warning("Redis indisponible, repli sur le cache LRU: %s", error)
            return None
        return self._decode(value) if value is not None else None

    def _prefetch(self, current_key: str) -> None:
        """
        Précharge des clés susceptibles d'être demandées prochainement.

        Args:
            current_key: Clé actuellement accédée
        """
        if not self.prefetch_enabled:
            return

        # Prédire les prochaines clés
        next_keys = self.access_pattern.predict_next_accesses(
            current_key, self.max_prefetch
        )

        # Ne précharger que les clés qui ne sont pas déjà en cours de préchargement
        keys_to_prefetch = [k for k in next_keys if k not in self.prefetching_keys]

        if not keys_to_prefetch:
            return

        logger.debug(
            f"Préchargement de {len(keys_to_prefetch)} clés: {keys_to_prefetch}"
        )

        # Marquer les clés comme en cours de préchargement
        self.prefetching_keys.update(keys_to_prefetch)

        try:
            if self.use_cluster:
                values = [self.client.get(key) for key in keys_to_prefetch]
            else:
                values = self.client.mget(keys_to_prefetch)
        except (redis.RedisError, OSError) as error:
            logger.warning("Préchargement Redis indisponible: %s", error)
            self.prefetching_keys.difference_update(keys_to_prefetch)
            return

        for key, value in zip(keys_to_prefetch, values):
            if value is None:
                self.prefetching_keys.discard(key)
                continue
            self.local_cache.set(key, self._decode(value))

    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.

        Args:
            key: Clé de cache

        Returns:
            Optional[Any]: Valeur cachée ou None
        """
        missing = object()
        value = self.local_cache.get(key, missing)
        hit = value is not missing
        if not hit:
            value = self._read_remote(key)
            hit = value is not None
            if hit:
                self.local_cache.set(key, value)

        self._update_stats(key, hit)

        if hit:
            # Si c'était une clé préchargée, compter comme un hit de préchargement
            if key in self.prefetching_keys:
                self.prefetch_hits += 1
                self.prefetching_keys.remove(key)

            # Précharger les prochaines clés probables
            self._prefetch(key)

            return value

        return None

    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """
        Récupère plusieurs valeurs du cache.

        Args:
            keys: Liste des clés

        Returns:
            Dict[str, Any]: Dictionnaire {clé: valeur} des valeurs trouvées
        """
        if not keys:
            return {}

        return {key: value for key in keys if (value := self.get(key)) is not None}

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Stocke une valeur dans le cache.

        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Durée de vie en secondes
        """
        ttl = ttl or self.default_ttl

        if key in self.prefetching_keys:
            self.prefetching_keys.remove(key)

        self.local_cache.set(key, value)
        try:
            self.client.setex(key, ttl, json.dumps(value))
        except (redis.RedisError, OSError) as error:
            logger.warning("Redis indisponible, écriture conservée en LRU: %s", error)

    def mset(self, mapping: Dict[str, Any], ttl: int = None) -> None:
        """
        Stocke plusieurs valeurs dans le cache.

        Args:
            mapping: Dictionnaire {clé: valeur}
            ttl: Durée de vie en secondes
        """
        ttl = ttl or self.default_ttl

        # Supprimer les clés du préchargement
        for key in mapping.keys():
            if key in self.prefetching_keys:
                self.prefetching_keys.remove(key)

        for key, value in mapping.items():
            self.local_cache.set(key, value)

        try:
            if self.use_cluster:
                for key, value in mapping.items():
                    self.client.setex(key, ttl, json.dumps(value))
            else:
                pipeline = self.client.pipeline()
                for key, value in mapping.items():
                    pipeline.setex(key, ttl, json.dumps(value))
                pipeline.execute()
        except (redis.RedisError, OSError) as error:
            logger.warning("Redis indisponible, écritures conservées en LRU: %s", error)

    def delete(self, key: str) -> None:
        """
        Supprime une valeur du cache.

        Args:
            key: Clé à supprimer
        """
        if key in self.prefetching_keys:
            self.prefetching_keys.remove(key)

        self.local_cache.delete(key)
        try:
            self.client.delete(key)
            self.client.delete(self._get_stat_key(key))
        except (redis.RedisError, OSError) as error:
            logger.warning("Redis indisponible, suppression LRU appliquée: %s", error)

    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache.

        Returns:
            Dict[str, Any]: Statistiques du cache
        """
        hot_keys = self.access_pattern.get_hot_keys()

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            ),
            "prefetch_hits": self.prefetch_hits,
            "prefetch_ratio": self.prefetch_hits / self.hits if self.hits > 0 else 0,
            "hot_keys": hot_keys,
            "prefetching_keys": list(self.prefetching_keys),
            "local": self.local_cache.get_stats(),
        }

    def clear_stats(self) -> None:
        """Réinitialise les statistiques du cache."""
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0
        self.access_pattern = AccessPattern()

    def get_key_stats(self, key: str) -> Dict[str, Any]:
        """
        Récupère les statistiques d'une clé spécifique.

        Args:
            key: Clé à analyser

        Returns:
            Dict[str, Any]: Statistiques de la clé
        """
        stat_key = self._get_stat_key(key)
        try:
            stats = self.client.get(stat_key)
        except (redis.RedisError, OSError) as error:
            logger.warning("Redis indisponible, statistiques par clé absentes: %s", error)
            return {"accesses": [], "hits": 0, "misses": 0, "created_at": None}

        if stats:
            return json.loads(stats)

        return {"accesses": [], "hits": 0, "misses": 0, "created_at": None}


# Exemple d'utilisation du cache amélioré
if __name__ == "__main__":
    # Configuration avec Redis standard
    cache = EnhancedDistributedCache(prefetch_enabled=True)

    # Simuler des accès pour générer des modèles
    for i in range(50):
        # Simuler un modèle d'accès A -> B -> C
        if i % 3 == 0:
            key = "article:1"
        elif i % 3 == 1:
            key = "article:2"
        else:
            key = "article:3"

        value = cache.get(key)
        if value is None:
            cache.set(
                key,
                {"title": f"Article {key.split(':')[1]}", "content": f"Contenu {i}"},
            )

    # Vérifier les statistiques
    print("Statistiques du cache:")
    stats = cache.get_stats()
    for stat, value in stats.items():
        if stat not in ["hot_keys", "prefetching_keys"]:
            print(f"  {stat}: {value}")

    print("\nClés populaires:")
    for key, count in stats["hot_keys"]:
        print(f"  {key}: {count} accès")

    print("\nModèles d'accès fréquents:")
    patterns = cache.access_pattern.get_frequent_patterns()
    for prev_key, next_keys in patterns.items():
        print(f"  Après {prev_key}, accès fréquents à:")
        for next_key, count in next_keys:
            print(f"    {next_key}: {count} fois")
