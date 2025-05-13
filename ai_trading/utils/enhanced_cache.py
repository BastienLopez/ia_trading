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
from rediscluster import RedisCluster

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
    """Cache distribué amélioré avec Redis Cluster et préchargement intelligent."""

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
        """
        self.use_cluster = use_cluster
        self.prefetch_enabled = prefetch_enabled
        self.max_prefetch = max_prefetch
        self.default_ttl = 3600  # 1 heure
        self.stats_ttl = stats_ttl

        # Initialiser le client Redis
        if use_cluster:
            startup_nodes = startup_nodes or [{"host": host, "port": port}]
            try:
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
        stats = self.client.get(stat_key)

        now = datetime.now().timestamp()
        if stats:
            stats = json.loads(stats)
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
        self.client.setex(stat_key, self.stats_ttl, json.dumps(stats))

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

        # On ne fait rien de plus ici, le préchargement se fera lors des prochains accès
        # C'est une stratégie passive de préchargement

    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.

        Args:
            key: Clé de cache

        Returns:
            Optional[Any]: Valeur cachée ou None
        """
        value = self.client.get(key)
        hit = value is not None

        self._update_stats(key, hit)

        if hit:
            # Si c'était une clé préchargée, compter comme un hit de préchargement
            if key in self.prefetching_keys:
                self.prefetch_hits += 1
                self.prefetching_keys.remove(key)

            # Précharger les prochaines clés probables
            self._prefetch(key)

            return json.loads(value)

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

        if self.use_cluster:
            # Redis Cluster n'a pas de mget natif, on fait des get individuels
            result = {}
            for key in keys:
                value = self.get(key)
                if value is not None:
                    result[key] = value
            return result
        else:
            # Redis standard peut utiliser mget
            values = self.client.mget(keys)
            result = {}

            for key, value in zip(keys, values):
                if value is not None:
                    hit = True
                    parsed_value = json.loads(value)
                    result[key] = parsed_value
                else:
                    hit = False

                self._update_stats(key, hit)

                if hit and key in self.prefetching_keys:
                    self.prefetch_hits += 1
                    self.prefetching_keys.remove(key)

            # Précharger les prochaines clés probables pour la dernière clé accédée
            if keys:
                self._prefetch(keys[-1])

            return result

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

        self.client.setex(key, ttl, json.dumps(value))

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

        if self.use_cluster:
            # Redis Cluster n'a pas de mset natif avec TTL, on fait des set individuels
            for key, value in mapping.items():
                self.set(key, value, ttl)
        else:
            # Redis standard peut utiliser pipeline
            pipeline = self.client.pipeline()
            for key, value in mapping.items():
                pipeline.setex(key, ttl, json.dumps(value))
            pipeline.execute()

    def delete(self, key: str) -> None:
        """
        Supprime une valeur du cache.

        Args:
            key: Clé à supprimer
        """
        if key in self.prefetching_keys:
            self.prefetching_keys.remove(key)

        self.client.delete(key)

        # Supprimer aussi les statistiques
        self.client.delete(self._get_stat_key(key))

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
        stats = self.client.get(stat_key)

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
