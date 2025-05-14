"""
Module de gestion de cache pour les prédictions de marché.

Ce module implémente un système de mise en cache pour les opérations intensives
dans les modules de prédiction, réduisant ainsi la latence et optimisant les performances.
"""

import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from datetime import datetime, timedelta
import threading
import pickle
import os
import zlib  # Pour la compression
import gzip  # Pour la compression gzip
import concurrent.futures  # Pour le préchargement parallèle
import sys

import pandas as pd

# Configuration du logger
from ai_trading.utils import setup_logger
logger = setup_logger("cache_manager")

class CacheManager:
    """
    Gestionnaire de cache pour les opérations de prédiction de marché.
    
    Implémente un système de cache à plusieurs niveaux avec gestion du TTL
    (Time To Live) et des stratégies d'éviction LRU (Least Recently Used).
    """
    
    def __init__(self, capacity: int = 100, ttl: int = 300, 
                 persist_path: Optional[str] = None,
                 enable_disk_cache: bool = True,
                 compression_level: int = 6,  # Niveau de compression (0-9)
                 compression_method: str = "zlib",  # zlib ou gzip
                 enable_predictive_loading: bool = True,  # Préchargement prédictif
                 max_prefetch_workers: int = 4  # Nombre max de workers pour préchargement
                ):
        """
        Initialise le gestionnaire de cache.
        
        Args:
            capacity: Capacité maximale du cache en mémoire
            ttl: Durée de vie par défaut des entrées en secondes
            persist_path: Chemin pour persister le cache sur disque
            enable_disk_cache: Activer la persistence sur disque
            compression_level: Niveau de compression (0-9, 0=pas de compression, 9=compression max)
            compression_method: Méthode de compression ('zlib' ou 'gzip')
            enable_predictive_loading: Activer le préchargement prédictif
            max_prefetch_workers: Nombre maximum de workers pour le préchargement
        """
        self.memory_cache = {}  # Cache mémoire principal
        self.capacity = capacity
        self.default_ttl = ttl
        self.enable_disk_cache = enable_disk_cache
        self.persist_path = persist_path
        
        # Configuration de la compression
        self.compression_level = min(9, max(0, compression_level))
        self.compression_method = compression_method
        self.use_compression = self.compression_level > 0
        
        # Configuration du préchargement prédictif
        self.enable_predictive_loading = enable_predictive_loading
        self.max_prefetch_workers = max_prefetch_workers
        self.prefetch_executor = None
        self.prefetch_patterns = {}  # Stocke les patterns d'accès
        
        if self.enable_predictive_loading:
            self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_prefetch_workers,
                thread_name_prefix="cache_prefetch"
            )
        
        # Dictionnaires pour gérer les métadonnées
        self.access_times = {}  # Timestamps d'accès
        self.expiry_times = {}  # Timestamps d'expiration
        self.access_frequency = {}  # Fréquence d'accès pour chaque clé
        self.access_pattern = {}  # Séquence d'accès récente
        
        # Verrou pour thread-safety
        self.lock = threading.RLock()
        
        # Statistiques du cache
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "compression_savings": 0,  # Espace économisé grâce à la compression
            "prefetch_hits": 0,  # Prédictions correctes de préchargement
            "prefetch_misses": 0  # Prédictions incorrectes de préchargement
        }
        
        # Création du répertoire de cache si nécessaire
        if self.enable_disk_cache and self.persist_path:
            os.makedirs(self.persist_path, exist_ok=True)
            self._load_disk_cache()
        
        logger.info(
            f"Cache initialisé avec capacité={capacity}, TTL={ttl}s, "
            f"compression={self.compression_method}(niveau={self.compression_level}), "
            f"préchargement prédictif={'activé' if self.enable_predictive_loading else 'désactivé'}"
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de l'entrée à récupérer
            
        Returns:
            Valeur associée à la clé ou None si absente/expirée
        """
        with self.lock:
            # Mise à jour des patterns d'accès
            if self.enable_predictive_loading:
                self._update_access_pattern(key)
            
            # Vérification dans le cache mémoire
            if key in self.memory_cache:
                # Vérification de l'expiration
                if self._is_expired(key):
                    self._remove(key)
                    self.stats["misses"] += 1
                    
                    # Tentative de récupération depuis le cache disque
                    if self.enable_disk_cache:
                        disk_value = self._get_from_disk(key)
                        if disk_value is not None:
                            self.stats["disk_hits"] += 1
                            # Rechargement en mémoire avec un nouveau TTL
                            self.set(key, disk_value)
                            return disk_value
                    
                    return None
                
                # Mise à jour du temps d'accès et de la fréquence
                self.access_times[key] = time.time()
                self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
                self.stats["hits"] += 1
                
                # Décompression si nécessaire
                value = self.memory_cache[key]
                if self.use_compression and isinstance(value, bytes):
                    value = self._decompress_value(value)
                    # Stockage de la valeur décompressée pour les accès futurs
                    self.memory_cache[key] = value
                
                # Préchargement prédictif basé sur le pattern d'accès
                if self.enable_predictive_loading:
                    self._prefetch_related_keys(key)
                
                return value
            
            # Clé non trouvée en mémoire, tentative sur disque
            if self.enable_disk_cache:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    self.stats["disk_hits"] += 1
                    # Chargement en mémoire avec un nouveau TTL
                    self.set(key, disk_value)
                    return disk_value
                self.stats["disk_misses"] += 1
            
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Ajoute ou met à jour une entrée dans le cache.
        
        Args:
            key: Clé de l'entrée
            value: Valeur à stocker
            ttl: Durée de vie spécifique en secondes (optionnel)
        """
        with self.lock:
            # Vérification si le cache est plein et éviction si nécessaire
            if len(self.memory_cache) >= self.capacity and key not in self.memory_cache:
                self._evict()
            
            # Compression si configurée et appropriée (pour les gros objets)
            stored_value = value
            if self.use_compression and self._should_compress(value):
                stored_value = self._compress_value(value)
            
            # Stockage en mémoire
            self.memory_cache[key] = stored_value
            
            # Mise à jour des métadonnées
            current_time = time.time()
            self.access_times[key] = current_time
            self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
            ttl_value = ttl if ttl is not None else self.default_ttl
            self.expiry_times[key] = current_time + ttl_value
            
            # Persistence sur disque
            if self.enable_disk_cache:
                self._save_to_disk(key, value)
    
    def prefetch(self, keys: List[str]) -> None:
        """
        Précharge un ensemble de clés en mémoire.
        
        Args:
            keys: Liste des clés à précharger
        """
        if not self.enable_disk_cache or not keys:
            return
        
        def _prefetch_worker(key_list):
            for k in key_list:
                if k not in self.memory_cache and not self._is_expired(k):
                    disk_value = self._get_from_disk(k)
                    if disk_value is not None:
                        self.set(k, disk_value)
        
        # Filtrer les clés déjà en mémoire
        keys_to_fetch = [k for k in keys if k not in self.memory_cache]
        
        if not keys_to_fetch:
            return
            
        # Préchargement dans un thread séparé pour ne pas bloquer
        if self.prefetch_executor:
            self.prefetch_executor.submit(_prefetch_worker, keys_to_fetch)
    
    def _update_access_pattern(self, key: str) -> None:
        """
        Met à jour les patterns d'accès pour le préchargement prédictif.
        
        Args:
            key: Clé accédée
        """
        # Maintenir une petite fenêtre d'accès récents
        recent_accesses = self.access_pattern.get("recent", [])
        recent_accesses.append(key)
        if len(recent_accesses) > 10:  # Taille de fenêtre fixe
            recent_accesses.pop(0)
        self.access_pattern["recent"] = recent_accesses
        
        # Si nous avons au moins 2 accès, regarder les paires
        if len(recent_accesses) >= 2:
            # Créer une paire (clé précédente -> clé actuelle)
            prev_key = recent_accesses[-2]
            curr_key = recent_accesses[-1]
            
            if prev_key not in self.prefetch_patterns:
                self.prefetch_patterns[prev_key] = {}
            
            # Incrémenter le compteur de cette transition
            if curr_key not in self.prefetch_patterns[prev_key]:
                self.prefetch_patterns[prev_key][curr_key] = 0
            self.prefetch_patterns[prev_key][curr_key] += 1
    
    def _prefetch_related_keys(self, key: str) -> None:
        """
        Précharge les clés probablement liées en fonction des patterns d'accès.
        
        Args:
            key: Clé de référence
        """
        if not self.enable_predictive_loading or key not in self.prefetch_patterns:
            return
        
        # Identifier les clés les plus susceptibles d'être demandées ensuite
        related_keys = []
        transitions = self.prefetch_patterns[key]
        
        # Sélectionner les top 3 transitions les plus fréquentes
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        for next_key, count in sorted_transitions[:3]:
            if count >= 2 and next_key not in self.memory_cache:  # Au moins 2 occurrences
                related_keys.append(next_key)
        
        if related_keys:
            # Précharger en arrière-plan
            self.prefetch(related_keys)
            
            # Mettre à jour les statistiques
            if any(k in self.memory_cache for k in related_keys):
                self.stats["prefetch_hits"] += 1
            else:
                self.stats["prefetch_misses"] += 1
    
    def _compress_value(self, value: Any) -> bytes:
        """
        Compresse une valeur pour le stockage.
        
        Args:
            value: Valeur à compresser
            
        Returns:
            Valeur compressée
        """
        # Sérialiser d'abord
        serialized = pickle.dumps(value)
        original_size = len(serialized)
        
        # Compresser selon la méthode configurée
        if self.compression_method == "gzip":
            compressed = gzip.compress(serialized, compresslevel=self.compression_level)
        else:  # zlib par défaut
            compressed = zlib.compress(serialized, level=self.compression_level)
        
        # Mise à jour des statistiques
        compressed_size = len(compressed)
        self.stats["compression_savings"] += (original_size - compressed_size)
        
        return compressed
    
    def _decompress_value(self, compressed_value: bytes) -> Any:
        """
        Décompresse une valeur du cache.
        
        Args:
            compressed_value: Valeur compressée
            
        Returns:
            Valeur décompressée
        """
        try:
            # Décompresser selon la méthode
            if self.compression_method == "gzip":
                decompressed = gzip.decompress(compressed_value)
            else:  # zlib par défaut
                decompressed = zlib.decompress(compressed_value)
            
            # Désérialiser
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Erreur lors de la décompression: {e}")
            return None
    
    def _should_compress(self, value: Any) -> bool:
        """
        Détermine si une valeur doit être compressée.
        
        Args:
            value: Valeur à évaluer
            
        Returns:
            True si la valeur doit être compressée
        """
        # Ne pas compresser les types primitifs ou petites chaînes
        if isinstance(value, (int, float, bool)) or (isinstance(value, str) and len(value) < 1024):
            return False
            
        # Ne pas compresser les valeurs déjà compressées
        if isinstance(value, bytes):
            return False
            
        # Compresser les dictionnaires, listes, DataFrame ou objets volumineux
        if isinstance(value, (dict, list, pd.DataFrame)) or sys.getsizeof(value) > 4096:
            return True
            
        return False
    
    def clear(self) -> None:
        """
        Vide complètement le cache.
        """
        with self.lock:
            self.memory_cache.clear()
            self.access_times.clear()
            self.expiry_times.clear()
            self.access_frequency.clear()
            self.access_pattern.clear()
            self.prefetch_patterns.clear()
            
            # Réinitialisation des statistiques
            for stat in self.stats:
                self.stats[stat] = 0
            
            logger.info("Cache vidé")
    
    def remove(self, key: str) -> None:
        """
        Supprime une entrée spécifique du cache.
        
        Args:
            key: Clé de l'entrée à supprimer
        """
        with self.lock:
            self._remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Renvoie les statistiques d'utilisation du cache.
        
        Returns:
            Dictionnaire des statistiques
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests) * 100 if total_requests > 0 else 0
            
            stats = {
                **self.stats,
                "total_entries": len(self.memory_cache),
                "capacity": self.capacity,
                "hit_rate": f"{hit_rate:.2f}%",
                "memory_usage_bytes": self._estimate_memory_usage()
            }
            
            return stats
    
    def _is_expired(self, key: str) -> bool:
        """
        Vérifie si une entrée est expirée.
        
        Args:
            key: Clé de l'entrée à vérifier
            
        Returns:
            True si l'entrée est expirée, False sinon
        """
        return time.time() > self.expiry_times.get(key, 0)
    
    def _remove(self, key: str) -> None:
        """
        Supprime une entrée du cache en mémoire.
        
        Args:
            key: Clé de l'entrée à supprimer
        """
        if key in self.memory_cache:
            del self.memory_cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.expiry_times:
            del self.expiry_times[key]
    
    def _evict(self) -> None:
        """
        Évince l'entrée la moins récemment utilisée.
        """
        if not self.memory_cache:
            return
        
        # Recherche de la clé la moins récemment utilisée
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Suppression
        self._remove(lru_key)
        self.stats["evictions"] += 1
        
        logger.debug(f"Éviction de l'entrée: {lru_key}")
    
    def _get_disk_path(self, key: str) -> str:
        """
        Obtient le chemin disque pour une clé.
        
        Args:
            key: Clé à convertir
            
        Returns:
            Chemin du fichier
        """
        if not self.persist_path:
            return ""
        
        # Création d'un nom de fichier sécurisé
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.persist_path, f"{hashed_key}.cache")
    
    def _save_to_disk(self, key: str, value: Any) -> None:
        """
        Sauvegarde une entrée sur disque.
        
        Args:
            key: Clé de l'entrée
            value: Valeur à sauvegarder
        """
        if not self.persist_path:
            return
        
        # Création du répertoire si nécessaire
        os.makedirs(self.persist_path, exist_ok=True)
        
        try:
            file_path = self._get_disk_path(key)
            disk_data = {
                "value": value,
                "expiry": self.expiry_times.get(key, 0)
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(disk_data, f)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du cache sur disque: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """
        Récupère une entrée depuis le cache disque.
        
        Args:
            key: Clé de l'entrée
            
        Returns:
            Valeur associée à la clé ou None si absente/expirée
        """
        if not self.persist_path:
            return None
        
        file_path = self._get_disk_path(key)
        
        try:
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                disk_data = pickle.load(f)
            
            # Vérification de l'expiration
            if time.time() > disk_data.get("expiry", 0):
                os.remove(file_path)
                return None
            
            return disk_data.get("value")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du cache disque: {e}")
            return None
    
    def _load_disk_cache(self) -> None:
        """
        Charge les entrées du cache disque en mémoire au démarrage.
        """
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        try:
            file_count = 0
            loaded_count = 0
            
            for filename in os.listdir(self.persist_path):
                if not filename.endswith('.cache'):
                    continue
                
                file_count += 1
                file_path = os.path.join(self.persist_path, filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        disk_data = pickle.load(f)
                    
                    # Vérification de l'expiration
                    if time.time() > disk_data.get("expiry", 0):
                        os.remove(file_path)
                        continue
                    
                    # Extraction de la clé originale si possible, sinon utiliser le nom du fichier
                    key = filename.split('.')[0]
                    value = disk_data.get("value")
                    
                    # Chargement en mémoire si la capacité le permet
                    if len(self.memory_cache) < self.capacity:
                        self.memory_cache[key] = value
                        self.access_times[key] = time.time()
                        self.expiry_times[key] = disk_data.get("expiry", 0)
                        loaded_count += 1
                
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du fichier cache {filename}: {e}")
            
            logger.info(f"Cache disque chargé: {loaded_count}/{file_count} entrées chargées en mémoire")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du cache disque: {e}")
    
    def _estimate_memory_usage(self) -> int:
        """
        Estime la consommation mémoire du cache.
        
        Returns:
            Estimation de la taille en octets
        """
        # Estimation simplifiée (peut être améliorée avec des bibliothèques spécialisées)
        total_size = 0
        
        try:
            # Taille des données
            for key, value in self.memory_cache.items():
                # Taille estimée de la clé
                key_size = len(key) * 2  # Unicode ~2 octets par caractère
                
                # Taille estimée de la valeur
                if isinstance(value, (str, bytes)):
                    value_size = len(value)
                elif isinstance(value, pd.DataFrame):
                    value_size = value.memory_usage(deep=True).sum()
                else:
                    # Estimation grossière pour les autres types
                    try:
                        value_size = len(pickle.dumps(value))
                    except:
                        value_size = 100  # Valeur par défaut
                
                total_size += key_size + value_size
            
            # Taille des métadonnées
            metadata_size = (len(self.access_times) + len(self.expiry_times)) * 16  # ~16 octets par timestamp
            
            return total_size + metadata_size
        
        except Exception:
            # Fallback si l'estimation échoue
            return len(self.memory_cache) * 1024  # Estimation grossière: ~1KB par entrée
    
    def purge_expired(self) -> int:
        """
        Supprime toutes les entrées expirées du cache.
        
        Returns:
            Nombre d'entrées supprimées
        """
        with self.lock:
            expired_keys = [key for key in list(self.memory_cache.keys()) if self._is_expired(key)]
            
            for key in expired_keys:
                self._remove(key)
            
            logger.info(f"{len(expired_keys)} entrées expirées purgées du cache")
            return len(expired_keys)

def cached(ttl: Optional[int] = None, key_fn: Optional[Callable] = None):
    """
    Décorateur pour mettre en cache les résultats de fonctions.
    
    Args:
        ttl: Durée de vie du cache en secondes
        key_fn: Fonction pour générer la clé de cache personnalisée
        
    Returns:
        Fonction décorée
    """
    def decorator(func):
        # Création d'un gestionnaire de cache local si nécessaire
        cache_name = f"func_cache_{func.__name__}"
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        cache = CacheManager(capacity=100, ttl=ttl or 300, persist_path=cache_dir)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Génération de la clé de cache
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Clé par défaut basée sur la fonction et ses arguments
                arg_key = str(args)
                kwarg_key = json.dumps(kwargs, sort_keys=True)
                cache_key = f"{func.__module__}.{func.__name__}:{arg_key}:{kwarg_key}"
            
            # Tentative de récupération depuis le cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Exécution de la fonction et mise en cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        # Ajout d'une méthode pour vider le cache
        wrapper.clear_cache = cache.clear
        
        return wrapper
    
    return decorator 