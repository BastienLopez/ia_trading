"""
Module de mise en cache intelligente des données.

Ce module implémente :
- Stratégie LRU (Least Recently Used)
- Préchargement des données fréquentes
- Compression des données historiques
- Gestion de la cohérence des données
"""

import logging
import os
import time
import threading
import pickle
import zstandard as zstd
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SmartCache:
    """
    Cache intelligent avec stratégie LRU, préchargement et compression.
    
    Caractéristiques:
    - Stratégie d'éviction LRU (Least Recently Used)
    - Préchargement automatique des données fréquemment utilisées
    - Compression des données historiques pour économiser la mémoire
    - Gestion de la cohérence des données avec horodatage et validation
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,  # Durée de vie en secondes (1 heure par défaut)
        compression_level: int = 3,  # Niveau de compression (0-22)
        preload_threshold: int = 5,  # Nombre d'accès pour déclencher le préchargement
        cache_dir: Optional[str] = None,
        persist: bool = True,
    ):
        """
        Initialise le cache intelligent.
        
        Args:
            max_size: Taille maximale du cache (nombre d'éléments)
            ttl: Durée de vie des éléments en secondes
            compression_level: Niveau de compression (0-22)
            preload_threshold: Nombre d'accès pour déclencher le préchargement
            cache_dir: Répertoire pour la persistance du cache
            persist: Activer la persistance du cache sur disque
        """
        self.max_size = max_size
        self.ttl = ttl
        self.compression_level = compression_level
        self.preload_threshold = preload_threshold
        self.persist = persist
        
        # Cache principal (OrderedDict pour LRU)
        self.cache = OrderedDict()
        
        # Statistiques d'accès pour le préchargement
        self.access_stats = {}
        
        # Verrou pour la thread-safety
        self.lock = threading.RLock()
        
        # Répertoire de cache
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".ai_trading_cache")
        
        if self.persist and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Charger le cache persistant
        if self.persist:
            self._load_persistent_cache()
        
        # Démarrer le thread de maintenance
        self.maintenance_thread = threading.Thread(target=self._maintenance_task, daemon=True)
        self.maintenance_thread.start()
        
        logger.info(f"Cache intelligent initialisé avec taille max: {max_size}, TTL: {ttl}s")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de l'élément
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur associée à la clé ou valeur par défaut
        """
        with self.lock:
            # Mettre à jour les statistiques d'accès
            self._update_access_stats(key)
            
            # Vérifier si la clé existe
            if key not in self.cache:
                return default
            
            # Récupérer l'élément
            timestamp, compressed, value = self.cache[key]
            
            # Vérifier si l'élément est expiré
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                return default
            
            # Déplacer l'élément à la fin (LRU)
            self.cache.move_to_end(key)
            
            # Décompresser si nécessaire
            if compressed:
                value = self._decompress(value)
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Ajoute ou met à jour une valeur dans le cache.
        
        Args:
            key: Clé de l'élément
            value: Valeur à stocker
            ttl: Durée de vie spécifique pour cet élément (None = valeur par défaut)
        """
        with self.lock:
            # Compresser les données volumineuses
            compressed = False
            original_value = value
            
            # Déterminer si la valeur doit être compressée
            if self._should_compress(value):
                value = self._compress(value)
                compressed = True
            
            # Ajouter l'élément au cache
            self.cache[key] = (time.time(), compressed, value)
            
            # Déplacer l'élément à la fin (LRU)
            self.cache.move_to_end(key)
            
            # Appliquer la stratégie LRU si nécessaire
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            
            # Persister le cache si nécessaire
            if self.persist:
                self._persist_item(key, (time.time(), compressed, value))
    
    def delete(self, key: str) -> bool:
        """
        Supprime un élément du cache.
        
        Args:
            key: Clé de l'élément à supprimer
            
        Returns:
            bool: True si l'élément a été supprimé, False sinon
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                
                # Supprimer du cache persistant
                if self.persist:
                    cache_file = os.path.join(self.cache_dir, self._hash_key(key))
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                
                return True
            return False
    
    def clear(self) -> None:
        """Vide le cache."""
        with self.lock:
            self.cache.clear()
            self.access_stats.clear()
            
            # Nettoyer le cache persistant
            if self.persist:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
    
    def preload(self, key: str, data_loader: Callable, force: bool = False) -> None:
        """
        Précharge des données dans le cache.
        
        Args:
            key: Clé pour les données préchargées
            data_loader: Fonction pour charger les données
            force: Forcer le préchargement même si la clé existe
        """
        with self.lock:
            if force or key not in self.cache:
                try:
                    data = data_loader()
                    self.set(key, data)
                    logger.debug(f"Données préchargées pour la clé: {key}")
                except Exception as e:
                    logger.error(f"Erreur lors du préchargement de {key}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le cache.
        
        Returns:
            Dict: Statistiques du cache
        """
        with self.lock:
            compressed_count = sum(1 for _, compressed, _ in self.cache.values() if compressed)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "compressed_items": compressed_count,
                "compression_ratio": compressed_count / max(1, len(self.cache)),
                "popular_keys": sorted(self.access_stats.items(), key=lambda x: x[1], reverse=True)[:10],
                "memory_usage_estimate": self._estimate_memory_usage()
            }
    
    def _update_access_stats(self, key: str) -> None:
        """Met à jour les statistiques d'accès pour une clé."""
        if key in self.access_stats:
            self.access_stats[key] += 1
        else:
            self.access_stats[key] = 1
    
    def _should_compress(self, value: Any) -> bool:
        """
        Détermine si une valeur doit être compressée.
        
        Args:
            value: Valeur à évaluer
            
        Returns:
            bool: True si la valeur doit être compressée
        """
        # Compresser les DataFrames volumineux
        if isinstance(value, pd.DataFrame) and len(value) > 1000:
            return True
        
        # Compresser les tableaux NumPy volumineux
        if isinstance(value, np.ndarray) and value.size > 10000:
            return True
        
        # Compresser les listes et dictionnaires volumineux
        if isinstance(value, (list, dict)) and len(value) > 5000:
            return True
        
        return False
    
    def _compress(self, data: Any) -> bytes:
        """
        Compresse des données.
        
        Args:
            data: Données à compresser
            
        Returns:
            bytes: Données compressées
        """
        try:
            # Sérialiser avec pickle
            serialized = pickle.dumps(data)
            
            # Compresser avec zstd
            compressor = zstd.ZstdCompressor(level=self.compression_level)
            compressed = compressor.compress(serialized)
            
            return compressed
        except Exception as e:
            logger.warning(f"Erreur lors de la compression: {e}. Utilisation des données non compressées.")
            return pickle.dumps(data)
    
    def _decompress(self, data: bytes) -> Any:
        """
        Décompresse des données.
        
        Args:
            data: Données compressées
            
        Returns:
            Any: Données décompressées
        """
        try:
            # Décompresser avec zstd
            decompressor = zstd.ZstdDecompressor()
            decompressed = decompressor.decompress(data)
            
            # Désérialiser avec pickle
            return pickle.loads(decompressed)
        except Exception as e:
            logger.warning(f"Erreur lors de la décompression: {e}. Tentative de désérialisation directe.")
            try:
                return pickle.loads(data)
            except:
                logger.error("Échec de la récupération des données.")
                return None
    
    def _hash_key(self, key: str) -> str:
        """
        Génère un hash pour une clé.
        
        Args:
            key: Clé à hasher
            
        Returns:
            str: Hash de la clé
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def _persist_item(self, key: str, item: Tuple) -> None:
        """
        Persiste un élément du cache sur disque.
        
        Args:
            key: Clé de l'élément
            item: Élément à persister (timestamp, compressed, value)
        """
        if not self.persist:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, self._hash_key(key))
            with open(cache_file, 'wb') as f:
                pickle.dump((key, item), f)
        except Exception as e:
            logger.error(f"Erreur lors de la persistance de {key}: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Charge le cache persistant depuis le disque."""
        if not os.path.exists(self.cache_dir):
            return
        
        try:
            # Charger chaque fichier de cache
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            key, item = pickle.load(f)
                            timestamp, compressed, value = item
                            
                            # Vérifier si l'élément est expiré
                            if time.time() - timestamp <= self.ttl:
                                self.cache[key] = item
                    except Exception as e:
                        logger.warning(f"Erreur lors du chargement du fichier {filename}: {e}")
            
            logger.info(f"Cache chargé depuis {self.cache_dir}: {len(self.cache)} éléments")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du cache persistant: {e}")
    
    def _maintenance_task(self) -> None:
        """Tâche de maintenance périodique du cache."""
        while True:
            time.sleep(300)  # Exécuter toutes les 5 minutes
            
            with self.lock:
                # Nettoyer les éléments expirés
                current_time = time.time()
                expired_keys = [
                    key for key, (timestamp, _, _) in self.cache.items()
                    if current_time - timestamp > self.ttl
                ]
                
                for key in expired_keys:
                    self.delete(key)
                
                # Précharger les données fréquemment utilisées
                for key, count in self.access_stats.items():
                    if count >= self.preload_threshold and key not in self.cache:
                        # Nous ne pouvons pas précharger directement car nous ne connaissons pas la fonction de chargement
                        # Cette logique serait implémentée par l'utilisateur via la méthode preload()
                        pass
                
                # Persister le cache complet périodiquement
                if self.persist:
                    for key, item in self.cache.items():
                        self._persist_item(key, item)
    
    def _estimate_memory_usage(self) -> int:
        """
        Estime l'utilisation mémoire du cache.
        
        Returns:
            int: Estimation de la mémoire utilisée en octets
        """
        total_size = 0
        for key, (timestamp, compressed, value) in self.cache.items():
            # Taille de la clé
            total_size += len(key.encode())
            
            # Taille de la valeur
            if compressed:
                total_size += len(value)
            else:
                try:
                    total_size += len(pickle.dumps(value))
                except:
                    # Estimation approximative si pickle échoue
                    total_size += sys.getsizeof(value)
        
        return total_size


class DataCache(SmartCache):
    """
    Extension du SmartCache spécialisée pour les données financières.
    
    Ajoute:
    - Gestion des séries temporelles
    - Préchargement intelligent basé sur les patterns d'accès
    - Validation de la cohérence des données
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        compression_level: int = 3,
        preload_threshold: int = 5,
        cache_dir: Optional[str] = None,
        persist: bool = True,
        data_sources: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialise le cache de données.
        
        Args:
            max_size: Taille maximale du cache
            ttl: Durée de vie des éléments
            compression_level: Niveau de compression
            preload_threshold: Seuil de préchargement
            cache_dir: Répertoire de cache
            persist: Activer la persistance
            data_sources: Dictionnaire de sources de données {nom: fonction_chargement}
        """
        super().__init__(max_size, ttl, compression_level, preload_threshold, cache_dir, persist)
        
        # Sources de données pour le préchargement
        self.data_sources = data_sources or {}
        
        # Patterns d'accès temporels
        self.time_patterns = {}
        
        # Métadonnées pour la validation de cohérence
        self.metadata = {}
        
        # Démarrer le thread de préchargement
        self.preload_thread = threading.Thread(target=self._preload_task, daemon=True)
        self.preload_thread.start()
    
    def get_timeseries(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        source: str = "default"
    ) -> pd.DataFrame:
        """
        Récupère des données de séries temporelles du cache.
        
        Args:
            symbol: Symbole de l'actif
            start_date: Date de début
            end_date: Date de fin
            interval: Intervalle de temps
            source: Source de données
            
        Returns:
            pd.DataFrame: Données de série temporelle
        """
        # Normaliser les dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Créer la clé de cache
        cache_key = f"timeseries:{symbol}:{interval}:{source}"
        
        # Enregistrer le pattern d'accès temporel
        self._record_time_pattern(cache_key, start_date, end_date)
        
        # Vérifier si les données sont dans le cache
        cached_data = self.get(cache_key)
        
        if cached_data is not None:
            # Filtrer les données selon la plage de dates demandée
            if isinstance(cached_data, pd.DataFrame) and 'date' in cached_data.columns:
                filtered_data = cached_data[
                    (cached_data['date'] >= start_date) & 
                    (cached_data['date'] <= end_date)
                ]
                
                # Si toutes les données demandées sont dans le cache
                if len(filtered_data) > 0 and filtered_data['date'].min() <= start_date and filtered_data['date'].max() >= end_date:
                    return filtered_data
        
        # Si les données ne sont pas dans le cache ou sont incomplètes
        if source in self.data_sources:
            # Charger les données depuis la source
            data_loader = self.data_sources[source]
            data = data_loader(symbol, start_date, end_date, interval)
            
            # Stocker dans le cache
            self.set(cache_key, data)
            
            # Enregistrer les métadonnées pour la validation de cohérence
            self._update_metadata(cache_key, {
                'symbol': symbol,
                'interval': interval,
                'source': source,
                'last_update': datetime.now(),
                'start_date': start_date,
                'end_date': end_date,
                'row_count': len(data)
            })
            
            # Filtrer selon la plage demandée
            if isinstance(data, pd.DataFrame) and 'date' in data.columns:
                return data[
                    (data['date'] >= start_date) & 
                    (data['date'] <= end_date)
                ]
            
            return data
        
        # Si la source n'est pas disponible
        logger.warning(f"Source de données '{source}' non disponible")
        return pd.DataFrame()
    
    def invalidate(self, pattern: str) -> int:
        """
        Invalide les entrées de cache correspondant à un pattern.
        
        Args:
            pattern: Pattern à invalider (ex: "timeseries:BTC:*")
            
        Returns:
            int: Nombre d'entrées invalidées
        """
        with self.lock:
            count = 0
            keys_to_delete = []
            
            # Trouver les clés correspondant au pattern
            for key in self.cache.keys():
                if self._match_pattern(key, pattern):
                    keys_to_delete.append(key)
            
            # Supprimer les clés
            for key in keys_to_delete:
                self.delete(key)
                count += 1
            
            return count
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """
        Vérifie si une clé correspond à un pattern.
        
        Args:
            key: Clé à vérifier
            pattern: Pattern avec wildcards (*)
            
        Returns:
            bool: True si la clé correspond au pattern
        """
        # Convertir le pattern en expression régulière
        import re
        regex = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex}$", key))
    
    def _record_time_pattern(self, key: str, start_date: datetime, end_date: datetime) -> None:
        """
        Enregistre un pattern d'accès temporel.
        
        Args:
            key: Clé d'accès
            start_date: Date de début
            end_date: Date de fin
        """
        with self.lock:
            if key not in self.time_patterns:
                self.time_patterns[key] = []
            
            # Enregistrer l'accès
            self.time_patterns[key].append({
                'timestamp': datetime.now(),
                'start_date': start_date,
                'end_date': end_date,
                'duration': (end_date - start_date).days
            })
            
            # Limiter le nombre d'entrées
            if len(self.time_patterns[key]) > 100:
                self.time_patterns[key] = self.time_patterns[key][-100:]
    
    def _update_metadata(self, key: str, metadata: Dict) -> None:
        """
        Met à jour les métadonnées pour une clé.
        
        Args:
            key: Clé de cache
            metadata: Métadonnées à stocker
        """
        with self.lock:
            self.metadata[key] = metadata
    
    def _preload_task(self) -> None:
        """Tâche de préchargement basée sur les patterns d'accès."""
        while True:
            time.sleep(600)  # Exécuter toutes les 10 minutes
            
            with self.lock:
                now = datetime.now()
                
                # Analyser les patterns pour chaque clé
                for key, patterns in self.time_patterns.items():
                    if not patterns:
                        continue
                    
                    # Ne considérer que les accès récents
                    recent_patterns = [p for p in patterns if (now - p['timestamp']).total_seconds() < 86400]
                    
                    if not recent_patterns:
                        continue
                    
                    # Calculer les moyennes
                    avg_duration = sum(p['duration'] for p in recent_patterns) / len(recent_patterns)
                    
                    # Extraire les composants de la clé
                    parts = key.split(':')
                    if len(parts) >= 4 and parts[0] == 'timeseries':
                        symbol = parts[1]
                        interval = parts[2]
                        source = parts[3]
                        
                        # Prévoir la prochaine plage de dates probable
                        last_pattern = max(recent_patterns, key=lambda p: p['timestamp'])
                        predicted_start = last_pattern['start_date']
                        predicted_end = last_pattern['end_date'] + timedelta(days=int(avg_duration * 0.5))
                        
                        # Précharger si la source est disponible
                        if source in self.data_sources:
                            logger.debug(f"Préchargement prévu pour {symbol} de {predicted_start} à {predicted_end}")
                            
                            try:
                                data_loader = self.data_sources[source]
                                data = data_loader(symbol, predicted_start, predicted_end, interval)
                                self.set(key, data)
                                
                                logger.debug(f"Données préchargées pour {key}")
                            except Exception as e:
                                logger.error(f"Erreur lors du préchargement de {key}: {e}")
    
    def validate_data(self, key: str) -> Tuple[bool, Optional[str]]:
        """
        Valide la cohérence des données en cache.
        
        Args:
            key: Clé à valider
            
        Returns:
            Tuple[bool, Optional[str]]: (valide, message d'erreur)
        """
        with self.lock:
            # Vérifier si les métadonnées existent
            if key not in self.metadata:
                return False, "Pas de métadonnées disponibles"
            
            # Vérifier si les données existent
            if key not in self.cache:
                return False, "Données non présentes dans le cache"
            
            # Récupérer les données et métadonnées
            _, compressed, value = self.cache[key]
            metadata = self.metadata[key]
            
            # Décompresser si nécessaire
            data = self._decompress(value) if compressed else value
            
            # Vérifier le type de données
            if not isinstance(data, pd.DataFrame):
                return False, "Les données ne sont pas un DataFrame"
            
            # Vérifier la cohérence
            if len(data) != metadata.get('row_count', 0):
                return False, f"Nombre de lignes incohérent: {len(data)} vs {metadata.get('row_count')}"
            
            # Vérifier les dates si c'est une série temporelle
            if 'date' in data.columns:
                min_date = data['date'].min()
                max_date = data['date'].max()
                
                if min_date > metadata.get('start_date', min_date):
                    return False, f"Date de début incohérente: {min_date} > {metadata.get('start_date')}"
                
                if max_date < metadata.get('end_date', max_date):
                    return False, f"Date de fin incohérente: {max_date} < {metadata.get('end_date')}"
            
            return True, None 