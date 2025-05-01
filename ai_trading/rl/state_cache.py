import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import collections
import threading
import pickle
import hashlib
import os
import json

class StateCache:
    """
    Système de cache pour les états fréquents en RL.
    
    Permet de:
    1. Éviter de recalculer les actions/valeurs pour des états fréquemment visités
    2. Maintenir des statistiques sur les états fréquents
    3. Optimiser la mémoire en gardant en cache seulement les états utiles
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        similarity_threshold: float = 0.001,
        hash_func: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        ttl: Optional[int] = None,
        enable_disk_cache: bool = False
    ):
        """
        Initialise le cache d'états.
        
        Args:
            capacity: Capacité maximale du cache (nombre d'entrées)
            similarity_threshold: Seuil de similarité pour considérer deux états identiques
            hash_func: Fonction de hachage personnalisée
            cache_dir: Répertoire pour la persistance du cache sur disque
            ttl: Durée de vie des entrées en secondes (None = pas d'expiration)
            enable_disk_cache: Activer la persistance sur disque
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.hash_func = hash_func
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.enable_disk_cache = enable_disk_cache
        
        # Cache principal (mémoire)
        self.cache = collections.OrderedDict()  # {hash: (state, value, metadata)}
        self.hits = 0
        self.misses = 0
        self.total_queries = 0
        
        # Verrou pour thread-safety
        self.lock = threading.RLock()
        
        # Créer le répertoire de cache si nécessaire
        if self.enable_disk_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Métriques par état
        self.state_metrics = {}  # {hash: {hits: int, last_access: float, ...}}
        
        # Pour mémoriser le dernier state et son hash
        self.last_state = None
        self.last_hash = None
    
    def _hash_state(self, state) -> str:
        """
        Génère un hash pour un état.
        
        Args:
            state: État à hacher
            
        Returns:
            Hash de l'état
        """
        if self.hash_func is not None:
            return self.hash_func(state)
        
        # Utiliser le hash personnalisé pour les tenseurs PyTorch
        if isinstance(state, torch.Tensor):
            # Détacher et convertir en numpy
            if state.requires_grad:
                state_data = state.detach().cpu().numpy()
            else:
                state_data = state.cpu().numpy()
            
            # Créer un hash basé sur les données
            state_bytes = state_data.tobytes()
            return hashlib.md5(state_bytes).hexdigest()
        
        # Utiliser le hash personnalisé pour les arrays numpy
        elif isinstance(state, np.ndarray):
            state_bytes = state.tobytes()
            return hashlib.md5(state_bytes).hexdigest()
        
        # Autres types: essayer pickle puis hash
        else:
            try:
                state_bytes = pickle.dumps(state)
                return hashlib.md5(state_bytes).hexdigest()
            except:
                # Fallback: hash par défaut
                return str(hash(str(state)))
    
    def _states_similar(self, state1, state2) -> bool:
        """
        Vérifie si deux états sont similaires.
        
        Args:
            state1: Premier état
            state2: Deuxième état
            
        Returns:
            True si les états sont similaires
        """
        # Cas PyTorch
        if isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor):
            # Assurer le même device
            if state1.device != state2.device:
                state2 = state2.to(state1.device)
            
            # Vérifier si les dimensions correspondent
            if state1.shape != state2.shape:
                return False
            
            # Calculer la distance
            diff = torch.abs(state1 - state2).mean().item()
            return diff < self.similarity_threshold
        
        # Cas NumPy
        elif isinstance(state1, np.ndarray) and isinstance(state2, np.ndarray):
            # Vérifier si les dimensions correspondent
            if state1.shape != state2.shape:
                return False
            
            # Calculer la distance
            diff = np.abs(state1 - state2).mean()
            return diff < self.similarity_threshold
        
        # Autres cas: égalité stricte
        return state1 == state2
    
    def _get_disk_cache_path(self, state_hash: str) -> str:
        """
        Génère le chemin du fichier de cache.
        
        Args:
            state_hash: Hash de l'état
            
        Returns:
            Chemin du fichier
        """
        if not self.cache_dir:
            return None
        
        # Utiliser les 2 premiers caractères comme sous-dossier pour éviter d'avoir trop de fichiers
        subdir = state_hash[:2]
        cache_subdir = os.path.join(self.cache_dir, subdir)
        os.makedirs(cache_subdir, exist_ok=True)
        
        return os.path.join(cache_subdir, f"{state_hash[2:]}.cache")
    
    def _save_to_disk(self, state_hash: str, state, value, metadata: Dict):
        """
        Enregistre une entrée du cache sur disque.
        
        Args:
            state_hash: Hash de l'état
            state: État
            value: Valeur associée
            metadata: Métadonnées
        """
        if not self.enable_disk_cache or not self.cache_dir:
            return
        
        cache_path = self._get_disk_cache_path(state_hash)
        if not cache_path:
            return
        
        try:
            data = {
                "value": value,
                "metadata": metadata,
                "timestamp": time.time()
            }
            
            # État des tenseurs PyTorch: convertir en numpy
            if isinstance(state, torch.Tensor):
                data["state"] = state.detach().cpu().numpy()
                data["is_torch"] = True
            else:
                data["state"] = state
                data["is_torch"] = False
            
            # Enregistrer en pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Erreur d'enregistrement sur disque: {e}")
    
    def _load_from_disk(self, state_hash: str) -> Optional[Tuple]:
        """
        Charge une entrée du cache depuis le disque.
        
        Args:
            state_hash: Hash de l'état
            
        Returns:
            Tuple (state, value, metadata) ou None si non trouvé
        """
        if not self.enable_disk_cache or not self.cache_dir:
            return None
        
        cache_path = self._get_disk_cache_path(state_hash)
        if not cache_path or not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Vérifier l'expiration si TTL défini
            if self.ttl is not None:
                timestamp = data.get("timestamp", 0)
                if time.time() - timestamp > self.ttl:
                    # Expiré
                    os.remove(cache_path)
                    return None
            
            state = data["state"]
            value = data["value"]
            metadata = data.get("metadata", {})
            
            # Convertir en tensor PyTorch si c'était le format d'origine
            if data.get("is_torch", False) and isinstance(state, np.ndarray):
                state = torch.tensor(state)
            
            return (state, value, metadata)
        except Exception as e:
            print(f"Erreur de chargement depuis le disque: {e}")
            # Supprimer le fichier corrompu
            try:
                os.remove(cache_path)
            except:
                pass
            return None
    
    def get(self, state, compute_fn: Optional[Callable] = None) -> Tuple[Any, Dict]:
        """
        Récupère une valeur du cache pour un état donné.
        Si la valeur n'est pas en cache et compute_fn est fourni,
        calcule la valeur et l'ajoute au cache.
        
        Args:
            state: État pour lequel chercher une valeur
            compute_fn: Fonction de calcul à utiliser si la valeur n'est pas en cache
            
        Returns:
            (valeur, info)
        """
        with self.lock:
            self.total_queries += 1
            
            # Calculer le hash de l'état actuel
            state_hash = self._hash_state(state)
            
            # Chercher d'abord le hash exact
            cache_entry = self.cache.get(state_hash)
            if cache_entry:
                cached_state, value, metadata = cache_entry
                
                # Vérifier la similarité
                if self._states_similar(state, cached_state):
                    # Hit !
                    self.hits += 1
                    
                    # Mettre à jour la position dans le cache (LRU)
                    # Supprimer puis réinsérer pour marquer comme récemment utilisé
                    self.cache.pop(state_hash)
                    self.cache[state_hash] = (cached_state, value, metadata)
                    
                    # Mettre à jour les métriques
                    if state_hash in self.state_metrics:
                        self.state_metrics[state_hash]["hits"] += 1
                        self.state_metrics[state_hash]["last_access"] = time.time()
                    
                    return value, {"cache_hit": True, "hash": state_hash}
            
            # Recherche par similarité uniquement si activée explicitement
            search_similar = getattr(self, 'search_similar', False)
            if search_similar:
                # Si pas trouvé par hash exact, chercher un état similaire
                for h, entry in list(self.cache.items()):
                    cached_state, cached_value, cached_metadata = entry
                    if self._states_similar(state, cached_state):
                        # Hit avec un état similaire !
                        self.hits += 1
                        
                        # Mettre à jour la position dans le cache (LRU)
                        self.cache.pop(h)
                        self.cache[h] = (cached_state, cached_value, cached_metadata)
                        
                        # Mettre à jour les métriques
                        if h in self.state_metrics:
                            self.state_metrics[h]["hits"] += 1
                            self.state_metrics[h]["similar_hits"] = self.state_metrics[h].get("similar_hits", 0) + 1
                            self.state_metrics[h]["last_access"] = time.time()
                        
                        return cached_value, {"cache_hit": True, "hash": h, "similar": True}
            
            # Chercher dans le cache disque
            disk_entry = self._load_from_disk(state_hash)
            if disk_entry:
                disk_state, disk_value, disk_metadata = disk_entry
                
                # Vérifier la similarité
                if self._states_similar(state, disk_state):
                    # Hit disque !
                    self.hits += 1
                    
                    # Ajouter au cache mémoire
                    self.cache[state_hash] = (disk_state, disk_value, disk_metadata)
                    
                    # Assurer la capacité maximale
                    if len(self.cache) > self.capacity:
                        # Supprimer l'entrée la plus ancienne (premier élément)
                        oldest_key = next(iter(self.cache))
                        self.cache.pop(oldest_key)
                    
                    # Mettre à jour les métriques
                    if state_hash in self.state_metrics:
                        self.state_metrics[state_hash]["hits"] += 1
                        self.state_metrics[state_hash]["disk_hits"] = self.state_metrics[state_hash].get("disk_hits", 0) + 1
                        self.state_metrics[state_hash]["last_access"] = time.time()
                    else:
                        self.state_metrics[state_hash] = {
                            "hits": 1,
                            "disk_hits": 1,
                            "last_access": time.time()
                        }
                    
                    return disk_value, {"cache_hit": True, "disk_hit": True, "hash": state_hash}
            
            # Miss - calculer la valeur si une fonction est fournie
            self.misses += 1
            
            if compute_fn is None:
                return None, {"cache_hit": False, "hash": state_hash}
            
            # Calculer la valeur
            value = compute_fn(state)
            
            # Ajouter au cache
            metadata = {
                "created_at": time.time(),
                "updated_at": time.time(),
                "access_count": 1
            }
            
            self.cache[state_hash] = (state, value, metadata)
            
            # Assurer la capacité maximale
            if len(self.cache) > self.capacity:
                # Supprimer l'entrée la plus ancienne (premier élément)
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
            
            # Mettre à jour les métriques
            self.state_metrics[state_hash] = {
                "hits": 0,
                "misses": 1,
                "last_access": time.time(),
                "created_at": time.time()
            }
            
            # Enregistrer sur disque si activé
            if self.enable_disk_cache:
                self._save_to_disk(state_hash, state, value, metadata)
            
            return value, {"cache_hit": False, "computed": True, "hash": state_hash}
    
    def put(self, state, value, metadata: Optional[Dict] = None) -> str:
        """
        Ajoute ou met à jour une entrée dans le cache.
        
        Args:
            state: État
            value: Valeur associée
            metadata: Métadonnées additionnelles
            
        Returns:
            Hash de l'état
        """
        with self.lock:
            state_hash = self._hash_state(state)
            
            # Métadonnées par défaut
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "updated_at": time.time(),
            })
            
            # Si l'entrée existe déjà, mettre à jour les métriques
            if state_hash in self.cache:
                old_state, old_value, old_metadata = self.cache[state_hash]
                metadata["access_count"] = old_metadata.get("access_count", 0) + 1
                metadata["created_at"] = old_metadata.get("created_at", time.time())
            else:
                metadata["access_count"] = 1
                metadata["created_at"] = time.time()
            
            # Mettre à jour le cache
            self.cache[state_hash] = (state, value, metadata)
            
            # Assurer la capacité maximale
            if len(self.cache) > self.capacity:
                # Supprimer l'entrée la plus ancienne (premier élément)
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
            
            # Mettre à jour les métriques de l'état
            if state_hash in self.state_metrics:
                self.state_metrics[state_hash]["updates"] = self.state_metrics[state_hash].get("updates", 0) + 1
                self.state_metrics[state_hash]["last_update"] = time.time()
            else:
                self.state_metrics[state_hash] = {
                    "hits": 0,
                    "misses": 0,
                    "updates": 1,
                    "last_update": time.time(),
                    "created_at": time.time()
                }
            
            # Enregistrer sur disque si activé
            if self.enable_disk_cache:
                self._save_to_disk(state_hash, state, value, metadata)
            
            return state_hash
    
    def remove(self, state_or_hash) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            state_or_hash: État ou hash de l'état à supprimer
            
        Returns:
            True si supprimé avec succès
        """
        with self.lock:
            # Déterminer le hash
            if isinstance(state_or_hash, str):
                state_hash = state_or_hash
                # Supprimer directement du cache mémoire
                removed = state_hash in self.cache
                if removed:
                    self.cache.pop(state_hash)
            else:
                # C'est un état, chercher par hash et par similarité
                state = state_or_hash
                state_hash = self._hash_state(state)
                
                # Essayer par hash direct
                removed = state_hash in self.cache
                if removed:
                    self.cache.pop(state_hash)
                else:
                    # Chercher un état similaire
                    for h, entry in list(self.cache.items()):
                        cached_state, _, _ = entry
                        if self._states_similar(state, cached_state):
                            self.cache.pop(h)
                            removed = True
                            state_hash = h
                            break
            
            # Supprimer du disque si activé
            if removed and self.enable_disk_cache:
                cache_path = self._get_disk_cache_path(state_hash)
                if cache_path and os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except:
                        pass
            
            # Supprimer les métriques
            if removed and state_hash in self.state_metrics:
                del self.state_metrics[state_hash]
            
            return removed
    
    def clear(self):
        """
        Vide entièrement le cache.
        """
        with self.lock:
            # Vider le cache mémoire
            self.cache.clear()
            self.state_metrics.clear()
            
            # Réinitialiser les compteurs
            self.hits = 0
            self.misses = 0
            self.total_queries = 0
            
            # Vider le cache disque
            if self.enable_disk_cache and self.cache_dir:
                try:
                    import shutil
                    shutil.rmtree(self.cache_dir)
                    os.makedirs(self.cache_dir, exist_ok=True)
                except Exception as e:
                    print(f"Erreur lors de la suppression du cache disque: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de performance du cache.
        
        Returns:
            Dictionnaire de métriques
        """
        with self.lock:
            metrics = {
                "hits": self.hits,
                "misses": self.misses,
                "total_queries": self.total_queries,
                "hit_rate": self.hits / max(1, self.total_queries),
                "memory_entries": len(self.cache),
                "capacity": self.capacity,
                "utilization": len(self.cache) / max(1, self.capacity)
            }
            
            # Ajouter des métriques sur les états les plus fréquents
            if self.state_metrics:
                # Top 10 des états les plus consultés
                top_hits = sorted(
                    [(h, m["hits"]) for h, m in self.state_metrics.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                metrics["top_states"] = [{"hash": h, "hits": hits} for h, hits in top_hits]
            
            return metrics
    
    def get_state_info(self, state_or_hash) -> Optional[Dict]:
        """
        Récupère les infos sur un état spécifique.
        
        Args:
            state_or_hash: État ou hash de l'état
            
        Returns:
            Infos sur l'état ou None si non trouvé
        """
        with self.lock:
            # Déterminer le hash
            if isinstance(state_or_hash, str):
                state_hash = state_or_hash
            else:
                state_hash = self._hash_state(state_or_hash)
            
            # Récupérer les métriques
            metrics = self.state_metrics.get(state_hash)
            
            # Vérifier si présent dans le cache
            cache_entry = self.cache.get(state_hash)
            
            if metrics or cache_entry:
                info = {
                    "hash": state_hash,
                    "in_memory_cache": cache_entry is not None
                }
                
                if metrics:
                    info.update(metrics)
                
                if cache_entry:
                    _, _, metadata = cache_entry
                    info["metadata"] = metadata
                
                return info
            
            return None
    
    def prune(self, min_hits: int = 0, ttl: Optional[int] = None):
        """
        Nettoie le cache en supprimant les entrées peu utilisées ou expirées.
        
        Args:
            min_hits: Nombre minimum de hits pour conserver une entrée
            ttl: Durée de vie en secondes (prioritaire sur self.ttl)
        """
        with self.lock:
            # Utiliser le TTL par défaut si non spécifié
            if ttl is None and self.ttl is not None:
                ttl = self.ttl
            
            # Entrées à supprimer
            to_remove = []
            
            # Parcourir toutes les entrées
            for state_hash, entry in list(self.cache.items()):
                _, _, metadata = entry
                keep = True
                
                # Vérifier le nombre de hits
                if min_hits > 0:
                    hits = self.state_metrics.get(state_hash, {}).get("hits", 0)
                    if hits < min_hits:
                        keep = False
                
                # Vérifier le TTL
                if ttl is not None and keep:
                    updated_at = metadata.get("updated_at", 0)
                    if time.time() - updated_at > ttl:
                        keep = False
                
                # Marquer pour suppression
                if not keep:
                    to_remove.append(state_hash)
            
            # Supprimer les entrées
            for state_hash in to_remove:
                self.cache.pop(state_hash, None)
                # Supprimer aussi du disque si activé
                if self.enable_disk_cache:
                    cache_path = self._get_disk_cache_path(state_hash)
                    if cache_path and os.path.exists(cache_path):
                        try:
                            os.remove(cache_path)
                        except:
                            pass
            
            return len(to_remove)
    
    def __len__(self) -> int:
        """
        Retourne le nombre d'entrées dans le cache.
        
        Returns:
            Nombre d'entrées
        """
        return len(self.cache)


class MultiLevelCache:
    """
    Cache multi-niveaux pour différents types d'états avec différentes stratégies.
    
    Permet de:
    1. Utiliser différents caches pour différents types d'états
    2. Avoir des stratégies adaptées à chaque type d'état
    3. Optimiser l'utilisation mémoire/disque en fonction de l'importance
    """
    
    def __init__(self, levels: Dict[str, Dict] = None):
        """
        Initialise le cache multi-niveaux.
        
        Args:
            levels: Dictionnaire de configuration des niveaux
                {nom_niveau: {capacity: int, threshold: float, ...}}
        """
        self.caches = {}
        self.default_level = "default"
        
        # Créer les différents niveaux de cache
        if levels:
            for level_name, config in levels.items():
                self.add_level(level_name, **config)
        
        # Créer un niveau par défaut si aucun n'est défini
        if not self.caches:
            self.add_level(self.default_level)
        
        # Fonction de sélection du niveau
        self.level_selector = None
    
    def add_level(self, name: str, **kwargs):
        """
        Ajoute un niveau de cache.
        
        Args:
            name: Nom du niveau
            **kwargs: Arguments pour le StateCache
        """
        self.caches[name] = StateCache(**kwargs)
    
    def set_level_selector(self, selector_fn: Callable[[Any], str]):
        """
        Définit une fonction pour sélectionner le niveau de cache approprié.
        
        Args:
            selector_fn: Fonction qui prend un état et retourne un nom de niveau
        """
        self.level_selector = selector_fn
    
    def _select_level(self, state) -> str:
        """
        Sélectionne le niveau de cache approprié pour un état.
        
        Args:
            state: État
            
        Returns:
            Nom du niveau
        """
        if self.level_selector:
            level = self.level_selector(state)
            if level in self.caches:
                return level
        
        return self.default_level
    
    def get(self, state, compute_fn: Optional[Callable] = None) -> Tuple[Any, Dict]:
        """
        Récupère une valeur du cache pour un état donné.
        
        Args:
            state: État pour lequel chercher une valeur
            compute_fn: Fonction de calcul à utiliser si la valeur n'est pas en cache
            
        Returns:
            (valeur, info)
        """
        level = self._select_level(state)
        result, info = self.caches[level].get(state, compute_fn)
        info["cache_level"] = level
        return result, info
    
    def put(self, state, value, metadata: Optional[Dict] = None) -> str:
        """
        Ajoute ou met à jour une entrée dans le cache.
        
        Args:
            state: État
            value: Valeur associée
            metadata: Métadonnées additionnelles
            
        Returns:
            Hash de l'état
        """
        level = self._select_level(state)
        return self.caches[level].put(state, value, metadata)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtient les métriques de tous les niveaux de cache.
        
        Returns:
            Dictionnaire de métriques
        """
        metrics = {"global": {}, "levels": {}}
        
        # Statistiques globales
        total_hits = 0
        total_misses = 0
        total_entries = 0
        
        # Collecter les métriques de chaque niveau
        for name, cache in self.caches.items():
            level_metrics = cache.get_metrics()
            metrics[name] = level_metrics
            
            # Agréger pour les statistiques globales
            total_hits += level_metrics.get("hits", 0)
            total_misses += level_metrics.get("misses", 0)
            total_entries += level_metrics.get("memory_entries", 0)
        
        # Calculer les statistiques globales
        metrics["global"]["hits"] = total_hits
        metrics["global"]["misses"] = total_misses
        metrics["global"]["total_queries"] = total_hits + total_misses
        metrics["global"]["hit_rate"] = total_hits / max(1, total_hits + total_misses)
        metrics["global"]["memory_entries"] = total_entries
        metrics["global"]["levels"] = list(self.caches.keys())
        
        return metrics
    
    def clear(self):
        """
        Vide entièrement tous les niveaux de cache.
        """
        for cache in self.caches.values():
            cache.clear()
    
    def prune(self, min_hits: int = 0, ttl: Optional[int] = None):
        """
        Nettoie tous les niveaux de cache en supprimant les entrées peu utilisées.
        
        Args:
            min_hits: Nombre minimum de hits pour conserver une entrée
            ttl: Durée de vie maximum en secondes
        """
        total_removed = 0
        
        # Nettoyer chaque niveau de cache
        for name, cache in self.caches.items():
            removed = cache.prune(min_hits=min_hits, ttl=ttl)
            total_removed += removed
            
        return total_removed
    
    def __len__(self) -> int:
        """
        Retourne le nombre total d'entrées dans tous les niveaux de cache.
        
        Returns:
            Nombre total d'entrées
        """
        return sum(len(cache) for cache in self.caches.values()) 