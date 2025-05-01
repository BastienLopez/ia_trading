#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la mise en cache des transformations fréquentes sur les données financières.
Permet d'éviter de recalculer des features techniques à chaque accès aux données.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps, lru_cache

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_cache_transform_fn(cache_size: int = 500):
    """
    Décorateur pour mettre en cache les résultats d'une fonction de transformation.
    
    Args:
        cache_size: Taille maximale du cache.
        
    Returns:
        Décorateur pour la fonction de transformation.
    """
    def decorator(transform_fn: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
        """
        Décore une fonction de transformation pour mettre en cache ses résultats.
        
        Args:
            transform_fn: Fonction de transformation à décorer.
            
        Returns:
            Fonction de transformation décorée avec cache.
        """
        # Cache local sous forme de dictionnaire
        cache = {}
        hits = 0
        misses = 0
        
        @wraps(transform_fn)
        def wrapper(x: Tensor) -> Tensor:
            """
            Wrapper autour de la fonction de transformation.
            
            Args:
                x: Tenseur d'entrée.
                
            Returns:
                Tenseur transformé (depuis le cache si disponible).
            """
            nonlocal hits, misses
            
            # Pour les tenseurs PyTorch
            if isinstance(x, torch.Tensor):
                # Créer une clé de hachage basée sur la forme et un échantillon de données
                shape = x.shape
                flat_data = x.detach().cpu().numpy().flatten()
                
                # Utiliser un sous-échantillon pour les grandes données
                if len(flat_data) > 1000:
                    data_sample = flat_data[:1000]
                else:
                    data_sample = flat_data
                
                # Créer une clé en combinant la forme et un hash des données
                key = (shape, hash(data_sample.tobytes()))
                
                # Vérifier si la clé est dans le cache
                if key in cache:
                    hits += 1
                    # Récupérer le résultat du cache
                    return cache[key]
                else:
                    misses += 1
                    # Calculer le résultat
                    result = transform_fn(x)
                    
                    # Mettre en cache
                    cache[key] = result
                    
                    # Gérer la taille du cache
                    if len(cache) > cache_size:
                        # Supprimer une clé au hasard (stratégie simple)
                        del cache[next(iter(cache))]
                    
                    return result
            else:
                # Pour les non-tenseurs, appeler directement la fonction
                return transform_fn(x)
        
        # Ajouter des fonctions utiles au wrapper
        def cache_info():
            return {"hits": hits, "misses": misses, "size": len(cache), "max_size": cache_size}
        
        def cache_clear():
            nonlocal cache, hits, misses
            cache.clear()
            hits = 0
            misses = 0
        
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        
        return wrapper
    
    return decorator


class CachedFeatureTransform:
    """
    Classe pour calculer et mettre en cache des transformations de features financières.
    Particulièrement utile pour les calculs d'indicateurs techniques coûteux.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_memory_cache_size: int = 1000,
        use_disk_cache: bool = True,
        create_dir_if_missing: bool = True
    ):
        """
        Initialise le gestionnaire de transformations avec cache.
        
        Args:
            cache_dir: Répertoire où stocker le cache persistant.
            max_memory_cache_size: Taille maximale du cache en mémoire.
            use_disk_cache: Si True, persiste le cache sur disque.
            create_dir_if_missing: Si True, crée le répertoire de cache s'il n'existe pas.
        """
        # Configurer le cache
        self.max_memory_cache_size = max_memory_cache_size
        self.use_disk_cache = use_disk_cache
        
        # Cache en mémoire
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cache sur disque
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.path.expanduser("~")) / ".ai_trading_cache"
        
        if create_dir_if_missing and use_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, data: Union[np.ndarray, Tensor], transform_name: str) -> str:
        """
        Génère une clé de cache unique pour un tenseur et une transformation.
        
        Args:
            data: Données à transformer.
            transform_name: Nom de la transformation.
            
        Returns:
            Clé de cache unique.
        """
        # Obtenir un hash des données
        if isinstance(data, torch.Tensor):
            # Utiliser un sous-échantillon pour l'empreinte (pour de grandes données)
            if data.numel() > 1000:
                sample = data.flatten()[:1000].detach().cpu().numpy()
            else:
                sample = data.detach().cpu().numpy()
        else:
            # Pour numpy, également sous-échantillonner
            if data.size > 1000:
                sample = data.flatten()[:1000]
            else:
                sample = data
        
        # Créer un hash simplifié basé sur la forme et un échantillon des valeurs
        shape_hash = hash(str(data.shape))
        data_hash = hash(str(sample.mean()) + str(sample.std()))
        
        # Combiner pour créer la clé
        return f"{transform_name}_{shape_hash}_{data_hash}"
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """
        Retourne le chemin du fichier cache sur disque.
        
        Args:
            cache_key: Clé du cache.
            
        Returns:
            Chemin vers le fichier cache.
        """
        # Créer un nom de fichier valide à partir de la clé
        safe_key = "".join(c if c.isalnum() else "_" for c in cache_key)
        return self.cache_dir / f"{safe_key}.pkl"
    
    def transform_with_cache(
        self,
        data: Union[np.ndarray, Tensor],
        transform_fn: Callable,
        transform_name: Optional[str] = None
    ) -> Union[np.ndarray, Tensor]:
        """
        Applique une transformation avec mise en cache des résultats.
        
        Args:
            data: Données à transformer.
            transform_fn: Fonction de transformation.
            transform_name: Nom de la transformation (pour identification dans le cache).
            
        Returns:
            Données transformées.
        """
        # Dériver le nom de la transformation si non fourni
        if transform_name is None:
            transform_name = transform_fn.__name__
        
        # Générer la clé de cache
        cache_key = self._get_cache_key(data, transform_name)
        
        # Vérifier le cache en mémoire
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        # Vérifier le cache sur disque si activé
        if self.use_disk_cache:
            disk_cache_path = self._get_disk_cache_path(cache_key)
            if disk_cache_path.exists():
                try:
                    with open(disk_cache_path, 'rb') as f:
                        transformed_data = pickle.load(f)
                    
                    # Mettre en cache mémoire pour un accès plus rapide
                    self.memory_cache[cache_key] = transformed_data
                    
                    # Gérer la taille du cache mémoire
                    if len(self.memory_cache) > self.max_memory_cache_size:
                        # Stratégie simple: supprimer une clé aléatoire
                        random_key = next(iter(self.memory_cache))
                        del self.memory_cache[random_key]
                    
                    self.cache_hits += 1
                    return transformed_data
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement du cache disque: {e}")
        
        # Cache miss - calculer la transformation
        self.cache_misses += 1
        transformed_data = transform_fn(data)
        
        # Mettre en cache en mémoire
        self.memory_cache[cache_key] = transformed_data
        
        # Gérer la taille du cache mémoire
        if len(self.memory_cache) > self.max_memory_cache_size:
            # Stratégie simple: supprimer une clé aléatoire
            random_key = next(iter(self.memory_cache))
            del self.memory_cache[random_key]
        
        # Mettre en cache sur disque si activé
        if self.use_disk_cache:
            try:
                with open(disk_cache_path, 'wb') as f:
                    pickle.dump(transformed_data, f)
            except Exception as e:
                logger.warning(f"Erreur lors de l'écriture du cache disque: {e}")
        
        return transformed_data
    
    def clear_memory_cache(self):
        """Vide le cache en mémoire."""
        self.memory_cache.clear()
        logger.info("Cache mémoire vidé")
    
    def clear_disk_cache(self):
        """Vide le cache sur disque."""
        if self.use_disk_cache and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Erreur lors de la suppression du fichier cache {cache_file}: {e}")
            logger.info("Cache disque vidé")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Retourne des statistiques sur l'utilisation du cache.
        
        Returns:
            Dictionnaire avec les statistiques.
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_max_size": self.max_memory_cache_size
        }


# Créer une instance par défaut
default_transform_cache = CachedFeatureTransform()

def cached_transform(
    data: Union[np.ndarray, Tensor],
    transform_fn: Callable,
    transform_name: Optional[str] = None,
    cache_manager: Optional[CachedFeatureTransform] = None
) -> Union[np.ndarray, Tensor]:
    """
    Fonction utilitaire pour appliquer une transformation avec cache.
    
    Args:
        data: Données à transformer.
        transform_fn: Fonction de transformation.
        transform_name: Nom de la transformation.
        cache_manager: Gestionnaire de cache à utiliser (utilise l'instance par défaut si None).
        
    Returns:
        Données transformées.
    """
    manager = cache_manager or default_transform_cache
    return manager.transform_with_cache(data, transform_fn, transform_name) 