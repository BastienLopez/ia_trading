#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'optimisation pour la réduction des temps d'opération.

Ce module implémente :
- Pré-calcul et mise en cache des résultats fréquents
- Utilisation de batchs de taille optimale (puissance de 2)
- Système de cache intelligent pour les prédictions
- Parallélisation des opérations indépendantes avec torch.nn.parallel
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import functools
import logging
import threading
import multiprocessing
from typing import Dict, List, Tuple, Any, Optional, Callable, Union, TypeVar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

# Type générique pour les résultats de fonctions
T = TypeVar('T')

# Cache global pour stocker les résultats précalculés
_RESULT_CACHE = {}
_PREDICTION_CACHE = {}
_CACHE_LOCK = threading.RLock()

def precalculate_and_cache(func: Callable) -> Callable:
    """
    Décorateur pour précalculer et mettre en cache les résultats d'une fonction.
    
    Args:
        func: La fonction dont les résultats seront mis en cache
        
    Returns:
        Une fonction décorée qui utilise les résultats en cache si disponibles
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Créer une clé de cache basée sur les arguments
        cache_key = (func.__name__, str(args), str(sorted(kwargs.items())))
        
        # Vérifier si le résultat est déjà en cache
        with _CACHE_LOCK:
            if cache_key in _RESULT_CACHE:
                logger.debug(f"Résultat en cache trouvé pour {func.__name__}")
                return _RESULT_CACHE[cache_key]
        
        # Exécuter la fonction et stocker le résultat en cache
        result = func(*args, **kwargs)
        
        with _CACHE_LOCK:
            _RESULT_CACHE[cache_key] = result
            
        return result
    
    return wrapper

def get_optimal_batch_size(min_size: int, max_size: int) -> int:
    """
    Détermine la taille de batch optimale (puissance de 2) pour les opérations.
    
    Args:
        min_size: Taille minimale requise
        max_size: Taille maximale autorisée
        
    Returns:
        La puissance de 2 la plus proche dans l'intervalle [min_size, max_size]
    """
    # Trouver la puissance de 2 la plus proche mais pas inférieure à min_size
    power = 1
    while power < min_size and power * 2 <= max_size:
        power *= 2
    
    # Si la puissance est inférieure à min_size mais que doubler dépasserait max_size,
    # on utilise la plus grande puissance de 2 disponible
    if power < min_size and power * 2 > max_size:
        return power
    
    # Si power est exactement min_size, on peut essayer de doubler si ça ne dépasse pas max_size
    if power == min_size and power * 2 <= max_size:
        power *= 2
    
    # Essayons de doubler plusieurs fois tant que ça ne dépasse pas max_size
    while power * 2 <= max_size:
        power *= 2
    
    return power

class PredictionCache:
    """Système de cache intelligent pour les prédictions."""
    
    def __init__(self, capacity: int = 1000, ttl: int = 3600):
        """
        Initialise le cache de prédictions.
        
        Args:
            capacity: Capacité maximale du cache
            ttl: Durée de vie des entrées en secondes (time to live)
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = {}  # {key: (value, timestamp)}
        self.lock = threading.RLock()
        
        # Démarrer un thread de nettoyage périodique
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Récupère une valeur du cache si elle existe et n'est pas expirée.
        
        Args:
            key: La clé de recherche
            
        Returns:
            La valeur associée ou None si absente ou expirée
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            value, timestamp = self.cache[key]
            current_time = time.time()
            
            # Vérifier si l'entrée a expiré
            if current_time - timestamp > self.ttl:
                del self.cache[key]
                return None
            
            return value
    
    def put(self, key: Any, value: Any) -> None:
        """
        Ajoute ou met à jour une entrée dans le cache.
        
        Args:
            key: La clé
            value: La valeur à associer
        """
        with self.lock:
            # Si le cache est plein, supprimer l'entrée la plus ancienne
            if len(self.cache) >= self.capacity and key not in self.cache:
                oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
                del self.cache[oldest_key]
            
            # Ajouter ou mettre à jour l'entrée
            self.cache[key] = (value, time.time())
    
    def _cleanup_loop(self) -> None:
        """Boucle de nettoyage périodique des entrées expirées."""
        while True:
            time.sleep(self.ttl / 10)  # Nettoyage toutes les ttl/10 secondes
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Supprime les entrées expirées du cache."""
        current_time = time.time()
        with self.lock:
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.debug(f"Nettoyage du cache: {len(expired_keys)} entrées supprimées")

def get_prediction_cache(name: str = "default") -> PredictionCache:
    """
    Récupère ou crée un cache de prédictions nommé.
    
    Args:
        name: Nom du cache
        
    Returns:
        L'instance de cache demandée
    """
    with _CACHE_LOCK:
        if name not in _PREDICTION_CACHE:
            _PREDICTION_CACHE[name] = PredictionCache()
        return _PREDICTION_CACHE[name]

class ParallelOperations:
    """Classe pour paralléliser les opérations indépendantes."""
    
    @staticmethod
    def parallel_map(func: Callable[[Any], T], items: List[Any], 
                   use_processes: bool = False, max_workers: Optional[int] = None) -> List[T]:
        """
        Exécute une fonction sur une liste d'éléments en parallèle.
        
        Args:
            func: La fonction à appliquer à chaque élément
            items: Liste des éléments à traiter
            use_processes: Utiliser des processus au lieu de threads
            max_workers: Nombre maximum de workers (None = automatique)
            
        Returns:
            Liste des résultats
        """
        if not max_workers:
            max_workers = multiprocessing.cpu_count()
        
        # Choisir l'exécuteur approprié
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            results = list(executor.map(func, items))
            
        return results
    
    @staticmethod
    def parallel_model_inference(model: nn.Module, inputs: List[torch.Tensor], 
                              batch_size: Optional[int] = None) -> List[torch.Tensor]:
        """
        Exécute l'inférence d'un modèle en parallèle sur plusieurs GPUs si disponibles.
        
        Args:
            model: Le modèle PyTorch
            inputs: Liste des tenseurs d'entrée
            batch_size: Taille de batch (None = taille optimale automatique)
            
        Returns:
            Liste des tenseurs de sortie
        """
        # Déterminer la taille de batch optimale si non spécifiée
        if batch_size is None:
            batch_size = get_optimal_batch_size(1, len(inputs))
        
        # Vérifier si plusieurs GPUs sont disponibles
        if torch.cuda.device_count() > 1:
            # Utiliser DataParallel pour paralléliser sur plusieurs GPUs
            parallel_model = nn.DataParallel(model)
            
            # Créer des batchs
            batches = [torch.stack(inputs[i:i+batch_size]) 
                      for i in range(0, len(inputs), batch_size)]
            
            # Exécuter l'inférence par batch
            outputs = []
            with torch.no_grad():
                for batch in batches:
                    batch_output = parallel_model(batch)
                    outputs.extend(batch_output)
            
            return outputs
        else:
            # Un seul GPU ou CPU, utiliser le modèle directement
            outputs = []
            with torch.no_grad():
                for input_tensor in inputs:
                    output = model(input_tensor.unsqueeze(0)).squeeze(0)
                    outputs.append(output)
            
            return outputs

# Exemple d'utilisation des fonctions de réduction des temps d'opération
def example_usage():
    # Exemple de pré-calcul et mise en cache
    @precalculate_and_cache
    def expensive_computation(x, y):
        """Simulation d'un calcul coûteux."""
        time.sleep(1)  # Simuler un long calcul
        return x * y + x
    
    # Exemple d'utilisation de taille de batch optimale
    optimal_batch = get_optimal_batch_size(100, 1000)
    
    # Exemple d'utilisation du cache de prédictions
    prediction_cache = get_prediction_cache("example")
    
    # Exemple de parallélisation
    def process_item(item):
        return item * 2
    
    items = list(range(100))
    parallel_ops = ParallelOperations()
    results = parallel_ops.parallel_map(process_item, items)
    
    return {
        "cached_result": expensive_computation(5, 10),
        "optimal_batch_size": optimal_batch,
        "parallel_results": results[:5]  # Juste les 5 premiers résultats
    } 