"""
Module d'adaptation en temps réel pour les prédictions de marché.

Ce module permet de mettre à jour les prédictions en temps réel
en fonction des nouvelles données de marché, des nouvelles de sentiment
et des changements de conditions de marché.
"""

import time
import threading
import queue
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import logging
from collections import deque, OrderedDict
import functools
import hashlib
import json
import sys
import concurrent.futures

# Import des modules de prédiction
from ai_trading.llm.predictions.model_ensemble import ModelEnsemble
from ai_trading.utils import setup_logger

logger = setup_logger("real_time_adapter")

# Décorateur pour la mise en cache des résultats de fonctions intensives
def cached(max_size=128, ttl=300):
    """
    Décorateur pour mettre en cache les résultats des fonctions.
    
    Args:
        max_size: Taille maximale du cache (nombre d'entrées)
        ttl: Durée de vie des entrées en secondes
    """
    def decorator(func):
        # Utiliser OrderedDict pour maintenir l'ordre d'insertion
        cache = OrderedDict()
        cache_info = {'hits': 0, 'misses': 0, 'maxsize': max_size}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Générer une clé de cache en utilisant les arguments de la fonction
            key_parts = [func.__name__]
            
            # Ajouter les arguments positionnels à la clé
            for arg in args:
                if isinstance(arg, (dict, list, set, tuple)):
                    # Pour les types complexes, utiliser leur représentation JSON
                    try:
                        key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
                    except (TypeError, ValueError):
                        # Si la sérialisation échoue, utiliser l'identifiant de l'objet
                        key_parts.append(str(id(arg)))
                else:
                    # Pour les types simples, utiliser leur représentation string
                    key_parts.append(str(arg))
            
            # Ajouter les arguments nommés à la clé (triés par nom)
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                if isinstance(v, (dict, list, set, tuple)):
                    try:
                        key_parts.append(f"{k}:{hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()}")
                    except (TypeError, ValueError):
                        key_parts.append(f"{k}:{id(v)}")
                else:
                    key_parts.append(f"{k}:{v}")
            
            # Créer la clé finale
            key = ":".join(key_parts)
            
            # Vérifier si le résultat est dans le cache et si l'entrée n'est pas expirée
            now = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp <= ttl:
                    # Déplacer l'entrée à la fin pour maintenir l'ordre LRU
                    cache.move_to_end(key)
                    cache_info['hits'] += 1
                    return result
            
            # Si pas de résultat en cache ou entrée expirée, calculer le résultat
            result = func(*args, **kwargs)
            
            # Stocker le résultat dans le cache avec un timestamp
            cache[key] = (result, now)
            cache_info['misses'] += 1
            
            # Si le cache dépasse la taille maximale, supprimer l'entrée la plus ancienne (LRU)
            if len(cache) > max_size:
                cache.popitem(last=False)
            
            return result
        
        # Attacher les informations sur le cache à la fonction
        wrapper.cache = cache
        wrapper.cache_info = cache_info
        wrapper.clear_cache = lambda: cache.clear()
        
        return wrapper
    return decorator


class RealTimeAdapter:
    """
    Classe permettant d'adapter les prédictions en temps réel.
    
    Cette classe permet de:
    - Mettre à jour les prédictions lorsque de nouvelles données arrivent
    - Détecter les changements significatifs de marché
    - Ajuster les prédictions en fonction de l'évolution récente
    - Intégrer des nouvelles données (de marché, de sentiment, etc.)
    
    Attributs:
        prediction_model: Modèle ou ensemble de modèles utilisé pour les prédictions
        update_frequency (float): Fréquence de mise à jour des prédictions (en secondes)
        change_detection_threshold (float): Seuil pour détecter un changement significatif
        max_history_size (int): Taille maximale de l'historique des prédictions
        backtest_mode (bool): Si True, fonctionne en mode backtest (pas de threads actifs)
        cache_size: Taille du cache pour les calculs intensifs
        cache_ttl: Durée de vie du cache en secondes
    """
    
    def __init__(self, 
                prediction_model: Any, 
                update_frequency: float = 60.0,
                change_detection_threshold: float = 0.1,
                max_history_size: int = 100,
                backtest_mode: bool = False,
                cache_size: int = 128,
                cache_ttl: int = 300):
        """
        Initialise l'adaptateur en temps réel.
        
        Args:
            prediction_model: Modèle ou ensemble utilisé pour les prédictions
            update_frequency: Fréquence de mise à jour en secondes
            change_detection_threshold: Seuil pour détecter un changement
            max_history_size: Nombre maximal de prédictions à conserver
            backtest_mode: Si True, désactive les threads actifs pour le backtest
            cache_size: Taille du cache pour les calculs intensifs
            cache_ttl: Durée de vie du cache en secondes
        """
        self.prediction_model = prediction_model
        self.update_frequency = update_frequency
        self.change_detection_threshold = change_detection_threshold
        self.max_history_size = max_history_size
        self.backtest_mode = backtest_mode
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        
        # File d'attente pour les nouvelles données
        self.data_queue = queue.Queue()
        
        # Historique des prédictions
        self.prediction_history = deque(maxlen=max_history_size)
        
        # État actuel du marché et dernière prédiction
        self.current_market_state = None
        self.last_prediction = None
        self.last_update_time = None
        
        # Callbacks pour les événements
        self.on_prediction_update = None
        self.on_significant_change = None
        
        # Threads de mise à jour et indicateur d'arrêt
        self.update_thread = None
        self._stop_event = threading.Event()
        
        # Statistiques de performance
        self.performance_stats = {
            'prediction_time': [],
            'data_processing_time': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("RealTimeAdapter initialisé avec fréquence de mise à jour de %.2f secondes", 
                   update_frequency)
    
    def start(self):
        """
        Démarre l'adaptateur en temps réel.
        
        Initialise et démarre le thread de mise à jour des prédictions.
        """
        if self.backtest_mode:
            logger.info("Adaptateur en mode backtest, pas de thread actif")
            return
        
        if self.update_thread is not None and self.update_thread.is_alive():
            logger.warning("L'adaptateur est déjà en cours d'exécution")
            return
        
        self._stop_event.clear()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Adaptateur en temps réel démarré")
    
    def stop(self):
        """
        Arrête l'adaptateur en temps réel.
        
        Arrête le thread de mise à jour des prédictions.
        """
        if self.backtest_mode:
            return
        
        self._stop_event.set()
        if self.update_thread is not None and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
            logger.info("Adaptateur en temps réel arrêté")
    
    def add_data(self, data: Dict[str, Any], data_timestamp: Optional[datetime] = None):
        """
        Ajoute de nouvelles données à la file d'attente pour mise à jour.
        
        Args:
            data: Dictionnaire contenant les nouvelles données
            data_timestamp: Horodatage des données (utilise datetime.now() si None)
        """
        if data_timestamp is None:
            data_timestamp = datetime.now()
        
        self.data_queue.put({
            'data': data,
            'timestamp': data_timestamp
        })
        
        logger.debug(f"Nouvelles données ajoutées à {data_timestamp}")
        
        # En mode backtest, traiter les données immédiatement
        if self.backtest_mode:
            self._process_new_data()
    
    def set_callback(self, event_type: str, callback: Callable):
        """
        Définit un callback pour un type d'événement donné.
        
        Args:
            event_type: Type d'événement ('update' ou 'change')
            callback: Fonction à appeler lors de l'événement
        """
        if event_type == 'update':
            self.on_prediction_update = callback
        elif event_type == 'change':
            self.on_significant_change = callback
        else:
            raise ValueError(f"Type d'événement inconnu: {event_type}")
    
    def get_latest_prediction(self) -> Dict[str, Any]:
        """
        Retourne la dernière prédiction disponible.
        
        Returns:
            Dict: Dernière prédiction avec métadonnées
        """
        if self.last_prediction is None:
            logger.warning("Aucune prédiction disponible")
            return {}
        
        return self.last_prediction
    
    def get_prediction_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retourne l'historique des prédictions.
        
        Args:
            n: Nombre de prédictions à retourner (toutes si None)
            
        Returns:
            List: Liste des n dernières prédictions
        """
        if n is None or n >= len(self.prediction_history):
            return list(self.prediction_history)
        else:
            return list(self.prediction_history)[-n:]
    
    def _update_loop(self):
        """
        Boucle principale de mise à jour des prédictions.
        
        Cette méthode s'exécute dans un thread séparé et:
        - Vérifie périodiquement les nouvelles données
        - Met à jour les prédictions selon la fréquence configurée
        """
        while not self._stop_event.is_set():
            current_time = datetime.now()
            
            # Vérifier si une mise à jour est nécessaire
            if (self.last_update_time is None or 
                (current_time - self.last_update_time).total_seconds() >= self.update_frequency):
                
                # Traiter les nouvelles données si disponibles
                if not self.data_queue.empty():
                    self._process_new_data()
                    self.last_update_time = current_time
                
                # Si aucune donnée n'est en file d'attente mais qu'une mise à jour est prévue
                elif self.current_market_state is not None:
                    self._update_prediction()
                    self.last_update_time = current_time
            
            # Attendre un court instant avant la prochaine vérification
            # Pour éviter de consommer trop de CPU
            time.sleep(1.0)
    
    def _process_new_data(self):
        """
        Traite les nouvelles données de la file d'attente.
        
        Cette méthode utilise un traitement par lots optimisé pour:
        - Récupérer toutes les données disponibles en une seule opération
        - Minimiser les calculs redondants
        - Réduire les allocations mémoire inutiles
        - Compresser les données historiques
        - Paralléliser certains calculs quand possible
        """
        # Variables pour les métriques de performance
        start_time = time.time()
        data_count = 0
        batch_metrics = {
            'outliers_removed': 0,
            'data_processed': 0,
            'compression_ratio': 0,
            'processing_time': 0,
            'validation_time': 0,
            'outlier_detection_time': 0,
            'compression_time': 0
        }
        
        try:
            # Phase 1: Collecte efficace des données (optimisation: traitement par lots)
            # Récupérer toutes les données d'un coup plutôt que de traiter élément par élément
            collected_data = []
            error_count = 0
            max_errors = 5
            max_batch_size = 250  # Augmenter la taille du lot pour plus d'efficacité
            
            # Mesurer le temps de validation
            validation_start = time.time()
            
            # Utiliser le traitement par lots dynamique avec adaptabilité à la charge
            queue_size = self.data_queue.qsize()
            optimal_batch_size = min(max_batch_size, max(50, int(queue_size * 0.8)))
            
            # Collecter les données en utilisant une approche plus efficace
            while not self.data_queue.empty() and error_count < max_errors and data_count < optimal_batch_size:
                try:
                    # Récupérer plusieurs éléments à la fois si possible (collecte groupée)
                    data_items = []
                    for _ in range(min(20, optimal_batch_size - data_count)):
                        if self.data_queue.empty():
                            break
                        data_items.append(self.data_queue.get(block=False))
                    
                    # Validation vectorisée des données
                    for data_item in data_items:
                        if self._validate_data_item(data_item):
                            collected_data.append(data_item)
                            data_count += 1
                        else:
                            error_count += 1
                        
                        self.data_queue.task_done()
                        
                except queue.Empty:
                    break
                except Exception as e:
                    error_count += 1
                    logger.error(f"Erreur lors de la récupération des données: {e} (erreur {error_count}/{max_errors})")
                    continue
            
            batch_metrics['validation_time'] = time.time() - validation_start
            batch_metrics['data_processed'] = data_count
            
            if not collected_data:
                return
            
            # Phase 2: Préparation et filtrage des données (vectorisation numpy)
            outlier_start = time.time()
            
            # Utiliser des array structurés numpy pour une manipulation plus efficace
            dtype = [('price', 'f8'), ('timestamp', 'f8'), ('index', 'i4')]
            
            # Convertir les données en numpy array structuré pour des opérations vectorisées
            # Utiliser des timestamps Unix pour faciliter les calculs
            timestamps_unix = np.array([item['timestamp'].timestamp() for item in collected_data])
            prices = np.array([item['data'].get('price', 0) for item in collected_data])
            
            # Créer un array structuré
            data_array = np.zeros(len(collected_data), dtype=dtype)
            data_array['price'] = prices
            data_array['timestamp'] = timestamps_unix
            data_array['index'] = np.arange(len(collected_data))
            
            # Trier les données par horodatage en une seule opération
            sorted_data_array = np.sort(data_array, order='timestamp')
            
            # Récupérer les indices originaux pour obtenir les objets d'origine
            sorted_indices = sorted_data_array['index']
            sorted_data = [collected_data[i] for i in sorted_indices]
            sorted_prices = sorted_data_array['price']
            
            # Filtrage des anomalies optimisé avec vectorisation numpy
            filtered_data, outliers_count = self._filter_outliers_optimized(sorted_data, sorted_prices)
            batch_metrics['outliers_removed'] = outliers_count
            
            batch_metrics['outlier_detection_time'] = time.time() - outlier_start
            
            if not filtered_data:
                return
            
            # Phase 3: Mise à jour de l'état avec les données les plus récentes (optimisation: mémoire)
            compression_start = time.time()
            
            latest_data = filtered_data[-1]['data']
            original_data_size = self._estimate_object_size(latest_data)
            
            # Vérifier et compléter les champs manquants essentiels
            if 'price' not in latest_data and self.current_market_state is not None and 'price' in self.current_market_state:
                latest_data['price'] = self.current_market_state['price']
            
            # Vérification de cohérence avec détection adaptative des sauts
            if self.current_market_state is not None and 'price' in latest_data and 'price' in self.current_market_state:
                current_price = self.current_market_state['price']
                new_price = latest_data['price']
                
                # Calcul dynamique du seuil basé sur la volatilité historique récente
                max_change_pct = 0.1  # Valeur par défaut
                
                # Si nous avons un historique de prix, calculer un seuil adaptatif
                if len(filtered_data) > 2:
                    # Calculer les rendements relatifs récents
                    recent_prices = [item['data'].get('price', 0) for item in filtered_data[-10:] if 'price' in item['data']]
                    if len(recent_prices) >= 3:
                        returns = np.diff(recent_prices) / recent_prices[:-1]
                        volatility = np.std(returns)
                        # Limiter la volatilité dans une plage raisonnable (0.5% à 20%)
                        max_change_pct = np.clip(volatility * 5, 0.005, 0.2)
                
                # Vérifier si le changement de prix est trop important
                price_change_pct = abs(new_price - current_price) / max(abs(current_price), 1e-10)
                
                if price_change_pct > max_change_pct:
                    direction = 1 if new_price > current_price else -1
                    latest_data['price'] = current_price * (1 + direction * max_change_pct)
                    latest_data['price_adjusted'] = True
                    latest_data['original_price'] = new_price  # Conserver l'original pour analyse
            
            # Optimisation de la mémoire: compression des données
            compressed_data = self._compress_market_data(latest_data)
            compressed_size = self._estimate_object_size(compressed_data)
            
            # Calculer le ratio de compression
            if original_data_size > 0:
                batch_metrics['compression_ratio'] = 1 - (compressed_size / original_data_size)
            
            # Mettre à jour l'état du marché avec les données compressées
            self.current_market_state = compressed_data
            
            batch_metrics['compression_time'] = time.time() - compression_start
            
            # Phase 4: Mesure des performances et mise à jour des prédictions
            total_processing_time = time.time() - start_time
            batch_metrics['processing_time'] = total_processing_time
            
            # Mettre à jour les métriques de performance
            self._update_performance_metrics(batch_metrics)
            
            # Alerter si le traitement prend trop de temps
            if total_processing_time > 0.1:  # Alerter si le traitement prend plus de 100ms
                logger.warning(f"Traitement des données lent: {total_processing_time:.3f}s pour {data_count} items")
            
            # Générer une nouvelle prédiction
            self._update_prediction()
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement des données: {e} (durée: {processing_time:.3f}s)")
            # Ajouter l'erreur aux statistiques de performance
            self.performance_stats.setdefault('errors', []).append(str(e))
    
    def _filter_outliers_optimized(self, data_items, prices):
        """
        Version optimisée du filtre des valeurs aberrantes utilisant des opérations vectorisées.
        
        Args:
            data_items: Liste des éléments de données
            prices: Tableau numpy des prix extraits
            
        Returns:
            Tuple: (Liste filtrée sans les valeurs aberrantes, nombre d'outliers supprimés)
        """
        if len(data_items) <= 2:
            return data_items, 0
        
        try:
            original_count = len(prices)
            
            # Détecter les valeurs aberrantes avec une méthode robuste adaptative
            # Utiliser la médiane et l'écart absolu médian (MAD) pour une meilleure robustesse
            median_price = np.median(prices)
            mad = np.median(np.abs(prices - median_price))
            
            # Adapter le seuil en fonction de la distribution (plus robuste que l'IQR pour les données financières)
            # Le facteur 1.4826 rend le MAD comparable à l'écart-type pour les distributions normales
            mad_factor = 1.4826 * mad
            
            # Utiliser un intervalle de confiance adaptatif basé sur la taille de l'échantillon
            # Plus nous avons de données, plus le seuil est serré
            confidence_factor = max(3.0, 5.0 - 0.05 * len(prices))
            
            # Définir les limites avec le MAD ajusté
            lower_bound = median_price - confidence_factor * mad_factor
            upper_bound = median_price + confidence_factor * mad_factor
            
            # Créer un masque booléen en une seule opération vectorisée
            valid_mask = (prices >= lower_bound) & (prices <= upper_bound)
            
            # Appliquer le masque pour obtenir les éléments valides
            filtered_items = [item for i, item in enumerate(data_items) if valid_mask[i]]
            
            # Nombre d'outliers supprimés
            outliers_removed = original_count - len(filtered_items)
            
            if not filtered_items and len(data_items) > 0:
                # Si tout est filtré, garder au moins l'élément le plus proche de la médiane
                closest_idx = np.argmin(np.abs(prices - median_price))
                return [data_items[closest_idx]], original_count - 1
            
            return filtered_items, outliers_removed
            
        except Exception as e:
            logger.error(f"Erreur lors du filtrage optimisé: {e}")
            return data_items, 0  # En cas d'erreur, retourner les données originales
    
    def _compress_market_data(self, market_data):
        """
        Compresse les données de marché pour optimiser l'utilisation de la mémoire.
        
        - Supprime les champs non essentiels
        - Arrondit les valeurs numériques pour réduire la précision inutile
        - Convertit les grands tableaux en versions compressées
        
        Args:
            market_data: Données de marché à compresser
            
        Returns:
            Dict: Données de marché compressées
        """
        # Liste des champs essentiels à conserver
        essential_fields = {'price', 'timestamp', 'trend', 'volume', 'asset', 'timeframe'}
        
        # Créer une copie pour éviter de modifier l'original
        compressed = {}
        
        # Ne conserver que les champs essentiels
        for field in essential_fields:
            if field in market_data:
                value = market_data[field]
                
                # Arrondir les valeurs numériques (prix, volume) pour économiser de la mémoire
                if field in ['price', 'volume'] and isinstance(value, float):
                    # Adapter la précision en fonction de la magnitude
                    if abs(value) < 0.01:
                        # Plus de précision pour les très petites valeurs
                        compressed[field] = round(value, 8)
                    elif abs(value) < 1.0:
                        compressed[field] = round(value, 6)
                    elif abs(value) < 1000.0:
                        compressed[field] = round(value, 4)
                    else:
                        # Moins de précision pour les grandes valeurs
                        compressed[field] = round(value, 2)
                else:
                    compressed[field] = value
        
        # Ajouter les champs nécessaires pour le suivi des erreurs/ajustements
        if 'price_adjusted' in market_data:
            compressed['price_adjusted'] = market_data['price_adjusted']
        
        # Compresser les horodatages en utilisant des timestamps Unix
        if isinstance(market_data.get('_timestamp'), datetime):
            compressed['_timestamp_unix'] = market_data['_timestamp'].timestamp()
        else:
            compressed['_timestamp_unix'] = datetime.now().timestamp()
        
        return compressed
    
    def _update_performance_metrics(self, batch_metrics):
        """
        Met à jour les statistiques de performance avec les métriques du dernier lot.
        
        Args:
            batch_metrics: Dictionnaire des métriques du dernier lot
        """
        # Mettre à jour les statistiques de temps de traitement
        self.performance_stats.setdefault('data_processing_time', []).append(batch_metrics['processing_time'])
        
        # Garder un historique limité pour éviter de consommer trop de mémoire
        max_history = 100
        if len(self.performance_stats['data_processing_time']) > max_history:
            self.performance_stats['data_processing_time'] = self.performance_stats['data_processing_time'][-max_history:]
        
        # Mettre à jour les statistiques agrégées
        self.performance_stats['total_data_processed'] = self.performance_stats.get('total_data_processed', 0) + batch_metrics['data_processed']
        self.performance_stats['total_outliers_removed'] = self.performance_stats.get('total_outliers_removed', 0) + batch_metrics['outliers_removed']
        
        # Calculer des moyennes glissantes
        self.performance_stats['avg_processing_time'] = np.mean(self.performance_stats['data_processing_time'])
        self.performance_stats['avg_compression_ratio'] = (
            self.performance_stats.get('avg_compression_ratio', 0) * 0.95 + 
            batch_metrics['compression_ratio'] * 0.05
        )
        
        # Stocker également les temps détaillés pour l'analyse de performance
        for metric in ['validation_time', 'outlier_detection_time', 'compression_time']:
            self.performance_stats.setdefault(metric, []).append(batch_metrics[metric])
            if len(self.performance_stats[metric]) > max_history:
                self.performance_stats[metric] = self.performance_stats[metric][-max_history:]
    
    def _estimate_object_size(self, obj):
        """
        Estime la taille approximative d'un objet Python en mémoire.
        
        Args:
            obj: L'objet dont on veut estimer la taille
            
        Returns:
            int: Taille estimée en octets
        """
        try:
            # Pour les objets simples, utiliser sys.getsizeof
            if isinstance(obj, (int, float, bool, str)):
                return sys.getsizeof(obj)
            
            # Pour les dictionnaires, estimer récursivement
            elif isinstance(obj, dict):
                size = sys.getsizeof(obj)
                for key, value in obj.items():
                    size += self._estimate_object_size(key)
                    size += self._estimate_object_size(value)
                return size
            
            # Pour les listes et tuples, estimer récursivement
            elif isinstance(obj, (list, tuple)):
                size = sys.getsizeof(obj)
                for item in obj:
                    size += self._estimate_object_size(item)
                return size
            
            # Pour les objets datetime
            elif isinstance(obj, datetime):
                return sys.getsizeof(obj)
            
            # Pour les tableaux numpy
            elif hasattr(obj, 'nbytes'):
                return obj.nbytes
            
            # Valeur par défaut pour les autres types
            return sys.getsizeof(obj)
            
        except Exception:
            # En cas d'erreur, utiliser une estimation de base
            return sys.getsizeof(str(obj))
    
    def _validate_data_item(self, data_item: Dict[str, Any]) -> bool:
        """
        Valide un élément de données pour s'assurer qu'il est correctement formaté.
        
        Args:
            data_item: Élément de données à valider
            
        Returns:
            bool: True si l'élément est valide
        """
        # Vérifier la structure de base
        if not isinstance(data_item, dict):
            logger.warning("Donnée ignorée: format non dictionnaire")
            return False
        
        if 'data' not in data_item or 'timestamp' not in data_item:
            logger.warning("Donnée ignorée: champs obligatoires 'data' ou 'timestamp' manquants")
            return False
        
        if not isinstance(data_item['data'], dict):
            logger.warning("Donnée ignorée: champ 'data' n'est pas un dictionnaire")
            return False
        
        # Vérifier les champs minimaux requis
        if 'price' not in data_item['data']:
            logger.warning("Donnée ignorée: champ 'price' manquant dans les données")
            return False
        
        # Vérifier le type des valeurs importantes
        if not isinstance(data_item['data']['price'], (int, float)):
            logger.warning(f"Donnée ignorée: 'price' n'est pas un nombre ({type(data_item['data']['price']).__name__})")
            return False
        
        return True
    
    def _filter_outliers(self, data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtre les données aberrantes dans un lot de données.
        
        Args:
            data_items: Liste des éléments de données à filtrer
            
        Returns:
            List: Liste filtrée sans les valeurs aberrantes
        """
        if len(data_items) <= 2:
            return data_items  # Pas assez de données pour détecter des anomalies
        
        # Extraire les prix
        prices = [item['data']['price'] for item in data_items if 'price' in item['data']]
        
        if not prices:
            return data_items
        
        # Calculer les statistiques
        median_price = np.median(prices)
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        
        # Définir les limites (utiliser l'écart interquartile)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filtrer les éléments
        filtered_items = []
        for item in data_items:
            price = item['data'].get('price')
            if price is not None and lower_bound <= price <= upper_bound:
                filtered_items.append(item)
            elif price is not None:
                logger.warning(f"Donnée aberrante détectée: prix={price} (limites: {lower_bound:.2f}-{upper_bound:.2f})")
                # Option: marquer comme aberrante mais conserver quand même
                item['data']['is_outlier'] = True
                item['data']['outlier_bounds'] = (lower_bound, upper_bound)
                # filtered_items.append(item)  # Décommentez pour conserver les données marquées
        
        return filtered_items
    
    @cached(max_size=128, ttl=60)
    def _generate_prediction(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère une nouvelle prédiction basée sur l'état actuel du marché.
        
        Args:
            market_state: État actuel du marché
            
        Returns:
            Prédiction générée
        """
        try:
            # Début du temps d'exécution
            start_time = time.time()
            
            # Extraire les informations clés
            asset = market_state.get("asset", "")
            timeframe = market_state.get("timeframe", "24h")
            
            # Vérifier si nous avons plusieurs actifs à traiter en même temps
            if isinstance(asset, list):
                # Utiliser la méthode optimisée de batch processing
                if hasattr(self.prediction_model, 'batch_predict_directions'):
                    predictions = self.prediction_model.batch_predict_directions(asset, timeframe)
                else:
                    # Fallback si la méthode batch n'est pas disponible
                    predictions = self._batch_predict_fallback(asset, timeframe, market_state)
                
                # Calculer le temps d'exécution moyen
                prediction_time = (time.time() - start_time) / len(asset)
                
                # Choisir la première prédiction comme retour principal
                primary_asset = asset[0]
                prediction = predictions.get(primary_asset, {})
                
                # Ajouter les autres prédictions
                prediction["batch_results"] = predictions
                prediction["average_prediction_time"] = prediction_time
                
                return prediction
            
            # Cas standard d'un seul actif
            # Générer la prédiction via le modèle
            prediction = self.prediction_model.predict(asset, timeframe)
            
            # Ajouter des métadonnées
            prediction["generation_time"] = time.time() - start_time
            prediction["timestamp"] = datetime.now().isoformat()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de prédiction: {e}")
            # Retourner une prédiction fallback
            return {
                "direction": "neutral",
                "confidence": 0.5,
                "analysis": f"Erreur de prédiction: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def _batch_predict_fallback(self, assets: List[str], timeframe: str, market_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Méthode fallback pour générer des prédictions en lot si la méthode batch native n'est pas disponible.
        
        Args:
            assets: Liste d'actifs à prédire
            timeframe: Horizon temporel
            market_state: État du marché
            
        Returns:
            Dictionnaire des prédictions par actif
        """
        results = {}
        futures = []
        
        # Utiliser un ThreadPoolExecutor pour paralléliser les requêtes
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(assets))) as executor:
            # Soumettre les tâches de prédiction
            for asset in assets:
                future = executor.submit(self.prediction_model.predict, asset, timeframe)
                futures.append((asset, future))
            
            # Collecter les résultats
            for asset, future in futures:
                try:
                    prediction = future.result(timeout=60)  # Timeout de 60 secondes par prédiction
                    results[asset] = prediction
                except Exception as e:
                    logger.error(f"Erreur lors de la prédiction pour {asset}: {e}")
                    # Prédiction fallback en cas d'erreur
                    results[asset] = {
                        "direction": "neutral",
                        "confidence": 0.5,
                        "analysis": f"Erreur: {str(e)}",
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    }
        
        return results
    
    @cached(max_size=64, ttl=30)
    def _is_significant_change(self, new_prediction: Dict[str, Any]) -> bool:
        """
        Détermine si une nouvelle prédiction représente un changement significatif.
        Version mise en cache pour optimiser les performances.
        
        Cette méthode analyse plusieurs facteurs pour déterminer si un changement est significatif:
        - Changement de direction de marché (bullish, bearish, neutral)
        - Changement significatif dans la valeur prédite (seuil configurable)
        - Prise en compte de la confiance de la prédiction
        - Filtrage des changements trop rapprochés pour éviter les faux signaux
        
        Args:
            new_prediction: Nouvelle prédiction à évaluer
            
        Returns:
            bool: True si le changement est significatif
        """
        if not self.prediction_history:
            return True
        
        previous_prediction = self.prediction_history[-1]
        
        # Vérifier s'il y a une erreur dans l'une des prédictions
        if 'error' in new_prediction or 'error' in previous_prediction:
            return False
        
        # Obtenir l'horodatage des prédictions
        new_timestamp = new_prediction.get('timestamp', datetime.now())
        prev_timestamp = previous_prediction.get('timestamp', datetime.now() - timedelta(seconds=60))
        
        # Éviter les changements trop fréquents (réduire les faux signaux)
        # Exiger au moins 1/3 de la période de mise à jour entre les changements significatifs
        min_interval = self.update_frequency / 3
        if (new_timestamp - prev_timestamp).total_seconds() < min_interval:
            return False
        
        # Pour les prédictions catégorielles (ex: direction du marché)
        if 'direction' in new_prediction and 'direction' in previous_prediction:
            # Vérifier si la direction a changé
            direction_changed = new_prediction['direction'] != previous_prediction['direction']
            
            # Si la confiance est disponible, l'utiliser pour filtrer les changements à faible confiance
            if 'confidence' in new_prediction and 'confidence' in previous_prediction:
                confidence_threshold = 0.6  # Ne considérer que les changements avec confiance suffisante
                confidence_sufficient = new_prediction['confidence'] >= confidence_threshold
                
                # Si la confiance est faible, exiger une confiance plus élevée que la précédente prédiction
                if not confidence_sufficient and new_prediction['confidence'] <= previous_prediction['confidence']:
                    return False
                
                return direction_changed and confidence_sufficient
            
            return direction_changed
        
        # Pour les prédictions numériques (ex: prix cible)
        elif 'prediction' in new_prediction and 'prediction' in previous_prediction:
            if isinstance(new_prediction['prediction'], (int, float)) and isinstance(previous_prediction['prediction'], (int, float)):
                prev_value = previous_prediction['prediction']
                new_value = new_prediction['prediction']
                
                # Éviter la division par zéro
                if abs(prev_value) < 1e-10:
                    absolute_change = abs(new_value - prev_value)
                    return absolute_change > self.change_detection_threshold
                
                # Calculer le changement relatif
                relative_change = abs(new_value - prev_value) / abs(prev_value)
                
                # Ajuster le seuil en fonction de la volatilité du marché si disponible
                adaptive_threshold = self.change_detection_threshold
                if ('market_state' in new_prediction and 
                    'volatility' in new_prediction['market_state'] and 
                    new_prediction['market_state']['volatility'] is not None):
                    
                    market_volatility = new_prediction['market_state']['volatility']
                    # Plus la volatilité est élevée, plus le seuil est élevé
                    volatility_factor = min(3.0, max(0.5, market_volatility * 10))
                    adaptive_threshold = self.change_detection_threshold * volatility_factor
                
                # Vérifier si le changement dépasse le seuil adaptatif
                change_significant = relative_change > adaptive_threshold
                
                # Vérifier si la tendance a changé (si disponible)
                trend_changed = False
                if ('market_state' in new_prediction and 
                    'market_state' in previous_prediction and
                    'trend' in new_prediction['market_state'] and 
                    'trend' in previous_prediction['market_state']):
                    
                    trend_changed = (new_prediction['market_state']['trend'] != 
                                    previous_prediction['market_state']['trend'])
                
                # Un changement est significatif si soit la valeur a suffisamment changé,
                # soit la tendance a changé et le changement est modéré
                return change_significant or (trend_changed and relative_change > adaptive_threshold / 2)
        
        # Pour les prédictions d'ensemble avec métadonnées étendues
        elif 'metadata' in new_prediction and 'metadata' in previous_prediction:
            # Déléguer à une méthode spécialisée pour les prédictions d'ensemble
            return self._is_significant_ensemble_change(new_prediction, previous_prediction)
        
        # Pour d'autres types de prédictions ou si pas assez d'informations
        return False
    
    def _is_significant_ensemble_change(self, new_prediction: Dict[str, Any], 
                                       previous_prediction: Dict[str, Any]) -> bool:
        """
        Analyse les changements significatifs pour les prédictions d'ensemble complexes.
        
        Args:
            new_prediction: Nouvelle prédiction d'ensemble
            previous_prediction: Prédiction d'ensemble précédente
            
        Returns:
            bool: True si le changement est significatif
        """
        # Vérifier si le ratio de consensus a suffisamment changé
        if ('consensus_ratio' in new_prediction and 
            'consensus_ratio' in previous_prediction):
            
            new_ratio = new_prediction['consensus_ratio']
            prev_ratio = previous_prediction['consensus_ratio']
            
            # Un changement majeur dans le consensus est significatif
            ratio_change = abs(new_ratio - prev_ratio)
            if ratio_change > 0.2:  # Seuil arbitraire de 20%
                return True
        
        # Vérifier si la prédiction principale a changé
        if ('prediction' in new_prediction and 
            'prediction' in previous_prediction):
            
            # Pour les valeurs catégorielles
            if (isinstance(new_prediction['prediction'], (str, int, bool)) and 
                isinstance(previous_prediction['prediction'], (str, int, bool))):
                
                return new_prediction['prediction'] != previous_prediction['prediction']
        
        return False
    
    def _get_prediction_summary(self, prediction: Dict[str, Any]) -> str:
        """
        Génère un résumé court de la prédiction pour le logging.
        
        Args:
            prediction: Prédiction à résumer
            
        Returns:
            str: Résumé de la prédiction
        """
        if 'direction' in prediction:
            return f"Direction: {prediction['direction']}"
        elif 'prediction' in prediction:
            if isinstance(prediction['prediction'], (int, float)):
                return f"Valeur: {prediction['prediction']:.4f}"
            else:
                return f"Prédiction: {prediction['prediction']}"
        else:
            return "Prédiction disponible"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques de performance.
        
        Returns:
            Dict: Statistiques de performance
        """
        stats = {
            'avg_prediction_time': np.mean(self.performance_stats['prediction_time']) if self.performance_stats['prediction_time'] else 0,
            'max_prediction_time': np.max(self.performance_stats['prediction_time']) if self.performance_stats['prediction_time'] else 0,
            'avg_data_processing_time': np.mean(self.performance_stats['data_processing_time']) if self.performance_stats['data_processing_time'] else 0,
            'cache_hits': self.performance_stats['cache_hits'],
            'cache_misses': self.performance_stats['cache_misses'],
            'cache_efficiency': self.performance_stats['cache_hits'] / (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0 else 0
        }
        
        return stats
    
    def clear_caches(self):
        """
        Vide les caches pour libérer de la mémoire.
        """
        # Accéder aux méthodes décorées et vider leurs caches
        try:
            self._generate_prediction.clear_cache()
            self._is_significant_change.clear_cache()
            logger.info("Caches vidés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du vidage des caches: {e}")
    
    def _update_prediction(self):
        """
        Met à jour la prédiction actuelle en fonction des nouvelles données.
        
        Cette méthode est appelée régulièrement par le thread de mise à jour
        pour générer une nouvelle prédiction basée sur l'état actuel.
        """
        try:
            # Vérifier si nous avons un état de marché valide
            if not self.current_market_state:
                logger.warning("Pas d'état de marché disponible pour la mise à jour de prédiction")
                return
            
            # Copier l'état pour éviter les modifications concurrentes
            market_state = dict(self.current_market_state)
            
            # Récupérer les actifs surveillés
            assets = market_state.get("assets", [])
            if not assets and "asset" in market_state:
                # Cas d'un seul actif
                assets = [market_state["asset"]]
                
            if not assets:
                logger.warning("Aucun actif spécifié pour la mise à jour de prédiction")
                return
                
            # Optimisation: Traiter tous les actifs en un seul appel
            if len(assets) > 1:
                # Passage en mode batch avec tous les actifs
                batch_market_state = dict(market_state)
                batch_market_state["asset"] = assets
                
                # Générer la prédiction pour tous les actifs
                batch_prediction = self._generate_prediction(batch_market_state)
                
                if "batch_results" in batch_prediction:
                    # Stocker les résultats individuels
                    batch_results = batch_prediction["batch_results"]
                    
                    # Mise à jour de la dernière prédiction avec les résultats consolidés
                    self.last_prediction = {
                        "timestamp": datetime.now().isoformat(),
                        "predictions": batch_results,
                        "consolidated": {
                            "bullish_count": sum(1 for p in batch_results.values() if p.get("direction") == "bullish"),
                            "bearish_count": sum(1 for p in batch_results.values() if p.get("direction") == "bearish"),
                            "neutral_count": sum(1 for p in batch_results.values() if p.get("direction") == "neutral"),
                            "average_confidence": sum(p.get("confidence", 0) for p in batch_results.values()) / len(batch_results)
                        },
                        "prediction_time": batch_prediction.get("average_prediction_time", 0) * len(assets)
                    }
                    
                    # Mise à jour du timestamp
                    self.last_update_time = datetime.now()
                    
                    # Vérifier si un changement significatif a eu lieu
                    if self._is_significant_change(self.last_prediction):
                        if self.on_significant_change:
                            self.on_significant_change(self.last_prediction)
                    
                    # Notification de mise à jour
                    if self.on_prediction_update:
                        self.on_prediction_update(self.last_prediction)
                        
                    return
            
            # Cas standard: prédiction individuelle ou fallback
            # Générer une nouvelle prédiction
            start_time = time.time()
            
            latest_prediction = self._generate_prediction(market_state)
            
            # Calculer le temps de prédiction
            prediction_time = time.time() - start_time
            
            # Ajouter des métadonnées
            latest_prediction["processed_at"] = datetime.now().isoformat()
            latest_prediction["prediction_time_seconds"] = prediction_time
            
            # Mise à jour de la dernière prédiction
            self.last_prediction = latest_prediction
            self.last_update_time = datetime.now()
            
            # Vérifier si un changement significatif a eu lieu
            if self._is_significant_change(latest_prediction):
                if self.on_significant_change:
                    self.on_significant_change(latest_prediction)
            
            # Notification de mise à jour
            if self.on_prediction_update:
                self.on_prediction_update(latest_prediction)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la prédiction: {e}")


class RealTimeMarketMonitor:
    """
    Classe pour surveiller le marché en temps réel et détecter les changements importants.
    
    Cette classe permet de:
    - Surveiller les données de marché
    - Détecter les changements de volatilité
    - Identifier les mouvements de prix importants
    - Générer des alertes sur des conditions spécifiques
    
    Attributs:
        observation_window (int): Nombre de points de données pour le calcul des indicateurs
        volatility_threshold (float): Seuil de changement de volatilité considéré comme significatif
        price_move_threshold (float): Seuil de mouvement de prix considéré comme significatif
        alert_callbacks (Dict): Fonctions à appeler lors de la détection d'événements
    """
    
    def __init__(self, 
                observation_window: int = 20,
                volatility_threshold: float = 2.0,
                price_move_threshold: float = 0.03):
        """
        Initialise le moniteur de marché en temps réel.
        
        Args:
            observation_window: Taille de la fenêtre d'observation
            volatility_threshold: Seuil de volatilité (multiplicateur)
            price_move_threshold: Seuil de mouvement de prix (en %)
        """
        self.observation_window = observation_window
        self.volatility_threshold = volatility_threshold
        self.price_move_threshold = price_move_threshold
        
        # Historique des données
        self.price_history = deque(maxlen=observation_window*2)
        self.volume_history = deque(maxlen=observation_window*2)
        
        # Indicateurs calculés
        self.current_volatility = None
        self.baseline_volatility = None
        self.current_trend = None
        
        # Callbacks pour les alertes
        self.alert_callbacks = {
            'volatility_spike': None,
            'significant_price_move': None,
            'trend_change': None,
            'volume_spike': None
        }
        
        logger.info("RealTimeMarketMonitor initialisé avec fenêtre d'observation de %d points",
                   observation_window)
    
    def add_market_data(self, price: float, volume: Optional[float] = None, timestamp: Optional[datetime] = None):
        """
        Ajoute une nouvelle donnée de marché et met à jour les indicateurs.
        
        Args:
            price: Prix actuel
            volume: Volume de transactions (optionnel)
            timestamp: Horodatage (utilise datetime.now() si None)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ajouter aux historiques
        self.price_history.append((price, timestamp))
        if volume is not None:
            self.volume_history.append((volume, timestamp))
        
        # Mettre à jour les indicateurs si assez de données
        if len(self.price_history) >= self.observation_window:
            self._update_indicators()
            self._check_alerts()
    
    def add_market_data_batch(self, prices: List[float], volumes: Optional[List[float]] = None, 
                             timestamps: Optional[List[datetime]] = None):
        """
        Ajoute un lot de données de marché.
        
        Args:
            prices: Liste des prix
            volumes: Liste des volumes (optionnel)
            timestamps: Liste des horodatages (utilise des horodatages séquentiels si None)
        """
        if timestamps is None:
            current_time = datetime.now()
            timestamps = [current_time - timedelta(minutes=i) for i in range(len(prices)-1, -1, -1)]
        
        for i, price in enumerate(prices):
            volume = volumes[i] if volumes is not None and i < len(volumes) else None
            timestamp = timestamps[i] if i < len(timestamps) else datetime.now()
            
            self.price_history.append((price, timestamp))
            if volume is not None:
                self.volume_history.append((volume, timestamp))
        
        # Mettre à jour les indicateurs
        if len(self.price_history) >= self.observation_window:
            self._update_indicators()
            self._check_alerts()
    
    def set_alert_callback(self, alert_type: str, callback: Callable):
        """
        Définit un callback pour un type d'alerte.
        
        Args:
            alert_type: Type d'alerte ('volatility_spike', 'significant_price_move', 'trend_change', 'volume_spike')
            callback: Fonction à appeler lors de l'alerte
        """
        if alert_type in self.alert_callbacks:
            self.alert_callbacks[alert_type] = callback
        else:
            raise ValueError(f"Type d'alerte inconnu: {alert_type}")
    
    def get_market_state(self) -> Dict[str, Any]:
        """
        Retourne l'état actuel du marché.
        
        Returns:
            Dict: État du marché avec indicateurs
        """
        if not self.price_history:
            return {'error': 'Pas de données disponibles'}
        
        current_price = self.price_history[-1][0]
        
        market_state = {
            'price': current_price,
            'timestamp': self.price_history[-1][1],
            'volatility': self.current_volatility,
            'baseline_volatility': self.baseline_volatility,
            'trend': self.current_trend,
            'price_change_24h': self._calculate_price_change(24),
            'price_change_1h': self._calculate_price_change(1)
        }
        
        if self.volume_history:
            market_state['volume'] = self.volume_history[-1][0]
        
        return market_state
    
    def _update_indicators(self):
        """
        Met à jour les indicateurs de marché.
        
        Calcule la volatilité actuelle, la tendance et d'autres indicateurs.
        Implémente une détection améliorée des changements de marché et filtre
        les données aberrantes.
        """
        # Extraire les prix récents
        if len(self.price_history) < self.observation_window:
            return
            
        # Convertir deque en liste pour pouvoir utiliser le slicing
        price_history_list = list(self.price_history)
        
        # Filtrer les valeurs aberrantes potentielles avec un filtre médian avant le calcul des indicateurs
        recent_prices_with_timestamps = price_history_list[-self.observation_window:]
        recent_prices = [p[0] for p in recent_prices_with_timestamps]
        recent_timestamps = [p[1] for p in recent_prices_with_timestamps]
        
        # Détection et correction des valeurs aberrantes avec filtre médian
        filtered_prices = self._filter_price_outliers(recent_prices)
        
        # Calculer la volatilité (écart-type des rendements)
        try:
            # Utiliser les prix filtrés pour le calcul des rendements
            price_returns = np.diff(filtered_prices) / filtered_prices[:-1]
            
            # Gérer les valeurs NaN ou infinies éventuelles
            price_returns = price_returns[~np.isnan(price_returns) & ~np.isinf(price_returns)]
            
            if len(price_returns) > 0:
                self.current_volatility = float(np.std(price_returns))
            else:
                # Si pas assez de valeurs valides, conserver la volatilité précédente
                logger.warning("Pas assez de valeurs valides pour calculer la volatilité")
                if self.current_volatility is None:
                    self.current_volatility = 0.0
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la volatilité: {e}")
            if self.current_volatility is None:
                self.current_volatility = 0.0
        
        # Calculer la volatilité de référence si pas encore définie
        try:
            if self.baseline_volatility is None and len(self.price_history) >= self.observation_window*2:
                baseline_start = max(0, len(price_history_list) - self.observation_window*2)
                baseline_end = max(0, len(price_history_list) - self.observation_window)
                baseline_prices = [p[0] for p in price_history_list[baseline_start:baseline_end]]
                
                # Filtrer les valeurs aberrantes
                filtered_baseline_prices = self._filter_price_outliers(baseline_prices)
                
                # Calculer les rendements
                baseline_returns = np.diff(filtered_baseline_prices) / filtered_baseline_prices[:-1]
                
                # Filtrer les valeurs NaN ou infinies
                baseline_returns = baseline_returns[~np.isnan(baseline_returns) & ~np.isinf(baseline_returns)]
                
                if len(baseline_returns) > 0:
                    self.baseline_volatility = float(np.std(baseline_returns))
                else:
                    # Fallback si pas assez de rendements valides
                    self.baseline_volatility = self.current_volatility
            elif self.baseline_volatility is None:
                self.baseline_volatility = self.current_volatility
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la volatilité de référence: {e}")
            self.baseline_volatility = self.current_volatility
        
        # Calculer la tendance
        try:
            # Utiliser une moyenne mobile exponentielle avec paramètres adaptatifs
            # La taille des fenêtres dépend de la volatilité du marché
            volatility_factor = 1.0
            if self.current_volatility is not None and self.baseline_volatility is not None and self.baseline_volatility > 0:
                volatility_factor = min(2.0, max(0.5, self.current_volatility / self.baseline_volatility))
            
            # Ajuster les périodes en fonction de la volatilité
            short_span = max(2, int(self.observation_window / (4 * volatility_factor)))
            long_span = max(short_span + 1, int(self.observation_window * volatility_factor))
            
            # Calculer les EMAs
            ema_short = self._calculate_ema(filtered_prices, span=short_span)
            ema_long = self._calculate_ema(filtered_prices, span=long_span)
            
            # Déterminer la tendance actuelle
            prev_trend = self.current_trend
            if ema_short > ema_long * 1.005:  # Seuil légèrement supérieur pour éviter les faux signaux
                self.current_trend = "bullish"
            elif ema_short < ema_long * 0.995:  # Seuil légèrement inférieur
                self.current_trend = "bearish"
            else:
                self.current_trend = "neutral"
            
            # Détecter un changement de tendance
            trend_changed = (prev_trend is not None and 
                            prev_trend != self.current_trend and 
                            prev_trend != "neutral")  # Ne pas considérer les changements depuis neutral
            
            if trend_changed and self.alert_callbacks['trend_change']:
                alert_data = {
                    'type': 'trend_change',
                    'previous_trend': prev_trend,
                    'current_trend': self.current_trend,
                    'ema_short': ema_short,
                    'ema_long': ema_long,
                    'timestamp': datetime.now()
                }
                try:
                    self.alert_callbacks['trend_change'](alert_data)
                except Exception as e:
                    logger.error(f"Erreur lors de l'appel du callback trend_change: {e}")
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la tendance: {e}")
            if self.current_trend is None:
                self.current_trend = "neutral"
    
    def _filter_price_outliers(self, prices: List[float]) -> List[float]:
        """
        Filtre les valeurs aberrantes dans une série de prix.
        
        Utilise une combinaison de techniques:
        1. Détection basée sur l'écart interquartile (IQR)
        2. Filtre médian pour les pics
        
        Args:
            prices: Liste des prix à filtrer
            
        Returns:
            List: Prix filtrés sans les valeurs aberrantes
        """
        if len(prices) < 3:
            return prices
        
        try:
            # Convertir en tableau numpy
            prices_array = np.array(prices)
            
            # 1. Méthode IQR pour la détection des outliers
            q1 = np.percentile(prices_array, 25)
            q3 = np.percentile(prices_array, 75)
            iqr = q3 - q1
            
            # Définir les limites (facteur 2.0 au lieu de 1.5 standard pour être moins strict)
            lower_bound = q1 - 2.0 * iqr
            upper_bound = q3 + 2.0 * iqr
            
            # Créer un masque pour les valeurs valides
            valid_mask = (prices_array >= lower_bound) & (prices_array <= upper_bound)
            
            # 2. Filtre médian
            # Calculer la médiane locale
            window_size = min(5, len(prices))
            if window_size % 2 == 0:  # S'assurer que la fenêtre est impaire
                window_size = max(3, window_size - 1)
                
            median_filtered = np.copy(prices_array)
            
            for i in range(len(prices_array)):
                # Calculer les indices de début et fin de la fenêtre
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(prices_array), i + window_size // 2 + 1)
                
                # Extraire la fenêtre
                window = prices_array[start_idx:end_idx]
                
                # Appliquer le filtre médian si la valeur actuelle est un outlier
                if not valid_mask[i]:
                    median_filtered[i] = np.median(window)
            
            return median_filtered.tolist()
            
        except Exception as e:
            logger.error(f"Erreur lors du filtrage des outliers: {e}")
            return prices  # En cas d'erreur, retourner les prix originaux
    
    def _check_alerts(self):
        """
        Vérifie les conditions d'alerte et déclenche les callbacks appropriés.
        
        Cette version améliorée implémente:
        - Détection de volatilité relative robuste
        - Filtrage des faux signaux
        - Calcul adaptatif des seuils
        """
        try:
            # Vérifier le pic de volatilité
            if (self.baseline_volatility is not None and 
                self.baseline_volatility > 0 and
                self.current_volatility > self.baseline_volatility * self.volatility_threshold):
                
                # Calculer le ratio pour l'inclure dans les données d'alerte
                volatility_ratio = self.current_volatility / self.baseline_volatility
                
                if self.alert_callbacks['volatility_spike']:
                    alert_data = {
                        'type': 'volatility_spike',
                        'current_volatility': self.current_volatility,
                        'baseline_volatility': self.baseline_volatility,
                        'ratio': volatility_ratio,
                        'threshold': self.volatility_threshold,
                        'timestamp': datetime.now()
                    }
                    try:
                        self.alert_callbacks['volatility_spike'](alert_data)
                    except Exception as e:
                        logger.error(f"Erreur lors de l'appel du callback volatility_spike: {e}")
            
            # Vérifier le mouvement de prix significatif
            price_change = self._calculate_recent_price_change()
            
            # Utiliser un seuil adaptatif basé sur la volatilité actuelle
            adaptive_threshold = self.price_move_threshold
            if self.current_volatility is not None and self.baseline_volatility is not None and self.baseline_volatility > 0:
                volatility_factor = min(2.0, max(0.5, self.current_volatility / self.baseline_volatility))
                adaptive_threshold = self.price_move_threshold * volatility_factor
            
            if abs(price_change) > adaptive_threshold:
                if self.alert_callbacks['significant_price_move']:
                    alert_data = {
                        'type': 'significant_price_move',
                        'price_change': price_change,
                        'threshold': adaptive_threshold,
                        'direction': 'up' if price_change > 0 else 'down',
                        'volatility_factor': volatility_factor if 'volatility_factor' in locals() else 1.0,
                        'timestamp': datetime.now()
                    }
                    try:
                        self.alert_callbacks['significant_price_move'](alert_data)
                    except Exception as e:
                        logger.error(f"Erreur lors de l'appel du callback significant_price_move: {e}")
                        
            # Vérifier le pic de volume si les données de volume sont disponibles
            if len(self.volume_history) >= self.observation_window:
                # Convertir deque en liste
                volume_history_list = list(self.volume_history)
                
                # Extraire les volumes récents
                recent_volumes = [v[0] for v in volume_history_list[-self.observation_window:]]
                
                if len(recent_volumes) > 1:
                    # Calculer la moyenne et l'écart-type du volume
                    avg_volume = np.mean(recent_volumes[:-1])  # Exclure le volume actuel
                    std_volume = np.std(recent_volumes[:-1])
                    
                    # Seuil pour un pic de volume (2 écarts-types)
                    volume_spike_threshold = avg_volume + 2 * std_volume
                    
                    current_volume = recent_volumes[-1]
                    if current_volume > volume_spike_threshold:
                        if self.alert_callbacks['volume_spike']:
                            volume_ratio = current_volume / avg_volume
                            alert_data = {
                                'type': 'volume_spike',
                                'current_volume': current_volume,
                                'average_volume': avg_volume,
                                'ratio': volume_ratio,
                                'threshold': volume_spike_threshold,
                                'timestamp': datetime.now()
                            }
                            try:
                                self.alert_callbacks['volume_spike'](alert_data)
                            except Exception as e:
                                logger.error(f"Erreur lors de l'appel du callback volume_spike: {e}")
        
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des alertes: {e}")
            # Continuer l'exécution malgré l'erreur
    
    def _calculate_ema(self, prices: List[float], span: int) -> float:
        """
        Calcule la moyenne mobile exponentielle.
        
        Args:
            prices: Liste des prix
            span: Période de l'EMA
            
        Returns:
            float: Valeur de l'EMA
        """
        if len(prices) < span:
            return np.mean(prices)
        
        alpha = 2 / (span + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_recent_price_change(self) -> float:
        """
        Calcule le changement de prix récent en pourcentage.
        
        Returns:
            float: Changement de prix en pourcentage
        """
        if len(self.price_history) < self.observation_window:
            return 0.0
        
        start_price = self.price_history[-self.observation_window][0]
        current_price = self.price_history[-1][0]
        
        return (current_price - start_price) / start_price
    
    def _calculate_price_change(self, hours: int) -> Optional[float]:
        """
        Calcule le changement de prix sur une période donnée.
        
        Args:
            hours: Nombre d'heures pour le calcul
            
        Returns:
            float: Changement de prix en pourcentage ou None si pas assez de données
        """
        if not self.price_history:
            return None
        
        current_price = self.price_history[-1][0]
        current_time = self.price_history[-1][1]
        
        # Trouver le prix il y a 'hours' heures
        for price, timestamp in reversed(self.price_history):
            if current_time - timestamp >= timedelta(hours=hours):
                return (current_price - price) / price
        
        return None 