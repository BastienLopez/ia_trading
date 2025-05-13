"""
Module de gestion avancée des timeouts et retries.

Ce module implémente :
- Backoff exponentiel pour les retries
- Circuit breaker pattern pour éviter les requêtes vouées à l'échec
- Fallback sur sources alternatives
- Monitoring des performances et de la fiabilité
"""

import logging
import time
import random
import functools
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import threading
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """États possibles du circuit breaker."""
    CLOSED = 'closed'  # Circuit fermé, les requêtes passent normalement
    OPEN = 'open'      # Circuit ouvert, les requêtes sont bloquées
    HALF_OPEN = 'half_open'  # Circuit semi-ouvert, test limité de requêtes


class CircuitBreaker:
    """
    Implémentation du pattern Circuit Breaker.
    
    Permet d'éviter les appels répétés à un service défaillant en
    ouvrant le circuit après un certain nombre d'échecs consécutifs.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3
    ):
        """
        Initialise le circuit breaker.
        
        Args:
            failure_threshold: Nombre d'échecs consécutifs avant ouverture du circuit
            recovery_timeout: Temps d'attente (en secondes) avant de passer en état semi-ouvert
            half_open_max_calls: Nombre max d'appels en état semi-ouvert
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
        self._lock = threading.RLock()
    
    def allow_request(self) -> bool:
        """
        Détermine si une requête peut être effectuée.
        
        Returns:
            bool: True si la requête est autorisée, False sinon
        """
        with self._lock:
            # En état fermé, toujours autoriser les requêtes
            if self.state == CircuitState.CLOSED:
                return True
            
            # En état ouvert, vérifier si le temps de récupération est écoulé
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout:
                    logger.info("Circuit passant en état semi-ouvert après le temps de récupération")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0  # Réinitialiser le compteur d'appels
                    return True  # Autoriser cette première requête après le timeout
                return False  # Sinon, bloquer les requêtes
            
            # En état semi-ouvert, limiter le nombre de requêtes
            if self.state == CircuitState.HALF_OPEN:
                # Si on a déjà atteint ou dépassé le nombre max d'appels, refuser de nouveaux appels
                if self.half_open_calls >= self.half_open_max_calls:
                    return False
                
                # Sinon, incrémenter le compteur et autoriser la requête
                self.half_open_calls += 1
                return True
            
            # Par défaut, autoriser (ne devrait jamais arriver)
            return False
    
    def record_success(self) -> None:
        """Enregistre un succès et ferme le circuit si nécessaire."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit fermé après un appel réussi en état semi-ouvert")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def record_failure(self) -> None:
        """Enregistre un échec et ouvre le circuit si nécessaire."""
        with self._lock:
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Échec en état semi-ouvert, circuit ouvert à nouveau")
                self.state = CircuitState.OPEN
                self.failure_count = self.failure_threshold  # Force le seuil
                return
            
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    logger.warning(f"Circuit ouvert après {self.failure_count} échecs consécutifs")
                    self.state = CircuitState.OPEN
    
    def get_state(self) -> CircuitState:
        """Retourne l'état actuel du circuit."""
        with self._lock:
            return self.state


class PerformanceMetrics:
    """
    Collecte et analyse les métriques de performance des requêtes.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialise le collecteur de métriques.
        
        Args:
            window_size: Nombre de requêtes à conserver dans l'historique
        """
        self.window_size = window_size
        self.response_times = []
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0
        self.last_status = None
        self._lock = threading.RLock()
    
    def record_success(self, response_time: float) -> None:
        """
        Enregistre une requête réussie.
        
        Args:
            response_time: Temps de réponse en secondes
        """
        with self._lock:
            self.response_times.append(response_time)
            if len(self.response_times) > self.window_size:
                self.response_times.pop(0)
            self.success_count += 1
            self.last_status = "success"
    
    def record_failure(self, is_timeout: bool = False) -> None:
        """
        Enregistre une requête échouée.
        
        Args:
            is_timeout: True si l'échec est dû à un timeout
        """
        with self._lock:
            self.failure_count += 1
            if is_timeout:
                self.timeout_count += 1
            self.last_status = "timeout" if is_timeout else "failure"
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques collectées.
        
        Returns:
            Dict: Métriques de performance
        """
        with self._lock:
            total_requests = self.success_count + self.failure_count
            success_rate = self.success_count / max(1, total_requests)
            
            metrics = {
                "total_requests": total_requests,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "timeout_count": self.timeout_count,
                "success_rate": success_rate,
                "last_status": self.last_status,
            }
            
            # Calculer les statistiques de temps de réponse
            if self.response_times:
                metrics.update({
                    "avg_response_time": round(statistics.mean(self.response_times), 2),
                    "min_response_time": min(self.response_times),
                    "max_response_time": max(self.response_times),
                    "p95_response_time": sorted(self.response_times)[int(len(self.response_times) * 0.95)],
                })
            
            return metrics
    
    def reset(self) -> None:
        """Réinitialise les métriques."""
        with self._lock:
            self.response_times = []
            self.success_count = 0
            self.failure_count = 0
            self.timeout_count = 0
            self.last_status = None


class ResilientRequester:
    """
    Effectue des requêtes avec gestion avancée des timeouts et retries.
    
    Caractéristiques:
    - Backoff exponentiel
    - Circuit breaker
    - Fallback sur sources alternatives
    - Monitoring des performances
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 10.0,
        jitter: float = 0.1,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        fallback_function: Optional[Callable] = None
    ):
        """
        Initialise le requester résilient.
        
        Args:
            max_retries: Nombre maximum de tentatives
            base_delay: Délai initial entre les tentatives (en secondes)
            max_delay: Délai maximum entre les tentatives (en secondes)
            timeout: Timeout pour chaque requête (en secondes)
            jitter: Facteur de variation aléatoire du délai (0-1)
            circuit_breaker_config: Configuration du circuit breaker
            fallback_function: Fonction à appeler en cas d'échec total
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.jitter = jitter
        self.fallback_function = fallback_function
        
        # Initialiser le circuit breaker
        cb_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get('failure_threshold', 5),
            recovery_timeout=cb_config.get('recovery_timeout', 30),
            half_open_max_calls=cb_config.get('half_open_max_calls', 3)
        )
        
        # Initialiser les métriques
        self.metrics = PerformanceMetrics()
    
    def request(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        retry_on_exceptions: Optional[Tuple[Exception, ...]] = None,
        **kwargs
    ) -> Any:
        """
        Effectue une requête avec retries et circuit breaker.
        
        Args:
            func: Fonction à appeler
            *args: Arguments positionnels pour la fonction
            timeout: Timeout spécifique pour cette requête
            retry_on_exceptions: Exceptions qui déclenchent un retry
            **kwargs: Arguments nommés pour la fonction
            
        Returns:
            Any: Résultat de la fonction
            
        Raises:
            Exception: Si toutes les tentatives échouent et qu'il n'y a pas de fallback
        """
        if not self.circuit_breaker.allow_request():
            logger.warning(f"Circuit ouvert, requête bloquée pour {func.__name__}")
            if self.fallback_function:
                return self.fallback_function(*args, **kwargs)
            raise CircuitBreakerError(f"Circuit ouvert pour {func.__name__}")
        
        retry_on = retry_on_exceptions or (Exception,)
        timeout_value = timeout or self.timeout
        
        for attempt in range(self.max_retries + 1):
            try:
                # Mesurer le temps de réponse
                start_time = time.time()
                
                # Appliquer le timeout
                result = self._call_with_timeout(func, timeout_value, *args, **kwargs)
                
                # Calculer le temps de réponse
                response_time = time.time() - start_time
                
                # Enregistrer le succès
                self.metrics.record_success(response_time)
                self.circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                is_timeout = isinstance(e, TimeoutError)
                self.metrics.record_failure(is_timeout=is_timeout)
                
                if attempt == self.max_retries:
                    # Dernière tentative échouée
                    self.circuit_breaker.record_failure()
                    
                    if self.fallback_function:
                        logger.warning(f"Toutes les tentatives ont échoué pour {func.__name__}, utilisation du fallback")
                        return self.fallback_function(*args, **kwargs)
                    
                    logger.error(f"Échec définitif après {attempt + 1} tentatives pour {func.__name__}: {e}")
                    raise
                
                # Vérifier si l'exception justifie un retry
                if not isinstance(e, retry_on):
                    logger.warning(f"Exception non retriable pour {func.__name__}: {e}")
                    raise
                
                # Calculer le délai avec backoff exponentiel
                delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                
                # Ajouter un jitter pour éviter les tempêtes de requêtes
                if self.jitter > 0:
                    delay = delay * (1 + random.uniform(-self.jitter, self.jitter))
                
                logger.warning(f"Tentative {attempt + 1}/{self.max_retries + 1} échouée pour {func.__name__}: {e}. Retry dans {delay:.2f}s")
                time.sleep(delay)
    
    def _call_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Appelle une fonction avec un timeout.
        
        Args:
            func: Fonction à appeler
            timeout: Timeout en secondes
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Any: Résultat de la fonction
            
        Raises:
            TimeoutError: Si la fonction dépasse le timeout
        """
        result = None
        exception = None
        completed = False
        
        def target():
            nonlocal result, exception, completed
            try:
                result = func(*args, **kwargs)
                completed = True
            except Exception as e:
                exception = e
                completed = True
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if not completed:
            raise TimeoutError(f"Timeout après {timeout}s pour {func.__name__}")
        
        if exception:
            raise exception
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance."""
        return self.metrics.get_metrics()
    
    def reset_metrics(self) -> None:
        """Réinitialise les métriques."""
        self.metrics.reset()
    
    def get_circuit_state(self) -> CircuitState:
        """Retourne l'état actuel du circuit breaker."""
        return self.circuit_breaker.get_state()


class CircuitBreakerError(Exception):
    """Exception levée lorsque le circuit est ouvert."""
    pass


class MultiSourceRequester:
    """
    Gère plusieurs sources de données avec fallback automatique.
    
    Permet d'essayer plusieurs sources en séquence jusqu'à obtenir une réponse valide.
    """
    
    def __init__(
        self,
        sources: Dict[str, Tuple[Callable, Dict[str, Any]]],
        source_priority: Optional[List[str]] = None,
        global_timeout: float = 30.0,
        requester_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise le requester multi-sources.
        
        Args:
            sources: Dictionnaire {nom_source: (fonction, config)}
            source_priority: Liste ordonnée des sources par priorité
            global_timeout: Timeout global pour toutes les sources
            requester_config: Configuration par défaut des requesters
        """
        self.sources = sources
        self.source_priority = source_priority or list(sources.keys())
        self.global_timeout = global_timeout
        
        # Créer un requester résilient pour chaque source
        self.requesters = {}
        default_config = requester_config or {}
        
        for source_name, (_, config) in sources.items():
            # Fusionner la config par défaut avec la config spécifique
            merged_config = {**default_config, **config}
            self.requesters[source_name] = ResilientRequester(**merged_config)
    
    def request(
        self,
        *args,
        preferred_sources: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Tuple[Any, str]:
        """
        Effectue une requête en essayant plusieurs sources.
        
        Args:
            *args: Arguments positionnels pour les fonctions sources
            preferred_sources: Sources à essayer en priorité
            timeout: Timeout global spécifique
            **kwargs: Arguments nommés pour les fonctions sources
            
        Returns:
            Tuple[Any, str]: (résultat, nom_source)
            
        Raises:
            AllSourcesFailedError: Si toutes les sources échouent
        """
        # Déterminer l'ordre des sources
        sources_to_try = preferred_sources or self.source_priority
        
        # Vérifier que toutes les sources demandées existent
        for source in sources_to_try:
            if source not in self.sources:
                logger.warning(f"Source inconnue: {source}, ignorée")
        
        # Filtrer les sources valides
        valid_sources = [s for s in sources_to_try if s in self.sources]
        
        # S'assurer qu'il y a au moins une source valide
        if not valid_sources:
            raise ValueError("Aucune source valide spécifiée")
        
        # Timeout global
        start_time = time.time()
        timeout_value = timeout or self.global_timeout
        
        # Erreurs par source
        errors = {}
        
        # Essayer chaque source dans l'ordre
        for source_name in valid_sources:
            # Vérifier le timeout global
            elapsed = time.time() - start_time
            if elapsed >= timeout_value:
                logger.warning(f"Timeout global atteint après {elapsed:.2f}s")
                break
            
            # Récupérer la fonction et le requester
            func, _ = self.sources[source_name]
            requester = self.requesters[source_name]
            
            # Calculer le timeout restant
            remaining_timeout = max(0.1, timeout_value - elapsed)
            
            try:
                logger.debug(f"Essai de la source: {source_name} (timeout: {remaining_timeout:.2f}s)")
                result = requester.request(func, *args, timeout=remaining_timeout, **kwargs)
                logger.info(f"Requête réussie avec la source: {source_name}")
                return result, source_name
            except Exception as e:
                logger.warning(f"Échec de la source {source_name}: {e}")
                errors[source_name] = str(e)
        
        # Si on arrive ici, toutes les sources ont échoué
        raise AllSourcesFailedError(f"Toutes les sources ont échoué: {errors}")
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne les métriques pour toutes les sources.
        
        Returns:
            Dict: {nom_source: métriques}
        """
        return {
            source_name: requester.get_metrics()
            for source_name, requester in self.requesters.items()
        }
    
    def get_circuit_states(self) -> Dict[str, CircuitState]:
        """
        Retourne l'état des circuits pour toutes les sources.
        
        Returns:
            Dict: {nom_source: état_circuit}
        """
        return {
            source_name: requester.get_circuit_state()
            for source_name, requester in self.requesters.items()
        }
    
    def reset_metrics(self) -> None:
        """Réinitialise les métriques pour toutes les sources."""
        for requester in self.requesters.values():
            requester.reset_metrics()


class AllSourcesFailedError(Exception):
    """Exception levée lorsque toutes les sources échouent."""
    pass


def resilient(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    timeout: float = 10.0,
    jitter: float = 0.1,
    retry_on_exceptions: Optional[Tuple[Exception, ...]] = None,
    fallback_function: Optional[Callable] = None,
    circuit_breaker_config: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Décorateur pour rendre une fonction résiliente.
    
    Args:
        max_retries: Nombre maximum de tentatives
        base_delay: Délai initial entre les tentatives (en secondes)
        max_delay: Délai maximum entre les tentatives (en secondes)
        timeout: Timeout pour chaque requête (en secondes)
        jitter: Facteur de variation aléatoire du délai (0-1)
        retry_on_exceptions: Exceptions qui déclenchent un retry
        fallback_function: Fonction à appeler en cas d'échec total
        circuit_breaker_config: Configuration du circuit breaker
        
    Returns:
        Callable: Décorateur
    """
    # Créer un requester résilient partagé pour cette fonction
    requester = ResilientRequester(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        timeout=timeout,
        jitter=jitter,
        circuit_breaker_config=circuit_breaker_config,
        fallback_function=fallback_function
    )
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return requester.request(
                func,
                *args,
                retry_on_exceptions=retry_on_exceptions,
                **kwargs
            )
        
        # Ajouter des attributs pour accéder aux métriques et à l'état
        wrapper.get_metrics = requester.get_metrics
        wrapper.reset_metrics = requester.reset_metrics
        wrapper.get_circuit_state = requester.get_circuit_state
        
        return wrapper
    
    return decorator 