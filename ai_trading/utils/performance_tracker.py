import time
import logging
import functools
import psutil
import os
import platform
import gc
import threading
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

# Configuration du logger
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Classe utilitaire pour suivre les performances CPU et mémoire."""
    
    def __init__(self, name: str = "default"):
        """Initialise un nouveau tracker de performances.
        
        Args:
            name: Nom identifiant ce tracker
        """
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.start_cpu_times = self.process.cpu_times()
        self.start_memory = self.process.memory_info()
        self.measurements = []
    
    def measure(self, label: str = None) -> Dict[str, Any]:
        """Mesure les performances actuelles.
        
        Args:
            label: Étiquette associée à cette mesure
            
        Returns:
            Un dictionnaire contenant les mesures
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Collecter les statistiques CPU
        cpu_times = self.process.cpu_times()
        cpu_percent = self.process.cpu_percent()
        
        # Collecter les statistiques mémoire
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Calculer les changements
        user_time_delta = cpu_times.user - self.start_cpu_times.user
        system_time_delta = cpu_times.system - self.start_cpu_times.system
        rss_delta = memory_info.rss - self.start_memory.rss
        vms_delta = memory_info.vms - self.start_memory.vms
        
        measurement = {
            "timestamp": current_time,
            "elapsed": elapsed,
            "label": label or f"measure_{len(self.measurements)}",
            "cpu": {
                "percent": cpu_percent,
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "user_time_delta": user_time_delta,
                "system_time_delta": system_time_delta
            },
            "memory": {
                "percent": memory_percent,
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "rss_delta": rss_delta,
                "vms_delta": vms_delta
            }
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def reset(self):
        """Réinitialise le tracker pour démarrer une nouvelle série de mesures."""
        self.start_time = time.time()
        self.start_cpu_times = self.process.cpu_times()
        self.start_memory = self.process.memory_info()
        self.measurements = []
    
    def summarize(self) -> Dict[str, Any]:
        """Résume les mesures collectées.
        
        Returns:
            Un dictionnaire contenant le résumé des mesures
        """
        if not self.measurements:
            return {"error": "No measurements recorded"}
        
        first = self.measurements[0]
        last = self.measurements[-1]
        
        total_elapsed = last["timestamp"] - first["timestamp"]
        
        # Résumé CPU
        user_time_total = last["cpu"]["user_time"] - first["cpu"]["user_time"]
        system_time_total = last["cpu"]["system_time"] - first["cpu"]["system_time"]
        
        # Calculer les moyennes et maxima
        cpu_percents = [m["cpu"]["percent"] for m in self.measurements]
        memory_percents = [m["memory"]["percent"] for m in self.measurements]
        rss_values = [m["memory"]["rss"] for m in self.measurements]
        vms_values = [m["memory"]["vms"] for m in self.measurements]
        
        return {
            "name": self.name,
            "total_elapsed": total_elapsed,
            "measurements_count": len(self.measurements),
            "cpu": {
                "avg_percent": np.mean(cpu_percents),
                "max_percent": np.max(cpu_percents),
                "user_time_total": user_time_total,
                "system_time_total": system_time_total
            },
            "memory": {
                "avg_percent": np.mean(memory_percents),
                "max_percent": np.max(memory_percents),
                "min_rss": np.min(rss_values),
                "max_rss": np.max(rss_values),
                "delta_rss": last["memory"]["rss"] - first["memory"]["rss"],
                "min_vms": np.min(vms_values),
                "max_vms": np.max(vms_values),
                "delta_vms": last["memory"]["vms"] - first["memory"]["vms"]
            }
        }
    
    def log_summary(self, level: int = logging.INFO):
        """Génère un résumé et l'enregistre dans le logger.
        
        Args:
            level: Niveau de logging à utiliser
        """
        summary = self.summarize()
        
        if "error" in summary:
            logger.log(level, f"Performance tracker '{self.name}': {summary['error']}")
            return
        
        # Formater les tailles mémoire pour lisibilité
        max_rss_mb = summary["memory"]["max_rss"] / (1024 * 1024)
        delta_rss_mb = summary["memory"]["delta_rss"] / (1024 * 1024)
        
        logger.log(level, f"=== Résumé de performance pour '{self.name}' ===")
        logger.log(level, f"Durée totale: {summary['total_elapsed']:.2f}s")
        logger.log(level, f"CPU moyen: {summary['cpu']['avg_percent']:.1f}%, max: {summary['cpu']['max_percent']:.1f}%")
        logger.log(level, f"Mémoire max: {max_rss_mb:.1f} MB, delta: {delta_rss_mb:.1f} MB")
        logger.log(level, f"Temps CPU: utilisateur={summary['cpu']['user_time_total']:.2f}s, système={summary['cpu']['system_time_total']:.2f}s")


def track_time(func=None, *, label=None):
    """Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Args:
        func: Fonction à décorer
        label: Étiquette pour identifier cette mesure
    
    Returns:
        Fonction décorée
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return f(*args, **kwargs)
            finally:
                end_time = time.time()
                elapsed = end_time - start_time
                func_name = label or f.__qualname__
                logger.info(f"Temps d'exécution de {func_name}: {elapsed:.4f}s")
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


def log_memory_usage(label=""):
    """Log l'utilisation actuelle de la mémoire.
    
    Args:
        label: Étiquette pour identifier cette mesure
    
    Returns:
        Un dictionnaire contenant les informations de mémoire
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convertir en MB pour plus de lisibilité
    rss_mb = memory_info.rss / (1024 * 1024)
    vms_mb = memory_info.vms / (1024 * 1024)
    
    memory_percent = process.memory_percent()
    
    # Collecter des statistiques de GC si disponibles
    gc_stats = None
    if hasattr(gc, 'get_stats'):
        gc_stats = gc.get_stats()
    
    memory_data = {
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "percent": memory_percent,
        "gc_stats": gc_stats
    }
    
    prefix = f"{label}: " if label else ""
    logger.info(f"{prefix}Utilisation mémoire - RSS: {rss_mb:.1f} MB, VMS: {vms_mb:.1f} MB, {memory_percent:.1f}%")
    
    return memory_data


def profile_memory_periodic(interval=5.0, label=None, stop_event=None):
    """Mesure l'utilisation de la mémoire périodiquement dans un thread séparé.
    
    Args:
        interval: Intervalle en secondes entre les mesures
        label: Préfixe d'étiquette pour les logs
        stop_event: Un threading.Event pour signaler l'arrêt du profilage
    
    Returns:
        Un thread de profilage démarré et un événement pour l'arrêter
    """
    if stop_event is None:
        stop_event = threading.Event()
    
    def _profile_thread():
        prefix = f"{label} " if label else ""
        iteration = 0
        measurements = []
        
        while not stop_event.is_set():
            memory_data = log_memory_usage(f"{prefix}Profilage périodique #{iteration}")
            measurements.append(memory_data)
            iteration += 1
            
            # Attendre l'intervalle ou jusqu'à ce que stop_event soit défini
            stop_event.wait(interval)
        
        # Log final
        if measurements:
            rss_values = [m["rss_mb"] for m in measurements]
            logger.info(f"{prefix}Résumé du profilage mémoire - "
                       f"Min: {min(rss_values):.1f} MB, "
                       f"Max: {max(rss_values):.1f} MB, "
                       f"Moy: {sum(rss_values)/len(rss_values):.1f} MB")
    
    profile_thread = threading.Thread(target=_profile_thread, daemon=True)
    profile_thread.start()
    
    return profile_thread, stop_event


# Classes de contexte pour le profilage mélange contrôlé
class TimeTracker:
    """Gestionnaire de contexte pour mesurer le temps d'exécution d'un bloc de code."""
    
    def __init__(self, name="opération"):
        self.name = name
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        logger.info(f"Temps d'exécution de {self.name}: {self.elapsed:.4f}s")


class MemoryTracker:
    """Gestionnaire de contexte pour mesurer l'utilisation de la mémoire d'un bloc de code."""
    
    def __init__(self, name="opération"):
        self.name = name
        self.process = psutil.Process(os.getpid())
    
    def __enter__(self):
        self.start_memory = self.process.memory_info()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = self.process.memory_info()
        
        # Calculer les deltas
        rss_delta = self.end_memory.rss - self.start_memory.rss
        vms_delta = self.end_memory.vms - self.start_memory.vms
        
        # Convertir en MB pour l'affichage
        rss_delta_mb = rss_delta / (1024 * 1024)
        end_rss_mb = self.end_memory.rss / (1024 * 1024)
        
        logger.info(f"Mémoire pour {self.name}: finale={end_rss_mb:.1f}MB, delta={rss_delta_mb:+.1f}MB")


class PerformanceContext:
    """Gestionnaire de contexte combinant la mesure du temps et de la mémoire."""
    
    def __init__(self, name="opération"):
        self.name = name
        self.time_tracker = TimeTracker(name)
        self.memory_tracker = MemoryTracker(name)
    
    def __enter__(self):
        self.time_tracker.__enter__()
        self.memory_tracker.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_tracker.__exit__(exc_type, exc_val, exc_tb)
        self.time_tracker.__exit__(exc_type, exc_val, exc_tb) 