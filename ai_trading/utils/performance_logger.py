"""
Module pour journaliser les métriques de performance du système.

Ce module permet de collecter et journaliser les métriques de performance
telles que l'utilisation du CPU, de la mémoire, du GPU, et les temps d'exécution.
"""

import datetime
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from ai_trading.utils.advanced_logging import get_logger

# Logger spécifique pour les métriques de performance
logger = get_logger("ai_trading.utils.performance")

# Chemin par défaut pour les fichiers de métriques
BASE_DIR = Path(__file__).parent.parent
METRICS_DIR = BASE_DIR / "info_retour" / "metrics"

# S'assurer que le répertoire de métriques existe
os.makedirs(METRICS_DIR, exist_ok=True)


class SystemMetricsCollector:
    """
    Collecteur de métriques système qui surveille régulièrement l'utilisation des ressources.
    """

    def __init__(
        self,
        interval: float = 10.0,
        log_to_file: bool = True,
        metrics_file: Optional[Path] = None,
        collect_gpu: bool = True,
    ):
        """
        Initialise le collecteur de métriques.

        Args:
            interval: Intervalle de collecte des métriques en secondes
            log_to_file: Si True, sauvegarde les métriques dans un fichier
            metrics_file: Fichier de métriques (optionnel)
            collect_gpu: Si True, collecte aussi les métriques GPU si disponibles
        """
        self.interval = interval
        self.log_to_file = log_to_file

        # Fichier de métriques
        self.metrics_file = metrics_file
        if self.metrics_file is None and log_to_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.metrics_file = METRICS_DIR / f"system_metrics_{timestamp}.json"

        # Métriques GPU
        self.collect_gpu = collect_gpu
        self._gpu_available = False

        # État interne
        self._stop_event = threading.Event()
        self._thread = None
        self._metrics_data = []

        # Vérifier la disponibilité du GPU
        if self.collect_gpu:
            try:
                import torch

                self._gpu_available = torch.cuda.is_available()
                if self._gpu_available:
                    logger.info(f"GPU disponible: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("Aucun GPU CUDA détecté")
            except ImportError:
                logger.warning(
                    "PyTorch non disponible, les métriques GPU ne seront pas collectées"
                )
                self._gpu_available = False

    def start(self) -> None:
        """Démarre la collecte de métriques dans un thread séparé."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Le collecteur de métriques est déjà en cours d'exécution")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info(f"Collecteur de métriques démarré (intervalle: {self.interval}s)")

    def stop(self) -> None:
        """Arrête la collecte de métriques."""
        if self._thread is None or not self._thread.is_alive():
            logger.warning("Le collecteur de métriques n'est pas en cours d'exécution")
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        logger.info("Collecteur de métriques arrêté")

        # Sauvegarder les données collectées
        if self.log_to_file and self._metrics_data:
            self._save_metrics()

    def _collect_loop(self) -> None:
        """Boucle principale de collecte des métriques."""
        while not self._stop_event.is_set():
            try:
                metrics = self.collect_metrics()
                self._metrics_data.append(metrics)
                logger.debug(f"Métriques collectées: {metrics}")
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des métriques: {str(e)}")

            # Attendre l'intervalle spécifié ou l'arrêt
            self._stop_event.wait(self.interval)

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collecte les métriques système actuelles.

        Returns:
            Un dictionnaire contenant les métriques
        """
        metrics = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "system": self._collect_system_metrics(),
            "memory": self._collect_memory_metrics(),
            "cpu": self._collect_cpu_metrics(),
            "disk": self._collect_disk_metrics(),
            "network": self._collect_network_metrics(),
        }

        if self._gpu_available:
            metrics["gpu"] = self._collect_gpu_metrics()

        return metrics

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques système générales."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "uptime": time.time() - psutil.boot_time(),
        }

    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques de mémoire."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent,
        }

    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques CPU."""
        return {
            "percent": psutil.cpu_percent(interval=0.1, percpu=False),
            "percent_per_cpu": psutil.cpu_percent(interval=0.1, percpu=True),
            "count": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "load_avg": [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()],
        }

    def _collect_disk_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques d'utilisation du disque."""
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()

        return {
            "total": disk_usage.total,
            "used": disk_usage.used,
            "free": disk_usage.free,
            "percent": disk_usage.percent,
            "read_count": disk_io.read_count if disk_io else None,
            "write_count": disk_io.write_count if disk_io else None,
            "read_bytes": disk_io.read_bytes if disk_io else None,
            "write_bytes": disk_io.write_bytes if disk_io else None,
        }

    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques réseau."""
        net_io = psutil.net_io_counters()

        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout,
        }

    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques GPU si disponibles."""
        try:
            import torch

            metrics = {
                "device_count": torch.cuda.device_count(),
                "devices": [],
            }

            for i in range(torch.cuda.device_count()):
                device_metrics = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i),
                }

                # Essayer d'obtenir l'utilisation si nvidia-smi est disponible
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    device_metrics.update(
                        {
                            "gpu_util_percent": util.gpu,
                            "memory_util_percent": util.memory,
                        }
                    )
                except:
                    pass

                metrics["devices"].append(device_metrics)

            return metrics
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des métriques GPU: {str(e)}")
            return {"error": str(e)}

    def _save_metrics(self) -> None:
        """Sauvegarde les métriques collectées dans un fichier."""
        if not self._metrics_data:
            logger.warning("Aucune métrique à sauvegarder")
            return

        try:
            # S'assurer que le répertoire parent existe
            os.makedirs(self.metrics_file.parent, exist_ok=True)

            with open(self.metrics_file, "w") as f:
                json.dump(self._metrics_data, f, indent=2)

            logger.info(f"Métriques sauvegardées dans {self.metrics_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métriques: {str(e)}")

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Retourne les dernières métriques collectées.

        Returns:
            Un dictionnaire contenant les dernières métriques, ou None si aucune n'est disponible
        """
        if not self._metrics_data:
            return None
        return self._metrics_data[-1]

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Retourne toutes les métriques collectées.

        Returns:
            Une liste de dictionnaires contenant toutes les métriques collectées
        """
        return self._metrics_data.copy()

    def clear_metrics(self) -> None:
        """Efface toutes les métriques collectées."""
        self._metrics_data.clear()
        logger.info("Métriques effacées")


class FunctionPerformanceTracker:
    """
    Tracker de performance pour mesurer et journaliser le temps d'exécution des fonctions.
    """

    def __init__(self, name: str, log_level: int = None):
        """
        Initialise le tracker de performance.

        Args:
            name: Nom du tracker
            log_level: Niveau de log (optionnel)
        """
        self.name = name
        self.logger = get_logger(f"ai_trading.performance.{name}")
        self.log_level = log_level if log_level is not None else logger.level
        self.timings = {}

    def start(self, task_name: str) -> None:
        """
        Démarre le timing pour une tâche spécifique.

        Args:
            task_name: Nom de la tâche
        """
        if task_name in self.timings:
            self.logger.warning(
                f"Le timing pour '{task_name}' a déjà été démarré, il sera réinitialisé"
            )

        self.timings[task_name] = {"start": time.time(), "end": None, "duration": None}
        self.logger.debug(f"Timing démarré pour {task_name}")

    def stop(self, task_name: str) -> Optional[float]:
        """
        Arrête le timing pour une tâche spécifique et retourne la durée.

        Args:
            task_name: Nom de la tâche

        Returns:
            La durée en secondes, ou None si la tâche n'a pas été démarrée
        """
        if task_name not in self.timings or self.timings[task_name]["start"] is None:
            self.logger.warning(f"Le timing pour '{task_name}' n'a pas été démarré")
            return None

        end_time = time.time()
        start_time = self.timings[task_name]["start"]
        duration = end_time - start_time

        self.timings[task_name]["end"] = end_time
        self.timings[task_name]["duration"] = duration

        self.logger.log(
            self.log_level, f"Timing pour {task_name}: {duration:.4f} secondes"
        )

        return duration

    def get_timing(self, task_name: str) -> Optional[Dict[str, float]]:
        """
        Retourne les données de timing pour une tâche spécifique.

        Args:
            task_name: Nom de la tâche

        Returns:
            Un dictionnaire contenant les données de timing,
            ou None si la tâche n'a pas été démarrée
        """
        if task_name not in self.timings:
            return None
        return self.timings[task_name].copy()

    def get_all_timings(self) -> Dict[str, Dict[str, float]]:
        """
        Retourne toutes les données de timing.

        Returns:
            Un dictionnaire contenant toutes les données de timing
        """
        return self.timings.copy()

    def reset(self) -> None:
        """Réinitialise toutes les données de timing."""
        self.timings.clear()
        self.logger.debug("Toutes les données de timing ont été réinitialisées")


# Instance globale du collecteur de métriques
_metrics_collector = None


def start_metrics_collection(
    interval: float = 30.0, log_to_file: bool = True
) -> SystemMetricsCollector:
    """
    Démarre la collecte des métriques système.

    Args:
        interval: Intervalle de collecte en secondes
        log_to_file: Si True, sauvegarde les métriques dans un fichier

    Returns:
        L'instance du collecteur de métriques
    """
    global _metrics_collector

    if _metrics_collector is not None:
        if (
            _metrics_collector._thread is not None
            and _metrics_collector._thread.is_alive()
        ):
            logger.info("Collecteur de métriques déjà en cours d'exécution")
            return _metrics_collector
        else:
            # Recréer un nouveau collecteur
            _metrics_collector = None

    # Créer un nouveau collecteur
    _metrics_collector = SystemMetricsCollector(
        interval=interval, log_to_file=log_to_file
    )
    _metrics_collector.start()

    return _metrics_collector


def stop_metrics_collection() -> None:
    """Arrête la collecte des métriques système."""
    global _metrics_collector

    if _metrics_collector is not None:
        _metrics_collector.stop()
        logger.info("Collecte des métriques système arrêtée")
    else:
        logger.warning("Aucun collecteur de métriques en cours d'exécution")


def get_performance_tracker(name: str) -> FunctionPerformanceTracker:
    """
    Crée un nouveau tracker de performance pour les fonctions.

    Args:
        name: Nom du tracker

    Returns:
        Un nouveau tracker de performance
    """
    return FunctionPerformanceTracker(name)
