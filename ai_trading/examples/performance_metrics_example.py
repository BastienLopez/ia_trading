"""
Exemple d'utilisation du module de métriques de performance.

Ce script montre comment utiliser les fonctionnalités du module performance_logger
pour collecter et journaliser les métriques système et les temps d'exécution.
"""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.utils.advanced_logging import get_logger, log_execution_time
from ai_trading.utils.performance_logger import (
    get_performance_tracker,
    start_metrics_collection,
    stop_metrics_collection,
)

# Configurer le logger
logger = get_logger("ai_trading.examples.performance")


# Exemple de fonction intensive en calcul pour démonstration
def intensive_task(size=1000):
    """Exécute une tâche intensive en calcul."""
    matrices = []
    for _ in range(10):
        # Créer une matrice aléatoire
        matrix = np.random.rand(size, size)
        # Calculer l'inverse
        inv_matrix = np.linalg.inv(matrix)
        # Multiplication de matrices
        result = np.matmul(matrix, inv_matrix)
        matrices.append(result)
    return matrices


# Exemple de fonction intensive en mémoire pour démonstration
def memory_intensive_task(size=5000):
    """Alloue beaucoup de mémoire temporairement."""
    large_arrays = []
    for _ in range(5):
        # Allouer un grand tableau
        array = np.random.rand(size, size)
        large_arrays.append(array)
    # Effectuer des opérations sur les tableaux
    result = sum(array.sum() for array in large_arrays)
    return result


# Démonstration d'utilisation du FunctionPerformanceTracker
def demonstrate_performance_tracker():
    """Démontre l'utilisation du FunctionPerformanceTracker."""
    logger.info("Démonstration du FunctionPerformanceTracker")

    # Créer un tracker de performance
    tracker = get_performance_tracker("demo")

    # Mesurer plusieurs tâches
    tasks = ["task1", "task2", "task3"]
    for task in tasks:
        tracker.start(task)
        # Simuler une tâche
        delay = random.uniform(0.5, 2.0)
        time.sleep(delay)
        duration = tracker.stop(task)
        logger.info(f"Tâche {task} terminée en {duration:.3f} secondes")

    # Récupérer tous les timings
    all_timings = tracker.get_all_timings()
    logger.info(f"Timings collectés: {all_timings}")

    # Réinitialiser le tracker
    tracker.reset()
    logger.info("Tracker réinitialisé")


# Démonstration d'utilisation du décorateur log_execution_time
@log_execution_time()
def timed_function(delay=1.0):
    """Fonction avec mesure du temps d'exécution."""
    logger.info(f"Démarrage de la fonction temporisée (délai: {delay}s)")
    time.sleep(delay)
    return "Terminé"


# Fonction principale pour la démonstration
def main():
    """Fonction principale pour démontrer l'utilisation du module de métriques de performance."""
    logger.info("Démarrage de la démonstration des métriques de performance")

    # 1. Démarrer la collecte des métriques système
    logger.info("Démarrage de la collecte des métriques système")
    collector = start_metrics_collection(interval=5.0)

    # 2. Démontrer l'utilisation du FunctionPerformanceTracker
    demonstrate_performance_tracker()

    # 3. Exécuter une tâche intensive en calcul
    logger.info("Exécution d'une tâche intensive en calcul")
    tracker = get_performance_tracker("intensive_tasks")

    tracker.start("calcul_intensif")
    intensive_task(size=500)
    tracker.stop("calcul_intensif")

    # 4. Exécuter une tâche intensive en mémoire
    logger.info("Exécution d'une tâche intensive en mémoire")
    tracker.start("memoire_intensive")
    memory_intensive_task(size=2000)
    tracker.stop("memoire_intensive")

    # 5. Utiliser le décorateur log_execution_time
    logger.info("Utilisation du décorateur log_execution_time")
    timed_function(1.5)

    # 6. Attendre un peu pour collecter plus de métriques
    logger.info("Collecte de métriques supplémentaires...")
    time.sleep(5)

    # 7. Récupérer et afficher les dernières métriques
    latest_metrics = collector.get_latest_metrics()
    if latest_metrics:
        logger.info("Dernières métriques système:")
        logger.info(f"- CPU: {latest_metrics['cpu']['percent']}%")
        logger.info(f"- Mémoire: {latest_metrics['memory']['percent']}%")
        logger.info(f"- Disque: {latest_metrics['disk']['percent']}%")
        if "gpu" in latest_metrics:
            devices = latest_metrics["gpu"].get("devices", [])
            for i, device in enumerate(devices):
                logger.info(
                    f"- GPU {i}: {device.get('memory_allocated', 0) / (1024 * 1024):.2f} MB utilisés"
                )

    # 8. Arrêter la collecte des métriques
    logger.info("Arrêt de la collecte des métriques système")
    stop_metrics_collection()

    # 9. Afficher le chemin des fichiers de métriques
    metrics_dir = Path(__file__).parent.parent / "info_retour" / "metrics"
    logger.info(f"Les métriques ont été sauvegardées dans: {metrics_dir}")

    logger.info("Démonstration des métriques de performance terminée")


if __name__ == "__main__":
    main()
