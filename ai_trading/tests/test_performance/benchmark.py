"""
Benchmarks pour les fonctionnalités transversales.

Ce module contient des benchmarks pour mesurer précisément les performances
des fonctionnalités de journalisation avancée, de collecte de métriques
et de gestion des checkpoints.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import timeit
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import torch.nn as nn

from ai_trading.utils.advanced_logging import (
    get_logger,
    log_exceptions,
    log_execution_time,
)
from ai_trading.utils.checkpoint_manager import CheckpointType, get_checkpoint_manager
from ai_trading.utils.performance_logger import (
    get_performance_tracker,
    start_metrics_collection,
    stop_metrics_collection,
)

# Logger pour les benchmarks
logger = get_logger("ai_trading.tests.performance.benchmark")


# Modèle simple pour les benchmarks
class BenchmarkModel(nn.Module):
    """Modèle PyTorch simple pour les benchmarks."""

    def __init__(self, size=1):
        super().__init__()
        self.fc1 = nn.Linear(10 * size, 64 * size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64 * size, 32 * size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32 * size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Fonctions utilitaires
def generate_test_data(size=1):
    """Génère des données de test pour les benchmarks."""
    return {
        "tensors": {
            "t1": torch.randn(100 * size, 10 * size),
            "t2": torch.randn(10 * size, 10 * size),
        },
        "arrays": {
            "a1": np.random.rand(100 * size, 10 * size),
            "a2": np.random.rand(10 * size, 10 * size),
        },
        "params": {"learning_rate": 0.01, "batch_size": 32, "epochs": 10},
    }


def benchmark_function(func, name, repeat=5, number=100):
    """Mesure le temps d'exécution d'une fonction."""
    logger.info(f"Benchmark de {name}")

    # Utiliser timeit pour une mesure précise
    timer = timeit.Timer(func)
    times = timer.repeat(repeat=repeat, number=number)

    # Calculer les statistiques
    avg_time = sum(times) / repeat / number
    min_time = min(times) / number
    max_time = max(times) / number

    logger.info(f"  Temps moyen par exécution: {avg_time*1000:.3f} ms")
    logger.info(f"  Temps min: {min_time*1000:.3f} ms")
    logger.info(f"  Temps max: {max_time*1000:.3f} ms")

    return {
        "name": name,
        "average_ms": avg_time * 1000,
        "min_ms": min_time * 1000,
        "max_ms": max_time * 1000,
        "total_runs": number * repeat,
    }


def benchmark_logging():
    """Benchmark des fonctionnalités de journalisation."""
    results = []

    # Configurer différents loggers pour le benchmark
    std_logger = get_logger(
        "ai_trading.benchmark.std",
        {
            "level": 10,  # DEBUG
            "console_handler": False,  # Pas de sortie console pour le benchmark
        },
    )

    json_logger = get_logger(
        "ai_trading.benchmark.json",
        {
            "json_handler": True,
            "level": 10,  # DEBUG
            "console_handler": False,
        },
    )

    # Benchmark 1: Journalisation standard
    def log_standard():
        std_logger.debug("Message de débogage pour benchmark")

    results.append(
        benchmark_function(
            log_standard, "Journalisation standard (DEBUG)", repeat=5, number=1000
        )
    )

    # Benchmark 2: Journalisation JSON
    def log_json():
        json_logger.debug("Message de débogage pour benchmark")

    results.append(
        benchmark_function(
            log_json, "Journalisation JSON (DEBUG)", repeat=5, number=1000
        )
    )

    # Benchmark 3: Journalisation avec données structurées
    data = {"value": 42, "list": [1, 2, 3], "nested": {"a": 1, "b": 2}}

    def log_with_data():
        std_logger.debug("Message avec données", extra={"data": data})

    results.append(
        benchmark_function(
            log_with_data,
            "Journalisation avec données structurées",
            repeat=5,
            number=1000,
        )
    )

    # Benchmark 4: Utilisation du décorateur log_execution_time
    @log_execution_time(std_logger)
    def dummy_function():
        x = 0
        for i in range(100):
            x += i
        return x

    results.append(
        benchmark_function(
            dummy_function, "Décorateur log_execution_time", repeat=5, number=100
        )
    )

    # Benchmark 5: Utilisation du décorateur log_exceptions
    @log_exceptions(std_logger)
    def function_with_no_exception():
        return 42

    results.append(
        benchmark_function(
            function_with_no_exception,
            "Décorateur log_exceptions (sans exception)",
            repeat=5,
            number=1000,
        )
    )

    return results


def benchmark_metrics():
    """Benchmark des fonctionnalités de collecte de métriques."""
    results = []

    # Benchmark 1: Démarrage/arrêt du collecteur
    def start_stop_collector():
        collector = start_metrics_collection(interval=60.0, log_to_file=False)
        stop_metrics_collection()

    results.append(
        benchmark_function(
            start_stop_collector,
            "Démarrage/arrêt du collecteur de métriques",
            repeat=3,
            number=10,
        )
    )

    # Benchmark 2: Collecte unique de métriques
    collector = start_metrics_collection(interval=60.0, log_to_file=False)

    def collect_metrics():
        metrics = collector.collect_metrics()
        return metrics

    results.append(
        benchmark_function(
            collect_metrics, "Collecte unique de métriques", repeat=5, number=20
        )
    )

    stop_metrics_collection()

    # Benchmark 3: Performance tracker
    tracker = get_performance_tracker("benchmark")

    def use_tracker():
        tracker.start("bench_task")
        # Simuler un traitement
        x = 0
        for i in range(1000):
            x += i
        tracker.stop("bench_task")

    results.append(
        benchmark_function(
            use_tracker, "Utilisation du performance tracker", repeat=5, number=100
        )
    )

    return results


def benchmark_checkpoints():
    """Benchmark des fonctionnalités de gestion des checkpoints."""
    results = []

    # Créer un répertoire temporaire pour les benchmarks
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configurer le gestionnaire de checkpoints
        checkpoint_manager = get_checkpoint_manager()
        checkpoint_manager.root_dir = Path(temp_dir)

        # Préparer les données de test
        small_model = BenchmarkModel(size=1)
        medium_model = BenchmarkModel(size=2)
        large_model = BenchmarkModel(size=4)

        small_data = generate_test_data(size=1)
        medium_data = generate_test_data(size=2)
        large_data = generate_test_data(size=4)

        # Benchmark 1: Sauvegarde de petit modèle
        def save_small_model():
            return checkpoint_manager.save_model(
                model=small_model,
                name="small_model",
                description="Petit modèle pour benchmark",
            )

        small_model_id = save_small_model()
        results.append(
            benchmark_function(
                save_small_model, "Sauvegarde de petit modèle", repeat=3, number=5
            )
        )

        # Benchmark 2: Sauvegarde de modèle moyen
        def save_medium_model():
            return checkpoint_manager.save_model(
                model=medium_model,
                name="medium_model",
                description="Modèle moyen pour benchmark",
            )

        medium_model_id = save_medium_model()
        results.append(
            benchmark_function(
                save_medium_model, "Sauvegarde de modèle moyen", repeat=3, number=5
            )
        )

        # Benchmark 3: Sauvegarde de grand modèle
        def save_large_model():
            return checkpoint_manager.save_model(
                model=large_model,
                name="large_model",
                description="Grand modèle pour benchmark",
            )

        large_model_id = save_large_model()
        results.append(
            benchmark_function(
                save_large_model, "Sauvegarde de grand modèle", repeat=3, number=3
            )
        )

        # Benchmark 4: Chargement de petit modèle
        new_small_model = BenchmarkModel(size=1)

        def load_small_model():
            checkpoint_manager.load_model(small_model_id, new_small_model)

        results.append(
            benchmark_function(
                load_small_model, "Chargement de petit modèle", repeat=3, number=10
            )
        )

        # Benchmark 5: Chargement de grand modèle
        new_large_model = BenchmarkModel(size=4)

        def load_large_model():
            checkpoint_manager.load_model(large_model_id, new_large_model)

        results.append(
            benchmark_function(
                load_large_model, "Chargement de grand modèle", repeat=3, number=5
            )
        )

        # Benchmark 6: Sauvegarde de données
        def save_small_data():
            return checkpoint_manager.save_checkpoint(
                obj=small_data,
                type=CheckpointType.STATE,
                prefix="small_data",
                description="Petites données pour benchmark",
            )

        small_data_id = save_small_data()
        results.append(
            benchmark_function(
                save_small_data, "Sauvegarde de petites données", repeat=3, number=5
            )
        )

        # Benchmark 7: Sauvegarde de grandes données
        def save_large_data():
            return checkpoint_manager.save_checkpoint(
                obj=large_data,
                type=CheckpointType.STATE,
                prefix="large_data",
                description="Grandes données pour benchmark",
            )

        large_data_id = save_large_data()
        results.append(
            benchmark_function(
                save_large_data, "Sauvegarde de grandes données", repeat=3, number=3
            )
        )

        # Benchmark 8: Chargement de données
        def load_small_data():
            return checkpoint_manager.load_checkpoint(small_data_id)

        results.append(
            benchmark_function(
                load_small_data, "Chargement de petites données", repeat=3, number=10
            )
        )

        def load_large_data():
            return checkpoint_manager.load_checkpoint(large_data_id)

        results.append(
            benchmark_function(
                load_large_data, "Chargement de grandes données", repeat=3, number=5
            )
        )

        # Benchmark 9: Listage des checkpoints
        def list_checkpoints():
            return checkpoint_manager.list_checkpoints()

        results.append(
            benchmark_function(
                list_checkpoints, "Listage des checkpoints", repeat=3, number=20
            )
        )

    return results


def run_all_benchmarks(output_file=None):
    """Exécute tous les benchmarks et sauvegarde les résultats."""
    logger.info("Démarrage des benchmarks")

    start_time = time.time()

    # Exécuter les benchmarks
    logging_results = benchmark_logging()
    metrics_results = benchmark_metrics()
    checkpoint_results = benchmark_checkpoints()

    # Regrouper les résultats
    all_results = {
        "timestamp": time.time(),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": time.time() - start_time,
        "system_info": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "results": {
            "logging": logging_results,
            "metrics": metrics_results,
            "checkpoints": checkpoint_results,
        },
    }

    # Afficher un résumé
    logger.info("\nRésumé des benchmarks:")
    logger.info(
        f"Temps total d'exécution: {all_results['duration_seconds']:.2f} secondes"
    )

    logger.info("\nJournalisation:")
    for result in logging_results:
        logger.info(f"  {result['name']}: {result['average_ms']:.3f} ms")

    logger.info("\nMétriques:")
    for result in metrics_results:
        logger.info(f"  {result['name']}: {result['average_ms']:.3f} ms")

    logger.info("\nCheckpoints:")
    for result in checkpoint_results:
        logger.info(f"  {result['name']}: {result['average_ms']:.3f} ms")

    # Sauvegarder les résultats si demandé
    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nRésultats sauvegardés dans {output_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarks des fonctionnalités transversales"
    )
    parser.add_argument(
        "--output", "-o", help="Fichier de sortie pour les résultats (JSON)"
    )
    parser.add_argument(
        "--logging-only",
        action="store_true",
        help="Exécuter uniquement les benchmarks de journalisation",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Exécuter uniquement les benchmarks de métriques",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Exécuter uniquement les benchmarks de checkpoints",
    )

    args = parser.parse_args()

    # Créer un fichier de sortie par défaut si non spécifié
    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"benchmark_results_{timestamp}.json"

    # Exécuter les benchmarks demandés
    if args.logging_only:
        results = {"results": {"logging": benchmark_logging()}}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Résultats sauvegardés dans {args.output}")
    elif args.metrics_only:
        results = {"results": {"metrics": benchmark_metrics()}}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Résultats sauvegardés dans {args.output}")
    elif args.checkpoints_only:
        results = {"results": {"checkpoints": benchmark_checkpoints()}}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Résultats sauvegardés dans {args.output}")
    else:
        run_all_benchmarks(args.output)
