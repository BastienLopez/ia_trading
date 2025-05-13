"""
Exemple d'utilisation des tests de performance.

Ce script démontre comment utiliser les tests de performance pour évaluer
l'impact des fonctionnalités transversales sur les performances du système.
"""

import argparse
import json
import os
import sys
import time

# Ajouter le répertoire parent au chemin Python
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai_trading.tests.test_performance.benchmark import run_all_benchmarks
from ai_trading.utils.advanced_logging import get_logger
from ai_trading.utils.checkpoint_manager import get_checkpoint_manager
from ai_trading.utils.performance_logger import (
    get_performance_tracker,
    start_metrics_collection,
    stop_metrics_collection,
)

# Logger pour cet exemple
logger = get_logger("ai_trading.tests.performance.archive.performance_test")


def simulate_workload(intensity=1, duration=1.0):
    """
    Simule une charge de travail pour les tests.

    Args:
        intensity: Niveau d'intensité (1=faible, 2=moyen, 3=élevé)
        duration: Durée approximative en secondes
    """
    start_time = time.time()

    # Ajuster la taille des données selon l'intensité
    size = 100 * intensity * intensity

    # Créer des tenseurs de grande taille
    tensors = []
    matrices = []

    while time.time() - start_time < duration:
        # Allocation de mémoire
        tensor = torch.randn(size, size // 2)
        tensors.append(tensor)

        # Calcul intensif
        matrix = tensor @ tensor.T
        matrices.append(matrix)

        # Libérer de la mémoire si trop gourmand
        if len(tensors) > 5:
            tensors.pop(0)
            matrices.pop(0)

    # Retourner quelque chose pour éviter l'optimisation
    return len(tensors), sum(matrix.sum().item() for matrix in matrices)


def test_logging_impact():
    """Teste l'impact de la journalisation sur les performances."""
    logger.info("Test de l'impact de la journalisation")
    tracker = get_performance_tracker("logging_impact")

    # Test 1: Sans journalisation
    tracker.start("without_logging")
    simulate_workload(intensity=2, duration=2.0)
    duration_without = tracker.stop("without_logging")

    # Test 2: Avec journalisation intensive
    test_logger = get_logger("ai_trading.test.intensive_logging", {"level": 10})

    tracker.start("with_logging")
    for i in range(100):
        test_logger.debug(f"Message de débogage {i}")
        if i % 10 == 0:
            test_logger.info(f"Étape {i}")
        simulate_workload(intensity=1, duration=0.02)
    duration_with = tracker.stop("with_logging")

    # Afficher les résultats
    logger.info(f"Temps sans journalisation: {duration_without:.2f}s")
    logger.info(f"Temps avec journalisation intensive: {duration_with:.2f}s")
    logger.info(
        f"Surcoût de la journalisation: {(duration_with - duration_without) / duration_without * 100:.1f}%"
    )

    return {
        "without_logging": duration_without,
        "with_logging": duration_with,
        "overhead_percent": (duration_with - duration_without) / duration_without * 100,
    }


def test_metrics_collection_impact():
    """Teste l'impact de la collecte des métriques sur les performances."""
    logger.info("Test de l'impact de la collecte des métriques")
    tracker = get_performance_tracker("metrics_impact")

    # Test 1: Sans collecte de métriques
    tracker.start("without_metrics")
    simulate_workload(intensity=2, duration=3.0)
    duration_without = tracker.stop("without_metrics")

    # Test 2: Avec collecte de métriques
    collector = start_metrics_collection(interval=0.5, log_to_file=False)

    tracker.start("with_metrics")
    simulate_workload(intensity=2, duration=3.0)
    duration_with = tracker.stop("with_metrics")

    stop_metrics_collection()

    # Afficher les résultats
    logger.info(f"Temps sans collecte de métriques: {duration_without:.2f}s")
    logger.info(f"Temps avec collecte de métriques: {duration_with:.2f}s")
    logger.info(
        f"Surcoût de la collecte: {(duration_with - duration_without) / duration_without * 100:.1f}%"
    )

    return {
        "without_metrics": duration_without,
        "with_metrics": duration_with,
        "overhead_percent": (duration_with - duration_without) / duration_without * 100,
    }


def test_checkpoint_impact():
    """Teste l'impact de la gestion des checkpoints sur les performances."""
    logger.info("Test de l'impact de la gestion des checkpoints")
    tracker = get_performance_tracker("checkpoint_impact")

    # Créer des données pour le test
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 200), torch.nn.ReLU(), torch.nn.Linear(200, 50)
    )

    data = {
        "tensors": {"input": torch.randn(1000, 100), "output": torch.randn(1000, 50)},
        "params": {"learning_rate": 0.01, "batch_size": 32},
    }

    # Test 1: Sans checkpoints
    tracker.start("without_checkpoints")
    for i in range(10):
        # Simuler un entraînement
        output = model(data["tensors"]["input"][:100])
        loss = torch.mean((output - data["tensors"]["output"][:100]) ** 2)

        # Simuler une charge de travail
        simulate_workload(intensity=1, duration=0.2)
    duration_without = tracker.stop("without_checkpoints")

    # Test 2: Avec checkpoints
    checkpoint_manager = get_checkpoint_manager()

    tracker.start("with_checkpoints")
    for i in range(10):
        # Simuler un entraînement
        output = model(data["tensors"]["input"][:100])
        loss = torch.mean((output - data["tensors"]["output"][:100]) ** 2)

        # Simuler une charge de travail
        simulate_workload(intensity=1, duration=0.2)

        # Sauvegarder un checkpoint à chaque itération
        if i % 2 == 0:
            checkpoint_manager.save_model(
                model=model,
                name=f"test_model_{i}",
                description=f"Modèle de test à l'itération {i}",
                metrics={"loss": loss.item()},
            )
    duration_with = tracker.stop("with_checkpoints")

    # Afficher les résultats
    logger.info(f"Temps sans checkpoints: {duration_without:.2f}s")
    logger.info(f"Temps avec checkpoints: {duration_with:.2f}s")
    logger.info(
        f"Surcoût des checkpoints: {(duration_with - duration_without) / duration_without * 100:.1f}%"
    )

    return {
        "without_checkpoints": duration_without,
        "with_checkpoints": duration_with,
        "overhead_percent": (duration_with - duration_without) / duration_without * 100,
    }


def visualize_benchmark_results(results, output_file=None):
    """
    Visualise les résultats des benchmarks.

    Args:
        results: Résultats des benchmarks
        output_file: Fichier de sortie pour le graphique
    """
    plt.figure(figsize=(12, 8))

    # Créer 3 sous-graphiques
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 1. Graphique pour la journalisation
    logging_results = results["results"]["logging"]
    names = [result["name"] for result in logging_results]
    avg_times = [result["average_ms"] for result in logging_results]
    min_times = [result["min_ms"] for result in logging_results]
    max_times = [result["max_ms"] for result in logging_results]

    x = np.arange(len(names))
    width = 0.6

    ax1.bar(x, avg_times, width, label="Temps moyen", color="skyblue")
    ax1.errorbar(
        x,
        avg_times,
        yerr=[
            [avg - min for avg, min in zip(avg_times, min_times)],
            [max - avg for avg, max in zip(avg_times, max_times)],
        ],
        fmt="o",
        color="black",
        capsize=5,
    )

    ax1.set_xlabel("Test")
    ax1.set_ylabel("Temps (ms)")
    ax1.set_title("Benchmarks de journalisation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # 2. Graphique pour les métriques
    metrics_results = results["results"]["metrics"]
    names = [result["name"] for result in metrics_results]
    avg_times = [result["average_ms"] for result in metrics_results]
    min_times = [result["min_ms"] for result in metrics_results]
    max_times = [result["max_ms"] for result in metrics_results]

    x = np.arange(len(names))

    ax2.bar(x, avg_times, width, label="Temps moyen", color="lightgreen")
    ax2.errorbar(
        x,
        avg_times,
        yerr=[
            [avg - min for avg, min in zip(avg_times, min_times)],
            [max - avg for avg, max in zip(avg_times, max_times)],
        ],
        fmt="o",
        color="black",
        capsize=5,
    )

    ax2.set_xlabel("Test")
    ax2.set_ylabel("Temps (ms)")
    ax2.set_title("Benchmarks de collecte de métriques")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # 3. Graphique pour les checkpoints
    checkpoint_results = results["results"]["checkpoints"]
    names = [result["name"] for result in checkpoint_results]
    avg_times = [result["average_ms"] for result in checkpoint_results]
    min_times = [result["min_ms"] for result in checkpoint_results]
    max_times = [result["max_ms"] for result in checkpoint_results]

    x = np.arange(len(names))

    ax3.bar(x, avg_times, width, label="Temps moyen", color="salmon")
    ax3.errorbar(
        x,
        avg_times,
        yerr=[
            [avg - min for avg, min in zip(avg_times, min_times)],
            [max - avg for avg, max in zip(avg_times, max_times)],
        ],
        fmt="o",
        color="black",
        capsize=5,
    )

    ax3.set_xlabel("Test")
    ax3.set_ylabel("Temps (ms)")
    ax3.set_title("Benchmarks de gestion des checkpoints")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Sauvegarder le graphique si demandé
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Graphique sauvegardé dans {output_file}")

    plt.show()


def main():
    """Fonction principale pour la démonstration des tests de performance."""
    parser = argparse.ArgumentParser(
        description="Tests de performance pour les fonctionnalités transversales"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Exécuter les benchmarks détaillés"
    )
    parser.add_argument(
        "--output", "-o", help="Fichier de sortie pour les résultats (JSON)"
    )
    parser.add_argument(
        "--plot", "-p", help="Fichier de sortie pour le graphique (PNG)"
    )

    args = parser.parse_args()

    logger.info("Démarrage des tests de performance")

    if args.benchmark:
        # Exécuter les benchmarks détaillés
        results = run_all_benchmarks(args.output)

        # Visualiser les résultats si demandé
        if args.plot:
            visualize_benchmark_results(results, args.plot)
    else:
        # Exécuter les tests d'impact simples
        logger.info("\n=== Tests d'impact des fonctionnalités ===\n")

        # 1. Test de l'impact de la journalisation
        logging_impact = test_logging_impact()

        # 2. Test de l'impact de la collecte des métriques
        metrics_impact = test_metrics_collection_impact()

        # 3. Test de l'impact des checkpoints
        checkpoint_impact = test_checkpoint_impact()

        # Résumé des résultats
        logger.info("\n=== Résumé des tests ===\n")
        logger.info(
            f"Impact de la journalisation: {logging_impact['overhead_percent']:.1f}% de surcoût"
        )
        logger.info(
            f"Impact de la collecte des métriques: {metrics_impact['overhead_percent']:.1f}% de surcoût"
        )
        logger.info(
            f"Impact des checkpoints: {checkpoint_impact['overhead_percent']:.1f}% de surcoût"
        )

        # Sauvegarder les résultats si demandé
        if args.output:
            results = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "impact_tests": {
                    "logging": logging_impact,
                    "metrics": metrics_impact,
                    "checkpoints": checkpoint_impact,
                },
            }

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nRésultats sauvegardés dans {args.output}")

    logger.info("Tests de performance terminés")


if __name__ == "__main__":
    main()
