#!/usr/bin/env python
"""
Exemple d'utilisation des optimisations multiprocessing et multithreading pour le système de trading.
Ce script démontre comment optimiser l'utilisation des ressources CPU sur différentes configurations.
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Ajouter le répertoire parent au path pour pouvoir importer les modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.utils.install_optimizations import set_cpu_optimization_env_vars
from ai_trading.utils.performance_tracker import PerformanceTracker, log_memory_usage

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_multiprocessing_configurations(data, sequence_length=50, batch_size=32):
    """
    Teste différentes configurations de multithreading/multiprocessing avec DataLoader.

    Args:
        data: Données financières (DataFrame, Tensor, etc.)
        sequence_length: Longueur des séquences temporelles
        batch_size: Taille des batchs

    Returns:
        DataFrame contenant les résultats des tests
    """
    logger.info("Test des configurations multiprocessing/multithreading...")

    # Configurations à tester
    configurations = [
        {
            "name": "Synchrone (1 process)",
            "workers": 0,
            "pin_memory": False,
            "prefetch": None,
        },
        {"name": "1 worker", "workers": 1, "pin_memory": False, "prefetch": 2},
        {"name": "2 workers", "workers": 2, "pin_memory": False, "prefetch": 2},
        {"name": "4 workers", "workers": 4, "pin_memory": False, "prefetch": 2},
        {
            "name": "4 workers + pin memory",
            "workers": 4,
            "pin_memory": True,
            "prefetch": 2,
        },
        {
            "name": "Auto workers",
            "workers": -1,
            "pin_memory": True,
            "prefetch": 2,
        },  # Auto-détection du nombre de workers
    ]

    results = []

    # Créer le dataset de base
    base_dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=True,  # Utiliser le chargement paresseux pour tester le multithreading
        chunk_size=5000,
        memory_optimize=True,
    )

    # Tester chaque configuration
    for config in configurations:
        logger.info(f"Test de la configuration: {config['name']}")

        # Créer le DataLoader selon la configuration
        dataloader = get_financial_dataloader(
            dataset=base_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["workers"],
            prefetch_factor=config["prefetch"],
            pin_memory=config["pin_memory"],
            optimize_memory=True,
            persistent_workers=True if config["workers"] > 0 else False,
        )

        # Initialiser le tracker de performance
        tracker = PerformanceTracker(name=config["name"])

        # Mesurer le temps d'itération sur les données
        start_time = time.time()
        total_batches = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            total_batches += 1
            if batch_idx % 20 == 0:  # Mesurer la performance tous les 20 batchs
                tracker.measure(f"batch_{batch_idx}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        batches_per_second = total_batches / max(elapsed_time, 0.001)

        # Obtenir un résumé des mesures de performance
        perf_summary = tracker.summarize()

        # Enregistrer les résultats
        results.append(
            {
                "name": config["name"],
                "workers": config["workers"],
                "pin_memory": config["pin_memory"],
                "prefetch": config["prefetch"],
                "total_time": elapsed_time,
                "batches_per_second": batches_per_second,
                "total_batches": total_batches,
                "avg_cpu_percent": perf_summary["cpu"]["avg_percent"],
                "max_cpu_percent": perf_summary["cpu"]["max_percent"],
                "max_memory_mb": perf_summary["memory"]["max_rss"] / (1024 * 1024),
            }
        )

        logger.info(f"Résultats pour {config['name']}:")
        logger.info(f"  Temps total: {elapsed_time:.2f}s")
        logger.info(f"  Batches/s: {batches_per_second:.2f}")
        logger.info(f"  CPU moyen: {perf_summary['cpu']['avg_percent']:.1f}%")
        logger.info(
            f"  Mémoire max: {perf_summary['memory']['max_rss'] / (1024 * 1024):.1f} MB"
        )

        # Nettoyage manuel pour libérer la mémoire
        del dataloader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    return results_df


def test_prefetch_configurations(
    data, sequence_length=50, batch_size=32, num_workers=4
):
    """
    Teste l'impact de différentes valeurs de prefetch_factor sur les performances.

    Args:
        data: Données financières
        sequence_length: Longueur des séquences temporelles
        batch_size: Taille des batchs
        num_workers: Nombre de workers à utiliser

    Returns:
        DataFrame avec les résultats des tests
    """
    logger.info("Test des configurations de prefetch_factor...")

    if num_workers == 0:
        logger.warning("prefetch_factor est ignoré quand num_workers=0. Test ignoré.")
        return pd.DataFrame()

    # Valeurs de prefetch_factor à tester
    prefetch_factors = [1, 2, 4, 8]

    results = []

    # Créer le dataset de base
    base_dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=True,
        chunk_size=5000,
        memory_optimize=True,
    )

    # Tester chaque configuration de prefetch_factor
    for prefetch in prefetch_factors:
        logger.info(f"Test avec prefetch_factor={prefetch}")

        # Créer le DataLoader
        dataloader = get_financial_dataloader(
            dataset=base_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            pin_memory=True,
            optimize_memory=True,
            persistent_workers=True,
        )

        # Mesurer le temps d'itération
        tracker = PerformanceTracker(name=f"prefetch_{prefetch}")

        start_time = time.time()
        total_batches = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            total_batches += 1
            if batch_idx % 20 == 0:
                tracker.measure(f"batch_{batch_idx}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        batches_per_second = total_batches / max(elapsed_time, 0.001)

        # Obtenir le résumé des performances
        perf_summary = tracker.summarize()

        # Enregistrer les résultats
        results.append(
            {
                "prefetch_factor": prefetch,
                "total_time": elapsed_time,
                "batches_per_second": batches_per_second,
                "total_batches": total_batches,
                "avg_cpu_percent": perf_summary["cpu"]["avg_percent"],
                "max_cpu_percent": perf_summary["cpu"]["max_percent"],
                "max_memory_mb": perf_summary["memory"]["max_rss"] / (1024 * 1024),
            }
        )

        logger.info(f"Résultats pour prefetch_factor={prefetch}:")
        logger.info(f"  Temps total: {elapsed_time:.2f}s")
        logger.info(f"  Batches/s: {batches_per_second:.2f}")
        logger.info(f"  CPU moyen: {perf_summary['cpu']['avg_percent']:.1f}%")

        # Nettoyage
        del dataloader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    return results_df


def test_persistent_workers(data, sequence_length=50, batch_size=32, num_workers=4):
    """
    Compare les performances avec et sans workers persistants.

    Args:
        data: Données financières
        sequence_length: Longueur des séquences temporelles
        batch_size: Taille des batchs
        num_workers: Nombre de workers à utiliser

    Returns:
        DataFrame avec les résultats des tests
    """
    logger.info("Test de l'impact des workers persistants...")

    if num_workers == 0:
        logger.warning(
            "persistent_workers est ignoré quand num_workers=0. Test ignoré."
        )
        return pd.DataFrame()

    results = []

    # Créer le dataset de base
    base_dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=True,
        chunk_size=5000,
        memory_optimize=True,
    )

    # Tester avec et sans workers persistants
    for persistent in [False, True]:
        logger.info(f"Test avec persistent_workers={persistent}")

        # Créer le DataLoader
        dataloader = get_financial_dataloader(
            dataset=base_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
            optimize_memory=True,
            persistent_workers=persistent,
        )

        # Simuler plusieurs époques
        num_epochs = 3
        total_time = 0

        for epoch in range(num_epochs):
            tracker = PerformanceTracker(name=f"persistent_{persistent}_epoch_{epoch}")

            start_time = time.time()
            total_batches = 0

            for batch_idx, (data, targets) in enumerate(dataloader):
                total_batches += 1
                if batch_idx % 20 == 0:
                    tracker.measure(f"batch_{batch_idx}")

            epoch_time = time.time() - start_time
            total_time += epoch_time

            logger.info(f"Époque {epoch+1}/{num_epochs}, temps: {epoch_time:.2f}s")

        # Calculer les métriques moyennes
        avg_epoch_time = total_time / num_epochs
        batches_per_second = total_batches / avg_epoch_time if avg_epoch_time > 0 else 0

        # Enregistrer les résultats
        results.append(
            {
                "persistent_workers": persistent,
                "avg_epoch_time": avg_epoch_time,
                "batches_per_second": batches_per_second,
                "total_batches": total_batches,
            }
        )

        logger.info(f"Résultats pour persistent_workers={persistent}:")
        logger.info(f"  Temps moyen par époque: {avg_epoch_time:.2f}s")
        logger.info(f"  Batches/s: {batches_per_second:.2f}")

        # Nettoyage
        del dataloader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    return results_df


def visualize_results(results_df, result_type="workers"):
    """
    Visualise les résultats des tests sous forme de graphiques.

    Args:
        results_df: DataFrame contenant les résultats
        result_type: Type de résultat à visualiser ("workers", "prefetch", ou "persistent")
    """
    plt.figure(figsize=(12, 8))

    if result_type == "workers":
        # Graphique pour les configurations de workers
        bar_data = results_df["batches_per_second"]
        bar_labels = results_df["name"]

        plt.subplot(2, 1, 1)
        plt.bar(bar_labels, bar_data)
        plt.title("Performance des différentes configurations de workers")
        plt.ylabel("Batches par seconde")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.bar(bar_labels, results_df["max_cpu_percent"])
        plt.title("Utilisation CPU maximale")
        plt.ylabel("CPU (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

    elif result_type == "prefetch":
        # Graphique pour les configurations de prefetch
        bar_data = results_df["batches_per_second"]
        bar_labels = results_df["prefetch_factor"].astype(str)

        plt.subplot(2, 1, 1)
        plt.bar(bar_labels, bar_data)
        plt.title("Impact du prefetch_factor sur les performances")
        plt.xlabel("Facteur de préchargement")
        plt.ylabel("Batches par seconde")

        plt.subplot(2, 1, 2)
        plt.bar(bar_labels, results_df["max_memory_mb"])
        plt.title("Impact du prefetch_factor sur l'utilisation mémoire")
        plt.xlabel("Facteur de préchargement")
        plt.ylabel("Mémoire maximale (MB)")

    elif result_type == "persistent":
        # Graphique pour les workers persistants
        bar_data = results_df["batches_per_second"]
        bar_labels = ["Non-persistant", "Persistant"]

        plt.bar(bar_labels, bar_data)
        plt.title("Impact des workers persistants sur les performances")
        plt.ylabel("Batches par seconde")

    plt.tight_layout()

    # Enregistrer le graphique
    output_dir = Path.home() / "ai_trading_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / f"multiprocessing_{result_type}_results.png")
    logger.info(
        f"Graphique enregistré dans {output_dir / f'multiprocessing_{result_type}_results.png'}"
    )

    # Afficher le graphique
    plt.show()


def main():
    """
    Fonction principale exécutant les tests d'optimisation multiprocessing.
    """
    logger.info(
        "=== Test des optimisations multiprocessing pour le système de trading ==="
    )

    # Appliquer les optimisations des variables d'environnement
    set_cpu_optimization_env_vars()

    # Générer des données de test
    logger.info("Génération des données de test...")
    data = generate_synthetic_market_data(
        n_points=100000,
        trend=0.0005,
        volatility=0.01,
        start_price=100.0,
        with_date=True,
        cyclic_pattern=True,
        include_volume=True,
    )

    # Paramètres pour les tests
    sequence_length = 50
    batch_size = 32

    # Mesurer l'utilisation mémoire initiale
    log_memory_usage("Initial")

    # 1. Tester différentes configurations de workers
    workers_results = test_multiprocessing_configurations(
        data, sequence_length, batch_size
    )

    # Visualiser les résultats des tests de workers
    if not workers_results.empty:
        visualize_results(workers_results, "workers")

        # Trouver la configuration optimale
        best_config = workers_results.loc[
            workers_results["batches_per_second"].idxmax()
        ]
        logger.info(f"Configuration optimale: {best_config['name']}")
        logger.info(f"Performance: {best_config['batches_per_second']:.2f} batches/s")

        # Utiliser le meilleur nombre de workers pour les tests suivants
        optimal_workers = best_config["workers"]
        if optimal_workers == -1:
            # Auto-détection, utiliser un nombre fixe pour les tests suivants
            import multiprocessing

            optimal_workers = multiprocessing.cpu_count()
    else:
        optimal_workers = 4  # Valeur par défaut

    # 2. Tester différentes valeurs de prefetch_factor
    if optimal_workers > 0:
        prefetch_results = test_prefetch_configurations(
            data, sequence_length, batch_size, optimal_workers
        )

        # Visualiser les résultats des tests de prefetch
        if not prefetch_results.empty:
            visualize_results(prefetch_results, "prefetch")

            # Trouver la valeur optimale de prefetch_factor
            best_prefetch = prefetch_results.loc[
                prefetch_results["batches_per_second"].idxmax()
            ]
            logger.info(
                f"Facteur de préchargement optimal: {best_prefetch['prefetch_factor']}"
            )
            logger.info(
                f"Performance: {best_prefetch['batches_per_second']:.2f} batches/s"
            )

    # 3. Tester l'impact des workers persistants
    if optimal_workers > 0:
        persistent_results = test_persistent_workers(
            data, sequence_length, batch_size, optimal_workers
        )

        # Visualiser les résultats des tests de workers persistants
        if not persistent_results.empty:
            visualize_results(persistent_results, "persistent")

    # Résumé des recommandations
    logger.info("\n=== Recommandations d'optimisation multiprocessing ===")

    if not workers_results.empty:
        best_config = workers_results.loc[
            workers_results["batches_per_second"].idxmax()
        ]
        logger.info(f"1. Nombre optimal de workers: {best_config['workers']}")
        logger.info(
            f"   Performance: {best_config['batches_per_second']:.2f} batches/s"
        )

    if optimal_workers > 0 and not prefetch_results.empty:
        best_prefetch = prefetch_results.loc[
            prefetch_results["batches_per_second"].idxmax()
        ]
        logger.info(
            f"2. Facteur de préchargement optimal: {best_prefetch['prefetch_factor']}"
        )
        logger.info(
            f"   Performance: {best_prefetch['batches_per_second']:.2f} batches/s"
        )

    if optimal_workers > 0 and not persistent_results.empty:
        best_persistent = persistent_results.loc[
            persistent_results["batches_per_second"].idxmax()
        ]
        persistent_value = best_persistent["persistent_workers"]
        logger.info(
            f"3. Workers persistants: {'Activés' if persistent_value else 'Désactivés'}"
        )
        logger.info(
            f"   Performance: {best_persistent['batches_per_second']:.2f} batches/s"
        )

    logger.info(
        "\nCes optimisations peuvent varier selon le matériel, la taille des données et le modèle utilisé."
    )


if __name__ == "__main__":
    main()
