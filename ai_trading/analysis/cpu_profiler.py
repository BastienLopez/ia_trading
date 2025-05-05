import cProfile
import io
import multiprocessing
import os
import pstats
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.data.data_optimizers import convert_to_hdf5, convert_to_parquet
from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data


def generate_test_data(n_points=50000, save_dir=None):
    """Génère des données de test et les sauvegarde optionnellement dans différents formats."""
    print(f"Génération de {n_points} points de données de test...")

    data = generate_synthetic_market_data(
        n_points=n_points,
        trend=0.0005,
        volatility=0.01,
        start_price=100.0,
        with_date=True,
        cyclic_pattern=True,
        include_volume=True,
    )

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder en CSV
        csv_path = save_dir / "test_data.csv"
        data.to_csv(csv_path)
        print(f"Données enregistrées au format CSV: {csv_path}")

        # Sauvegarder en Parquet si disponible
        try:
            parquet_path = save_dir / "test_data.parquet"
            convert_to_parquet(data, parquet_path)
            print(f"Données enregistrées au format Parquet: {parquet_path}")
        except ImportError:
            print("PyArrow non disponible, format Parquet ignoré")

        # Sauvegarder en HDF5 si disponible
        try:
            hdf5_path = save_dir / "test_data.h5"
            convert_to_hdf5(data, hdf5_path)
            print(f"Données enregistrées au format HDF5: {hdf5_path}")
        except ImportError:
            print("PyTables non disponible, format HDF5 ignoré")

    return data


def profile_dataset_creation(
    data, sequence_length=50, lazy_loading=False, chunk_size=None
):
    """Profile la création du dataset."""
    print(
        f"\nProfil de création du dataset (lazy_loading={lazy_loading}, chunk_size={chunk_size})"
    )

    # Utiliser cProfile pour mesurer les performances
    profiler = cProfile.Profile()
    profiler.enable()

    # Créer le dataset
    dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=lazy_loading,
        chunk_size=chunk_size,
    )

    # Arrêter le profiling
    profiler.disable()

    # Imprimer les statistiques
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    ps.print_stats(20)  # Afficher les 20 fonctions les plus gourmandes
    print(s.getvalue())

    return dataset


def profile_dataloader_performance(dataset, batch_size=32, num_workers_list=None):
    """Profile la performance du DataLoader avec différents nombres de workers."""
    if num_workers_list is None:
        # Tester avec différents nombres de workers
        num_workers_list = [0, 1, 2, 4, 8]

    # Limiter au nombre maximal de CPUs disponibles
    max_workers = multiprocessing.cpu_count()
    num_workers_list = [min(w, max_workers) for w in num_workers_list]

    results = {}

    for num_workers in num_workers_list:
        print(f"\nTest DataLoader avec {num_workers} workers:")

        # Créer le DataLoader avec le nombre de workers spécifié
        dataloader = get_financial_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        # Mesurer le temps d'itération
        start_time = time.time()

        # Parcourir tous les batches
        total_batches = 0
        for _ in tqdm(dataloader, desc=f"Traitement avec {num_workers} workers"):
            total_batches += 1

        elapsed_time = time.time() - start_time
        batches_per_second = total_batches / max(elapsed_time, 0.001)

        print(f"  Temps total: {elapsed_time:.2f}s")
        print(f"  Batches/s: {batches_per_second:.2f}")

        results[num_workers] = {
            "total_time": elapsed_time,
            "batches_per_second": batches_per_second,
            "total_batches": total_batches,
        }

    # Afficher le résumé des performances
    print("\nRésumé des performances DataLoader:")
    for workers, perf in results.items():
        print(f"  {workers} workers: {perf['batches_per_second']:.2f} batches/s")

    # Déterminer le nombre optimal de workers
    optimal_workers = max(results.items(), key=lambda x: x[1]["batches_per_second"])[0]
    print(f"\nNombre optimal de workers: {optimal_workers}")

    return results, optimal_workers


def profile_async_vs_sync_loading(data_path, sequence_length=50, batch_size=32):
    """Compare le chargement synchrone vs asynchrone des données."""
    print("\nComparaison du chargement synchrone vs asynchrone:")

    # Test avec chargement synchrone (lazy_loading=False)
    start_time = time.time()
    sync_dataset = FinancialDataset(
        data=data_path,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=False,
    )
    sync_dataloader = get_financial_dataloader(
        dataset=sync_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Chargement synchrone
        pin_memory=False,
    )
    sync_load_time = time.time() - start_time

    # Parcourir tous les batches
    sync_iter_start = time.time()
    for _ in tqdm(sync_dataloader, desc="Chargement synchrone"):
        pass
    sync_iter_time = time.time() - sync_iter_start

    print(
        f"Chargement synchrone - Temps d'initialisation: {sync_load_time:.2f}s, Temps d'itération: {sync_iter_time:.2f}s"
    )

    # Test avec chargement asynchrone (lazy_loading=True et workers>0)
    start_time = time.time()
    async_dataset = FinancialDataset(
        data=data_path,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=True,
        chunk_size=5000,
    )
    async_dataloader = get_financial_dataloader(
        dataset=async_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Chargement asynchrone
        prefetch_factor=2,
        pin_memory=True,
    )
    async_load_time = time.time() - start_time

    # Parcourir tous les batches
    async_iter_start = time.time()
    for _ in tqdm(async_dataloader, desc="Chargement asynchrone"):
        pass
    async_iter_time = time.time() - async_iter_start

    print(
        f"Chargement asynchrone - Temps d'initialisation: {async_load_time:.2f}s, Temps d'itération: {async_iter_time:.2f}s"
    )

    # Calculer l'accélération
    total_sync_time = sync_load_time + sync_iter_time
    total_async_time = async_load_time + async_iter_time
    speedup = total_sync_time / max(total_async_time, 0.001)

    print(f"\nAccélération du chargement asynchrone: {speedup:.2f}x")

    return {
        "sync": {
            "init_time": sync_load_time,
            "iter_time": sync_iter_time,
            "total_time": total_sync_time,
        },
        "async": {
            "init_time": async_load_time,
            "iter_time": async_iter_time,
            "total_time": total_async_time,
        },
        "speedup": speedup,
    }


def profile_prefetch_factors(dataset, batch_size=32, num_workers=4):
    """Profile l'impact de différents facteurs de prefetch."""
    if num_workers == 0:
        print("Le prefetch_factor est ignoré quand num_workers=0. Test ignoré.")
        return {}

    prefetch_factors = [1, 2, 4, 8, 16]
    results = {}

    print("\nTest de différents facteurs de préchargement (prefetch_factor):")

    for factor in prefetch_factors:
        print(f"\nTest avec prefetch_factor={factor}:")

        # Créer le DataLoader avec le facteur de préchargement spécifié
        dataloader = get_financial_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=factor,
            pin_memory=torch.cuda.is_available(),
        )

        # Mesurer le temps d'itération
        start_time = time.time()

        # Parcourir tous les batches
        total_batches = 0
        for _ in tqdm(dataloader, desc=f"prefetch_factor={factor}"):
            total_batches += 1

        elapsed_time = time.time() - start_time
        batches_per_second = total_batches / max(elapsed_time, 0.001)

        print(f"  Temps total: {elapsed_time:.2f}s")
        print(f"  Batches/s: {batches_per_second:.2f}")

        results[factor] = {
            "total_time": elapsed_time,
            "batches_per_second": batches_per_second,
            "total_batches": total_batches,
        }

    # Afficher le résumé des performances
    print("\nRésumé des performances prefetch_factor:")
    for factor, perf in results.items():
        print(f"  prefetch_factor={factor}: {perf['batches_per_second']:.2f} batches/s")

    # Déterminer le facteur optimal de préchargement
    optimal_factor = max(results.items(), key=lambda x: x[1]["batches_per_second"])[0]
    print(f"\nFacteur optimal de préchargement: {optimal_factor}")

    return results, optimal_factor


def run_cpu_profiling():
    """Exécute l'ensemble des profilages CPU et recommande des optimisations."""
    print(
        "=== Profilage CPU pour l'optimisation des performances de chargement des données ===\n"
    )

    # Générer ou charger les données de test
    data_dir = Path("ai_trading/profiling_data")
    data_file = data_dir / "test_data.parquet"

    if data_file.exists():
        print(f"Chargement des données de test depuis {data_file}")
        data = pd.read_parquet(data_file)
    else:
        data = generate_test_data(n_points=50000, save_dir=data_dir)

    print(f"Données de taille: {data.shape}")

    # 1. Profiler la création du dataset (standard vs lazy loading)
    std_dataset = profile_dataset_creation(data, lazy_loading=False)
    lazy_dataset = profile_dataset_creation(data, lazy_loading=True, chunk_size=5000)

    # 2. Profiler la performance du DataLoader avec différents nombres de workers
    _, optimal_workers = profile_dataloader_performance(lazy_dataset, batch_size=32)

    # 3. Profiler l'impact de différents facteurs de prefetch
    if optimal_workers > 0:
        _, optimal_prefetch = profile_prefetch_factors(
            lazy_dataset, batch_size=32, num_workers=optimal_workers
        )
    else:
        optimal_prefetch = 2  # Valeur par défaut

    # 4. Comparer le chargement synchrone vs asynchrone
    if hasattr(data, "to_parquet"):
        parquet_path = data_dir / "test_data.parquet"
        if not parquet_path.exists():
            data.to_parquet(parquet_path)
        async_results = profile_async_vs_sync_loading(parquet_path)

    # Résumé des recommandations
    print("\n=== Recommandations d'optimisation CPU pour le chargement des données ===")
    print(
        f"1. Utilisation de lazy_loading=True pour réduire la charge mémoire et CPU à l'initialisation"
    )
    print(f"2. Nombre optimal de workers pour DataLoader: {optimal_workers}")
    print(f"3. Facteur optimal de préchargement (prefetch_factor): {optimal_prefetch}")
    if "async_results" in locals():
        print(
            f"4. Utilisation du chargement asynchrone pour une accélération de {async_results['speedup']:.2f}x"
        )

    print(
        "\nRemarque: Les optimisations optimales peuvent varier selon la taille des données et la configuration matérielle."
    )


if __name__ == "__main__":
    run_cpu_profiling()
