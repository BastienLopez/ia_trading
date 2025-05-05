"""
Démonstration des avantages du chargement paresseux (lazy loading) et mise en cache des features
pour les grands datasets financiers.

Ce script permet de comparer différentes approches de chargement de données:
1. Chargement standard (en mémoire)
2. Chargement paresseux (lazy loading)
3. Chargement paresseux avec mise en cache des features

Il mesure les performances (temps et mémoire) pour chaque approche.
"""

import argparse
import gc
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from ai_trading.data.data_optimizers import convert_to_parquet

# Import des modules de notre projet
from ai_trading.data.financial_dataset import (
    FinancialDataset,
    get_feature_transform_fn,
    get_financial_dataloader,
)
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data

# Chemin pour sauvegarder les résultats
RESULTS_DIR = Path("ai_trading/info_retour/visualisations/misc")
DATA_DIR = Path("ai_trading/info_retour/data/demo")


def get_memory_usage():
    """Retourne l'utilisation mémoire du processus actuel en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def create_demo_data(n_points=500000, save_dir=None):
    """
    Crée des données synthétiques volumineuses pour la démonstration.

    Args:
        n_points: Nombre de points de données
        save_dir: Répertoire où sauvegarder les données

    Returns:
        DataFrame avec données synthétiques et chemin vers fichier Parquet si sauvegardé
    """
    logger.info(f"Génération de {n_points} points de données synthétiques...")

    data = generate_synthetic_market_data(
        n_points=n_points,
        trend=0.0001,
        volatility=0.01,
        start_price=100.0,
        include_volume=True,
        cyclic_pattern=True,
        seasonal_periods=20,
        with_anomalies=True,
    )

    # Ajouter quelques indicateurs techniques pour augmenter la dimensionnalité
    logger.info("Calcul des indicateurs techniques...")

    # Moyennes mobiles
    data["ma_5"] = data["close"].rolling(5).mean().fillna(method="bfill")
    data["ma_20"] = data["close"].rolling(20).mean().fillna(method="bfill")
    data["ma_50"] = data["close"].rolling(50).mean().fillna(method="bfill")

    # Bandes de Bollinger
    data["volatility"] = data["close"].rolling(20).std().fillna(0)
    data["upper_band"] = data["ma_20"] + 2 * data["volatility"]
    data["lower_band"] = data["ma_20"] - 2 * data["volatility"]

    # RSI simplifié
    delta = data["close"].diff().fillna(0)
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss

    avg_gain = gain.rolling(14).mean().fillna(0)
    avg_loss = loss.rolling(14).mean().fillna(0)

    rs = avg_gain / avg_loss.replace(0, 1e-8)  # Éviter division par zéro
    data["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    data["ema_12"] = data["close"].ewm(span=12).mean()
    data["ema_26"] = data["close"].ewm(span=26).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9).mean()

    # Momentum
    data["momentum"] = data["close"].pct_change(periods=10).fillna(0)

    logger.info(f"Dataset créé avec {len(data)} lignes et {len(data.columns)} colonnes")

    # Sauvegarder en format optimisé si demandé
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # Sauvegarder en CSV standard et Parquet optimisé
        csv_path = save_dir / "demo_data.csv"
        data.to_csv(csv_path)
        logger.info(f"Données sauvegardées en CSV: {csv_path}")

        parquet_path = save_dir / "demo_data.parquet"
        convert_to_parquet(
            data,
            str(parquet_path),
            compression="snappy",
            partition_cols=None,  # Pas de partitionnement pour cet exemple simple
        )
        logger.info(f"Données sauvegardées en Parquet optimisé: {parquet_path}")

        return data, str(parquet_path)

    return data, None


def create_complex_transform():
    """
    Crée une fonction de transformation complexe qui bénéficiera de la mise en cache.

    Returns:
        Fonction de transformation avec cache
    """

    @get_feature_transform_fn(cache_size=500)
    def complex_transform(tensor):
        """Transformation complexe simulée qui bénéficiera du cache."""
        # Simuler un calcul complexe/coûteux
        time.sleep(
            0.001
        )  # Petite pause artificiellement pour simuler un calcul coûteux

        # Normalisation par fenêtre
        window_mean = tensor.mean(dim=0, keepdim=True)
        window_std = tensor.std(dim=0, keepdim=True) + 1e-8
        normalized = (tensor - window_mean) / window_std

        # Quelques opérations supplémentaires coûteuses
        ema_fast = torch.zeros_like(tensor)
        ema_slow = torch.zeros_like(tensor)
        alpha_fast = 0.2
        alpha_slow = 0.05

        # Calcul d'EMA manuellement (pour simuler un calcul complexe)
        ema_fast[0] = tensor[0]
        ema_slow[0] = tensor[0]
        for i in range(1, len(tensor)):
            ema_fast[i] = alpha_fast * tensor[i] + (1 - alpha_fast) * ema_fast[i - 1]
            ema_slow[i] = alpha_slow * tensor[i] + (1 - alpha_slow) * ema_slow[i - 1]

        # Combiner les features
        result = torch.cat(
            [
                normalized,
                ema_fast.unsqueeze(-1),
                ema_slow.unsqueeze(-1),
                (ema_fast - ema_slow).unsqueeze(-1),
            ],
            dim=-1,
        )

        return result

    return complex_transform


def benchmark_loading_methods(
    data, parquet_path=None, sequence_length=50, batch_size=32
):
    """
    Compare différentes méthodes de chargement de données.

    Args:
        data: DataFrame contenant les données
        parquet_path: Chemin vers fichier Parquet (optionnel)
        sequence_length: Longueur des séquences
        batch_size: Taille des batchs

    Returns:
        Dictionnaire avec résultats des benchmarks
    """
    results = {}
    memory_snapshots = []

    # Caractéristiques des données
    data_size = len(data)
    feature_count = len(data.columns)

    # Nombre d'exemples à extraire pour le test
    n_samples_to_read = 1000

    # Créer la transformation complexe
    complex_transform = create_complex_transform()

    # Enregistrer l'utilisation mémoire avant chargement
    gc.collect()  # Forcer le nettoyage mémoire
    initial_memory = get_memory_usage()
    memory_snapshots.append(("Initial", initial_memory))

    logger.info(
        f"\n{'='*60}\nBenchmark des méthodes de chargement de données\n{'='*60}"
    )
    logger.info(
        f"Taille du dataset: {data_size} points, {feature_count} caractéristiques"
    )
    logger.info(f"Mémoire initiale: {initial_memory:.2f} MB\n")

    # ---------- TEST 1: CHARGEMENT STANDARD ----------
    logger.info(f"\n[1/4] Test du chargement standard (en mémoire)...")

    start_time = time.time()

    standard_dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=False,
        transform=None,  # Sans transformation pour le moment
    )

    init_time = time.time() - start_time
    post_init_memory = get_memory_usage()
    memory_snapshots.append(("Standard - Après init", post_init_memory))

    # Tester accès séquentiel
    start_time = time.time()
    for i in range(n_samples_to_read):
        sequence, target = standard_dataset[i]
    sequential_time = time.time() - start_time

    # Tester accès aléatoire
    start_time = time.time()
    indices = np.random.randint(0, len(standard_dataset), n_samples_to_read)
    for i in indices:
        sequence, target = standard_dataset[i]
    random_time = time.time() - start_time

    # Mesures de mémoire et nettoyage
    post_access_memory = get_memory_usage()
    memory_snapshots.append(("Standard - Après accès", post_access_memory))

    results["standard"] = {
        "init_time": init_time,
        "sequential_time": sequential_time,
        "random_time": random_time,
        "memory_usage": post_access_memory - initial_memory,
    }

    logger.info(f"Temps d'initialisation: {init_time:.4f}s")
    logger.info(
        f"Temps accès séquentiel ({n_samples_to_read} échantillons): {sequential_time:.4f}s"
    )
    logger.info(
        f"Temps accès aléatoire ({n_samples_to_read} échantillons): {random_time:.4f}s"
    )
    logger.info(f"Utilisation mémoire: {post_access_memory - initial_memory:.2f} MB")

    # Nettoyer la mémoire
    del standard_dataset
    gc.collect()
    time.sleep(1)  # Laisser le temps au système de libérer la mémoire

    # ---------- TEST 2: CHARGEMENT PARESSEUX ----------
    logger.info(f"\n[2/4] Test du chargement paresseux (lazy loading)...")

    start_time = time.time()

    lazy_dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=True,
        chunk_size=10000,
        transform=None,
    )

    init_time = time.time() - start_time
    post_init_memory = get_memory_usage()
    memory_snapshots.append(("Lazy - Après init", post_init_memory))

    # Tester accès séquentiel
    start_time = time.time()
    for i in range(n_samples_to_read):
        sequence, target = lazy_dataset[i]
    sequential_time = time.time() - start_time
    post_seq_memory = get_memory_usage()
    memory_snapshots.append(("Lazy - Après accès séquentiel", post_seq_memory))

    # Tester accès aléatoire
    start_time = time.time()
    indices = np.random.randint(0, len(lazy_dataset), n_samples_to_read)
    for i in indices:
        sequence, target = lazy_dataset[i]
    random_time = time.time() - start_time

    # Mesures de mémoire et nettoyage
    post_access_memory = get_memory_usage()
    memory_snapshots.append(("Lazy - Après accès aléatoire", post_access_memory))

    results["lazy"] = {
        "init_time": init_time,
        "sequential_time": sequential_time,
        "random_time": random_time,
        "memory_usage": post_access_memory - initial_memory,
    }

    logger.info(f"Temps d'initialisation: {init_time:.4f}s")
    logger.info(
        f"Temps accès séquentiel ({n_samples_to_read} échantillons): {sequential_time:.4f}s"
    )
    logger.info(
        f"Temps accès aléatoire ({n_samples_to_read} échantillons): {random_time:.4f}s"
    )
    logger.info(f"Utilisation mémoire: {post_access_memory - initial_memory:.2f} MB")

    # Nettoyer la mémoire
    del lazy_dataset
    gc.collect()
    time.sleep(1)

    # ---------- TEST 3: CHARGEMENT AVEC TRANSFORMATION COMPLEXE ----------
    logger.info(f"\n[3/4] Test avec transformation complexe (sans cache)...")

    # Dataset avec transformation mais sans cache
    start_time = time.time()

    transform_nocache_dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=False,
        transform=lambda x: complex_transform(x),  # Wrapper sans cache
    )

    init_time = time.time() - start_time

    # Premier accès (transformations calculées)
    start_time = time.time()
    for i in range(min(50, n_samples_to_read)):  # Limiter car plus lent
        sequence, target = transform_nocache_dataset[i]
    first_access_time = time.time() - start_time

    # Deuxième accès aux mêmes éléments (transformations recalculées)
    start_time = time.time()
    for i in range(min(50, n_samples_to_read)):
        sequence, target = transform_nocache_dataset[i]
    second_access_time = time.time() - start_time

    results["transform_nocache"] = {
        "init_time": init_time,
        "first_access_time": first_access_time,
        "second_access_time": second_access_time,
    }

    logger.info(f"Temps d'initialisation: {init_time:.4f}s")
    logger.info(f"Premier accès (50 échantillons): {first_access_time:.4f}s")
    logger.info(f"Deuxième accès (50 échantillons): {second_access_time:.4f}s")
    logger.info(f"Ratio temps: {second_access_time/first_access_time:.2f}x")

    # Nettoyer la mémoire
    del transform_nocache_dataset
    gc.collect()
    time.sleep(1)

    # ---------- TEST 4: CHARGEMENT AVEC TRANSFORMATION ET CACHE ----------
    logger.info(f"\n[4/4] Test avec transformation complexe (avec cache)...")

    # Dataset avec transformation et cache
    start_time = time.time()

    transform_cache_dataset = FinancialDataset(
        data=data,
        sequence_length=sequence_length,
        is_train=True,
        lazy_loading=True,
        chunk_size=10000,
        transform=complex_transform,  # Avec cache
    )

    init_time = time.time() - start_time

    # Premier accès (mise en cache)
    start_time = time.time()
    for i in range(min(50, n_samples_to_read)):
        sequence, target = transform_cache_dataset[i]
    first_access_time = time.time() - start_time

    # Deuxième accès aux mêmes éléments (utilisation du cache)
    start_time = time.time()
    for i in range(min(50, n_samples_to_read)):
        sequence, target = transform_cache_dataset[i]
    second_access_time = time.time() - start_time

    results["transform_cache"] = {
        "init_time": init_time,
        "first_access_time": first_access_time,
        "second_access_time": second_access_time,
    }

    logger.info(f"Temps d'initialisation: {init_time:.4f}s")
    logger.info(f"Premier accès (50 échantillons): {first_access_time:.4f}s")
    logger.info(f"Deuxième accès (50 échantillons): {second_access_time:.4f}s")
    logger.info(
        f"Ratio d'accélération: {first_access_time/max(second_access_time, 0.0001):.2f}x"
    )

    # ---------- BENCHMARK AVEC DATALOADER ----------
    if parquet_path:
        logger.info(f"\nTest DataLoader avec fichier Parquet et lazy loading...")

        # Dataset optimisé depuis fichier Parquet avec lazy loading
        parquet_dataset = FinancialDataset(
            data=parquet_path,
            sequence_length=sequence_length,
            is_train=True,
            lazy_loading=True,
            chunk_size=10000,
            transform=complex_transform,
        )

        # DataLoader optimisé
        dataloader = get_financial_dataloader(
            parquet_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
        )

        # Mesurer le temps pour itérer sur quelques batchs
        start_time = time.time()
        batch_count = 0

        for batch_data in dataloader:
            batch_sequences, batch_targets = batch_data
            # Faire une opération simple pour s'assurer que le chargement est terminé
            if torch.cuda.is_available():
                batch_sequences = batch_sequences.cuda()
                _ = batch_sequences + 1.0

            batch_count += 1
            if batch_count >= 20:
                break

        dataloader_time = time.time() - start_time

        logger.info(f"Temps pour 20 batchs avec DataLoader: {dataloader_time:.4f}s")
        logger.info(f"Temps moyen par batch: {dataloader_time/20:.4f}s")

        results["dataloader"] = {
            "batch_time": dataloader_time / 20,
            "total_time": dataloader_time,
        }

    # Générer des graphiques de résultats
    plot_benchmark_results(results, memory_snapshots)

    return results


def plot_benchmark_results(results, memory_snapshots):
    """
    Génère des graphiques de comparaison des différentes méthodes de chargement.

    Args:
        results: Dictionnaire avec les résultats des benchmarks
        memory_snapshots: Liste des mesures de mémoire aux différentes étapes
    """
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Graphique d'utilisation mémoire
    plt.figure(figsize=(10, 6))
    labels = [snap[0] for snap in memory_snapshots]
    memory_values = [snap[1] for snap in memory_snapshots]

    plt.bar(labels, memory_values, color="skyblue")
    plt.title("Utilisation mémoire des différentes méthodes de chargement")
    plt.ylabel("Mémoire utilisée (MB)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "memory_usage.png")

    # Graphique de temps d'initialisation
    plt.figure(figsize=(10, 6))
    init_times = []
    labels = []

    for method, data in results.items():
        if "init_time" in data:
            init_times.append(data["init_time"])
            labels.append(method.replace("_", " ").title())

    plt.bar(labels, init_times, color="lightgreen")
    plt.title("Temps d'initialisation des différentes méthodes")
    plt.ylabel("Temps (s)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "init_times.png")

    # Graphique de comparaison des temps d'accès
    if "standard" in results and "lazy" in results:
        plt.figure(figsize=(10, 6))
        methods = ["standard", "lazy"]
        sequential_times = [results[m]["sequential_time"] for m in methods]
        random_times = [results[m]["random_time"] for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        plt.bar(x - width / 2, sequential_times, width, label="Accès séquentiel")
        plt.bar(x + width / 2, random_times, width, label="Accès aléatoire")

        plt.xlabel("Méthode de chargement")
        plt.ylabel("Temps (s)")
        plt.title("Comparaison des temps d'accès")
        plt.xticks(x, [m.replace("_", " ").title() for m in methods])
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "access_times.png")

    # Graphique pour transformation avec et sans cache
    if "transform_nocache" in results and "transform_cache" in results:
        plt.figure(figsize=(10, 6))
        methods = ["transform_nocache", "transform_cache"]
        first_access = [results[m]["first_access_time"] for m in methods]
        second_access = [results[m]["second_access_time"] for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        plt.bar(x - width / 2, first_access, width, label="Premier accès")
        plt.bar(x + width / 2, second_access, width, label="Deuxième accès")

        plt.xlabel("Méthode de chargement")
        plt.ylabel("Temps (s)")
        plt.title("Impact de la mise en cache des transformations")
        plt.xticks(x, ["Sans cache", "Avec cache"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "transform_caching.png")

    logger.info(f"Graphiques sauvegardés dans le répertoire {RESULTS_DIR}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Démonstration du chargement paresseux et mise en cache des features"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=500000,
        help="Nombre de points de données à générer",
    )
    parser.add_argument(
        "--seq-length", type=int, default=50, help="Longueur des séquences"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Taille des batchs")
    args = parser.parse_args()

    # Créer les répertoires nécessaires
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Vérifier si les données existent déjà
    parquet_path = DATA_DIR / "demo_data.parquet"
    if parquet_path.exists():
        logger.info(f"Utilisation des données existantes: {parquet_path}")
        data = pd.read_parquet(parquet_path)
        parquet_path_str = str(parquet_path)
    else:
        # Générer et sauvegarder les données
        data, parquet_path_str = create_demo_data(
            n_points=args.points, save_dir=DATA_DIR
        )

    # Exécuter les benchmarks
    results = benchmark_loading_methods(
        data,
        parquet_path=parquet_path_str,
        sequence_length=args.seq_length,
        batch_size=args.batch_size,
    )

    # Afficher les résultats finaux
    logger.info("\nRésumé des performances:")
    logger.info(f"{'='*60}")

    if "standard" in results and "lazy" in results:
        init_speedup = results["standard"]["init_time"] / max(
            results["lazy"]["init_time"], 0.0001
        )
        mem_reduction = (
            results["standard"]["memory_usage"]
            / max(results["lazy"]["memory_usage"], 1)
        ) * 100 - 100

        logger.info(
            f"Accélération initialisation (lazy vs standard): {init_speedup:.2f}x"
        )
        logger.info(f"Réduction mémoire (lazy vs standard): {mem_reduction:.2f}%")

    if "transform_nocache" in results and "transform_cache" in results:
        cache_speedup = results["transform_nocache"]["second_access_time"] / max(
            results["transform_cache"]["second_access_time"], 0.0001
        )
        logger.info(
            f"Accélération avec cache pour transformations: {cache_speedup:.2f}x"
        )

    logger.info(f"{'='*60}")
    logger.info(f"Démonstration terminée. Consultez les graphiques dans {RESULTS_DIR}")


if __name__ == "__main__":
    main()
