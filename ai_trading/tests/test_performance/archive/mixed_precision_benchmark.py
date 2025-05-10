"""
Test de performance pour l'entraînement en précision mixte.

Ce script effectue des tests de performance comparatifs entre l'entraînement
en précision standard (FP32) et en précision mixte (FP16) pour évaluer le gain
de performance et la réduction de mémoire.
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
import torch
import torch.nn as nn
import torch.optim as optim

from ai_trading.utils.advanced_logging import get_logger
from ai_trading.utils.mixed_precision import (
    MixedPrecisionWrapper,
    is_mixed_precision_supported,
    setup_mixed_precision,
    test_mixed_precision_performance,
)

# Logger pour ce script
logger = get_logger("ai_trading.tests.performance.archive.mixed_precision")


class BenchmarkModel(nn.Module):
    """Modèle PyTorch pour les benchmarks de précision mixte."""

    def __init__(self, input_size=128, hidden_size=256, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


def benchmark_mixed_precision(
    model_size="small", batch_sizes=[32, 64, 128, 256], iterations=100
):
    """
    Exécute des benchmarks de précision mixte avec différentes tailles de batch.

    Args:
        model_size: Taille du modèle ('small', 'medium', 'large')
        batch_sizes: Liste des tailles de batch à tester
        iterations: Nombre d'itérations pour chaque test

    Returns:
        dict: Résultats des benchmarks
    """
    logger.info(f"Exécution des benchmarks de précision mixte (modèle: {model_size})")

    # Définir les dimensions selon la taille du modèle
    if model_size == "small":
        input_size, hidden_size = 128, 256
    elif model_size == "medium":
        input_size, hidden_size = 256, 512
    elif model_size == "large":
        input_size, hidden_size = 512, 1024
    else:
        raise ValueError(f"Taille de modèle non reconnue: {model_size}")

    # Créer le modèle
    model = BenchmarkModel(input_size, hidden_size)

    # Résultats
    results = {
        "model_size": model_size,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "batch_results": [],
        "cuda_available": torch.cuda.is_available(),
        "mp_supported": is_mixed_precision_supported(),
    }

    # Si CUDA n'est pas disponible, retourner les résultats vides
    if not results["cuda_available"]:
        logger.warning("CUDA n'est pas disponible, benchmarks impossibles")
        return results

    # Configurer la précision mixte si prise en charge
    if results["mp_supported"]:
        setup_mixed_precision()

    # Exécuter les tests pour chaque taille de batch
    for batch_size in batch_sizes:
        logger.info(f"Test avec batch_size={batch_size}...")

        # Exécuter le test de performance
        perf_results = test_mixed_precision_performance(
            model=model,
            input_shape=(input_size,),
            batch_size=batch_size,
            iterations=iterations,
        )

        # Ajouter les résultats
        batch_result = {
            "batch_size": batch_size,
            "fp32_time": perf_results["fp32_time"],
            "fp16_time": perf_results["fp16_time"],
            "speedup": perf_results["speedup"],
            "fp32_memory_mb": perf_results["fp32_memory_mb"],
            "fp16_memory_mb": perf_results["fp16_memory_mb"],
            "memory_reduction": perf_results["memory_reduction"],
        }

        results["batch_results"].append(batch_result)

    return results


def visualize_benchmark_results(results, output_file=None):
    """
    Visualise les résultats des benchmarks de précision mixte.

    Args:
        results: Résultats des benchmarks
        output_file: Fichier de sortie pour le graphique
    """
    # Créer la figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Extraire les données
    batch_sizes = [r["batch_size"] for r in results["batch_results"]]
    speedups = [r["speedup"] for r in results["batch_results"]]
    memory_reductions = [r["memory_reduction"] for r in results["batch_results"]]

    # Graphique d'accélération
    ax1.bar(range(len(batch_sizes)), speedups, color="royalblue")
    ax1.set_xlabel("Taille de batch")
    ax1.set_ylabel("Accélération (x)")
    ax1.set_title("Accélération avec précision mixte")
    ax1.set_xticks(range(len(batch_sizes)))
    ax1.set_xticklabels(batch_sizes)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(speedups):
        ax1.text(i, v + 0.05, f"{v:.2f}x", ha="center")

    # Graphique de réduction mémoire
    ax2.bar(range(len(batch_sizes)), memory_reductions, color="forestgreen")
    ax2.set_xlabel("Taille de batch")
    ax2.set_ylabel("Réduction mémoire (x)")
    ax2.set_title("Réduction mémoire avec précision mixte")
    ax2.set_xticks(range(len(batch_sizes)))
    ax2.set_xticklabels(batch_sizes)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(memory_reductions):
        ax2.text(i, v + 0.05, f"{v:.2f}x", ha="center")

    # Ajuster la mise en page
    plt.tight_layout()

    # Ajouter des informations sur le modèle
    plt.figtext(
        0.5,
        0.01,
        f"Modèle: {results['model_size']} (input: {results['input_size']}, hidden: {results['hidden_size']})",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "lavender", "alpha": 0.5, "pad": 5},
    )

    # Sauvegarder si demandé
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Graphique sauvegardé dans {output_file}")

    plt.show()


def run_training_benchmark(batch_size=64, epochs=5, use_mixed_precision=True):
    """
    Exécute un benchmark d'entraînement avec ou sans précision mixte.

    Args:
        batch_size: Taille du batch
        epochs: Nombre d'époques
        use_mixed_precision: Utiliser la précision mixte

    Returns:
        dict: Résultats du benchmark
    """
    logger.info(f"Benchmark d'entraînement (mixed_precision={use_mixed_precision})")

    if not torch.cuda.is_available():
        logger.warning("CUDA n'est pas disponible, benchmark impossible")
        return {"error": "CUDA non disponible"}

    # Créer un modèle et des données
    model = BenchmarkModel(input_size=128, hidden_size=256)
    model.to("cuda")

    # Générer des données synthétiques
    inputs = torch.randn(batch_size * 10, 128, device="cuda")
    targets = torch.randint(0, 10, (batch_size * 10,), device="cuda")

    # Créer dataset et dataloader
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Optimiseur et critère
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Configurer la précision mixte si demandé
    if use_mixed_precision and is_mixed_precision_supported():
        setup_mixed_precision()
        wrapper = MixedPrecisionWrapper(model, optimizer)
    else:
        wrapper = None

    # Fonction forward
    def forward_fn(batch):
        x, _ = batch
        return model(x)

    # Fonction loss
    def loss_fn(outputs, batch):
        _, y = batch
        return criterion(outputs, y)

    # Mesurer le temps et la mémoire
    start_time = time.time()
    peak_memory = 0

    # Boucle d'entraînement
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            # Mise à jour avec ou sans mixed precision
            if wrapper:
                loss = wrapper.training_step(batch, forward_fn, loss_fn)
            else:
                # Entraînement standard
                optimizer.zero_grad()
                outputs = forward_fn(batch)
                loss = loss_fn(outputs, batch)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            # Mesurer la mémoire
            current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
            peak_memory = max(peak_memory, current_memory)

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    # Calculer le temps total
    total_time = time.time() - start_time

    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "mixed_precision": use_mixed_precision,
        "total_time": total_time,
        "time_per_epoch": total_time / epochs,
        "peak_memory_mb": peak_memory,
    }


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Benchmarks de précision mixte")
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Taille du modèle pour les benchmarks",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Tailles de batch à tester",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Nombre d'itérations pour chaque test",
    )
    parser.add_argument(
        "--output", "-o", help="Fichier de sortie pour les résultats (JSON)"
    )
    parser.add_argument(
        "--plot", "-p", help="Fichier de sortie pour le graphique (PNG)"
    )
    parser.add_argument(
        "--training", action="store_true", help="Exécuter le benchmark d'entraînement"
    )

    args = parser.parse_args()

    # Afficher les informations système
    logger.info(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Appareil CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"Précision mixte supportée: {is_mixed_precision_supported()}")

    if args.training:
        # Exécuter le benchmark d'entraînement
        fp32_results = run_training_benchmark(use_mixed_precision=False)
        fp16_results = run_training_benchmark(use_mixed_precision=True)

        # Afficher les résultats comparatifs
        if "error" not in fp32_results and "error" not in fp16_results:
            speedup = fp32_results["total_time"] / fp16_results["total_time"]
            memory_reduction = (
                fp32_results["peak_memory_mb"] / fp16_results["peak_memory_mb"]
            )

            logger.info("\n=== Résultats du benchmark d'entraînement ===")
            logger.info(f"Temps FP32: {fp32_results['total_time']:.2f}s")
            logger.info(f"Temps FP16: {fp16_results['total_time']:.2f}s")
            logger.info(f"Accélération: {speedup:.2f}x")
            logger.info(f"Mémoire FP32: {fp32_results['peak_memory_mb']:.2f} MB")
            logger.info(f"Mémoire FP16: {fp16_results['peak_memory_mb']:.2f} MB")
            logger.info(f"Réduction mémoire: {memory_reduction:.2f}x")

            # Sauvegarder les résultats si demandé
            if args.output:
                results = {
                    "fp32": fp32_results,
                    "fp16": fp16_results,
                    "speedup": speedup,
                    "memory_reduction": memory_reduction,
                }
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Résultats sauvegardés dans {args.output}")
    else:
        # Exécuter les benchmarks de performance
        results = benchmark_mixed_precision(
            model_size=args.model_size,
            batch_sizes=args.batch_sizes,
            iterations=args.iterations,
        )

        # Afficher un résumé
        if results["cuda_available"] and results["mp_supported"]:
            logger.info("\n=== Résumé des benchmarks ===")
            for r in results["batch_results"]:
                logger.info(
                    f"Batch {r['batch_size']}: Accélération {r['speedup']:.2f}x, "
                    f"Réduction mémoire {r['memory_reduction']:.2f}x"
                )

        # Sauvegarder les résultats si demandé
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Résultats sauvegardés dans {args.output}")

        # Visualiser les résultats si demandé
        if args.plot and results["cuda_available"] and results["mp_supported"]:
            visualize_benchmark_results(results, args.plot)


if __name__ == "__main__":
    main()
