#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du module de quantification de modèles.

Ce script montre comment utiliser les différentes fonctionnalités de quantification
pour optimiser les modèles de deep learning en termes de taille et de vitesse d'inférence.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_trading.utils.model_quantization import (
    benchmark_inference_speed,
    compare_model_performance,
    export_quantized_model,
    quantize_model_dynamic,
    quantize_model_static,
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TradingModel(nn.Module):
    """Modèle simple pour la prédiction de prix d'actifs financiers."""

    def __init__(self, input_dim=10, hidden_dim=50, output_dim=1):
        super().__init__()
        # Couches simples pour la démonstration
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.features(x)
        return self.regressor(features)


class ConvTradingModel(nn.Module):
    """Modèle CNN pour traiter des séries temporelles de prix."""

    def __init__(self, input_channels=1, sequence_length=50, output_dim=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(2),  # Réduire la taille de moitié
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # Calculer la taille d'entrée pour la couche linéaire
        self.fc_input_dim = 32 * (sequence_length // 2)
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        conv_out = self.conv_layers(x)
        # Reshape pour la couche fully connected
        flat = conv_out.reshape(batch_size, -1)
        return self.fc(flat)


def generate_dummy_data(n_samples=1000, input_dim=10, time_steps=None):
    """
    Génère des données synthétiques pour entraîner et tester nos modèles.

    Args:
        n_samples: Nombre d'échantillons
        input_dim: Dimension d'entrée pour le modèle MLP
        time_steps: Nombre de pas de temps pour le modèle CNN (si None, données pour MLP)

    Returns:
        Données X et cibles y
    """
    if time_steps is None:
        # Données pour le modèle MLP
        X = np.random.randn(n_samples, input_dim).astype(np.float32)
        # Formule simple: somme pondérée des caractéristiques + bruit
        weights = np.random.randn(input_dim)
        y = np.dot(X, weights) + 0.1 * np.random.randn(n_samples)
        y = y.reshape(-1, 1).astype(np.float32)
        return torch.tensor(X), torch.tensor(y)
    else:
        # Données pour le modèle CNN (séries temporelles)
        X = np.random.randn(n_samples, 1, time_steps).astype(np.float32)
        # Les cibles sont basées sur la moyenne mobile des derniers points
        y = np.mean(X[:, 0, -5:], axis=1, keepdims=True)
        return torch.tensor(X), torch.tensor(y)


def train_model(
    model,
    train_data,
    train_targets,
    val_data,
    val_targets,
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    """
    Entraîne un modèle sur les données fournies.

    Args:
        model: Le modèle PyTorch à entraîner
        train_data: Données d'entraînement X
        train_targets: Cibles d'entraînement y
        val_data: Données de validation X
        val_targets: Cibles de validation y
        epochs: Nombre d'époques d'entraînement
        batch_size: Taille des batchs
        learning_rate: Taux d'apprentissage

    Returns:
        Le modèle entraîné et l'historique des pertes
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    n_samples = len(train_data)
    n_batches = (n_samples + batch_size - 1) // batch_size

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Mélanger les données
        indices = torch.randperm(n_samples)
        train_data_shuffled = train_data[indices]
        train_targets_shuffled = train_targets[indices]

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            # Obtenir le batch
            batch_data = train_data_shuffled[start_idx:end_idx]
            batch_targets = train_targets_shuffled[start_idx:end_idx]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end_idx - start_idx)

        # Calculer la perte moyenne sur l'époque
        epoch_loss /= n_samples

        # Évaluer sur la validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_targets).item()
        model.train()

        # Enregistrer les métriques
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            f"Époque {epoch+1}/{epochs} - Perte entraînement: {epoch_loss:.4f} - Perte validation: {val_loss:.4f}"
        )

    return model, history


def calibrate_model(model, calib_data, batch_size=32):
    """
    Calibre un modèle préparé pour la quantification statique.

    Args:
        model: Le modèle préparé pour la quantification
        calib_data: Données de calibration
        batch_size: Taille des batchs pour la calibration
    """
    model.eval()
    n_samples = len(calib_data)
    n_batches = (n_samples + batch_size - 1) // batch_size

    logger.info("Calibration du modèle...")
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            # Obtenir le batch
            batch_data = calib_data[start_idx:end_idx]

            # Passer les données dans le modèle pour calibration
            _ = model(batch_data)


def plot_results(
    original_model, quantized_model, test_data, test_targets, save_path=None
):
    """
    Trace les prédictions des modèles originaux et quantifiés.

    Args:
        original_model: Le modèle PyTorch original
        quantized_model: Le modèle PyTorch quantifié
        test_data: Données de test
        test_targets: Cibles de test
        save_path: Chemin où sauvegarder le graphique
    """
    with torch.no_grad():
        original_preds = original_model(test_data).numpy()
        quantized_preds = quantized_model(test_data).numpy()

    # Convertir cibles en numpy
    test_targets = test_targets.numpy()

    # Créer le graphique
    plt.figure(figsize=(10, 6))

    # Tracer un échantillon des données
    sample_size = min(100, len(test_data))
    indices = np.random.choice(len(test_data), sample_size, replace=False)

    plt.scatter(indices, test_targets[indices], color="blue", label="Cibles réelles")
    plt.scatter(
        indices,
        original_preds[indices],
        color="green",
        alpha=0.5,
        label="Prédictions modèle original",
    )
    plt.scatter(
        indices,
        quantized_preds[indices],
        color="red",
        alpha=0.5,
        label="Prédictions modèle quantifié",
    )

    plt.title("Comparaison des prédictions: Original vs Quantifié")
    plt.xlabel("Échantillon")
    plt.ylabel("Valeur prédite")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique sauvegardé dans {save_path}")

    plt.show()


def demonstration_mlp():
    """Démonstration de quantification avec un modèle MLP simple."""
    logger.info("\n" + "=" * 80)
    logger.info("DÉMONSTRATION DE QUANTIFICATION AVEC UN MODÈLE MLP")
    logger.info("=" * 80)

    # Configuration
    input_dim = 20
    hidden_dim = 100
    output_dim = 1
    n_samples = 10000

    # Générer des données
    logger.info("Génération des données...")
    X, y = generate_dummy_data(n_samples=n_samples, input_dim=input_dim)

    # Diviser en ensembles d'entraînement, validation et test
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size

    train_data, train_targets = X[:train_size], y[:train_size]
    val_data, val_targets = (
        X[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
    )
    test_data, test_targets = X[train_size + val_size :], y[train_size + val_size :]

    # Créer et entraîner le modèle
    logger.info("Création et entraînement du modèle...")
    model = TradingModel(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )
    model, history = train_model(
        model,
        train_data,
        train_targets,
        val_data,
        val_targets,
        epochs=5,
        batch_size=64,
        learning_rate=0.001,
    )

    # Évaluer le modèle original
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        original_preds = model(test_data)
        original_loss = criterion(original_preds, test_targets).item()

    logger.info(f"Performance du modèle original - MSE: {original_loss:.4f}")

    # 1. Quantification dynamique
    logger.info("\nDémonstration de la quantification dynamique...")
    dynamic_quantized_model = quantize_model_dynamic(model)

    # Évaluer le modèle quantifié dynamiquement
    with torch.no_grad():
        dynamic_preds = dynamic_quantized_model(test_data)
        dynamic_loss = criterion(dynamic_preds, test_targets).item()

    logger.info(
        f"Performance du modèle quantifié dynamiquement - MSE: {dynamic_loss:.4f}"
    )

    # Comparer les performances (taille et vitesse)
    logger.info("\nComparaison des performances...")
    benchmark_results = benchmark_inference_speed(
        model,
        dynamic_quantized_model,
        test_data,
        num_iterations=100,
        warmup_iterations=10,
    )

    logger.info(
        f"Temps d'inférence modèle original: {benchmark_results['original_inference_time_ms']:.2f} ms"
    )
    logger.info(
        f"Temps d'inférence modèle quantifié: {benchmark_results['quantized_inference_time_ms']:.2f} ms"
    )
    logger.info(f"Accélération: {benchmark_results['speedup_factor']:.2f}x")

    # 2. Quantification statique (si supportée)
    try:
        # Vérifier qu'un backend de quantification est disponible
        backends = []
        for backend in ["fbgemm", "qnnpack"]:
            try:
                torch.backends.quantized.engine = backend
                backends.append(backend)
            except:
                pass

        if not backends:
            logger.warning(
                "Aucun backend de quantification statique n'est disponible sur ce système"
            )
        else:
            logger.info(
                f"\nDémonstration de la quantification statique avec backend {backends[0]}..."
            )
            torch.backends.quantized.engine = backends[0]

            # Préparation pour la quantification statique
            model.eval()  # Assurez-vous que le modèle est en mode évaluation

            # Définir une fonction de calibration
            def calibration_fn(prepared_model):
                calibrate_model(prepared_model, val_data)

            # Quantification statique complète
            static_quantized_model = quantize_model_static(
                model, calibration_fn, qconfig_name=backends[0]
            )

            # Évaluer le modèle quantifié statiquement
            with torch.no_grad():
                static_preds = static_quantized_model(test_data)
                static_loss = criterion(static_preds, test_targets).item()

            logger.info(
                f"Performance du modèle quantifié statiquement - MSE: {static_loss:.4f}"
            )

            # Exporter le modèle quantifié
            export_dir = Path("ai_trading/info_retour/examples/model_quantization")
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / "static_quantized_model.pt"

            export_quantized_model(
                static_quantized_model, str(export_path), input_sample=test_data[:1]
            )
            logger.info(f"Modèle quantifié exporté dans {export_path}")

            # Tracer les résultats
            plot_path = export_dir / "quantization_comparison.png"
            plot_results(
                model,
                static_quantized_model,
                test_data[:100],
                test_targets[:100],
                str(plot_path),
            )

    except Exception as e:
        logger.warning(f"Quantification statique non supportée ou erreur: {e}")

        # Créer le répertoire pour les résultats même en cas d'erreur
        export_dir = Path("ai_trading/info_retour/examples/model_quantization")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Créer un rapport d'erreur
        with open(export_dir / "quantization_error_report.txt", "w") as f:
            f.write(f"Erreur de quantification statique: {e}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {torch.__version__}\n")


def demonstration_cnn():
    """Démonstration de quantification avec un modèle CNN."""
    logger.info("\n" + "=" * 80)
    logger.info("DÉMONSTRATION DE QUANTIFICATION AVEC UN MODÈLE CNN")
    logger.info("=" * 80)

    # Configuration
    sequence_length = 50
    n_samples = 5000

    # Générer des données
    logger.info("Génération des données...")
    X, y = generate_dummy_data(n_samples=n_samples, time_steps=sequence_length)

    # Diviser en ensembles d'entraînement, validation et test
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size

    train_data, train_targets = X[:train_size], y[:train_size]
    val_data, val_targets = (
        X[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
    )
    test_data, test_targets = X[train_size + val_size :], y[train_size + val_size :]

    # Créer et entraîner le modèle
    logger.info("Création et entraînement du modèle CNN...")
    model = ConvTradingModel(
        input_channels=1, sequence_length=sequence_length, output_dim=1
    )
    model, history = train_model(
        model,
        train_data,
        train_targets,
        val_data,
        val_targets,
        epochs=5,
        batch_size=64,
        learning_rate=0.001,
    )

    # Évaluer le modèle original
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        original_preds = model(test_data)
        original_loss = criterion(original_preds, test_targets).item()

    logger.info(f"Performance du modèle CNN original - MSE: {original_loss:.4f}")

    # Quantification dynamique
    logger.info("\nQuantification dynamique du modèle CNN...")
    dynamic_quantized_model = quantize_model_dynamic(model)

    # Évaluer le modèle quantifié
    with torch.no_grad():
        quantized_preds = dynamic_quantized_model(test_data)
        quantized_loss = criterion(quantized_preds, test_targets).item()

    logger.info(f"Performance du modèle CNN quantifié - MSE: {quantized_loss:.4f}")

    # Comparer les performances
    def test_fn(m):
        start_time = time()
        with torch.no_grad():
            for _ in range(10):
                outputs = m(test_data)
        end_time = time()

        with torch.no_grad():
            preds = m(test_data)
            mse = criterion(preds, test_targets).item()

        return {"inference_time": (end_time - start_time) * 1000 / 10, "mse": mse}  # ms

    comparison = compare_model_performance(model, dynamic_quantized_model, test_fn)

    logger.info("\nComparaison des performances CNN:")
    if comparison.get("original_size") is not None:
        logger.info(
            f"Taille originale: {comparison.get('original_size')/1024/1024:.2f} Mo"
        )
    else:
        logger.info("Taille originale: Non disponible")

    if comparison.get("quantized_size") is not None:
        logger.info(
            f"Taille quantifiée: {comparison.get('quantized_size')/1024/1024:.2f} Mo"
        )
    else:
        logger.info("Taille quantifiée: Non disponible")

    if comparison.get("size_reduction_percent") is not None:
        logger.info(
            f"Réduction de taille: {comparison.get('size_reduction_percent', 0):.2f}%"
        )
    else:
        logger.info("Réduction de taille: Non disponible")

    if "original_metrics" in comparison and "quantized_metrics" in comparison:
        orig_metrics = comparison["original_metrics"]
        quant_metrics = comparison["quantized_metrics"]

        if "inference_time" in orig_metrics and "inference_time" in quant_metrics:
            speedup = orig_metrics["inference_time"] / quant_metrics["inference_time"]
            logger.info(
                f"Temps d'inférence original: {orig_metrics['inference_time']:.2f} ms"
            )
            logger.info(
                f"Temps d'inférence quantifié: {quant_metrics['inference_time']:.2f} ms"
            )
            logger.info(f"Accélération: {speedup:.2f}x")

        if "mse" in orig_metrics and "mse" in quant_metrics:
            accuracy_change = quant_metrics["mse"] - orig_metrics["mse"]
            logger.info(f"MSE original: {orig_metrics['mse']:.4f}")
            logger.info(f"MSE quantifié: {quant_metrics['mse']:.4f}")
            logger.info(f"Changement de MSE: {accuracy_change:.4f}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Démonstration de quantification de modèles"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn", "both"],
        default="both",
        help="Type de modèle à démontrer (mlp, cnn, ou both)",
    )
    args = parser.parse_args()

    # Créer le répertoire de résultats s'il n'existe pas
    results_dir = Path("ai_trading/info_retour/examples/model_quantization")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Exécuter les démonstrations selon les arguments
    if args.model in ["mlp", "both"]:
        demonstration_mlp()

    if args.model in ["cnn", "both"]:
        demonstration_cnn()


if __name__ == "__main__":
    main()
