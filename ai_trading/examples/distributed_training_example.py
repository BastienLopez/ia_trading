"""
Exemple d'utilisation du parallélisme de données multi-GPU avec DistributedDataParallel (DDP).

Ce script démontre comment accélérer l'entraînement d'un modèle en répartissant
les données sur plusieurs GPUs en utilisant DistributedDataParallel (DDP) de PyTorch.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.utils.distributed_training import (
    DDPModelWrapper,
    count_available_gpus,
    get_gpu_memory_usage,
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Créer un répertoire pour les résultats
RESULTS_DIR = Path(__file__).parent / "results" / "distributed_training"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class DistributedTradingModel(nn.Module):
    """
    Un modèle de trading qui sera entraîné avec DDP.

    Ce modèle est suffisamment grand pour bénéficier de l'entraînement distribué.
    """

    def __init__(self, input_dim=128, hidden_dim=512, num_layers=4, dropout=0.2):
        """
        Initialise le modèle.

        Args:
            input_dim: Dimension d'entrée
            hidden_dim: Dimension des couches cachées
            num_layers: Nombre de couches LSTM
            dropout: Taux de dropout
        """
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)

        # Bloc LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Couches fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Couche de sortie (prédiction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor d'entrée de forme [batch_size, seq_len, input_dim]

        Returns:
            Prédictions de forme [batch_size, 1]
        """
        # Reconfigurer pour batch norm si nécessaire
        batch_size, seq_len, features = x.size()

        # Reshape et normaliser chaque timestep
        if seq_len == 1:
            # Cas spécial pour une séquence de longueur 1
            x = self.input_layer(x.squeeze(1))
            x = self.batch_norm1(x)
            x = x.unsqueeze(1)
        else:
            # Cas général
            x = x.reshape(-1, features)
            x = self.input_layer(x)
            x = self.batch_norm1(x)
            x = x.reshape(batch_size, seq_len, -1)

        # LSTM
        x, _ = self.lstm(x)

        # Utiliser seulement la dernière sortie de la séquence
        x = x[:, -1, :]

        # Couches fully connected
        x = self.fc_layers(x)

        # Normalisation de la sortie
        x = self.sigmoid(x)

        return x


def generate_synthetic_data(
    num_samples=10000, seq_length=20, input_dim=128, batch_size=64
):
    """
    Génère des données synthétiques pour tester l'entraînement distribué.

    Args:
        num_samples: Nombre total d'échantillons
        seq_length: Longueur de chaque séquence
        input_dim: Dimension des features d'entrée
        batch_size: Taille du batch

    Returns:
        DataLoader d'entraînement, DataLoader de validation
    """
    # Générer des séquences aléatoires
    X = torch.randn(num_samples, seq_length, input_dim)

    # Générer des cibles
    # (par exemple, la somme des 3 premières features de la dernière timestep > 0)
    y = ((X[:, -1, 0] + X[:, -1, 1] + X[:, -1, 2]) > 0).float().unsqueeze(1)

    # Ajouter un peu de bruit
    y = torch.clamp(y + 0.05 * torch.randn_like(y), 0.0, 1.0)

    # Diviser en ensembles d'entraînement et de validation
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Créer les datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    return train_dataset, val_dataset


def benchmark_distributed_vs_single(
    train_dataset, val_dataset, num_epochs=5, batch_size=64, use_mixed_precision=True
):
    """
    Compare les performances entre l'entraînement distribué et l'entraînement sur un seul GPU.

    Args:
        train_dataset: Dataset d'entraînement
        val_dataset: Dataset de validation
        num_epochs: Nombre d'époques d'entraînement
        batch_size: Taille du batch par GPU
        use_mixed_precision: Si True, utilise l'entraînement en précision mixte

    Returns:
        Dictionnaire avec les résultats du benchmark
    """
    # Vérifier si plusieurs GPUs sont disponibles
    num_gpus = count_available_gpus()

    if num_gpus < 2:
        logger.warning(
            "Moins de 2 GPUs disponibles, impossible de comparer l'entraînement distribué"
        )
        logger.info(f"Nombre de GPUs détectés: {num_gpus}")
        return None

    logger.info(f"Benchmark avec {num_gpus} GPUs disponibles")

    # Afficher les informations sur les GPUs
    gpu_info = get_gpu_memory_usage()
    for gpu in gpu_info:
        logger.info(
            f"GPU {gpu['device']}: {gpu['name']}, Mémoire libre: {gpu['free_gb']:.2f} GB"
        )

    # Fonction de création du modèle
    def create_model():
        return DistributedTradingModel(
            input_dim=train_dataset[0][0].shape[-1], hidden_dim=512, num_layers=4
        )

    # Fonction de création de l'optimiseur
    def create_optimizer(model_params):
        return optim.Adam(model_params, lr=0.001)

    # Fonction de création de la fonction de perte
    def create_criterion():
        return nn.BCELoss()

    # Créer le checkpoint dir
    checkpoint_dir = RESULTS_DIR / "checkpoints"

    # --- Entraînement distribué ---
    logger.info("\n=== Entraînement avec DDP sur plusieurs GPUs ===")

    ddp_start_time = time.time()

    # Utiliser le wrapper DDP
    ddp_wrapper = DDPModelWrapper(
        model_fn=create_model,
        optimizer_fn=create_optimizer,
        criterion_fn=create_criterion,
        world_size=num_gpus,  # Utiliser tous les GPUs disponibles
        mixed_precision=use_mixed_precision,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Entraîner en mode distribué
    ddp_wrapper.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        epochs=num_epochs,
        num_workers=4,
    )

    ddp_training_time = time.time() - ddp_start_time
    logger.info(f"Temps d'entraînement DDP: {ddp_training_time:.2f} secondes")

    # --- Entraînement sur un seul GPU ---
    logger.info("\n=== Entraînement sur un seul GPU ===")

    # Libérer la mémoire GPU
    torch.cuda.empty_cache()

    single_gpu_start_time = time.time()

    # Créer le modèle et le déplacer sur le GPU
    single_model = create_model().to("cuda:0")
    optimizer = create_optimizer(single_model.parameters())
    criterion = create_criterion()

    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * num_gpus,  # Même batch_size total que DDP
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * num_gpus * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Créer un GradScaler pour la précision mixte si nécessaire
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

    # Entraînement single-GPU
    for epoch in range(num_epochs):
        # Phase d'entraînement
        single_model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")

            # Forward pass avec précision mixte si activée
            if use_mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    outputs = single_model(inputs)
                    loss = criterion(outputs, targets)

                # Backward et optimize avec scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass standard
                outputs = single_model(inputs)
                loss = criterion(outputs, targets)

                # Backward et optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Époque {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}"
                )

        # Phase de validation
        single_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")

                outputs = single_model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Calculer les pertes moyennes
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        logger.info(
            f"Époque {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    single_gpu_training_time = time.time() - single_gpu_start_time
    logger.info(
        f"Temps d'entraînement Single-GPU: {single_gpu_training_time:.2f} secondes"
    )

    # Calculer le gain de performance
    speedup = (
        single_gpu_training_time / ddp_training_time if ddp_training_time > 0 else 0
    )
    logger.info(f"Accélération avec DDP: {speedup:.2f}x")

    # Résultats du benchmark
    benchmark_results = {
        "num_gpus": num_gpus,
        "ddp_training_time": ddp_training_time,
        "single_gpu_training_time": single_gpu_training_time,
        "speedup": speedup,
        "batch_size_per_gpu": batch_size,
        "total_batch_size_ddp": batch_size * num_gpus,
        "epochs": num_epochs,
        "mixed_precision": use_mixed_precision,
    }

    # Sauvegarder les résultats du benchmark
    save_benchmark_results(benchmark_results)

    return benchmark_results


def save_benchmark_results(results):
    """
    Sauvegarde les résultats du benchmark et génère des graphiques.

    Args:
        results: Dictionnaire contenant les résultats du benchmark
    """
    # Sauvegarder les données brutes
    results_file = RESULTS_DIR / "benchmark_results.txt"
    with open(results_file, "w") as f:
        f.write("=== Benchmark d'entraînement distribué ===\n\n")
        f.write(f"Nombre de GPUs: {results['num_gpus']}\n")
        f.write(f"Batch size par GPU: {results['batch_size_per_gpu']}\n")
        f.write(f"Batch size total DDP: {results['total_batch_size_ddp']}\n")
        f.write(f"Nombre d'époques: {results['epochs']}\n")
        f.write(
            f"Précision mixte: {'Activée' if results['mixed_precision'] else 'Désactivée'}\n\n"
        )

        f.write("Temps d'entraînement:\n")
        f.write(f"  DDP (multi-GPU): {results['ddp_training_time']:.2f} secondes\n")
        f.write(f"  Single-GPU: {results['single_gpu_training_time']:.2f} secondes\n\n")

        f.write(f"Accélération: {results['speedup']:.2f}x\n")

    # Créer un graphique de comparaison des temps d'entraînement
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["Single-GPU", f'DDP ({results["num_gpus"]} GPUs)'],
        [results["single_gpu_training_time"], results["ddp_training_time"]],
        color=["blue", "green"],
    )

    plt.title("Comparaison des temps d'entraînement")
    plt.ylabel("Temps (secondes)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(
        [results["single_gpu_training_time"], results["ddp_training_time"]]
    ):
        plt.text(i, v + 1, f"{v:.1f}s", ha="center")

    # Ajouter une ligne pour l'accélération idéale
    ideal_time = results["single_gpu_training_time"] / results["num_gpus"]
    plt.axhline(
        y=ideal_time,
        color="r",
        linestyle="--",
        label=f'Accélération idéale ({results["num_gpus"]}x)',
    )

    plt.text(1.1, ideal_time - 5, f"Idéal: {ideal_time:.1f}s", color="red")
    plt.legend()

    plt.savefig(RESULTS_DIR / "training_time_comparison.png")
    logger.info(
        f"Graphique de comparaison sauvegardé dans {RESULTS_DIR / 'training_time_comparison.png'}"
    )


def main(args):
    """
    Fonction principale pour exécuter la démonstration.

    Args:
        args: Arguments de la ligne de commande
    """
    logger.info("Démarrage de l'exemple d'entraînement distribué")

    # Vérifier CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA n'est pas disponible. Cet exemple nécessite des GPUs.")
        return

    # Générer des données synthétiques
    logger.info("Génération des données synthétiques...")
    train_dataset, val_dataset = generate_synthetic_data(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        input_dim=args.input_dim,
        batch_size=args.batch_size,
    )

    # Exécuter le benchmark
    logger.info("Exécution du benchmark d'entraînement distribué vs single-GPU...")
    results = benchmark_distributed_vs_single(
        train_dataset,
        val_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        use_mixed_precision=not args.no_mixed_precision,
    )

    if results:
        # Résumé final
        logger.info("\n=== Résumé du benchmark ===")
        logger.info(
            f"Entraînement distribué sur {results['num_gpus']} GPUs: {results['ddp_training_time']:.2f} secondes"
        )
        logger.info(
            f"Entraînement sur un seul GPU: {results['single_gpu_training_time']:.2f} secondes"
        )
        logger.info(
            f"Accélération: {results['speedup']:.2f}x (théorique idéale: {results['num_gpus']:.1f}x)"
        )

        # Efficacité parallèle
        efficiency = (results["speedup"] / results["num_gpus"]) * 100
        logger.info(f"Efficacité parallèle: {efficiency:.1f}%")

    logger.info(
        f"Exemple d'entraînement distribué terminé! Voir les résultats dans: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    # Analyser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(
        description="Exemple d'entraînement distribué multi-GPU"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Nombre d'échantillons pour les données synthétiques",
    )
    parser.add_argument(
        "--seq_length", type=int, default=20, help="Longueur des séquences"
    )
    parser.add_argument("--input_dim", type=int, default=128, help="Dimension d'entrée")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Taille du batch par GPU"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Nombre d'époques d'entraînement"
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Désactiver l'entraînement en précision mixte",
    )

    args = parser.parse_args()

    main(args)
