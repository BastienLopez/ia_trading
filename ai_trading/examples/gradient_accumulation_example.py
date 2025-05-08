"""
Exemple d'utilisation de l'accumulation de gradient pour simuler de plus grands batchs.

Ce script démontre comment utiliser l'accumulation de gradient pour entraîner
un modèle avec des batchs virtuellement plus grands que ce que la mémoire GPU permettrait.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ai_trading.utils.gpu_rtx_optimizer import setup_rtx_optimization
from ai_trading.utils.gradient_accumulation import (
    GradientAccumulator,
    train_with_gradient_accumulation,
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Créer un répertoire pour les résultats
RESULTS_DIR = Path(__file__).parent / "results" / "gradient_accumulation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SimpleModel(nn.Module):
    """Un modèle simple pour tester l'accumulation de gradient."""

    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_synthetic_data(num_samples=1000, input_dim=10):
    """Génère des données synthétiques pour l'entraînement."""
    X = torch.randn(num_samples, input_dim)
    # Fonction non linéaire pour les cibles (y = sum(x^2) + noise)
    y = torch.sum(X * X, dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
    return X, y


def train_with_different_accumulation(
    train_loader, val_loader, input_dim, device, epochs=5
):
    """
    Entraîne plusieurs modèles avec différentes étapes d'accumulation
    pour comparer les performances.
    """
    # Configurations à tester
    accumulation_steps_list = [1, 2, 4, 8]
    results = {}

    for accumulation_steps in accumulation_steps_list:
        logger.info(f"\nTest avec {accumulation_steps} étapes d'accumulation:")

        # Créer un modèle frais pour chaque test
        model = SimpleModel(input_dim=input_dim).to(device)

        # Calculer la taille de batch effective
        effective_batch_size = train_loader.batch_size * accumulation_steps
        logger.info(f"Taille de batch effective: {effective_batch_size}")

        # Créer l'optimiseur et le critère
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Historique d'entraînement
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Entraînement avec accumulation de gradient
            train_metrics = train_with_gradient_accumulation(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                accumulation_steps=accumulation_steps,
                gradient_clip=1.0,
            )

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Enregistrer les métriques
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(avg_val_loss)

            logger.info(
                f"Époque {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        # Stocker les résultats pour ce nombre d'étapes d'accumulation
        results[accumulation_steps] = history

    return results


def plot_comparison(results, save_path):
    """Génère un graphique comparant les différentes configurations."""
    plt.figure(figsize=(12, 10))

    # Plot des pertes d'entraînement
    plt.subplot(2, 1, 1)
    for steps, history in results.items():
        plt.plot(
            history["train_loss"], label=f"{steps} étapes (batch effectif={steps*32})"
        )

    plt.title("Perte d'entraînement par étapes d'accumulation")
    plt.xlabel("Époques")
    plt.ylabel("Perte (MSE)")
    plt.legend()
    plt.grid(True)

    # Plot des pertes de validation
    plt.subplot(2, 1, 2)
    for steps, history in results.items():
        plt.plot(
            history["val_loss"], label=f"{steps} étapes (batch effectif={steps*32})"
        )

    plt.title("Perte de validation par étapes d'accumulation")
    plt.xlabel("Époques")
    plt.ylabel("Perte (MSE)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Graphique de comparaison sauvegardé dans {save_path}")


def demo_gradient_accumulator_class():
    """
    Démontre l'utilisation de la classe GradientAccumulator
    pour un contrôle manuel de l'accumulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Générer des données
    X, y = generate_synthetic_data(num_samples=500, input_dim=10)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Créer un modèle
    model = SimpleModel(input_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Initialiser l'accumulateur de gradient
    accumulator = GradientAccumulator(
        model=model, optimizer=optimizer, accumulation_steps=4, gradient_clip=1.0
    )

    logger.info("\nDémonstration de la classe GradientAccumulator:")

    # Entraînement manuel avec l'accumulateur
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass avec accumulation
            accumulator.backward(loss)

            # Accumuler la perte
            total_loss += loss.item()

            # Mettre à jour les poids si nécessaire
            if accumulator.step():
                logger.info(
                    f"Mise à jour des poids effectuée après {accumulator.accumulation_steps} batchs"
                )

        # Afficher la perte moyenne
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Époque {epoch+1}/3 - Loss: {avg_loss:.4f}")

    return "Démonstration GradientAccumulator terminée"


def main():
    """Fonction principale pour exécuter les démonstrations."""
    # Appliquer les optimisations RTX si disponible
    if torch.cuda.is_available():
        setup_rtx_optimization()
        device = torch.device("cuda")
        logger.info(f"Utilisation de CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA non disponible, utilisation du CPU")

    # Générer des données synthétiques
    input_dim = 10
    X_train, y_train = generate_synthetic_data(num_samples=2000, input_dim=input_dim)
    X_val, y_val = generate_synthetic_data(num_samples=500, input_dim=input_dim)

    # Créer les datasets et dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Utiliser une petite taille de batch pour simuler des contraintes de mémoire
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Entraîner avec différentes configurations d'accumulation
    results = train_with_different_accumulation(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        device=device,
        epochs=5,
    )

    # Générer un graphique comparant les résultats
    plot_path = RESULTS_DIR / "gradient_accumulation_comparison.png"
    plot_comparison(results, save_path=plot_path)

    # Démontrer l'utilisation de la classe GradientAccumulator
    demo_gradient_accumulator_class()

    logger.info("\nDémonstration terminée!")
    logger.info(f"Voir les résultats dans: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
