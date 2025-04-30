"""
Exemple d'utilisation des optimisations pour RTX 3070.
Ce script montre comment implémenter les recommandations d'optimisation
pour accélérer l'entraînement sur GPU RTX série 30xx.
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Importer notre module d'optimisation RTX
from ai_trading.utils.gpu_rtx_optimizer import (
    MixedPrecisionTrainer,
    optimize_batch_size,
    setup_rtx_optimization,
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rtx_example")


# Modèle simple pour la démonstration
class SimpleFinancialModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=1):
        super(SimpleFinancialModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # Prendre uniquement la dernière sortie de la séquence
        last_output = lstm_out[:, -1, :]
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def generate_dummy_financial_data(n_samples=1000, seq_length=20, n_features=10):
    """Génère des données financières factices pour la démonstration."""
    # Générer des séquences aléatoires
    X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    # Générer des cibles (simple somme des dernières valeurs + bruit)
    y = np.sum(X[:, -5:, 0], axis=1, keepdims=True).astype(np.float32)
    y += np.random.randn(n_samples, 1).astype(np.float32) * 0.1

    return X, y


def compare_training_performance(
    model, X_train, y_train, epochs=5, batch_sizes=[32, 64]
):
    """
    Compare les performances d'entraînement avec différentes optimisations.

    Args:
        model: Le modèle à entraîner
        X_train, y_train: Données d'entraînement
        epochs: Nombre d'époques
        batch_sizes: Liste des tailles de batch à tester

    Returns:
        results: Dictionnaire contenant les résultats de performance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # Convertir les données en tensors PyTorch
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Configurations à tester
    configurations = [
        {"name": "Standard (FP32)", "mixed_precision": False, "tensor_cores": False},
        {
            "name": "Mixed Precision (FP16)",
            "mixed_precision": True,
            "tensor_cores": False,
        },
        {"name": "Tensor Cores", "mixed_precision": False, "tensor_cores": True},
        {"name": "Full Optimization", "mixed_precision": True, "tensor_cores": True},
    ]

    for batch_size in batch_sizes:
        for config in configurations:
            config_name = f"{config['name']} (BS={batch_size})"
            logger.info(f"Test de la configuration: {config_name}")

            # Réinitialiser le modèle pour chaque test
            model_copy = SimpleFinancialModel(
                input_dim=X_train.shape[2], hidden_dim=128, output_dim=y_train.shape[1]
            ).to(device)

            # Appliquer les optimisations RTX selon la configuration
            setup_rtx_optimization(
                enable_tensor_cores=config["tensor_cores"],
                enable_mixed_precision=config["mixed_precision"],
                optimize_memory_allocation=True,
            )

            # Créer le DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4 if device.type == "cuda" else 0,
            )

            # Préparer l'optimiseur
            optimizer = optim.Adam(model_copy.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Utiliser notre classe MixedPrecisionTrainer si mixed_precision est activé
            if config["mixed_precision"]:
                trainer = MixedPrecisionTrainer(model_copy, optimizer)

            # Mesurer le temps d'entraînement
            start_time = time.time()
            epoch_times = []
            batch_times = []
            losses = []

            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_loss = 0.0
                batch_count = 0

                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    batch_start = time.time()

                    # Transférer les données sur GPU
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Entraînement avec ou sans mixed precision
                    if config["mixed_precision"]:
                        loss = trainer.training_step(inputs, targets, criterion)
                    else:
                        # Entraînement standard
                        optimizer.zero_grad()
                        outputs = model_copy(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        loss = loss.item()

                    # Enregistrer les métriques
                    epoch_loss += loss
                    batch_count += 1
                    batch_times.append(time.time() - batch_start)

                    # Limiter le nombre de batch pour la démonstration
                    if batch_idx >= 20:
                        break

                # Calculer la perte moyenne de l'époque
                avg_epoch_loss = epoch_loss / batch_count
                losses.append(avg_epoch_loss)

                # Enregistrer le temps de l'époque
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)

                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s"
                )

            # Calculer les métriques
            total_time = time.time() - start_time
            avg_epoch_time = np.mean(epoch_times)
            avg_batch_time = np.mean(batch_times) * 1000  # en ms

            # Enregistrer les résultats
            results[config_name] = {
                "total_time": total_time,
                "avg_epoch_time": avg_epoch_time,
                "avg_batch_time": avg_batch_time,
                "final_loss": losses[-1],
            }

            logger.info(
                f"Configuration {config_name}: "
                f"Temps total: {total_time:.2f}s, "
                f"Temps moyen par époque: {avg_epoch_time:.2f}s, "
                f"Temps moyen par batch: {avg_batch_time:.2f}ms, "
                f"Perte finale: {losses[-1]:.6f}"
            )

    return results


def plot_performance_comparison(results):
    """Affiche un graphique comparatif des performances."""
    # Extraire les données pour le graphique
    configs = list(results.keys())
    batch_times = [results[config]["avg_batch_time"] for config in configs]
    total_times = [results[config]["total_time"] for config in configs]

    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique 1: Temps moyen par batch
    ax1.bar(configs, batch_times, color="skyblue")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Temps moyen par batch (ms)")
    ax1.set_title("Comparaison du temps de traitement par batch")
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # Graphique 2: Temps total d'entraînement
    ax2.bar(configs, total_times, color="salmon")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Temps total (s)")
    ax2.set_title("Comparaison du temps total d'entraînement")
    ax2.set_xticklabels(configs, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("rtx_optimization_performance.png")
    logger.info(
        "Graphique de performance enregistré dans 'rtx_optimization_performance.png'"
    )


def find_optimal_batch_size_demo(model, input_shape):
    """Démontre l'utilitaire de recherche de taille de batch optimale."""
    logger.info("\nRecherche de la taille de batch optimale pour RTX 3070...")

    # Trouver la taille de batch optimale
    optimal_batch_size = optimize_batch_size(
        model=model, input_shape=input_shape, max_batch_size=256, start_batch_size=8
    )

    logger.info(f"Taille de batch optimale pour ce modèle: {optimal_batch_size}")
    return optimal_batch_size


def main():
    """Fonction principale pour démontrer les optimisations RTX."""
    logger.info("Démarrage de la démonstration d'optimisation RTX 3070")

    # Vérifier si le GPU est disponible
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA n'est pas disponible. La démonstration fonctionnera sur CPU."
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU détecté: {gpu_name}")

    # Activer les optimisations RTX
    rtx_configured = setup_rtx_optimization(
        enable_tensor_cores=True,
        enable_mixed_precision=True,
        optimize_memory_allocation=True,
    )

    if rtx_configured:
        logger.info("Optimisations RTX configurées avec succès")

    # Générer des données fictives
    seq_length = 20
    n_features = 10
    X_train, y_train = generate_dummy_financial_data(
        n_samples=2000, seq_length=seq_length, n_features=n_features
    )

    logger.info(
        f"Données générées: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
    )

    # Créer un modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFinancialModel(
        input_dim=n_features, hidden_dim=128, output_dim=y_train.shape[1]
    ).to(device)

    # Trouver la taille de batch optimale
    optimal_batch_size = find_optimal_batch_size_demo(model, (seq_length, n_features))

    # Comparer les performances avec différentes configurations
    # Utiliser la taille optimale et une taille non-optimale pour comparaison
    batch_sizes = [optimal_batch_size, 32]
    results = compare_training_performance(
        model=model, X_train=X_train, y_train=y_train, epochs=3, batch_sizes=batch_sizes
    )

    # Afficher les résultats sous forme de graphique
    plot_performance_comparison(results)

    logger.info("\nConclusion de la démonstration")
    logger.info(
        "Les optimisations RTX 3070 permettent d'obtenir des gains significatifs de performance"
    )
    logger.info("Recommandations principales pour RTX 3070:")
    logger.info("1. Utiliser torch.cuda.amp pour la précision mixte")
    logger.info("2. Configurer PYTORCH_CUDA_ALLOC_CONF pour optimiser la mémoire")
    logger.info(
        "3. Activer les Tensor Cores avec torch.backends.cuda.matmul.allow_tf32 = True"
    )
    logger.info("4. Utiliser des tailles de batch multiples de 8 pour les Tensor Cores")


if __name__ == "__main__":
    main()
