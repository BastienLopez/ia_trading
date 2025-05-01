"""
Exemple d'utilisation de l'entraînement en précision mixte (Mixed Precision Training).

Ce script démontre comment utiliser torch.cuda.amp via notre module dédié
pour réduire la consommation mémoire et accélérer l'entraînement.
"""

import os
import sys
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ai_trading.utils.mixed_precision import (
    MixedPrecisionWrapper,
    setup_mixed_precision,
    test_mixed_precision_performance,
    is_mixed_precision_supported
)
from ai_trading.utils.gpu_rtx_optimizer import setup_rtx_optimization

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Créer un répertoire pour les résultats
RESULTS_DIR = Path(__file__).parent / "results" / "mixed_precision"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ConvNet(nn.Module):
    """Un modèle CNN simple pour tester la précision mixte."""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def generate_synthetic_data(num_samples=1000, image_size=32, channels=3, num_classes=10):
    """Génère des données synthétiques pour l'entraînement."""
    X = torch.randn(num_samples, channels, image_size, image_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_model(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    device, 
    epochs=5, 
    use_mixed_precision=False
):
    """
    Entraîne le modèle avec ou sans précision mixte.

    Args:
        model: Le modèle à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        optimizer: L'optimiseur à utiliser
        criterion: La fonction de perte
        device: L'appareil sur lequel effectuer les calculs
        epochs: Nombre d'époques d'entraînement
        use_mixed_precision: Si True, utilise la précision mixte

    Returns:
        Un dictionnaire contenant les métriques d'entraînement
    """
    history = {
        "train_loss": [], 
        "val_loss": [], 
        "train_acc": [], 
        "val_acc": [],
        "time_per_epoch": []
    }
    
    # Initialiser le wrapper de précision mixte si nécessaire
    if use_mixed_precision:
        mp_wrapper = MixedPrecisionWrapper(model, optimizer, enabled=True)
        logger.info("Entraînement avec précision mixte (FP16)")
    else:
        logger.info("Entraînement avec précision standard (FP32)")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()
        
        # Entraînement
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if use_mixed_precision:
                # Utiliser le wrapper pour la précision mixte
                def forward_fn(batch):
                    return model(batch)
                
                def loss_fn(outputs, batch):
                    return criterion(outputs, batch[1])
                
                loss = mp_wrapper.training_step((inputs, targets), forward_fn, loss_fn)
            else:
                # Approche standard
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Temps écoulé pour cette époque
        epoch_time = time.time() - epoch_start_time
        history["time_per_epoch"].append(epoch_time)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Pour la validation, nous n'avons pas besoin de précision mixte
                # car nous ne calculons pas de gradients
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculer les métriques moyennes
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Enregistrer les métriques
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        logger.info(
            f"Époque {epoch+1}/{epochs} - "
            f"Temps: {epoch_time:.2f}s - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Train Acc: {train_acc:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val Acc: {val_acc:.4f}"
        )
    
    return history


def compare_precision_modes(batch_size=64, epochs=5):
    """
    Compare les performances entre la précision standard et la précision mixte.
    
    Args:
        batch_size: Taille du batch à utiliser
        epochs: Nombre d'époques d'entraînement
        
    Returns:
        Tuple contenant les historiques d'entraînement pour les deux modes
    """
    # Vérifier si CUDA est disponible
    if not torch.cuda.is_available():
        logger.warning("CUDA n'est pas disponible, impossible d'utiliser la précision mixte")
        return None, None
    
    # Configurer le périphérique
    device = torch.device("cuda")
    
    # Générer des données synthétiques
    X_train, y_train = generate_synthetic_data(num_samples=5000)
    X_val, y_val = generate_synthetic_data(num_samples=1000)
    
    # Créer les datasets et dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    # Entraînement avec précision standard (FP32)
    logger.info("\n=== Entraînement avec précision standard (FP32) ===")
    model_fp32 = ConvNet().to(device)
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history_fp32 = train_model(
        model=model_fp32,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_fp32,
        criterion=criterion,
        device=device,
        epochs=epochs,
        use_mixed_precision=False
    )
    
    # Entraînement avec précision mixte (FP16)
    logger.info("\n=== Entraînement avec précision mixte (FP16) ===")
    model_fp16 = ConvNet().to(device)
    optimizer_fp16 = optim.Adam(model_fp16.parameters(), lr=0.001)
    
    history_fp16 = train_model(
        model=model_fp16,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_fp16,
        criterion=criterion,
        device=device,
        epochs=epochs,
        use_mixed_precision=True
    )
    
    return history_fp32, history_fp16


def plot_comparison(history_fp32, history_fp16, save_path):
    """
    Génère un graphique comparant les performances des deux modes.
    
    Args:
        history_fp32: Historique d'entraînement avec précision standard
        history_fp16: Historique d'entraînement avec précision mixte
        save_path: Chemin où sauvegarder le graphique
    """
    if history_fp32 is None or history_fp16 is None:
        logger.warning("Impossible de générer le graphique: historiques manquants")
        return
    
    # Créer la figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Perte d'entraînement
    axes[0, 0].plot(history_fp32["train_loss"], label="FP32", marker="o")
    axes[0, 0].plot(history_fp16["train_loss"], label="FP16", marker="s")
    axes[0, 0].set_title("Perte d'entraînement")
    axes[0, 0].set_xlabel("Époque")
    axes[0, 0].set_ylabel("Perte")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Perte de validation
    axes[0, 1].plot(history_fp32["val_loss"], label="FP32", marker="o")
    axes[0, 1].plot(history_fp16["val_loss"], label="FP16", marker="s")
    axes[0, 1].set_title("Perte de validation")
    axes[0, 1].set_xlabel("Époque")
    axes[0, 1].set_ylabel("Perte")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Précision d'entraînement
    axes[1, 0].plot(history_fp32["train_acc"], label="FP32", marker="o")
    axes[1, 0].plot(history_fp16["train_acc"], label="FP16", marker="s")
    axes[1, 0].set_title("Précision d'entraînement")
    axes[1, 0].set_xlabel("Époque")
    axes[1, 0].set_ylabel("Précision")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Comparaison des temps d'entraînement
    total_time_fp32 = sum(history_fp32["time_per_epoch"])
    total_time_fp16 = sum(history_fp16["time_per_epoch"])
    speedup = total_time_fp32 / total_time_fp16 if total_time_fp16 > 0 else 0
    
    bars = axes[1, 1].bar(
        ["FP32", "FP16"], 
        [total_time_fp32, total_time_fp16],
        color=["blue", "orange"]
    )
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f"{height:.2f}s",
            ha="center", 
            va="bottom"
        )
    
    axes[1, 1].set_title(f"Temps total d'entraînement (Accélération: {speedup:.2f}x)")
    axes[1, 1].set_ylabel("Temps (secondes)")
    axes[1, 1].grid(True, axis="y")
    
    # Finaliser et sauvegarder
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Graphique de comparaison sauvegardé dans {save_path}")


def run_performance_test():
    """
    Exécute un test de performance spécifique pour la précision mixte.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA n'est pas disponible, test de performance impossible")
        return
    
    # Créer un modèle pour le test
    model = ConvNet().to(torch.device("cuda"))
    
    # Exécuter le test
    logger.info("\n=== Test de performance de la précision mixte ===")
    results = test_mixed_precision_performance(
        model=model,
        input_shape=(3, 32, 32),  # Format d'image CIFAR (C, H, W)
        batch_size=64,
        iterations=100
    )
    
    # Sauvegarder les résultats
    with open(RESULTS_DIR / "performance_results.txt", "w") as f:
        f.write("Test de performance Mixed Precision\n")
        f.write("=================================\n\n")
        f.write(f"Temps FP32: {results['fp32_time']:.4f}s\n")
        f.write(f"Temps FP16: {results['fp16_time']:.4f}s\n")
        f.write(f"Accélération: {results['speedup']:.2f}x\n\n")
        f.write(f"Mémoire FP32: {results['fp32_memory_mb']:.2f} MB\n")
        f.write(f"Mémoire FP16: {results['fp16_memory_mb']:.2f} MB\n")
        f.write(f"Réduction mémoire: {results['memory_reduction']:.2f}x\n")
    
    # Créer un graphique de comparaison
    labels = ["FP32", "FP16"]
    times = [results["fp32_time"], results["fp16_time"]]
    memories = [results["fp32_memory_mb"], results["fp16_memory_mb"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Graphique des temps
    bars1 = ax1.bar(labels, times, color=["blue", "orange"])
    ax1.set_title(f"Temps d'exécution (Accélération: {results['speedup']:.2f}x)")
    ax1.set_ylabel("Temps (secondes)")
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f"{height:.4f}s",
            ha="center", 
            va="bottom"
        )
    
    # Graphique des mémoires
    bars2 = ax2.bar(labels, memories, color=["blue", "orange"])
    ax2.set_title(f"Utilisation mémoire (Réduction: {results['memory_reduction']:.2f}x)")
    ax2.set_ylabel("Mémoire (MB)")
    
    # Ajouter les valeurs sur les barres
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f"{height:.2f} MB",
            ha="center", 
            va="bottom"
        )
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "performance_comparison.png")
    logger.info(f"Graphique de performance sauvegardé dans {RESULTS_DIR / 'performance_comparison.png'}")


def main():
    """Fonction principale pour exécuter les démonstrations."""
    logger.info("Démarrage de l'exemple de précision mixte")
    
    # Vérifier si la précision mixte est supportée
    if is_mixed_precision_supported():
        logger.info("La précision mixte est supportée sur ce matériel")
        # Configurer la précision mixte
        setup_mixed_precision()
    else:
        logger.warning("La précision mixte n'est pas supportée sur ce matériel")
        logger.warning("L'exemple s'exécutera en mode standard (FP32)")
    
    # Appliquer les optimisations RTX si disponible
    if torch.cuda.is_available():
        setup_rtx_optimization()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Utilisation de CUDA: {device_name}")
    else:
        logger.warning("CUDA n'est pas disponible, utilisation du CPU")
    
    # Exécuter le test de performance
    run_performance_test()
    
    # Comparer les modes de précision
    logger.info("\nComparaison des modes de précision...")
    history_fp32, history_fp16 = compare_precision_modes(batch_size=64, epochs=3)
    
    # Générer le graphique de comparaison
    if history_fp32 and history_fp16:
        plot_comparison(
            history_fp32, 
            history_fp16, 
            save_path=RESULTS_DIR / "training_comparison.png"
        )
    
    logger.info("\nExample de précision mixte terminé!")
    logger.info(f"Voir les résultats dans: {RESULTS_DIR}")


if __name__ == "__main__":
    main() 