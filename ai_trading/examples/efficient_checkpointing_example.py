"""
Exemple d'utilisation de l'Efficient Checkpointing pour les modèles de trading.

Ce script démontre comment utiliser les techniques d'Efficient Checkpointing pour
optimiser l'espace disque et la vitesse d'entraînement lors de la sauvegarde des modèles.
"""

import os
import sys
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.utils.efficient_checkpointing import (
    save_model_weights_only,
    AsyncCheckpointSaver,
    CheckpointManager,
    compare_checkpoint_sizes
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Créer un répertoire pour les résultats
RESULTS_DIR = Path(__file__).parent / "results" / "efficient_checkpointing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class TradingModel(nn.Module):
    """Modèle de réseau de neurones pour la prédiction de trading."""
    
    def __init__(self, input_dim=50, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Couche d'entrée
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        # Couches cachées
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Couche de sortie (prédiction de mouvement de prix)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def generate_synthetic_trading_data(num_samples=10000, input_dim=50):
    """Génère des données synthétiques pour simuler des features de trading."""
    X = torch.randn(num_samples, input_dim)  # Features de marché simulées
    
    # Créer une tendance synthétique basée sur certaines features
    weights = torch.randn(input_dim)
    y = torch.matmul(X, weights.unsqueeze(1)) + 0.1 * torch.randn(num_samples, 1)
    
    # Convertir en labels binaires (hausse/baisse)
    y_binary = (y > 0).float()
    
    return X, y_binary


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Entraîne le modèle pendant une époque."""
    model.train()
    total_loss = 0.0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Valide le modèle sur un ensemble de données."""
    model.eval()
    total_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * X.size(0)
            
            # Calculer la précision (classification binaire)
            predictions = (outputs > 0.5).float()
            correct += (predictions == y).sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    
    return avg_loss, accuracy


def benchmark_checkpoint_methods(model, optimizer, data_size_mb):
    """Compare les performances des différentes méthodes de checkpoint."""
    logger.info("\n=== Benchmark des méthodes de checkpoint ===")
    
    # Préparer les répertoires
    benchmark_dir = RESULTS_DIR / "benchmark"
    benchmark_dir.mkdir(exist_ok=True)
    
    # Méthode 1: Sauvegarde standard (modèle + optimiseur)
    start_time = time.time()
    full_path = benchmark_dir / "full_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, full_path)
    full_time = time.time() - start_time
    full_size = os.path.getsize(full_path) / (1024 * 1024)
    
    # Méthode 2: Sauvegarde des poids uniquement
    start_time = time.time()
    weights_path = benchmark_dir / "weights_only.pt"
    save_model_weights_only(model, weights_path)
    weights_time = time.time() - start_time
    weights_size = os.path.getsize(weights_path) / (1024 * 1024)
    
    # Méthode 3: Sauvegarde asynchrone
    async_saver = AsyncCheckpointSaver()
    start_time = time.time()
    async_path = benchmark_dir / "async_checkpoint.pt"
    async_saver.save_checkpoint(model, async_path, optimizer=optimizer, weights_only=False)
    async_saver.wait_for_completion()
    async_time = time.time() - start_time
    async_size = os.path.getsize(async_path) / (1024 * 1024)
    
    # Afficher les résultats
    logger.info("\n=== Résultats du benchmark ===")
    logger.info(f"Taille des données d'entraînement: {data_size_mb:.2f} MB")
    logger.info(f"Méthode standard    : {full_time:.4f}s, {full_size:.2f} MB")
    logger.info(f"Poids uniquement    : {weights_time:.4f}s, {weights_size:.2f} MB " +
                f"({weights_size/full_size*100:.1f}% de la taille complète)")
    logger.info(f"Async + poids complet: {async_time:.4f}s, {async_size:.2f} MB")
    
    # Calculer les économies potentielles pour un entraînement complet
    num_epochs = 100
    checkpoints_per_epoch = 1
    
    standard_total = full_size * num_epochs * checkpoints_per_epoch
    optimized_total = weights_size * num_epochs * checkpoints_per_epoch
    
    logger.info("\n=== Projections pour un entraînement complet ===")
    logger.info(f"Estimation pour {num_epochs} époques avec {checkpoints_per_epoch} checkpoints/époque:")
    logger.info(f"Taille standard      : {standard_total:.2f} MB")
    logger.info(f"Taille optimisée     : {optimized_total:.2f} MB")
    logger.info(f"Espace économisé     : {standard_total - optimized_total:.2f} MB " +
                f"({(1 - optimized_total/standard_total) * 100:.1f}%)")
    
    with open(RESULTS_DIR / "benchmark_results.txt", "w") as f:
        f.write("=== Benchmark des méthodes de checkpoint ===\n")
        f.write(f"Taille des données d'entraînement: {data_size_mb:.2f} MB\n")
        f.write(f"Méthode standard    : {full_time:.4f}s, {full_size:.2f} MB\n")
        f.write(f"Poids uniquement    : {weights_time:.4f}s, {weights_size:.2f} MB " +
                f"({weights_size/full_size*100:.1f}% de la taille complète)\n")
        f.write(f"Async + poids complet: {async_time:.4f}s, {async_size:.2f} MB\n\n")
        
        f.write("=== Projections pour un entraînement complet ===\n")
        f.write(f"Estimation pour {num_epochs} époques avec {checkpoints_per_epoch} checkpoints/époque:\n")
        f.write(f"Taille standard      : {standard_total:.2f} MB\n")
        f.write(f"Taille optimisée     : {optimized_total:.2f} MB\n")
        f.write(f"Espace économisé     : {standard_total - optimized_total:.2f} MB " +
                f"({(1 - optimized_total/standard_total) * 100:.1f}%)\n")
    
    return {
        "full_time": full_time,
        "full_size": full_size,
        "weights_time": weights_time,
        "weights_size": weights_size,
        "async_time": async_time,
        "async_size": async_size
    }


def train_with_efficient_checkpointing(
    model, train_loader, val_loader, optimizer, criterion, device, 
    num_epochs=10, checkpoint_interval=2
):
    """
    Entraîne le modèle en utilisant l'Efficient Checkpointing.
    
    Args:
        model: Le modèle à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        optimizer: L'optimiseur
        criterion: La fonction de perte
        device: Périphérique d'entraînement (CPU/GPU)
        num_epochs: Nombre d'époques d'entraînement
        checkpoint_interval: Nombre d'époques entre les sauvegardes
    """
    # Initialiser le gestionnaire de checkpoints
    checkpoint_dir = RESULTS_DIR / "training_checkpoints"
    checkpoint_mgr = CheckpointManager(
        base_dir=checkpoint_dir,
        max_checkpoints=5,  # Conserver uniquement les 5 derniers checkpoints
        async_save=True,    # Utiliser des sauvegardes asynchrones
        weights_only=True,  # Sauvegarder uniquement les poids
        save_format="model_epoch_{epoch:03d}.pt"
    )
    
    # Journal d'entraînement
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
        "checkpoint_times": []
    }
    
    logger.info(f"\n=== Début de l'entraînement avec Efficient Checkpointing ===")
    logger.info(f"Checkpoints sauvegardés dans: {checkpoint_dir}")
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Entraîner une époque
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Valider
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Mesurer le temps d'époque
        epoch_time = time.time() - epoch_start
        
        # Sauvegarder le checkpoint si nécessaire
        checkpoint_time = 0
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            checkpoint_start = time.time()
            
            # Sauvegarder le modèle avec des métadonnées
            metadata = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "timestamp": time.time()
            }
            
            checkpoint_mgr.save(model, optimizer=optimizer, epoch=epoch+1, extra_data=metadata)
            
            checkpoint_time = time.time() - checkpoint_start
        
        # Enregistrer les métriques
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_accuracy"].append(val_acc)
        training_history["epoch_times"].append(epoch_time)
        training_history["checkpoint_times"].append(checkpoint_time)
        
        # Afficher les métriques
        logger.info(f"Époque {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.4f}, "
                   f"Temps: {epoch_time:.2f}s "
                   f"(+ Checkpoint: {checkpoint_time:.2f}s)")
    
    # Attendre la fin des sauvegardes asynchrones
    if checkpoint_mgr.async_save and checkpoint_mgr.async_saver:
        logger.info("Attente de la fin des sauvegardes asynchrones...")
        checkpoint_mgr.async_saver.wait_for_completion()
    
    # Sauvegarder l'historique d'entraînement
    with open(RESULTS_DIR / "training_history.txt", "w") as f:
        f.write("=== Historique d'entraînement ===\n")
        f.write(f"{'Époque':<10}{'Train Loss':<15}{'Val Loss':<15}{'Val Acc':<15}"
                f"{'Temps (s)':<15}{'Checkpoint (s)':<15}\n")
        
        for i in range(num_epochs):
            f.write(f"{i+1:<10}{training_history['train_loss'][i]:<15.4f}"
                    f"{training_history['val_loss'][i]:<15.4f}"
                    f"{training_history['val_accuracy'][i]:<15.4f}"
                    f"{training_history['epoch_times'][i]:<15.2f}"
                    f"{training_history['checkpoint_times'][i]:<15.2f}\n")
    
    logger.info(f"\n=== Entraînement terminé ===")
    logger.info(f"Historique d'entraînement sauvegardé dans: {RESULTS_DIR / 'training_history.txt'}")
    
    return training_history, checkpoint_mgr


def restore_and_test_model(checkpoint_mgr, model, test_loader, criterion, device):
    """
    Restaure et teste un modèle à partir du dernier checkpoint.
    
    Args:
        checkpoint_mgr: Le gestionnaire de checkpoints
        model: Le modèle à restaurer
        test_loader: DataLoader pour les données de test
        criterion: La fonction de perte
        device: Périphérique d'entraînement (CPU/GPU)
    """
    logger.info("\n=== Test de restauration du modèle ===")
    
    # Charger le dernier checkpoint
    metadata = checkpoint_mgr.load_latest(model, optimizer=None, device=device)
    
    if not metadata:
        logger.error("Échec de la restauration du modèle")
        return
    
    # Tester le modèle
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    logger.info(f"Modèle restauré de l'époque {metadata.get('epoch', 'inconnue')}")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    with open(RESULTS_DIR / "restoration_test.txt", "w") as f:
        f.write("=== Test de restauration du modèle ===\n")
        f.write(f"Modèle restauré de l'époque {metadata.get('epoch', 'inconnue')}\n")
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
    
    return test_loss, test_acc


def main():
    """Fonction principale pour exécuter les démonstrations."""
    logger.info("Démarrage de l'exemple d'Efficient Checkpointing")
    
    # Vérifier CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du dispositif: {device}")
    
    # Paramètres
    input_dim = 50
    batch_size = 64
    learning_rate = 0.001
    
    # Générer des données synthétiques
    logger.info("Génération des données synthétiques...")
    X, y = generate_synthetic_trading_data(num_samples=10000, input_dim=input_dim)
    
    # Diviser en ensembles d'entraînement, validation et test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    test_size = len(X) - train_size - val_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Créer les DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Estimer la taille des données
    data_size_bytes = (X.element_size() * X.nelement() + 
                       y.element_size() * y.nelement())
    data_size_mb = data_size_bytes / (1024 * 1024)
    
    # Créer le modèle et l'optimiseur
    model = TradingModel(input_dim=input_dim, hidden_dim=128, num_layers=3)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Benchmark des méthodes de checkpoint
    logger.info("Exécution du benchmark des méthodes de checkpoint...")
    benchmark_results = benchmark_checkpoint_methods(model, optimizer, data_size_mb)
    
    # Entraîner avec Efficient Checkpointing
    logger.info("Démarrage de l'entraînement avec Efficient Checkpointing...")
    training_history, checkpoint_mgr = train_with_efficient_checkpointing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=10,
        checkpoint_interval=2
    )
    
    # Restaurer et tester le modèle
    logger.info("Test de restauration du modèle...")
    test_results = restore_and_test_model(
        checkpoint_mgr=checkpoint_mgr,
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    logger.info(f"\nExemple d'Efficient Checkpointing terminé!")
    logger.info(f"Voir les résultats dans: {RESULTS_DIR}")


if __name__ == "__main__":
    main() 