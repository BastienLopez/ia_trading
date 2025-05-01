"""
Exemple d'utilisation de l'Activation Checkpointing pour économiser la mémoire VRAM.

Ce script démontre comment utiliser l'Activation Checkpointing pour réduire la
consommation mémoire pendant l'entraînement de modèles profonds, au prix d'un
léger surcoût computationnel.
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
import matplotlib.pyplot as plt
import argparse

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.utils.activation_checkpointing import (
    apply_activation_checkpointing,
    CheckpointedModule,
    analyze_memory_usage,
    analyze_checkpointing_savings,
    ActivationCheckpointingModifier
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Créer un répertoire pour les résultats
RESULTS_DIR = Path(__file__).parent / "results" / "activation_checkpointing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class DeepTradingModel(nn.Module):
    """
    Un modèle de trading avec plusieurs couches profondes.
    
    Ce modèle est intentionnellement profond pour démontrer les avantages
    de l'Activation Checkpointing.
    """
    
    def __init__(self, input_dim=64, hidden_dim=256, num_blocks=8, use_checkpointing=False):
        """
        Initialise le modèle.
        
        Args:
            input_dim: Dimension d'entrée
            hidden_dim: Dimension des couches cachées
            num_blocks: Nombre de blocs résiduels
            use_checkpointing: Si True, applique l'Activation Checkpointing
        """
        super().__init__()
        
        # Couche d'entrée
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Blocs résiduels
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ResidualBlock(hidden_dim, hidden_dim // 2)
            if use_checkpointing:
                block = CheckpointedModule(block)
            self.blocks.append(block)
        
        # Couche de sortie
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        x = self.input_norm(x)
        
        # Passer à travers les blocs résiduels
        for block in self.blocks:
            x = block(x)
        
        # Couche de sortie
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """
    Bloc résiduel avec plusieurs couches linéaires.
    
    Ce bloc contient des couches qui consomment beaucoup de mémoire
    pour les activations intermédiaires.
    """
    
    def __init__(self, dim, hidden_dim):
        """
        Initialise le bloc résiduel.
        
        Args:
            dim: Dimension d'entrée/sortie
            hidden_dim: Dimension cachée
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        """Forward pass avec connexion résiduelle."""
        identity = x
        out = self.layers(x)
        return out + identity


def generate_trading_data(num_samples=1000, input_dim=64, batch_size=32):
    """
    Génère des données synthétiques pour démontrer le modèle.
    
    Args:
        num_samples: Nombre d'échantillons
        input_dim: Dimension des caractéristiques d'entrée
        batch_size: Taille des batchs
        
    Returns:
        Dataloader d'entraînement, Dataloader de validation
    """
    # Générer des données aléatoires
    X = torch.randn(num_samples, input_dim)
    
    # Créer un pattern simple pour les labels
    # Les premiers éléments de chaque échantillon déterminent le label
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).float().unsqueeze(1)
    
    # Ajouter du bruit
    y = y + 0.05 * torch.randn_like(y)
    y = torch.clamp(y, 0.0, 1.0)
    
    # Diviser en train/val
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Créer les datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Entraîne le modèle pendant une époque.
    
    Args:
        model: Le modèle à entraîner
        dataloader: DataLoader pour les données d'entraînement
        optimizer: L'optimiseur
        criterion: La fonction de perte
        device: Périphérique (CPU/GPU)
        
    Returns:
        Perte moyenne sur l'époque
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """
    Valide le modèle.
    
    Args:
        model: Le modèle à valider
        dataloader: DataLoader pour les données de validation
        criterion: La fonction de perte
        device: Périphérique (CPU/GPU)
        
    Returns:
        Perte moyenne, précision moyenne
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Calculer les métriques
            predictions = (outputs > 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def benchmark_training_memory(model_with_checkpointing, model_without_checkpointing, train_loader, val_loader, num_epochs=2):
    """
    Compare les performances d'entraînement avec et sans checkpointing.
    
    Args:
        model_with_checkpointing: Modèle avec checkpointing
        model_without_checkpointing: Modèle sans checkpointing
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        num_epochs: Nombre d'époques d'entraînement
        
    Returns:
        Dictionnaire avec les résultats du benchmark
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Placer les modèles sur le bon périphérique
    model_with_checkpointing = model_with_checkpointing.to(device)
    model_without_checkpointing = model_without_checkpointing.to(device)
    
    # Préparer les optimiseurs et le critère
    optimizer_with = optim.Adam(model_with_checkpointing.parameters(), lr=0.001)
    optimizer_without = optim.Adam(model_without_checkpointing.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Analyser la mémoire des modèles
    logger.info("Analyse de la mémoire pour le modèle avec checkpointing...")
    sample_batch = next(iter(train_loader))[0]
    memory_with = analyze_memory_usage(model_with_checkpointing, sample_batch.shape[1:], batch_size=train_loader.batch_size)
    
    logger.info("Analyse de la mémoire pour le modèle sans checkpointing...")
    memory_without = analyze_memory_usage(model_without_checkpointing, sample_batch.shape[1:], batch_size=train_loader.batch_size)
    
    # Mesurer le temps d'entraînement
    logger.info("\n=== Benchmark des performances d'entraînement ===")
    
    # Entraîner le modèle avec checkpointing
    torch.cuda.empty_cache()
    start_time = time.time()
    train_losses_with = []
    val_losses_with = []
    accuracies_with = []
    peak_memory_with = []
    
    logger.info("Entraînement du modèle AVEC checkpointing...")
    for epoch in range(num_epochs):
        # Mesurer la mémoire maximale
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Entraîner
        train_loss = train_epoch(model_with_checkpointing, train_loader, optimizer_with, criterion, device)
        val_loss, accuracy = validate(model_with_checkpointing, val_loader, criterion, device)
        
        # Enregistrer les métriques
        train_losses_with.append(train_loss)
        val_losses_with.append(val_loss)
        accuracies_with.append(accuracy)
        
        # Enregistrer la mémoire maximale
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            peak_memory_with.append(peak_mem)
            logger.info(f"Époque {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Accuracy: {accuracy:.4f}, Mémoire max: {peak_mem:.2f} MB")
        else:
            logger.info(f"Époque {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Accuracy: {accuracy:.4f}")
    
    time_with = time.time() - start_time
    logger.info(f"Temps d'entraînement avec checkpointing: {time_with:.2f}s")
    
    # Entraîner le modèle sans checkpointing
    try:
        torch.cuda.empty_cache()
        start_time = time.time()
        train_losses_without = []
        val_losses_without = []
        accuracies_without = []
        peak_memory_without = []
        
        logger.info("\nEntraînement du modèle SANS checkpointing...")
        for epoch in range(num_epochs):
            # Mesurer la mémoire maximale
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Entraîner
            train_loss = train_epoch(model_without_checkpointing, train_loader, optimizer_without, criterion, device)
            val_loss, accuracy = validate(model_without_checkpointing, val_loader, criterion, device)
            
            # Enregistrer les métriques
            train_losses_without.append(train_loss)
            val_losses_without.append(val_loss)
            accuracies_without.append(accuracy)
            
            # Enregistrer la mémoire maximale
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                peak_memory_without.append(peak_mem)
                logger.info(f"Époque {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Accuracy: {accuracy:.4f}, Mémoire max: {peak_mem:.2f} MB")
            else:
                logger.info(f"Époque {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Accuracy: {accuracy:.4f}")
        
        time_without = time.time() - start_time
        logger.info(f"Temps d'entraînement sans checkpointing: {time_without:.2f}s")
        
        memory_reduction = None
        if torch.cuda.is_available() and peak_memory_without and peak_memory_with:
            memory_reduction = (max(peak_memory_without) - max(peak_memory_with)) / max(peak_memory_without) * 100
            logger.info(f"Réduction de mémoire: {memory_reduction:.2f}%")
        
        speed_overhead = ((time_with - time_without) / time_without) * 100
        logger.info(f"Surcoût de calcul: {speed_overhead:.2f}%")
        
        oom_without = False
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("OOM ERROR: Entraînement sans checkpointing impossible - CUDA out of memory!")
            oom_without = True
            train_losses_without = None
            val_losses_without = None
            accuracies_without = None
            peak_memory_without = None
            time_without = None
            memory_reduction = 100  # Considéré comme 100% car le modèle ne tient pas en mémoire
            speed_overhead = None
        else:
            raise e
    
    # Compiler les résultats
    benchmark_results = {
        "memory_with_checkpointing_mb": memory_with["total_memory_mb"],
        "memory_without_checkpointing_mb": memory_without["total_memory_mb"],
        "memory_reduction_percent": memory_reduction,
        "training_time_with_checkpointing_s": time_with,
        "training_time_without_checkpointing_s": time_without,
        "speed_overhead_percent": speed_overhead,
        "train_losses_with": train_losses_with,
        "train_losses_without": train_losses_without,
        "val_losses_with": val_losses_with,
        "val_losses_without": val_losses_without,
        "accuracies_with": accuracies_with,
        "accuracies_without": accuracies_without,
        "peak_memory_with_mb": peak_memory_with,
        "peak_memory_without_mb": peak_memory_without,
        "out_of_memory_without_checkpointing": oom_without
    }
    
    # Sauvegarder les résultats
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
    with open(results_file, 'w') as f:
        f.write("=== Benchmark de l'Activation Checkpointing ===\n\n")
        
        f.write("Consommation mémoire statique:\n")
        f.write(f"  Avec checkpointing: {results['memory_with_checkpointing_mb']:.2f} MB\n")
        f.write(f"  Sans checkpointing: {results['memory_without_checkpointing_mb']:.2f} MB\n")
        
        if results['memory_reduction_percent'] is not None:
            f.write(f"  Réduction de mémoire: {results['memory_reduction_percent']:.2f}%\n")
        
        f.write("\nTemps d'entraînement:\n")
        f.write(f"  Avec checkpointing: {results['training_time_with_checkpointing_s']:.2f}s\n")
        
        if results['training_time_without_checkpointing_s'] is not None:
            f.write(f"  Sans checkpointing: {results['training_time_without_checkpointing_s']:.2f}s\n")
            
            if results['speed_overhead_percent'] is not None:
                f.write(f"  Surcoût de calcul: {results['speed_overhead_percent']:.2f}%\n")
        else:
            f.write("  Sans checkpointing: ÉCHEC (Out of Memory)\n")
        
        # Ajouter des métriques détaillées si disponibles
        if results['peak_memory_with_mb']:
            f.write("\nMémoire maximale par époque (MB):\n")
            f.write("  Avec checkpointing: " + 
                   ", ".join([f"{mem:.2f}" for mem in results['peak_memory_with_mb']]) + "\n")
            
            if results['peak_memory_without_mb']:
                f.write("  Sans checkpointing: " + 
                       ", ".join([f"{mem:.2f}" for mem in results['peak_memory_without_mb']]) + "\n")
    
    # Créer des graphiques si possible
    try:
        # Graphique de perte
        if results['train_losses_with'] and results['val_losses_with']:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(results['train_losses_with']) + 1)
            
            plt.plot(epochs, results['train_losses_with'], 'b-', label='Train avec checkpointing')
            plt.plot(epochs, results['val_losses_with'], 'b--', label='Val avec checkpointing')
            
            if results['train_losses_without'] and results['val_losses_without']:
                plt.plot(epochs, results['train_losses_without'], 'r-', label='Train sans checkpointing')
                plt.plot(epochs, results['val_losses_without'], 'r--', label='Val sans checkpointing')
            
            plt.xlabel('Époque')
            plt.ylabel('Perte')
            plt.title('Comparaison des pertes avec et sans Activation Checkpointing')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(RESULTS_DIR / "loss_comparison.png")
            
            logger.info(f"Graphique de perte sauvegardé: {RESULTS_DIR / 'loss_comparison.png'}")
        
        # Graphique de mémoire
        if results['peak_memory_with_mb']:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(results['peak_memory_with_mb']) + 1)
            
            plt.plot(epochs, results['peak_memory_with_mb'], 'b-', label='Avec checkpointing')
            
            if results['peak_memory_without_mb']:
                plt.plot(epochs, results['peak_memory_without_mb'], 'r-', label='Sans checkpointing')
            
            plt.xlabel('Époque')
            plt.ylabel('Mémoire VRAM maximale (MB)')
            plt.title('Consommation mémoire avec et sans Activation Checkpointing')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(RESULTS_DIR / "memory_comparison.png")
            
            logger.info(f"Graphique de mémoire sauvegardé: {RESULTS_DIR / 'memory_comparison.png'}")
    except Exception as e:
        logger.warning(f"Erreur lors de la création des graphiques: {e}")


def main(args):
    """
    Fonction principale pour exécuter la démonstration.
    
    Args:
        args: Arguments de la ligne de commande
    """
    logger.info("Démarrage de l'exemple d'Activation Checkpointing")
    
    # Vérifier CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du périphérique: {device}")
    
    # Paramètres du modèle
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    num_blocks = args.num_blocks
    
    # Créer les modèles
    logger.info(f"Création des modèles avec {num_blocks} blocs et dimension cachée {hidden_dim}...")
    model_with_checkpointing = DeepTradingModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        use_checkpointing=True
    )
    
    model_without_checkpointing = DeepTradingModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        use_checkpointing=False
    )
    
    # Générer les données
    logger.info("Génération des données synthétiques...")
    train_loader, val_loader = generate_trading_data(
        num_samples=args.num_samples,
        input_dim=input_dim,
        batch_size=args.batch_size
    )
    
    # Exécuter le benchmark
    logger.info("Exécution du benchmark d'entraînement...")
    results = benchmark_training_memory(
        model_with_checkpointing,
        model_without_checkpointing,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs
    )
    
    # Résumé final
    logger.info("\n=== Résumé ===")
    if results["out_of_memory_without_checkpointing"]:
        logger.info("Le modèle sans checkpointing ne peut pas être entraîné en raison de contraintes de mémoire.")
        logger.info("L'Activation Checkpointing a permis d'entraîner le modèle avec la VRAM disponible.")
    else:
        logger.info(f"Réduction de mémoire: {results['memory_reduction_percent']:.2f}%")
        logger.info(f"Surcoût de calcul: {results['speed_overhead_percent']:.2f}%")
    
    logger.info(f"\nExemple d'Activation Checkpointing terminé! Voir les résultats dans: {RESULTS_DIR}")


if __name__ == "__main__":
    # Analyser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Exemple d'Activation Checkpointing")
    parser.add_argument("--input_dim", type=int, default=64, 
                        help="Dimension d'entrée")
    parser.add_argument("--hidden_dim", type=int, default=256, 
                        help="Dimension cachée des couches")
    parser.add_argument("--num_blocks", type=int, default=8, 
                        help="Nombre de blocs résiduels")
    parser.add_argument("--num_samples", type=int, default=5000, 
                        help="Nombre d'échantillons pour les données synthétiques")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Taille du batch")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Nombre d'époques d'entraînement")
    
    args = parser.parse_args()
    
    main(args) 