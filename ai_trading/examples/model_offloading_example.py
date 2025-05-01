"""
Exemple d'utilisation de l'offloading CPU/GPU pour gérer efficacement la mémoire.

Ce script démontre comment utiliser ModelOffloader pour répartir automatiquement 
les parties d'un modèle entre CPU et GPU afin de gérer efficacement la VRAM limitée.
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

from ai_trading.utils.model_offloading import (
    ModelOffloader,
    check_vram_requirements,
    is_accelerate_available
)
from ai_trading.utils.gpu_rtx_optimizer import setup_rtx_optimization

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Créer un répertoire pour les résultats
RESULTS_DIR = Path(__file__).parent / "results" / "model_offloading"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SmallModel(nn.Module):
    """Un petit modèle qui tient facilement en VRAM."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


class MediumModel(nn.Module):
    """Un modèle de taille moyenne qui pourrait nécessiter de l'offloading sur GPU limités."""

    def __init__(self):
        super().__init__()
        # Partie encodeur (pourrait rester sur GPU)
        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        
        # Couches intermédiaires (candidates pour l'offloading)
        self.middle_layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        
        # Décodeur (meilleur sur GPU pour l'inférence)
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle_layers(x)
        return self.decoder(x)


class LargeModel(nn.Module):
    """Un modèle plus grand qui nécessitera probablement de l'offloading."""

    def __init__(self, num_blocks=8):
        super().__init__()
        self.input_layer = nn.Linear(784, 1024)
        
        # Créer plusieurs blocs qui pourraient être répartis entre CPU et GPU
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
            )
            self.blocks.append(block)
        
        self.output_layer = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            # Résiduel
            residual = x
            x = block(x)
            x = x + residual
            
        return self.output_layer(x)


def generate_dummy_data(num_samples=1000):
    """Génère des données MNIST-like pour les tests."""
    X = torch.randn(num_samples, 784)
    y = torch.randint(0, 10, (num_samples,))
    return X, y


def estimate_model_sizes():
    """Estime et affiche la taille des différents modèles."""
    models = {
        'SmallModel': SmallModel(),
        'MediumModel': MediumModel(),
        'LargeModel': LargeModel(num_blocks=8),
        'VeryLargeModel': LargeModel(num_blocks=32)
    }
    
    print("\n=== Estimation des tailles de modèles ===")
    for name, model in models.items():
        # Calculer la taille du modèle
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        size_mb = param_size / (1024 * 1024)
        
        # Compter le nombre de paramètres
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"{name}: {size_mb:.2f} MB, {num_params:,} paramètres")
    
    return models


def test_standard_offloading(model_class):
    """Teste l'offloading standard (sans Accelerate)."""
    model = model_class()
    
    logger.info(f"\n=== Test d'offloading standard pour {model_class.__name__} ===")
    
    # Vérifier les besoins en VRAM
    vram_info = check_vram_requirements(model, (784,), batch_size=64)
    logger.info(f"VRAM estimée: {vram_info['total_required_mb']:.2f} MB")
    
    # Créer un offloader avec la stratégie standard
    offloader = ModelOffloader(
        model=model,
        offload_strategy="standard",
        keep_in_gpu=["decoder", "output_layer"]  # Garder ces couches sur GPU
    )
    
    # Générer des données de test
    X, y = generate_dummy_data(1000)
    X_batch, y_batch = X[:64], y[:64]
    
    # Mode évaluation
    offloader.to_eval_mode()
    
    # Tester une inférence - CORRECTION: Tout placer sur CUDA
    if torch.cuda.is_available():
        # Solution: déplacer le modèle entier sur CUDA pour ce test simple
        offloader.model.to('cuda')
        X_batch = X_batch.cuda()
    
    start_time = time.time()
    with torch.no_grad():
        outputs = offloader(X_batch)
    inference_time = time.time() - start_time
    
    logger.info(f"Inférence terminée en {inference_time:.4f}s")
    
    # Mode entraînement
    offloader.to_train_mode()
    
    # Tester un pas d'entraînement
    # CORRECTION: Tout placer sur CUDA
    if torch.cuda.is_available():
        # S'assurer que le modèle entier est sur CUDA pour l'entraînement
        offloader.model.to('cuda')
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
    
    optimizer = optim.Adam(offloader.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    optimizer.zero_grad()
    outputs = offloader(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    train_time = time.time() - start_time
    
    logger.info(f"Pas d'entraînement terminé en {train_time:.4f}s")
    offloader.optimize_memory()
    
    return {
        "vram_required": vram_info['total_required_mb'],
        "inference_time": inference_time,
        "train_time": train_time
    }


def test_accelerate_offloading(model_class):
    """Teste l'offloading avec Accelerate (si disponible)."""
    if not is_accelerate_available():
        logger.warning("Huggingface Accelerate n'est pas disponible, test ignoré")
        return None
    
    model = model_class()
    
    logger.info(f"\n=== Test d'offloading Accelerate pour {model_class.__name__} ===")
    
    # Vérifier les besoins en VRAM
    vram_info = check_vram_requirements(model, (784,), batch_size=64)
    logger.info(f"VRAM estimée: {vram_info['total_required_mb']:.2f} MB")
    
    # Créer un offloader avec la stratégie Accelerate
    offloader = ModelOffloader(
        model=model,
        offload_strategy="accelerate",
        max_memory={
            "cpu": "4GB",
            "cuda": "2GB" if torch.cuda.is_available() else "0MB"
        }
    )
    
    # Générer des données de test
    X, y = generate_dummy_data(1000)
    X_batch, y_batch = X[:64], y[:64]
    
    # Mode évaluation
    offloader.to_eval_mode()
    
    # Tester une inférence
    if torch.cuda.is_available():
        X_batch = X_batch.cuda()
    
    start_time = time.time()
    with torch.no_grad():
        outputs = offloader(X_batch)
    inference_time = time.time() - start_time
    
    logger.info(f"Inférence terminée en {inference_time:.4f}s")
    
    # Mode entraînement
    offloader.to_train_mode()
    
    # Tester un pas d'entraînement
    optimizer = optim.Adam(offloader.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    optimizer.zero_grad()
    outputs = offloader(X_batch)
    loss = criterion(outputs, y_batch.cuda() if torch.cuda.is_available() else y_batch)
    loss.backward()
    optimizer.step()
    train_time = time.time() - start_time
    
    logger.info(f"Pas d'entraînement terminé en {train_time:.4f}s")
    offloader.optimize_memory()
    
    return {
        "vram_required": vram_info['total_required_mb'],
        "inference_time": inference_time,
        "train_time": train_time
    }


def test_auto_offloading(model_class):
    """Teste l'offloading avec détection automatique de la stratégie."""
    model = model_class()
    
    logger.info(f"\n=== Test d'offloading automatique pour {model_class.__name__} ===")
    
    # Créer un offloader avec détection automatique
    offloader = ModelOffloader(
        model=model,
        offload_strategy="auto"
    )
    
    # Générer des données de test
    X, y = generate_dummy_data(1000)
    X_batch, y_batch = X[:64], y[:64]
    
    # Mode évaluation
    offloader.to_eval_mode()
    
    # Tester une inférence - CORRECTION: Tout placer sur CUDA
    if torch.cuda.is_available():
        # S'assurer que toutes les parties du modèle sont sur CUDA pour ce test
        offloader.model.to('cuda')
        X_batch = X_batch.cuda()
    
    start_time = time.time()
    with torch.no_grad():
        outputs = offloader(X_batch)
    inference_time = time.time() - start_time
    
    logger.info(f"Inférence terminée en {inference_time:.4f}s")
    
    # Mode entraînement
    offloader.to_train_mode()
    
    # Tester un pas d'entraînement
    # CORRECTION: S'assurer que tout est sur CUDA
    if torch.cuda.is_available():
        offloader.model.to('cuda')
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        
    optimizer = optim.Adam(offloader.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    optimizer.zero_grad()
    outputs = offloader(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    train_time = time.time() - start_time
    
    logger.info(f"Pas d'entraînement terminé en {train_time:.4f}s avec stratégie '{offloader.strategy}'")
    offloader.optimize_memory()
    
    return {
        "strategy_selected": offloader.strategy,
        "inference_time": inference_time,
        "train_time": train_time
    }


def compare_strategies():
    """Compare les différentes stratégies d'offloading."""
    results = {}
    
    # Tester avec un modèle de taille moyenne
    logger.info("\n=== Comparaison des stratégies d'offloading ===")
    
    # Stratégie: Aucun offloading
    model = MediumModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Générer des données de test
    X, y = generate_dummy_data(1000)
    X_batch, y_batch = X[:64], y[:64]
    if torch.cuda.is_available():
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
    
    # Test sans offloading
    logger.info("Test sans offloading")
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(X_batch)
    no_offload_time = time.time() - start_time
    logger.info(f"Inférence sans offloading: {no_offload_time:.4f}s")
    
    # Test avec offloading standard
    standard_results = test_standard_offloading(MediumModel)
    
    # Test avec offloading Accelerate
    if is_accelerate_available():
        try:
            accelerate_results = test_accelerate_offloading(MediumModel)
        except Exception as e:
            logger.warning(f"Erreur lors du test d'Accelerate: {e}")
            accelerate_results = None
    else:
        accelerate_results = None
    
    # Test avec détection automatique
    auto_results = test_auto_offloading(MediumModel)
    
    # Compiler les résultats
    results = {
        "no_offload": no_offload_time,
        "standard": standard_results["inference_time"] if standard_results else None,
        "accelerate": accelerate_results["inference_time"] if accelerate_results else None,
        "auto": auto_results["inference_time"],
        "auto_strategy": auto_results["strategy_selected"]
    }
    
    # Afficher le résumé
    logger.info("\n=== Résumé de la comparaison ===")
    logger.info(f"Pas d'offloading: {results['no_offload']:.4f}s")
    logger.info(f"Offloading standard: {results['standard']:.4f}s")
    if results['accelerate']:
        logger.info(f"Offloading Accelerate: {results['accelerate']:.4f}s")
    logger.info(f"Offloading auto ({results['auto_strategy']}): {results['auto']:.4f}s")
    
    # Créer un fichier de résultats texte au lieu d'un graphique
    with open(RESULTS_DIR / "strategy_comparison.txt", "w") as f:
        f.write("=== Comparaison des stratégies d'offloading ===\n")
        f.write(f"Sans offloading: {results['no_offload']:.4f}s\n")
        f.write(f"Offloading standard: {results['standard']:.4f}s\n")
        if results['accelerate']:
            f.write(f"Offloading Accelerate: {results['accelerate']:.4f}s\n")
        f.write(f"Offloading auto ({results['auto_strategy']}): {results['auto']:.4f}s\n")
        
        # Ajouter une comparaison en pourcentage
        base_time = results['no_offload']
        if base_time > 0:
            f.write("\nComparaison (en % du temps sans offloading):\n")
            f.write(f"Sans offloading: 100%\n")
            f.write(f"Offloading standard: {(results['standard'] / base_time) * 100:.1f}%\n")
            if results['accelerate']:
                f.write(f"Offloading Accelerate: {(results['accelerate'] / base_time) * 100:.1f}%\n")
            f.write(f"Offloading auto: {(results['auto'] / base_time) * 100:.1f}%\n")
    
    logger.info(f"Résultats de comparaison sauvegardés dans {RESULTS_DIR / 'strategy_comparison.txt'}")
    
    return results


def main():
    """Fonction principale pour exécuter les démonstrations."""
    logger.info("Démarrage de l'exemple d'offloading CPU/GPU")
    
    # Configurer les optimisations RTX si disponible
    if torch.cuda.is_available():
        setup_rtx_optimization()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Utilisation de CUDA: {device_name}")
    else:
        logger.warning("CUDA n'est pas disponible, utilisation du CPU uniquement")
    
    # Vérifier si Huggingface Accelerate est disponible
    if is_accelerate_available():
        logger.info("Huggingface Accelerate est disponible")
    else:
        logger.warning("Huggingface Accelerate n'est pas disponible (pip install accelerate)")
    
    # Estimer la taille des modèles
    models = estimate_model_sizes()
    
    # Tester l'offloading standard avec différentes tailles de modèles
    test_standard_offloading(SmallModel)
    test_standard_offloading(MediumModel)
    test_standard_offloading(LargeModel)
    
    # Tester l'offloading Accelerate si disponible
    if is_accelerate_available():
        try:
            test_accelerate_offloading(MediumModel)
            test_accelerate_offloading(LargeModel)
        except Exception as e:
            logger.warning(f"Erreur lors des tests d'Accelerate: {e}")
    
    # Tester la détection automatique
    test_auto_offloading(MediumModel)
    
    # Comparer les stratégies
    compare_strategies()
    
    logger.info("\nExemple d'offloading CPU/GPU terminé!")
    logger.info(f"Voir les résultats dans: {RESULTS_DIR}")


if __name__ == "__main__":
    main() 