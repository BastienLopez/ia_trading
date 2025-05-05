#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation des optimisations des opérations critiques.

Ce script démontre comment utiliser les optimisations suivantes :
- torch.jit.script pour les fonctions fréquemment appelées
- torch.vmap pour les opérations vectorisées
- torch.compile() pour optimiser les modèles
- cudnn.benchmark pour optimiser les convolutions
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importer le module d'optimisation des opérations critiques
from ai_trading.optim.critical_operations import (
    use_jit_script,
    fast_matrix_multiply,
    optimize_model_with_compile,
    enable_cudnn_benchmark,
    VectorizedOperations,
    configure_performance_settings,
    benchmark_function
)

def run_jit_script_example():
    """Exemple d'utilisation de torch.jit.script pour les fonctions."""
    logger.info("=== Exemple d'utilisation de torch.jit.script ===")
    
    # Définir une fonction simple qui peut bénéficier de JIT
    def slow_activation(x, alpha=0.1):
        """Fonction d'activation LeakyReLU implémentée manuellement."""
        result = torch.empty_like(x)
        for i in range(x.numel()):
            val = x.flatten()[i]
            result.flatten()[i] = val if val > 0 else alpha * val
        return result
    
    # Version scriptée avec JIT
    @use_jit_script
    def fast_activation(x, alpha=0.1):
        """Fonction d'activation LeakyReLU optimisée avec JIT."""
        return torch.where(x > 0, x, alpha * x)
    
    # Générer des données aléatoires
    x = torch.randn(1000, 1000)
    
    # Mesurer le temps d'exécution pour la version lente
    start_time = time.time()
    slow_result = slow_activation(x)
    slow_time = time.time() - start_time
    logger.info(f"Temps d'exécution de slow_activation: {slow_time:.6f} secondes")
    
    # Mesurer le temps d'exécution pour la version rapide
    start_time = time.time()
    fast_result = fast_activation(x)
    fast_time = time.time() - start_time
    logger.info(f"Temps d'exécution de fast_activation: {fast_time:.6f} secondes")
    
    # Calculer le speedup
    speedup = slow_time / fast_time
    logger.info(f"Speedup avec JIT: {speedup:.2f}x")
    
    # Vérifier que les résultats sont identiques
    max_diff = torch.max(torch.abs(slow_result - fast_result)).item()
    logger.info(f"Différence maximale entre les résultats: {max_diff}")
    
    return speedup

def run_vmap_example():
    """Exemple d'utilisation de torch.vmap pour les opérations vectorisées."""
    logger.info("\n=== Exemple d'utilisation de torch.vmap ===")
    
    # Vérifier si vmap est disponible
    if not hasattr(torch, 'vmap'):
        logger.warning("torch.vmap n'est pas disponible dans cette version de PyTorch")
        return 0
    
    # Créer une instance de VectorizedOperations
    vec_ops = VectorizedOperations()
    
    # Générer des données aléatoires
    batch_size = 100
    m = 50
    n = 40
    
    matrices = torch.randn(batch_size, m, n)
    vectors = torch.randn(batch_size, n)
    
    # Version traditionnelle avec boucle for
    def slow_batch_matrix_vector_product(matrices, vectors):
        results = []
        for i in range(matrices.shape[0]):
            results.append(torch.mv(matrices[i], vectors[i]))
        return torch.stack(results)
    
    # Mesurer le temps d'exécution pour la version lente
    start_time = time.time()
    slow_results = slow_batch_matrix_vector_product(matrices, vectors)
    slow_time = time.time() - start_time
    logger.info(f"Temps d'exécution avec boucle for: {slow_time:.6f} secondes")
    
    # Mesurer le temps d'exécution pour la version vectorisée
    start_time = time.time()
    fast_results = vec_ops.batch_matrix_vector_product(matrices, vectors)
    fast_time = time.time() - start_time
    logger.info(f"Temps d'exécution avec vmap: {fast_time:.6f} secondes")
    
    # Calculer le speedup
    speedup = slow_time / fast_time
    logger.info(f"Speedup avec vmap: {speedup:.2f}x")
    
    # Vérifier que les résultats sont identiques
    max_diff = torch.max(torch.abs(slow_results - fast_results)).item()
    logger.info(f"Différence maximale entre les résultats: {max_diff}")
    
    return speedup

def run_compile_example():
    """Exemple d'utilisation de torch.compile pour optimiser les modèles."""
    logger.info("\n=== Exemple d'utilisation de torch.compile ===")
    
    # Vérifier si compile est disponible
    if not hasattr(torch, 'compile'):
        logger.warning("torch.compile n'est pas disponible dans cette version de PyTorch")
        return 0
    
    # Créer un modèle simple mais suffisamment complexe
    class ComplexModel(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Créer le modèle
    model = ComplexModel()
    
    # Optimiser le modèle avec torch.compile
    compiled_model = optimize_model_with_compile(model)
    
    # Générer des données aléatoires
    batch_size = 1024
    input_dim = 64
    x = torch.randn(batch_size, input_dim)
    
    # Échauffer les modèles (première inférence est souvent plus lente)
    with torch.no_grad():
        _ = model(x)
        _ = compiled_model(x)
    
    # Mesurer le temps d'exécution pour le modèle standard
    with torch.no_grad():
        start_time = time.time()
        for _ in range(10):
            _ = model(x)
        standard_time = (time.time() - start_time) / 10
        logger.info(f"Temps d'exécution moyen du modèle standard: {standard_time:.6f} secondes")
    
    # Mesurer le temps d'exécution pour le modèle compilé
    with torch.no_grad():
        start_time = time.time()
        for _ in range(10):
            _ = compiled_model(x)
        compiled_time = (time.time() - start_time) / 10
        logger.info(f"Temps d'exécution moyen du modèle compilé: {compiled_time:.6f} secondes")
    
    # Calculer le speedup
    speedup = standard_time / compiled_time
    logger.info(f"Speedup avec torch.compile: {speedup:.2f}x")
    
    return speedup

def run_cudnn_benchmark_example():
    """Exemple d'utilisation de cudnn.benchmark pour optimiser les convolutions."""
    logger.info("\n=== Exemple d'utilisation de cudnn.benchmark ===")
    
    # Vérifier si CUDA est disponible
    if not torch.cuda.is_available():
        logger.warning("CUDA n'est pas disponible, impossible de tester cudnn.benchmark")
        return 0
    
    # Créer un modèle CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8 * 8, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    # Créer le modèle et le déplacer sur GPU
    model = SimpleCNN().cuda()
    
    # Générer des données aléatoires
    batch_size = 64
    x = torch.randn(batch_size, 3, 32, 32).cuda()
    
    # Désactiver cudnn.benchmark pour le premier test
    torch.backends.cudnn.benchmark = False
    
    # Échauffer le modèle
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # Mesurer le temps sans cudnn.benchmark
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(10):
            _ = model(x)
        end_time.record()
        
        torch.cuda.synchronize()
        no_benchmark_time = start_time.elapsed_time(end_time) / 10
        logger.info(f"Temps d'exécution moyen sans cudnn.benchmark: {no_benchmark_time:.3f} ms")
    
    # Activer cudnn.benchmark
    enable_cudnn_benchmark(True)
    
    # Échauffer le modèle à nouveau
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # Mesurer le temps avec cudnn.benchmark
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(10):
            _ = model(x)
        end_time.record()
        
        torch.cuda.synchronize()
        benchmark_time = start_time.elapsed_time(end_time) / 10
        logger.info(f"Temps d'exécution moyen avec cudnn.benchmark: {benchmark_time:.3f} ms")
    
    # Calculer le speedup
    speedup = no_benchmark_time / benchmark_time
    logger.info(f"Speedup avec cudnn.benchmark: {speedup:.2f}x")
    
    return speedup

def main(args):
    """
    Fonction principale qui exécute les exemples d'optimisation.
    
    Args:
        args: Arguments de ligne de commande
    """
    logger.info("Démonstration des optimisations des opérations critiques")
    
    # Configurer les paramètres de performance optimaux
    configure_performance_settings()
    
    # Exécuter les exemples
    speedups = {}
    
    if args.all or args.jit:
        speedups['jit'] = run_jit_script_example()
    
    if args.all or args.vmap:
        speedups['vmap'] = run_vmap_example()
    
    if args.all or args.compile:
        speedups['compile'] = run_compile_example()
    
    if args.all or args.cudnn:
        speedups['cudnn'] = run_cudnn_benchmark_example()
    
    # Afficher un résumé des speedups
    if speedups:
        logger.info("\n=== Résumé des optimisations ===")
        for name, speedup in speedups.items():
            logger.info(f"- {name}: {speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démonstration des optimisations des opérations critiques")
    parser.add_argument("--all", action="store_true", help="Exécuter tous les exemples")
    parser.add_argument("--jit", action="store_true", help="Exécuter l'exemple de torch.jit.script")
    parser.add_argument("--vmap", action="store_true", help="Exécuter l'exemple de torch.vmap")
    parser.add_argument("--compile", action="store_true", help="Exécuter l'exemple de torch.compile")
    parser.add_argument("--cudnn", action="store_true", help="Exécuter l'exemple de cudnn.benchmark")
    
    args = parser.parse_args()
    
    # Si aucun argument n'est spécifié, exécuter tous les exemples
    if not (args.all or args.jit or args.vmap or args.compile or args.cudnn):
        args.all = True
    
    main(args) 