#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation des optimisations de réduction des temps d'opération.

Ce script démontre comment utiliser les fonctionnalités suivantes :
- Pré-calcul et mise en cache des résultats fréquents
- Utilisation de batchs de taille optimale (puissance de 2)
- Système de cache intelligent pour les prédictions
- Parallélisation des opérations indépendantes
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

# Importer les fonctionnalités de réduction des temps d'opération
from ai_trading.optim.operation_time_reduction import (
    precalculate_and_cache,
    get_optimal_batch_size,
    PredictionCache,
    get_prediction_cache,
    ParallelOperations
)

def demo_precalculate_and_cache():
    """Démo de l'utilisation du décorateur precalculate_and_cache."""
    logger.info("=== Démonstration du pré-calcul et mise en cache ===")
    
    # Définition d'une fonction coûteuse à mettre en cache
    @precalculate_and_cache
    def calculate_fibonacci(n):
        """Calcule le n-ième nombre de Fibonacci (implémentation inefficace pour la démo)."""
        if n <= 1:
            return n
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    
    # Premier appel (calcul complet)
    logger.info("Premier appel de calculate_fibonacci(30)...")
    start_time = time.time()
    result1 = calculate_fibonacci(30)
    elapsed1 = time.time() - start_time
    logger.info(f"Résultat: {result1}, temps: {elapsed1:.4f} secondes")
    
    # Deuxième appel (devrait utiliser le cache)
    logger.info("Deuxième appel de calculate_fibonacci(30)...")
    start_time = time.time()
    result2 = calculate_fibonacci(30)
    elapsed2 = time.time() - start_time
    logger.info(f"Résultat: {result2}, temps: {elapsed2:.4f} secondes")
    
    # Calcul du speedup
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float('inf')
    logger.info(f"Speedup avec mise en cache: {speedup:.2f}x")
    
    return speedup

def demo_optimal_batch_size():
    """Démo de l'utilisation de la taille de batch optimale."""
    logger.info("\n=== Démonstration de l'utilisation de batchs de taille optimale ===")
    
    # Créer un modèle simple pour la démo
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Créer le modèle
    model = SimpleModel()
    model.eval()
    
    # Générer des données aléatoires pour la démo
    num_samples = 1024
    data = torch.randn(num_samples, 100)
    
    # Tester différentes tailles de batch
    batch_sizes = [1, 4, 16, 32, 64, 128, 256, 512, 1024]
    times = []
    
    for batch_size in batch_sizes:
        # Calculer par batch
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = data[i:i+batch_size]
                _ = model(batch)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        logger.info(f"Batch size {batch_size}: {elapsed:.4f} secondes")
    
    # Trouver la taille de batch la plus rapide
    fastest_idx = times.index(min(times))
    fastest_batch_size = batch_sizes[fastest_idx]
    
    # Calculer la taille de batch optimale théorique
    optimal_batch_size = get_optimal_batch_size(1, num_samples)
    
    logger.info(f"Taille de batch la plus rapide empiriquement: {fastest_batch_size}")
    logger.info(f"Taille de batch optimale théorique: {optimal_batch_size}")
    
    return {
        "batch_sizes": batch_sizes,
        "times": times,
        "fastest_batch_size": fastest_batch_size,
        "optimal_batch_size": optimal_batch_size
    }

def demo_prediction_cache():
    """Démo de l'utilisation du cache intelligent pour les prédictions."""
    logger.info("\n=== Démonstration du cache intelligent pour les prédictions ===")
    
    # Créer un "modèle" fictif (juste une fonction) qui prend du temps à calculer
    def slow_model(input_data):
        """Simuler une prédiction lente."""
        time.sleep(0.1)  # Simuler un calcul lent
        return input_data * 2
    
    # Créer un wrapper qui utilise le cache
    def cached_model(input_data, cache_name="demo_cache"):
        """Version qui utilise le cache."""
        # Convertir l'entrée en une clé de cache (utiliser une représentation string)
        cache_key = str(input_data)
        
        # Récupérer le cache
        cache = get_prediction_cache(cache_name)
        
        # Vérifier si le résultat est en cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Calculer le résultat et le mettre en cache
        result = slow_model(input_data)
        cache.put(cache_key, result)
        return result
    
    # Effectuer plusieurs prédictions, certaines avec les mêmes entrées
    inputs = [1, 2, 3, 1, 2, 3, 4, 5]
    
    # Mesurer le temps sans cache
    logger.info("Prédictions sans cache:")
    start_time = time.time()
    results_no_cache = [slow_model(x) for x in inputs]
    elapsed_no_cache = time.time() - start_time
    logger.info(f"Temps total: {elapsed_no_cache:.4f} secondes")
    
    # Mesurer le temps avec cache
    logger.info("Prédictions avec cache:")
    start_time = time.time()
    results_with_cache = [cached_model(x) for x in inputs]
    elapsed_with_cache = time.time() - start_time
    logger.info(f"Temps total: {elapsed_with_cache:.4f} secondes")
    
    # Calculer le speedup
    speedup = elapsed_no_cache / elapsed_with_cache if elapsed_with_cache > 0 else float('inf')
    logger.info(f"Speedup avec cache de prédictions: {speedup:.2f}x")
    
    # Vérifier que les résultats sont identiques
    logger.info(f"Résultats identiques: {results_no_cache == results_with_cache}")
    
    return speedup

def demo_parallel_operations():
    """Démo de la parallélisation des opérations indépendantes."""
    logger.info("\n=== Démonstration de la parallélisation des opérations ===")
    
    # Définir une opération qui prend du temps
    def slow_operation(x):
        """Simulation d'une opération lente."""
        time.sleep(0.1)
        return x * 2
    
    # Créer des données de test
    data = list(range(20))
    
    # Exécuter en séquentiel
    logger.info("Exécution séquentielle:")
    start_time = time.time()
    results_sequential = [slow_operation(x) for x in data]
    elapsed_sequential = time.time() - start_time
    logger.info(f"Temps total: {elapsed_sequential:.4f} secondes")
    
    # Exécuter en parallèle avec des threads
    logger.info("Exécution parallèle avec threads:")
    parallel_ops = ParallelOperations()
    start_time = time.time()
    results_parallel = parallel_ops.parallel_map(slow_operation, data)
    elapsed_parallel = time.time() - start_time
    logger.info(f"Temps total: {elapsed_parallel:.4f} secondes")
    
    # Calculer le speedup
    speedup = elapsed_sequential / elapsed_parallel if elapsed_parallel > 0 else float('inf')
    logger.info(f"Speedup avec parallélisation: {speedup:.2f}x")
    
    # Vérifier que les résultats sont identiques
    logger.info(f"Résultats identiques: {results_sequential == results_parallel}")
    
    return speedup

def main(args):
    """
    Fonction principale qui exécute les démonstrations.
    
    Args:
        args: Arguments de ligne de commande
    """
    logger.info("Démonstration des optimisations de réduction des temps d'opération")
    
    # Stocker les résultats de speedup
    speedups = {}
    
    # Exécuter les démonstrations selon les arguments
    if args.all or args.cache:
        speedups['cache'] = demo_precalculate_and_cache()
    
    if args.all or args.batch:
        batch_results = demo_optimal_batch_size()
        # Calculer le speedup en comparant la taille de batch la plus lente et la plus rapide
        slowest_time = max(batch_results['times'])
        fastest_time = min(batch_results['times'])
        speedups['batch'] = slowest_time / fastest_time if fastest_time > 0 else float('inf')
    
    if args.all or args.prediction:
        speedups['prediction'] = demo_prediction_cache()
    
    if args.all or args.parallel:
        speedups['parallel'] = demo_parallel_operations()
    
    # Afficher un résumé des speedups
    if speedups:
        logger.info("\n=== Résumé des optimisations ===")
        for name, speedup in speedups.items():
            logger.info(f"- {name}: {speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démonstration des optimisations de réduction des temps d'opération")
    parser.add_argument("--all", action="store_true", help="Exécuter toutes les démonstrations")
    parser.add_argument("--cache", action="store_true", help="Exécuter la démo de mise en cache des résultats")
    parser.add_argument("--batch", action="store_true", help="Exécuter la démo de taille de batch optimale")
    parser.add_argument("--prediction", action="store_true", help="Exécuter la démo de cache de prédictions")
    parser.add_argument("--parallel", action="store_true", help="Exécuter la démo de parallélisation")
    
    args = parser.parse_args()
    
    # Si aucun argument n'est spécifié, exécuter toutes les démonstrations
    if not (args.all or args.cache or args.batch or args.prediction or args.parallel):
        args.all = True
    
    main(args) 