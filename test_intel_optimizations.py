#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import logging
import sys

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importer notre module d'optimisations Intel
try:
    from ai_trading.utils.intel_optimizations import (
        optimize_for_intel, 
        print_optimization_info,
        is_intel_cpu
    )
except ImportError:
    logger.error("Module ai_trading.utils.intel_optimizations non trouvé.")
    logger.info("Vérifiez que le module est dans votre PYTHONPATH.")
    sys.exit(1)

def test_matrix_multiplication(size=2000, runs=3):
    """Test de multiplication de matrices pour vérifier les performances."""
    logger.info(f"Test de multiplication de matrices {size}x{size}")
    
    # Créer deux matrices aléatoires
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    # Mesurer le temps de multiplication sans optimisation
    times = []
    for i in range(runs):
        start = time.time()
        C = np.matmul(A, B)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    logger.info(f"Temps moyen pour multiplication de matrices: {avg_time:.4f} secondes")
    return avg_time

if __name__ == "__main__":
    logger.info("=== Test des optimisations Intel ===")
    
    # Vérifier si nous avons un CPU Intel
    if not is_intel_cpu():
        logger.warning("CPU non-Intel détecté. Les optimisations pourraient ne pas être efficaces.")
    
    # Test sans optimisations
    logger.info("Exécution sans optimisations Intel...")
    time_without_opt = test_matrix_multiplication()
    
    # Appliquer les optimisations
    logger.info("Application des optimisations Intel...")
    optimize_for_intel()
    
    # Afficher les informations d'optimisation
    logger.info("Informations sur les optimisations:")
    print_optimization_info()
    
    # Test avec optimisations
    logger.info("Exécution avec optimisations Intel...")
    time_with_opt = test_matrix_multiplication()
    
    # Calculer l'amélioration
    if time_without_opt > 0:
        improvement = (time_without_opt - time_with_opt) / time_without_opt * 100
        logger.info(f"Amélioration des performances: {improvement:.2f}%")
    
    logger.info("Test terminé.") 