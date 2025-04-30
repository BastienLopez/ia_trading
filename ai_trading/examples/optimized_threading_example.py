#!/usr/bin/env python
"""
Exemple d'utilisation des optimisations de multithreading/multiprocessing pour le système de trading.
Ce script montre comment utiliser les optimisations de threading pour accélérer le chargement et
le traitement des données financières.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.utils.performance_tracker import track_time, PerformanceTracker
from ai_trading.utils.threading_optimizer import (
    ThreadingOptimizer, 
    parallel_map, 
    optimize_system_for_training
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def generate_test_data(n_points=50000):
    """Génère des données de test pour l'exemple."""
    logger.info(f"Génération de {n_points} points de données synthétiques...")
    
    data = generate_synthetic_market_data(
        n_points=n_points,
        trend=0.0005,
        volatility=0.01,
        start_price=100.0,
        with_date=True,
        cyclic_pattern=True,
        include_volume=True
    )
    
    return data

@track_time
def preprocessing_sequential(data):
    """Effectue un prétraitement séquentiel des données."""
    logger.info("Prétraitement séquentiel des données...")
    
    result = []
    for i in range(len(data)):
        # Simuler un traitement lourd
        row = data.iloc[i]
        processed = {
            'log_return': np.log(row['close'] / row['open']) if row['open'] > 0 else 0,
            'volatility': np.sqrt(np.square(row['high'] - row['low'])),
            'momentum': row['close'] - data.iloc[max(0, i-10):i+1]['close'].mean() if i > 0 else 0,
            'volume_ratio': row['volume'] / data.iloc[max(0, i-5):i+1]['volume'].mean() if i > 0 else 1
        }
        result.append(processed)
    
    return pd.DataFrame(result)

def process_row(row_data):
    """Traite une seule ligne de données (pour parallel_map)."""
    index, row = row_data
    data_window = row['data_window']
    
    # Simuler un traitement lourd
    processed = {
        'log_return': np.log(row['close'] / row['open']) if row['open'] > 0 else 0,
        'volatility': np.sqrt(np.square(row['high'] - row['low'])),
        'momentum': row['close'] - data_window['close'].mean() if len(data_window) > 0 else 0,
        'volume_ratio': row['volume'] / data_window['volume'].mean() if len(data_window) > 0 else 1
    }
    return processed

@track_time
def preprocessing_parallel(data, num_workers=None):
    """Effectue un prétraitement parallèle des données."""
    logger.info(f"Prétraitement parallèle des données avec {num_workers} workers...")
    
    # Préparation des données avec fenêtres glissantes
    items = []
    for i in range(len(data)):
        window = data.iloc[max(0, i-10):i]
        items.append((i, {
            'open': data.iloc[i]['open'],
            'close': data.iloc[i]['close'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low'],
            'volume': data.iloc[i]['volume'],
            'data_window': window
        }))
    
    # Traitement parallèle
    results = parallel_map(
        process_row, 
        items, 
        max_workers=num_workers, 
        use_processes=True,
        show_progress=True
    )
    
    return pd.DataFrame(results)

def compare_dataloader_configs(dataset, configs):
    """Compare différentes configurations de DataLoader."""
    logger.info("Comparaison de différentes configurations de DataLoader...")
    
    results = {}
    
    for name, config in configs.items():
        logger.info(f"Test avec configuration: {name}")
        
        # Créer le DataLoader avec cette configuration
        dataloader = get_financial_dataloader(
            dataset=dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            prefetch_factor=config.get('prefetch_factor'),
            pin_memory=config.get('pin_memory', False),
            persistent_workers=config.get('persistent_workers', False),
            auto_threading=False  # Désactiver l'auto-threading pour utiliser les valeurs fournies
        )
        
        # Mesurer les performances
        tracker = PerformanceTracker(name=name)
        
        # Parcourir tous les batches
        start_time = time.time()
        batch_count = 0
        
        for _ in dataloader:
            batch_count += 1
            if batch_count % 10 == 0:
                tracker.measure(f"batch_{batch_count}")
        
        elapsed_time = time.time() - start_time
        batches_per_second = batch_count / elapsed_time
        
        # Enregistrer le résultat
        results[name] = {
            'batches_per_second': batches_per_second,
            'total_time': elapsed_time,
            'batch_count': batch_count
        }
        
        # Afficher le résumé
        tracker.log_summary()
        logger.info(f"Performance {name}: {batches_per_second:.2f} batches/s, temps total: {elapsed_time:.2f}s")
    
    # Trouver la meilleure configuration
    best_config = max(results.items(), key=lambda x: x[1]['batches_per_second'])
    logger.info(f"Meilleure configuration: {best_config[0]} avec {best_config[1]['batches_per_second']:.2f} batches/s")
    
    return results, best_config[0]

def main():
    """Fonction principale d'exemple."""
    logger.info("=== Exemple d'optimisation multithreading/multiprocessing pour le trading ===")
    
    # 1. Détection du système et recommandations
    threading_optimizer = ThreadingOptimizer()
    optimal_workers = threading_optimizer.calculate_optimal_workers()
    
    logger.info(f"Configuration système détectée:")
    logger.info(f"- CPU: {threading_optimizer.cpu_count} cœurs logiques, {threading_optimizer.physical_cores} cœurs physiques")
    logger.info(f"- Hyperthreading: {'Présent' if threading_optimizer.has_hyperthreading else 'Absent'}")
    logger.info(f"- Mémoire: {threading_optimizer.total_memory_gb:.1f} GB total")
    
    logger.info(f"Nombre optimal de workers recommandé:")
    logger.info(f"- DataLoader: {optimal_workers['dataloader']} workers")
    logger.info(f"- Prétraitement: {optimal_workers['preprocessing']} workers")
    logger.info(f"- Entraînement: {optimal_workers['training']} threads")
    
    # 2. Générer des données de test
    data = generate_test_data(n_points=20000)
    
    # 3. Comparer le prétraitement séquentiel vs parallèle
    logger.info("\n=== Comparaison prétraitement séquentiel vs parallèle ===")
    
    # Prétraitement séquentiel
    sequential_result = preprocessing_sequential(data)
    
    # Prétraitement parallèle
    parallel_result = preprocessing_parallel(data, num_workers=optimal_workers['preprocessing'])
    
    # 4. Créer un dataset et comparer différentes configs de DataLoader
    logger.info("\n=== Comparaison des configurations DataLoader ===")
    
    # Créer le dataset
    dataset = FinancialDataset(
        data=data,
        sequence_length=50,
        is_train=True,
        lazy_loading=True,
        chunk_size=5000
    )
    
    # Différentes configurations à tester
    dataloader_configs = {
        "Séquentiel": {
            "batch_size": 32,
            "num_workers": 0
        },
        "Workers_2": {
            "batch_size": 32,
            "num_workers": 2,
            "prefetch_factor": 2,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": True
        },
        "Workers_4": {
            "batch_size": 32,
            "num_workers": 4,
            "prefetch_factor": 2,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": True
        },
        "Optimal": {
            "batch_size": 32,
            "num_workers": optimal_workers['dataloader'],
            "prefetch_factor": min(4, max(2, optimal_workers['dataloader'] // 2)),
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": True
        }
    }
    
    # Comparer les configurations
    results, best_config = compare_dataloader_configs(dataset, dataloader_configs)
    
    # 5. Démontrer l'utilisation de DataLoader auto-optimisé
    logger.info("\n=== Démonstration du DataLoader auto-optimisé ===")
    
    # Configurer le système pour l'entraînement
    logger.info("Configuration du système pour l'entraînement...")
    optimize_system_for_training()
    
    # Créer un DataLoader auto-optimisé
    auto_dataloader = get_financial_dataloader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=-1,  # -1 déclenche l'auto-détection
        auto_threading=True
    )
    
    # Mesurer les performances
    tracker = PerformanceTracker(name="auto_optimized")
    
    # Parcourir tous les batches
    start_time = time.time()
    batch_count = 0
    
    for _ in auto_dataloader:
        batch_count += 1
        if batch_count % 10 == 0:
            tracker.measure(f"batch_{batch_count}")
    
    elapsed_time = time.time() - start_time
    batches_per_second = batch_count / elapsed_time
    
    # Afficher le résumé
    tracker.log_summary()
    logger.info(f"Performance auto-optimisée: {batches_per_second:.2f} batches/s, temps total: {elapsed_time:.2f}s")
    
    # 6. Résumé final
    logger.info("\n=== Résumé des optimisations de threading ===")
    
    # Calculer le speedup par rapport à la version séquentielle
    seq_performance = results["Séquentiel"]["batches_per_second"]
    auto_speedup = batches_per_second / seq_performance
    best_speedup = results[best_config]["batches_per_second"] / seq_performance
    
    logger.info(f"Accélération avec la meilleure configuration manuelle ({best_config}): {best_speedup:.2f}x")
    logger.info(f"Accélération avec la configuration auto-optimisée: {auto_speedup:.2f}x")
    
    logger.info("\nRecommandations:")
    if auto_speedup > best_speedup:
        logger.info("- Utiliser l'auto-optimisation (num_workers=-1, auto_threading=True)")
    else:
        logger.info(f"- Utiliser la configuration manuelle {best_config}")
    
    logger.info(f"- Pour le prétraitement des données, utiliser {optimal_workers['preprocessing']} workers")
    logger.info(f"- Pour les bibliothèques numériques (NumPy/PyTorch), limiter à {optimal_workers['training']} threads")

if __name__ == "__main__":
    main() 