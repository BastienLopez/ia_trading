"""
Script de test pour l'optimisation mémoire du dashboard.

Ce script teste l'efficacité des fonctionnalités d'optimisation mémoire
et génère des statistiques sur les gains de performance.
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Ajouter le chemin du projet à l'import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.dashboard.memory_optimizer import (
    optimize_dataframe,
    get_df_memory_usage,
    downscale_for_visualization,
    chunk_dataframe,
    disk_cache,
    clean_cache,
    DataFramePool,
    configure_cache_dir
)

# Configuration pour les tests
TEST_CACHE_DIR = os.path.join(os.path.dirname(__file__), "test_cache")
configure_cache_dir(TEST_CACHE_DIR)

def generate_large_df(rows=100000, cols=20):
    """Génère un DataFrame volumineux pour les tests."""
    np.random.seed(42)
    
    # Créer un DataFrame avec différents types de données
    data = {}
    
    # Colonnes d'entiers
    for i in range(int(cols * 0.3)):
        data[f"int_col_{i}"] = np.random.randint(0, 1000, size=rows)
    
    # Colonnes flottantes
    for i in range(int(cols * 0.3)):
        data[f"float_col_{i}"] = np.random.random(size=rows) * 100
    
    # Colonnes catégorielles
    for i in range(int(cols * 0.2)):
        n_categories = 50
        data[f"cat_col_{i}"] = np.random.choice(
            [f"category_{j}" for j in range(n_categories)], 
            size=rows
        )
    
    # Colonnes date/heure
    for i in range(int(cols * 0.1)):
        start_date = datetime(2020, 1, 1)
        data[f"date_col_{i}"] = pd.date_range(start=start_date, periods=rows, freq="h")
    
    # Colonnes booléennes
    for i in range(int(cols * 0.1)):
        data[f"bool_col_{i}"] = np.random.choice([True, False], size=rows)
        
    return pd.DataFrame(data)

def test_optimize_dataframe():
    """Teste l'optimisation de DataFrame et mesure les gains."""
    print("\n=== Test d'optimisation de DataFrame ===")
    
    # Générer un DataFrame de test
    print("Génération d'un grand DataFrame...")
    df = generate_large_df(100000, 15)
    
    # Mesurer l'utilisation mémoire avant optimisation
    before_mem, before_detail = get_df_memory_usage(df)
    print(f"Mémoire avant optimisation: {before_mem:.2f} MB")
    
    # Top 5 colonnes par utilisation mémoire
    top_cols = sorted(before_detail.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 des colonnes les plus volumineuses avant optimisation:")
    for col, mem in top_cols:
        print(f"  {col}: {mem:.2f} MB")
    
    # Optimiser le DataFrame
    print("\nOptimisation du DataFrame...")
    start_time = time.time()
    optimized_df = optimize_dataframe(df, verbose=True)
    optimization_time = time.time() - start_time
    
    # Mesurer l'utilisation mémoire après optimisation
    after_mem, after_detail = get_df_memory_usage(optimized_df)
    print(f"Temps d'optimisation: {optimization_time:.2f} secondes")
    
    # Vérifier que les données sont équivalentes
    print("\nVérification de l'intégrité des données...")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(optimized_df[col]):
            # Pour les colonnes numériques, vérifier que les valeurs sont proches
            if not np.allclose(df[col].fillna(0), optimized_df[col].fillna(0), rtol=1e-4, atol=1e-4):
                print(f"ATTENTION: Potentielle perte de précision dans la colonne {col}")
        else:
            # Pour les autres types, vérifier l'égalité
            if not df[col].equals(optimized_df[col]):
                print(f"ATTENTION: Différences dans la colonne {col}")
    
    # Calcul du gain
    memory_reduction = before_mem - after_mem
    memory_reduction_pct = 100 * (1 - after_mem / before_mem)
    print(f"\nRéduction mémoire: {memory_reduction:.2f} MB ({memory_reduction_pct:.1f}%)")
    
    # Afficher les résultats
    print_memory_comparison(before_detail, after_detail)
    
    return before_mem, after_mem, optimization_time

def test_downscaling():
    """Teste la réduction d'échelle pour visualisation."""
    print("\n=== Test de réduction d'échelle pour visualisation ===")
    
    # Générer un grand DataFrame
    rows = 100000
    df = pd.DataFrame({
        'x': np.random.randn(rows).cumsum(),
        'y': np.random.randn(rows).cumsum(),
        'z': np.random.randn(rows).cumsum(),
        'timestamp': pd.date_range(start='2020-01-01', periods=rows, freq='1min')
    })
    
    print(f"DataFrame d'origine: {len(df):,} points")
    
    # Tester différentes tailles de downscaling
    max_points_options = [5000, 10000, 20000, 50000]
    results = []
    
    for max_points in max_points_options:
        start_time = time.time()
        downscaled_df = downscale_for_visualization(df, max_points=max_points)
        downscale_time = time.time() - start_time
        
        # Mesurer l'utilisation mémoire
        original_mem = df.memory_usage(deep=True).sum() / 1024**2
        downscaled_mem = downscaled_df.memory_usage(deep=True).sum() / 1024**2
        
        reduction_pct = 100 * (1 - len(downscaled_df) / len(df))
        
        print(f"Downscaling à {max_points:,} points:")
        print(f"  Taille résultante: {len(downscaled_df):,} points")
        print(f"  Réduction: {reduction_pct:.1f}%")
        print(f"  Mémoire: {downscaled_mem:.2f} MB (vs {original_mem:.2f} MB)")
        print(f"  Temps: {downscale_time:.3f} secondes")
        
        results.append({
            'max_points': max_points,
            'actual_points': len(downscaled_df),
            'memory_mb': downscaled_mem,
            'reduction_pct': reduction_pct,
            'time_sec': downscale_time
        })
    
    # Afficher les résultats
    print_downscaling_results(results)
    
    return results

def test_data_frame_pool():
    """Teste le pool de DataFrames."""
    print("\n=== Test du pool de DataFrames ===")
    
    # Initialiser le pool
    df_pool = DataFramePool(max_pool_size=5)
    
    # Créer plusieurs DataFrames
    dataframes = []
    for i in range(10):
        size = np.random.randint(10000, 50000)
        df = generate_large_df(rows=size, cols=5)
        dataframes.append((f"df_{i}", df))
        print(f"Créé DataFrame {i}: {size:,} lignes")
    
    # Ajouter les DataFrames au pool
    print("\nAjout des DataFrames au pool...")
    for key, df in dataframes:
        before_mem = df.memory_usage(deep=True).sum() / 1024**2
        df_pool.add(key, df)
        print(f"Ajouté {key}: {before_mem:.2f} MB")
    
    # Récupérer quelques DataFrames
    print("\nRécupération de DataFrames depuis le pool...")
    for i in range(0, 10, 2):
        key = f"df_{i}"
        start_time = time.time()
        retrieved_df = df_pool.get(key)
        retrieval_time = time.time() - start_time
        
        if retrieved_df is not None:
            print(f"Récupéré {key}: {len(retrieved_df):,} lignes en {retrieval_time:.3f} secondes")
        else:
            print(f"DataFrame {key} non trouvé dans le pool (attendu pour pool_size=5)")
    
    # Afficher la taille du pool
    pool_size = len(df_pool.pool)
    print(f"\nTaille finale du pool: {pool_size} (max={df_pool.max_pool_size})")
    
    # Vider le pool
    df_pool.clear()
    print("Pool vidé.")
    
    return pool_size

def test_disk_cache():
    """Teste le cache sur disque."""
    print("\n=== Test du cache sur disque ===")
    
    # Nettoyer le répertoire de cache
    if os.path.exists(TEST_CACHE_DIR):
        for file in os.listdir(TEST_CACHE_DIR):
            os.remove(os.path.join(TEST_CACHE_DIR, file))
    
    # Fonction qui sera mise en cache
    @disk_cache(TEST_CACHE_DIR, expiry_hours=1)
    def expensive_computation(size, seed):
        """Simulation d'une opération coûteuse."""
        print(f"Exécution du calcul coûteux (taille={size}, seed={seed})...")
        np.random.seed(seed)
        time.sleep(1)  # Simuler un long traitement
        return pd.DataFrame({
            'data': np.random.randn(size).cumsum(),
            'timestamp': pd.date_range(start='2022-01-01', periods=size, freq='1min')
        })
    
    # Première exécution (sans cache)
    print("\nPremière exécution (devrait calculer):")
    start_time = time.time()
    result1 = expensive_computation(10000, 42)
    first_time = time.time() - start_time
    print(f"Temps: {first_time:.3f} seconds")
    
    # Deuxième exécution (avec cache)
    print("\nDeuxième exécution (devrait utiliser le cache):")
    start_time = time.time()
    result2 = expensive_computation(10000, 42)
    second_time = time.time() - start_time
    print(f"Temps: {second_time:.3f} seconds")
    
    # Exécution avec paramètres différents (sans cache)
    print("\nExécution avec paramètres différents (devrait calculer):")
    start_time = time.time()
    result3 = expensive_computation(10000, 43)
    third_time = time.time() - start_time
    print(f"Temps: {third_time:.3f} seconds")
    
    # Vérifier le contenu du cache
    cache_files = os.listdir(TEST_CACHE_DIR)
    print(f"\nFichiers dans le cache: {len(cache_files)}")
    
    # Nettoyage du cache
    count, size_mb = clean_cache(TEST_CACHE_DIR, max_age_hours=0)
    print(f"Nettoyage du cache: {count} fichiers supprimés ({size_mb:.2f} MB)")
    
    # Calculer le speedup
    speedup = first_time / second_time if second_time > 0 else float('inf')
    print(f"\nAccélération avec cache: {speedup:.1f}x")
    
    return speedup

def print_memory_comparison(before_detail, after_detail):
    """Affiche une comparaison de l'utilisation mémoire avant/après."""
    print("\nTop 10 colonnes - Comparaison mémoire avant/après optimisation:")
    print("-" * 80)
    print(f"{'Colonne':<20} {'Avant (MB)':<15} {'Après (MB)':<15} {'Réduction (%)':<15}")
    print("-" * 80)
    
    # Filtrer pour les 10 colonnes les plus volumineuses
    top_columns = sorted(before_detail.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for col, before_mem in top_columns:
        after_mem = after_detail.get(col, 0)
        reduction = 100 * (1 - after_mem / before_mem) if before_mem > 0 else 0
        print(f"{col:<20} {before_mem:<15.2f} {after_mem:<15.2f} {reduction:<15.1f}")
    
    print("-" * 80)

def print_downscaling_results(results):
    """Affiche les résultats de downscaling."""
    print("\nRésultats du downscaling:")
    print("-" * 80)
    print(f"{'Max Points':<15} {'Points Réduits':<15} {'Mémoire (MB)':<15} {'Réduction (%)':<15} {'Temps (s)':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['max_points']:<15,} {result['actual_points']:<15,} {result['memory_mb']:<15.2f} {result['reduction_pct']:<15.1f} {result['time_sec']:<15.3f}")
    
    print("-" * 80)
    
    avg_reduction = np.mean([r['reduction_pct'] for r in results])
    print(f"Réduction moyenne: {avg_reduction:.1f}%")

def run_all_tests():
    """Exécute tous les tests d'optimisation mémoire."""
    print("=== TESTS D'OPTIMISATION MÉMOIRE ===")
    
    # Créer le répertoire de test s'il n'existe pas
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)
    
    # Exécuter les tests
    before_mem, after_mem, optimization_time = test_optimize_dataframe()
    downscaling_results = test_downscaling()
    pool_size = test_data_frame_pool()
    speedup = test_disk_cache()
    
    # Résumé des résultats
    print("\n=== RÉSUMÉ DES TESTS ===")
    print(f"Optimisation DataFrame: {before_mem:.2f} MB → {after_mem:.2f} MB ({100*(1-after_mem/before_mem):.1f}% réduction)")
    print(f"Temps d'optimisation: {optimization_time:.2f} secondes")
    print(f"Downscaling: Réduction mémoire moyenne: {np.mean([r['reduction_pct'] for r in downscaling_results]):.1f}%")
    print(f"Pool de DataFrames: {pool_size} frames dans un pool limité à 5")
    print(f"Cache sur disque: Accélération de {speedup:.1f}x")
    
    print("\nTests complétés.")

if __name__ == "__main__":
    run_all_tests() 