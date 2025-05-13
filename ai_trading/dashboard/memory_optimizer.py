"""
Module d'optimisation de la mémoire pour le dashboard de trading.

Ce module fournit des fonctions pour réduire la consommation mémoire,
mettre en cache efficacement les résultats et nettoyer les données.
"""

import gc
import os
import pickle
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_df_memory_usage(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Calcule l'utilisation mémoire d'un DataFrame avec détail par colonne.

    Args:
        df: Le DataFrame à analyser

    Returns:
        Tuple contenant (taille_totale_en_MB, dict_détail_par_colonne)
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024**2  # En MB
    column_memory = {col: memory_usage[i] / 1024**2 for i, col in enumerate(df.columns)}

    return total_memory, column_memory


def optimize_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Optimise l'utilisation mémoire d'un DataFrame en convertissant les types de données.

    Args:
        df: Le DataFrame à optimiser
        verbose: Si True, affiche les détails de l'optimisation

    Returns:
        DataFrame optimisé
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"Mémoire initiale : {start_mem:.2f} MB")

    # Copie pour éviter de modifier l'original
    result = df.copy()

    # Optimisation des colonnes numériques
    for col in result.select_dtypes(include=["int"]).columns:
        col_min, col_max = result[col].min(), result[col].max()

        # Conversion en type int plus petit si possible
        if col_min >= 0:
            if col_max < 255:
                result[col] = result[col].astype(np.uint8)
            elif col_max < 65535:
                result[col] = result[col].astype(np.uint16)
            elif col_max < 4294967295:
                result[col] = result[col].astype(np.uint32)
            else:
                result[col] = result[col].astype(np.uint64)
        else:
            if col_min > -128 and col_max < 127:
                result[col] = result[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                result[col] = result[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                result[col] = result[col].astype(np.int32)
            else:
                result[col] = result[col].astype(np.int64)

    # Optimisation des colonnes flottantes
    for col in result.select_dtypes(include=["float"]).columns:
        # Tester si les valeurs sont suffisamment précises pour float32
        if result[col].round(4).equals(result[col]):
            result[col] = result[col].astype(np.float32)
        else:
            result[col] = result[col].astype(np.float64)

    # Optimisation des colonnes catégorielles
    for col in result.select_dtypes(include=["object"]).columns:
        if (
            result[col].nunique() / len(result) < 0.5
        ):  # Si moins de 50% de valeurs uniques
            result[col] = result[col].astype("category")

    # Optimisation des dates - Correction du FutureWarning
    for col in result.select_dtypes(include=["datetime"]).columns:
        # Remplacer l'utilisation dépréciée de errors='ignore'
        try:
            result[col] = pd.to_datetime(result[col])
        except (ValueError, TypeError):
            # Conserver les valeurs d'origine si la conversion échoue
            pass

    end_mem = result.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (1 - end_mem / start_mem)

    if verbose:
        print(f"Mémoire finale : {end_mem:.2f} MB")
        print(f"Réduction : {reduction:.1f}%")

    return result


def disk_cache(cache_dir: str, expiry_hours: int = 24):
    """
    Décorateur pour mettre en cache les résultats de fonction sur le disque.

    Args:
        cache_dir: Répertoire où stocker les fichiers de cache
        expiry_hours: Durée de validité du cache en heures

    Returns:
        Fonction décorée
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Créer le répertoire de cache s'il n'existe pas
            os.makedirs(cache_dir, exist_ok=True)

            # Créer une clé de cache basée sur les arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = "_".join(key_parts)

            # Hacher la clé si elle est trop longue
            if len(key) > 100:
                import hashlib

                key = hashlib.md5(key.encode()).hexdigest()

            cache_file = os.path.join(cache_dir, f"{key}.pkl")

            # Vérifier si le cache existe et est valide
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(hours=expiry_hours):
                    try:
                        with open(cache_file, "rb") as f:
                            return pickle.load(f)
                    except (pickle.PickleError, EOFError):
                        # En cas d'erreur, supprimer le cache corrompu
                        os.remove(cache_file)

            # Calculer le résultat
            result = func(*args, **kwargs)

            # Sauvegarder dans le cache
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator


def clean_cache(cache_dir: str, max_age_hours: int = 48):
    """
    Nettoie les fichiers de cache obsolètes.

    Args:
        cache_dir: Répertoire contenant les fichiers de cache
        max_age_hours: Âge maximum des fichiers en heures
    """
    if not os.path.exists(cache_dir):
        return

    now = datetime.now()
    count = 0
    size = 0

    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(cache_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            # Supprimer les fichiers trop anciens
            if now - file_time > timedelta(hours=max_age_hours):
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                count += 1
                size += file_size

    return count, size / 1024**2  # Retourne nombre de fichiers et taille en MB


class DataFramePool:
    """
    Pool de DataFrames pour éviter la création excessive de copies.
    """

    def __init__(self, max_pool_size: int = 10):
        """
        Initialise le pool de DataFrames.

        Args:
            max_pool_size: Nombre maximum de DataFrames à conserver en mémoire
        """
        self.pool = {}
        self.last_access = {}
        self.max_pool_size = max_pool_size

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Récupère un DataFrame du pool.

        Args:
            key: Clé identifiant le DataFrame

        Returns:
            DataFrame s'il existe dans le pool, None sinon
        """
        if key in self.pool:
            self.last_access[key] = time.time()
            return self.pool[key].copy()
        return None

    def add(self, key: str, df: pd.DataFrame):
        """
        Ajoute un DataFrame au pool.

        Args:
            key: Clé pour stocker le DataFrame
            df: DataFrame à stocker
        """
        # Vérifier si le pool est plein
        if len(self.pool) >= self.max_pool_size:
            # Supprimer l'élément le moins récemment utilisé
            oldest_key = min(self.last_access.items(), key=lambda x: x[1])[0]
            del self.pool[oldest_key]
            del self.last_access[oldest_key]

        # Optimiser et stocker le DataFrame
        self.pool[key] = optimize_dataframe(df)
        self.last_access[key] = time.time()

    def clear(self):
        """Vide complètement le pool."""
        self.pool.clear()
        self.last_access.clear()
        gc.collect()  # Forcer le ramassage de la mémoire


def memory_profile(func: Callable) -> Callable:
    """
    Décorateur pour profiler l'utilisation mémoire d'une fonction.

    Args:
        func: Fonction à profiler

    Returns:
        Fonction décorée avec profilage mémoire
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Collecter la mémoire avant
        gc.collect()
        start_mem = psutil.Process().memory_info().rss / 1024**2

        # Exécuter la fonction
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Collecter la mémoire après
        gc.collect()
        end_mem = psutil.Process().memory_info().rss / 1024**2

        print(f"Fonction: {func.__name__}")
        print(f"Temps d'exécution: {end_time - start_time:.2f} sec")
        print(f"Mémoire avant: {start_mem:.2f} MB")
        print(f"Mémoire après: {end_mem:.2f} MB")
        print(f"Différence: {end_mem - start_mem:.2f} MB")

        return result

    return wrapper


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
    """
    Divise un grand DataFrame en chunks plus petits pour le traitement.

    Args:
        df: DataFrame à diviser
        chunk_size: Taille de chaque chunk

    Returns:
        Liste de DataFrames plus petits
    """
    if len(df) <= chunk_size:
        return [df]

    chunks = []
    num_chunks = (len(df) // chunk_size) + (1 if len(df) % chunk_size > 0 else 0)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunks.append(df.iloc[start_idx:end_idx].copy())

    return chunks


def downscale_for_visualization(
    df: pd.DataFrame, max_points: int = 5000
) -> pd.DataFrame:
    """
    Réduit le nombre de points dans un DataFrame pour la visualisation.

    Args:
        df: DataFrame à échantillonner
        max_points: Nombre maximum de points à conserver

    Returns:
        DataFrame échantillonné
    """
    # Vérifications de sécurité
    if df is None:
        print("WARNING: DataFrame None reçu dans downscale_for_visualization")
        # Retourner un DataFrame vide mais avec des colonnes
        return pd.DataFrame(columns=["timestamp", "value"])

    if df.empty:
        print("WARNING: DataFrame vide reçu dans downscale_for_visualization")
        return df

    # Si le DataFrame est déjà assez petit, le retourner tel quel
    if len(df) <= max_points:
        return df

    try:
        # Méthode 1: Échantillonnage systématique
        step = max(1, len(df) // max_points)  # S'assurer que step est au moins 1
        result = df.iloc[::step].reset_index(drop=True)

        # Vérifier que nous avons bien des données
        if result.empty:
            print(
                f"WARNING: Échantillonnage a résulté en DataFrame vide (step={step}, len original={len(df)})"
            )
            # Échantillonner différemment si nécessaire
            if len(df) > 0:
                # Prendre simplement les premiers 'max_points' éléments
                return df.head(max_points)

        return result
    except Exception as e:
        print(f"ERROR dans downscale_for_visualization: {str(e)}")
        # En cas d'erreur, retourner le DataFrame original
        return df


# Singleton pour le pool de DataFrames global
_global_df_pool = DataFramePool()


def get_global_df_pool() -> DataFramePool:
    """
    Récupère l'instance singleton du pool de DataFrames global.

    Returns:
        Instance du pool global
    """
    return _global_df_pool


# Configuration par défaut
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "info_retour", "cache"
)


def configure_cache_dir(path: str):
    """
    Configure le répertoire de cache.

    Args:
        path: Chemin du répertoire de cache
    """
    global CACHE_DIR
    CACHE_DIR = path
    os.makedirs(CACHE_DIR, exist_ok=True)
