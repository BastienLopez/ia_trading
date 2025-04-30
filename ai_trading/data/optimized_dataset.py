"""
Module pour l'optimisation des datasets financiers avec compression zstd.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader
from ai_trading.data.compressed_storage import CompressedStorage

logger = logging.getLogger(__name__)


class OptimizedFinancialDataset(FinancialDataset):
    """
    Version optimisée du dataset financier utilisant la compression zstd pour les fichiers volumineux.
    Permet de réduire l'espace disque et d'améliorer les performances de chargement.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path, torch.Tensor],
        sequence_length: int = 50,
        target_column: str = "close",
        predict_n_ahead: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_train: bool = True,
        use_shared_memory: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        compression_level: int = 3,
        use_compressed_cache: bool = True,
    ):
        """
        Initialise le dataset financier optimisé.

        Args:
            data: Données financières sous forme de DataFrame, chemin vers un fichier, ou Tensor.
            sequence_length: Longueur des séquences à extraire.
            target_column: Nom de la colonne cible pour la prédiction.
            predict_n_ahead: Nombre de pas de temps à prédire dans le futur.
            transform: Fonction de transformation pour les séquences.
            target_transform: Fonction de transformation pour les cibles.
            is_train: Si True, utilise toutes les données sauf la dernière sequence_length.
                    Si False, utilise seulement les dernières sequence_length données.
            use_shared_memory: Si True, utilise la mémoire partagée pour le multiprocessing.
            cache_dir: Répertoire où stocker les fichiers de cache compressés.
            compression_level: Niveau de compression zstd (1-22).
            use_compressed_cache: Si True, utilise la compression pour le cache.
        """
        # Initialiser le dataset parent
        super().__init__(
            data=data,
            sequence_length=sequence_length,
            target_column=target_column,
            predict_n_ahead=predict_n_ahead,
            transform=transform,
            target_transform=target_transform,
            is_train=is_train,
            use_shared_memory=use_shared_memory,
        )

        # Stocker les paramètres de compression
        self.compression_level = compression_level
        self.use_compressed_cache = use_compressed_cache

        # Configurer le répertoire de cache
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "ai_trading_cache"
        else:
            self.cache_dir = Path(cache_dir)

        # Créer le répertoire de cache s'il n'existe pas
        os.makedirs(self.cache_dir, exist_ok=True)

        # Créer le gestionnaire de stockage compressé
        self.storage = CompressedStorage(compression_level=compression_level)

        # Dictionnaire pour stocker les chemins des fichiers de cache
        self.cache_files = {}

    def cache_data(self, name: str, data: Any, format: str = "pickle") -> Path:
        """
        Met en cache des données sous forme compressée.

        Args:
            name: Nom unique pour les données mises en cache.
            data: Données à mettre en cache (DataFrame, tableau NumPy, etc.).
            format: Format de sérialisation ('parquet', 'csv', 'pickle', 'numpy').

        Returns:
            Chemin du fichier de cache.
        """
        if not self.use_compressed_cache:
            return None

        # Créer un nom de fichier unique basé sur les paramètres
        file_path = self.cache_dir / f"{name}.{format}.zst"
        self.cache_files[name] = file_path

        # Sauvegarder les données selon leur type
        if isinstance(data, pd.DataFrame):
            self.storage.save_dataframe(data, file_path, format=format)
        elif isinstance(data, np.ndarray):
            self.storage.save_numpy(data, file_path)
        elif isinstance(data, (dict, list)):
            self.storage.save_json(data, file_path)
        elif isinstance(data, torch.Tensor):
            # Convertir le tenseur en tableau NumPy pour la sauvegarde
            numpy_data = data.cpu().numpy()
            self.storage.save_numpy(numpy_data, file_path)
        else:
            raise TypeError(f"Type de données non pris en charge pour la mise en cache: {type(data)}")

        logger.info(f"Données '{name}' mises en cache dans {file_path}")
        return file_path

    def load_cached_data(self, name: str, format: str = "pickle", tensor_dtype=None) -> Any:
        """
        Charge des données depuis le cache compressé.

        Args:
            name: Nom unique des données mises en cache.
            format: Format de sérialisation ('parquet', 'csv', 'pickle', 'numpy').
            tensor_dtype: Type de données pour la conversion en tenseur (si applicable).

        Returns:
            Données chargées (DataFrame, tableau NumPy, tenseur, etc.).
        """
        if not self.use_compressed_cache or name not in self.cache_files:
            return None

        file_path = self.cache_files[name]
        if not file_path.exists():
            logger.warning(f"Fichier de cache non trouvé: {file_path}")
            return None

        # Charger les données selon le format
        if format in ["parquet", "csv", "pickle"]:
            data = self.storage.load_dataframe(file_path, format=format)
        elif format == "numpy":
            data = self.storage.load_numpy(file_path)
            # Convertir en tenseur si demandé
            if tensor_dtype is not None:
                data = torch.tensor(data, dtype=tensor_dtype)
        elif format == "json":
            data = self.storage.load_json(file_path)
        else:
            raise ValueError(f"Format non pris en charge: {format}")

        logger.info(f"Données '{name}' chargées depuis {file_path}")
        return data

    def precompute_features(self, compute_fn: Callable, name: str) -> torch.Tensor:
        """
        Précalcule et met en cache des caractéristiques.

        Args:
            compute_fn: Fonction qui calcule les caractéristiques.
            name: Nom unique pour les caractéristiques.

        Returns:
            Tenseur des caractéristiques précalculées.
        """
        # Vérifier si les caractéristiques sont déjà en cache
        features = self.load_cached_data(name, format="numpy", tensor_dtype=torch.float32)
        if features is not None:
            return features

        # Calculer les caractéristiques
        try:
            features = compute_fn()
        except TypeError:
            # Si compute_fn n'est pas callable, créer une fonction lambda qui retourne la valeur
            if callable(compute_fn):
                features = compute_fn()
            else:
                # Si c'est déjà un tenseur, l'utiliser directement
                features = compute_fn

        # Mettre en cache les caractéristiques
        self.cache_data(name, features, format="numpy")

        return features

    def clear_cache(self, names: Optional[List[str]] = None) -> None:
        """
        Supprime les fichiers de cache.

        Args:
            names: Liste des noms de cache à supprimer. Si None, supprime tous les caches.
        """
        if names is None:
            names = list(self.cache_files.keys())

        for name in names:
            if name in self.cache_files:
                file_path = self.cache_files[name]
                if file_path.exists():
                    try:
                        os.remove(file_path)
                        logger.info(f"Cache supprimé: {file_path}")
                    except OSError as e:
                        logger.error(f"Erreur lors de la suppression du cache {file_path}: {e}")
                del self.cache_files[name]

    def optimize_dataframe(self, df: pd.DataFrame, cache_name: str = "optimized_df") -> pd.DataFrame:
        """
        Optimise un DataFrame en mettant en cache la version compressée.

        Args:
            df: DataFrame à optimiser.
            cache_name: Nom du cache.

        Returns:
            DataFrame optimisé (peut être le même si l'optimisation échoue).
        """
        if not self.use_compressed_cache:
            return df

        try:
            # Mettre en cache le DataFrame
            self.cache_data(cache_name, df, format="parquet")
            
            # Charger depuis le cache pour vérifier
            cached_df = self.load_cached_data(cache_name, format="parquet")
            
            if cached_df is not None and cached_df.shape == df.shape:
                return cached_df
            else:
                logger.warning(f"Échec de l'optimisation du DataFrame '{cache_name}'")
                return df
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation du DataFrame: {e}")
            return df

    def compress_raw_data(self, data_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Compresse un fichier de données brutes et retourne le chemin du fichier compressé.

        Args:
            data_path: Chemin du fichier à compresser.
            output_path: Chemin du fichier compressé. Si None, ajoute l'extension .zst.

        Returns:
            Chemin du fichier compressé.
        """
        return self.storage.compress_file(data_path, output_path)

    def load_from_compressed(self, path: Union[str, Path], format: str = "parquet") -> pd.DataFrame:
        """
        Charge un DataFrame directement depuis un fichier compressé.

        Args:
            path: Chemin du fichier compressé.
            format: Format des données ('parquet', 'csv', 'pickle').

        Returns:
            DataFrame chargé.
        """
        return self.storage.load_dataframe(path, format=format)


def get_optimized_dataloader(
    dataset: OptimizedFinancialDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Crée un DataLoader optimisé pour le dataset compressé.

    Args:
        dataset: Instance d'OptimizedFinancialDataset.
        batch_size: Taille des batchs.
        shuffle: Si True, mélange les données.
        num_workers: Nombre de workers pour le chargement des données.
        prefetch_factor: Multiplicateur pour le prefetching.
        pin_memory: Si True, copie les tenseurs dans la mémoire CUDA pinned.
        drop_last: Si True, supprime le dernier batch s'il est incomplet.
        persistent_workers: Si True, garde les workers en vie entre les itérations.

    Returns:
        DataLoader optimisé.
    """
    return get_financial_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


def convert_to_compressed(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                        compression_level: int = 3) -> Path:
    """
    Convertit un fichier de données financières en format compressé zstd.

    Args:
        input_path: Chemin du fichier d'entrée.
        output_path: Chemin du fichier de sortie. Si None, ajoute l'extension .zst.
        compression_level: Niveau de compression (1-22).

    Returns:
        Chemin du fichier compressé.
    """
    storage = CompressedStorage(compression_level=compression_level)
    return storage.compress_file(input_path, output_path)


def load_market_data_compressed(
    path: Union[str, Path], format: str = "parquet", cache_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Charge des données de marché à partir d'un fichier compressé.

    Args:
        path: Chemin du fichier compressé.
        format: Format des données ('parquet', 'csv', 'pickle').
        cache_dir: Répertoire de cache. Si None, utilise le répertoire temporaire.

    Returns:
        DataFrame des données de marché.
    """
    storage = CompressedStorage()
    return storage.load_dataframe(path, format=format) 