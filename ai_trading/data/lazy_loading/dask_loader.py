#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le chargement paresseux avec Dask.
Fournit des fonctions et des classes pour charger et traiter efficacement de grands ensembles de données.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import dask
    import dask.dataframe as dd
    from dask.delayed import delayed
    HAVE_DASK = True
except ImportError:
    logger.error("Dask n'est pas installé. Veuillez l'installer avec: pip install 'dask[complete]'")
    HAVE_DASK = False


def read_csv_lazy(
    path: Union[str, Path], 
    blocksize: Optional[int] = None,
    **kwargs
) -> "dd.DataFrame":
    """
    Lit un fichier CSV de manière paresseuse avec Dask.
    
    Args:
        path: Chemin vers le fichier CSV ou dossier contenant plusieurs CSV
        blocksize: Taille des blocs pour la lecture paresseuse (None = auto)
        **kwargs: Arguments supplémentaires passés à dd.read_csv
        
    Returns:
        DataFrame Dask contenant les données
    """
    if not HAVE_DASK:
        raise ImportError("Dask est requis pour utiliser cette fonction.")
    
    if isinstance(path, Path):
        path = str(path)
    
    # Déterminer si c'est un fichier ou un dossier
    if os.path.isdir(path):
        # Lire tous les CSV dans le dossier
        pattern = os.path.join(path, '*.csv')
        return dd.read_csv(pattern, blocksize=blocksize, **kwargs)
    else:
        # Lire un seul fichier
        return dd.read_csv(path, blocksize=blocksize, **kwargs)


def read_parquet_lazy(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
    **kwargs
) -> "dd.DataFrame":
    """
    Lit un fichier Parquet de manière paresseuse avec Dask.
    
    Args:
        path: Chemin vers le fichier ou dossier Parquet
        columns: Liste des colonnes à lire (None = toutes)
        filters: Filtres à appliquer lors de la lecture
        **kwargs: Arguments supplémentaires passés à dd.read_parquet
        
    Returns:
        DataFrame Dask contenant les données
    """
    if not HAVE_DASK:
        raise ImportError("Dask est requis pour utiliser cette fonction.")
    
    if isinstance(path, Path):
        path = str(path)
    
    return dd.read_parquet(path, columns=columns, filters=filters, **kwargs)


class DaskDataset(Dataset):
    """Dataset PyTorch basé sur un DataFrame Dask pour le chargement paresseux."""
    
    def __init__(
        self,
        ddf: "dd.DataFrame",
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        npartitions: Optional[int] = None,
        compute_meta: bool = True
    ):
        """
        Initialise le dataset Dask.
        
        Args:
            ddf: DataFrame Dask
            feature_columns: Liste des colonnes à utiliser comme features
            target_column: Nom de la colonne cible (None = pas de cible)
            transform: Fonction de transformation des features
            target_transform: Fonction de transformation de la cible
            npartitions: Nombre de partitions à utiliser (None = inchangé)
            compute_meta: Si True, calcule les métadonnées au démarrage
        """
        if not HAVE_DASK:
            raise ImportError("Dask est requis pour utiliser cette classe.")
        
        # Repartitionner si nécessaire
        if npartitions is not None and npartitions != ddf.npartitions:
            self.ddf = ddf.repartition(npartitions=npartitions)
        else:
            self.ddf = ddf
        
        # Colonnes de features et cible
        self.feature_columns = feature_columns or list(ddf.columns)
        if target_column and target_column in self.feature_columns:
            self.feature_columns.remove(target_column)
        self.target_column = target_column
        
        # Transformations
        self.transform = transform
        self.target_transform = target_transform
        
        # Calculer la longueur et les métadonnées si demandé
        if compute_meta:
            self._length = len(self.ddf)
            self._meta = self.ddf._meta
        else:
            self._length = None
            self._meta = None
    
    def __len__(self) -> int:
        """Retourne le nombre d'éléments dans le dataset."""
        if self._length is None:
            self._length = len(self.ddf)
        return self._length
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Récupère un élément du dataset.
        
        Args:
            idx: Index de l'élément à récupérer
            
        Returns:
            Tenseur de features ou tuple (features, target)
        """
        # Récupérer la ligne du dataframe
        partition_idx = self.ddf.partitions.npartitions_without_index
        if partition_idx is not None:
            # Trouver la partition qui contient cet index
            row = self.ddf.partitions[idx // partition_idx].iloc[idx % partition_idx].compute()
        else:
            # Fallback si nous ne connaissons pas la structure des partitions
            row = self.ddf.iloc[idx].compute()
        
        # Extraire les features
        features = row[self.feature_columns].values.astype(np.float32)
        features_tensor = torch.tensor(features)
        
        # Appliquer la transformation des features si spécifiée
        if self.transform:
            features_tensor = self.transform(features_tensor)
        
        # Si pas de cible, retourner seulement les features
        if self.target_column is None:
            return features_tensor
        
        # Extraire la cible
        target = row[self.target_column]
        target_tensor = torch.tensor(float(target))
        
        # Appliquer la transformation de la cible si spécifiée
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)
        
        return features_tensor, target_tensor


class DaskDataLoader(DataLoader):
    """DataLoader optimisé pour les datasets Dask."""
    
    def __init__(
        self,
        ddf: "dd.DataFrame",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        npartitions: Optional[int] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Initialise le DataLoader Dask.
        
        Args:
            ddf: DataFrame Dask
            batch_size: Taille des batchs
            shuffle: Si True, mélange les données
            num_workers: Nombre de workers pour le chargement
            feature_columns: Liste des colonnes de features
            target_column: Nom de la colonne cible
            transform: Fonction de transformation des features
            target_transform: Fonction de transformation de la cible
            npartitions: Nombre de partitions à utiliser
            pin_memory: Si True, utilise la mémoire épinglée
            drop_last: Si True, ignore le dernier batch incomplet
            **kwargs: Arguments supplémentaires pour DataLoader
        """
        if not HAVE_DASK:
            raise ImportError("Dask est requis pour utiliser cette classe.")
        
        # Optimiser le nombre de partitions en fonction de la taille de batch et du nombre de workers
        if npartitions is None:
            # Heuristique : 2-4 partitions par worker est souvent optimal
            optimal_partitions = max(1, num_workers * 2)
            npartitions = min(optimal_partitions, ddf.npartitions)
        
        # Créer le dataset
        dataset = DaskDataset(
            ddf=ddf,
            feature_columns=feature_columns,
            target_column=target_column,
            transform=transform,
            target_transform=target_transform,
            npartitions=npartitions
        )
        
        # Initialiser le DataLoader parent
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        ) 