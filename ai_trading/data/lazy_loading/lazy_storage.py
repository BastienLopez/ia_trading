#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le chargement paresseux (lazy loading) des données financières.
Permet de charger efficacement de grands ensembles de données sans surcharger la mémoire.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérification des modules optionnels
HAVE_PYARROW = False
HAVE_HDF5 = False
HAVE_ZSTD = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    pass

try:
    import h5py
    HAVE_HDF5 = True
except ImportError:
    pass

try:
    import zstandard as zstd
    HAVE_ZSTD = True
except ImportError:
    pass


class LazyFileReader:
    """
    Classe pour lire des fichiers de manière paresseuse (lazy loading).
    Permet de charger des parties spécifiques de fichiers volumineux sans tout charger en mémoire.
    """
    
    def __init__(self, file_path: Union[str, Path], chunk_size: int = 10000, cache_size: int = 10):
        """
        Initialise le lecteur de fichier paresseux.
        
        Args:
            file_path: Chemin vers le fichier à lire.
            chunk_size: Nombre de lignes à lire par chunk.
            cache_size: Nombre de chunks à garder en cache.
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self._file_type = self._detect_file_type()
        self._file_length = self._get_file_length()
        self._chunk_cache = {}  # Cache pour les chunks de données
        self._cache_order = []  # Ordre LRU pour les chunks
        
    def _detect_file_type(self) -> str:
        """
        Détecte le type de fichier à partir de son extension.
        
        Returns:
            Type de fichier ('csv', 'parquet', 'hdf5', 'zstd', ou 'unknown').
        """
        suffix = self.file_path.suffix.lower()
        if suffix == '.csv':
            return 'csv'
        elif suffix == '.parquet':
            if not HAVE_PYARROW:
                raise ImportError("PyArrow est requis pour lire des fichiers Parquet. Installez-le avec: pip install pyarrow")
            return 'parquet'
        elif suffix in ['.h5', '.hdf5']:
            if not HAVE_HDF5:
                raise ImportError("h5py est requis pour lire des fichiers HDF5. Installez-le avec: pip install h5py")
            return 'hdf5'
        elif suffix == '.zst':
            if not HAVE_ZSTD:
                raise ImportError("zstandard est requis pour lire des fichiers compressés zstd. Installez-le avec: pip install zstandard")
            return 'zstd'
        else:
            logger.warning(f"Type de fichier inconnu: {suffix}")
            return 'unknown'
    
    def _get_file_length(self) -> int:
        """
        Détermine le nombre de lignes dans le fichier sans le charger entièrement.
        
        Returns:
            Nombre de lignes dans le fichier.
        """
        try:
            if self._file_type == 'csv':
                # Pour CSV, compter les lignes efficacement
                with open(self.file_path, 'r') as f:
                    # Soustraire 1 pour l'en-tête
                    return sum(1 for _ in f) - 1
            elif self._file_type == 'parquet':
                # Pour Parquet, utiliser les métadonnées
                return pq.read_metadata(self.file_path).num_rows
            elif self._file_type == 'hdf5':
                # Pour HDF5, ouvrir le fichier et vérifier la taille du premier dataset
                with h5py.File(self.file_path, 'r') as f:
                    # Utiliser le premier dataset ou groupe
                    key = list(f.keys())[0]
                    if isinstance(f[key], h5py.Dataset):
                        return len(f[key])
                    else:
                        # Si c'est un groupe, essayer de trouver un dataset
                        for sub_key in f[key].keys():
                            if isinstance(f[key][sub_key], h5py.Dataset):
                                return len(f[key][sub_key])
                        # Si aucun dataset n'est trouvé
                        return 0
            elif self._file_type == 'zstd':
                # Pour les fichiers zstd, décompresser et estimer
                # Ce n'est pas optimal mais fonctionne pour l'estimation
                decompressor = zstd.ZstdDecompressor()
                with open(self.file_path, 'rb') as f:
                    reader = decompressor.stream_reader(f)
                    # Lire les 100 premières lignes pour estimer
                    sample_lines = 0
                    sample_size = 0
                    for _ in range(100):
                        line = reader.read(1024)
                        if not line:
                            break
                        sample_lines += line.count(b'\n')
                        sample_size += len(line)
                    
                    if sample_lines > 0 and sample_size > 0:
                        # Estimer le nombre total de lignes
                        file_size = self.file_path.stat().st_size
                        decompression_ratio = 3  # Estimation conservatrice
                        estimated_size = file_size * decompression_ratio
                        return int((estimated_size / sample_size) * sample_lines)
                    else:
                        logger.warning("Impossible d'estimer la taille du fichier zstd")
                        return 1000  # Valeur par défaut
            else:
                # Type de fichier inconnu, estimation par défaut
                logger.warning(f"Impossible de déterminer la taille pour le type de fichier: {self._file_type}")
                return 1000  # Valeur par défaut
        except Exception as e:
            logger.error(f"Erreur lors de la détermination de la taille du fichier: {e}")
            return 1000  # Valeur par défaut
    
    @lru_cache(maxsize=32)
    def get_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """
        Récupère un chunk spécifique du fichier.
        
        Args:
            chunk_idx: Index du chunk à récupérer.
            
        Returns:
            DataFrame contenant les données du chunk.
        """
        # Vérifier si le chunk est déjà dans le cache
        if chunk_idx in self._chunk_cache:
            # Mettre à jour l'ordre du cache
            self._cache_order.remove(chunk_idx)
            self._cache_order.append(chunk_idx)
            return self._chunk_cache[chunk_idx]
        
        # Calculer les indices de début et de fin
        start_idx = chunk_idx * self.chunk_size
        end_idx = min((chunk_idx + 1) * self.chunk_size, self._file_length)
        
        # Charger le chunk selon le type de fichier
        chunk_data = None
        
        try:
            if self._file_type == 'csv':
                # Pour CSV, utiliser skiprows et nrows
                chunk_data = pd.read_csv(
                    self.file_path,
                    skiprows=range(1, start_idx + 1),  # +1 pour l'en-tête
                    nrows=end_idx - start_idx
                )
            elif self._file_type == 'parquet':
                # Pour Parquet, utiliser pyarrow pour une lecture partielle
                table = pq.read_table(
                    self.file_path, 
                    row_groups=[chunk_idx % pq.read_metadata(self.file_path).num_row_groups]
                )
                chunk_data = table.to_pandas()
            elif self._file_type == 'hdf5':
                # Pour HDF5, utiliser des slices
                with h5py.File(self.file_path, 'r') as f:
                    key = list(f.keys())[0]
                    if isinstance(f[key], h5py.Dataset):
                        dataset = f[key]
                        chunk_data = pd.DataFrame(dataset[start_idx:end_idx])
                    else:
                        # Si c'est un groupe, trouver un dataset
                        for sub_key in f[key].keys():
                            if isinstance(f[key][sub_key], h5py.Dataset):
                                dataset = f[key][sub_key]
                                chunk_data = pd.DataFrame(dataset[start_idx:end_idx])
                                break
            elif self._file_type == 'zstd':
                # Pour zstd, décompresser d'abord
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                try:
                    # Décompresser dans un fichier temporaire
                    with open(self.file_path, 'rb') as f_in:
                        with open(temp_file.name, 'wb') as f_out:
                            decompressor = zstd.ZstdDecompressor()
                            decompressor.copy_stream(f_in, f_out)
                    
                    # Lire le fichier temporaire
                    chunk_data = pd.read_csv(
                        temp_file.name,
                        skiprows=range(1, start_idx + 1),  # +1 pour l'en-tête
                        nrows=end_idx - start_idx
                    )
                finally:
                    # Supprimer le fichier temporaire
                    os.unlink(temp_file.name)
            else:
                raise ValueError(f"Type de fichier non pris en charge: {self._file_type}")
                
            # Mettre en cache le chunk
            self._chunk_cache[chunk_idx] = chunk_data
            self._cache_order.append(chunk_idx)
            
            # Gérer la taille du cache
            while len(self._cache_order) > self.cache_size:
                oldest_chunk = self._cache_order.pop(0)
                del self._chunk_cache[oldest_chunk]
                
            return chunk_data
        
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du chunk {chunk_idx}: {e}")
            # Retourner un DataFrame vide en cas d'erreur
            return pd.DataFrame()
    
    def get_row(self, row_idx: int) -> pd.Series:
        """
        Récupère une ligne spécifique du fichier.
        
        Args:
            row_idx: Index de la ligne à récupérer.
            
        Returns:
            Série pandas contenant les données de la ligne.
        """
        if row_idx < 0 or row_idx >= self._file_length:
            raise IndexError(f"Index de ligne {row_idx} hors limites [0, {self._file_length})")
        
        # Déterminer le chunk contenant cette ligne
        chunk_idx = row_idx // self.chunk_size
        local_idx = row_idx % self.chunk_size
        
        # Récupérer le chunk
        chunk = self.get_chunk(chunk_idx)
        
        # Retourner la ligne
        if local_idx < len(chunk):
            return chunk.iloc[local_idx]
        else:
            raise IndexError(f"Index local {local_idx} hors limites pour le chunk {chunk_idx}")
    
    def get_rows(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Récupère une plage de lignes du fichier.
        
        Args:
            start_idx: Index de début.
            end_idx: Index de fin (exclusif).
            
        Returns:
            DataFrame contenant les lignes demandées.
        """
        if start_idx < 0 or start_idx >= self._file_length:
            raise IndexError(f"Index de début {start_idx} hors limites [0, {self._file_length})")
        if end_idx <= start_idx or end_idx > self._file_length:
            end_idx = min(end_idx, self._file_length)
            if end_idx <= start_idx:
                return pd.DataFrame()
        
        # Déterminer les chunks nécessaires
        start_chunk = start_idx // self.chunk_size
        end_chunk = (end_idx - 1) // self.chunk_size
        
        if start_chunk == end_chunk:
            # Si tout est dans un seul chunk
            chunk = self.get_chunk(start_chunk)
            local_start = start_idx % self.chunk_size
            local_end = (end_idx - 1) % self.chunk_size + 1
            return chunk.iloc[local_start:local_end]
        else:
            # Si les données sont réparties sur plusieurs chunks
            chunks = []
            
            # Premier chunk (partiel)
            first_chunk = self.get_chunk(start_chunk)
            local_start = start_idx % self.chunk_size
            chunks.append(first_chunk.iloc[local_start:])
            
            # Chunks intermédiaires (complets)
            for chunk_idx in range(start_chunk + 1, end_chunk):
                chunks.append(self.get_chunk(chunk_idx))
            
            # Dernier chunk (partiel)
            last_chunk = self.get_chunk(end_chunk)
            local_end = (end_idx - 1) % self.chunk_size + 1
            chunks.append(last_chunk.iloc[:local_end])
            
            # Concaténer tous les chunks
            return pd.concat(chunks, ignore_index=True)
    
    def get_length(self) -> int:
        """
        Retourne le nombre total de lignes dans le fichier.
        
        Returns:
            Nombre de lignes.
        """
        return self._file_length
    
    def get_column_names(self) -> List[str]:
        """
        Retourne les noms des colonnes du fichier.
        
        Returns:
            Liste des noms de colonnes.
        """
        # Lire la première ligne pour obtenir les noms de colonnes
        if self._file_type == 'csv':
            return pd.read_csv(self.file_path, nrows=0).columns.tolist()
        elif self._file_type == 'parquet':
            return pq.read_schema(self.file_path).names
        elif self._file_type == 'hdf5':
            with h5py.File(self.file_path, 'r') as f:
                key = list(f.keys())[0]
                if isinstance(f[key], h5py.Dataset):
                    # Pour un dataset simple, utiliser les attributs
                    if hasattr(f[key], 'dtype') and hasattr(f[key].dtype, 'names'):
                        return list(f[key].dtype.names)
                    else:
                        # Sinon, utiliser des indices génériques
                        return [f"col_{i}" for i in range(f[key].shape[1])]
                else:
                    # Pour un groupe, chercher dans le premier dataset
                    for sub_key in f[key].keys():
                        if isinstance(f[key][sub_key], h5py.Dataset):
                            if hasattr(f[key][sub_key], 'dtype') and hasattr(f[key][sub_key].dtype, 'names'):
                                return list(f[key][sub_key].dtype.names)
                            else:
                                return [f"col_{i}" for i in range(f[key][sub_key].shape[1])]
                    return []
        elif self._file_type == 'zstd':
            # Pour zstd, décompresser temporairement
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            try:
                # Décompresser dans un fichier temporaire (juste assez pour l'en-tête)
                with open(self.file_path, 'rb') as f_in:
                    decompressor = zstd.ZstdDecompressor()
                    reader = decompressor.stream_reader(f_in)
                    with open(temp_file.name, 'wb') as f_out:
                        # Lire juste assez pour obtenir l'en-tête
                        chunk = reader.read(4096)
                        f_out.write(chunk)
                
                # Lire l'en-tête
                return pd.read_csv(temp_file.name, nrows=0).columns.tolist()
            finally:
                # Supprimer le fichier temporaire
                os.unlink(temp_file.name)
        else:
            logger.warning(f"Type de fichier non pris en charge pour get_column_names: {self._file_type}")
            return []


class LazyDataset(Dataset):
    """
    Dataset qui charge les données paresseusement depuis un fichier.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        sequence_length: int = 50,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        predict_n_ahead: int = 1,
        chunk_size: int = 10000,
        cache_size: int = 10,
    ):
        """
        Initialise le dataset paresseux.
        
        Args:
            file_path: Chemin vers le fichier de données.
            sequence_length: Longueur des séquences à extraire.
            target_column: Nom de la colonne cible.
            feature_columns: Liste des colonnes à utiliser comme features.
            transform: Fonction de transformation pour les séquences.
            target_transform: Fonction de transformation pour les cibles.
            predict_n_ahead: Nombre de pas de temps à prédire dans le futur.
            chunk_size: Taille des chunks à charger.
            cache_size: Nombre de chunks à garder en cache.
        """
        self.file_path = Path(file_path)
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.transform = transform
        self.target_transform = target_transform
        self.predict_n_ahead = predict_n_ahead
        
        # Initialiser le lecteur de fichier paresseux
        self.reader = LazyFileReader(
            file_path=file_path,
            chunk_size=chunk_size,
            cache_size=cache_size
        )
        
        # Longueur totale du dataset
        self.data_length = self.reader.get_length()
        
        # Nombre total d'exemples (séquences) dans le dataset
        if self.data_length > self.sequence_length + self.predict_n_ahead - 1:
            self.num_examples = self.data_length - self.sequence_length - self.predict_n_ahead + 1
        else:
            self.num_examples = 0
            logger.warning(
                f"Impossible de créer des séquences de longueur {self.sequence_length} "
                f"avec {self.predict_n_ahead} pas d'avance à partir de données de longueur {self.data_length}"
            )
        
        # Récupérer les noms des colonnes
        all_columns = self.reader.get_column_names()
        
        # Déterminer les indices des features et de la cible
        if self.feature_columns is None:
            # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes sauf la cible
            self.feature_indices = list(range(len(all_columns)))
            if self.target_column is not None and self.target_column in all_columns:
                target_idx = all_columns.index(self.target_column)
                self.feature_indices.remove(target_idx)
                self.target_idx = target_idx
            else:
                self.target_idx = None
        else:
            # Utiliser les colonnes spécifiées
            self.feature_indices = [all_columns.index(col) for col in self.feature_columns if col in all_columns]
            if self.target_column is not None and self.target_column in all_columns:
                self.target_idx = all_columns.index(self.target_column)
            else:
                self.target_idx = None
    
    def __len__(self) -> int:
        """
        Retourne le nombre d'exemples dans le dataset.
        
        Returns:
            Nombre d'exemples.
        """
        return self.num_examples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Récupère une séquence et sa cible correspondante.
        
        Args:
            idx: Index de la séquence.
            
        Returns:
            Tuple (séquence, cible).
        """
        if idx < 0 or idx >= self.num_examples:
            raise IndexError(f"Index {idx} hors limites [0, {self.num_examples})")
        
        # Calculer les indices pour la séquence
        start_idx = idx
        end_idx = idx + self.sequence_length
        
        # Récupérer la séquence
        sequence_data = self.reader.get_rows(start_idx, end_idx)
        
        # Extraire les features
        if self.feature_indices:
            # Utiliser iloc si feature_indices contient des entiers
            if all(isinstance(i, int) for i in self.feature_indices):
                features = sequence_data.iloc[:, self.feature_indices].values
            else:
                # Sinon utiliser les noms de colonnes
                features = sequence_data[self.feature_indices].values
        else:
            features = sequence_data.values
        
        # Convertir en tenseur
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Appliquer la transformation si spécifiée
        if self.transform:
            features_tensor = self.transform(features_tensor)
        
        # Récupérer la cible
        target_tensor = None
        if self.target_idx is not None:
            target_idx = end_idx - 1 + self.predict_n_ahead
            if target_idx < self.data_length:
                target_row = self.reader.get_row(target_idx)
                target_value = target_row.iloc[self.target_idx] if hasattr(target_row, 'iloc') else target_row[self.target_idx]
                target_tensor = torch.tensor(target_value, dtype=torch.float32)
                
                # Appliquer la transformation si spécifiée
                if self.target_transform:
                    target_tensor = self.target_transform(target_tensor)
        
        return features_tensor, target_tensor


def get_lazy_dataloader(
    file_path: Union[str, Path],
    sequence_length: int = 50,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    predict_n_ahead: int = 1,
    chunk_size: int = 10000,
    cache_size: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """
    Crée un DataLoader optimisé avec chargement paresseux des données.
    
    Args:
        file_path: Chemin vers le fichier de données.
        sequence_length: Longueur des séquences.
        target_column: Nom de la colonne cible.
        feature_columns: Liste des colonnes à utiliser comme features.
        transform: Fonction de transformation pour les séquences.
        target_transform: Fonction de transformation pour les cibles.
        predict_n_ahead: Nombre de pas de temps à prédire dans le futur.
        chunk_size: Taille des chunks à charger.
        cache_size: Nombre de chunks à garder en cache.
        batch_size: Taille des batchs.
        shuffle: Si True, mélange les données.
        num_workers: Nombre de workers pour le chargement parallèle.
        prefetch_factor: Nombre de batchs à précharger par worker.
        pin_memory: Si True, utilise la mémoire paginée pour un transfert plus rapide vers le GPU.
        **dataloader_kwargs: Arguments supplémentaires pour le DataLoader.
        
    Returns:
        DataLoader optimisé.
    """
    # Créer le dataset
    dataset = LazyDataset(
        file_path=file_path,
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        transform=transform,
        target_transform=target_transform,
        predict_n_ahead=predict_n_ahead,
        chunk_size=chunk_size,
        cache_size=cache_size
    )
    
    # Créer le DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        **dataloader_kwargs
    ) 