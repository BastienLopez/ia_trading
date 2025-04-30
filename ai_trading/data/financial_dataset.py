"""
Module fournissant des implémentations optimisées de Dataset et DataLoader pour les données financières.
Ces classes permettent un chargement efficace et prétraitement des données pour les modèles d'apprentissage.
"""

import gc
import logging
import os
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Configuration du logger
logger = logging.getLogger(__name__)

# Vérifiation de l'existence des modules optionnels
HAVE_PYARROW = False
HAVE_HDF5 = False
HAVE_LRU_CACHE = False

try:
    import pyarrow.parquet as pq

    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False

try:
    import h5py

    HAVE_HDF5 = True
except ImportError:
    HAVE_HDF5 = False

try:
    HAVE_LRU_CACHE = True
except ImportError:
    HAVE_LRU_CACHE = False


def optimize_memory(force_cuda_empty: bool = False) -> Tuple[float, Optional[float]]:
    """
    Optimise l'utilisation de la mémoire en forçant le garbage collection
    et en vidant le cache CUDA si nécessaire.

    Args:
        force_cuda_empty: Si True, force le vidage du cache CUDA même si la mémoire
                          n'est pas critique

    Returns:
        Tuple contenant (mémoire RAM libérée en MB, mémoire CUDA libérée en MB ou None)
    """
    # Mesurer la mémoire avant gc
    import psutil

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Mesurer la mémoire CUDA si disponible
    cuda_mem_before = None
    if torch.cuda.is_available():
        cuda_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

    # Forcer le garbage collection Python
    collected = gc.collect()

    # Libérer le cache CUDA si demandé ou si nécessaire
    cuda_mem_freed = None
    if torch.cuda.is_available():
        # Vérifier si la mémoire CUDA est critique (>80% utilisée)
        max_mem_allocated = torch.cuda.max_memory_allocated()
        # Éviter la division par zéro
        if force_cuda_empty or (
            max_mem_allocated > 0
            and torch.cuda.memory_allocated() / max_mem_allocated > 0.8
        ):
            # Mesurer avant la libération
            before = torch.cuda.memory_allocated() / (1024 * 1024)

            # Libérer le cache
            torch.cuda.empty_cache()

            # Mesurer après la libération
            after = torch.cuda.memory_allocated() / (1024 * 1024)
            cuda_mem_freed = before - after

    # Mesurer la mémoire après gc
    mem_after = process.memory_info().rss / (1024 * 1024)
    ram_freed = mem_before - mem_after

    return ram_freed, cuda_mem_freed


class MemoryManager:
    """
    Gestionnaire de mémoire intelligent pour l'optimisation automatique
    de l'utilisation RAM/CUDA pendant les opérations de chargement de données.
    """

    def __init__(
        self,
        check_interval: int = 1000,
        ram_threshold: float = 0.8,
        cuda_threshold: float = 0.8,
        enabled: bool = True,
    ):
        """
        Initialise le gestionnaire de mémoire.

        Args:
            check_interval: Nombre d'accès entre les vérifications de mémoire
            ram_threshold: Seuil d'utilisation de la RAM (0.0-1.0) pour déclencher l'optimisation
            cuda_threshold: Seuil d'utilisation CUDA (0.0-1.0) pour déclencher l'optimisation
            enabled: Si False, désactive les optimisations automatiques
        """
        self.check_interval = check_interval
        self.ram_threshold = ram_threshold
        self.cuda_threshold = cuda_threshold
        self.enabled = enabled
        self.access_count = 0

        # Mesurer la mémoire disponible au démarrage
        import psutil

        self.total_ram = psutil.virtual_memory().total / (1024 * 1024)

        # Déterminer si CUDA est disponible
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.total_cuda = torch.cuda.get_device_properties(0).total_memory / (
                1024 * 1024
            )

    def check_and_optimize(self, force: bool = False) -> bool:
        """
        Vérifie l'utilisation de la mémoire et optimise si nécessaire.

        Args:
            force: Si True, force l'optimisation indépendamment des seuils

        Returns:
            True si l'optimisation a été effectuée, False sinon
        """
        if not self.enabled:
            return False

        # Incrémenter le compteur d'accès
        self.access_count += 1

        # Vérifier si on doit faire l'optimisation
        if not force and self.access_count % self.check_interval != 0:
            return False

        # Vérifier l'utilisation de la mémoire RAM
        import psutil

        mem_info = psutil.virtual_memory()
        ram_usage = mem_info.percent / 100.0

        # Vérifier l'utilisation de la mémoire CUDA
        cuda_usage = 0.0
        if self.cuda_available:
            allocated = torch.cuda.memory_allocated()
            max_mem = torch.cuda.max_memory_allocated()
            cuda_usage = allocated / max_mem if max_mem > 0 else 0.0

        # Optimiser si l'un des seuils est dépassé ou si forcé
        if (
            force
            or ram_usage > self.ram_threshold
            or (self.cuda_available and cuda_usage > self.cuda_threshold)
        ):
            force_cuda = force or (
                self.cuda_available and cuda_usage > self.cuda_threshold
            )
            mem_freed, cuda_freed = optimize_memory(force_cuda_empty=force_cuda)

            # Log les résultats si significatifs
            return mem_freed > 10 or (cuda_freed is not None and cuda_freed > 10)

        return False


class FinancialDataset(Dataset):
    """
    Dataset optimisé pour les données financières avec prise en charge de différents formats et conversions.
    Supporte la création de séquences temporelles et normalisations.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray, str, torch.Tensor],
        sequence_length: int = 50,
        target_column: Optional[str] = "close",
        feature_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        predict_n_ahead: int = 1,
        is_train: bool = True,
        device: str = "cpu",
        use_shared_memory: bool = True,
        dtype: torch.dtype = torch.float32,
        lazy_loading: bool = False,
        cache_size: int = 100,
        precompute_features: bool = False,
        chunk_size: Optional[int] = None,
        memory_optimize: bool = True,
    ):
        """
        Initialise le dataset financier avec prétraitement optimisé.

        Args:
            data: Données source (DataFrame, ndarray, chemin de fichier ou tensor)
            sequence_length: Longueur des séquences à générer
            target_column: Nom de la colonne contenant la valeur cible (pour DataFrame)
            feature_columns: Liste des colonnes à utiliser comme features (pour DataFrame)
            transform: Fonction de transformation des features
            target_transform: Fonction de transformation des cibles
            predict_n_ahead: Nombre de pas de temps à prédire dans le futur
            is_train: Si True, mode entraînement (utilise les targets), sinon mode prédiction
            device: Dispositif sur lequel charger les données ("cpu" ou "cuda")
            use_shared_memory: Si True, utilise la mémoire partagée pour les workers
            dtype: Type de données à utiliser pour les tenseurs
            lazy_loading: Si True, charge les données à la demande plutôt qu'en totalité au début
            cache_size: Taille du cache LRU pour les séquences (si lazy_loading=True)
            precompute_features: Si True, prétraite et met en cache certaines features coûteuses
            chunk_size: Taille des chunks pour le chargement paresseux (None=auto)
            memory_optimize: Si True, active la gestion automatique de la mémoire
        """
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.transform = transform
        self.target_transform = target_transform
        self.predict_n_ahead = predict_n_ahead
        self.is_train = is_train
        self.device = device
        self.use_shared_memory = use_shared_memory
        self.dtype = dtype
        self.lazy_loading = lazy_loading
        self.precompute_features = precompute_features

        # Initialisation des caches
        self.feature_cache = {}
        self.sequence_cache = {}
        self.cache_size = cache_size

        # Déterminer automatiquement la taille de chunk si non spécifiée
        # (Utiliser ~100MB par chunk comme règle empirique)
        self.chunk_size = chunk_size

        # Méta-informations sur les données source pour le lazy loading
        self.data_source = data
        self.data_info = self._init_data_source(data)

        # Initialiser le gestionnaire de mémoire si activé
        self.memory_manager = (
            MemoryManager(enabled=memory_optimize) if memory_optimize else None
        )

        # Si pas de chargement paresseux, charger les données immédiatement
        if not lazy_loading:
            self.features, self.targets = self._load_and_preprocess_data(data)

            # Si on utilise la mémoire partagée pour les workers, on la configure
            if self.use_shared_memory:
                # Vérifier si la mémoire partagée est supportée
                if hasattr(torch.multiprocessing, "get_context") and callable(
                    getattr(self.features, "share_memory_", None)
                ):
                    self.features = self.features.share_memory_()
                    if self.targets is not None:
                        self.targets = self.targets.share_memory_()
                else:
                    logger.warning(
                        "Mémoire partagée non supportée ou désactivée. "
                        "Les performances multiprocessus peuvent être affectées."
                    )
        else:
            # En mode lazy loading, on définit des variables qui seront utilisées à la demande
            self.features = None
            self.targets = None

            # Initialiser les chunks pour le lazy loading
            self._init_chunks()

            # Précharger les premiers chunks si spécifié
            if self.precompute_features:
                self._precompute_initial_chunks()

        # Calcul du nombre d'exemples
        if is_train:
            # Calcul du nombre d'exemples pour l'entraînement
            calculated_examples = (
                self.data_info["length"] - sequence_length - predict_n_ahead + 1
            )
            # S'assurer que le nombre est positif pour éviter les erreurs
            self._num_examples = max(calculated_examples, 0)
        else:
            # Calcul du nombre d'exemples pour l'inférence/test
            calculated_examples = self.data_info["length"] - sequence_length + 1
            # S'assurer que le nombre est positif pour éviter les erreurs
            self._num_examples = max(calculated_examples, 0)

        # Si le nombre d'exemples est 0, avertir l'utilisateur
        if self._num_examples <= 0:
            logger.warning(
                f"Dataset financier initialisé avec un nombre d'exemples insuffisant ou négatif. Cela peut être dû à une séquence trop longue ({sequence_length}) par rapport au nombre de données ({self.data_info['length']}). Utilisation d'une valeur par défaut."
            )
            # Valeur par défaut pour éviter les erreurs
            self._num_examples = 1

        logger.info(f"Dataset financier initialisé avec {self._num_examples} exemples")

        # Exécuter un garbage collection initial pour libérer la mémoire après initialisation
        if memory_optimize:
            optimize_memory(force_cuda_empty=False)

    def _init_data_source(
        self, data: Union[pd.DataFrame, np.ndarray, str, torch.Tensor]
    ) -> Dict:
        """
        Initialise les informations sur la source de données sans charger le contenu complet.

        Args:
            data: Source de données

        Returns:
            Dictionnaire avec métadonnées sur les données
        """
        info = {"type": None, "length": 0, "columns": None, "path": None}

        # Traitement selon le type de données
        if isinstance(data, pd.DataFrame):
            info["type"] = "dataframe"
            info["length"] = len(data)
            info["columns"] = list(data.columns)

        elif isinstance(data, np.ndarray):
            info["type"] = "ndarray"
            info["length"] = data.shape[0]
            info["columns"] = list(range(data.shape[1]))

        elif isinstance(data, torch.Tensor):
            info["type"] = "tensor"
            info["length"] = data.size(0)
            info["columns"] = list(range(data.size(1)))

        elif isinstance(data, str) and os.path.exists(data):
            info["type"] = "file"
            info["path"] = data
            ext = os.path.splitext(data)[1].lower()

            # Déterminer le nombre de lignes et colonnes sans charger le fichier entier
            if ext == ".csv":
                # Pour CSV, lire seulement l'en-tête pour récupérer les colonnes
                try:
                    df_sample = pd.read_csv(data, nrows=5)
                    info["columns"] = list(df_sample.columns)

                    # Obtenir le nombre total de lignes
                    with open(data, "r") as f:
                        info["length"] = sum(1 for _ in f) - 1  # -1 pour l'en-tête
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture du fichier CSV: {e}")
                    # Valeur par défaut
                    info["length"] = 1000

            elif ext == ".parquet":
                if HAVE_PYARROW:
                    try:
                        parquet_file = pq.ParquetFile(data)
                        info["length"] = parquet_file.metadata.num_rows
                        info["columns"] = parquet_file.schema.names
                    except Exception as e:
                        logger.warning(
                            f"Erreur lors de la lecture du fichier Parquet: {e}"
                        )
                        info["length"] = 1000
                else:
                    try:
                        # Utiliser pandas sans charger toutes les données
                        df_sample = pd.read_parquet(data, engine="auto")
                        info["columns"] = list(df_sample.columns)
                        info["length"] = len(df_sample)
                    except Exception as e:
                        logger.warning(
                            f"Erreur lors de la lecture du fichier Parquet: {e}"
                        )
                        info["length"] = 1000

            elif ext in [".h5", ".hdf5"]:
                if HAVE_HDF5:
                    try:
                        with h5py.File(data, "r") as f:
                            # H5 peut avoir plusieurs datasets
                            dset_name = list(f.keys())[0]
                            dset = f[dset_name]
                            info["length"] = dset.shape[0]
                            info["columns"] = list(range(dset.shape[1]))
                    except Exception as e:
                        logger.warning(
                            f"Erreur lors de la lecture du fichier HDF5: {e}"
                        )
                        info["length"] = 1000
                else:
                    try:
                        # Utiliser pandas sans charger toutes les données
                        df_sample = pd.read_hdf(data, start=0, stop=5)
                        info["columns"] = list(df_sample.columns)
                        info["length"] = len(df_sample)
                    except Exception as e:
                        logger.warning(
                            f"Erreur lors de la lecture du fichier HDF5: {e}"
                        )
                        info["length"] = 1000

        return info

    def _init_chunks(self):
        """Initialise les chunks pour le chargement paresseux."""
        if not self.lazy_loading:
            return

        # Déterminer la taille des chunks si non spécifiée
        if self.chunk_size is None:
            # Règle empirique: ~100k points par chunk
            self.chunk_size = min(100000, self.data_info["length"])

        # Calculer le nombre de chunks nécessaires
        self.num_chunks = (
            self.data_info["length"] + self.chunk_size - 1
        ) // self.chunk_size

        # Créer des informations sur les chunks
        self.chunks = []
        for i in range(self.num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.data_info["length"])
            self.chunks.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "loaded": False,
                    "features": None,
                    "targets": None,
                }
            )

        logger.info(
            f"Données divisées en {self.num_chunks} chunks de taille ~{self.chunk_size}"
        )

    def _precompute_initial_chunks(self):
        """Précharge les premiers chunks de données pour accès rapide."""
        if not self.lazy_loading or not self.precompute_features:
            return

        # Précharger le premier chunk
        self._load_chunk(0)

        # Si on a plusieurs chunks et assez de mémoire, précharger aussi le deuxième
        if self.num_chunks > 1 and self.chunks[0]["features"] is not None:
            # Estimer la taille mémoire du premier chunk
            chunk_mem = (
                self.chunks[0]["features"].element_size()
                * self.chunks[0]["features"].nelement()
            )
            # Si moins de ~500MB, charger le deuxième chunk aussi
            if chunk_mem < 500 * 1024 * 1024:  # 500MB
                self._load_chunk(1)

    def _load_chunk(self, chunk_idx: int):
        """
        Charge un chunk spécifique de données.

        Args:
            chunk_idx: Index du chunk à charger
        """
        if chunk_idx >= len(self.chunks) or self.chunks[chunk_idx]["loaded"]:
            return

        chunk = self.chunks[chunk_idx]
        start_idx = chunk["start_idx"]
        end_idx = chunk["end_idx"]

        if self.data_info["type"] == "file":
            # Charger le chunk depuis le fichier
            path = self.data_info["path"]
            ext = os.path.splitext(path)[1].lower()

            try:
                if ext == ".csv":
                    # Utiliser skiprows et nrows pour sélectionner seulement le chunk
                    df_chunk = pd.read_csv(
                        path,
                        skiprows=range(1, start_idx + 1),
                        nrows=end_idx - start_idx,
                    )

                elif ext == ".parquet":
                    if HAVE_PYARROW:
                        table = pq.read_table(path, start_idx, end_idx - start_idx)
                        df_chunk = table.to_pandas()
                    else:
                        df_chunk = pd.read_parquet(path, engine="auto")
                        df_chunk = df_chunk.iloc[start_idx:end_idx]

                elif ext in [".h5", ".hdf5"]:
                    if HAVE_HDF5:
                        with h5py.File(path, "r") as f:
                            dset_name = list(f.keys())[0]
                            dset = f[dset_name]
                            df_chunk = pd.DataFrame(dset[start_idx:end_idx])
                    else:
                        df_chunk = pd.read_hdf(path, start=start_idx, stop=end_idx)

                # Traiter le DataFrame comme dans _load_and_preprocess_data
                if self.feature_columns is not None:
                    features_df = df_chunk[self.feature_columns]
                else:
                    numeric_cols = df_chunk.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    if self.target_column in numeric_cols and self.is_train:
                        numeric_cols.remove(self.target_column)
                    features_df = df_chunk[numeric_cols]

                features = torch.tensor(features_df.values, dtype=self.dtype)

                # Charger les cibles si nécessaire
                targets = None
                if self.is_train and self.target_column is not None:
                    if self.target_column in df_chunk.columns:
                        targets = torch.tensor(
                            df_chunk[self.target_column].values, dtype=self.dtype
                        )

                # Stocker dans le chunk
                self.chunks[chunk_idx]["features"] = features
                self.chunks[chunk_idx]["targets"] = targets
                self.chunks[chunk_idx]["loaded"] = True

            except Exception as e:
                logger.error(f"Erreur lors du chargement du chunk {chunk_idx}: {e}")

        elif self.data_info["type"] in ["dataframe", "ndarray", "tensor"]:
            # Pour les données déjà en mémoire, on extrait simplement le sous-ensemble
            data_subset = (
                self.data_source.iloc[start_idx:end_idx]
                if self.data_info["type"] == "dataframe"
                else self.data_source[start_idx:end_idx]
            )
            features, targets = self._load_and_preprocess_data(data_subset)

            # Stocker dans le chunk
            self.chunks[chunk_idx]["features"] = features
            self.chunks[chunk_idx]["targets"] = targets
            self.chunks[chunk_idx]["loaded"] = True

    def _get_data_from_chunks(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Récupère les données d'un point spécifique depuis les chunks.

        Args:
            idx: Index global du point de données

        Returns:
            Tenseurs de features et cible pour l'index spécifié
        """
        # Déterminer quel chunk contient l'index
        chunk_idx = idx // self.chunk_size
        if chunk_idx >= len(self.chunks):
            raise IndexError(f"Index {idx} hors limites pour {len(self.chunks)} chunks")

        # Charger le chunk s'il n'est pas déjà chargé
        if not self.chunks[chunk_idx]["loaded"]:
            self._load_chunk(chunk_idx)

        # Calculer l'index relatif dans le chunk
        local_idx = idx - self.chunks[chunk_idx]["start_idx"]

        # Vérifier si nous avons assez de données pour la séquence
        if local_idx + self.sequence_length > len(self.chunks[chunk_idx]["features"]):
            # La séquence s'étend sur deux chunks
            start_features = self.chunks[chunk_idx]["features"][local_idx:]

            # Charger le chunk suivant si nécessaire et s'il existe
            remaining_seq_len = self.sequence_length - len(start_features)
            if chunk_idx + 1 < len(self.chunks):
                if not self.chunks[chunk_idx + 1]["loaded"]:
                    self._load_chunk(chunk_idx + 1)

                # Obtenir le reste de la séquence du chunk suivant
                if remaining_seq_len <= len(self.chunks[chunk_idx + 1]["features"]):
                    end_features = self.chunks[chunk_idx + 1]["features"][
                        :remaining_seq_len
                    ]
                    sequence = torch.cat([start_features, end_features], dim=0)
                else:
                    # Pas assez de données, remplir avec des zéros
                    padding = torch.zeros(
                        (
                            remaining_seq_len
                            - len(self.chunks[chunk_idx + 1]["features"]),
                        )
                        + self.chunks[chunk_idx + 1]["features"].shape[1:],
                        dtype=self.dtype,
                    )
                    end_features = torch.cat(
                        [self.chunks[chunk_idx + 1]["features"], padding], dim=0
                    )
                    sequence = torch.cat([start_features, end_features], dim=0)
            else:
                # Pas de chunk suivant, remplir avec des zéros
                padding = torch.zeros(
                    (remaining_seq_len,) + self.chunks[chunk_idx]["features"].shape[1:],
                    dtype=self.dtype,
                )
                sequence = torch.cat([start_features, padding], dim=0)
        else:
            # La séquence est contenue dans un seul chunk
            sequence = self.chunks[chunk_idx]["features"][
                local_idx : local_idx + self.sequence_length
            ]

        # Pour la cible, on doit vérifier si elle est dans le même chunk ou le suivant
        target = None
        if self.is_train:
            target_idx = local_idx + self.sequence_length + self.predict_n_ahead - 1

            # Vérifier si la cible est dans le même chunk
            if target_idx < len(self.chunks[chunk_idx]["targets"]):
                target = self.chunks[chunk_idx]["targets"][target_idx]
            else:
                # La cible est dans le chunk suivant
                next_chunk_idx = chunk_idx + 1
                if next_chunk_idx < len(self.chunks):
                    if not self.chunks[next_chunk_idx]["loaded"]:
                        self._load_chunk(next_chunk_idx)

                    # Calculer l'index dans le chunk suivant
                    next_local_idx = target_idx - (
                        len(self.chunks[chunk_idx]["targets"])
                    )

                    # S'assurer que l'index est valide
                    if (
                        0
                        <= next_local_idx
                        < len(self.chunks[next_chunk_idx]["targets"])
                    ):
                        target = self.chunks[next_chunk_idx]["targets"][next_local_idx]
                    else:
                        # Index hors limites
                        target = torch.zeros(1, dtype=self.dtype)
                else:
                    # Pas de chunk suivant
                    target = torch.zeros(1, dtype=self.dtype)

        return sequence, target

    def _load_and_preprocess_data(
        self, data: Union[pd.DataFrame, np.ndarray, str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Charge et prétraite les données selon différents formats d'entrée.

        Args:
            data: Données à charger et prétraiter

        Returns:
            Tuple avec features et cibles (ou None pour les cibles en mode prédiction)
        """
        features = None
        targets = None

        # Chargement selon le type
        if isinstance(data, str) and os.path.exists(data):
            # Déterminer le format du fichier par l'extension
            ext = os.path.splitext(data)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(data, index_col=0, parse_dates=True)
            elif ext == ".parquet":
                if not HAVE_PYARROW:
                    logger.warning(
                        "PyArrow non disponible - essai avec pandas.read_parquet directement. "
                        "Pour de meilleures performances, installez pyarrow."
                    )
                df = pd.read_parquet(data)
            elif ext in [".h5", ".hdf5"]:
                if not HAVE_HDF5:
                    logger.warning(
                        "h5py non disponible - essai avec pandas.read_hdf directement. "
                        "Pour de meilleures performances, installez h5py."
                    )
                df = pd.read_hdf(data)
            else:
                raise ValueError(f"Format de fichier non supporté: {ext}")

            data = df  # Continuer le traitement comme un DataFrame

        # Traitement basé sur le type de données maintenant chargé
        if isinstance(data, pd.DataFrame):
            # Sélection des colonnes pour les features
            if self.feature_columns is not None:
                features_df = data[self.feature_columns]
            else:
                # Utilise toutes les colonnes numériques par défaut
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if self.target_column in numeric_cols and self.is_train:
                    numeric_cols.remove(self.target_column)
                features_df = data[numeric_cols]

            # Conversion en tenseur PyTorch de manière efficace
            features = torch.tensor(features_df.values, dtype=self.dtype)

            # Cibles si en mode entraînement
            if self.is_train and self.target_column is not None:
                if self.target_column in data.columns:
                    targets = torch.tensor(
                        data[self.target_column].values, dtype=self.dtype
                    )
                else:
                    raise ValueError(
                        f"Colonne cible {self.target_column} introuvable dans le DataFrame"
                    )

        elif isinstance(data, np.ndarray):
            # Conversion directe en tenseur
            features = torch.tensor(data, dtype=self.dtype)
            # Pour ndarray, on suppose que la dernière colonne est la cible si nécessaire
            if self.is_train and data.shape[1] > 1:
                targets = torch.tensor(data[:, -1], dtype=self.dtype)
                features = torch.tensor(data[:, :-1], dtype=self.dtype)

        elif isinstance(data, torch.Tensor):
            # Utilisation directe du tenseur
            features = data.to(dtype=self.dtype)
            # Pour tensor, même logique que ndarray
            if self.is_train and data.size(1) > 1:
                targets = data[:, -1].to(dtype=self.dtype)
                features = data[:, :-1].to(dtype=self.dtype)

        else:
            raise TypeError(f"Type de données non supporté: {type(data)}")

        return features, targets

    def __len__(self) -> int:
        """Retourne le nombre d'exemples dans le dataset."""
        return self._num_examples

    # Utiliser lru_cache si disponible pour les séquences fréquemment accédées
    def _get_sequence_cached(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Récupère une séquence avec mise en cache LRU."""
        if idx in self.sequence_cache:
            return self.sequence_cache[idx]

        # Si le cache est plein, supprimer l'élément le moins récemment utilisé
        if len(self.sequence_cache) >= self.cache_size:
            # Simple LRU: supprimer le premier élément (le plus ancien)
            self.sequence_cache.pop(next(iter(self.sequence_cache)))

        # En mode lazy loading, récupérer depuis les chunks
        if self.lazy_loading:
            sequence, target = self._get_data_from_chunks(idx)
        else:
            # Extraire la séquence de features
            seq_start = idx
            seq_end = idx + self.sequence_length
            sequence = self.features[seq_start:seq_end]

            # En mode entrainement, récupérer la cible
            if self.is_train:
                target_idx = seq_end + self.predict_n_ahead - 1
                if target_idx < len(self.targets):
                    target = self.targets[target_idx]
                else:
                    # Cas où la cible serait hors limites
                    target = torch.zeros(1, dtype=self.dtype)
            else:
                target = None

        # Mettre en cache
        self.sequence_cache[idx] = (sequence, target)
        return sequence, target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Récupère un exemple (séquence + cible) à l'index spécifié.

        Args:
            idx: Index de l'exemple à récupérer

        Returns:
            Tuple contenant (séquence de features, valeur cible)
        """
        # Récupérer la séquence (avec cache si disponible)
        sequence, target = self._get_sequence_cached(idx)

        # Appliquer les transformations si définies
        if self.transform:
            sequence = self.transform(sequence)

        if self.target_transform and target is not None:
            target = self.target_transform(target)

        # Optimiser la mémoire périodiquement si le gestionnaire est activé
        if self.memory_manager:
            self.memory_manager.check_and_optimize()

        return sequence, target


# Fonction LRU cache pour le prétraitement des features
def get_feature_transform_fn(cache_size: int = 1000):
    """
    Crée une fonction de transformation de features avec mise en cache LRU.

    Args:
        cache_size: Taille du cache LRU

    Returns:
        Fonction décorateur pour mise en cache des transformations
    """
    if HAVE_LRU_CACHE:

        def feature_transform_decorator(transform_fn):
            # Cache des transformations, utilise l'ID du tenseur comme clé
            cache = {}

            def cached_transform(tensor):
                # Identifier le tenseur par son id et sa somme (hachage simple mais efficace)
                tensor_id = id(tensor)
                tensor_sum = float(tensor.sum())
                key = (tensor_id, tensor_sum)

                if key in cache:
                    # Réutiliser le résultat mis en cache
                    return cache[key]
                else:
                    # Calculer la transformation
                    result = transform_fn(tensor)

                    # Mettre à jour le cache, supprimer les anciens éléments si nécessaire
                    if len(cache) >= cache_size:
                        # Simple LRU: supprimer le premier élément (le plus ancien)
                        if cache:
                            cache.pop(next(iter(cache)))

                    # Stocker dans le cache
                    cache[key] = result
                    return result

            return cached_transform

        return feature_transform_decorator
    else:
        # Version simplifiée sans cache si lru_cache n'est pas disponible
        def feature_transform_decorator(transform_fn):
            return transform_fn

        return feature_transform_decorator


# Fonction d'initialisation des workers pour le DataLoader
def memory_optimized_worker_init(worker_id, original_worker_init_fn=None):
    """
    Initialise un worker avec des optimisations de mémoire.

    Args:
        worker_id: ID du worker
        original_worker_init_fn: Fonction d'initialisation utilisateur optionnelle
    """
    # Désactiver le GC automatique et le faire manuellement pour un meilleur contrôle
    gc.disable()
    gc.collect()  # Nettoyage initial

    # Réactiver le GC mais avec un seuil plus élevé pour moins de collections automatiques
    gc.enable()

    # Exécuter la fonction d'initialisation utilisateur si fournie
    if original_worker_init_fn:
        original_worker_init_fn(worker_id)


def get_financial_dataloader(
    dataset: FinancialDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    worker_init_fn: Optional[Callable] = None,
    optimize_memory: bool = True,
) -> DataLoader:
    """
    Crée un DataLoader optimisé pour les données financières.

    Args:
        dataset: Instance de FinancialDataset à utiliser
        batch_size: Taille des batchs
        shuffle: Si True, mélange les données à chaque époque
        num_workers: Nombre de processus pour le chargement parallèle
        prefetch_factor: Nombre de batchs à précharger par worker (None si num_workers=0)
        pin_memory: Si True, copie les tenseurs en mémoire page-locked pour transfert GPU plus rapide
        drop_last: Si True, élimine le dernier batch incomplet
        worker_init_fn: Fonction d'initialisation personnalisée pour les workers
        optimize_memory: Si True, active l'optimisation automatique de la mémoire

    Returns:
        DataLoader optimisé
    """
    # Vérifier si num_workers doit être ajusté selon le système
    if num_workers > 0 and os.name == "nt":  # Windows
        # Sur Windows, les processus sont plus coûteux en ressources
        num_workers = min(num_workers, os.cpu_count() or 4)

    # Définir une fonction partielle pour l'initialisation des workers qui utilise la fonction globale
    if optimize_memory:
        worker_init_fn_to_use = partial(
            memory_optimized_worker_init, original_worker_init_fn=worker_init_fn
        )
    else:
        worker_init_fn_to_use = worker_init_fn

    # Paramètres du DataLoader selon la version de PyTorch
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": drop_last,
        "worker_init_fn": worker_init_fn_to_use,
    }

    # Ajouter prefetch_factor et persistent_workers si la version de PyTorch le supporte
    # et seulement si num_workers > 0
    if (
        hasattr(DataLoader, "__init__")
        and "prefetch_factor" in DataLoader.__init__.__code__.co_varnames
        and num_workers > 0
        and prefetch_factor is not None
    ):
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    # Créer le DataLoader avec les options d'optimisation
    dataloader = DataLoader(dataset, **loader_kwargs)

    logger.info(
        f"DataLoader financier créé: batch_size={batch_size}, "
        f"num_workers={num_workers}"
    )

    return dataloader


class GarbageCollectionDataLoader(DataLoader):
    """
    DataLoader qui nettoie automatiquement la mémoire après chaque époque.

    Cette classe étend le DataLoader standard de PyTorch pour effectuer un garbage
    collection automatique et libérer le cache CUDA à la fin de chaque époque,
    afin d'optimiser l'utilisation de la mémoire pendant l'entraînement.
    """

    def __init__(
        self,
        dataset: Dataset,
        gc_frequency: int = 10,  # Fréquence de garbage collection (en nombre de batchs)
        cuda_empty_frequency: int = 100,  # Fréquence de vidage du cache CUDA
        *args,
        **kwargs,
    ):
        """
        Initialise le DataLoader avec garbage collection.

        Args:
            dataset: Dataset à utiliser
            gc_frequency: Fréquence de garbage collection (en nombre de batchs)
            cuda_empty_frequency: Fréquence de vidage du cache CUDA (en nombre de batchs)
            *args, **kwargs: Arguments passés au DataLoader parent
        """
        super().__init__(dataset, *args, **kwargs)
        self.gc_frequency = gc_frequency
        self.cuda_empty_frequency = cuda_empty_frequency
        self.batch_count = 0

    def __iter__(self):
        """Retourne un itérateur sur les données avec garbage collection périodique."""
        self.batch_count = 0
        for batch in super().__iter__():
            self.batch_count += 1

            # Appliquer le garbage collection périodiquement
            if self.gc_frequency > 0 and self.batch_count % self.gc_frequency == 0:
                gc.collect()

            # Vider le cache CUDA périodiquement si disponible
            if (
                torch.cuda.is_available()
                and self.cuda_empty_frequency > 0
                and self.batch_count % self.cuda_empty_frequency == 0
            ):
                torch.cuda.empty_cache()

            yield batch

        # Nettoyer à la fin de l'époque
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
