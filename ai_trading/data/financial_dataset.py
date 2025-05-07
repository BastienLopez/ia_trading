"""
Module fournissant des implémentations optimisées de Dataset et DataLoader pour les données financières.
Ces classes permettent un chargement efficace et prétraitement des données pour les modèles d'apprentissage.
"""

import gc
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    HAVE_THREADING = True
except ImportError:
    HAVE_THREADING = False

# Configuration du logger
logger = logging.getLogger(__name__)

# Vérifiation de l'existence des modules optionnels
HAVE_PYARROW = False
HAVE_HDF5 = False
HAVE_LRU_CACHE = False
HAVE_THREADING_OPTIMIZER = False

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

try:
    from ai_trading.utils.threading_optimizer import ThreadingOptimizer

    HAVE_THREADING_OPTIMIZER = True
except ImportError:
    HAVE_THREADING_OPTIMIZER = False


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
    Dataset pour les données financières avec support pour le chargement paresseux.

    Prend en charge plusieurs formats de données source : DataFrame, ndarray, Tensor, fichier CSV, etc.
    Optimisé pour la mémoire avec chargement paresseux et mémoire partagée.
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
        async_prefetch: bool = False,  # Activer le pré-chargement asynchrone
        prefetch_num_chunks: int = 2,  # Nombre de chunks à précharger à l'avance
        max_prefetch_queue_size: int = 5,  # Taille maximale de la file d'attente de préchargement
    ):
        """
        Initialise le dataset.

        Args:
            data: Source de données (DataFrame, ndarray, Tensor, ou chemin de fichier)
            sequence_length: Longueur des séquences à générer
            target_column: Nom/index de la colonne cible pour la prédiction
            feature_columns: Liste des colonnes à utiliser comme features
            transform: Fonction de transformation des séquences
            target_transform: Fonction de transformation des cibles
            predict_n_ahead: Nombre de pas à prédire dans le futur
            is_train: Si True, inclut les cibles pour l'entraînement
            device: Périphérique pour les tenseurs ('cpu' ou 'cuda')
            use_shared_memory: Utiliser la mémoire partagée pour les tensors
            dtype: Type de données des tensors
            lazy_loading: Activer le chargement paresseux
            cache_size: Taille du cache pour les séquences
            precompute_features: Pré-calculer et mettre en cache les features
            chunk_size: Taille des chunks pour le lazy loading
            memory_optimize: Activer l'optimisation mémoire
            async_prefetch: Pré-chargement asynchrone des chunks
            prefetch_num_chunks: Nombre de chunks à précharger à l'avance
            max_prefetch_queue_size: Taille maximale de la file d'attente de préchargement
        """
        super().__init__()
        self.data_source = data
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
        self.memory_optimize = memory_optimize
        self.async_prefetch = async_prefetch
        self.prefetch_num_chunks = prefetch_num_chunks
        self.max_prefetch_queue_size = max_prefetch_queue_size

        # Taille du cache pour les séquences
        self._cache_size = cache_size
        self._sequence_cache = {}
        self._cache_keys = []
        self.precompute_features = precompute_features
        self.chunk_size = chunk_size

        # Nombre d'exemples dans le dataset
        self._num_examples = 0

        # Initialisation du gestionnaire de mémoire si demandé
        self.memory_manager = (
            MemoryManager(enabled=memory_optimize) if memory_optimize else None
        )

        # Initialisation des attributs de préchargement
        if self.async_prefetch:
            self._prefetch_lock = threading.Lock()
            self._prefetched_chunks = set()
            self._current_chunk_idx = 0
            self._prefetch_queue = queue.Queue(maxsize=max_prefetch_queue_size)
            self._stop_prefetch = threading.Event()
            self._thread_pool = ThreadPoolExecutor(max_workers=2)

        # Initialisation des données en fonction du mode (lazy ou complet)
        data_source = self._init_data_source(data)

        self.data_tensor = data_source.get("data_tensor")
        self.target_tensor = data_source.get("target_tensor")
        self.column_names = data_source.get("column_names")
        self.feature_indices = data_source.get("feature_indices")
        self.target_index = data_source.get("target_index")
        self.data_info = data_source  # Sauvegarder toutes les informations de la source

        if self.lazy_loading:
            # En mode chargement paresseux, on initialise les chunks et les métadonnées
            self._init_chunks()
            self._init_prefetch_system()
            # Calculer le nombre d'exemples en mode lazy loading
            data_length = self.data_info.get("length", 0)
            if data_length > self.sequence_length + self.predict_n_ahead - 1:
                self._num_examples = (
                    data_length - self.sequence_length - self.predict_n_ahead + 1
                )
            else:
                self._num_examples = 0
        else:
            # En mode chargement complet, on calcule le nombre d'exemples
            if self.data_tensor is not None:
                data_length = len(self.data_tensor)
                if data_length > self.sequence_length + self.predict_n_ahead - 1:
                    self._num_examples = (
                        data_length - self.sequence_length - self.predict_n_ahead + 1
                    )
                else:
                    self._num_examples = 0
            else:
                # Si aucune donnée n'est chargée, définir un nombre d'exemples par défaut
                self._num_examples = 0
                logger.warning(
                    "Aucune donnée valide n'a été chargée. Le dataset est vide."
                )

        # S'assurer que le nombre d'exemples est raisonnable
        if self._num_examples <= 0:
            logger.warning(
                f"Le nombre d'exemples calculé est invalide: {self._num_examples}. "
                f"Vérifiez la longueur des données ({self.data_info.get('length', 0)}) par rapport à "
                f"la longueur de séquence ({self.sequence_length}) et prediction_ahead ({self.predict_n_ahead})."
            )
            # Valeur par défaut pour éviter les erreurs
            self._num_examples = max(
                1, self.data_info.get("length", 1) - self.sequence_length
            )

    def _init_data_source(
        self, data: Union[pd.DataFrame, np.ndarray, str, torch.Tensor]
    ) -> Dict:
        """
        Initialise les informations sur la source de données et charge les données.

        Args:
            data: Source de données

        Returns:
            Dictionnaire avec métadonnées sur les données et les tenseurs chargés
        """
        info = {
            "type": None,
            "length": 0,
            "columns": None,
            "path": None,
            "data_tensor": None,
            "target_tensor": None,
        }

        # Traitement selon le type de données
        if isinstance(data, pd.DataFrame):
            info["type"] = "dataframe"
            info["length"] = len(data)
            info["columns"] = list(data.columns)

            # Créer les tenseurs de données et de cibles
            if not self.lazy_loading:
                # Extraire les features
                if self.feature_columns is not None:
                    feature_cols = self.feature_columns
                else:
                    feature_cols = list(data.columns)
                    if self.target_column in feature_cols:
                        feature_cols.remove(self.target_column)

                # Convertir en tenseurs
                feature_data = data[feature_cols].values
                info["data_tensor"] = torch.tensor(feature_data, dtype=self.dtype)

                # Extraire la colonne cible si spécifiée
                if self.target_column and self.target_column in data.columns:
                    target_data = data[self.target_column].values
                    info["target_tensor"] = torch.tensor(target_data, dtype=self.dtype)
                    info["target_index"] = data.columns.get_loc(self.target_column)

                # Stocker les indices des colonnes features
                info["feature_indices"] = [
                    data.columns.get_loc(col)
                    for col in feature_cols
                    if col in data.columns
                ]
                info["column_names"] = feature_cols

            # Stocker les informations même en mode lazy loading
            if self.feature_columns is not None:
                feature_cols = self.feature_columns
            else:
                feature_cols = list(data.columns)
                if self.target_column in feature_cols:
                    feature_cols.remove(self.target_column)

            info["feature_indices"] = [
                data.columns.get_loc(col) for col in feature_cols if col in data.columns
            ]
            info["column_names"] = feature_cols
            if self.target_column and self.target_column in data.columns:
                info["target_index"] = data.columns.get_loc(self.target_column)

        elif isinstance(data, np.ndarray):
            info["type"] = "ndarray"
            info["length"] = data.shape[0]

            # Nombre de colonnes
            n_cols = data.shape[1]
            info["columns"] = list(range(n_cols))

            if not self.lazy_loading:
                # Si les indices des colonnes de caractéristiques ne sont pas spécifiés, utiliser toutes les colonnes sauf la cible
                if self.feature_columns is not None:
                    feature_indices = [
                        int(col)
                        for col in self.feature_columns
                        if isinstance(col, (int, str)) and int(col) < n_cols
                    ]
                else:
                    # Si la colonne cible est un index, l'exclure des features
                    target_idx = (
                        int(self.target_column)
                        if isinstance(self.target_column, (int, str))
                        and self.target_column.isdigit()
                        else -1
                    )
                    feature_indices = [i for i in range(n_cols) if i != target_idx]

                # Extraire les données
                info["data_tensor"] = torch.tensor(
                    data[:, feature_indices], dtype=self.dtype
                )

                # Extraire la colonne cible si spécifiée
                if self.target_column is not None:
                    target_idx = (
                        int(self.target_column)
                        if isinstance(self.target_column, (int, str))
                        and (
                            isinstance(self.target_column, int)
                            or self.target_column.isdigit()
                        )
                        else -1
                    )
                    if 0 <= target_idx < n_cols:
                        info["target_tensor"] = torch.tensor(
                            data[:, target_idx], dtype=self.dtype
                        )
                        info["target_index"] = target_idx

                info["feature_indices"] = feature_indices
                info["column_names"] = [str(idx) for idx in feature_indices]

        elif isinstance(data, torch.Tensor):
            info["type"] = "tensor"
            info["length"] = data.size(0)

            # Pour un tenseur, on suppose que c'est déjà dans le bon format
            if data.dim() == 2:
                n_cols = data.size(1)
                info["columns"] = list(range(n_cols))

                if not self.lazy_loading:
                    # Si les indices des colonnes de caractéristiques ne sont pas spécifiés, utiliser toutes les colonnes sauf la cible
                    if self.feature_columns is not None:
                        feature_indices = [
                            int(col)
                            for col in self.feature_columns
                            if isinstance(col, (int, str)) and int(col) < n_cols
                        ]
                    else:
                        # Si la colonne cible est un index, l'exclure des features
                        target_idx = (
                            int(self.target_column)
                            if isinstance(self.target_column, (int, str))
                            and (
                                isinstance(self.target_column, int)
                                or str(self.target_column).isdigit()
                            )
                            else -1
                        )
                        feature_indices = [i for i in range(n_cols) if i != target_idx]

                    # Extraire les données
                    info["data_tensor"] = data[:, feature_indices].to(dtype=self.dtype)

                    # Extraire la colonne cible si spécifiée
                    if self.target_column is not None:
                        target_idx = (
                            int(self.target_column)
                            if isinstance(self.target_column, (int, str))
                            and (
                                isinstance(self.target_column, int)
                                or str(self.target_column).isdigit()
                            )
                            else -1
                        )
                        if 0 <= target_idx < n_cols:
                            info["target_tensor"] = data[:, target_idx].to(
                                dtype=self.dtype
                            )
                            info["target_index"] = target_idx

                    info["feature_indices"] = feature_indices
                    info["column_names"] = [str(idx) for idx in feature_indices]
            else:
                # Si le tenseur est unidimensionnel, on le considère comme une seule colonne
                info["columns"] = [0]

                if not self.lazy_loading:
                    info["data_tensor"] = data.view(-1, 1).to(dtype=self.dtype)
                    info["feature_indices"] = [0]
                    info["column_names"] = ["0"]

        elif isinstance(data, str) and os.path.exists(data):
            info["type"] = "file"
            info["path"] = data
            ext = os.path.splitext(data)[1].lower()

            # Charger les données depuis le fichier si nécessaire
            if not self.lazy_loading:
                if ext == ".csv":
                    try:
                        # Charger le CSV et convertir les colonnes object en float si possible
                        df = pd.read_csv(data)

                        # Convertir les colonnes object en numériques lorsque possible
                        for col in df.select_dtypes(include=["object"]).columns:
                            try:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                            except Exception:
                                # Si la conversion échoue, conserver comme string
                                pass

                        # Supprimer les colonnes non numériques qui ne peuvent pas être converties en tenseurs
                        numeric_cols = df.select_dtypes(
                            include=["number"]
                        ).columns.tolist()
                        if not numeric_cols:
                            logger.error(
                                f"Aucune colonne numérique trouvée dans le fichier CSV: {data}"
                            )
                            info["length"] = len(df)
                            return info

                        df = df[numeric_cols]

                        n_rows, n_cols = df.shape
                        info["length"] = n_rows
                        info["columns"] = list(df.columns)

                        # Extraire les features
                        if self.feature_columns is not None:
                            feature_cols = [
                                col for col in self.feature_columns if col in df.columns
                            ]
                        else:
                            feature_cols = list(df.columns)
                            if self.target_column in feature_cols:
                                feature_cols.remove(self.target_column)

                        # Convertir en tenseurs
                        feature_data = df[feature_cols].values
                        info["data_tensor"] = torch.tensor(
                            feature_data, dtype=self.dtype
                        )

                        # Extraire la colonne cible si spécifiée
                        if self.target_column and self.target_column in df.columns:
                            target_data = df[self.target_column].values
                            info["target_tensor"] = torch.tensor(
                                target_data, dtype=self.dtype
                            )
                            info["target_index"] = df.columns.get_loc(
                                self.target_column
                            )

                        # Stocker les indices des colonnes features
                        info["feature_indices"] = [
                            df.columns.get_loc(col) for col in feature_cols
                        ]
                        info["column_names"] = feature_cols
                    except Exception as e:
                        logger.error(f"Erreur lors du chargement du fichier CSV: {e}")
                        info["error"] = str(e)

                # Autres formats de fichier (parquet, hdf5, etc.)
                elif ext == ".parquet" and HAVE_PYARROW:
                    try:
                        # Charger les métadonnées Parquet
                        parquet_metadata = pq.read_metadata(data)

                        # Charger les données parquet
                        df = pq.read_table(data).to_pandas()

                        n_rows, n_cols = df.shape
                        info["length"] = n_rows
                        info["columns"] = list(df.columns)

                        # Extraire les features
                        if self.feature_columns is not None:
                            feature_cols = [
                                col for col in self.feature_columns if col in df.columns
                            ]
                        else:
                            feature_cols = list(df.columns)
                            if self.target_column in feature_cols:
                                feature_cols.remove(self.target_column)

                        # Convertir en tenseurs
                        feature_data = df[feature_cols].values
                        info["data_tensor"] = torch.tensor(
                            feature_data, dtype=self.dtype
                        )

                        # Extraire la colonne cible si spécifiée
                        if self.target_column and self.target_column in df.columns:
                            target_data = df[self.target_column].values
                            info["target_tensor"] = torch.tensor(
                                target_data, dtype=self.dtype
                            )
                            info["target_index"] = df.columns.get_loc(
                                self.target_column
                            )

                        # Stocker les indices des colonnes features
                        info["feature_indices"] = [
                            df.columns.get_loc(col) for col in feature_cols
                        ]
                        info["column_names"] = feature_cols
                    except Exception as e:
                        logger.error(
                            f"Erreur lors du chargement du fichier Parquet: {e}"
                        )
                        info["error"] = str(e)

                # Format HDF5
                elif ext in [".h5", ".hdf5"] and HAVE_HDF5:
                    try:
                        # Charger les données depuis le fichier HDF5
                        df = pd.read_hdf(data, key="data")

                        n_rows, n_cols = df.shape
                        info["length"] = n_rows
                        info["columns"] = list(df.columns)

                        # Extraire les features
                        if self.feature_columns is not None:
                            feature_cols = [
                                col for col in self.feature_columns if col in df.columns
                            ]
                        else:
                            feature_cols = list(df.columns)
                            if self.target_column in feature_cols:
                                feature_cols.remove(self.target_column)

                        # Convertir en tenseurs
                        feature_data = df[feature_cols].values
                        info["data_tensor"] = torch.tensor(
                            feature_data, dtype=self.dtype
                        )

                        # Extraire la colonne cible si spécifiée
                        if self.target_column and self.target_column in df.columns:
                            target_data = df[self.target_column].values
                            info["target_tensor"] = torch.tensor(
                                target_data, dtype=self.dtype
                            )
                            info["target_index"] = df.columns.get_loc(
                                self.target_column
                            )

                        # Stocker les indices des colonnes features
                        info["feature_indices"] = [
                            df.columns.get_loc(col) for col in feature_cols
                        ]
                        info["column_names"] = feature_cols
                    except Exception as e:
                        logger.error(f"Erreur lors du chargement du fichier HDF5: {e}")
                        info["error"] = str(e)
            else:
                # En mode lazy loading, on ne charge pas les données entièrement mais on récupère les métadonnées
                if ext == ".csv":
                    try:
                        # Lire les premières lignes pour obtenir la structure
                        df_sample = pd.read_csv(data, nrows=5)
                        info["columns"] = list(df_sample.columns)

                        # Obtenir le nombre total de lignes
                        with open(data, "r") as f:
                            info["length"] = sum(1 for _ in f) - 1  # -1 pour l'en-tête

                        # Déterminer les indices des colonnes features
                        if self.feature_columns is not None:
                            feature_cols = [
                                col
                                for col in self.feature_columns
                                if col in df_sample.columns
                            ]
                        else:
                            feature_cols = list(df_sample.columns)
                            if self.target_column in feature_cols:
                                feature_cols.remove(self.target_column)

                        info["feature_indices"] = [
                            df_sample.columns.get_loc(col)
                            for col in feature_cols
                            if col in df_sample.columns
                        ]
                        info["column_names"] = feature_cols

                        # Enregistrer l'indice de la colonne cible
                        if (
                            self.target_column
                            and self.target_column in df_sample.columns
                        ):
                            info["target_index"] = df_sample.columns.get_loc(
                                self.target_column
                            )
                    except Exception as e:
                        logger.warning(f"Erreur lors de la lecture du fichier CSV: {e}")
                        info["length"] = 1000  # Valeur par défaut

                elif ext == ".parquet" and HAVE_PYARROW:
                    try:
                        # Lire les métadonnées du fichier parquet
                        parquet_metadata = pq.read_metadata(data)
                        info["length"] = parquet_metadata.num_rows

                        # Lire l'en-tête pour obtenir les noms de colonnes (version simplifiée)
                        # Ne pas utiliser nrows avec read_table car ce n'est pas supporté par toutes les versions
                        df_sample = pq.read_table(data).to_pandas().head(5)
                        info["columns"] = list(df_sample.columns)

                        # Déterminer les indices des colonnes features
                        if self.feature_columns is not None:
                            feature_cols = [
                                col
                                for col in self.feature_columns
                                if col in df_sample.columns
                            ]
                        else:
                            feature_cols = list(df_sample.columns)
                            if self.target_column in feature_cols:
                                feature_cols.remove(self.target_column)

                        info["feature_indices"] = [
                            df_sample.columns.get_loc(col)
                            for col in feature_cols
                            if col in df_sample.columns
                        ]
                        info["column_names"] = feature_cols

                        # Enregistrer l'indice de la colonne cible
                        if (
                            self.target_column
                            and self.target_column in df_sample.columns
                        ):
                            info["target_index"] = df_sample.columns.get_loc(
                                self.target_column
                            )
                    except Exception as e:
                        logger.warning(
                            f"Erreur lors de la lecture des métadonnées Parquet: {e}"
                        )
                        info["length"] = 1000  # Valeur par défaut

                elif ext in [".h5", ".hdf5"] and HAVE_HDF5:
                    try:
                        # Lire les métadonnées du fichier HDF5
                        with pd.HDFStore(data, mode="r") as store:
                            # Vérifier si la clé 'data' existe
                            if "data" in store.keys():
                                info["length"] = store.get_storer("data").nrows
                                # Lire quelques lignes pour obtenir la structure
                                df_sample = pd.read_hdf(
                                    data, key="data", start=0, stop=5
                                )
                                info["columns"] = list(df_sample.columns)

                                # Déterminer les indices des colonnes features
                                if self.feature_columns is not None:
                                    feature_cols = [
                                        col
                                        for col in self.feature_columns
                                        if col in df_sample.columns
                                    ]
                                else:
                                    feature_cols = list(df_sample.columns)
                                    if self.target_column in feature_cols:
                                        feature_cols.remove(self.target_column)

                                info["feature_indices"] = [
                                    df_sample.columns.get_loc(col)
                                    for col in feature_cols
                                    if col in df_sample.columns
                                ]
                                info["column_names"] = feature_cols

                                # Enregistrer l'indice de la colonne cible
                                if (
                                    self.target_column
                                    and self.target_column in df_sample.columns
                                ):
                                    info["target_index"] = df_sample.columns.get_loc(
                                        self.target_column
                                    )
                            else:
                                logger.warning(
                                    f"Clé 'data' non trouvée dans le fichier HDF5: {data}"
                                )
                                info["length"] = 1000  # Valeur par défaut
                    except Exception as e:
                        logger.warning(
                            f"Erreur lors de la lecture des métadonnées HDF5: {e}"
                        )
                        info["length"] = 1000  # Valeur par défaut

        # Stocker les informations de colonnes pour référence future
        if "column_names" not in info and info["columns"] is not None:
            info["column_names"] = info["columns"]

        # Vérifier que les données ont été correctement chargées
        if not self.lazy_loading and "data_tensor" not in info:
            logger.warning("Les données n'ont pas été correctement chargées")

        return info

    def _init_chunks(self):
        """Initialise les chunks pour le lazy loading."""
        # Déterminer la taille du dataset
        if isinstance(self.data_source, pd.DataFrame):
            data_length = len(self.data_source)
        elif isinstance(self.data_source, np.ndarray):
            data_length = len(self.data_source)
        elif isinstance(self.data_source, torch.Tensor):
            data_length = len(self.data_source)
        elif isinstance(self.data_source, str) and os.path.exists(self.data_source):
            # Estimer la taille à partir du fichier ou autres métadonnées
            if self.data_info and "length" in self.data_info:
                data_length = self.data_info["length"]
            else:
                # Charger les métadonnées du fichier pour obtenir la taille
                ext = os.path.splitext(self.data_source)[1].lower()
                if ext == ".csv":
                    # Pour les fichiers CSV, compter les lignes rapidement
                    with open(self.data_source, "r") as f:
                        data_length = sum(1 for _ in f) - 1  # -1 pour l'en-tête
                elif ext == ".parquet" and HAVE_PYARROW:
                    # Pour les fichiers Parquet, utiliser les métadonnées
                    data_length = pq.read_metadata(self.data_source).num_rows
                elif ext in [".h5", ".hdf5"] and HAVE_HDF5:
                    # Pour les fichiers HDF5, utiliser h5py
                    with h5py.File(self.data_source, "r") as f:
                        dset_name = list(f.keys())[0]
                        data_length = len(f[dset_name])
                else:
                    # Fallback: charger le fichier
                    logger.warning(
                        f"Impossible de déterminer la taille du fichier {self.data_source} "
                        f"sans le charger entièrement. Cela peut prendre du temps."
                    )
                    try:
                        temp_data = pd.read_csv(self.data_source)
                        data_length = len(temp_data)
                        del temp_data  # Libérer la mémoire
                    except Exception as e:
                        logger.error(f"Erreur lors de la lecture du fichier: {e}")
                        # Valeur par défaut pour éviter une erreur
                        data_length = 1000
        else:
            # Valeur par défaut pour éviter une erreur avec source de données non supportée
            logger.warning(f"Source de données non supportée: {type(self.data_source)}")
            data_length = 1000

        # Si aucune taille de chunk spécifiée, déterminer automatiquement
        if self.chunk_size is None:
            # Utiliser une taille de chunk de ~100MB ou 5000 lignes par défaut
            self.chunk_size = min(5000, max(1000, data_length // 10))

        # Créer les indices de chunks
        self._chunk_indices = []
        self._chunk_boundaries = []
        self._loaded_chunks = {}

        # Diviser les données en chunks seulement si data_length > 0
        if data_length > 0:
            for i in range(0, data_length, self.chunk_size):
                start_idx = i
                end_idx = min(i + self.chunk_size, data_length)
                self._chunk_indices.append((start_idx, end_idx))
                # Calculer les limites pour l'indexation des séquences
                seq_start = max(0, i - self.sequence_length - self.predict_n_ahead + 1)
                if i > 0:
                    seq_start = i  # Pour éviter les chevauchements entre chunks
                seq_end = end_idx
                self._chunk_boundaries.append((seq_start, seq_end))

        # Stocker le nombre total de chunks
        self.num_chunks = len(self._chunk_indices)

        logger.info(
            f"Données divisées en {self.num_chunks} chunks de taille ~{self.chunk_size}"
        )

        # Mettre à jour les informations de données
        if self.data_info is None:
            self.data_info = {}
        self.data_info["length"] = data_length

    def _init_prefetch_system(self):
        """Initialise le système de préchargement asynchrone si activé."""
        if not self.async_prefetch:
            return

        # Vérifier si les attributs nécessaires ont été initialisés
        if not hasattr(self, "_prefetch_lock") or not hasattr(
            self, "_prefetched_chunks"
        ):
            # Initialiser les attributs de préchargement s'ils n'existent pas encore
            self._prefetch_lock = threading.Lock()
            self._prefetched_chunks = set()
            self._current_chunk_idx = 0
            self._prefetch_queue = queue.Queue(maxsize=self.max_prefetch_queue_size)
            self._stop_prefetch = threading.Event()
            self._thread_pool = ThreadPoolExecutor(max_workers=2)

        # Préchargement initial des premiers chunks
        for i in range(min(self.prefetch_num_chunks, len(self._chunk_indices))):
            self._prefetch_chunk(i)

        # Démarrer un thread de surveillance qui gère le préchargement continu
        def prefetch_monitor_thread():
            while not self._stop_prefetch.is_set():
                try:
                    # Attendre un court moment
                    self._stop_prefetch.wait(0.1)

                    # Vérifier si des chunks supplémentaires doivent être préchargés
                    with self._prefetch_lock:
                        for i in range(
                            self._current_chunk_idx,
                            min(
                                self._current_chunk_idx + self.prefetch_num_chunks,
                                len(self._chunk_indices),
                            ),
                        ):
                            if (
                                i not in self._prefetched_chunks
                                and not self._prefetch_queue.full()
                            ):
                                self._prefetch_chunk(i)
                except Exception as e:
                    logger.error(f"Erreur dans le thread de préchargement: {e}")

        # Démarrer le thread de surveillance
        self._prefetch_thread = threading.Thread(
            target=prefetch_monitor_thread, daemon=True
        )
        self._prefetch_thread.start()

    def _prefetch_chunk(self, chunk_idx):
        """Précharge un chunk de données de manière asynchrone.

        Args:
            chunk_idx: Index du chunk à précharger
        """
        if not self.async_prefetch or chunk_idx in self._prefetched_chunks:
            return

        with self._prefetch_lock:
            if chunk_idx in self._prefetched_chunks:
                return

            self._prefetched_chunks.add(chunk_idx)

        # Soumettre la tâche de chargement au thread pool
        future = self._thread_pool.submit(self._load_chunk, chunk_idx)

        # Callback pour ajouter le résultat à la file d'attente une fois chargé
        def done_callback(fut):
            try:
                chunk_result = fut.result()
                self._prefetch_queue.put((chunk_idx, chunk_result))
            except Exception as e:
                logger.error(f"Erreur lors du préchargement du chunk {chunk_idx}: {e}")
                with self._prefetch_lock:
                    if chunk_idx in self._prefetched_chunks:
                        self._prefetched_chunks.remove(chunk_idx)

        future.add_done_callback(done_callback)

    def _load_chunk(self, chunk_idx: int):
        """
        Charge un chunk de données en mémoire.

        Args:
            chunk_idx: Index du chunk à charger

        Returns:
            Les données et cibles du chunk chargé
        """
        start_idx, end_idx = self._chunk_indices[chunk_idx]

        # Charger et prétraiter les données du chunk
        if isinstance(self.data_source, pd.DataFrame):
            chunk_data = self.data_source.iloc[start_idx:end_idx]

            # Convertir les colonnes object/string en float si possible ou les supprimer
            for col in chunk_data.select_dtypes(include=["object"]).columns:
                try:
                    chunk_data[col] = pd.to_numeric(chunk_data[col], errors="coerce")
                except:
                    logger.warning(
                        f"Impossible de convertir la colonne {col} en numérique, elle sera supprimée"
                    )
                    chunk_data = chunk_data.drop(columns=[col])

            # S'assurer que toutes les données sont numériques
            chunk_data = chunk_data.select_dtypes(include=["number"])

            # Vérifier si feature_indices est spécifié
            if self.feature_indices is not None:
                # Extraire les colonnes d'intérêt en utilisant les indices numériques
                column_indices = [
                    i for i in self.feature_indices if i < chunk_data.shape[1]
                ]
                if column_indices:
                    feature_data = chunk_data.iloc[:, column_indices].values
                else:
                    feature_data = chunk_data.values
            else:
                feature_data = chunk_data.values

            # Convertir en tenseur PyTorch
            chunk_tensor = torch.tensor(feature_data, dtype=self.dtype)

            if self.target_column is not None and self.is_train:
                if isinstance(self.target_column, list):
                    # Vérifier si toutes les colonnes cibles existent
                    available_targets = [
                        col for col in self.target_column if col in chunk_data.columns
                    ]
                    if not available_targets:
                        logger.warning(
                            f"Aucune des colonnes cibles {self.target_column} n'existe dans les données"
                        )
                        chunk_targets = None
                    else:
                        # Cibles multiples
                        chunk_targets = torch.tensor(
                            chunk_data[available_targets].values, dtype=self.dtype
                        )
                else:
                    # Vérifier si la colonne cible existe
                    if (
                        isinstance(self.target_column, str)
                        and self.target_column not in chunk_data.columns
                    ):
                        logger.warning(
                            f"Colonne cible {self.target_column} introuvable dans les données"
                        )
                        chunk_targets = None
                    elif (
                        isinstance(self.target_column, int)
                        and self.target_column >= chunk_data.shape[1]
                    ):
                        logger.warning(
                            f"Indice cible {self.target_column} hors limites"
                        )
                        chunk_targets = None
                    else:
                        # Cible unique
                        if isinstance(self.target_column, str):
                            target_values = chunk_data[self.target_column].values
                        else:  # int
                            target_values = chunk_data.iloc[
                                :, self.target_column
                            ].values

                        chunk_targets = torch.tensor(target_values, dtype=self.dtype)
            else:
                chunk_targets = None

        elif isinstance(self.data_source, np.ndarray):
            # Vérifier si le tableau contient des types object
            chunk_data = self.data_source[start_idx:end_idx]

            # Traitement pour les tableaux contenant des objets
            if "object" in str(chunk_data.dtype):
                # Essayer de convertir les colonnes objet en numérique
                numeric_data = np.zeros(
                    (chunk_data.shape[0], chunk_data.shape[1]), dtype=np.float32
                )

                for i in range(chunk_data.shape[1]):
                    try:
                        numeric_data[:, i] = pd.to_numeric(
                            chunk_data[:, i], errors="coerce"
                        )
                    except:
                        # En cas d'échec, initialiser à NaN
                        numeric_data[:, i] = np.nan

                # Remplacer les NaN par des zéros
                numeric_data = np.nan_to_num(numeric_data, nan=0.0)
                chunk_data = numeric_data

            # Extraire les features si les indices sont spécifiés
            if self.feature_indices is not None:
                # S'assurer que les indices sont valides
                valid_indices = [
                    i for i in self.feature_indices if i < chunk_data.shape[1]
                ]
                if valid_indices:
                    feature_data = chunk_data[:, valid_indices]
                else:
                    feature_data = chunk_data
            else:
                feature_data = chunk_data

            # Convertir en tenseur PyTorch
            chunk_tensor = torch.tensor(feature_data, dtype=self.dtype)

            if self.target_index is not None and self.is_train:
                if isinstance(self.target_index, list):
                    # Cibles multiples
                    valid_indices = [
                        i for i in self.target_index if i < chunk_data.shape[1]
                    ]
                    if valid_indices:
                        chunk_targets = torch.tensor(
                            chunk_data[:, valid_indices], dtype=self.dtype
                        )
                    else:
                        chunk_targets = None
                else:
                    # Cible unique
                    if 0 <= self.target_index < chunk_data.shape[1]:
                        chunk_targets = torch.tensor(
                            chunk_data[:, self.target_index], dtype=self.dtype
                        )
                    else:
                        chunk_targets = None
            else:
                chunk_targets = None

        elif isinstance(self.data_source, torch.Tensor):
            # Traitement pour tenseur PyTorch
            chunk_data = self.data_source[start_idx:end_idx]

            # Extraire les features si les indices sont spécifiés
            if self.feature_indices is not None:
                # S'assurer que les indices sont valides pour ce tenseur
                valid_indices = [
                    i for i in self.feature_indices if i < chunk_data.shape[1]
                ]
                if valid_indices:
                    chunk_tensor = chunk_data[:, valid_indices].to(dtype=self.dtype)
                else:
                    chunk_tensor = chunk_data.to(dtype=self.dtype)
            else:
                chunk_tensor = chunk_data.to(dtype=self.dtype)

            if self.target_index is not None and self.is_train:
                if isinstance(self.target_index, list):
                    # Cibles multiples
                    valid_indices = [
                        i for i in self.target_index if i < chunk_data.shape[1]
                    ]
                    if valid_indices:
                        chunk_targets = chunk_data[:, valid_indices].to(
                            dtype=self.dtype
                        )
                    else:
                        chunk_targets = None
                else:
                    # Cible unique
                    if 0 <= self.target_index < chunk_data.shape[1]:
                        chunk_targets = chunk_data[:, self.target_index].to(
                            dtype=self.dtype
                        )
                    else:
                        chunk_targets = None
            else:
                chunk_targets = None

        elif isinstance(self.data_source, str) and os.path.exists(self.data_source):
            # Charger le chunk depuis un fichier
            ext = os.path.splitext(self.data_source)[1].lower()

            try:
                if ext == ".csv":
                    # Charger le chunk depuis un CSV
                    chunk_data = pd.read_csv(
                        self.data_source,
                        skiprows=range(1, start_idx + 1),
                        nrows=end_idx - start_idx,
                    )
                elif ext == ".parquet" and HAVE_PYARROW:
                    # Charger le chunk depuis un Parquet
                    try:
                        # Utiliser read_table pour lire le fichier parquet complet puis sélectionner le chunk
                        table = pq.read_table(self.data_source)
                        # Convertir en DataFrame et sélectionner la plage d'indices
                        chunk_data = table.to_pandas().iloc[start_idx:end_idx]
                    except Exception as e:
                        logger.error(
                            f"Erreur lors de la lecture du fichier Parquet: {e}"
                        )
                        # Fallback: créer un DataFrame vide comme fallback
                        chunk_data = pd.DataFrame()
                elif ext in [".h5", ".hdf5"] and HAVE_HDF5:
                    # Charger le chunk depuis un HDF5
                    with h5py.File(self.data_source, "r") as f:
                        dset_name = list(f.keys())[0]
                        chunk_data = pd.DataFrame(f[dset_name][start_idx:end_idx])
                else:
                    raise ValueError(f"Format de fichier non supporté: {ext}")

                # Convertir les colonnes object/string en float si possible ou les supprimer
                for col in chunk_data.select_dtypes(include=["object"]).columns:
                    try:
                        chunk_data[col] = pd.to_numeric(
                            chunk_data[col], errors="coerce"
                        )
                    except:
                        logger.warning(
                            f"Impossible de convertir la colonne {col} en numérique, elle sera supprimée"
                        )
                        chunk_data = chunk_data.drop(columns=[col])

                # S'assurer que toutes les données sont numériques
                chunk_data = chunk_data.select_dtypes(include=["number"])

                # Vérifier si le DataFrame est vide après les conversions
                if chunk_data.empty:
                    raise ValueError("Aucune donnée numérique valide dans le chunk")

                # Extraire les features si feature_indices est spécifié
                if self.feature_indices is not None:
                    # S'assurer que les indices sont valides
                    valid_indices = [
                        i for i in self.feature_indices if i < chunk_data.shape[1]
                    ]
                    if valid_indices:
                        feature_data = chunk_data.iloc[:, valid_indices].values
                    else:
                        feature_data = chunk_data.values
                else:
                    feature_data = chunk_data.values

                # Convertir en tenseur PyTorch
                chunk_tensor = torch.tensor(feature_data, dtype=self.dtype)

                if self.target_column is not None and self.is_train:
                    if isinstance(self.target_column, list):
                        # Vérifier si toutes les colonnes cibles existent
                        available_targets = [
                            col
                            for col in self.target_column
                            if col in chunk_data.columns
                        ]
                        if not available_targets:
                            logger.warning(
                                f"Aucune des colonnes cibles {self.target_column} n'existe dans les données"
                            )
                            chunk_targets = None
                        else:
                            # Cibles multiples
                            chunk_targets = torch.tensor(
                                chunk_data[available_targets].values, dtype=self.dtype
                            )
                    else:
                        # Vérifier si la colonne cible existe
                        if (
                            isinstance(self.target_column, str)
                            and self.target_column not in chunk_data.columns
                        ):
                            logger.warning(
                                f"Colonne cible {self.target_column} introuvable dans les données"
                            )
                            chunk_targets = None
                        elif (
                            isinstance(self.target_column, int)
                            and self.target_column >= chunk_data.shape[1]
                        ):
                            logger.warning(
                                f"Indice cible {self.target_column} hors limites"
                            )
                            chunk_targets = None
                        else:
                            # Cible unique - extraire en tant que vecteur 1D
                            if isinstance(self.target_column, str):
                                target_values = chunk_data[self.target_column].values
                            else:  # int
                                target_values = chunk_data.iloc[
                                    :, self.target_column
                                ].values

                            chunk_targets = torch.tensor(
                                target_values, dtype=self.dtype
                            )
                else:
                    chunk_targets = None
            except Exception as e:
                logger.error(
                    f"Erreur lors du chargement du chunk depuis le fichier: {e}"
                )
                # Créer un tenseur vide comme fallback
                chunk_tensor = torch.zeros((end_idx - start_idx, 5), dtype=self.dtype)
                chunk_targets = None
        else:
            raise TypeError(
                f"Type de données non pris en charge: {type(self.data_source)}"
            )

        # S'assurer que les cibles sont des scalaires si c'est un tenseur 1D
        if chunk_targets is not None and chunk_targets.dim() == 1:
            chunk_targets = chunk_targets.unsqueeze(-1)

        return (chunk_tensor, chunk_targets)

    def _get_data_from_chunks(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Récupère les données et cibles à partir des chunks.

        Args:
            idx: Index global dans le dataset

        Returns:
            Tuple (données, cibles)
        """
        # Trouver quel chunk contient cet index
        chunk_idx = next(
            (
                i
                for i, (start, end) in enumerate(self._chunk_boundaries)
                if start <= idx < end
            ),
            None,
        )

        if chunk_idx is None:
            raise IndexError(f"Index {idx} en dehors des limites du dataset")

        # Calculer l'index local dans le chunk
        local_idx = idx - self._chunk_boundaries[chunk_idx][0]

        # Mettre à jour l'indice du chunk actuel pour le préchargement
        if self.async_prefetch and chunk_idx != self._current_chunk_idx:
            with self._prefetch_lock:
                self._current_chunk_idx = chunk_idx

        # Récupérer ou charger le chunk principal
        if chunk_idx not in self._loaded_chunks:
            self._loaded_chunks[chunk_idx] = self._load_chunk(chunk_idx)
        chunk_tensor, chunk_targets = self._loaded_chunks[chunk_idx]

        # Construire la séquence
        seq_length = self.sequence_length
        ahead = self.predict_n_ahead

        # Vérifier si la séquence entre dans le chunk actuel
        if local_idx + seq_length <= len(chunk_tensor):
            # La séquence tient dans le chunk actuel
            x = chunk_tensor[local_idx : local_idx + seq_length]

            # Extraire la cible si disponible
            if chunk_targets is not None and self.is_train:
                target_idx = local_idx + seq_length - 1 + ahead
                if target_idx < len(chunk_targets):
                    y = chunk_targets[target_idx]
                else:
                    # La cible est dans le chunk suivant
                    next_chunk_idx = chunk_idx + 1
                    if next_chunk_idx < len(self._chunk_indices):
                        # Précharger le chunk suivant si nécessaire
                        if next_chunk_idx not in self._loaded_chunks:
                            self._loaded_chunks[next_chunk_idx] = self._load_chunk(
                                next_chunk_idx
                            )

                        # Déterminer l'index relatif dans le chunk suivant
                        next_chunk_tensor, next_chunk_targets = self._loaded_chunks[
                            next_chunk_idx
                        ]
                        next_local_idx = target_idx - len(chunk_tensor)

                        if next_local_idx < len(next_chunk_targets):
                            y = next_chunk_targets[next_local_idx]
                        else:
                            y = None
                    else:
                        y = None
            else:
                y = None
        else:
            # La séquence s'étend au-delà du chunk actuel
            start_chunk_data = chunk_tensor[local_idx:]

            # Déterminer combien de données restantes sont nécessaires
            remaining_length = seq_length - len(start_chunk_data)

            # Charger le chunk suivant si nécessaire
            next_chunk_idx = chunk_idx + 1
            if next_chunk_idx < len(self._chunk_indices):
                if next_chunk_idx not in self._loaded_chunks:
                    self._loaded_chunks[next_chunk_idx] = self._load_chunk(
                        next_chunk_idx
                    )
                next_chunk_tensor, next_chunk_targets = self._loaded_chunks[
                    next_chunk_idx
                ]

                # Extraire les données restantes du chunk suivant
                if remaining_length <= len(next_chunk_tensor):
                    end_chunk_data = next_chunk_tensor[:remaining_length]

                    # Concaténer les deux parties
                    x = torch.cat([start_chunk_data, end_chunk_data], dim=0)

                    # Pour la cible, elle sera toujours dans le chunk suivant
                    if next_chunk_targets is not None and self.is_train:
                        target_idx = seq_length - len(start_chunk_data) - 1 + ahead
                        if target_idx < len(next_chunk_targets):
                            y = next_chunk_targets[target_idx]
                        else:
                            y = None
                    else:
                        y = None
                else:
                    # Même le chunk suivant ne contient pas assez de données
                    # Dans ce cas, on complète avec des zéros
                    padding_size = remaining_length - len(next_chunk_tensor)
                    padding = torch.zeros(
                        (padding_size,) + next_chunk_tensor.shape[1:],
                        dtype=next_chunk_tensor.dtype,
                    )

                    end_chunk_data = torch.cat([next_chunk_tensor, padding], dim=0)
                    x = torch.cat([start_chunk_data, end_chunk_data], dim=0)
                    y = None
            else:
                # Il n'y a pas de chunk suivant, on complète avec des zéros
                padding = torch.zeros(
                    (remaining_length,) + chunk_tensor.shape[1:],
                    dtype=chunk_tensor.dtype,
                )
                x = torch.cat([start_chunk_data, padding], dim=0)
                y = None

        return x, y

    def __len__(self) -> int:
        """Retourne le nombre d'exemples dans le dataset."""
        return self._num_examples

    # Utiliser lru_cache si disponible pour les séquences fréquemment accédées
    def _get_sequence_cached(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Récupère une séquence depuis le cache ou la génère.

        Args:
            idx: Index de la séquence

        Returns:
            Tuple (séquence, cible)
        """
        # Vérifier si la séquence est dans le cache
        if idx in self._sequence_cache:
            return self._sequence_cache[idx]

        # Sinon, générer la séquence
        if self.lazy_loading:
            # Mode lazy loading, utiliser les chunks
            try:
                x, y = self._get_data_from_chunks(idx)
            except Exception as e:
                logger.error(
                    f"Erreur lors de la récupération des données du chunk pour idx={idx}: {e}"
                )
                raise
        else:
            # Mode chargement complet, extraire directement du tenseur
            if self.data_tensor is None:
                raise RuntimeError("Les données n'ont pas été chargées correctement")

            # Extraire la séquence
            start_idx = idx
            end_idx = idx + self.sequence_length

            # Vérifier les limites
            if end_idx > len(self.data_tensor):
                raise IndexError(
                    f"Index {idx} hors limites pour la longueur des données {len(self.data_tensor)}"
                )

            x = self.data_tensor[start_idx:end_idx]

            # Extraire la cible si disponible
            if self.target_tensor is not None and self.is_train:
                target_idx = end_idx - 1 + self.predict_n_ahead
                if target_idx < len(self.target_tensor):
                    y = self.target_tensor[target_idx]
                else:
                    y = None
            else:
                y = None

        # Stocker dans le cache avant les transformations
        # Maintenir la taille du cache limitée
        if len(self._sequence_cache) >= self._cache_size:
            # Supprimer l'élément le plus ancien si nous utilisons un cache simple
            if not HAVE_LRU_CACHE:
                if self._cache_keys:
                    oldest_key = self._cache_keys.pop(0)
                    if oldest_key in self._sequence_cache:
                        del self._sequence_cache[oldest_key]
            else:
                # Si LRU_CACHE est disponible, il gérera automatiquement la taille
                # Mais nous devons quand même limiter manuellement
                keys_to_remove = list(self._sequence_cache.keys())[
                    : len(self._sequence_cache) - self._cache_size + 1
                ]
                for k in keys_to_remove:
                    del self._sequence_cache[k]

        # Ajouter la nouvelle séquence au cache
        self._sequence_cache[idx] = (x, y)

        # Ajouter la clé à la liste des clés pour le suivi LRU simple
        if not HAVE_LRU_CACHE:
            self._cache_keys.append(idx)

        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Récupère un exemple du dataset à partir de son index.

        Args:
            idx: Index de l'exemple à récupérer

        Returns:
            Tuple contenant (séquence, cible) où cible peut être None en mode prédiction
        """
        # Vérifier les limites
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} hors limites pour dataset de taille {len(self)}"
            )

        # Vérifier si le gestionnaire de mémoire doit optimiser
        if self.memory_manager:
            self.memory_manager.check_and_optimize()

        # Récupérer la séquence et la cible sans transformation
        sequence, target = self._get_sequence_cached(idx)

        # Appliquer les transformations si spécifiées
        if self.transform is not None:
            sequence_transformed = self.transform(sequence)
        else:
            sequence_transformed = sequence

        if target is not None and self.target_transform is not None:
            target_transformed = self.target_transform(target)
        else:
            target_transformed = target

        # S'assurer que target n'est jamais None pour éviter les erreurs de collate dans DataLoader
        if target_transformed is None:
            # Créer un tenseur vide compatible avec la forme attendue
            if sequence_transformed.dim() > 1:
                # Si la séquence a plus d'une dimension, créer un tenseur cible avec la même première dimension
                target_transformed = torch.zeros(
                    1,
                    dtype=sequence_transformed.dtype,
                    device=sequence_transformed.device,
                )
            else:
                # Sinon, créer un scalaire
                target_transformed = torch.zeros(
                    [],
                    dtype=sequence_transformed.dtype,
                    device=sequence_transformed.device,
                )
        elif target_transformed.dim() > 0:
            # Si la cible a plus d'une dimension, la convertir en scalaire pour le test
            target_transformed = target_transformed.squeeze()
            # S'assurer qu'on a un scalaire si c'est une cible unique
            if target_transformed.dim() == 0:
                # Parfait, c'est un scalaire
                pass
            else:
                # Si c'est un tenseur avec une seule valeur, le convertir en scalaire
                if target_transformed.numel() == 1:
                    target_transformed = target_transformed.item()
                    target_transformed = torch.tensor(
                        target_transformed,
                        dtype=sequence_transformed.dtype,
                        device=sequence_transformed.device,
                    )

        return sequence_transformed, target_transformed


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
    persistent_workers: bool = True,  # Garder les workers en vie entre les époques
    auto_threading: bool = True,  # Nouvelle option pour optimiser automatiquement les workers
) -> DataLoader:
    """
    Crée un DataLoader optimisé pour les données financières.

    Args:
        dataset: Instance de FinancialDataset
        batch_size: Taille des batchs
        shuffle: Si True, mélange les données à chaque époque
        num_workers: Nombre de processus de chargement (-1 pour auto-détection)
        prefetch_factor: Nombre de batchs à précharger par worker (si num_workers > 0)
        pin_memory: Si True, épingle la mémoire pour transfert plus rapide vers GPU
        drop_last: Si True, supprime le dernier batch s'il est incomplet
        worker_init_fn: Fonction d'initialisation personnalisée pour les workers
        optimize_memory: Si True, optimise l'utilisation de la mémoire pendant le chargement
        persistent_workers: Si True, garde les workers en vie entre les époques
        auto_threading: Si True, utilise ThreadingOptimizer pour la configuration optimale

    Returns:
        DataLoader optimisé
    """
    # Si auto_threading est activé et l'optimiseur de threading est disponible,
    # obtenir la configuration optimale
    if auto_threading and HAVE_THREADING_OPTIMIZER and num_workers == -1:
        try:
            # Créer l'optimiseur de threading
            threading_optimizer = ThreadingOptimizer()

            # Obtenir la taille approximative du dataset
            dataset_size = len(dataset)

            # Obtenir la configuration optimale pour le DataLoader
            config = threading_optimizer.get_dataloader_config(
                data_size=dataset_size,
                batch_size=batch_size,
                persistent_workers=persistent_workers,
            )

            # Appliquer la configuration optimale
            num_workers = config["num_workers"]
            prefetch_factor = config["prefetch_factor"]
            pin_memory = config["pin_memory"]
            persistent_workers = config["persistent_workers"]

            logger.info(
                f"Configuration DataLoader auto-optimisée: num_workers={num_workers}, "
                f"prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers}"
            )
        except Exception as e:
            logger.warning(
                f"Erreur lors de l'auto-optimisation du threading: {e}. Utilisation des valeurs par défaut."
            )
    elif num_workers == -1:
        # Auto-détection simple du nombre de workers si ThreadingOptimizer n'est pas disponible
        import multiprocessing

        num_workers = multiprocessing.cpu_count()
        logger.info(f"Auto-détection du nombre de workers: {num_workers}")

    # Limiter le nombre de workers en fonction de la taille du dataset
    if len(dataset) < batch_size * 10 and num_workers > 2:
        adjusted_workers = min(2, num_workers)
        logger.info(
            f"Dataset petit, réduction du nombre de workers de {num_workers} à {adjusted_workers}"
        )
        num_workers = adjusted_workers

    # Worker init function for memory optimization
    if worker_init_fn is None and optimize_memory:
        worker_init_fn = memory_optimized_worker_init
    elif worker_init_fn is not None and optimize_memory:
        # Wrap the user-provided function with our memory optimization
        original_fn = worker_init_fn
        worker_init_fn = lambda x: memory_optimized_worker_init(x, original_fn)

    # Create a GarbageCollectionDataLoader if optimize_memory, else standard DataLoader
    if optimize_memory:
        if num_workers > 0:
            loader_kwargs = {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": drop_last,
                "prefetch_factor": prefetch_factor,
                "worker_init_fn": worker_init_fn,
                "persistent_workers": persistent_workers if num_workers > 0 else False,
            }
        else:
            # No need for prefetch_factor or persistent_workers when num_workers=0
            loader_kwargs = {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": 0,
                "pin_memory": pin_memory,
                "drop_last": drop_last,
                "worker_init_fn": worker_init_fn,
            }

        return GarbageCollectionDataLoader(**loader_kwargs)
    else:
        if num_workers > 0:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=pin_memory,
                drop_last=drop_last,
                worker_init_fn=worker_init_fn,
            )


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
