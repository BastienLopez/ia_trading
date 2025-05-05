"""
Module contenant des utilitaires pour optimiser le stockage et le chargement des données financières.
Ces fonctions permettent de convertir et stocker des données dans des formats optimisés
comme Parquet et HDF5 pour un accès plus rapide et une utilisation mémoire réduite.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Configuration du logger
logger = logging.getLogger(__name__)

# Importation des modules optionnels avec gestion des erreurs
HAVE_PYARROW = False
HAVE_HDF5 = False
HAVE_TABLES = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAVE_PYARROW = True
except ImportError:
    logger.warning(
        "Module PyArrow non disponible. Pour l'installer: pip install pyarrow. "
        "Les fonctionnalités Parquet ne seront pas disponibles."
    )

try:
    import tables
    HAVE_TABLES = True
except ImportError:
    logger.warning(
        "Module tables (PyTables) non disponible. Pour l'installer: pip install tables. "
        "Certaines fonctionnalités HDF5 peuvent être limitées."
    )

try:
    import h5py
    HAVE_HDF5 = True
except ImportError:
    logger.warning(
        "Module h5py non disponible. Pour l'installer: pip install h5py. "
        "Les fonctionnalités HDF5 ne seront pas disponibles."
    )

# Importer tqdm pour les barres de progression si disponible
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Version simplifiée de tqdm si non disponible
    def tqdm(iterable, **kwargs):
        return iterable


def convert_to_parquet(
    data: Union[pd.DataFrame, str],
    output_path: str,
    compression: str = "snappy",
    row_group_size: int = 100000,
    use_dictionary: bool = True,
    partition_cols: Optional[List[str]] = None,
) -> str:
    """
    Convertit des données en format Parquet pour un stockage et chargement optimisés.

    Args:
        data: DataFrame ou chemin vers un fichier CSV à convertir
        output_path: Chemin de sortie pour le fichier Parquet
        compression: Algorithme de compression ('snappy', 'gzip', 'brotli', 'zstd', 'lz4', None)
        row_group_size: Taille des groupes de lignes pour l'optimisation d'accès
        use_dictionary: Si True, utilise l'encodage par dictionnaire pour les colonnes textuelles
        partition_cols: Colonnes selon lesquelles partitionner les données (ex: ['date', 'symbol'])

    Returns:
        Chemin vers le fichier Parquet créé
    """
    if not HAVE_PYARROW:
        raise ImportError(
            "Le module PyArrow est requis pour cette fonctionnalité. "
            "Installez-le avec: pip install pyarrow"
        )

    # Charger les données si fournies sous forme de chemin
    if isinstance(data, str):
        if os.path.exists(data):
            ext = os.path.splitext(data)[1].lower()
            if ext == ".csv":
                logger.info(f"Chargement du fichier CSV: {data}")
                data = pd.read_csv(data, index_col=0, parse_dates=True)
            elif ext == ".parquet":
                logger.warning(f"Le fichier est déjà au format Parquet: {data}")
                return data
            else:
                raise ValueError(
                    f"Format de fichier non supporté pour la conversion: {ext}"
                )
        else:
            raise FileNotFoundError(f"Fichier introuvable: {data}")

    # Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convertir l'index en colonne si c'est un DatetimeIndex pour optimiser les partitions
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()

    # Optimisation des types de données pour réduire l'empreinte mémoire
    for col in data.select_dtypes(include=["float64"]).columns:
        # Si les valeurs sont entières, convertir en entier
        if data[col].dropna().apply(lambda x: x.is_integer()).all():
            data[col] = data[col].astype("Int64")  # Int64 peut contenir des NaN
        # Sinon, utiliser float32 pour les valeurs décimales
        else:
            data[col] = data[col].astype("float32")

    # Convertir en Table PyArrow
    table = pa.Table.from_pandas(data)

    # Écrire en format Parquet
    if partition_cols:
        logger.info(
            f"Écriture des données partitionnées par {partition_cols} dans: {output_path}"
        )
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=partition_cols,
            compression=compression,
            row_group_size=row_group_size,
            use_dictionary=use_dictionary,
        )
        return output_path
    else:
        logger.info(f"Écriture des données en format Parquet dans: {output_path}")
        pq.write_table(
            table,
            output_path,
            compression=compression,
            row_group_size=row_group_size,
            use_dictionary=use_dictionary,
        )
        return output_path


def convert_to_hdf5(
    data: Union[pd.DataFrame, str],
    output_path: str,
    key: str = "data",
    mode: str = "w",
    format: str = "table",
    complevel: int = 9,
    complib: str = "blosc:lz4",
    min_itemsize: Optional[Dict[str, int]] = None,
) -> str:
    """
    Convertit des données en format HDF5 pour un stockage et chargement optimisés.

    Args:
        data: DataFrame ou chemin vers un fichier à convertir
        output_path: Chemin de sortie pour le fichier HDF5
        key: Clé sous laquelle stocker les données dans le fichier HDF5
        mode: Mode d'ouverture du fichier ('w' pour écraser, 'a' pour ajouter)
        format: Format de stockage ('table' ou 'fixed')
        complevel: Niveau de compression (0-9)
        complib: Bibliothèque de compression ('zlib', 'lzo', 'bzip2', 'blosc')
        min_itemsize: Tailles minimales pour les colonnes de type chaîne

    Returns:
        Chemin vers le fichier HDF5 créé
    """
    if not (HAVE_HDF5 or HAVE_TABLES):
        raise ImportError(
            "Les modules h5py ou tables sont requis pour cette fonctionnalité. "
            "Installez-les avec: pip install h5py tables"
        )

    # Charger les données si fournies sous forme de chemin
    if isinstance(data, str):
        if os.path.exists(data):
            ext = os.path.splitext(data)[1].lower()
            if ext == ".csv":
                logger.info(f"Chargement du fichier CSV: {data}")
                data = pd.read_csv(data, index_col=0, parse_dates=True)
            elif ext in [".h5", ".hdf5"]:
                logger.warning(f"Le fichier est déjà au format HDF5: {data}")
                return data
            elif ext == ".parquet":
                if not HAVE_PYARROW:
                    raise ImportError(
                        "PyArrow est requis pour lire des fichiers Parquet. "
                        "Installez-le avec: pip install pyarrow"
                    )
                logger.info(f"Chargement du fichier Parquet: {data}")
                data = pd.read_parquet(data)
            else:
                raise ValueError(
                    f"Format de fichier non supporté pour la conversion: {ext}"
                )
        else:
            raise FileNotFoundError(f"Fichier introuvable: {data}")

    # Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Optimiser les types de données
    for col in data.select_dtypes(include=["float64"]).columns:
        # Si les valeurs sont entières, convertir en entier
        if data[col].dropna().apply(lambda x: x.is_integer()).all():
            data[col] = data[col].astype("Int64")
        # Sinon, utiliser float32 pour les valeurs décimales
        else:
            data[col] = data[col].astype("float32")

    # Déterminer les tailles minimales pour les colonnes string
    if min_itemsize is None:
        min_itemsize = {}
        for col in data.select_dtypes(include=["object"]).columns:
            try:
                # Calculer la longueur maximale de chaîne + marge de sécurité
                max_len = data[col].str.len().max()
                if pd.notna(max_len):
                    min_itemsize[col] = int(max_len * 1.2)  # 20% de marge de sécurité
            except (AttributeError, TypeError):
                # Si ce n'est pas une chaîne ou s'il y a des valeurs mixtes
                continue

    # Écrire en format HDF5
    logger.info(f"Écriture des données en format HDF5 dans: {output_path}")
    data.to_hdf(
        output_path,
        key=key,
        mode=mode,
        format=format,
        complevel=complevel,
        complib=complib,
        min_itemsize=min_itemsize if min_itemsize else None,
    )

    return output_path


def optimize_dataset_storage(
    data_dir: str,
    output_dir: Optional[str] = None,
    format: str = "parquet",
    recursive: bool = True,
    pattern: str = "*.csv",
    partition_dates: bool = True,
) -> List[str]:
    """
    Optimise le stockage de multiples datasets dans un répertoire.

    Args:
        data_dir: Répertoire contenant les fichiers à optimiser
        output_dir: Répertoire de sortie (si None, utilise data_dir/optimized)
        format: Format de sortie ('parquet' ou 'hdf5')
        recursive: Si True, recherche récursivement dans les sous-répertoires
        pattern: Motif de fichiers à traiter
        partition_dates: Si True, partitionne les données par date pour Parquet

    Returns:
        Liste des chemins vers les fichiers optimisés
    """
    # Vérifier si les modules nécessaires sont disponibles
    if format.lower() == "parquet" and not HAVE_PYARROW:
        raise ImportError(
            "Le module PyArrow est requis pour le format Parquet. "
            "Installez-le avec: pip install pyarrow"
        )
    elif format.lower() == "hdf5" and not (HAVE_HDF5 or HAVE_TABLES):
        raise ImportError(
            "Les modules h5py ou tables sont requis pour le format HDF5. "
            "Installez-les avec: pip install h5py tables"
        )

    # Paramètres par défaut
    if output_dir is None:
        output_dir = os.path.join(data_dir, "optimized")

    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Trouver tous les fichiers correspondant au motif
    data_dir_path = Path(data_dir)
    if recursive:
        file_paths = list(data_dir_path.glob(f"**/{pattern}"))
    else:
        file_paths = list(data_dir_path.glob(pattern))

    logger.info(f"Trouvé {len(file_paths)} fichiers à optimiser")

    output_files = []
    files_iter = (
        tqdm(file_paths, desc="Optimisation des fichiers") if HAS_TQDM else file_paths
    )

    for file_path in files_iter:
        # Créer un chemin de sortie relatif
        rel_path = file_path.relative_to(data_dir_path)
        rel_dir = os.path.dirname(rel_path)

        # Déterminer le nom du fichier de sortie
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if format.lower() == "parquet":
            # Pour Parquet avec partitionnement, créer un répertoire par fichier
            if partition_dates:
                output_file = os.path.join(output_dir, rel_dir, file_name)
                # S'assurer que c'est un répertoire pour le partitionnement
                os.makedirs(output_file, exist_ok=True)
            else:
                output_file = os.path.join(output_dir, rel_dir, f"{file_name}.parquet")
        else:  # hdf5
            output_file = os.path.join(output_dir, rel_dir, f"{file_name}.h5")

        # Créer le répertoire de sortie si nécessaire
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Lire le fichier source
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Identifier les colonnes de date pour le partitionnement
            date_cols = []
            if partition_dates and format.lower() == "parquet":
                # Si l'index est un DatetimeIndex, l'inclure dans les partitions
                if isinstance(data.index, pd.DatetimeIndex):
                    # Convertir l'index en colonnes de date
                    data = data.reset_index()
                    # Partitionner par année, mois
                    data["year"] = data.iloc[:, 0].dt.year
                    data["month"] = data.iloc[:, 0].dt.month
                    date_cols = ["year", "month"]

                # Chercher d'autres colonnes de date
                for col in data.columns:
                    if pd.api.types.is_datetime64_any_dtype(data[col]):
                        data[f"{col}_year"] = data[col].dt.year
                        data[f"{col}_month"] = data[col].dt.month
                        date_cols.extend([f"{col}_year", f"{col}_month"])

            # Convertir selon le format spécifié
            if format.lower() == "parquet":
                output_path = convert_to_parquet(
                    data, output_file, partition_cols=date_cols if date_cols else None
                )
            else:  # hdf5
                output_path = convert_to_hdf5(data, output_file)

            output_files.append(output_path)

        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation de {file_path}: {e}")

    logger.info(
        f"Optimisation terminée. {len(output_files)} fichiers traités avec succès."
    )
    return output_files
