"""
Module pour le chargement paresseux (lazy loading) des données financières.
Permet de charger efficacement de grands ensembles de données sans surcharger la mémoire.
"""

import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Vérification des modules optionnels
HAVE_DASK = False

try:
    HAVE_DASK = True
    logger.info("Dask est disponible pour le chargement paresseux.")
except ImportError:
    logger.warning(
        "Dask n'est pas installé. Certaines fonctionnalités de chargement paresseux seront limitées."
    )
    logger.warning("Pour installer Dask, exécutez: pip install 'dask[complete]'")

from .batch_inference import BatchInferenceOptimizer, batch_inference
from .cached_transform import (
    CachedFeatureTransform,
    CachedTransform,
    cached_transform,
    get_cache_transform_fn,
)

# Importer les composants principaux
from .lazy_storage import LazyDataset, LazyFileReader, get_lazy_dataloader

# Si Dask est disponible, importer les composants spécifiques
if HAVE_DASK:
    try:
        # Importer les composants qui dépendent de Dask
        pass

        __all__ = [
            "LazyFileReader",
            "LazyDataset",
            "get_lazy_dataloader",
            "CachedTransform",
            "get_cache_transform_fn",
            "CachedFeatureTransform",
            "cached_transform",
            "BatchInferenceOptimizer",
            "batch_inference",
            "DaskDataLoader",
            "read_parquet_lazy",
            "read_csv_lazy",
            "HAVE_DASK",
        ]
    except ImportError as e:
        logger.warning(f"Erreur lors de l'importation de certains modules Dask: {e}")
        __all__ = [
            "LazyFileReader",
            "LazyDataset",
            "get_lazy_dataloader",
            "CachedTransform",
            "get_cache_transform_fn",
            "CachedFeatureTransform",
            "cached_transform",
            "BatchInferenceOptimizer",
            "batch_inference",
            "HAVE_DASK",
        ]
else:
    __all__ = [
        "LazyFileReader",
        "LazyDataset",
        "get_lazy_dataloader",
        "CachedTransform",
        "get_cache_transform_fn",
        "CachedFeatureTransform",
        "cached_transform",
        "BatchInferenceOptimizer",
        "batch_inference",
        "HAVE_DASK",
    ]


def is_dask_available():
    """
    Vérifie si Dask est disponible pour le chargement paresseux avancé.

    Returns:
        bool: True si Dask est disponible, False sinon
    """
    return HAVE_DASK


def install_dask():
    """
    Tente d'installer Dask automatiquement.

    Returns:
        bool: True si l'installation a réussi, False sinon
    """
    try:
        import subprocess
        import sys

        print("Installation de Dask...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "dask[complete]"]
        )

        # Vérifier que l'installation a réussi

        global HAVE_DASK
        HAVE_DASK = True
        print("Dask installé avec succès!")
        return True
    except Exception as e:
        print(f"Échec de l'installation de Dask: {e}")
        return False
