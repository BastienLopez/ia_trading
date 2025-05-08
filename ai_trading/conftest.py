#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration générale pour les tests pytest.
Ce fichier est automatiquement détecté par pytest sans besoin d'importation.
"""

import logging
import os
import warnings

import pytest

# Configurer le niveau de journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_trading.tests")


# Filtrer les avertissements connus
def pytest_configure(config):
    """Configuration exécutée avant le début des tests."""
    # Supprimer les avertissements de dépréciation
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Supprimer les avertissements de futurs changements
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Supprimer les avertissements spécifiques à tensorflow/jax
    warnings.filterwarnings("ignore", message="jax.xla_computation is deprecated")

    # Supprimer les avertissements de imghdr (tweepy)
    warnings.filterwarnings("ignore", message="'imghdr' is deprecated")

    # Supprimer les avertissements relatifs à distutils
    warnings.filterwarnings(
        "ignore", message="distutils Version classes are deprecated"
    )

    # Supprimer les avertissements relatifs à OAuthHandler
    warnings.filterwarnings("ignore", message="OAuthHandler is deprecated")

    # Supprimer les avertissements de type de pandas
    warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated")

    # Supprimer les avertissements de option pandas
    warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

    # Supprimer les avertissements de sklearn kmeans
    warnings.filterwarnings(
        "ignore", message="The default value of `n_init` will change"
    )

    # Supprimer les avertissements de Keras
    warnings.filterwarnings("ignore", message="keras.initializers.serialize")

    # Supprimer les avertissements de twisted
    warnings.filterwarnings(
        "ignore", message="twisted.internet.defer.returnValue was deprecated"
    )

    # Supprimer les avertissements de pandas overflow in cast
    warnings.filterwarnings("ignore", message="overflow encountered in cast")

    # Supprimer les avertissements relatifs aux d'initializers keras
    warnings.filterwarnings(
        "ignore", message="The `keras.initializers.serialize()` API should only be used"
    )

    # Rendre les tests compatibles avec CUDA
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Réduire les journaux TensorFlow


# Marqueurs personnalisés pour les tests
def pytest_addoption(parser):
    """Ajouter des options de ligne de commande."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Exécuter les tests lents",
    )
    parser.addoption(
        "--run-gpu", action="store_true", default=False, help="Exécuter les tests GPU"
    )


# Configurer les marqueurs personnalisés
def pytest_configure(config):
    """Configuration de marqueurs personnalisés pour les tests."""
    config.addinivalue_line("markers", "slow: marque les tests qui prennent du temps")
    config.addinivalue_line("markers", "gpu: marque les tests qui nécessitent un GPU")


# Sauter les tests longs sauf si --run-slow est spécifié
def pytest_collection_modifyitems(config, items):
    """Modifier les tests collectés en fonction des options."""
    run_slow = (
        config.getoption("--run-slow") or os.environ.get("RUN_SLOW_TESTS", "0") == "1"
    )
    run_gpu = (
        config.getoption("--run-gpu") or os.environ.get("RUN_GPU_TESTS", "0") == "1"
    )

    skip_slow = pytest.mark.skip(
        reason="Test trop long - utilisez --run-slow pour l'exécuter"
    )
    skip_gpu = pytest.mark.skip(reason="Test GPU - utilisez --run-gpu pour l'exécuter")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)


# Fixture pour vérifier la disponibilité de CUDA
@pytest.fixture(scope="session")
def cuda_available():
    """Vérifie si CUDA est disponible pour les tests."""
    import torch

    return torch.cuda.is_available()


# Fixture pour créer un environnement de test temporaire
@pytest.fixture(scope="function")
def temp_dir(tmpdir):
    """Crée un répertoire temporaire pour les tests."""
    return tmpdir
