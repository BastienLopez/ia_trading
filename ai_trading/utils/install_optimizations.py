#!/usr/bin/env python
"""
Script d'installation des dépendances pour l'optimisation CPU/GPU du système de trading.
Ce script vérifie et installe les packages nécessaires pour les performances optimales.
"""

import logging
import os
import platform
import subprocess
import sys

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Définir les dépendances requises
CPU_OPTIMIZATION_DEPS = [
    "psutil",  # Pour le suivi de la mémoire et des processus
    "pyarrow",  # Pour le stockage et le chargement efficace des données
    "pandas",  # Pour la manipulation des données (vérifier la version récente)
    "h5py",  # Pour le stockage HDF5 optimisé
    "threadpoolctl",  # Pour contrôler le nombre de threads dans les bibliothèques scientifiques
    "pympler",  # Pour l'analyse avancée de l'utilisation de la mémoire
    "py-spy",  # Pour le profilage CPU en temps réel
    "line_profiler",  # Pour le profilage de ligne de code
]

# Dépendances optionnelles mais recommandées
RECOMMENDED_DEPS = [
    "numba",  # Pour la compilation JIT de code Python numérique
    "bottleneck",  # Accélération des opérations pandas
    "memory_profiler",  # Pour le profilage mémoire détaillé
    "fastparquet",  # Alternative à pyarrow pour les fichiers parquet
]


def check_python_version():
    """Vérifie que la version de Python est compatible avec les optimisations."""
    python_version = sys.version_info

    if python_version.major < 3 or (
        python_version.major == 3 and python_version.minor < 8
    ):
        logger.warning(
            f"Python {python_version.major}.{python_version.minor} détecté. "
            f"Recommandé: Python 3.8 ou supérieur pour des performances optimales."
        )
        return False
    else:
        logger.info(
            f"Python {python_version.major}.{python_version.minor} détecté - OK"
        )
        return True


def get_installed_packages():
    """Obtient la liste des packages installés et leurs versions."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = {}

        for line in result.stdout.splitlines():
            if "==" in line:
                name, version = line.split("==", 1)
                packages[name.lower()] = version

        return packages
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de la récupération des packages installés: {e}")
        return {}


def check_dependencies(required_deps, installed_packages, install_missing=False):
    """
    Vérifie si les dépendances requises sont installées et les installe si nécessaire.

    Args:
        required_deps: Liste des dépendances à vérifier
        installed_packages: Dictionnaire des packages installés
        install_missing: Si True, installe les packages manquants

    Returns:
        Tuple de (packages manquants, packages installés)
    """
    missing = []
    installed = []

    for dep in required_deps:
        if dep.lower() in installed_packages:
            installed.append(f"{dep}=={installed_packages[dep.lower()]}")
        else:
            missing.append(dep)

    if missing and install_missing:
        logger.info(f"Installation des dépendances manquantes: {', '.join(missing)}")
        try:
            for package in missing:
                logger.info(f"Installation de {package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package], check=True
                )
            logger.info("Installation des dépendances terminée avec succès.")

            # Mettre à jour la liste des packages installés
            installed_packages = get_installed_packages()
            installed = []
            still_missing = []

            for dep in required_deps:
                if dep.lower() in installed_packages:
                    installed.append(f"{dep}=={installed_packages[dep.lower()]}")
                else:
                    still_missing.append(dep)

            missing = still_missing
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de l'installation des dépendances: {e}")

    return missing, installed


def check_system_optimization():
    """Vérifie si le système est configuré de manière optimale pour les performances."""
    system = platform.system()
    optimizations = []
    warnings = []

    if system == "Windows":
        try:
            # Vérifier si le système utilise les fichiers à haute performance
            import psutil

            # Vérifier les priorités de processus
            current_process = psutil.Process()
            priority = current_process.nice()

            if priority > 0:  # Plus bas = priorité plus élevée sur Windows
                optimizations.append(
                    f"Priorité de processus actuelle: {priority} (Normal)"
                )
            else:
                optimizations.append(
                    f"Priorité de processus actuelle: {priority} (Élevée)"
                )

            # Vérifier si l'hyperthreading est activé
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)

            if logical_cores and physical_cores and logical_cores > physical_cores:
                ratio = logical_cores / physical_cores
                optimizations.append(
                    f"Hyperthreading détecté: {physical_cores} cœurs physiques, {logical_cores} cœurs logiques (ratio {ratio:.1f})"
                )
            else:
                warnings.append(
                    "Hyperthreading non détecté ou non disponible. Performances potentiellement réduites pour le multi-threading."
                )

        except ImportError:
            warnings.append(
                "psutil non installé, impossible de vérifier les paramètres système avancés."
            )

    elif system == "Linux":
        try:
            # Vérifier les configurations Linux spécifiques
            import resource

            # Vérifier les limites de fichiers ouverts
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < 4096:
                warnings.append(
                    f"Limite de fichiers ouverts trop basse: {soft}. Recommandé: 4096+"
                )
            else:
                optimizations.append(f"Limite de fichiers ouverts: {soft} - OK")

            # Vérifier si l'utilisateur peut verrouiller la mémoire
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
                if soft == 0:
                    warnings.append(
                        "Verrouillage mémoire désactivé. Performances potentiellement réduites pour les opérations intensives."
                    )
                else:
                    optimizations.append(f"Verrouillage mémoire activé: {soft} - OK")
            except Exception:
                warnings.append(
                    "Impossible de vérifier les paramètres de verrouillage mémoire."
                )

        except ImportError:
            warnings.append(
                "Module resource non disponible, impossible de vérifier les paramètres système avancés."
            )

    # Optimisations communes à tous les systèmes
    try:
        import numpy as np

        optimizations.append(f"NumPy version {np.__version__}")

        try:
            # Vérifier la configuration de NumPy
            np_config = np.__config__
            if hasattr(np_config, "show") and callable(np_config.show):
                config_info = np_config.show()
                if (
                    "mkl" in str(config_info).lower()
                    or "blas" in str(config_info).lower()
                ):
                    optimizations.append("NumPy utilise MKL ou BLAS - OK")
                else:
                    warnings.append(
                        "NumPy n'utilise pas MKL/BLAS - performances potentiellement réduites pour les calculs matriciels."
                    )
        except Exception:
            pass

    except ImportError:
        warnings.append(
            "NumPy non installé, impossible de vérifier les optimisations NumPy."
        )

    # Vérification des threads OpenMP et MKL
    try:
        import threadpoolctl

        pools = threadpoolctl.threadpool_info()
        if pools:
            for pool in pools:
                optimizations.append(
                    f"Pool de threads détecté: {pool['user_api']} avec {pool.get('num_threads', 'inconnu')} threads"
                )
        else:
            warnings.append("Aucun pool de threads OpenMP/MKL/BLAS détecté.")

    except ImportError:
        warnings.append(
            "threadpoolctl non installé, impossible de vérifier les configurations de threads."
        )

    return optimizations, warnings


def set_cpu_optimization_env_vars():
    """
    Configure les variables d'environnement pour optimiser les performances CPU.
    """
    optimizations = {}

    # NumPy et bibliothèques scientifiques
    # Utiliser le maximum de threads disponibles pour les calculs
    import multiprocessing

    num_cores = multiprocessing.cpu_count()

    # MKL (Intel Math Kernel Library)
    optimizations["MKL_NUM_THREADS"] = str(num_cores)
    optimizations["NUMEXPR_NUM_THREADS"] = str(num_cores)

    # OpenBLAS
    optimizations["OPENBLAS_NUM_THREADS"] = str(num_cores)

    # OpenMP (utilisé par scikit-learn, XGBoost, etc.)
    optimizations["OMP_NUM_THREADS"] = str(num_cores)

    # Paramètres de performance pour torch.DataLoader
    optimizations["OMP_WAIT_POLICY"] = "ACTIVE"  # Réduit la latence du thread switching

    # Indique au garbage collector Python d'être plus agressif (moins de pauses GC mais plus fréquentes)
    optimizations["PYTHONgc"] = "1"

    # Appliquer les variables d'environnement
    for key, value in optimizations.items():
        os.environ[key] = value
        logger.info(f"Variable d'environnement définie: {key}={value}")

    return optimizations


def main(install_missing=True, check_system=True):
    """
    Fonction principale pour vérifier et installer les optimisations CPU.

    Args:
        install_missing: Si True, installe les dépendances manquantes
        check_system: Si True, vérifie la configuration système
    """
    logger.info("=== Vérification des optimisations CPU pour le système de trading ===")

    # Vérifier la version de Python
    python_ok = check_python_version()

    # Obtenir les packages installés
    installed_packages = get_installed_packages()

    # Vérifier les dépendances requises
    logger.info("Vérification des dépendances requises pour l'optimisation CPU...")
    missing, installed = check_dependencies(
        CPU_OPTIMIZATION_DEPS, installed_packages, install_missing
    )

    if missing:
        logger.warning(f"Dépendances manquantes: {', '.join(missing)}")
    else:
        logger.info("Toutes les dépendances requises sont installées.")

    # Vérifier les dépendances recommandées
    logger.info("Vérification des dépendances recommandées...")
    missing_rec, installed_rec = check_dependencies(
        RECOMMENDED_DEPS, installed_packages, False
    )

    if missing_rec:
        logger.info(
            f"Dépendances recommandées manquantes: {', '.join(missing_rec)}\n"
            f"Pour installer: pip install {' '.join(missing_rec)}"
        )

    # Vérifier les optimisations système
    if check_system:
        logger.info("Vérification de la configuration système...")
        optimizations, warnings = check_system_optimization()

        for opt in optimizations:
            logger.info(f"Optimisation: {opt}")

        for warn in warnings:
            logger.warning(f"Attention: {warn}")

    # Configurer les variables d'environnement pour l'optimisation
    logger.info(
        "Configuration des variables d'environnement pour l'optimisation CPU..."
    )
    set_cpu_optimization_env_vars()

    # Résumé final
    logger.info("\n=== Résumé de l'optimisation CPU ===")

    if not missing:
        logger.info("✓ Toutes les dépendances requises sont installées.")
    else:
        logger.warning(f"✗ Dépendances manquantes: {', '.join(missing)}")

    if not missing_rec:
        logger.info("✓ Toutes les dépendances recommandées sont installées.")
    else:
        logger.info(f"ℹ Dépendances recommandées manquantes: {', '.join(missing_rec)}")

    if check_system and not warnings:
        logger.info("✓ Configuration système optimale.")
    elif check_system:
        logger.warning(
            f"⚠ {len(warnings)} avertissements sur la configuration système."
        )

    logger.info("\nPour optimiser davantage les performances CPU:")
    logger.info(
        "1. Utilisez 'get_financial_dataloader()' avec num_workers=-1 pour auto-détection"
    )
    logger.info(
        "2. Activez async_prefetch=True dans FinancialDataset pour le chargement asynchrone"
    )
    logger.info(
        "3. Mesurez les performances avec PerformanceTracker pour identifier les goulots d'étranglement"
    )
    logger.info(
        "4. Utilisez le décorateur @track_time pour profiler les fonctions critiques"
    )


if __name__ == "__main__":
    # Analyser les arguments de ligne de commande
    import argparse

    parser = argparse.ArgumentParser(
        description="Installe et configure les optimisations CPU pour le système de trading"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Ne pas installer les dépendances manquantes",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Ne pas vérifier la configuration système",
    )

    args = parser.parse_args()

    main(install_missing=not args.no_install, check_system=not args.no_check)
