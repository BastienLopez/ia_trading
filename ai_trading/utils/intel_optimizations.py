"""
Module pour optimiser les performances sur les processeurs Intel.
Active les optimisations Intel MKL et configure le nombre de threads pour torch et OpenMP.
"""

import logging
import multiprocessing
import os
import platform
from typing import Any, Dict, Optional

# Vérifier si torch est installé avant de l'importer
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Vérifier si numpy est installé avant de l'importer
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Vérifier si tensorflow est installé avant de l'importer
try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

logger = logging.getLogger(__name__)

# Configurations par défaut
DEFAULT_CONFIG = {
    "use_mkl": True,  # Utiliser Intel MKL
    "intra_op_threads": None,  # Nombre de threads pour les opérations internes (None = auto)
    "inter_op_threads": 2,  # Nombre de threads entre opérations (parallélisme)
    "omp_threads": None,  # Nombre de threads OpenMP (None = auto)
    "mkl_threads": None,  # Nombre de threads MKL (None = auto)
    "use_mkldnn": True,  # Utiliser MKLDNN (oneDNN) pour accélérer les convolutions
    "disable_avx512": False,  # Désactiver AVX-512 si problèmes de stabilité
    "optimize_memory": True,  # Optimiser l'utilisation de la mémoire
    "set_torch_deterministic": False,  # Rendre PyTorch déterministe (affecte les performances)
}


def is_intel_cpu() -> bool:
    """
    Détecte si le CPU est un processeur Intel.

    Returns:
        True si Intel, False sinon
    """
    cpu_info = platform.processor()
    return "Intel" in cpu_info


def get_optimal_threads(max_threads: Optional[int] = None) -> int:
    """
    Détermine le nombre optimal de threads à utiliser.

    Args:
        max_threads: Nombre maximum de threads à utiliser (None = auto)

    Returns:
        Nombre optimal de threads
    """
    # Obtenir le nombre de coeurs logiques disponibles
    cpu_count = multiprocessing.cpu_count()

    if max_threads is None:
        # Si le nombre de threads n'est pas spécifié, utiliser un nombre optimal
        if cpu_count <= 4:
            # Pour les petits CPU, utiliser tous les coeurs
            return cpu_count
        else:
            # Pour les CPU plus grands, laisser au moins un coeur libre
            # pour les autres processus du système
            return max(1, cpu_count - 1)
    else:
        # Utiliser le nombre spécifié, mais ne pas dépasser le nombre de coeurs
        return min(max_threads, cpu_count)


def configure_environment_variables(config: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Configure les variables d'environnement pour optimiser MKL et OpenMP.

    Args:
        config: Configuration d'optimisation

    Returns:
        Dictionnaire des variables d'environnement modifiées
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Fusionner avec la configuration par défaut
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value

    # Calculer le nombre de threads si non spécifié
    if config["omp_threads"] is None:
        config["omp_threads"] = get_optimal_threads()

    if config["mkl_threads"] is None:
        config["mkl_threads"] = config["omp_threads"]

    if config["intra_op_threads"] is None:
        config["intra_op_threads"] = config["omp_threads"]

    # Variables d'environnement à définir
    env_vars = {}

    # Configuration OpenMP
    env_vars["OMP_NUM_THREADS"] = str(config["omp_threads"])
    env_vars["OMP_SCHEDULE"] = "STATIC"
    env_vars["OMP_PROC_BIND"] = "CLOSE"

    # Configuration MKL
    if config["use_mkl"]:
        env_vars["MKL_NUM_THREADS"] = str(config["mkl_threads"])
        env_vars["MKL_DOMAIN_NUM_THREADS"] = f"BLAS={config['mkl_threads']}"

        if is_intel_cpu():
            # Optimisations spécifiques aux processeurs Intel
            env_vars["MKL_DYNAMIC"] = "false"  # Désactiver l'ajustement dynamique

            if config["disable_avx512"]:
                # Désactiver AVX-512 si demandé (problèmes de stabilité ou throttling)
                env_vars["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"

            # Pour les opérations à mémoire intensive
            if config["optimize_memory"]:
                env_vars["MKL_FAST_MEMORY_LIMIT"] = "0"

        # Activer MKL-DNN si disponible
        if config["use_mkldnn"]:
            env_vars["TF_ENABLE_ONEDNN_OPTS"] = "1"

    # Appliquer les variables d'environnement
    for var, value in env_vars.items():
        os.environ[var] = value
        logger.info(f"Environnement: {var}={value}")

    return env_vars


def configure_torch(config: Dict[str, Any] = None) -> bool:
    """
    Configure PyTorch pour optimiser les performances sur les processeurs Intel.

    Args:
        config: Configuration d'optimisation

    Returns:
        True si la configuration a réussi, False sinon
    """
    if not HAS_TORCH:
        logger.warning("PyTorch n'est pas installé, impossible de configurer")
        return False

    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Fusionner avec la configuration par défaut
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value

    try:
        # Définir le nombre de threads pour les opérations internes
        if config["intra_op_threads"] is not None:
            torch.set_num_threads(config["intra_op_threads"])
            logger.info(f"torch.set_num_threads({config['intra_op_threads']})")

        # Définir le nombre de threads pour les opérations inter-processus
        if config["inter_op_threads"] is not None:
            torch.set_num_interop_threads(config["inter_op_threads"])
            logger.info(f"torch.set_num_interop_threads({config['inter_op_threads']})")

        # Activer les optimisations MKL si disponibles et demandées
        if (
            config["use_mkl"]
            and hasattr(torch, "backends")
            and hasattr(torch.backends, "mkl")
        ):
            torch.backends.mkl.is_available()
            if hasattr(torch.backends.mkl, "enabled"):
                torch.backends.mkl.enabled = True
                logger.info("Activé: torch.backends.mkl.enabled = True")

        # Activer MKL-DNN/OneDNN si disponible et demandé
        if (
            config["use_mkldnn"]
            and hasattr(torch, "backends")
            and hasattr(torch.backends, "mkldnn")
        ):
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                logger.info("Activé: torch.backends.mkldnn.enabled = True")

        # Configurer le mode déterministe si demandé
        if config["set_torch_deterministic"]:
            torch.use_deterministic_algorithms(True)
            logger.info("Mode déterministe activé (peut affecter les performances)")

        return True

    except Exception as e:
        logger.error(f"Erreur lors de la configuration de PyTorch: {e}")
        return False


def configure_numpy(config: Dict[str, Any] = None) -> bool:
    """
    Configure NumPy pour optimiser les performances sur les processeurs Intel.

    Args:
        config: Configuration d'optimisation

    Returns:
        True si la configuration a réussi, False sinon
    """
    if not HAS_NUMPY:
        logger.warning("NumPy n'est pas installé, impossible de configurer")
        return False

    try:
        # Vérifier si NumPy utilise MKL
        using_mkl = "mkl" in np.__config__.get_info("blas_mkl_info").get(
            "libraries", []
        )

        if using_mkl:
            logger.info("NumPy utilise Intel MKL")
        else:
            logger.info("NumPy n'utilise pas Intel MKL")

        return True
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de NumPy MKL: {e}")
        return False


def configure_tensorflow(config: Dict[str, Any] = None) -> bool:
    """
    Configure TensorFlow pour optimiser les performances sur les processeurs Intel.

    Args:
        config: Configuration d'optimisation

    Returns:
        True si la configuration a réussi, False sinon
    """
    if not HAS_TF:
        logger.warning("TensorFlow n'est pas installé, impossible de configurer")
        return False

    if config is None:
        config = DEFAULT_CONFIG.copy()

    try:
        # Configurer le parallélisme intra-op
        if config["intra_op_threads"] is not None:
            tf.config.threading.set_intra_op_parallelism_threads(
                config["intra_op_threads"]
            )
            logger.info(f"TensorFlow intra_op_threads = {config['intra_op_threads']}")

        # Configurer le parallélisme inter-op
        if config["inter_op_threads"] is not None:
            tf.config.threading.set_inter_op_parallelism_threads(
                config["inter_op_threads"]
            )
            logger.info(f"TensorFlow inter_op_threads = {config['inter_op_threads']}")

        # Activer les optimisations MKL-DNN/OneDNN si demandé
        if config["use_mkldnn"]:
            # Déjà configuré via la variable d'environnement TF_ENABLE_ONEDNN_OPTS
            pass

        return True

    except Exception as e:
        logger.error(f"Erreur lors de la configuration de TensorFlow: {e}")
        return False


def optimize_for_intel(
    config: Dict[str, Any] = None,
    optimize_torch: bool = True,
    optimize_numpy: bool = True,
    optimize_tensorflow: bool = True,
) -> Dict[str, bool]:
    """
    Optimise les performances pour les processeurs Intel.

    Args:
        config: Configuration d'optimisation
        optimize_torch: Optimiser PyTorch
        optimize_numpy: Optimiser NumPy
        optimize_tensorflow: Optimiser TensorFlow

    Returns:
        Dictionnaire des résultats de configuration pour chaque bibliothèque
    """
    if not is_intel_cpu():
        logger.warning(
            "CPU non-Intel détecté, les optimisations pourraient ne pas être efficaces"
        )

    results = {}

    # Configurer les variables d'environnement
    env_vars = configure_environment_variables(config)
    results["environment"] = len(env_vars) > 0

    # Configurer PyTorch
    if optimize_torch and HAS_TORCH:
        results["torch"] = configure_torch(config)

    # Configurer NumPy
    if optimize_numpy and HAS_NUMPY:
        results["numpy"] = configure_numpy(config)

    # Configurer TensorFlow
    if optimize_tensorflow and HAS_TF:
        results["tensorflow"] = configure_tensorflow(config)

    # Afficher un résumé des optimisations
    logger.info("Optimisations Intel activées:")
    for lib, success in results.items():
        logger.info(f"  - {lib}: {'Réussi' if success else 'Échec'}")

    return results


def get_optimization_info() -> Dict[str, Any]:
    """
    Recueille des informations sur l'état actuel des optimisations.

    Returns:
        Dictionnaire d'informations sur les optimisations
    """
    info = {
        "cpu": {
            "is_intel": is_intel_cpu(),
            "processor": platform.processor(),
            "cpu_count": multiprocessing.cpu_count(),
        },
        "environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "non défini"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "non défini"),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get(
                "TF_ENABLE_ONEDNN_OPTS", "non défini"
            ),
        },
    }

    # Informations sur PyTorch
    if HAS_TORCH:
        info["torch"] = {
            "version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        }

        # Vérifier si MKL est disponible
        if hasattr(torch, "backends") and hasattr(torch.backends, "mkl"):
            if hasattr(torch.backends.mkl, "is_available"):
                info["torch"]["mkl_available"] = torch.backends.mkl.is_available()
            if hasattr(torch.backends.mkl, "enabled"):
                info["torch"]["mkl_enabled"] = torch.backends.mkl.enabled

        # Vérifier si MKL-DNN est disponible
        if hasattr(torch, "backends") and hasattr(torch.backends, "mkldnn"):
            if hasattr(torch.backends.mkldnn, "is_available"):
                info["torch"]["mkldnn_available"] = torch.backends.mkldnn.is_available()
            if hasattr(torch.backends.mkldnn, "enabled"):
                info["torch"]["mkldnn_enabled"] = torch.backends.mkldnn.enabled

    # Informations sur NumPy
    if HAS_NUMPY:
        info["numpy"] = {
            "version": np.__version__,
        }

        # Vérifier si NumPy utilise MKL
        try:
            blas_info = np.__config__.get_info("blas_mkl_info")
            info["numpy"]["using_mkl"] = "mkl" in blas_info.get("libraries", [])
            info["numpy"]["blas_info"] = blas_info
        except Exception:
            info["numpy"]["using_mkl"] = "inconnu"

    # Informations sur TensorFlow
    if HAS_TF:
        info["tensorflow"] = {
            "version": tf.__version__,
        }

        try:
            info["tensorflow"][
                "intra_op_parallelism"
            ] = tf.config.threading.get_intra_op_parallelism_threads()
            info["tensorflow"][
                "inter_op_parallelism"
            ] = tf.config.threading.get_inter_op_parallelism_threads()
        except Exception:
            pass

    return info


def print_optimization_info() -> None:
    """
    Affiche les informations sur les optimisations Intel.
    """
    info = get_optimization_info()

    print("\n========== INFORMATIONS D'OPTIMISATION INTEL ==========")

    # Informations CPU
    print(f"\nCPU:")
    print(f"  Processeur: {info['cpu']['processor']}")
    print(f"  Nombre de coeurs: {info['cpu']['cpu_count']}")
    print(f"  Est Intel: {'Oui' if info['cpu']['is_intel'] else 'Non'}")

    # Variables d'environnement
    print(f"\nVariables d'environnement:")
    for var, val in info["environment"].items():
        print(f"  {var}: {val}")

    # PyTorch
    if "torch" in info:
        print(f"\nPyTorch (version {info['torch']['version']}):")
        print(f"  Nombre de threads: {info['torch']['num_threads']}")
        print(f"  Nombre de threads interop: {info['torch']['num_interop_threads']}")

        if "mkl_available" in info["torch"]:
            print(
                f"  MKL disponible: {'Oui' if info['torch']['mkl_available'] else 'Non'}"
            )
        if "mkl_enabled" in info["torch"]:
            print(f"  MKL activé: {'Oui' if info['torch']['mkl_enabled'] else 'Non'}")
        if "mkldnn_available" in info["torch"]:
            print(
                f"  MKL-DNN disponible: {'Oui' if info['torch']['mkldnn_available'] else 'Non'}"
            )
        if "mkldnn_enabled" in info["torch"]:
            print(
                f"  MKL-DNN activé: {'Oui' if info['torch']['mkldnn_enabled'] else 'Non'}"
            )

    # NumPy
    if "numpy" in info:
        print(f"\nNumPy (version {info['numpy']['version']}):")
        print(f"  Utilise MKL: {info['numpy']['using_mkl']}")

    # TensorFlow
    if "tensorflow" in info:
        print(f"\nTensorFlow (version {info['tensorflow']['version']}):")
        if "intra_op_parallelism" in info["tensorflow"]:
            print(
                f"  Parallélisme intra-op: {info['tensorflow']['intra_op_parallelism']}"
            )
        if "inter_op_parallelism" in info["tensorflow"]:
            print(
                f"  Parallélisme inter-op: {info['tensorflow']['inter_op_parallelism']}"
            )

    print("\n======================================================")


if __name__ == "__main__":
    # Configurer le logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Appliquer les optimisations Intel
    optimize_for_intel()

    # Afficher les informations d'optimisation
    print_optimization_info()
