#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module qui détecte et configure l'environnement Docker pour maximiser les performances.
Ce module vérifie si l'application s'exécute dans un conteneur Docker,
et active automatiquement les optimisations spécifiques à Linux.
"""

import logging
import os
import platform
import sys
from typing import Any, Dict

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Variables globales
IS_DOCKER = False
IS_LINUX = platform.system() == "Linux"


def detect_docker() -> bool:
    """
    Détecte si l'application s'exécute dans un conteneur Docker.

    Returns:
        bool: True si l'environnement est Docker
    """
    global IS_DOCKER

    # Si déjà détecté, retourner la valeur mémorisée
    if IS_DOCKER:
        return True

    # Méthode 1: Vérifier l'existence de /.dockerenv
    if os.path.exists("/.dockerenv"):
        IS_DOCKER = True
        logger.info("Environnement Docker détecté (/.dockerenv)")
        return True

    # Méthode 2: Vérifier cgroup
    try:
        with open("/proc/1/cgroup", "r") as f:
            if "docker" in f.read():
                IS_DOCKER = True
                logger.info("Environnement Docker détecté (/proc/1/cgroup)")
                return True
    except (FileNotFoundError, PermissionError):
        pass

    # Méthode 3: Vérifier hostname
    try:
        with open("/etc/hostname", "r") as f:
            if (
                len(f.read().strip()) == 12
            ):  # Les hostname Docker font souvent 12 caractères
                try:
                    with open("/proc/self/cgroup", "r") as cgroup:
                        if "docker" in cgroup.read():
                            IS_DOCKER = True
                            logger.info(
                                "Environnement Docker détecté (hostname + cgroup)"
                            )
                            return True
                except (FileNotFoundError, PermissionError):
                    pass
    except (FileNotFoundError, PermissionError):
        pass

    # Si on arrive ici, ce n'est probablement pas Docker
    logger.info("Environnement Docker non détecté")
    return False


def detect_environment() -> Dict[str, Any]:
    """
    Détecte l'environnement d'exécution et ses caractéristiques.

    Returns:
        Dict: Informations sur l'environnement
    """
    env_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "is_64bit": sys.maxsize > 2**32,
        "is_docker": detect_docker(),
        "is_linux": IS_LINUX,
        "cpu_count": os.cpu_count(),
    }

    # Informations spécifiques à Docker
    if env_info["is_docker"]:
        # Récupérer les limites de ressources du conteneur si possible
        try:
            # Mémoire limitée ?
            memory_limit = None
            try:
                with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                    memory_limit = int(f.read().strip()) / (
                        1024 * 1024
                    )  # Convertir en MB
            except (FileNotFoundError, PermissionError):
                pass

            # CPU limité ?
            cpu_limit = None
            try:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                    quota = int(f.read().strip())
                    if quota > 0:
                        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f2:
                            period = int(f2.read().strip())
                            if period > 0:
                                cpu_limit = quota / period
            except (FileNotFoundError, PermissionError):
                pass

            env_info["docker_memory_limit_mb"] = memory_limit
            env_info["docker_cpu_limit"] = cpu_limit
        except Exception as e:
            logger.warning(
                f"Erreur lors de la récupération des limites Docker: {str(e)}"
            )

    # Récupérer les informations GPU si possible
    try:
        import torch

        env_info["cuda_available"] = torch.cuda.is_available()
        if env_info["cuda_available"]:
            env_info["cuda_device_count"] = torch.cuda.device_count()
            env_info["cuda_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        env_info["cuda_available"] = False

    return env_info


def apply_docker_optimizations() -> Dict[str, bool]:
    """
    Applique automatiquement les optimisations pour l'environnement Docker.

    Returns:
        Dict: Résultat des optimisations appliquées
    """
    optimizations = {
        "torch_compile_enabled": False,
        "deepspeed_enabled": False,
        "distributed_training_enabled": False,
        "mixed_precision_enabled": False,
        "environment_is_docker": IS_DOCKER or detect_docker(),
        "environment_is_linux": IS_LINUX,
    }

    # Si nous ne sommes pas dans Docker, rien à faire
    if not optimizations["environment_is_docker"]:
        logger.info(
            "L'environnement n'est pas Docker, pas d'optimisations spécifiques appliquées"
        )
        return optimizations

    logger.info("Application des optimisations pour l'environnement Docker")

    # 1. Activer torch.compile si disponible (devrait l'être sous Linux)
    try:
        import torch

        if hasattr(torch, "compile"):
            # Référencer notre module cross-platform pour s'assurer qu'il utilise la bonne implémentation
            try:
                from ai_trading.optim.cross_platform_torch_compile import (
                    is_compile_available,
                )

                if is_compile_available():
                    optimizations["torch_compile_enabled"] = True
                    logger.info(
                        "torch.compile est disponible et utilisable dans Docker"
                    )
            except ImportError:
                # Si notre module n'est pas disponible, vérifier directement
                try:
                    # Test simple pour voir si torch.compile fonctionne
                    @torch.compile
                    def simple_fn(x):
                        return x + 1

                    # Test avec un tensor
                    simple_fn(torch.tensor([1.0]))
                    optimizations["torch_compile_enabled"] = True
                    logger.info(
                        "torch.compile est disponible et utilisable dans Docker"
                    )
                except Exception as e:
                    logger.warning(f"torch.compile n'est pas utilisable: {str(e)}")
    except ImportError:
        logger.warning("torch n'est pas disponible")

    # 2. Activer DeepSpeed si disponible
    try:
        optimizations["deepspeed_enabled"] = True
        logger.info("DeepSpeed est disponible dans Docker")
    except ImportError:
        try:
            # Vérifier notre wrapper DeepSpeed
            from ai_trading.utils.deepspeed_wrapper import is_deepspeed_available

            if is_deepspeed_available():
                optimizations["deepspeed_enabled"] = True
                logger.info("DeepSpeed est disponible via notre wrapper")
            else:
                logger.info("Mode compatible DeepSpeed utilisé via notre wrapper")
                optimizations["deepspeed_enabled"] = True
        except ImportError:
            logger.warning("DeepSpeed n'est pas disponible")

    # 3. Activer l'entraînement distribué si plusieurs GPUs
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                optimizations["distributed_training_enabled"] = True
                logger.info(
                    f"Training distribué activé pour {torch.cuda.device_count()} GPUs"
                )
            except ImportError:
                logger.warning("torch.distributed n'est pas disponible")
        else:
            logger.info("Training distribué non activé (moins de 2 GPUs)")
    except ImportError:
        logger.warning("torch n'est pas disponible pour le training distribué")

    # 4. Activer la précision mixte
    try:
        import torch

        if torch.cuda.is_available():
            optimizations["mixed_precision_enabled"] = True
            logger.info("Précision mixte activée pour GPU")
    except ImportError:
        logger.warning("torch n'est pas disponible pour la précision mixte")

    return optimizations


def get_docker_deployment_info() -> Dict[str, Any]:
    """
    Retourne des informations sur le déploiement Docker actuel.

    Returns:
        Dict: Informations sur le déploiement Docker
    """
    if not (IS_DOCKER or detect_docker()):
        return {"is_docker": False}

    info = {
        "is_docker": True,
        "container_id": None,
        "image_id": None,
        "environment_variables": {},
    }

    # Récupérer l'ID du conteneur
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                if "docker" in line:
                    info["container_id"] = line.strip().split("/")[-1]
                    break
    except (FileNotFoundError, PermissionError, IndexError):
        pass

    # Récupérer les variables d'environnement liées à Docker
    for key, value in os.environ.items():
        if key.startswith("DOCKER_") or key in ["HOSTNAME", "PATH", "PYTHONPATH"]:
            info["environment_variables"][key] = value

    return info


def is_running_in_docker() -> bool:
    """
    Fonction simple pour vérifier si l'application s'exécute dans Docker.

    Returns:
        bool: True si Docker, False sinon
    """
    return IS_DOCKER or detect_docker()


def setup_for_docker():
    """
    Configure l'environnement pour un fonctionnement optimal sous Docker.
    Cette fonction doit être appelée au démarrage de l'application.
    """
    if not (IS_DOCKER or detect_docker()):
        logger.info(
            "Pas de configuration Docker - l'application ne s'exécute pas dans Docker"
        )
        return False

    logger.info("Configuration de l'environnement pour Docker")

    # Obtenir les informations sur l'environnement
    env_info = detect_environment()

    # Appliquer les optimisations
    optimizations = apply_docker_optimizations()

    # Afficher un résumé
    logger.info(f"Configuration Docker terminée. Résumé des optimisations:")
    for name, status in optimizations.items():
        logger.info(f"  - {name}: {'Activé' if status else 'Désactivé'}")

    return True


# Si exécuté directement, afficher les informations sur l'environnement
if __name__ == "__main__":
    print("=== Détection de l'environnement Docker ===")
    is_docker = detect_docker()
    print(f"Environnement Docker: {'Oui' if is_docker else 'Non'}")

    if is_docker:
        print("\n=== Informations sur l'environnement ===")
        env_info = detect_environment()
        for key, value in env_info.items():
            print(f"{key}: {value}")

        print("\n=== Application des optimisations Docker ===")
        optimizations = apply_docker_optimizations()
        for name, status in optimizations.items():
            print(f"{name}: {'Activé' if status else 'Désactivé'}")

        print("\n=== Informations sur le déploiement Docker ===")
        deploy_info = get_docker_deployment_info()
        for key, value in deploy_info.items():
            print(f"{key}: {value}")

    # Si sur Linux mais pas dans Docker, l'indiquer
    if IS_LINUX and not is_docker:
        print("\nL'application s'exécute sur Linux mais pas dans Docker.")
        print("Certaines optimisations Linux peuvent quand même être appliquées.")

    print("\nAppel de setup_for_docker():")
    result = setup_for_docker()
    print(
        f"Résultat: {'Configuration appliquée' if result else 'Pas de configuration nécessaire'}"
    )
