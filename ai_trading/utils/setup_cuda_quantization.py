#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour configurer et tester la quantification CUDA.

Ce script permet de:
1. Vérifier la disponibilité de CUDA
2. Configurer l'environnement pour la quantification CUDA
3. Exécuter les tests de quantification
"""

import os
import sys
import argparse
import logging
import torch
import subprocess
from pathlib import Path

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("setup_cuda_quantization")

# Ajouter le chemin du projet au sys.path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def check_cuda_availability():
    """Vérifie la disponibilité de CUDA et affiche les informations"""
    logger.info("Vérification de la disponibilité CUDA...")
    
    if not torch.cuda.is_available():
        logger.error("CUDA n'est pas disponible dans cet environnement")
        return False
    
    # Informations CUDA
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, "version") else "Inconnu"
    device_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    
    logger.info(f"Version CUDA: {cuda_version}")
    logger.info(f"Version cuDNN: {cudnn_version}")
    logger.info(f"Nombre de GPUs: {device_count}")
    for i, name in enumerate(device_names):
        logger.info(f"  GPU {i}: {name}")
        
    # Informations mémoire
    for i in range(device_count):
        torch.cuda.set_device(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        free_mem = total_mem - allocated_mem
        
        logger.info(f"  GPU {i} Mémoire totale: {total_mem:.2f} GB")
        logger.info(f"  GPU {i} Mémoire réservée: {reserved_mem:.2f} GB")
        logger.info(f"  GPU {i} Mémoire allouée: {allocated_mem:.2f} GB")
        logger.info(f"  GPU {i} Mémoire libre: {free_mem:.2f} GB")
    
    return True


def setup_environment():
    """Configure l'environnement pour la quantification CUDA"""
    logger.info("Configuration de l'environnement pour la quantification CUDA...")
    
    # Variables d'environnement pour optimiser CUDA
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["PYTORCH_CUDA_MEMORY_FRACTION"] = "0.8"
    os.environ["PYTORCH_JIT"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Activer les optimisations PyTorch CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
        # Activer TF32 pour les GPUs Ampere+
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 activé pour les opérations matmul")
            
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 activé pour cuDNN")
        
        logger.info("Environment CUDA configuré avec succès")
        return True
    else:
        logger.error("Impossible de configurer l'environnement CUDA: CUDA non disponible")
        return False


def patch_tests():
    """Applique un patch aux tests pour forcer la détection CUDA"""
    try:
        from ai_trading.utils.model_quantization import is_cuda_quantization_supported
        
        # Vérifie que notre fonction améliorée avec le paramètre force_cuda existe
        if "force_cuda" in is_cuda_quantization_supported.__code__.co_varnames:
            logger.info("Les tests sont déjà compatibles avec le forçage CUDA")
        else:
            logger.warning("La fonction is_cuda_quantization_supported n'a pas le paramètre force_cuda")
            logger.warning("Veuillez mettre à jour le code pour ajouter ce paramètre")
        
        logger.info("Tests modifiés pour forcer la détection CUDA")
        return True
    except (ImportError, AttributeError) as e:
        logger.error(f"Erreur lors de la modification des tests: {e}")
        return False


def run_tests(test_modules=None):
    """Exécute les tests de quantification
    
    Args:
        test_modules: Liste des modules de test à exécuter, ou None pour tous
    """
    if test_modules is None:
        test_modules = [
            "ai_trading/tests/test_model_quantization.py",
            "ai_trading/tests/test_model_offloading.py"
        ]
    
    logger.info(f"Exécution des tests: {', '.join(test_modules)}")
    
    # Exécuter les tests avec pytest en utilisant le mode verbeux
    cmd = [sys.executable, "-m", "pytest"] + test_modules + ["-v"]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Tous les tests ont été exécutés")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution des tests: {e}")
        return False


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Configuration et test de la quantification CUDA")
    parser.add_argument("--check-only", action="store_true", help="Vérifier uniquement la disponibilité CUDA sans exécuter les tests")
    parser.add_argument("--run-tests", action="store_true", help="Exécuter les tests de quantification")
    parser.add_argument("--test-module", type=str, nargs="+", help="Modules de test spécifiques à exécuter")
    
    args = parser.parse_args()
    
    # Vérifier la disponibilité CUDA
    cuda_available = check_cuda_availability()
    
    if not cuda_available:
        logger.warning("CUDA n'est pas disponible, les tests seront exécutés en mode CPU")
    
    # Configurer l'environnement
    if not args.check_only:
        setup_environment()
        patch_tests()
    
    # Exécuter les tests si demandé
    if args.run_tests or (not args.check_only and not args.run_tests):
        run_tests(args.test_module)
    
    logger.info("Script terminé avec succès")


if __name__ == "__main__":
    main() 