#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal unifié pour nettoyer les ressources CUDA et le projet AI Trading.
Ce script regroupe toutes les fonctionnalités de nettoyage en un seul outil :
- Libération des ressources CUDA (PyTorch et TensorFlow)
- Nettoyage des fichiers temporaires et caches
- Suppression des modèles obsolètes
"""

import os
import sys
import time
import datetime
import shutil
import importlib.util
from pathlib import Path
import gc

# Chemin racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
AI_TRADING_DIR = ROOT_DIR / "ai_trading"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------

# Dossiers à nettoyer complètement
DIRS_TO_CLEAN = [
    AI_TRADING_DIR / "tmp_test_models",
    AI_TRADING_DIR / "__pycache__",
    ROOT_DIR / "__pycache__",
]

# Extensions de fichiers à supprimer
EXTENSIONS_TO_CLEAN = [
    ".pyc", 
    ".pyo", 
    ".pyd", 
    ".log",
    ".prof",  # Profiling files
    ".coverage",
    ".egg-info"
]

# Noms de dossiers à rechercher et supprimer
FOLDERS_TO_FIND_AND_CLEAN = [
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints"
]

# Répertoires contenant des modèles et résultats
MODEL_DIRS = [
    AI_TRADING_DIR / "tmp_test_models",
    AI_TRADING_DIR / "models" / "saved_models",
    AI_TRADING_DIR / "results"
]

# Âge maximum des modèles en jours avant suppression (par défaut: 30 jours)
MAX_MODEL_AGE_DAYS = 30

# Chemin vers le script de nettoyage CUDA
CUDA_SCRIPT_PATH = ROOT_DIR / "scripts" / "clean_cuda.py"

# -----------------------------------------------------------------
# Fonctions pour nettoyer les fichiers temporaires
# -----------------------------------------------------------------

def clean_directory(directory, extensions_to_clean=None, delete_empty_folders=False):
    """
    Nettoie un répertoire en supprimant les fichiers avec les extensions spécifiées
    et éventuellement les dossiers vides.
    """
    if not os.path.exists(directory):
        print(f"Le répertoire {directory} n'existe pas.")
        return

    print(f"Nettoyage du répertoire: {directory}")
    for root, dirs, files in os.walk(directory, topdown=False):
        # Supprime les fichiers avec les extensions spécifiées
        if extensions_to_clean:
            for file in files:
                file_path = os.path.join(root, file)
                if any(file.endswith(ext) for ext in extensions_to_clean):
                    try:
                        os.remove(file_path)
                        print(f"  Suppression du fichier: {file_path}")
                    except Exception as e:
                        print(f"  Erreur lors de la suppression de {file_path}: {e}")

        # Supprime les dossiers vides si demandé
        if delete_empty_folders:
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Vérifie si le dossier est vide
                        os.rmdir(dir_path)
                        print(f"  Suppression du dossier vide: {dir_path}")
                except Exception as e:
                    print(f"  Erreur lors de la suppression du dossier {dir_path}: {e}")


def clean_specific_folders(base_dir, folder_names):
    """
    Recherche et supprime les dossiers spécifiques dans tout le projet.
    """
    print(f"Recherche et suppression des dossiers spécifiques dans {base_dir}...")
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name in folder_names:
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"  Suppression du dossier: {dir_path}")
                except Exception as e:
                    print(f"  Erreur lors de la suppression de {dir_path}: {e}")


def clean_folders_to_delete():
    """Nettoie les dossiers qui doivent être complètement supprimés."""
    for directory in DIRS_TO_CLEAN:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Suppression complète du dossier: {directory}")
            except Exception as e:
                print(f"Erreur lors de la suppression de {directory}: {e}")


def clean_temp_files():
    """Fonction qui nettoie tous les fichiers temporaires."""
    print("\n=== Début du nettoyage des fichiers temporaires ===")
    
    # Nettoie les dossiers spécifiques à supprimer complètement
    clean_folders_to_delete()
    
    # Nettoie tous les fichiers avec extensions spécifiées dans le projet
    clean_directory(ROOT_DIR, EXTENSIONS_TO_CLEAN, delete_empty_folders=True)
    
    # Recherche et supprime les dossiers de cache spécifiques
    clean_specific_folders(ROOT_DIR, FOLDERS_TO_FIND_AND_CLEAN)
    
    print("=== Nettoyage des fichiers temporaires terminé ===")


# -----------------------------------------------------------------
# Fonctions pour nettoyer les modèles obsolètes
# -----------------------------------------------------------------

def clean_old_files(directory, max_age_days=MAX_MODEL_AGE_DAYS, file_extensions=None):
    """
    Supprime les fichiers plus anciens que max_age_days dans le répertoire spécifié.
    Peut être filtré par extensions de fichiers si spécifié.
    """
    if not os.path.exists(directory):
        print(f"Le répertoire {directory} n'existe pas.")
        return

    print(f"Nettoyage des fichiers obsolètes dans: {directory}")
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    for root, _, files in os.walk(directory):
        for file in files:
            # Filtre par extension si spécifié
            if file_extensions and not any(file.endswith(ext) for ext in file_extensions):
                continue

            file_path = os.path.join(root, file)
            
            # Vérifie l'âge du fichier
            file_age = now - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    age_days = file_age / (24 * 60 * 60)
                    print(f"  Suppression du fichier obsolète ({age_days:.1f} jours): {file_path}")
                except Exception as e:
                    print(f"  Erreur lors de la suppression de {file_path}: {e}")


def clean_empty_result_folders(base_dir=AI_TRADING_DIR / "results"):
    """Supprime les dossiers de résultats vides."""
    if not os.path.exists(base_dir):
        print(f"Le répertoire {base_dir} n'existe pas.")
        return

    print(f"Suppression des dossiers de résultats vides dans: {base_dir}")
    
    # Parcourt le dossier de résultats de bas en haut pour supprimer les sous-dossiers vides d'abord
    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Si le dossier est vide (pas de fichiers et plus de sous-dossiers car déjà traités)
        if not files and not dirs:
            # Ne supprime pas le dossier racine des résultats
            if root != str(base_dir):
                try:
                    os.rmdir(root)
                    print(f"  Suppression du dossier vide: {root}")
                except Exception as e:
                    print(f"  Erreur lors de la suppression du dossier {root}: {e}")


def clean_old_models(max_age_days=None):
    """Fonction qui nettoie les modèles et résultats obsolètes."""
    if max_age_days is not None:
        try:
            max_age = int(max_age_days)
        except ValueError:
            print(f"Erreur: l'âge maximum doit être un nombre entier: {max_age_days}")
            return 1
    else:
        max_age = MAX_MODEL_AGE_DAYS

    print(f"\n=== Début du nettoyage des modèles et résultats obsolètes (> {max_age} jours) ===")
    
    # Extensions de fichiers de modèles
    model_extensions = [".h5", ".keras", ".pb", ".pt", ".pth", ".ckpt", ".tf", ".model"]
    
    # Extensions de fichiers de résultats
    results_extensions = [".csv", ".json", ".pkl", ".png", ".jpg", ".pdf", ".npy"]
    
    for directory in MODEL_DIRS:
        if os.path.exists(directory):
            # Détermine les extensions à utiliser selon le dossier
            if "models" in str(directory).lower():
                clean_old_files(directory, max_age, model_extensions)
            elif "results" in str(directory).lower():
                clean_old_files(directory, max_age, results_extensions)
            else:
                clean_old_files(directory, max_age)
    
    # Supprime les dossiers de résultats vides
    clean_empty_result_folders()
    
    print("=== Nettoyage des modèles et résultats obsolètes terminé ===")


# -----------------------------------------------------------------
# Fonctions pour nettoyer les ressources CUDA
# -----------------------------------------------------------------

def import_module_from_path(module_name, file_path):
    """Importe dynamiquement un module à partir d'un chemin de fichier."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clean_cuda():
    """Fonction qui nettoie les ressources CUDA."""
    print(f"\n=== Début du nettoyage des ressources CUDA ===")
    
    if not os.path.exists(CUDA_SCRIPT_PATH):
        # Si le script n'existe pas, on utilise une implémentation directe
        print(f"Le script externe clean_cuda.py n'existe pas, utilisation de l'implémentation interne.")
        try:
            # Nettoyage PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    print("Nettoyage de la mémoire CUDA pour PyTorch...")
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
                    print("Mémoire PyTorch libérée.")
            except ImportError:
                print("PyTorch n'est pas installé.")
            except Exception as e:
                print(f"Erreur lors du nettoyage PyTorch: {e}")
                
            # Nettoyage TensorFlow
            try:
                import tensorflow as tf
                print("Nettoyage de la mémoire pour TensorFlow...")
                if hasattr(tf, 'config') and hasattr(tf.config, 'list_physical_devices'):
                    if tf.config.list_physical_devices('GPU'):
                        tf.keras.backend.clear_session()
                        print("Session TensorFlow réinitialisée.")
            except ImportError:
                print("TensorFlow n'est pas installé.")
            except Exception as e:
                print(f"Erreur lors du nettoyage TensorFlow: {e}")
                
            # Force garbage collection
            print("Exécution du garbage collector...")
            gc.collect()
            print("Garbage collection terminée.")
            
            print("=== Nettoyage des ressources CUDA terminé ===")
            return 0
            
        except Exception as e:
            print(f"Erreur lors du nettoyage CUDA interne: {e}")
            return 1
    
    try:
        # Importe et exécute le script de nettoyage CUDA
        cuda_cleaner = import_module_from_path("cuda_cleaner", CUDA_SCRIPT_PATH)
        
        # Exécute les fonctions du script importé
        if hasattr(cuda_cleaner, "clean_pytorch"):
            cuda_cleaner.clean_pytorch()
        
        if hasattr(cuda_cleaner, "clean_tensorflow"):
            cuda_cleaner.clean_tensorflow()
        
        if hasattr(cuda_cleaner, "force_gc"):
            cuda_cleaner.force_gc()
            
        print("=== Nettoyage des ressources CUDA terminé ===")
        return 0
    
    except Exception as e:
        print(f"Erreur lors de l'exécution du script de nettoyage CUDA: {e}")
        return 1

# -----------------------------------------------------------------
# Fonction principale
# -----------------------------------------------------------------

def main():
    """
    Fonction principale qui exécute toutes les opérations de nettoyage.
    """
    print("=== DÉBUT DU NETTOYAGE COMPLET DU PROJET ===\n")
    start_time = time.time()
    
    # Étape 1: Nettoyage des fichiers temporaires
    temp_status = 0
    try:
        clean_temp_files()
    except Exception as e:
        print(f"Erreur lors du nettoyage des fichiers temporaires: {e}")
        temp_status = 1
    
    # Étape 2: Nettoyage des modèles obsolètes
    models_status = None
    try:
        models_status = clean_old_models()
    except Exception as e:
        print(f"Erreur lors du nettoyage des modèles obsolètes: {e}")
        models_status = 1
    
    # Étape 3: Nettoyage des ressources CUDA
    cuda_status = 0
    try:
        cuda_status = clean_cuda()
    except Exception as e:
        print(f"Erreur lors du nettoyage des ressources CUDA: {e}")
        cuda_status = 1
    
    # Afficher un résumé des résultats
    print("\n=== RÉSUMÉ DU NETTOYAGE ===")
    all_success = True
    
    statuses = [
        ("Nettoyage des fichiers temporaires", temp_status),
        ("Nettoyage des modèles obsolètes", models_status if models_status is not None else "Aucun retour"),
        ("Nettoyage des ressources CUDA", cuda_status)
    ]
    
    for name, status in statuses:
        if isinstance(status, str):
            result = status
            print(f"- {name}: {result}")
            all_success = False
        else:
            result = "Réussi" if status == 0 else f"Échoué (code {status})"
            print(f"- {name}: {result}")
            if status != 0:
                all_success = False
    
    # Calculer le temps d'exécution
    execution_time = time.time() - start_time
    print(f"\nTemps d'exécution total: {execution_time:.2f} secondes")
    
    if all_success:
        print("\n=== NETTOYAGE COMPLET TERMINÉ AVEC SUCCÈS ===")
        return 0
    else:
        print("\n=== NETTOYAGE TERMINÉ AVEC DES ERREURS ===")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 