#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du module d'optimisation système.
Ce script montre comment configurer et appliquer les optimisations
système pour améliorer les performances de l'application d'IA trading.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import psutil

# Ajout du répertoire parent au path pour importer les modules du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_trading.utils.system_optimizer import SystemOptimizer, optimize_system


def create_example_config():
    """Crée un fichier de configuration d'exemple."""
    config = {
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        },
        "system_limits": {
            "file_limit": 8192,
            "process_limit": 4096,
            "memory_limit_mb": 0,  # Pas de limite
        },
        "disk_optimization": {"tmp_in_ram": True, "use_ssd_for_cache": True},
        "memory_optimization": {"swappiness": 10, "cache_pressure": 50},
        "logging": {
            "level": "INFO",
            "max_file_size_mb": 10,
            "rotation_count": 5,
            "use_json_format": False,
        },
    }

    # Créer un fichier temporaire pour stocker la configuration
    with tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".json") as f:
        json.dump(config, f, indent=4)
        config_path = f.name

    print(f"Configuration d'exemple créée dans: {config_path}")
    return config_path


def example_with_default_config():
    """Exemple d'utilisation avec la configuration par défaut."""
    print("\n--- Exemple avec configuration par défaut ---")

    # Créer l'optimiseur avec la configuration par défaut
    optimizer = SystemOptimizer()

    # Afficher les informations système détectées
    system_info = optimizer.system_info
    print(f"Système d'exploitation: {system_info['os']} {system_info['os_release']}")
    print(
        f"CPU: {system_info['physical_cpu']} cœurs physiques, {system_info['cpu_count']} threads"
    )
    print(f"Mémoire totale: {system_info['memory_total'] / (1024**3):.2f} GB")

    # Optimiser uniquement les variables d'environnement
    env_vars = optimizer.optimize_environment_variables()
    print("\nVariables d'environnement optimisées:")
    for var, value in optimizer.config["env_vars"].items():
        print(f"  {var} = {value}")

    # Configurer le logging
    logger = optimizer.setup_logging()
    logger.info("Test du système de logging configuré")
    print("\nLogging configuré avec succès. Vérifiez le dossier ./logs")


def example_with_custom_config():
    """Exemple d'utilisation avec une configuration personnalisée."""
    print("\n--- Exemple avec configuration personnalisée ---")

    # Créer un fichier de configuration d'exemple
    config_path = create_example_config()

    try:
        # Créer l'optimiseur avec la configuration personnalisée
        optimizer = SystemOptimizer(config_path)

        # Appliquer toutes les optimisations
        print("\nApplication de toutes les optimisations...")
        start_time = time.time()
        optimizations = optimizer.optimize_all()
        end_time = time.time()

        print(f"Optimisations appliquées en {end_time - start_time:.2f} secondes")

        # Afficher le statut des optimisations
        status = optimizer.get_optimization_status()
        print("\nStatut des optimisations:")
        print(
            f"  Variables d'environnement: {len(status['optimizations'].get('environment_variables', {}))}"
        )

        if "system_limits" in status["optimizations"]:
            print(f"  Limites système: {status['optimizations']['system_limits']}")

        if (
            "disk_optimization" in status["optimizations"]
            and "temp_dir" in status["optimizations"]["disk_optimization"]
        ):
            print(
                f"  Répertoire temporaire: {status['optimizations']['disk_optimization']['temp_dir']}"
            )

        if "memory_optimization" in status["optimizations"]:
            print(
                f"  Optimisations mémoire: {status['optimizations']['memory_optimization']}"
            )

        if "logging" in status["optimizations"]:
            print(f"  Configuration logging: {status['optimizations']['logging']}")

    finally:
        # Nettoyer le fichier de configuration temporaire
        try:
            os.unlink(config_path)
            print(f"\nFichier de configuration temporaire supprimé: {config_path}")
        except:
            pass


def example_optimize_system_helper():
    """Exemple d'utilisation de la fonction utilitaire optimize_system."""
    print("\n--- Exemple avec la fonction utilitaire optimize_system ---")

    # Utiliser la fonction utilitaire pour optimiser le système
    print("Optimisation du système en une seule étape...")
    optimizer = optimize_system()

    # Vérifier l'état de la mémoire après optimisation
    mem = psutil.virtual_memory()
    print(f"\nUtilisation mémoire après optimisation:")
    print(f"  Total: {mem.total / (1024**3):.2f} GB")
    print(f"  Disponible: {mem.available / (1024**3):.2f} GB")
    print(f"  Pourcentage utilisé: {mem.percent}%")

    # Vérifier l'utilisation CPU
    print(f"\nUtilisation CPU: {psutil.cpu_percent(interval=1)}%")

    # Afficher quelques variables d'environnement optimisées
    print("\nQuelques variables d'environnement optimisées:")
    env_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED"]
    for var in env_vars:
        if var in os.environ:
            print(f"  {var} = {os.environ[var]}")


def run_intensive_task():
    """
    Simule une tâche intensive pour montrer l'effet des optimisations.
    Cette fonction est utilisée comme exemple pour mesurer l'impact des optimisations.
    """
    print("\n--- Mesure de l'impact des optimisations sur une tâche intensive ---")

    # Mesurer les performances sans optimisations
    print("Exécution de la tâche sans optimisations...")
    start_time = time.time()
    # Simuler une charge de travail intensive (calcul de matrice)
    matrix_size = 1000
    matrix_a = [[i + j for j in range(matrix_size)] for i in range(matrix_size)]
    matrix_b = [[i * j for j in range(matrix_size)] for i in range(matrix_size)]
    result = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    for i in range(matrix_size):
        for j in range(matrix_size):
            for k in range(matrix_size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    baseline_time = time.time() - start_time
    print(f"Temps de calcul sans optimisations: {baseline_time:.2f} secondes")

    # Appliquer les optimisations
    optimizer = optimize_system()

    # Mesurer les performances avec optimisations
    print("Exécution de la tâche avec optimisations...")
    start_time = time.time()
    # Même charge de travail
    matrix_a = [[i + j for j in range(matrix_size)] for i in range(matrix_size)]
    matrix_b = [[i * j for j in range(matrix_size)] for i in range(matrix_size)]
    result = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    for i in range(matrix_size):
        for j in range(matrix_size):
            for k in range(matrix_size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    optimized_time = time.time() - start_time
    print(f"Temps de calcul avec optimisations: {optimized_time:.2f} secondes")

    # Calculer l'amélioration
    if baseline_time > 0:
        improvement = (baseline_time - optimized_time) / baseline_time * 100
        print(f"Amélioration: {improvement:.2f}%")


if __name__ == "__main__":
    print("=================================================")
    print("   EXEMPLE D'UTILISATION DU SYSTEM OPTIMIZER")
    print("=================================================")

    example_with_default_config()
    example_with_custom_config()
    example_optimize_system_helper()
    run_intensive_task()

    print("\n=================================================")
    print("       DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    print("=================================================")
