#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation des outils de profilage pour mesurer les performances
des différentes parties du code.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Importer les outils de profilage
from ai_trading.utils.profiling import ProfilingManager, profile_block, profile_function


# Fonction d'exemple pour démontrer le profilage cProfile
def process_data(size=1000000):
    """Fonction qui effectue des opérations intensives sur des données."""

    # Création d'un grand tableau
    data = np.random.random((size // 1000, 1000))

    # Opérations coûteuses
    result1 = np.mean(data, axis=1)
    result2 = np.std(data, axis=0)
    result3 = np.linalg.norm(data, axis=1)

    # Simulation d'une opération de tri
    sorted_data = np.sort(data, axis=1)

    # Calcul final
    final_result = np.sum(result1) + np.sum(result2) + np.sum(result3)

    return final_result


# Fonction optimisée pour comparer avec la version non optimisée
def process_data_optimized(size=1000000):
    """Version optimisée de la fonction de traitement des données."""

    # Création d'un grand tableau (identique pour comparaison équitable)
    data = np.random.random((size // 1000, 1000))

    # Calcul des statistiques en une seule passe
    result1 = np.mean(data, axis=1)

    # Utilisation de views au lieu de copies
    result3 = np.sqrt(np.sum(data * data, axis=1))

    # Éviter le tri complet si seules quelques valeurs sont nécessaires
    # Dans cet exemple, on simule qu'on a juste besoin des valeurs min/max
    min_vals = np.min(data, axis=1)
    max_vals = np.max(data, axis=1)

    # Calcul final optimisé
    final_result = (
        np.sum(result1) + np.sum(result3) + np.sum(min_vals) + np.sum(max_vals)
    )

    return final_result


# Exemple d'utilisation du décorateur de profilage
@profile_function(method="cprofile")
def run_intensive_task():
    """Fonction qui exécute une tâche intensive avec profilage."""
    return process_data(size=100000)


def main():
    """Fonction principale démontrant plusieurs méthodes de profilage."""

    print("=" * 80)
    print("EXEMPLE DE PROFILAGE AVEC AI TRADING")
    print("=" * 80)

    # Créer un gestionnaire de profilage avec le bon chemin
    output_dir = (
        Path(__file__).parent.parent / "info_retour" / "optimisation" / "profiling"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler = ProfilingManager(output_dir=str(output_dir))
    print(f"Les résultats de profilage seront sauvegardés dans: {profiler.output_dir}")

    # 1. Profilage simple avec cProfile
    print("\n1. PROFILAGE AVEC CPROFILE")
    print("-" * 50)
    result = profiler.profile_with_cprofile(process_data, 50000)
    print(f"Résultat de la fonction: {result['result']}")
    print(f"Rapport de profilage sauvegardé dans: {result['profile_path']}")
    print("\nTop 5 fonctions les plus lentes:")
    for i, func in enumerate(result["top_functions"][:5], 1):
        print(
            f"{i}. {func['function']} - {func['cumtime']:.4f}s cumulé, {func['tottime']:.4f}s total"
        )

    # 2. Comparaison de deux implémentations avec le gestionnaire de profilage
    print("\n2. COMPARAISON DE DEUX IMPLÉMENTATIONS")
    print("-" * 50)

    # Profiler la version non optimisée
    result1 = profiler.profile_with_cprofile(process_data, 50000)

    # Profiler la version optimisée
    result2 = profiler.profile_with_cprofile(process_data_optimized, 50000)

    # Comparer les résultats
    speedup = (
        result1["top_functions"][0]["cumtime"] / result2["top_functions"][0]["cumtime"]
        if result2["top_functions"]
        else 0
    )

    print(f"Version non optimisée: {result1['result']}")
    print(f"Version optimisée: {result2['result']}")
    print(f"Accélération: {speedup:.2f}x")

    # 3. Utilisation du contexte de profilage
    print("\n3. UTILISATION DU CONTEXTE DE PROFILAGE")
    print("-" * 50)

    print("Profilage d'un bloc de code avec le temps d'exécution:")
    with profile_block(name="Traitement par blocs", method="time"):
        # Simulation d'un traitement par blocs
        for i in range(3):
            print(f"  - Traitement du bloc {i+1}/3...")
            time.sleep(0.2)  # Simuler un traitement
            np.random.random((1000, 1000))  # Création de données

    # 4. Utilisation du décorateur de profilage
    print("\n4. UTILISATION DU DÉCORATEUR DE PROFILAGE")
    print("-" * 50)
    result = run_intensive_task()
    print(f"Résultat de la tâche: {result}")

    # 5. Profilage conditionnel basé sur les variables d'environnement
    print("\n5. PROFILAGE CONDITIONNEL")
    print("-" * 50)

    if os.environ.get("PROFILE_ENABLED", "0") == "1":
        print("Profilage activé via la variable d'environnement PROFILE_ENABLED")
        with profile_block(name="Tâche conditionnelle", method="cprofile"):
            process_data(10000)
    else:
        print("Profilage désactivé. Pour activer, définissez PROFILE_ENABLED=1")
        print("Exemple: PROFILE_ENABLED=1 python profiling_example.py")

    print("\nExemple de profilage terminé!")


if __name__ == "__main__":
    main()
