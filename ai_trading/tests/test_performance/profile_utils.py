"""
Utilitaires de profilage pour l'analyse des performances.

Ce module fournit des outils pour profiler en détail les fonctions
et identifier les goulots d'étranglement dans les fonctionnalités transversales.
"""

import cProfile
import functools
import io
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

from ai_trading.utils.advanced_logging import get_logger

# Logger pour ce module
logger = get_logger("ai_trading.tests.performance.profile_utils")


@contextmanager
def profile_time(name=None):
    """
    Context manager pour mesurer le temps d'exécution.

    Args:
        name: Nom de l'opération pour l'affichage
    """
    operation = name or "Opération"
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{operation} terminée en {elapsed:.6f} secondes")


@contextmanager
def profile_memory(name=None):
    """
    Context manager pour mesurer l'utilisation mémoire.

    Args:
        name: Nom de l'opération pour l'affichage
    """
    operation = name or "Opération"
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    yield

    current_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = current_snapshot.compare_to(start_snapshot, "lineno")

    logger.info(f"Utilisation mémoire pour {operation}:")
    total = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    logger.info(f"Mémoire totale utilisée: {total / 1024 / 1024:.2f} MB")

    # Afficher les 5 principales sources d'allocation mémoire
    top_stats = stats[:5]
    for stat in top_stats:
        if stat.size_diff > 0:  # Afficher seulement les augmentations
            logger.info(f"{stat.size_diff / 1024:.1f} KB: {stat.traceback.format()[0]}")


def profile_function(func=None, output_file=None, sort_by="cumulative"):
    """
    Décorateur pour profiler une fonction avec cProfile.

    Args:
        func: Fonction à profiler
        output_file: Fichier de sortie pour les statistiques (optionnel)
        sort_by: Critère de tri pour les statistiques (par défaut: 'cumulative')

    Returns:
        Fonction décorée
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                result = f(*args, **kwargs)
                profiler.disable()
                return result
            finally:
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
                ps.print_stats(20)  # Limiter à 20 entrées

                # Log the profiling info
                logger.info(f"Profil de {f.__name__}:\n{s.getvalue()}")

                # Sauvegarder dans un fichier si demandé
                if output_file:
                    with open(output_file, "w") as out_file:
                        ps = pstats.Stats(profiler, stream=out_file).sort_stats(sort_by)
                        ps.print_stats()

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def run_full_profile(
    func, args=None, kwargs=None, output_dir=None, include_memory=True
):
    """
    Exécute un profilage complet (temps CPU et mémoire) d'une fonction.

    Args:
        func: Fonction à profiler
        args: Arguments positionnels (liste)
        kwargs: Arguments nommés (dict)
        output_dir: Répertoire pour les fichiers de sortie
        include_memory: Activer le profilage mémoire

    Returns:
        tuple: (résultat de la fonction, temps d'exécution, stats CPU, stats mémoire)
    """
    args = args or []
    kwargs = kwargs or {}

    # Créer le répertoire de sortie si nécessaire
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_filename = f"{func.__name__}_profile_{time.strftime('%Y%m%d_%H%M%S')}"
        cpu_file = output_dir / f"{base_filename}_cpu.prof"
        memory_file = output_dir / f"{base_filename}_memory.txt"
    else:
        cpu_file = None
        memory_file = None

    # Profiler le temps d'exécution
    profiler = cProfile.Profile()

    if include_memory:
        tracemalloc.start()
        memory_start = tracemalloc.take_snapshot()

    start_time = time.time()
    profiler.enable()

    # Exécuter la fonction
    result = func(*args, **kwargs)

    # Arrêter les profileurs
    profiler.disable()
    end_time = time.time()

    if include_memory:
        memory_end = tracemalloc.take_snapshot()
        tracemalloc.stop()
        memory_stats = memory_end.compare_to(memory_start, "lineno")
    else:
        memory_stats = None

    # Traiter les statistiques CPU
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)

    # Sauvegarder les statistiques CPU
    if cpu_file:
        with open(cpu_file, "w") as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
            stats.print_stats()
        logger.info(f"Statistiques CPU sauvegardées dans {cpu_file}")

    # Sauvegarder les statistiques mémoire
    if include_memory and memory_file:
        with open(memory_file, "w") as f:
            f.write("Statistiques d'utilisation mémoire\n")
            f.write("=" * 50 + "\n\n")

            total = sum(stat.size_diff for stat in memory_stats if stat.size_diff > 0)
            f.write(f"Mémoire totale utilisée: {total / 1024 / 1024:.2f} MB\n\n")

            f.write("Top 20 allocations mémoire:\n")
            for stat in memory_stats[:20]:
                if stat.size_diff > 0:
                    f.write(
                        f"{stat.size_diff / 1024:.1f} KB: {stat.traceback.format()[0]}\n"
                    )
                    for line in stat.traceback.format()[1:]:
                        f.write(f"    {line}\n")
                    f.write("\n")

        logger.info(f"Statistiques mémoire sauvegardées dans {memory_file}")

    # Afficher un résumé
    elapsed = end_time - start_time
    logger.info(f"Profilage de {func.__name__} terminé en {elapsed:.4f} secondes")

    if include_memory:
        total_memory = sum(
            stat.size_diff for stat in memory_stats if stat.size_diff > 0
        )
        logger.info(f"Mémoire utilisée: {total_memory / 1024 / 1024:.2f} MB")

    logger.info(f"Statistiques CPU:\n{s.getvalue()}")

    return {
        "result": result,
        "elapsed_time": elapsed,
        "cpu_stats": profiler,
        "memory_stats": memory_stats if include_memory else None,
        "cpu_file": cpu_file,
        "memory_file": memory_file,
    }


def profile_comparison(
    func1, func2, args1=None, args2=None, kwargs1=None, kwargs2=None, names=None
):
    """
    Compare les performances de deux fonctions.

    Args:
        func1: Première fonction
        func2: Deuxième fonction
        args1: Arguments positionnels pour func1
        args2: Arguments positionnels pour func2
        kwargs1: Arguments nommés pour func1
        kwargs2: Arguments nommés pour func2
        names: Tuple avec les noms des fonctions (optionnel)

    Returns:
        dict: Résultats de la comparaison
    """
    args1 = args1 or []
    args2 = args2 or []
    kwargs1 = kwargs1 or {}
    kwargs2 = kwargs2 or {}

    name1 = names[0] if names and len(names) > 0 else func1.__name__
    name2 = names[1] if names and len(names) > 1 else func2.__name__

    logger.info(f"Comparaison des performances: {name1} vs {name2}")

    # Profiler la première fonction
    with profile_time(name1) as p1:
        if isinstance(args1, list):
            result1 = func1(*args1, **kwargs1)
        else:
            result1 = func1(args1, **kwargs1)
    time1 = p1.__exit__(None, None, None) or 0

    # Profiler la deuxième fonction
    with profile_time(name2) as p2:
        if isinstance(args2, list):
            result2 = func2(*args2, **kwargs2)
        else:
            result2 = func2(args2, **kwargs2)
    time2 = p2.__exit__(None, None, None) or 0

    # Calculer la différence de performance
    if time1 > 0:
        difference = (time2 - time1) / time1 * 100
        faster = name1 if time1 < time2 else name2
        slower = name2 if time1 < time2 else name1

        logger.info(f"{faster} est {abs(difference):.2f}% plus rapide que {slower}")
    else:
        difference = 0
        logger.info("Impossible de calculer la différence (temps trop court)")

    return {
        "func1": {"name": name1, "time": time1, "result": result1},
        "func2": {"name": name2, "time": time2, "result": result2},
        "difference_percent": difference,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utilitaires de profilage")
    parser.add_argument(
        "--test", action="store_true", help="Exécuter des tests de démonstration"
    )

    args = parser.parse_args()

    if args.test:
        # Test des fonctionnalités de profilage
        logger.info("Test des utilitaires de profilage")

        # Fonction de test 1
        def test_function_1():
            """Fonction de test avec allocation mémoire modérée."""
            result = []
            for i in range(100000):
                result.append(i * 2)
            time.sleep(0.1)
            return len(result)

        # Fonction de test 2
        def test_function_2():
            """Fonction de test avec allocation mémoire importante."""
            result = []
            for i in range(500000):
                result.append(i * 2)
            time.sleep(0.2)
            return len(result)

        # Tester avec le décorateur
        logger.info("Test du décorateur profile_function")
        decorated = profile_function(test_function_1)
        decorated()

        # Tester avec run_full_profile
        logger.info("Test de run_full_profile")
        profile_result = run_full_profile(test_function_2, output_dir="profile_results")

        # Tester la comparaison
        logger.info("Test de profile_comparison")
        comparison = profile_comparison(
            test_function_1,
            test_function_2,
            names=("Fonction légère", "Fonction lourde"),
        )
