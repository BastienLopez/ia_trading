#!/usr/bin/env python
"""
Tests pour le module d'optimisation du multithreading et multiprocessing.
"""

import logging
import os
import sys
import time
import unittest

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.utils.threading_optimizer import ThreadingOptimizer, parallel_map

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TestThreadingOptimizer(unittest.TestCase):
    """Tests pour la classe ThreadingOptimizer."""

    def setUp(self):
        """Initialisation pour chaque test."""
        self.optimizer = ThreadingOptimizer()

    def test_detection_resources(self):
        """Teste que les ressources système sont correctement détectées."""
        # Vérifier que les valeurs sont raisonnables
        self.assertGreater(self.optimizer.cpu_count, 0)
        self.assertGreater(self.optimizer.physical_cores, 0)
        self.assertGreaterEqual(self.optimizer.cpu_count, self.optimizer.physical_cores)
        self.assertGreater(self.optimizer.total_memory_gb, 0)

    def test_optimal_workers_calculation(self):
        """Teste que le calcul du nombre optimal de workers est cohérent."""
        optimal = self.optimizer.calculate_optimal_workers()

        # Vérifier que tous les types de tâches sont présents
        self.assertIn("dataloader", optimal)
        self.assertIn("training", optimal)
        self.assertIn("preprocessing", optimal)
        self.assertIn("inference", optimal)

        # Vérifier que les valeurs sont cohérentes
        for task, workers in optimal.items():
            self.assertGreater(workers, 0)
            self.assertLessEqual(workers, self.optimizer.cpu_count)

    def test_dataloader_config(self):
        """Teste que la configuration du dataloader est correcte."""
        # Test avec un petit dataset
        small_config = self.optimizer.get_dataloader_config(
            data_size=100, batch_size=32
        )
        self.assertIn("num_workers", small_config)
        self.assertIn("prefetch_factor", small_config)
        self.assertIn("pin_memory", small_config)

        # Test avec un grand dataset
        large_config = self.optimizer.get_dataloader_config(
            data_size=100000, batch_size=32
        )

        # Le grand dataset devrait avoir plus ou le même nombre de workers que le petit
        self.assertGreaterEqual(
            large_config["num_workers"], small_config["num_workers"]
        )

    def test_thread_limits_configuration(self):
        """Teste que la configuration des limites de threads fonctionne."""
        # Configurer avec des valeurs explicites
        limits = self.optimizer.configure_thread_limits(
            numpy_threads=2, torch_threads=3, omp_threads=4
        )

        self.assertEqual(limits["numpy"], 2)
        self.assertEqual(limits["torch"], 3)
        self.assertEqual(limits["omp"], 4)

        # Vérifier que les variables d'environnement sont définies
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "4")
        self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "2")


# Ajouter cette fonction au niveau global du module pour la rendre picklable
def factorial_test_function(n):
    """Fonction de calcul de factorielle pour les tests."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


class TestParallelProcessing(unittest.TestCase):
    """Tests pour les fonctions de traitement parallèle."""

    def test_parallel_map_threads(self):
        """Teste la fonction parallel_map avec des threads."""

        # Fonction de test: attendre puis retourner l'entrée * 2
        def process_item(x):
            time.sleep(0.01)  # Simuler un traitement
            return x * 2

        # Test avec une liste de 20 éléments
        items = list(range(20))

        # Exécution séquentielle pour comparaison
        start_time = time.time()
        results_seq = [process_item(item) for item in items]
        seq_time = time.time() - start_time

        # Exécution parallèle avec threads
        start_time = time.time()
        results_parallel = parallel_map(
            process_item, items, max_workers=4, use_processes=False
        )
        parallel_time = time.time() - start_time

        # Vérifier que les résultats sont identiques
        self.assertEqual(results_seq, results_parallel)

        # La version parallèle devrait être plus rapide, mais pas garantie
        # donc on ne fait pas de test strict sur le temps
        logger.info(
            f"Temps séquentiel: {seq_time:.4f}s, Temps parallèle: {parallel_time:.4f}s"
        )

    def test_parallel_map_processes(self):
        """Teste la fonction parallel_map avec des processus."""
        # Utiliser la fonction globale au lieu d'une fonction locale
        # pour éviter les problèmes de pickle avec multiprocessing

        # Test avec une liste de nombres pour calculer les factorielles
        items = [100, 200, 300, 400, 500]  # Réduire la taille pour accélérer les tests

        # Exécuter avec des processus (utile pour CPU-bound)
        results = parallel_map(
            factorial_test_function, items, use_processes=True, max_workers=2
        )

        # Vérifier que nous avons obtenu des résultats pour tous les éléments
        self.assertEqual(len(results), len(items))

        # Les résultats devraient être des nombres très grands
        for r in results:
            self.assertGreater(r, 0)


def run_performance_comparison(iterations=5):
    """
    Exécute une comparaison de performance entre différentes configurations.

    Args:
        iterations: Nombre d'itérations pour chaque test
    """
    print("\n=== Comparaison de performance des configurations de threading ===")

    # Simuler une tâche de traitement CPU-intensive
    def cpu_intensive_task(size):
        matrix = [[i * j for j in range(size)] for i in range(size)]
        result = 0
        for i in range(size):
            for j in range(size):
                result += matrix[i][j]
        return result

    # Paramètres du test
    matrix_size = 500
    item_count = 8

    # Liste des configurations à tester
    configs = [
        {"workers": 1, "name": "1 worker (séquentiel)"},
        {"workers": 2, "name": "2 workers"},
        {"workers": 4, "name": "4 workers"},
        {"workers": 8, "name": "8 workers"},
    ]

    results = {}

    # Tester chaque configuration
    for config in configs:
        workers = config["workers"]
        name = config["name"]

        # Exécuter plusieurs fois pour une moyenne stable
        times = []
        for _ in range(iterations):
            items = [matrix_size] * item_count

            start_time = time.time()
            parallel_map(
                cpu_intensive_task, items, max_workers=workers, use_processes=True
            )
            elapsed = time.time() - start_time

            times.append(elapsed)

        # Calculer la moyenne et l'écart-type
        avg_time = sum(times) / len(times)
        results[name] = avg_time

        print(f"{name}: {avg_time:.4f}s")

    # Trouver la configuration la plus rapide
    fastest = min(results, key=results.get)
    speedup = results["1 worker (séquentiel)"] / results[fastest]

    print(f"\nLa configuration la plus rapide est {fastest}")
    print(f"Accélération par rapport au séquentiel: {speedup:.2f}x")

    return results, fastest


if __name__ == "__main__":
    # Exécuter les tests unitaires
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

    # Exécuter la comparaison de performance
    print("\nExécution de la comparaison de performance...")
    results, fastest = run_performance_comparison(iterations=3)

    # Afficher le résumé final
    print("\n=== Résumé des optimisations de threading ===")
    optimizer = ThreadingOptimizer()
    optimal = optimizer.calculate_optimal_workers()

    print(f"Configuration recommandée pour ce système:")
    print(f"- DataLoader: {optimal['dataloader']} workers")
    print(f"- Entraînement: {optimal['training']} threads")
    print(f"- Prétraitement: {optimal['preprocessing']} processus")

    if fastest == "1 worker (séquentiel)":
        print(
            "\nRecommandation: Pour ce type de tâche et ce système, le traitement séquentiel est plus efficace."
        )
    else:
        worker_count = int(fastest.split()[0])
        print(
            f"\nRecommandation: Pour ce type de tâche, utiliser {worker_count} workers est optimal."
        )
        print(
            f"Cette configuration offre une accélération de {results['1 worker (séquentiel)'] / results[fastest]:.2f}x par rapport au traitement séquentiel."
        )
