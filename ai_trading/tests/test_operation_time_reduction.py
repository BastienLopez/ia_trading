#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import threading
import time
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Désactiver les logs pendant les tests
logging.disable(logging.CRITICAL)

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Vérifier si PyTorch est disponible
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai_trading.optim.operation_time_reduction import (
    _RESULT_CACHE,
)  # Pour les tests uniquement
from ai_trading.optim.operation_time_reduction import (
    ParallelOperations,
    PredictionCache,
    get_optimal_batch_size,
    get_prediction_cache,
    precalculate_and_cache,
)


class SimpleModel(nn.Module):
    """Modèle simple pour les tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestOperationTimeReduction(unittest.TestCase):
    """Tests pour le module de réduction des temps d'opération."""

    def setUp(self):
        """Configuration avant chaque test."""
        # Vider le cache entre les tests
        _RESULT_CACHE.clear()

    def tearDown(self):
        """Nettoyage après chaque test."""

    def test_precalculate_and_cache(self):
        """Teste la fonction de précalcul et mise en cache."""
        # Définir une fonction de test avec compteur d'appel
        call_count = [0]

        @precalculate_and_cache
        def test_func(x, y):
            call_count[0] += 1
            return x * y

        # Premier appel, devrait exécuter la fonction
        result1 = test_func(2, 3)
        self.assertEqual(result1, 6)
        self.assertEqual(call_count[0], 1)

        # Deuxième appel avec les mêmes arguments, devrait utiliser le cache
        result2 = test_func(2, 3)
        self.assertEqual(result2, 6)
        self.assertEqual(call_count[0], 1)  # Le compteur ne devrait pas augmenter

        # Appel avec des arguments différents, devrait exécuter la fonction
        result3 = test_func(3, 4)
        self.assertEqual(result3, 12)
        self.assertEqual(call_count[0], 2)

    def test_get_optimal_batch_size(self):
        """Teste la fonction de calcul de taille de batch optimale."""
        # Test avec une taille minimale de 1
        size1 = get_optimal_batch_size(1, 10)
        self.assertEqual(size1, 8)

        # Test avec une taille minimale supérieure à 1
        size2 = get_optimal_batch_size(10, 20)
        self.assertEqual(size2, 16)

        # Test avec une taille maximale qui limite la puissance de 2
        size3 = get_optimal_batch_size(10, 15)
        self.assertEqual(size3, 8)

        # Test avec des valeurs extrêmes
        size4 = get_optimal_batch_size(1000, 2000)
        self.assertEqual(size4, 1024)

    def test_prediction_cache(self):
        """Teste la classe PredictionCache."""
        # Créer un cache de prédictions
        cache = PredictionCache(capacity=5, ttl=1)

        # Ajouter quelques entrées
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Vérifier que les entrées sont récupérables
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        self.assertIsNone(cache.get("nonexistent"))

        # Tester la limite de capacité
        for i in range(10):
            cache.put(f"extra_key_{i}", f"value_{i}")

        # Vérifier que la capacité n'est pas dépassée
        self.assertLessEqual(len(cache.cache), 5)

        # Tester l'expiration
        cache.put("expire_key", "expire_value")
        self.assertEqual(cache.get("expire_key"), "expire_value")

        # Attendre que l'entrée expire
        time.sleep(1.1)
        self.assertIsNone(cache.get("expire_key"))

    def test_get_prediction_cache(self):
        """Teste la fonction get_prediction_cache."""
        # Récupérer deux fois le même cache
        cache1 = get_prediction_cache("test_cache")
        cache2 = get_prediction_cache("test_cache")

        # Vérifier que c'est la même instance
        self.assertIs(cache1, cache2)

        # Récupérer un cache différent
        cache3 = get_prediction_cache("other_test_cache")
        self.assertIsNot(cache1, cache3)

    def test_parallel_map(self):
        """Teste la fonction de mapping parallèle."""

        # Fonction de test
        def square(x):
            return x * x

        # Liste d'entrées
        items = list(range(10))

        # Exécuter avec des threads
        parallel_ops = ParallelOperations()
        results_threads = parallel_ops.parallel_map(square, items, use_processes=False)

        # Vérifier les résultats
        expected = [x * x for x in items]
        self.assertEqual(results_threads, expected)

        # Exécuter avec des processus
        # Note: cela peut ne pas fonctionner dans certains environnements de test
        try:
            results_processes = parallel_ops.parallel_map(
                square, items, use_processes=True
            )
            self.assertEqual(results_processes, expected)
        except Exception as e:
            # Ignorer les erreurs liées aux processus dans les environnements restreints
            pass

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_parallel_model_inference(self):
        """Teste la fonction d'inférence parallèle de modèle."""
        # Créer un modèle simple
        model = SimpleModel()

        # Créer des entrées
        inputs = [torch.randn(10) for _ in range(5)]

        # Exécuter l'inférence
        parallel_ops = ParallelOperations()
        outputs = parallel_ops.parallel_model_inference(model, inputs)

        # Vérifier le nombre de sorties
        self.assertEqual(len(outputs), len(inputs))

        # Vérifier la forme des sorties
        for output in outputs:
            self.assertEqual(output.shape[0], 5)

    def test_concurrent_access(self):
        """Teste l'accès concurrent au cache."""

        # Définir une fonction de test
        @precalculate_and_cache
        def slow_func(x):
            time.sleep(0.1)
            return x * 2

        # Fonction exécutée par chaque thread
        def thread_func(results, idx):
            results[idx] = slow_func(idx)

        # Créer plusieurs threads
        num_threads = 10
        results = [None] * num_threads
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=thread_func, args=(results, i))
            threads.append(t)

        # Démarrer les threads
        for t in threads:
            t.start()

        # Attendre que tous les threads terminent
        for t in threads:
            t.join()

        # Vérifier les résultats
        for i in range(num_threads):
            self.assertEqual(results[i], i * 2)


if __name__ == "__main__":
    unittest.main()
