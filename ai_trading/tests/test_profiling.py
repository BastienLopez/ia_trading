#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading.utils.profiling import ProfilingManager, profile_block, profile_function


# Fonction à profiler pour les tests
def fibonacci(n):
    """Fonction Fibonacci récursive (inefficace) pour tester le profilage."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_iterative(n):
    """Fonction Fibonacci itérative (efficace) pour tester le profilage."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


class TestProfiling(unittest.TestCase):
    """Tests pour le module de profilage."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.profiler = ProfilingManager(output_dir=self.temp_dir.name)

    def tearDown(self):
        """Nettoyage après chaque test."""
        self.temp_dir.cleanup()

    def test_profile_with_cprofile(self):
        """Teste le profilage avec cProfile."""
        # Profiler une fonction simple
        result = self.profiler.profile_with_cprofile(fibonacci_iterative, 20)

        # Vérifier que le résultat contient les bonnes clés
        self.assertIn("result", result)
        self.assertIn("profile_path", result)
        self.assertIn("text_report", result)
        self.assertIn("top_functions", result)

        # Vérifier que le résultat est correct
        self.assertEqual(result["result"], 6765)  # fibonacci(20) = 6765

        # Vérifier que le fichier de profil a été créé
        self.assertTrue(os.path.exists(result["profile_path"]))

    def test_profile_function_decorator(self):
        """Teste le décorateur profile_function."""
        # Capturer la sortie standard
        captured_output = io.StringIO()

        # Créer une fonction décorée
        @profile_function(method="time")
        def test_func():
            return fibonacci_iterative(20)

        # Exécuter la fonction avec la sortie redirigée
        with contextlib.redirect_stdout(captured_output):
            result = test_func()

        # Vérifier que le résultat est correct
        self.assertEqual(result, 6765)

        # Vérifier que le profilage a généré une sortie
        output = captured_output.getvalue()
        self.assertIn("Temps d'exécution pour test_func", output)

    def test_profile_block_context(self):
        """Teste le contexte profile_block."""
        # Capturer la sortie standard
        captured_output = io.StringIO()

        # Utiliser le contexte avec la sortie redirigée
        with contextlib.redirect_stdout(captured_output):
            with profile_block(name="Test Block", method="time"):
                result = fibonacci_iterative(20)

        # Vérifier que le résultat est correct
        self.assertEqual(result, 6765)

        # Vérifier que le profilage a généré une sortie
        output = captured_output.getvalue()
        self.assertIn("Temps d'exécution pour Test Block", output)

    def test_parse_cprofile_stats(self):
        """Teste l'analyse des statistiques cProfile."""
        # Profiler une fonction pour obtenir des statistiques
        result = self.profiler.profile_with_cprofile(fibonacci, 10)

        # Vérifier que les statistiques sont analysées correctement
        self.assertIsInstance(result["top_functions"], list)

        if result["top_functions"]:  # Si des fonctions sont détectées
            # Vérifier que chaque entrée a les bonnes clés
            first_entry = result["top_functions"][0]
            self.assertIn("ncalls", first_entry)
            self.assertIn("tottime", first_entry)
            self.assertIn("percall", first_entry)
            self.assertIn("cumtime", first_entry)
            self.assertIn("function", first_entry)

            # Vérifier que les temps sont des nombres flottants
            self.assertIsInstance(first_entry["tottime"], float)
            self.assertIsInstance(first_entry["percall"], float)
            self.assertIsInstance(first_entry["cumtime"], float)

    @unittest.skipIf(not os.environ.get("RUN_GPU_TESTS"), "Tests GPU désactivés")
    def test_pytorch_profiler(self):
        """
        Teste le profilage PyTorch (exécuté uniquement si PyTorch est disponible
        et que RUN_GPU_TESTS est défini).
        """
        try:
            import torch
            import torch.nn as nn

            # Créer un modèle simple
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 5)

                def forward(self, x):
                    return self.fc(x)

            model = SimpleModel()
            input_data = torch.randn(32, 10)

            # Profiler le modèle
            result = self.profiler.profile_with_torch(model, input_data)

            # Vérifier que le résultat contient les bonnes clés
            self.assertIn("trace_path", result)
            self.assertIn("bottlenecks", result)
            self.assertIn("memory_stats", result)

            # Vérifier que le fichier de trace a été créé
            self.assertTrue(os.path.exists(result["trace_path"]))

        except ImportError:
            self.skipTest("PyTorch n'est pas disponible")

    def test_nsight_availability(self):
        """Teste la détection de disponibilité de NVIDIA Nsight."""
        # Cette méthode ne teste pas l'exécution de Nsight, juste la détection
        availability = self.profiler._check_nsight_available()

        # La valeur peut être True ou False selon l'environnement, on vérifie juste le type
        self.assertIsInstance(availability, bool)


if __name__ == "__main__":
    unittest.main()
