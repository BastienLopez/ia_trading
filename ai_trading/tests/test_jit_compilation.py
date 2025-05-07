#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading.utils.jit_compilation import (
    TF_AVAILABLE,
    TORCH_AVAILABLE,
    XLA_AVAILABLE,
    TorchScriptCompiler,
    compile_model,
    enable_tensorflow_xla,
    optimize_function,
)


# Simple fonction pour tester l'optimisation de fonction
def torch_add(a, b):
    return a + b


class TestJitCompilation(unittest.TestCase):
    """Tests pour le module de compilation JIT."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Créer le répertoire info_retour/models/jit pour les tests
        self.info_retour_dir = Path("ai_trading/info_retour/models/jit").absolute()
        os.makedirs(self.info_retour_dir, exist_ok=True)

    def tearDown(self):
        """Nettoyage après chaque test."""
        self.temp_dir.cleanup()

        # Nettoyer les fichiers de test créés dans info_retour/models/jit
        for file in self.info_retour_dir.glob("test_*.pt"):
            if file.exists():
                os.remove(file)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_torchscript_compiler_init(self):
        """Teste l'initialisation du compilateur TorchScript."""

        # Initialiser le compilateur avec un répertoire personnalisé
        compiler = TorchScriptCompiler(save_dir=self.temp_dir.name)

        # Vérifier que le répertoire est correctement configuré
        self.assertEqual(str(compiler.save_dir), self.temp_dir.name)

        # Vérifier que le répertoire existe
        self.assertTrue(os.path.exists(self.temp_dir.name))

        # Initialiser le compilateur sans répertoire personnalisé (utilise info_retour/models/jit par défaut)
        compiler_default = TorchScriptCompiler()

        # Vérifier que le répertoire par défaut est correctement configuré (en tenant compte des backslashes sur Windows)
        default_path = str(compiler_default.save_dir)
        self.assertTrue(
            "info_retour/models/jit" in default_path.replace("\\", "/")
            or "info_retour\\models\\jit" in default_path,
            f"Chemin attendu non trouvé dans {default_path}",
        )

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_trace_model(self):
        """Teste le traçage d'un modèle PyTorch."""
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
        input_data = torch.randn(1, 10)

        # Initialiser le compilateur par défaut (utilise info_retour/models/jit)
        compiler = TorchScriptCompiler()

        # Tracer le modèle
        traced_model = compiler.trace_model(
            model=model, example_inputs=input_data, name="test_trace_model"
        )

        # Vérifier que le modèle tracé est du bon type
        self.assertIsInstance(traced_model, torch.jit.ScriptModule)

        # Vérifier que le fichier de modèle a été créé au bon endroit
        model_file_path = self.info_retour_dir / "test_trace_model.pt"
        self.assertTrue(os.path.exists(model_file_path))

        # Vérifier que le modèle tracé donne le même résultat que le modèle original
        with torch.no_grad():
            original_output = model(input_data)
            traced_output = traced_model(input_data)

            self.assertTrue(torch.allclose(original_output, traced_output, atol=1e-5))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_script_model(self):
        """Teste le scripting d'un modèle PyTorch."""
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
        input_data = torch.randn(1, 10)

        # Initialiser le compilateur par défaut (utilise info_retour/models/jit)
        compiler = TorchScriptCompiler()

        # Scripter le modèle
        scripted_model = compiler.script_model(model=model, name="test_script_model")

        # Vérifier que le modèle scripté est du bon type
        self.assertIsInstance(scripted_model, torch.jit.ScriptModule)

        # Vérifier que le fichier de modèle a été créé au bon endroit
        model_file_path = self.info_retour_dir / "test_script_model.pt"
        self.assertTrue(os.path.exists(model_file_path))

        # Vérifier que le modèle scripté donne le même résultat que le modèle original
        with torch.no_grad():
            original_output = model(input_data)
            scripted_output = scripted_model(input_data)

            self.assertTrue(torch.allclose(original_output, scripted_output, atol=1e-5))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_benchmark_model(self):
        """Teste le benchmark entre modèle original et compilé."""
        import os
        from pathlib import Path

        import torch
        import torch.nn as nn

        # Créer un modèle simple mais avec suffisamment de calculs pour être mesurable
        class BenchmarkModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(100, 200),
                    nn.ReLU(),
                    nn.Linear(200, 100),
                    nn.ReLU(),
                    nn.Linear(100, 50),
                )

            def forward(self, x):
                return self.layers(x)

        model = BenchmarkModel()
        input_data = torch.randn(32, 100)

        # Créer le dossier pour les tests
        info_retour_dir = Path("ai_trading/info_retour/models/jit").absolute()
        os.makedirs(info_retour_dir, exist_ok=True)

        # Initialiser le compilateur sans spécifier de save_dir (utilise info_retour/models/jit par défaut)
        compiler = TorchScriptCompiler()

        # Scripter le modèle avec un nom pour le sauvegarder
        scripted_model = compiler.script_model(model=model, name="benchmark_test_model")

        # Vérifier que le fichier a été créé au bon endroit
        model_file_path = info_retour_dir / "benchmark_test_model.pt"
        self.assertTrue(
            os.path.exists(model_file_path),
            f"Le fichier de modèle n'a pas été créé à l'emplacement attendu: {model_file_path}",
        )

        # Exécuter le benchmark
        results = compiler.benchmark_model(
            model=model,
            scripted_model=scripted_model,
            input_data=input_data,
            num_warmup=5,
            num_iter=10,
        )

        # Vérifier que les résultats contiennent les bonnes clés
        self.assertIn("original_time", results)
        self.assertIn("scripted_time", results)
        self.assertIn("speedup", results)

        # Les temps doivent être positifs
        self.assertGreater(results["original_time"], 0)
        self.assertGreater(results["scripted_time"], 0)

        # Nettoyer après le test
        if os.path.exists(model_file_path):
            os.remove(model_file_path)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_load_compiled_model(self):
        """Teste le chargement d'un modèle compilé."""
        import torch
        import torch.nn as nn

        # Créer et sauvegarder un modèle
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()

        # Initialiser le compilateur par défaut (utilise info_retour/models/jit)
        compiler = TorchScriptCompiler()

        # Scripter et sauvegarder le modèle
        _ = compiler.script_model(model=model, name="test_model_to_load")

        # Vérifier que le fichier a été créé
        model_file_path = self.info_retour_dir / "test_model_to_load.pt"
        self.assertTrue(os.path.exists(model_file_path))

        # Charger le modèle
        loaded_model = compiler.load_compiled_model("test_model_to_load")

        # Vérifier que le modèle chargé est du bon type
        self.assertIsInstance(loaded_model, torch.jit.ScriptModule)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_optimize_function(self):
        """Teste l'optimisation d'une fonction avec TorchScript."""
        import torch

        # Optimiser une simple fonction d'addition
        optimized_add = optimize_function(torch_add)

        # Créer des tenseurs pour tester
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        # Vérifier que la fonction optimisée donne le même résultat
        original_result = torch_add(a, b)
        optimized_result = optimized_add(a, b)

        self.assertTrue(torch.allclose(original_result, optimized_result))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_compile_model_torch(self):
        """Teste la fonction utilitaire compile_model avec PyTorch."""
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
        input_data = torch.randn(1, 10)

        # Compiler le modèle
        compiled_model = compile_model(
            model=model, example_inputs=input_data, framework="torch", method="trace"
        )

        # Vérifier que le modèle compilé est du bon type
        self.assertIsInstance(compiled_model, torch.jit.ScriptModule)

        # Vérifier que le modèle compilé donne le même résultat
        with torch.no_grad():
            original_output = model(input_data)
            compiled_output = compiled_model(input_data)

            self.assertTrue(torch.allclose(original_output, compiled_output, atol=1e-5))

    @unittest.skipIf(
        not TF_AVAILABLE or not XLA_AVAILABLE,
        "TensorFlow avec XLA n'est pas disponible",
    )
    def test_tensorflow_xla(self):
        """Teste l'activation de XLA pour TensorFlow."""
        enabled = enable_tensorflow_xla()
        self.assertIsInstance(enabled, bool)


if __name__ == "__main__":
    unittest.main()
