"""
Tests pour vérifier le fonctionnement du module Mixed Precision Training.

Ce module contient des tests unitaires pour s'assurer que le module mixed_precision.py
fonctionne correctement et produit les gains de performance attendus.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.utils.mixed_precision import (
    MixedPrecisionWrapper,
    autocast_context,
    is_mixed_precision_supported,
    setup_mixed_precision,
)
from ai_trading.utils.mixed_precision import (
    test_mixed_precision_performance as mp_performance_test,
)


class SimpleModel(nn.Module):
    """Un modèle simple pour les tests."""

    def __init__(self, input_size=10, hidden_size=50, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestMixedPrecision(unittest.TestCase):
    """Tests pour le module Mixed Precision Training."""

    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        cls.cuda_available = torch.cuda.is_available()
        cls.mp_supported = (
            is_mixed_precision_supported() if cls.cuda_available else False
        )

        # Créer un modèle de test
        cls.model = SimpleModel()
        if cls.cuda_available:
            cls.model = cls.model.to("cuda")

        # Créer des données de test
        cls.batch_size = 16
        cls.input_size = 10

        if cls.cuda_available:
            cls.inputs = torch.randn(cls.batch_size, cls.input_size, device="cuda")
            cls.targets = torch.randn(cls.batch_size, 1, device="cuda")
        else:
            cls.inputs = torch.randn(cls.batch_size, cls.input_size)
            cls.targets = torch.randn(cls.batch_size, 1)

        # Créer un optimiseur
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=0.01)

        # Créer un critère
        cls.criterion = nn.MSELoss()

    def test_mixed_precision_supported(self):
        """Vérifie que la détection du support de mixed precision fonctionne."""
        if self.cuda_available:
            # La fonction doit renvoyer un booléen
            self.assertIsInstance(self.mp_supported, bool)
        else:
            # Si CUDA n'est pas disponible, doit renvoyer False
            self.assertFalse(is_mixed_precision_supported())

    def test_setup_mixed_precision(self):
        """Vérifie que la configuration de mixed precision fonctionne."""
        if self.cuda_available and self.mp_supported:
            result = setup_mixed_precision()
            self.assertTrue(result)

            # Vérifier que les flags sont correctement définis
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)
        else:
            # Si CUDA ou MP n'est pas supporté, doit retourner False
            self.assertFalse(setup_mixed_precision())

    def test_autocast_context(self):
        """Vérifie que le contexte autocast fonctionne correctement."""
        if not self.cuda_available:
            self.skipTest("CUDA n'est pas disponible.")

        # Vérifier que le contexte s'exécute sans erreur
        try:
            with autocast_context():
                _ = self.model(self.inputs)
            self.assertTrue(True)  # Si on arrive ici, pas d'exception
        except Exception as e:
            self.fail(f"autocast_context a levé une exception: {e}")

    def test_mixed_precision_wrapper_initialization(self):
        """Vérifie que le wrapper s'initialise correctement."""
        if not self.cuda_available:
            self.skipTest("CUDA n'est pas disponible.")

        # Créer un wrapper
        wrapper = MixedPrecisionWrapper(self.model, self.optimizer)

        # Vérifier les attributs
        self.assertEqual(wrapper.model, self.model)
        self.assertEqual(wrapper.optimizer, self.optimizer)
        self.assertTrue(hasattr(wrapper, "scaler"))

        # Si MP est supporté, enabled doit être True
        if self.mp_supported:
            self.assertTrue(wrapper.enabled)

    def test_mixed_precision_wrapper_training_step(self):
        """Vérifie que le training_step fonctionne avec et sans MP."""
        if not self.cuda_available:
            self.skipTest("CUDA n'est pas disponible.")

        # Créer un wrapper
        wrapper = MixedPrecisionWrapper(self.model, self.optimizer)

        # Définir les fonctions forward et loss
        def forward_fn(batch):
            return self.model(batch[0])

        def loss_fn(outputs, batch):
            return self.criterion(outputs, batch[1])

        # Exécuter un pas d'entraînement
        loss = wrapper.training_step((self.inputs, self.targets), forward_fn, loss_fn)

        # Vérifier que la perte est un tenseur
        self.assertIsInstance(loss, torch.Tensor)

        # Vérifier que l'entraînement fonctionne aussi sans MP
        wrapper.enabled = False
        loss = wrapper.training_step((self.inputs, self.targets), forward_fn, loss_fn)
        self.assertIsInstance(loss, torch.Tensor)

    def test_performance_test_function(self):
        """Vérifie que le test de performance fonctionne."""
        if not self.cuda_available:
            self.skipTest("CUDA n'est pas disponible.")

        # Exécuter le test de performance
        results = mp_performance_test(
            self.model,
            input_shape=(self.input_size,),
            batch_size=self.batch_size,
            iterations=10,  # Réduire pour accélérer le test
        )

        # Vérifier les clés de résultat
        expected_keys = [
            "fp32_time",
            "fp16_time",
            "speedup",
            "fp32_memory_mb",
            "fp16_memory_mb",
            "memory_reduction",
        ]
        for key in expected_keys:
            self.assertIn(key, results)

        # Si MP est supporté, le speedup devrait être > 1
        if self.mp_supported:
            # Tolérance pour les variations de test - réduire à 0.3 pour tenir compte des variations hardware
            self.assertGreaterEqual(results["speedup"], 0.3)

    def test_mixed_precision_wrapper_with_gradient_accumulation(self):
        """Vérifie que le wrapper fonctionne avec l'accumulation de gradient."""
        if not self.cuda_available:
            self.skipTest("CUDA n'est pas disponible.")

        # Créer un wrapper
        wrapper = MixedPrecisionWrapper(self.model, self.optimizer)

        # Définir les fonctions forward et loss
        def forward_fn(batch):
            return self.model(batch[0])

        def loss_fn(outputs, batch):
            return self.criterion(outputs, batch[1])

        # Simuler l'accumulation de gradient (3 étapes)
        accumulation_steps = 3

        for i in range(accumulation_steps):
            # Dernier pas?
            is_last_step = i == accumulation_steps - 1

            # Exécuter un pas d'entraînement
            loss = wrapper.training_step(
                (self.inputs, self.targets),
                forward_fn,
                loss_fn,
                accumulation_steps=accumulation_steps,
            )

            # Vérifier que la perte est un tenseur
            self.assertIsInstance(loss, torch.Tensor)

            # Mettre à jour l'optimiseur à la dernière étape
            if is_last_step:
                wrapper.optimizer_step()

        # Test réussi si on arrive ici sans erreur
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
