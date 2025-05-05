#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import tempfile
from pathlib import Path
import logging

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

from ai_trading.optim.critical_operations import (
    use_jit_script,
    fast_matrix_multiply,
    optimize_model_with_compile,
    enable_cudnn_benchmark,
    VectorizedOperations,
    configure_performance_settings,
    benchmark_function
)


class TestCriticalOperations(unittest.TestCase):
    """Tests pour le module d'optimisation des opérations critiques."""

    def setUp(self):
        """Configuration avant chaque test."""
        pass
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        pass
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_use_jit_script_decorator(self):
        """Teste le décorateur use_jit_script sur une fonction simple."""
        @use_jit_script
        def add_tensors(a, b):
            return a + b
        
        # Vérifier que la fonction est bien scriptée
        self.assertTrue(hasattr(add_tensors, '_compilation_unit'))
        
        # Tester la fonction
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = add_tensors(a, b)
        
        # Vérifier le résultat
        expected = torch.tensor([5, 7, 9])
        self.assertTrue(torch.all(result == expected))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_fast_matrix_multiply(self):
        """Teste la fonction de multiplication matricielle optimisée avec JIT."""
        # Créer des tenseurs pour le test
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        
        # Appeler la fonction optimisée
        result = fast_matrix_multiply(a, b)
        
        # Calculer le résultat attendu
        expected = torch.matmul(a, b)
        
        # Vérifier l'égalité
        self.assertTrue(torch.allclose(result, expected))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_optimize_model_with_compile(self):
        """Teste l'optimisation d'un modèle avec torch.compile."""
        if not hasattr(torch, 'compile'):
            self.skipTest("torch.compile n'est pas disponible dans cette version de PyTorch")
        
        # Créer un modèle simple
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.fc(x)
        
        # Créer une instance du modèle
        model = SimpleModel()
        
        try:
            # Optimiser le modèle
            optimized_model = optimize_model_with_compile(model)
            
            # Vérifier que le modèle a été optimisé
            self.assertIsNotNone(optimized_model)
            
            # Tester que le modèle optimisé fonctionne
            input_data = torch.randn(1, 10)
            output = optimized_model(input_data)
            
            # Vérifier la forme de la sortie
            self.assertEqual(output.shape, torch.Size([1, 5]))
        except RuntimeError as e:
            # Si torch.compile n'est pas supporté sur cette plateforme, skipTest
            self.skipTest(f"torch.compile a échoué: {str(e)}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_enable_cudnn_benchmark(self):
        """Teste l'activation de cudnn.benchmark."""
        # Sauvegarder l'état initial
        initial_state = torch.backends.cudnn.benchmark
        
        try:
            # Activer cudnn.benchmark
            enable_cudnn_benchmark(True)
            
            # Vérifier que l'état a changé comme prévu si CUDA est disponible
            if torch.cuda.is_available():
                self.assertTrue(torch.backends.cudnn.benchmark)
            
            # Désactiver cudnn.benchmark
            enable_cudnn_benchmark(False)
            
            # Vérifier que l'état a changé comme prévu si CUDA est disponible
            if torch.cuda.is_available():
                self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            # Restaurer l'état initial
            torch.backends.cudnn.benchmark = initial_state
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_vectorized_operations(self):
        """Teste les opérations vectorisées."""
        # Créer une instance de la classe
        vec_ops = VectorizedOperations()
        
        # Tester batch_matrix_vector_product
        batch_size = 3
        matrix_dim_m = 4
        matrix_dim_n = 5
        
        # Créer des tenseurs pour le test
        matrices = torch.randn(batch_size, matrix_dim_m, matrix_dim_n)
        vectors = torch.randn(batch_size, matrix_dim_n)
        
        # Calculer avec vmap si disponible
        try:
            result = vec_ops.batch_matrix_vector_product(matrices, vectors)
            
            # Calculer le résultat attendu manuellement
            expected = torch.stack([
                torch.mv(matrices[i], vectors[i]) for i in range(batch_size)
            ])
            
            # Vérifier l'égalité
            self.assertTrue(torch.allclose(result, expected, atol=1e-5))
            
            # Vérifier la forme de la sortie
            self.assertEqual(result.shape, torch.Size([batch_size, matrix_dim_m]))
        except RuntimeError as e:
            self.skipTest(f"torch.vmap a échoué: {str(e)}")
        
        # Tester batch_pairwise_distance
        n_points = 3
        m_points = 2
        dim = 4
        
        # Créer des tenseurs pour le test
        x = torch.randn(batch_size, n_points, dim)
        y = torch.randn(batch_size, m_points, dim)
        
        try:
            # Calculer avec vmap
            result = vec_ops.batch_pairwise_distance(x, y)
            
            # Calculer le résultat attendu manuellement
            expected = torch.stack([
                torch.cdist(x[i].unsqueeze(0), y[i].unsqueeze(0)).squeeze(0)
                for i in range(batch_size)
            ])
            
            # Vérifier l'égalité
            self.assertTrue(torch.allclose(result, expected, atol=1e-5))
            
            # Vérifier la forme de la sortie
            self.assertEqual(result.shape, torch.Size([batch_size, n_points, m_points]))
        except RuntimeError as e:
            self.skipTest(f"torch.vmap a échoué: {str(e)}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_benchmark_function(self):
        """Teste le décorateur benchmark_function."""
        # Fonction à benchmarker
        @benchmark_function
        def test_func(n):
            total = 0
            for i in range(n):
                total += i
            return total
        
        # Exécuter la fonction
        result = test_func(1000)
        
        # Vérifier que le résultat est correct
        expected = sum(range(1000))
        self.assertEqual(result, expected)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch n'est pas disponible")
    def test_configure_performance_settings(self):
        """Teste la configuration des paramètres de performance."""
        # Sauvegarder l'état initial de cudnn.benchmark
        initial_benchmark_state = torch.backends.cudnn.benchmark
        
        try:
            # Appeler la fonction de configuration
            configure_performance_settings()
            
            # Vérifier que cudnn.benchmark a été activé si CUDA est disponible
            if torch.cuda.is_available():
                self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            # Restaurer l'état initial
            torch.backends.cudnn.benchmark = initial_benchmark_state


if __name__ == "__main__":
    unittest.main() 