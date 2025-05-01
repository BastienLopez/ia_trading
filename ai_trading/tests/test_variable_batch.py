import unittest
import sys
import numpy as np
import torch
import psutil
import time
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.variable_batch import VariableBatchSampler, BatchOptimizer

# Créer un tampon simple pour les tests
class MockBuffer:
    def __init__(self, size=1000):
        self.data = [(torch.randn(4), i) for i in range(size)]
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.data), batch_size)
        return [self.data[i] for i in indices]
    
    def sample_batch(self, batch_size, beta=None):
        samples = self.sample(batch_size)
        states = torch.stack([s[0] for s in samples])
        values = torch.tensor([s[1] for s in samples])
        info = {"batch_size": batch_size, "beta": beta}
        return (states, values), info

# Fonction de perte simple pour les tests
def _test_loss_fn(model, states, actions=None, rewards=None, next_states=None, dones=None):
    """
    Fonction de perte simple pour les tests.
    Accepte soit juste les états, soit le tuple complet (states, actions, rewards, next_states, dones).
    """
    if actions is None:  # Interface simplifiée pour tests
        return ((model(states) - torch.mean(states, dim=1))**2).mean()
    else:
        # Interface RL standard
        q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = rewards  # Version simplifiée pour les tests
        return ((q_values - targets)**2).mean()

# Modèle simple pour les tests
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

class TestVariableBatchSampler(unittest.TestCase):
    
    def setUp(self):
        """Prépare l'environnement de test"""
        # Créer un buffer pour les tests
        self.buffer = MockBuffer(size=1000)
        
        # Créer l'échantillonneur avec des paramètres de test
        self.sampler = VariableBatchSampler(
            buffer=self.buffer,
            base_batch_size=32,
            min_batch_size=16,
            max_batch_size=128,
            target_gpu_util=0.8,
            target_cpu_util=0.7,
            target_ram_util=0.9,
            adaptation_speed=0.1,
            check_interval=5
        )
    
    def test_initialization(self):
        """Teste l'initialisation correcte de l'échantillonneur"""
        self.assertEqual(self.sampler.base_batch_size, 32)
        self.assertEqual(self.sampler.min_batch_size, 16)
        self.assertEqual(self.sampler.max_batch_size, 128)
        self.assertEqual(self.sampler.current_batch_size, 32)
        self.assertEqual(self.sampler.iterations, 0)
        self.assertEqual(len(self.sampler.sample_times), 0)
        self.assertEqual(self.sampler.strategy, "auto")
    
    def test_resource_utilization(self):
        """Teste la récupération des ressources système"""
        resources = self.sampler.get_resource_utilization()
        
        # Vérifier que les métriques de base sont présentes
        self.assertIn("ram", resources)
        self.assertIn("cpu", resources)
        
        # Vérifier que les valeurs sont dans une plage raisonnable
        self.assertGreaterEqual(resources["ram"], 0.0)
        self.assertLessEqual(resources["ram"], 1.0)
        self.assertGreaterEqual(resources["cpu"], 0.0)
        self.assertLessEqual(resources["cpu"], 1.0)
    
    def test_sample_basic(self):
        """Teste l'échantillonnage de base"""
        batch, info = self.sampler.sample()
        
        # Vérifier que l'échantillon a la bonne forme
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)  # (states, values)
        
        # Vérifier que les états et valeurs ont la bonne taille
        states, values = batch
        self.assertEqual(states.shape[0], self.sampler.current_batch_size)
        self.assertEqual(values.shape[0], self.sampler.current_batch_size)
        
        # Vérifier les infos
        self.assertEqual(info["batch_size"], self.sampler.current_batch_size)
        self.assertEqual(info["iterations"], 1)
        self.assertIn("sample_time", info)
        self.assertIn("resources", info)
    
    def test_batch_adaptation(self):
        """Teste l'adaptation de la taille des batchs"""
        # Échantillonner suffisamment pour déclencher une adaptation
        initial_batch_size = self.sampler.current_batch_size
        
        # Prendre assez d'échantillons pour atteindre check_interval
        for _ in range(self.sampler.check_interval):
            self.sampler.sample()
        
        # Vérifier que l'historique a été mis à jour
        self.assertEqual(len(self.sampler.sample_times), self.sampler.check_interval)
        self.assertGreaterEqual(len(self.sampler.batch_sizes_history), 1)
    
    def test_custom_schedule(self):
        """Teste l'utilisation d'une fonction de planification personnalisée"""
        # Fonction qui double la taille du batch à chaque itération (jusqu'au max)
        def custom_schedule(iteration):
            return min(32 * (2 ** (iteration // 5)), 256)
        
        # Créer un échantillonneur avec cette fonction
        sampler = VariableBatchSampler(
            buffer=self.buffer,
            base_batch_size=32,
            max_batch_size=256,
            schedule_fn=custom_schedule,
            check_interval=5
        )
        
        # Échantillonner 15 fois (3 adaptations)
        batch_sizes = []
        for _ in range(15):
            sampler.sample()
            if len(sampler.batch_sizes_history) > 0:
                batch_sizes.append(sampler.batch_sizes_history[-1])
        
        # Vérifier que la taille des batchs a augmenté
        if batch_sizes:
            self.assertGreater(batch_sizes[-1], sampler.base_batch_size)
    
    def test_strategy_selection(self):
        """Teste la sélection de stratégie"""
        # Tester chaque stratégie valide
        valid_strategies = ["auto", "gpu", "cpu", "ram", "performance"]
        for strategy in valid_strategies:
            self.sampler.set_strategy(strategy)
            self.assertEqual(self.sampler.strategy, strategy)
        
        # Tester une stratégie invalide
        with self.assertRaises(ValueError):
            self.sampler.set_strategy("invalid_strategy")
    
    def test_get_metrics(self):
        """Teste la récupération des métriques"""
        # Échantillonner plusieurs fois
        for _ in range(10):
            self.sampler.sample()
        
        metrics = self.sampler.get_metrics()
        
        # Vérifier que les métriques de base sont présentes
        self.assertIn("current_batch_size", metrics)
        self.assertIn("iterations", metrics)
        self.assertIn("strategy", metrics)
        
        # Vérifier les valeurs
        self.assertEqual(metrics["iterations"], 10)
        self.assertEqual(metrics["strategy"], "auto")
    
    def test_reset(self):
        """Teste la réinitialisation de l'échantillonneur"""
        # Échantillonner plusieurs fois
        for _ in range(10):
            self.sampler.sample()
        
        # Vérifier que l'état a changé
        self.assertEqual(self.sampler.iterations, 10)
        self.assertEqual(len(self.sampler.sample_times), 10)
        
        # Réinitialiser
        self.sampler.reset()
        
        # Vérifier que l'état a été réinitialisé
        self.assertEqual(self.sampler.iterations, 0)
        self.assertEqual(len(self.sampler.sample_times), 0)
        self.assertEqual(self.sampler.current_batch_size, self.sampler.base_batch_size)


class TestBatchOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Prépare l'environnement de test"""
        # Créer un modèle simple
        self.model = SimpleModel()
        
        # Créer un buffer pour les tests
        self.buffer = MockBuffer(size=1000)
        
        # Créer l'optimiseur avec des paramètres de test
        self.optimizer = BatchOptimizer(
            model=self.model,
            buffer=self.buffer,
            loss_fn=_test_loss_fn,
            min_batch_size=16,
            max_batch_size=128,
            warmup_iters=2,
            test_iters=3,
            search_method="grid"
        )
    
    def test_initialization(self):
        """Teste l'initialisation correcte de l'optimiseur"""
        self.assertEqual(self.optimizer.min_batch_size, 16)
        self.assertEqual(self.optimizer.max_batch_size, 128)
        self.assertEqual(self.optimizer.warmup_iters, 2)
        self.assertEqual(self.optimizer.test_iters, 3)
        self.assertEqual(self.optimizer.search_method, "grid")
        self.assertEqual(len(self.optimizer.results), 0)
        self.assertIsNone(self.optimizer.optimal_batch_size)
    
    def test_test_batch_size(self):
        """Teste l'évaluation d'une taille de batch spécifique"""
        throughput = self.optimizer.test_batch_size(32)
        
        # Vérifier que le débit est un nombre positif
        self.assertGreater(throughput, 0.0)
    
    def test_grid_search(self):
        """Teste la recherche par grille"""
        # Utiliser une recherche plus limitée pour les tests
        self.optimizer.min_batch_size = 16
        self.optimizer.max_batch_size = 64
        
        best_size = self.optimizer._grid_search()
        
        # Vérifier que le résultat est dans la plage attendue
        self.assertGreaterEqual(best_size, 16)
        self.assertLessEqual(best_size, 64)
        
        # Vérifier que des résultats ont été enregistrés
        self.assertGreater(len(self.optimizer.results), 0)
    
    def test_binary_search(self):
        """Teste la recherche binaire"""
        # Utiliser une recherche plus limitée pour les tests
        self.optimizer.min_batch_size = 16
        self.optimizer.max_batch_size = 64
        self.optimizer.search_method = "binary"
        
        best_size = self.optimizer._binary_search()
        
        # Vérifier que le résultat est dans la plage attendue
        self.assertGreaterEqual(best_size, 16)
        self.assertLessEqual(best_size, 64)
        
        # Vérifier que des résultats ont été enregistrés
        self.assertGreater(len(self.optimizer.results), 0)
    
    def test_find_optimal_batch_size(self):
        """Teste la recherche de la taille optimale"""
        # Utiliser une recherche plus limitée pour les tests
        self.optimizer.min_batch_size = 16
        self.optimizer.max_batch_size = 32
        
        optimal_size = self.optimizer.find_optimal_batch_size()
        
        # Vérifier que le résultat est dans la plage attendue
        self.assertGreaterEqual(optimal_size, 16)
        self.assertLessEqual(optimal_size, 32)
        
        # Vérifier que l'optimal a été enregistré
        self.assertEqual(self.optimizer.optimal_batch_size, optimal_size)
    
    def test_get_results(self):
        """Teste la récupération des résultats"""
        # Effectuer quelques tests
        throughput16 = self.optimizer.test_batch_size(16)
        self.optimizer.results[16] = throughput16
        
        throughput32 = self.optimizer.test_batch_size(32)
        self.optimizer.results[32] = throughput32
        
        results = self.optimizer.get_results()
        
        # Vérifier que les résultats ont été enregistrés
        self.assertEqual(len(results), 2)
        self.assertIn(16, results)
        self.assertIn(32, results)


if __name__ == "__main__":
    unittest.main() 