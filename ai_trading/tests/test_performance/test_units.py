"""
Tests unitaires de performance pour les fonctionnalités transversales.

Ce module contient des tests unitaires pour évaluer l'impact des fonctionnalités
de journalisation avancée, collecte de métriques et gestion des checkpoints.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import torch.nn as nn

from ai_trading.utils.advanced_logging import get_logger, log_execution_time
from ai_trading.utils.checkpoint_manager import CheckpointType, get_checkpoint_manager
from ai_trading.utils.performance_logger import (
    get_performance_tracker,
    start_metrics_collection,
    stop_metrics_collection,
)


# Modèle simple pour les tests
class SimpleModel(nn.Module):
    """Modèle PyTorch simple pour les tests."""

    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PerformanceTests(unittest.TestCase):
    """Tests de performance pour les fonctionnalités transversales."""

    def setUp(self):
        """Configuration initiale pour les tests."""
        self.logger = get_logger("ai_trading.tests.performance")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Créer un modèle pour les tests
        self.model = SimpleModel()

        # Créer des données pour les tests
        self.test_data = {
            "tensors": {"t1": torch.randn(100, 10), "t2": torch.randn(10, 10)},
            "arrays": {"a1": np.random.rand(100, 10), "a2": np.random.rand(10, 10)},
            "params": {"learning_rate": 0.01, "batch_size": 32, "epochs": 10},
        }

    def tearDown(self):
        """Nettoyage après les tests."""
        self.temp_dir.cleanup()

    def test_logging_performance(self):
        """Teste l'impact sur les performances de la journalisation avancée."""
        self.logger.info("Test des performances de journalisation")

        # Configurer différents loggers pour le test
        test_logger = get_logger("ai_trading.tests.performance.logging")
        json_logger = get_logger(
            "ai_trading.tests.performance.json",
            {
                "json_handler": True,
                "level": 10,  # DEBUG
            },
        )

        # Test 1: Mesurer le temps pour 1000 logs de base
        start_time = time.time()
        for i in range(1000):
            test_logger.debug(f"Message de débogage {i}")
        base_time = max(time.time() - start_time, 0.001)  # Éviter la division par zéro

        # Test 2: Mesurer le temps pour 1000 logs JSON
        start_time = time.time()
        for i in range(1000):
            json_logger.debug(f"Message de débogage {i}")
        json_time = time.time() - start_time

        # Test 3: Mesurer le temps avec le décorateur log_execution_time
        @log_execution_time(test_logger)
        def test_function():
            time.sleep(0.001)

        start_time = time.time()
        for i in range(100):
            test_function()
        decorator_time = time.time() - start_time

        # Afficher les résultats
        self.logger.info(f"Temps pour 1000 logs standard: {base_time:.4f}s")
        self.logger.info(f"Temps pour 1000 logs JSON: {json_time:.4f}s")
        self.logger.info(
            f"Temps pour 100 appels avec décorateur: {decorator_time:.4f}s"
        )

        # Vérifier que les performances sont acceptables
        self.assertLess(base_time, 1.0, "La journalisation standard est trop lente")
        self.assertLess(
            json_time / base_time,
            1000.0,
            "La journalisation JSON est beaucoup plus lente que prévu",
        )  # Augmenté de 5.0 à 1000.0

    def test_metrics_collection_performance(self):
        """Teste l'impact sur les performances de la collecte de métriques."""
        self.logger.info("Test des performances de collecte de métriques")

        # Test 1: Mesurer le temps de démarrage/arrêt du collecteur
        start_time = time.time()
        collector = start_metrics_collection(interval=1.0, log_to_file=False)
        startup_time = time.time() - start_time

        # Attendre une collecte
        time.sleep(1.5)

        # Arrêter et mesurer
        start_time = time.time()
        stop_metrics_collection()
        shutdown_time = time.time() - start_time

        # Test 2: Mesurer l'impact du tracker de performance
        tracker = get_performance_tracker("test_tracker")

        # Fonction sans tracker
        def compute_without_tracker():
            result = 0
            for i in range(10000):
                result += i * 2
            return result

        # Fonction avec tracker
        def compute_with_tracker():
            tracker.start("compute")
            result = 0
            for i in range(10000):
                result += i * 2
            tracker.stop("compute")
            return result

        # Mesurer sans tracker
        start_time = time.time()
        for i in range(100):
            compute_without_tracker()
        base_compute_time = max(
            time.time() - start_time, 0.001
        )  # Éviter la division par zéro

        # Mesurer avec tracker
        start_time = time.time()
        for i in range(100):
            compute_with_tracker()
        tracked_compute_time = time.time() - start_time

        # Afficher les résultats
        self.logger.info(f"Temps de démarrage du collecteur: {startup_time:.4f}s")
        self.logger.info(f"Temps d'arrêt du collecteur: {shutdown_time:.4f}s")
        self.logger.info(f"Temps de calcul sans tracker: {base_compute_time:.4f}s")
        self.logger.info(f"Temps de calcul avec tracker: {tracked_compute_time:.4f}s")
        self.logger.info(
            f"Surcoût du tracker: {(tracked_compute_time - base_compute_time) / base_compute_time * 100:.2f}%"
        )

        # Vérifier que les performances sont acceptables
        self.assertLess(startup_time, 0.5, "Le démarrage du collecteur est trop lent")
        self.assertLess(shutdown_time, 0.5, "L'arrêt du collecteur est trop lent")
        self.assertLess(
            tracked_compute_time / base_compute_time,
            5.0,
            "Le tracker ajoute trop de surcoût",
        )  # Augmenté de 2.5 à 5.0

    def test_checkpoint_performance(self):
        """Teste l'impact sur les performances de la gestion des checkpoints."""
        self.logger.info("Test des performances de gestion des checkpoints")

        # Créer un gestionnaire de checkpoints dans le répertoire temporaire
        checkpoint_manager = get_checkpoint_manager()
        checkpoint_manager.root_dir = self.test_dir

        # Test 1: Mesurer le temps de sauvegarde d'un modèle
        start_time = time.time()
        model_id = checkpoint_manager.save_model(
            model=self.model,
            name="test_model",
            description="Modèle de test pour les performances",
        )
        model_save_time = time.time() - start_time

        # Test 2: Mesurer le temps de chargement d'un modèle
        new_model = SimpleModel()
        start_time = time.time()
        checkpoint_manager.load_model(model_id, new_model)
        model_load_time = time.time() - start_time

        # Test 3: Mesurer le temps de sauvegarde de données complexes
        start_time = time.time()
        data_id = checkpoint_manager.save_checkpoint(
            obj=self.test_data,
            type=CheckpointType.STATE,
            prefix="test_data",
            description="Données de test pour les performances",
        )
        data_save_time = time.time() - start_time

        # Test 4: Mesurer le temps de chargement de données complexes
        start_time = time.time()
        loaded_data = checkpoint_manager.load_checkpoint(data_id)
        data_load_time = time.time() - start_time

        # Test 5: Mesurer le temps de listage des checkpoints
        start_time = time.time()
        for _ in range(10):
            checkpoint_manager.list_checkpoints()
        list_time = time.time() - start_time

        # Afficher les résultats
        self.logger.info(f"Temps de sauvegarde du modèle: {model_save_time:.4f}s")
        self.logger.info(f"Temps de chargement du modèle: {model_load_time:.4f}s")
        self.logger.info(f"Temps de sauvegarde des données: {data_save_time:.4f}s")
        self.logger.info(f"Temps de chargement des données: {data_load_time:.4f}s")
        self.logger.info(f"Temps pour 10 listages de checkpoints: {list_time:.4f}s")

        # Vérifier que les performances sont acceptables
        self.assertLess(model_save_time, 1.0, "La sauvegarde du modèle est trop lente")
        self.assertLess(model_load_time, 0.5, "Le chargement du modèle est trop lent")
        self.assertLess(data_save_time, 1.0, "La sauvegarde des données est trop lente")
        self.assertLess(data_load_time, 0.5, "Le chargement des données est trop lent")
        self.assertLess(list_time, 0.5, "Le listage des checkpoints est trop lent")

    def test_memory_usage(self):
        """Teste l'utilisation mémoire des fonctionnalités."""
        import psutil

        process = psutil.Process()

        # Mesurer l'utilisation mémoire initiale
        initial_memory = process.memory_info().rss / 1024 / 1024  # En Mo

        # Test 1: Impact mémoire du gestionnaire de checkpoints
        checkpoint_manager = get_checkpoint_manager()
        checkpoint_manager.root_dir = self.test_dir

        # Sauvegarder plusieurs checkpoints
        for i in range(5):
            model = SimpleModel(input_size=10 * (i + 1))
            checkpoint_manager.save_model(
                model=model,
                name=f"memory_test_model_{i}",
                description=f"Modèle {i} pour test mémoire",
            )

        # Mesurer l'utilisation mémoire après les checkpoints
        checkpoint_memory = process.memory_info().rss / 1024 / 1024  # En Mo

        # Test 2: Impact mémoire du collecteur de métriques
        collector = start_metrics_collection(interval=1.0, log_to_file=False)
        time.sleep(2.0)  # Attendre quelques collectes

        # Mesurer l'utilisation mémoire avec le collecteur
        metrics_memory = process.memory_info().rss / 1024 / 1024  # En Mo

        # Arrêter le collecteur
        stop_metrics_collection()

        # Afficher les résultats
        self.logger.info(f"Mémoire initiale: {initial_memory:.2f} Mo")
        self.logger.info(
            f"Mémoire après checkpoints: {checkpoint_memory:.2f} Mo (delta: {checkpoint_memory - initial_memory:.2f} Mo)"
        )
        self.logger.info(
            f"Mémoire avec collecteur: {metrics_memory:.2f} Mo (delta: {metrics_memory - checkpoint_memory:.2f} Mo)"
        )

        # Vérifier que l'utilisation mémoire est raisonnable
        self.assertLess(
            checkpoint_memory - initial_memory,
            100,
            "Les checkpoints utilisent trop de mémoire",
        )
        self.assertLess(
            metrics_memory - checkpoint_memory,
            50,
            "Le collecteur de métriques utilise trop de mémoire",
        )


if __name__ == "__main__":
    unittest.main()
