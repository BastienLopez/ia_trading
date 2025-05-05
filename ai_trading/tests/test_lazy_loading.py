#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests pour le module de chargement paresseux (lazy loading).
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ajouter le répertoire parent au path pour importer les modules du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_trading.data.lazy_loading import (
    BatchInferenceOptimizer,
    CachedFeatureTransform,
    LazyDataset,
    LazyFileReader,
    get_cache_transform_fn,
)


class TestLazyFileReader(unittest.TestCase):
    """Tests pour la classe LazyFileReader."""

    def setUp(self):
        """Initialiser les éléments nécessaires pour chaque test."""
        # Créer un répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()

        # Créer un DataFrame synthétique
        self.df = pd.DataFrame(
            {
                "close": np.random.randn(1000),
                "open": np.random.randn(1000),
                "high": np.random.randn(1000),
                "low": np.random.randn(1000),
                "volume": np.abs(np.random.randn(1000)) * 1000,
            }
        )

        # Sauvegarder en CSV pour les tests
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
        self.df.to_csv(self.csv_path)

        # Sauvegarder en parquet si possible
        try:
            self.parquet_path = os.path.join(self.temp_dir, "test_data.parquet")
            self.df.to_parquet(self.parquet_path)
        except ImportError:
            self.parquet_path = None

    def tearDown(self):
        """Nettoyer après chaque test."""
        # Supprimer le répertoire temporaire
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Tester l'initialisation du lecteur."""
        reader = LazyFileReader(self.csv_path)
        self.assertEqual(reader._file_type, "csv")
        self.assertEqual(reader._file_length, 1000)
        self.assertEqual(len(reader.get_column_names()), 6)  # 5 colonnes + l'index

    def test_get_row(self):
        """Tester la récupération d'une ligne."""
        reader = LazyFileReader(self.csv_path, chunk_size=100)

        # Récupérer la ligne 50
        row = reader.get_row(50)

        # Vérifier que la ligne correspond aux données originales
        self.assertAlmostEqual(row["close"], self.df.iloc[50]["close"], places=6)
        self.assertAlmostEqual(row["volume"], self.df.iloc[50]["volume"], places=6)

    def test_get_rows(self):
        """Tester la récupération d'une plage de lignes."""
        reader = LazyFileReader(self.csv_path, chunk_size=100)

        # Récupérer les lignes 150-250
        rows = reader.get_rows(150, 250)

        # Vérifier le nombre de lignes
        self.assertEqual(len(rows), 100)

        # Vérifier que les données correspondent aux originales
        self.assertAlmostEqual(
            rows.iloc[0]["close"], self.df.iloc[150]["close"], places=6
        )
        self.assertAlmostEqual(
            rows.iloc[99]["close"], self.df.iloc[249]["close"], places=6
        )

    def test_cache_efficiency(self):
        """Tester l'efficacité du cache."""
        reader = LazyFileReader(self.csv_path, chunk_size=100, cache_size=5)

        # Premier accès (cold)
        reader.get_chunk(2)

        # Deuxième accès (warm) - devrait être plus rapide, mais difficile à tester
        # On vérifie juste que ça ne lance pas d'erreur
        reader.get_chunk(2)

        # Vérifier la taille du cache
        self.assertEqual(len(reader._chunk_cache), 1)

        # Remplir le cache
        for i in range(10):
            reader.get_chunk(i)

        # Vérifier que la taille du cache est limitée
        self.assertLessEqual(len(reader._chunk_cache), 5)

    @unittest.skipIf(
        not os.path.exists(os.path.expandvars("${TEMP}/test_data.parquet")),
        "Test parquet ignoré si pyarrow n'est pas installé",
    )
    def test_parquet_support(self):
        """Tester le support des fichiers parquet."""
        if self.parquet_path:
            reader = LazyFileReader(self.parquet_path)
            self.assertEqual(reader._file_type, "parquet")

            # Récupérer la première ligne
            row = reader.get_row(0)

            # Vérifier les données
            self.assertAlmostEqual(row["close"], self.df.iloc[0]["close"], places=6)


class TestLazyDataset(unittest.TestCase):
    """Tests pour la classe LazyDataset."""

    def setUp(self):
        """Initialiser les éléments nécessaires pour chaque test."""
        # Créer un répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()

        # Créer un DataFrame synthétique
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "close": np.random.randn(
                    1000
                ).cumsum(),  # Prix cumulés pour simuler une série temporelle
                "open": np.random.randn(1000).cumsum(),
                "high": np.random.randn(1000).cumsum()
                + 1.0,  # Toujours plus haut que close
                "low": np.random.randn(1000).cumsum()
                - 1.0,  # Toujours plus bas que close
                "volume": np.abs(np.random.randn(1000)) * 1000,
            }
        )

        # Sauvegarder en CSV pour les tests
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
        self.df.to_csv(self.csv_path)

        # Transformation simple
        self.transform = lambda x: x * 2

    def tearDown(self):
        """Nettoyer après chaque test."""
        # Supprimer le répertoire temporaire
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Tester l'initialisation du dataset."""
        dataset = LazyDataset(
            file_path=self.csv_path, sequence_length=10, target_column="close"
        )

        # Vérifier les attributs
        self.assertEqual(dataset.sequence_length, 10)
        self.assertEqual(dataset.target_column, "close")

        # Vérifier le nombre d'exemples
        # 1000 lignes - 10 séquence - 1 prédiction + 1 = 990 exemples
        self.assertEqual(len(dataset), 990)

    def test_getitem(self):
        """Tester la récupération d'un item."""
        dataset = LazyDataset(
            file_path=self.csv_path, sequence_length=10, target_column="close"
        )

        # Récupérer le premier exemple
        sequence, target = dataset[0]

        # Vérifier la forme de la séquence
        self.assertEqual(sequence.shape, (10, len(dataset.feature_indices)))

        # Vérifier que la cible est un scalaire
        self.assertTrue(isinstance(target.item(), float))

    def test_transform(self):
        """Tester l'application d'une transformation."""
        dataset = LazyDataset(
            file_path=self.csv_path,
            sequence_length=10,
            target_column="close",
            transform=self.transform,
        )

        # Récupérer un exemple
        sequence, target = dataset[0]

        # Récupérer les données brutes
        reader = LazyFileReader(self.csv_path)
        raw_data = reader.get_rows(0, 10)

        # Vérifier que la transformation a été appliquée
        expected = raw_data.iloc[:, dataset.feature_indices].values * 2
        expected_tensor = torch.tensor(expected, dtype=torch.float32)

        # Comparer avec une tolérance
        self.assertTrue(torch.allclose(sequence, expected_tensor, rtol=1e-5))

    def test_target_column(self):
        """Tester la sélection de la colonne cible."""
        dataset = LazyDataset(
            file_path=self.csv_path,
            sequence_length=10,
            target_column="volume",  # Changer la colonne cible
        )

        # Récupérer un exemple
        sequence, target = dataset[0]

        # Vérifier que le target est bien la valeur de volume
        reader = LazyFileReader(self.csv_path)
        raw_data = reader.get_row(10)  # Target est à seq_len + predict_ahead - 1

        self.assertAlmostEqual(target.item(), raw_data["volume"], places=5)


class TestCachedTransform(unittest.TestCase):
    """Tests pour les fonctionnalités de cache des transformations."""

    def test_get_cache_transform_fn(self):
        """Tester le décorateur de cache pour les transformations."""
        # Définir une transformation avec compteur d'appels
        call_count = 0

        @get_cache_transform_fn(cache_size=10)
        def test_transform(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Créer un tenseur
        tensor = torch.randn(5, 3)

        # Premier appel (cold cache)
        result1 = test_transform(tensor)
        self.assertEqual(call_count, 1)

        # Deuxième appel avec le même tenseur (should hit cache)
        result2 = test_transform(tensor)
        self.assertEqual(call_count, 1)  # Le compteur ne doit pas augmenter

        # Vérifier que les résultats sont identiques
        self.assertTrue(torch.all(result1 == result2))

        # Appel avec un nouveau tenseur
        new_tensor = torch.randn(5, 3)
        result3 = test_transform(new_tensor)
        self.assertEqual(call_count, 2)  # Le compteur doit augmenter

    def test_cached_feature_transform(self):
        """Tester la classe CachedFeatureTransform."""
        # Créer une instance de CachedFeatureTransform
        cache_manager = CachedFeatureTransform(
            cache_dir=tempfile.mkdtemp(), max_memory_cache_size=5
        )

        # Fonction de transformation
        def transform_fn(data):
            return data * 2

        # Créer des données
        data = torch.randn(10, 5)

        # Premier appel (cold cache)
        result1 = cache_manager.transform_with_cache(data, transform_fn)

        # Deuxième appel (warm cache)
        result2 = cache_manager.transform_with_cache(data, transform_fn)

        # Vérifier que les résultats sont identiques
        self.assertTrue(torch.all(result1 == result2))

        # Vérifier les stats du cache
        stats = cache_manager.get_cache_stats()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)

        # Nettoyer
        cache_manager.clear_memory_cache()
        cache_manager.clear_disk_cache()
        shutil.rmtree(cache_manager.cache_dir)


class TestBatchInference(unittest.TestCase):
    """Tests pour BatchInferenceOptimizer."""

    def setUp(self):
        """Initialiser les éléments nécessaires pour chaque test."""

        # Créer un modèle simple
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        self.model = SimpleModel()

        # Créer des données d'exemple
        self.data = torch.randn(100, 10)

    def test_batch_inference_optimizer(self):
        """Tester l'optimiseur d'inférence par lots."""
        # Créer l'optimiseur
        optimizer = BatchInferenceOptimizer(
            model=self.model, batch_size=32, device="cpu", num_workers=0
        )

        # Effectuer les prédictions
        predictions = optimizer.predict(self.data, return_numpy=True)

        # Vérifier la forme des prédictions
        self.assertEqual(predictions.shape, (100, 1))

        # Vérifier que les prédictions sont cohérentes
        with torch.no_grad():
            expected = self.model(self.data).numpy()

        self.assertTrue(np.allclose(predictions, expected, rtol=1e-5))

    def test_batch_inference_function(self):
        """Tester la fonction utilitaire batch_inference."""
        from ai_trading.data.lazy_loading import batch_inference

        # Effectuer les prédictions
        predictions = batch_inference(
            model=self.model, data=self.data, batch_size=32, device="cpu", num_workers=0
        )

        # Vérifier la forme des prédictions
        self.assertEqual(predictions.shape, (100, 1))


if __name__ == "__main__":
    unittest.main()
