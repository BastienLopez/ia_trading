import os
import tempfile
import unittest
from pathlib import Path

import torch

from ai_trading.data.optimized_dataset import (
    OptimizedFinancialDataset,
    convert_to_compressed,
    get_optimized_dataloader,
    load_market_data_compressed,
)
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data


class TestOptimizedDataset(unittest.TestCase):
    """Tests pour la classe OptimizedFinancialDataset."""

    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Paramètres pour les tests
        self.n_points = 500
        self.sequence_length = 30
        self.batch_size = 16

        # Générer des données de test
        self.synthetic_data = generate_synthetic_market_data(
            n_points=self.n_points,
            trend=0.001,
            volatility=0.02,
            start_price=100.0,
            include_volume=True,
            cyclic_pattern=True,
        )

        # Créer quelques fichiers de test
        self.csv_path = os.path.join(self.temp_path, "test_data.csv")
        self.synthetic_data.to_csv(self.csv_path, index=False)

        # Créer un répertoire de cache dédié aux tests
        self.cache_dir = os.path.join(self.temp_path, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        """Nettoyage après les tests."""
        self.temp_dir.cleanup()

    def test_init_and_basic_functionality(self):
        """Teste l'initialisation et les fonctionnalités de base."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            target_column="close",
            is_train=True,
            cache_dir=self.cache_dir,
            compression_level=1,  # Niveau rapide pour les tests
            use_compressed_cache=True,
        )

        # Vérifier les dimensions
        expected_num_examples = len(self.synthetic_data) - self.sequence_length - 1 + 1
        self.assertEqual(len(dataset), expected_num_examples)

        # Récupérer un exemple et vérifier sa forme
        sequence, target = dataset[0]
        self.assertEqual(sequence.shape[0], self.sequence_length)
        self.assertEqual(sequence.dim(), 2)  # (seq_len, features)
        self.assertEqual(target.dim(), 0)  # Scalaire

    def test_cache_and_load_dataframe(self):
        """Teste la mise en cache et le chargement d'un DataFrame."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            cache_dir=self.cache_dir,
            use_compressed_cache=True,
        )

        # Mettre en cache le DataFrame
        cache_name = "test_df"
        dataset.cache_data(cache_name, self.synthetic_data, format="parquet")

        # Vérifier que le fichier de cache existe
        cache_path = dataset.cache_files[cache_name]
        self.assertTrue(cache_path.exists())

        # Charger le DataFrame depuis le cache
        loaded_df = dataset.load_cached_data(cache_name, format="parquet")

        # Vérifier que le DataFrame chargé est correct
        self.assertEqual(loaded_df.shape, self.synthetic_data.shape)
        self.assertEqual(list(loaded_df.columns), list(self.synthetic_data.columns))

    def test_precompute_features(self):
        """Teste le précalcul et la mise en cache des caractéristiques."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            cache_dir=self.cache_dir,
            use_compressed_cache=True,
        )

        # Créer directement un tenseur et le passer comme caractéristiques précalculées
        features_tensor = torch.rand(10, 5)
        features_name = "test_direct_features"

        # Mettre en cache les caractéristiques directement
        dataset.cache_data(features_name, features_tensor, format="numpy")

        # Charger les caractéristiques du cache
        loaded_features = dataset.load_cached_data(
            features_name, format="numpy", tensor_dtype=torch.float32
        )

        # Vérifier que les caractéristiques ont été mises en cache correctement
        self.assertIsNotNone(loaded_features)
        self.assertEqual(loaded_features.shape, features_tensor.shape)

    def test_clear_cache(self):
        """Teste la suppression du cache."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            cache_dir=self.cache_dir,
            use_compressed_cache=True,
        )

        # Mettre en cache plusieurs données
        dataset.cache_data("df1", self.synthetic_data, format="parquet")
        dataset.cache_data("df2", self.synthetic_data, format="csv")

        # Vérifier que les fichiers de cache existent
        self.assertEqual(len(dataset.cache_files), 2)

        # Supprimer un cache spécifique
        dataset.clear_cache(["df1"])

        # Vérifier qu'un seul cache a été supprimé
        self.assertEqual(len(dataset.cache_files), 1)
        self.assertNotIn("df1", dataset.cache_files)
        self.assertIn("df2", dataset.cache_files)

        # Supprimer tous les caches restants
        dataset.clear_cache()

        # Vérifier que tous les caches ont été supprimés
        self.assertEqual(len(dataset.cache_files), 0)

    def test_optimize_dataframe(self):
        """Teste l'optimisation d'un DataFrame."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            cache_dir=self.cache_dir,
            use_compressed_cache=True,
        )

        # Optimiser le DataFrame
        optimized_df = dataset.optimize_dataframe(self.synthetic_data, "optimized_test")

        # Vérifier que le DataFrame optimisé est correct
        self.assertEqual(optimized_df.shape, self.synthetic_data.shape)
        self.assertEqual(list(optimized_df.columns), list(self.synthetic_data.columns))

        # Vérifier que le fichier de cache existe
        self.assertIn("optimized_test", dataset.cache_files)
        cache_path = dataset.cache_files["optimized_test"]
        self.assertTrue(cache_path.exists())

    def test_compress_raw_data(self):
        """Teste la compression d'un fichier de données brutes."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            cache_dir=self.cache_dir,
            use_compressed_cache=True,
        )

        # Compresser le fichier CSV
        compressed_path = dataset.compress_raw_data(self.csv_path)

        # Vérifier que le fichier compressé existe
        self.assertTrue(compressed_path.exists())

        # Vérifier que le fichier compressé est plus petit que l'original
        self.assertLess(
            compressed_path.stat().st_size, Path(self.csv_path).stat().st_size
        )

    def test_load_from_compressed(self):
        """Teste le chargement d'un DataFrame depuis un fichier compressé."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            cache_dir=self.cache_dir,
            use_compressed_cache=True,
        )

        # Sauvegarder le DataFrame au format parquet compressé
        test_path = os.path.join(self.temp_path, "test_compressed.parquet.zst")
        dataset.storage.save_dataframe(self.synthetic_data, test_path, format="parquet")

        # Charger le DataFrame depuis le fichier compressé
        loaded_df = dataset.load_from_compressed(test_path, format="parquet")

        # Vérifier que le DataFrame chargé est correct
        self.assertEqual(loaded_df.shape, self.synthetic_data.shape)
        self.assertEqual(list(loaded_df.columns), list(self.synthetic_data.columns))

    def test_dataloader(self):
        """Teste le DataLoader optimisé."""
        # Créer un dataset optimisé
        dataset = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            cache_dir=self.cache_dir,
            use_compressed_cache=True,
        )

        # Créer un DataLoader
        dataloader = get_optimized_dataloader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Pas de workers pour les tests
        )

        # Vérifier qu'on peut itérer sur le DataLoader
        batch_count = 0
        for batch_sequences, batch_targets in dataloader:
            # Vérifier les dimensions des batchs
            self.assertEqual(batch_sequences.shape[1], self.sequence_length)

            # Limiter le nombre de batchs pour les tests
            batch_count += 1
            if batch_count >= 3:
                break

        self.assertGreater(batch_count, 0)

    def test_utility_functions(self):
        """Teste les fonctions utilitaires du module."""
        # Tester convert_to_compressed
        compressed_path = convert_to_compressed(
            input_path=self.csv_path,
            compression_level=1,  # Niveau rapide pour les tests
        )
        self.assertTrue(compressed_path.exists())

        # Tester load_market_data_compressed
        # Sauvegarder d'abord le DataFrame au format parquet compressé
        test_path = os.path.join(self.temp_path, "market_data.parquet.zst")
        storage = OptimizedFinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
        ).storage
        storage.save_dataframe(self.synthetic_data, test_path, format="parquet")

        # Charger les données de marché compressées
        loaded_df = load_market_data_compressed(test_path, format="parquet")

        # Vérifier que le DataFrame chargé est correct
        self.assertEqual(loaded_df.shape, self.synthetic_data.shape)


if __name__ == "__main__":
    unittest.main()
