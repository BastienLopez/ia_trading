"""
Tests pour le module financial_dataset.py.
"""

import os
import tempfile
import unittest
from pathlib import Path

import torch

from ai_trading.data.financial_dataset import FinancialDataset, get_financial_dataloader
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data

# Vérifier si modules optionnels existent
try:
    from ai_trading.data.data_optimizers import convert_to_hdf5, convert_to_parquet

    HAVE_OPTIMIZERS = True
except ImportError:
    HAVE_OPTIMIZERS = False


class TestFinancialDataset(unittest.TestCase):
    """Tests pour la classe FinancialDataset."""

    def setUp(self):
        """Préparation des données pour les tests."""
        # Créer des données synthétiques pour les tests
        self.n_points = 500
        self.sequence_length = 50
        self.batch_size = 32

        # Générer des données
        self.synthetic_data = generate_synthetic_market_data(
            n_points=self.n_points,
            trend=0.001,
            volatility=0.02,
            start_price=100.0,
            include_volume=True,
            cyclic_pattern=True,
        )

        # Créer un répertoire temporaire pour les fichiers de test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Sauvegarder en CSV (format universel)
        self.csv_path = os.path.join(self.temp_path, "data.csv")
        self.synthetic_data.to_csv(self.csv_path)

        # Essayer de sauvegarder dans d'autres formats si les modules sont disponibles
        if HAVE_OPTIMIZERS:
            try:
                self.parquet_path = convert_to_parquet(
                    self.synthetic_data, os.path.join(self.temp_path, "data.parquet")
                )
            except ImportError:
                self.parquet_path = None

            try:
                self.hdf5_path = convert_to_hdf5(
                    self.synthetic_data,
                    os.path.join(self.temp_path, "data.h5"),
                    key="data",
                )
            except ImportError:
                self.hdf5_path = None
        else:
            self.parquet_path = None
            self.hdf5_path = None

        # Créer un tensor pour les tests
        self.tensor_data = torch.tensor(
            self.synthetic_data[["open", "high", "low", "close", "volume"]].values,
            dtype=torch.float32,
        )

    def tearDown(self):
        """Nettoyer après les tests."""
        self.temp_dir.cleanup()

    def test_init_from_dataframe(self):
        """Tester l'initialisation à partir d'un DataFrame."""
        dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            target_column="close",
            is_train=True,
        )

        # Vérifier les dimensions
        expected_num_examples = len(self.synthetic_data) - self.sequence_length - 1 + 1
        self.assertEqual(len(dataset), expected_num_examples)

        # Récupérer un exemple et vérifier sa forme
        sequence, target = dataset[0]
        self.assertEqual(sequence.shape[0], self.sequence_length)
        self.assertEqual(sequence.dim(), 2)  # (seq_len, features)
        self.assertEqual(target.dim(), 0)  # Scalaire

    def test_init_from_csv(self):
        """Tester l'initialisation à partir d'un fichier CSV."""
        dataset = FinancialDataset(
            data=self.csv_path, sequence_length=self.sequence_length, is_train=True
        )

        # Récupérer un exemple et vérifier sa forme
        sequence, target = dataset[0]
        self.assertEqual(sequence.shape[0], self.sequence_length)

    @unittest.skipIf(
        not HAVE_OPTIMIZERS
        or not hasattr(unittest.TestCase, "parquet_path")
        or not unittest.TestCase.parquet_path,
        "Module PyArrow non disponible",
    )
    def test_init_from_parquet(self):
        """Tester l'initialisation à partir d'un fichier Parquet."""
        if not self.parquet_path:
            self.skipTest("Fichier Parquet non disponible")

        dataset = FinancialDataset(
            data=self.parquet_path, sequence_length=self.sequence_length, is_train=True
        )

        # Récupérer un exemple et vérifier sa forme
        sequence, target = dataset[0]
        self.assertEqual(sequence.shape[0], self.sequence_length)

    @unittest.skipIf(
        not HAVE_OPTIMIZERS
        or not hasattr(unittest.TestCase, "hdf5_path")
        or not unittest.TestCase.hdf5_path,
        "Module h5py non disponible",
    )
    def test_init_from_hdf5(self):
        """Tester l'initialisation à partir d'un fichier HDF5."""
        if not self.hdf5_path:
            self.skipTest("Fichier HDF5 non disponible")

        dataset = FinancialDataset(
            data=self.hdf5_path, sequence_length=self.sequence_length, is_train=True
        )

        # Récupérer un exemple et vérifier sa forme
        sequence, target = dataset[0]
        self.assertEqual(sequence.shape[0], self.sequence_length)

    def test_init_from_tensor(self):
        """Tester l'initialisation à partir d'un tensor."""
        dataset = FinancialDataset(
            data=self.tensor_data, sequence_length=self.sequence_length, is_train=True
        )

        # Récupérer un exemple et vérifier sa forme
        sequence, target = dataset[0]
        self.assertEqual(sequence.shape[0], self.sequence_length)

    def test_dataloader_single_worker(self):
        """Tester le DataLoader avec un seul worker."""
        dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            is_train=True,
        )

        dataloader = get_financial_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            prefetch_factor=None,
        )

        # Vérifier que l'on peut itérer sur le DataLoader
        batch_count = 0
        for batch_data in dataloader:
            batch_sequences, batch_targets = batch_data
            self.assertEqual(batch_sequences.shape[0], self.batch_size)
            self.assertEqual(batch_sequences.shape[1], self.sequence_length)
            batch_count += 1

            # Limiter le nombre de batchs pour le test
            if batch_count >= 3:
                break

        self.assertGreater(batch_count, 0)

    def test_dataloader_multi_workers(self):
        """Tester le DataLoader avec plusieurs workers."""
        # Réduire le nombre de workers pour les tests (éviter problèmes sur CI)
        num_workers = min(2, os.cpu_count() or 1)

        dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            is_train=True,
            use_shared_memory=True,
        )

        try:
            dataloader = get_financial_dataloader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=2,
            )

            # Vérifier que l'on peut itérer sur le DataLoader
            batch_count = 0
            for batch_data in dataloader:
                batch_sequences, batch_targets = batch_data
                self.assertEqual(batch_sequences.shape[0], self.batch_size)
                self.assertEqual(batch_sequences.shape[1], self.sequence_length)
                batch_count += 1

                # Limiter le nombre de batchs pour le test
                if batch_count >= 3:
                    break

            self.assertGreater(batch_count, 0)
        except RuntimeError as e:
            # Capturer les erreurs liées au multiprocessing et les passer
            # (peut arriver sur certains systèmes/configurations)
            if "multiprocessing" in str(e).lower():
                self.skipTest(f"Test multi-workers ignoré: {e}")
            else:
                raise

    def test_predict_n_ahead(self):
        """Tester la prédiction à plusieurs pas de temps dans le futur."""
        predict_n_ahead = 5
        dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            target_column="close",
            predict_n_ahead=predict_n_ahead,
            is_train=True,
        )

        # Récupérer un exemple
        sequence, target = dataset[0]

        # Vérifier que la cible correspond bien au prix n pas dans le futur
        expected_target_idx = self.sequence_length + predict_n_ahead - 1
        expected_target = torch.tensor(
            self.synthetic_data["close"].iloc[expected_target_idx], dtype=torch.float32
        )
        self.assertTrue(torch.isclose(target, expected_target, rtol=1e-4))

    def test_transform_functions(self):
        """Tester les fonctions de transformation."""

        # Définir des fonctions de transformation simples
        def transform_seq(x):
            return x / 100.0

        def transform_target(y):
            return y * 2.0

        dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            target_column="close",
            transform=transform_seq,
            target_transform=transform_target,
            is_train=True,
        )

        # Récupérer un exemple avec transformations
        sequence, target = dataset[0]

        # Récupérer un exemple sans transformations pour comparer
        dataset_no_transform = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            target_column="close",
            is_train=True,
        )
        sequence_no_transform, target_no_transform = dataset_no_transform[0]

        # Vérifier les transformations
        self.assertTrue(
            torch.allclose(sequence, sequence_no_transform / 100.0, rtol=1e-4)
        )
        self.assertTrue(torch.isclose(target, target_no_transform * 2.0, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
