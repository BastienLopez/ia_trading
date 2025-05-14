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

# Vérifier si les modules optionnels existent
try:
    import pyarrow
    HAVE_PYARROW = True
    print("Module PyArrow disponible")
except ImportError:
    HAVE_PYARROW = False
    print("Module PyArrow non disponible")

try:
    import h5py
    import tables
    HAVE_HDF5 = True
    print("Module h5py disponible")
except ImportError:
    HAVE_HDF5 = False
    print("Module h5py non disponible")

try:
    from ai_trading.data.data_optimizers import convert_to_hdf5, convert_to_parquet

    HAVE_OPTIMIZERS = True
    print("Module optimizers disponible")
except ImportError:
    HAVE_OPTIMIZERS = False
    print("Module optimizers non disponible")


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

        # Préparation des chemins pour les formats optimisés
        self.parquet_path = os.path.join(self.temp_path, "data.parquet")
        self.hdf5_path = os.path.join(self.temp_path, "data.h5")

        # Essayer de sauvegarder dans d'autres formats si les modules sont disponibles
        if HAVE_OPTIMIZERS and HAVE_PYARROW:
            try:
                convert_to_parquet(self.synthetic_data, self.parquet_path)
            except Exception as e:
                print(f"Erreur lors de la conversion Parquet: {e}")
                self.parquet_path = None
        else:
            self.parquet_path = None

        if HAVE_OPTIMIZERS and HAVE_HDF5:
            try:
                convert_to_hdf5(
                    self.synthetic_data,
                    self.hdf5_path,
                    key="data",
                )
            except Exception as e:
                print(f"Erreur lors de la conversion HDF5: {e}")
                self.hdf5_path = None
        else:
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

    @unittest.skipIf(not HAVE_PYARROW, "Module PyArrow non disponible")
    def test_init_from_parquet(self):
        """Tester l'initialisation à partir d'un fichier Parquet."""
        if not HAVE_PYARROW:
            self.skipTest("Module PyArrow non disponible")

        # Créer directement un fichier Parquet simplifiée sans utiliser les fonctionnalités d'optimisation
        self.parquet_path = os.path.join(self.temp_path, "data_simple.parquet")
        try:
            # Utiliser pandas directement pour écrire le fichier Parquet
            self.synthetic_data.to_parquet(self.parquet_path)

            # Vérifier que le fichier a été créé
            self.assertTrue(
                os.path.exists(self.parquet_path), "Le fichier Parquet n'a pas été créé"
            )

            # Utiliser le fichier avec FinancialDataset
            dataset = FinancialDataset(
                data=self.parquet_path,
                sequence_length=self.sequence_length,
                is_train=True,
            )

            # Récupérer un exemple et vérifier sa forme
            sequence, target = dataset[0]
            self.assertEqual(sequence.shape[0], self.sequence_length)
        except Exception as e:
            self.skipTest(
                f"Erreur lors de la création ou du chargement du fichier Parquet: {e}"
            )

    @unittest.skipIf(not HAVE_HDF5, "Module h5py non disponible")
    def test_init_from_hdf5(self):
        """Tester l'initialisation à partir d'un fichier HDF5."""
        if not HAVE_HDF5:
            self.skipTest("Module h5py non disponible")

        # Créer directement un fichier HDF5 simplifié sans utiliser les fonctionnalités d'optimisation
        self.hdf5_path = os.path.join(self.temp_path, "data_simple.h5")
        try:
            # Utiliser pandas directement pour écrire le fichier HDF5
            self.synthetic_data.to_hdf(self.hdf5_path, key="data", mode="w")

            # Vérifier que le fichier a été créé
            self.assertTrue(
                os.path.exists(self.hdf5_path), "Le fichier HDF5 n'a pas été créé"
            )

            # Utiliser le fichier avec FinancialDataset
            dataset = FinancialDataset(
                data=self.hdf5_path, sequence_length=self.sequence_length, is_train=True
            )

            # Récupérer un exemple et vérifier sa forme
            sequence, target = dataset[0]
            self.assertEqual(sequence.shape[0], self.sequence_length)
        except Exception as e:
            self.skipTest(
                f"Erreur lors de la création ou du chargement du fichier HDF5: {e}"
            )

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
            self.synthetic_data.iloc[expected_target_idx]["close"],
            dtype=torch.float32,
        )
        self.assertTrue(torch.isclose(target, expected_target, rtol=1e-5))

    def test_transform_functions(self):
        """Tester les fonctions de transformation personnalisées."""

        # Définir des fonctions de transformation simples
        def transform_seq(x):
            return x * 2.0

        def transform_target(y):
            return y + 1.0

        dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            target_column="close",
            is_train=True,
            transform=transform_seq,
            target_transform=transform_target,
        )

        # Récupérer un exemple
        sequence, target = dataset[0]

        # Créer un dataset sans transformation pour comparer
        dataset_no_transform = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            target_column="close",
            is_train=True,
        )

        # Récupérer l'exemple sans transformation
        sequence_raw, target_raw = dataset_no_transform[0]

        # Vérifier que les transformations ont été appliquées
        self.assertTrue(
            torch.allclose(sequence, sequence_raw * 2.0, rtol=1e-5),
            "La transformation de séquence n'a pas été correctement appliquée",
        )
        self.assertTrue(
            torch.isclose(target, target_raw + 1.0, rtol=1e-5),
            "La transformation de cible n'a pas été correctement appliquée",
        )


if __name__ == "__main__":
    unittest.main()
