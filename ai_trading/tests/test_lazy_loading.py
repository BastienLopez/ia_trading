"""
Tests pour les fonctionnalités de chargement paresseux (lazy loading)
et mise en cache des features du module financial_dataset.py.
"""

import gc
import os
import tempfile
import time
import unittest

import numpy as np
import psutil
import torch

from ai_trading.data.financial_dataset import (
    FinancialDataset,
    get_feature_transform_fn,
    get_financial_dataloader,
)
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data


def get_memory_usage():
    """Retourne l'utilisation mémoire en MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB


class TestLazyLoading(unittest.TestCase):
    """Tests pour les fonctionnalités de chargement paresseux et mise en cache des features."""

    def setUp(self):
        """Préparation des données pour les tests."""
        # Créer un répertoire temporaire pour les fichiers
        self.temp_dir = tempfile.TemporaryDirectory()

        # Paramètres pour les tests
        self.sequence_length = 50
        self.batch_size = 16

        # Générer des données synthétiques avec les paramètres corrects
        self.synthetic_data = generate_synthetic_market_data(
            n_points=1000,  # Nombre de points à générer
            trend=0.0005,  # Tendance positive
            volatility=0.01,  # Volatilité modérée
            start_price=100.0,  # Prix de départ
            with_date=True,  # Inclure dates
            cyclic_pattern=True,  # Inclure cycles
            include_volume=True,  # Inclure volume
        )

        # Enregistrer les données dans différents formats pour les tests
        self.csv_path = os.path.join(self.temp_dir.name, "synthetic_data.csv")
        self.synthetic_data.to_csv(self.csv_path, index=True)

        try:
            # Enregistrer en format Parquet si disponible
            self.parquet_path = os.path.join(
                self.temp_dir.name, "synthetic_data.parquet"
            )
            self.synthetic_data.to_parquet(self.parquet_path, index=True)
        except ImportError:
            self.parquet_path = None
            print("Avertissement: PyArrow non disponible, tests Parquet ignorés")

        try:
            # Enregistrer en format HDF5 si disponible
            self.hdf5_path = os.path.join(self.temp_dir.name, "synthetic_data.h5")
            self.synthetic_data.to_hdf(self.hdf5_path, key="data", index=True)
        except ImportError:
            self.hdf5_path = None
            print("Avertissement: h5py/pytables non disponible, tests HDF5 ignorés")

        # Créer un tenseur pour les tests
        features = self.synthetic_data.drop(columns=["close"]).values
        targets = self.synthetic_data["close"].values
        self.tensor_features = torch.tensor(features, dtype=torch.float32)
        self.tensor_targets = torch.tensor(targets, dtype=torch.float32)

        # Afficher la taille des données
        print(
            f"\nDonnées synthétiques: {len(self.synthetic_data)} points ({self.synthetic_data.shape})"
        )
        print(f"Utilisation mémoire initiale: {get_memory_usage():.2f} MB")

    def tearDown(self):
        """Nettoyer après les tests."""
        # Forcer le nettoyage de la mémoire
        gc.collect()
        # Nettoyer les fichiers temporaires
        self.temp_dir.cleanup()

    def test_memory_usage_comparison(self):
        """Compare l'utilisation mémoire entre chargement normal et paresseux."""
        # Forcer le garbage collector avant de mesurer
        gc.collect()

        # Mesurer la mémoire initiale
        initial_memory = get_memory_usage()

        # Créer un dataset standard (chargement complet)
        eager_dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            is_train=True,
            lazy_loading=False,
        )

        # Mesurer la mémoire après chargement complet
        eager_memory = get_memory_usage()

        # Nettoyer et forcer le garbage collector
        del eager_dataset
        gc.collect()
        time.sleep(1)  # Laisser le temps au système de libérer la mémoire

        # Créer un dataset avec lazy loading
        lazy_dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            is_train=True,
            lazy_loading=True,
            chunk_size=10000,
        )

        # Mesurer la mémoire avec lazy loading sans charger de données
        lazy_init_memory = get_memory_usage()

        # Accéder à quelques éléments pour charger partiellement les données
        for i in range(0, 50, 10):
            lazy_dataset[i]

        # Mesurer la mémoire après chargement partiel
        lazy_partial_memory = get_memory_usage()

        # Afficher les résultats
        print(f"\nUtilisation mémoire (MB):")
        print(f"Initiale: {initial_memory:.2f}")
        print(
            f"Chargement complet: {eager_memory:.2f} (delta: {eager_memory - initial_memory:.2f})"
        )
        print(
            f"Lazy loading (init): {lazy_init_memory:.2f} (delta: {lazy_init_memory - initial_memory:.2f})"
        )
        print(
            f"Lazy loading (partiel): {lazy_partial_memory:.2f} (delta: {lazy_partial_memory - initial_memory:.2f})"
        )

        # Vérifier que le lazy loading utilise moins de mémoire à l'initialisation
        self.assertLess(
            lazy_init_memory - initial_memory,
            eager_memory - initial_memory,
            "Le lazy loading devrait utiliser significativement moins de mémoire à l'initialisation",
        )

    def test_lazy_loading_data_consistency(self):
        """Vérifie que les données chargées paresseusement sont identiques au chargement complet."""
        # Créer datasets avec et sans lazy loading, en utilisant un sous-ensemble des données
        subset_data = self.synthetic_data.iloc[
            :200
        ]  # Limitation à 200 points pour éviter les indices hors limites

        eager_dataset = FinancialDataset(
            data=subset_data,
            sequence_length=self.sequence_length,
            is_train=True,
            lazy_loading=False,
        )

        lazy_dataset = FinancialDataset(
            data=subset_data,
            sequence_length=self.sequence_length,
            is_train=True,
            lazy_loading=True,
            chunk_size=100,  # Taille de chunk plus petite pour ce test
        )

        # Calculer le nombre réel d'exemples disponibles
        max_valid_idx = len(eager_dataset) - 1
        print(f"Nombre maximum d'indices valides: {max_valid_idx}")

        # Comparer seulement les indices valides
        test_indices = [0, 10, 20, 30]
        for i in test_indices:
            if i <= max_valid_idx:
                eager_seq, eager_target = eager_dataset[i]
                lazy_seq, lazy_target = lazy_dataset[i]

                # Vérifier l'égalité des séquences
                np.testing.assert_allclose(
                    eager_seq.numpy(),
                    lazy_seq.numpy(),
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Séquences différentes à l'index {i}",
                )

                # Vérifier l'égalité des cibles si présentes
                if eager_target is not None and lazy_target is not None:
                    np.testing.assert_allclose(
                        eager_target.numpy(),
                        lazy_target.numpy(),
                        rtol=1e-4,
                        atol=1e-4,
                        err_msg=f"Cibles différentes à l'index {i}",
                    )
            else:
                print(f"Index {i} hors limites, skip")

    def test_file_lazy_loading(self):
        """Teste le chargement paresseux à partir d'un fichier."""
        if not self.parquet_path:
            self.skipTest("Format Parquet requis pour ce test")

        # Mesurer la mémoire initiale
        initial_memory = get_memory_usage()

        # Créer un dataset avec lazy loading depuis un fichier
        lazy_dataset = FinancialDataset(
            data=self.parquet_path,
            sequence_length=self.sequence_length,
            is_train=True,
            lazy_loading=True,
            chunk_size=5000,
        )

        # Mesurer la mémoire après initialisation
        init_memory = get_memory_usage()

        # Vérifier que l'initialisation n'a pas chargé toutes les données
        self.assertLess(
            init_memory - initial_memory,
            50,  # Une limite raisonnable pour l'overhead d'initialisation (en MB)
            "L'initialisation lazy depuis fichier ne devrait pas charger toutes les données",
        )

        # Accéder à quelques éléments pour vérifier que le chargement fonctionne
        # Utiliser des indices plus petits pour éviter les erreurs hors limites
        valid_indices = [0, 10, 20]
        for i in valid_indices:
            sequence, target = lazy_dataset[i]
            # Vérifier que les séquences ont la bonne forme
            self.assertEqual(sequence.shape[0], self.sequence_length)
            if target is not None:
                self.assertEqual(target.dim(), 0)  # target devrait être un scalaire

    def test_cached_feature_transform(self):
        """Teste la mise en cache de transformations de features coûteuses."""

        # Créer une transformation "coûteuse" pour simuler un calcul intensif
        @get_feature_transform_fn(cache_size=100)
        def expensive_transform(tensor):
            # Simuler un calcul coûteux (ex: calcul d'indicateurs techniques)
            time.sleep(0.01)  # Petite pause pour simuler un calcul intense
            return tensor * 2.0  # Transformation simple pour le test

        # Créer un dataset avec la transformation
        dataset = FinancialDataset(
            data=self.synthetic_data.iloc[:1000],  # Petit subset pour le test
            sequence_length=self.sequence_length,
            is_train=True,
            transform=expensive_transform,
        )

        # Premier accès (devrait être lent car pas encore en cache)
        start_time = time.time()
        first_seq, _ = dataset[0]
        first_access_time = time.time() - start_time

        # Deuxième accès au même élément (devrait être plus rapide grâce au cache)
        start_time = time.time()
        second_seq, _ = dataset[0]
        second_access_time = time.time() - start_time

        # Afficher les temps d'accès
        print(f"\nTemps d'accès (s):")
        print(f"Premier accès: {first_access_time:.4f}")
        print(f"Deuxième accès (cached): {second_access_time:.4f}")
        print(
            f"Ratio d'accélération: {first_access_time / max(second_access_time, 0.0001):.2f}x"
        )

        # Vérifier que le deuxième accès est significativement plus rapide
        self.assertLess(
            second_access_time,
            first_access_time * 0.5,  # Au moins 2x plus rapide
            "Le deuxième accès devrait être significativement plus rapide grâce au cache",
        )

        # Vérifier que les séquences sont identiques (même transformation)
        np.testing.assert_allclose(
            first_seq.numpy(),
            second_seq.numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Les séquences transformées devraient être identiques",
        )

    def test_dataloader_with_lazy_loading(self):
        """Teste l'intégration du chargement paresseux avec DataLoader."""
        # Créer un dataset avec lazy loading
        lazy_dataset = FinancialDataset(
            data=self.synthetic_data,
            sequence_length=self.sequence_length,
            is_train=True,
            lazy_loading=True,
            chunk_size=100,
        )

        # Créer un DataLoader optimisé avec un seul worker pour éviter les problèmes de sérialisation
        dataloader = get_financial_dataloader(
            lazy_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Utiliser 0 worker pour éviter les problèmes de sérialisation
            prefetch_factor=None,
        )

        # Parcourir quelques batchs pour vérifier l'intégration
        batch_count = 0
        for batch_data in dataloader:
            batch_sequences, batch_targets = batch_data
            # Vérifier que les batchs ont la bonne forme
            self.assertEqual(batch_sequences.shape[0], self.batch_size)
            self.assertEqual(batch_sequences.shape[1], self.sequence_length)
            batch_count += 1

            # Limiter à 5 batchs pour le test
            if batch_count >= 5:
                break

        # Vérifier qu'on a bien pu itérer sur les batchs
        self.assertEqual(batch_count, 5, "Devrait pouvoir itérer sur 5 batchs")

    @unittest.skipIf(not torch.cuda.is_available(), "GPU requis pour ce test")
    def test_precomputed_features_gpu(self):
        """Teste le prétraitement des features avec transfert GPU."""

        # Définir une transformation avec prétraitement et cache
        @get_feature_transform_fn(cache_size=50)
        def gpu_transform(tensor):
            # Simuler un calcul GPU
            if torch.cuda.is_available():
                gpu_tensor = tensor.cuda()
                result = gpu_tensor * 2.0  # Opération simple
                return result.cpu()  # Retourner au CPU
            else:
                return tensor * 2.0

        # Dataset avec prétraitement des features
        dataset = FinancialDataset(
            data=self.synthetic_data.iloc[:1000],
            sequence_length=self.sequence_length,
            is_train=True,
            transform=gpu_transform,
            precompute_features=True,
        )

        # Accéder à plusieurs éléments
        for i in range(10):
            sequence, _ = dataset[i]
            self.assertEqual(sequence.shape[0], self.sequence_length)

        # Préchargement des données GPU
        dataloader = get_financial_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Simplifier pour le test GPU
            pin_memory=True,  # Important pour transfert GPU efficace
        )

        # Simuler un traitement par batch avec GPU
        device = torch.device("cuda")
        for batch_data in dataloader:
            batch_sequences, batch_targets = batch_data
            # Transférer sur GPU
            batch_sequences = batch_sequences.to(device)
            if batch_targets is not None:
                batch_targets = batch_targets.to(device)

            # Vérifier que les tenseurs sont bien sur GPU
            self.assertTrue(batch_sequences.is_cuda)
            if batch_targets is not None:
                self.assertTrue(batch_targets.is_cuda)

            # Simuler une opération GPU
            result = batch_sequences + 1.0
            self.assertTrue(result.is_cuda)

            break  # Un seul batch suffit pour le test

        # Si on arrive ici, le test est réussi


if __name__ == "__main__":
    unittest.main()
