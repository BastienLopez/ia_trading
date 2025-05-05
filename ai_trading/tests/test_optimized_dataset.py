import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

try:
    from ai_trading.data.optimized_dataset import (
        OptimizedDataset,
        CompressedDataset,
        ParallelDataLoader,
    )
    from ai_trading.data.compressed_storage import CompressedStorage
    DEPS_AVAILABLE = True
except ImportError:
    # Mocks si les modules ne sont pas disponibles
    DEPS_AVAILABLE = False
    OptimizedDataset = MagicMock()
    CompressedDataset = MagicMock()
    ParallelDataLoader = MagicMock()
    CompressedStorage = MagicMock()


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required libraries not available")
class TestOptimizedDataset(unittest.TestCase):
    """Tests pour la classe OptimizedDataset."""

    def setUp(self):
        """Configuration pour les tests."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        # Créer un répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer un petit dataset de test
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'date': dates,
            'price': np.random.rand(100) * 100 + 50,
            'volume': np.random.randint(1000, 10000, 100),
            'returns': np.random.randn(100) * 0.02
        })
        
        # Créer une instance du dataset
        self.dataset = OptimizedDataset(
            data=self.test_data,
            cache_dir=self.temp_dir
        )
        
    def tearDown(self):
        """Nettoyage après les tests."""
        # Supprimer les fichiers temporaires
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)
        
    def test_init_with_data(self):
        """Teste l'initialisation avec des données."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        self.assertEqual(len(self.dataset), len(self.test_data))
        pd.testing.assert_frame_equal(self.dataset.data, self.test_data)
        
    def test_getitem(self):
        """Teste l'accès aux éléments."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        item = self.dataset[10]
        self.assertIsInstance(item, pd.Series)
        self.assertEqual(item.name, 10)
        
    def test_len(self):
        """Teste la méthode __len__."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        self.assertEqual(len(self.dataset), 100)
        
    def test_preprocess(self):
        """Teste le prétraitement des données."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        # Définir une fonction de prétraitement simple
        def add_ma(df):
            df['ma5'] = df['price'].rolling(5).mean()
            return df.dropna()
            
        # Appliquer le prétraitement
        self.dataset.preprocess(add_ma)
        
        # Vérifier que la colonne a été ajoutée
        self.assertIn('ma5', self.dataset.data.columns)
        # Vérifier que les lignes NaN ont été supprimées
        self.assertEqual(len(self.dataset), 96)  # 100 - 4 (rolling 5)
        
    def test_save_load(self):
        """Teste la sauvegarde et le chargement."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        # Sauvegarder
        path = os.path.join(self.temp_dir, "test_dataset.parquet")
        self.dataset.save(path)
        
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(path))
        
        # Charger dans un nouveau dataset
        loaded = OptimizedDataset()
        loaded.load(path)
        
        # Vérifier que les données sont identiques
        self.assertEqual(len(loaded), len(self.dataset))
        pd.testing.assert_frame_equal(loaded.data, self.dataset.data)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required libraries not available")
class TestCompressedDataset(unittest.TestCase):
    """Tests pour la classe CompressedDataset."""
    
    def setUp(self):
        """Configuration pour les tests."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        # Créer un répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer un petit dataset de test
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'date': dates,
            'price': np.random.rand(100) * 100 + 50,
            'volume': np.random.randint(1000, 10000, 100),
            'returns': np.random.randn(100) * 0.02
        })
        
        # Créer une instance du dataset compressé
        self.dataset = CompressedDataset(
            data=self.test_data,
            cache_dir=self.temp_dir,
            compression_level=3
        )
        
    def tearDown(self):
        """Nettoyage après les tests."""
        # Supprimer les fichiers temporaires
        for filename in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, filename))
            except:
                pass  # Ignorer les erreurs
        os.rmdir(self.temp_dir)
        
    def test_init_with_compression(self):
        """Teste l'initialisation avec compression."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        self.assertEqual(len(self.dataset), len(self.test_data))
        self.assertEqual(self.dataset.compression_level, 3)
        self.assertIsInstance(self.dataset.storage, CompressedStorage)
        
    def test_compress_decompress(self):
        """Teste la compression et décompression."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        # Sauvegarder de manière compressée
        path = os.path.join(self.temp_dir, "compressed_dataset")
        self.dataset.save_compressed(path)
        
        # Vérifier que le fichier compressé existe
        self.assertTrue(os.path.exists(f"{path}.zst"))
        
        # Charger à partir du fichier compressé
        loaded = CompressedDataset()
        loaded.load_compressed(path)
        
        # Vérifier que les données sont identiques
        self.assertEqual(len(loaded), len(self.dataset))
        pd.testing.assert_frame_equal(loaded.data, self.dataset.data)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required libraries not available")
class TestParallelDataLoader(unittest.TestCase):
    """Tests pour la classe ParallelDataLoader."""
    
    def setUp(self):
        """Configuration pour les tests."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        # Créer un répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer plusieurs petits datasets de test
        self.test_files = []
        for i in range(3):
            dates = pd.date_range(start=f'2023-0{i+1}-01', periods=50)
            data = pd.DataFrame({
                'date': dates,
                'price': np.random.rand(50) * 100 + 50,
                'symbol': f'ASSET_{i}',
                'returns': np.random.randn(50) * 0.02
            })
            
            # Sauvegarder chaque dataset
            path = os.path.join(self.temp_dir, f"data_{i}.parquet")
            data.to_parquet(path)
            self.test_files.append(path)
            
        # Créer le chargeur parallèle
        self.loader = ParallelDataLoader(
            num_workers=2,
            batch_size=10
        )
        
    def tearDown(self):
        """Nettoyage après les tests."""
        # Supprimer les fichiers temporaires
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)
        
    def test_load_files(self):
        """Teste le chargement parallèle des fichiers."""
        if not DEPS_AVAILABLE:
            self.skipTest("Required libraries not available")
            
        # Charger les fichiers
        result = self.loader.load_files(self.test_files)
        
        # Vérifier que les données sont correctes
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 50 * 3)  # 3 fichiers de 50 lignes chacun
        
        # Vérifier que tous les symboles sont présents
        unique_symbols = result['symbol'].unique()
        self.assertEqual(len(unique_symbols), 3)
        self.assertTrue('ASSET_0' in unique_symbols)
        self.assertTrue('ASSET_1' in unique_symbols)
        self.assertTrue('ASSET_2' in unique_symbols)


if __name__ == "__main__":
    unittest.main()
