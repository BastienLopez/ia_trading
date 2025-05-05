import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import numpy as np
import pandas as pd
import pytest

try:
    from ai_trading.data.compressed_storage import (
        CompressedStorage,
        CompressedFrame,
        COMPRESSION_LEVEL_DEFAULT,
        get_compressor,
    )
    ZSTD_AVAILABLE = True
except ImportError:
    # Mock des classes si le module zstandard n'est pas installé
    ZSTD_AVAILABLE = False
    CompressedStorage = MagicMock()
    CompressedFrame = MagicMock()
    COMPRESSION_LEVEL_DEFAULT = 3
    get_compressor = MagicMock()


@pytest.mark.skipif(not ZSTD_AVAILABLE, reason="Zstandard library not available")
class TestCompressedStorage(unittest.TestCase):
    """Test pour le système de stockage compressé."""

    def setUp(self):
        """Configuration pour les tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = CompressedStorage(base_path=self.temp_dir)

        # Créer des données de test
        self.test_data = pd.DataFrame(
            {"feature1": np.random.rand(100), "feature2": np.random.rand(100)}
        )

    def tearDown(self):
        """Nettoyage après les tests."""
        # Supprimer les fichiers temporaires
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def test_save_load_frame(self):
        """Test de la sauvegarde et du chargement d'un DataFrame."""
        if not ZSTD_AVAILABLE:
            self.skipTest("Zstandard library not available")
            
        frame_id = "test_frame"
        
        # Sauvegarder le DataFrame
        self.storage.save_frame(frame_id, self.test_data)
        
        # Vérifier que le fichier a été créé
        expected_path = os.path.join(self.temp_dir, f"{frame_id}.zst")
        self.assertTrue(os.path.exists(expected_path))
        
        # Charger le DataFrame
        loaded_data = self.storage.load_frame(frame_id)
        
        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(self.test_data, loaded_data)

    def test_frame_metadata(self):
        """Test des métadonnées de frame."""
        if not ZSTD_AVAILABLE:
            self.skipTest("Zstandard library not available")
            
        frame_id = "metadata_test"
        metadata = {"source": "test", "version": 1}
        
        # Sauvegarder avec métadonnées
        self.storage.save_frame(frame_id, self.test_data, metadata=metadata)
        
        # Récupérer les métadonnées
        loaded_metadata = self.storage.get_metadata(frame_id)
        
        # Vérifier les métadonnées
        self.assertEqual(metadata["source"], loaded_metadata["source"])
        self.assertEqual(metadata["version"], loaded_metadata["version"])

    def test_compressed_frame(self):
        """Test de la classe CompressedFrame."""
        if not ZSTD_AVAILABLE:
            self.skipTest("Zstandard library not available")
            
        frame = CompressedFrame(self.test_data)
        
        # Compresser les données
        compressed_data = frame.compress()
        self.assertIsInstance(compressed_data, bytes)
        
        # Décompresser les données
        decompressed_frame = CompressedFrame.decompress(compressed_data)
        pd.testing.assert_frame_equal(self.test_data, decompressed_frame)

    def test_compression_levels(self):
        """Test des différents niveaux de compression."""
        if not ZSTD_AVAILABLE:
            self.skipTest("Zstandard library not available")
            
        # Tester avec différents niveaux
        for level in [1, 5, 10]:
            frame = CompressedFrame(self.test_data)
            compressed_data = frame.compress(level=level)
            
            # Décompresser et vérifier
            decompressed = CompressedFrame.decompress(compressed_data)
            pd.testing.assert_frame_equal(self.test_data, decompressed)

    def test_compressor_selection(self):
        """Test de la sélection du compresseur."""
        if not ZSTD_AVAILABLE:
            self.skipTest("Zstandard library not available")
            
        # Tester le compresseur par défaut
        compressor = get_compressor()
        self.assertIsNotNone(compressor)
        
        # Tester avec un niveau spécifié
        compressor = get_compressor(level=7)
        self.assertIsNotNone(compressor)


if __name__ == "__main__":
    unittest.main()
