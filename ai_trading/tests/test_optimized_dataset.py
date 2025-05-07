import io
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ai_trading.data.compressed_storage import CompressedStorage

# Importer les classes réellement disponibles dans le module
from ai_trading.data.optimized_dataset import convert_to_compressed


# Tester directement les fonctions de compression sans utiliser OptimizedFinancialDataset
class TestCompressedFunctions(unittest.TestCase):
    """Tests pour les fonctions de compression."""

    def setUp(self):
        """Configuration pour les tests."""
        # Créer un répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()

        # Créer un petit dataset de test
        dates = pd.date_range(start="2023-01-01", periods=100)
        self.test_data = pd.DataFrame(
            {
                "close": np.random.rand(100) * 100 + 50,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

    def tearDown(self):
        """Nettoyage après les tests."""
        # Supprimer les fichiers temporaires
        for filename in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, filename))
            except:
                pass
        os.rmdir(self.temp_dir)

    def test_compression_storage(self):
        """Test de la classe CompressedStorage."""
        # Créer un fichier à compresser
        raw_path = os.path.join(self.temp_dir, "test_data.parquet")
        self.test_data.to_parquet(raw_path)

        # Créer un objet de stockage compressé
        storage = CompressedStorage(compression_level=3)

        # Compresser le fichier
        compressed_path = storage.compress_file(raw_path)

        # Vérifier que le fichier compressé existe
        self.assertTrue(os.path.exists(compressed_path))

        # Décompresser le fichier
        output_path = os.path.join(self.temp_dir, "decompressed.parquet")
        storage.decompress_file(compressed_path, output_path)

        # Vérifier que le fichier décompressé existe
        self.assertTrue(os.path.exists(output_path))

        # Lire les données décompressées
        loaded_data = pd.read_parquet(output_path)

        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(self.test_data, loaded_data)

    def test_compress_dataframe(self):
        """Test de la compression d'un DataFrame."""
        # Compresser des données dans un buffer mémoire
        storage = CompressedStorage(compression_level=3)
        buffer = io.BytesIO()
        self.test_data.to_parquet(buffer)
        buffer.seek(0)

        # Compresser les données
        compressed_data = storage.compress_data(buffer.getvalue())

        # Vérifier que les données sont compressées
        self.assertIsInstance(compressed_data, bytes)

        # Décompresser les données
        decompressed_data = storage.decompress_data(compressed_data)

        # Charger les données décompressées dans un DataFrame
        buffer = io.BytesIO(decompressed_data)
        loaded_data = pd.read_parquet(buffer)

        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(self.test_data, loaded_data)

    def test_compression_json(self):
        """Test de la compression JSON."""
        # Créer des données JSON de test
        test_json = {
            "meta": {"version": 1, "timestamp": "2023-01-01", "source": "test"},
            "stats": {"mean": 100.5, "std": 15.2, "min": 50.0, "max": 150.0},
        }

        # Compresser les données JSON
        storage = CompressedStorage(compression_level=3)
        path = os.path.join(self.temp_dir, "test.json.zst")
        storage.save_json(test_json, path)

        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(path))

        # Charger les données
        loaded_json = storage.load_json(path)

        # Vérifier que les données sont identiques
        self.assertEqual(test_json["meta"]["version"], loaded_json["meta"]["version"])
        self.assertEqual(test_json["meta"]["source"], loaded_json["meta"]["source"])
        self.assertEqual(test_json["stats"]["mean"], loaded_json["stats"]["mean"])
        self.assertEqual(test_json["stats"]["std"], loaded_json["stats"]["std"])

    def test_convert_to_compressed(self):
        """Test de la fonction convert_to_compressed."""
        # Créer un fichier à compresser
        raw_path = os.path.join(self.temp_dir, "raw_data.parquet")
        self.test_data.to_parquet(raw_path)

        # Compresser avec la fonction de haut niveau
        compressed_path = convert_to_compressed(raw_path, compression_level=3)

        # Vérifier que le fichier compressé existe
        self.assertTrue(os.path.exists(compressed_path))

        # Vérifier qu'il se termine par .zst (en convertissant Path en str si nécessaire)
        compressed_path_str = str(compressed_path)
        self.assertTrue(compressed_path_str.endswith(".zst"))

        # Charger les données compressées
        storage = CompressedStorage()
        output_path = os.path.join(self.temp_dir, "loaded.parquet")
        storage.decompress_file(compressed_path, output_path)
        loaded_data = pd.read_parquet(output_path)

        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(self.test_data, loaded_data)


if __name__ == "__main__":
    unittest.main()
