import io
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

# Importer les composants réellement disponibles
from ai_trading.data.compressed_storage import (
    CompressedStorage,
    compressed_to_dataframe,
    dataframe_to_compressed,
    optimize_compression_level,
)


# La classe de test n'a plus besoin d'être conditionnellement ignorée
class TestCompressedStorage(unittest.TestCase):
    """Test pour le système de stockage compressé."""

    def setUp(self):
        """Configuration pour les tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = CompressedStorage()

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

    def test_save_load_dataframe(self):
        """Test de la sauvegarde et du chargement d'un DataFrame."""
        # Utiliser dataframe_to_compressed et compressed_to_dataframe au lieu de save_frame
        output_path = os.path.join(self.temp_dir, "test_frame.zst")

        # Sauvegarder le DataFrame en utilisant la fonction disponible
        dataframe_to_compressed(self.test_data, output_path, format="parquet")

        # Vérifier que le fichier a été créé
        self.assertTrue(os.path.exists(output_path))

        # Charger le DataFrame
        loaded_data = compressed_to_dataframe(output_path, format="parquet")

        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(self.test_data, loaded_data)

    def test_json_metadata(self):
        """Test de la sauvegarde et du chargement de métadonnées JSON."""
        metadata = {"source": "test", "version": 1}
        output_path = os.path.join(self.temp_dir, "metadata.json.zst")

        # Utiliser la méthode save_json de CompressedStorage
        self.storage.save_json(metadata, output_path)

        # Vérifier que le fichier a été créé
        self.assertTrue(os.path.exists(output_path))

        # Charger les métadonnées
        loaded_metadata = self.storage.load_json(output_path)

        # Vérifier les métadonnées
        self.assertEqual(metadata["source"], loaded_metadata["source"])
        self.assertEqual(metadata["version"], loaded_metadata["version"])

    def test_compression_functions(self):
        """Test des fonctions de compression des données."""
        # Convertir DataFrame en bytes pour le test
        buffer = io.BytesIO()
        self.test_data.to_parquet(buffer)
        data_bytes = buffer.getvalue()

        # Compresser les données avec CompressedStorage
        compressed_data = self.storage.compress_data(data_bytes)
        self.assertIsInstance(compressed_data, bytes)

        # Décompresser les données
        decompressed_data = self.storage.decompress_data(compressed_data)

        # Charger à nouveau dans un DataFrame et vérifier
        buffer = io.BytesIO(decompressed_data)
        df_decompressed = pd.read_parquet(buffer)
        pd.testing.assert_frame_equal(self.test_data, df_decompressed)

    def test_compression_levels(self):
        """Test des différents niveaux de compression."""
        # Convertir DataFrame en bytes pour le test
        buffer = io.BytesIO()
        self.test_data.to_parquet(buffer)
        data_bytes = buffer.getvalue()

        sizes = {}

        # Tester différents niveaux de compression
        for level in [1, 5, 10]:
            # Créer un nouveau CompressedStorage avec le niveau spécifié
            storage = CompressedStorage(compression_level=level)

            # Compresser les données
            compressed = storage.compress_data(data_bytes)

            # Stocker la taille pour comparaison
            sizes[level] = len(compressed)

            # Décompresser et vérifier
            decompressed = storage.decompress_data(compressed)

            # Charger à nouveau dans un DataFrame et vérifier
            buffer = io.BytesIO(decompressed)
            df_decompressed = pd.read_parquet(buffer)
            pd.testing.assert_frame_equal(self.test_data, df_decompressed)

        # Vérifier que niveau supérieur = meilleure compression (taille plus petite)
        self.assertLessEqual(sizes[10], sizes[5])
        self.assertLessEqual(sizes[5], sizes[1])

    def test_compression_optimization(self):
        """Test de l'optimisation du niveau de compression."""
        buffer = io.BytesIO()
        self.test_data.to_parquet(buffer)
        data_bytes = buffer.getvalue()

        # Tester optimize_compression_level avec des niveaux limités pour la rapidité
        best_level, results = optimize_compression_level(data_bytes, test_levels=[1, 3])

        # Vérifier le résultat
        self.assertIn(best_level, [1, 3])
        self.assertEqual(len(results), 2)  # Deux niveaux testés

        # Vérifier que les résultats contiennent les métriques attendues
        for level in [1, 3]:
            self.assertIn(level, results)
            self.assertIn("compression_ratio", results[level])
            self.assertIn("compression_time", results[level])


if __name__ == "__main__":
    unittest.main()
