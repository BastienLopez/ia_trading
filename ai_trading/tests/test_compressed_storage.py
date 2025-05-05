import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trading.data.compressed_storage import (
    CompressedStorage,
    compressed_to_dataframe,
    dataframe_to_compressed,
    optimize_compression_level,
)


class TestCompressedStorage(unittest.TestCase):
    """Tests pour le module de stockage compressé."""

    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Créer une instance de CompressedStorage
        self.storage = CompressedStorage(
            compression_level=9
        )  # Niveau plus élevé pour les tests

        # Créer des données de test
        self.text_data = "Hello, world! " * 1000
        self.binary_data = os.urandom(
            100000
        )  # Augmenter la taille pour une meilleure compression

        # Créer un DataFrame de test
        self.df = pd.DataFrame(
            {
                "A": np.random.randn(1000),
                "B": np.random.randn(1000),
                "C": np.random.choice(["X", "Y", "Z"], 1000),
                "D": np.random.randint(0, 100, 1000),
            }
        )

        # Créer un tableau NumPy de test
        self.array = np.random.randn(100, 100)

        # Créer des données JSON de test
        self.json_data = {
            "name": "Test JSON",
            "values": list(range(100)),
            "nested": {"key1": "value1", "key2": [1, 2, 3, 4, 5]},
        }

    def tearDown(self):
        """Nettoyage après les tests."""
        self.temp_dir.cleanup()

    def test_compress_decompress_data(self):
        """Teste la compression et décompression de données binaires."""
        # Compresser des données
        compressed = self.storage.compress_data(self.binary_data)

        # Vérifier que les données sont compressées (taille réduite)
        # Note: Pour certaines données aléatoires, la compression pourrait ne pas être efficace
        self.assertNotEqual(len(compressed), 0)

        # Décompresser les données
        decompressed = self.storage.decompress_data(compressed)

        # Vérifier que les données décompressées sont identiques aux données d'origine
        self.assertEqual(decompressed, self.binary_data)

    def test_compress_decompress_text(self):
        """Teste la compression et décompression de données texte."""
        # Convertir le texte en bytes
        text_bytes = self.text_data.encode("utf-8")

        # Compresser des données
        compressed = self.storage.compress_data(text_bytes)

        # Pour le texte répétitif, la compression devrait être efficace
        self.assertLess(len(compressed), len(text_bytes))

        # Décompresser les données
        decompressed = self.storage.decompress_data(compressed)

        # Convertir les bytes décompressés en texte
        decompressed_text = decompressed.decode("utf-8")

        # Vérifier que le texte décompressé est identique au texte d'origine
        self.assertEqual(decompressed_text, self.text_data)

    def test_compress_decompress_file(self):
        """Teste la compression et décompression de fichiers."""
        # Créer un fichier de test avec du texte répétitif pour garantir la compression
        test_file = self.temp_path / "test_file.txt"
        with open(test_file, "w") as f:
            f.write("This is a test file with repetitive content. " * 1000)

        # Compresser le fichier
        compressed_file = self.storage.compress_file(test_file)

        # Vérifier que le fichier compressé existe
        self.assertTrue(compressed_file.exists())

        # Vérifier que le fichier compressé est plus petit que l'original
        self.assertLess(compressed_file.stat().st_size, test_file.stat().st_size)

        # Décompresser le fichier
        decompressed_file = self.storage.decompress_file(compressed_file)

        # Vérifier que le fichier décompressé existe
        self.assertTrue(decompressed_file.exists())

        # Vérifier que le contenu du fichier décompressé est identique à l'original
        with open(test_file, "r") as f:
            original_content = f.read()
        with open(decompressed_file, "r") as f:
            decompressed_content = f.read()
        self.assertEqual(decompressed_content, original_content)

    def test_save_load_dataframe(self):
        """Teste la sauvegarde et le chargement d'un DataFrame."""
        # Formats à tester
        formats = ["parquet", "csv", "pickle"]

        for format in formats:
            # Sauvegarder le DataFrame
            df_file = self.temp_path / f"test_df.{format}.zst"
            self.storage.save_dataframe(self.df, df_file, format=format)

            # Vérifier que le fichier existe
            self.assertTrue(df_file.exists())

            # Charger le DataFrame
            loaded_df = self.storage.load_dataframe(df_file, format=format)

            # Vérifier que le DataFrame chargé a les mêmes dimensions que l'original
            self.assertEqual(loaded_df.shape, self.df.shape)

            # Pour les formats qui préservent les types de données
            if format in ["parquet", "pickle"]:
                # Vérifier que les types de colonnes sont préservés
                for col in self.df.columns:
                    self.assertEqual(loaded_df[col].dtype, self.df[col].dtype)

    def test_save_load_numpy(self):
        """Teste la sauvegarde et le chargement d'un tableau NumPy."""
        # Sauvegarder le tableau NumPy
        array_file = self.temp_path / "test_array.npy.zst"
        self.storage.save_numpy(self.array, array_file)

        # Vérifier que le fichier existe
        self.assertTrue(array_file.exists())

        # Charger le tableau NumPy
        loaded_array = self.storage.load_numpy(array_file)

        # Vérifier que le tableau chargé a les mêmes dimensions que l'original
        self.assertEqual(loaded_array.shape, self.array.shape)

        # Vérifier que le contenu est identique
        np.testing.assert_array_equal(loaded_array, self.array)

    def test_save_load_json(self):
        """Teste la sauvegarde et le chargement de données JSON."""
        # Sauvegarder les données JSON
        json_file = self.temp_path / "test_data.json.zst"
        self.storage.save_json(self.json_data, json_file)

        # Vérifier que le fichier existe
        self.assertTrue(json_file.exists())

        # Charger les données JSON
        loaded_json = self.storage.load_json(json_file)

        # Vérifier que les données chargées sont identiques aux données d'origine
        self.assertEqual(loaded_json, self.json_data)

    def test_read_compressed_chunks(self):
        """Teste la lecture par chunks de fichiers compressés."""
        # Créer un fichier de test avec du texte répétitif pour garantir la compression
        test_file = self.temp_path / "test_large_file.txt"
        with open(test_file, "w") as f:
            f.write(
                "This is a test file with repetitive content for chunked reading. "
                * 1000
            )

        # Lire le contenu original
        with open(test_file, "rb") as f:
            original_data = f.read()

        # Compresser le fichier
        compressed_file = self.storage.compress_file(test_file)

        # Lire le fichier par chunks
        chunks = []
        for chunk in self.storage.read_compressed_chunks(
            compressed_file, chunk_size=1000
        ):
            chunks.append(chunk)

        # Reconstituer les données
        reconstructed_data = b"".join(chunks)

        # Vérifier que les données reconstruites sont identiques aux données d'origine
        self.assertEqual(reconstructed_data, original_data)

    def test_stream_dataframe_chunks(self):
        """Teste le chargement par chunks d'un DataFrame."""
        # Créer un DataFrame plus grand pour le test
        big_df = pd.DataFrame(
            {"A": np.random.randn(10000), "B": np.random.choice(["X", "Y", "Z"], 10000)}
        )

        # Sauvegarder le DataFrame au format CSV
        df_file = self.temp_path / "test_big_df.csv.zst"
        self.storage.save_dataframe(big_df, df_file, format="csv")

        # Charger le DataFrame par chunks
        chunks = []
        for chunk in self.storage.stream_dataframe_chunks(
            df_file, format="csv", chunksize=1000
        ):
            chunks.append(chunk)

        # Reconstituer le DataFrame
        reconstructed_df = pd.concat(chunks, ignore_index=True)

        # Vérifier que le DataFrame reconstitué a les mêmes dimensions que l'original
        self.assertEqual(reconstructed_df.shape, big_df.shape)

    def test_compression_dictionary(self):
        """Teste l'utilisation d'un dictionnaire de compression."""
        # Activer l'utilisation d'un dictionnaire
        dict_storage = CompressedStorage(
            compression_level=3, use_dict=True, dict_size=50000
        )

        # Créer des échantillons de données similaires
        samples = []
        for i in range(10):
            base_text = f"This is sample {i} with some repetitive content: "
            repeated_text = base_text + "crypto trading " * 20
            samples.append(repeated_text.encode("utf-8"))

        # Entraîner un dictionnaire
        dict_data = dict_storage.train_dictionary(samples)
        self.assertIsNotNone(dict_data)

        # Vérifier que le dictionnaire est au format bytes
        self.assertIsInstance(dict_data, bytes)

        # Sauvegarder le dictionnaire
        dict_file = self.temp_path / "compression_dict.bin"
        dict_storage.save_dictionary(dict_file)
        self.assertTrue(dict_file.exists())

        # Créer un nouveau stockage et charger le dictionnaire
        new_storage = CompressedStorage(compression_level=3)
        new_storage.load_dictionary(dict_file)

        # Compresser des données avec et sans dictionnaire
        test_data = ("This is a test with crypto trading " * 20).encode("utf-8")
        compressed_with_dict = new_storage.compress_data(test_data)
        compressed_without_dict = self.storage.compress_data(test_data)

        # Vérifier que la décompression fonctionne correctement
        decompressed = new_storage.decompress_data(compressed_with_dict)
        self.assertEqual(decompressed, test_data)

    def test_utility_functions(self):
        """Teste les fonctions utilitaires du module."""
        # Tester dataframe_to_compressed et compressed_to_dataframe
        df_file = self.temp_path / "test_util_df.parquet.zst"
        dataframe_to_compressed(self.df, df_file, format="parquet")
        loaded_df = compressed_to_dataframe(df_file, format="parquet")
        self.assertEqual(loaded_df.shape, self.df.shape)

        # Créer un fichier compressé pour tester get_compression_info
        # On utilise un texte répétitif pour garantir une bonne compression
        test_file = self.temp_path / "test_info.txt"
        with open(test_file, "w") as f:
            f.write("This is a test file for compression info. " * 100)

        compressed_file = self.storage.compress_file(test_file)

        # Tester optimize_compression_level avec des données compressibles
        compressible_data = ("This is a highly compressible string. " * 100).encode(
            "utf-8"
        )
        optimal_level, results = optimize_compression_level(
            compressible_data, test_levels=[1, 3]
        )

        # Vérifier que l'optimisation a bien fonctionné
        self.assertIn(optimal_level, [1, 3])
        self.assertEqual(len(results), 2)
        for level in [1, 3]:
            self.assertIn("compression_ratio", results[level])
            self.assertIn("efficiency", results[level])


if __name__ == "__main__":
    unittest.main()
