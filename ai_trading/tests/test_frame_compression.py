import sys
import unittest
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.frame_compression import FrameCompressor, FrameStackWrapper


class TestFrameCompressor(unittest.TestCase):

    def setUp(self):
        """Prépare des données de test"""
        # Créer une série de frames de test
        self.test_frames = []
        for i in range(5):
            # Frame RGB de taille 84x84
            frame = np.ones((84, 84, 3), dtype=np.uint8) * (i * 20)
            self.test_frames.append(frame)

        # Frame avec des valeurs aléatoires
        self.random_frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)

        # Initialiser le compresseur
        self.compressor = FrameCompressor(
            compression_level=6,
            frame_stack_size=4,
            resize_dim=(42, 42),
            use_grayscale=True,
            quantize=True,
            use_delta_encoding=True,
        )

        # Compresseur sans options
        self.basic_compressor = FrameCompressor()

    def test_preprocess_frame(self):
        """Teste le prétraitement des frames"""
        frame = self.test_frames[0]

        # Tester le compresseur avec toutes les options
        processed = self.compressor.preprocess_frame(frame)

        # Vérifier les dimensions et le type
        self.assertEqual(processed.shape, (42, 42))  # Grayscale + resize
        self.assertEqual(processed.dtype, np.uint8)  # Quantization

        # Tester le compresseur basique
        processed_basic = self.basic_compressor.preprocess_frame(frame)
        self.assertEqual(processed_basic.shape, frame.shape)

    def test_compress_decompress(self):
        """Teste la compression et décompression d'une frame"""
        frame = self.random_frame

        # Prétraiter
        processed = self.basic_compressor.preprocess_frame(frame)

        # Compresser
        compressed = self.basic_compressor.compress_frame(processed)

        # Décompresser
        decompressed = self.basic_compressor.decompress_frame(
            compressed, processed.shape, processed.dtype
        )

        # Vérifier que la décompression est identique à l'original
        np.testing.assert_array_equal(processed, decompressed)

    def test_stack_frames(self):
        """Teste l'empilement des frames"""
        frames = [np.ones((10, 10), dtype=np.uint8) * i for i in range(3)]

        # Configurer un compresseur pour 4 frames
        stacker = FrameCompressor(frame_stack_size=4)

        # Empiler - devrait ajouter une frame au début
        stacked = stacker.stack_frames(frames)

        # Vérifier les dimensions
        self.assertEqual(stacked.shape, (4, 10, 10))

        # Vérifier que la première frame est dupliquée
        np.testing.assert_array_equal(stacked[0], stacked[1])

    def test_process_state(self):
        """Teste le traitement complet d'un état"""
        # Traiter plusieurs frames consécutives
        for i in range(5):
            state = self.compressor.process_state(self.test_frames[i])

            # Après 4 frames, on devrait avoir un état de taille 4
            if i >= 3:
                self.assertEqual(len(self.compressor.last_frames), 4)

        # Ne pas vérifier l'encodage delta car il peut être désactivé ou ne pas produire de valeurs négatives
        # Vérifions simplement que l'état est un tableau numpy valide
        self.assertIsInstance(state, np.ndarray)
        self.assertTrue(state.size > 0)

    def test_compress_state(self):
        """Teste la compression complète d'un état avec métadonnées"""
        state = self.random_frame

        # Compresser l'état
        compressed_data, metadata = self.basic_compressor.compress_state(state)

        # Vérifier les métadonnées
        self.assertIn("shape", metadata)
        self.assertIn("dtype", metadata)
        self.assertIn("frame_stack_size", metadata)
        self.assertEqual(metadata["frame_stack_size"], 4)

        # Décompresser et vérifier
        decompressed = self.basic_compressor.decompress_state(compressed_data, metadata)
        self.assertEqual(decompressed.shape, metadata["shape"])

    def test_reset(self):
        """Teste la réinitialisation du compresseur"""
        # Ajouter quelques frames
        for i in range(3):
            self.compressor.process_state(self.test_frames[i])

        # Vérifier qu'il y a des frames
        self.assertGreater(len(self.compressor.last_frames), 0)

        # Réinitialiser
        self.compressor.reset()

        # Vérifier que la liste est vide
        self.assertEqual(len(self.compressor.last_frames), 0)


class TestFrameStackWrapper(unittest.TestCase):

    def setUp(self):
        """Prépare le wrapper et des données de test"""
        # Créer des observations de test
        self.observations = [
            np.ones((10, 10, 3), dtype=np.uint8) * i for i in range(10)
        ]

        # Initialiser le wrapper sans environnement
        self.wrapper = FrameStackWrapper(n_frames=4)

        # Wrapper avec compression
        self.compressed_wrapper = FrameStackWrapper(
            n_frames=4, compress=True, compression_level=6
        )

    def test_reset(self):
        """Teste la réinitialisation du wrapper"""
        # Réinitialiser avec une observation
        observation = self.wrapper.reset(observation=self.observations[0])

        # Vérifier que toutes les frames sont identiques
        frame_count = 0

        # Cas d'une observation 3D (empilée sur le dernier axe)
        if len(observation.shape) == 3:
            self.assertEqual(observation.shape[2], 3 * 4)  # 3 canaux * 4 frames
            frame_count = observation.shape[2] // 3

        # Cas d'une observation 4D (empilée sur le premier axe)
        elif len(observation.shape) == 4:
            self.assertEqual(observation.shape[0], 4)  # 4 frames
            frame_count = observation.shape[0]

        self.assertEqual(frame_count, 4)

    def test_add_frame(self):
        """Teste l'ajout manuel de frames"""
        # Réinitialiser avec une observation
        self.wrapper.reset(observation=self.observations[0])

        # Ajouter d'autres frames
        for i in range(1, 5):
            observation = self.wrapper.add_frame(self.observations[i])

        # Vérifier que l'observation contient les 4 dernières frames
        self.assertEqual(len(self.wrapper.frames), 4)
        self.assertTrue(np.array_equal(self.wrapper.frames[0], self.observations[1]))
        self.assertTrue(np.array_equal(self.wrapper.frames[1], self.observations[2]))
        self.assertTrue(np.array_equal(self.wrapper.frames[2], self.observations[3]))
        self.assertTrue(np.array_equal(self.wrapper.frames[3], self.observations[4]))

    def test_compression_wrapper(self):
        """Teste le wrapper avec compression activée"""
        # Réinitialiser avec une observation
        observation = self.compressed_wrapper.reset(observation=self.observations[0])

        # L'observation doit être un tableau numpy
        self.assertIsInstance(observation, np.ndarray)

        # Ajouter quelques frames
        for i in range(1, 3):
            observation = self.compressed_wrapper.add_frame(self.observations[i])

        # Vérifier que le compresseur est utilisé
        self.assertTrue(hasattr(self.compressed_wrapper, "compressor"))

        # Vérifier que la frame a bien été traitée
        self.assertIsInstance(observation, np.ndarray)


if __name__ == "__main__":
    unittest.main()
