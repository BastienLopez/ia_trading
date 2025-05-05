import sys
import unittest
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

try:
    import cv2
    from ai_trading.rl.frame_compression import FrameCompressor, FrameStackWrapper
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # Créer des mocks pour les classes
    FrameCompressor = MagicMock()
    FrameStackWrapper = MagicMock()

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV (cv2) library not available")
class TestFrameCompressor(unittest.TestCase):
    """Tests pour le compresseur de frames basé sur CV2."""

    def setUp(self):
        """Configuration pour les tests."""
        # Skip si OpenCV n'est pas disponible
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
        
        # Créer un compresseur avec les paramètres corrects selon l'implémentation réelle
        self.compressor = FrameCompressor(
            compression_level=5,
            resize_dim=(100, 100),
            use_grayscale=True
        )
        
        # Créer une frame de test 200x200 RGB
        self.test_frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Ajouter un motif pour éviter une frame totalement uniforme
        self.test_frame[50:150, 50:150] = 200
        
    def test_compression_params(self):
        """Teste que les paramètres de compression sont correctement définis."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
            
        self.assertEqual(self.compressor.compression_level, 5)
        self.assertEqual(self.compressor.resize_dim, (100, 100))
        self.assertTrue(self.compressor.use_grayscale)
        
    def test_compress_decompress(self):
        """Teste que la compression et décompression fonctionne."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
            
        # Préprocesser la frame avec les méthodes disponibles
        processed = self.compressor.preprocess_frame(self.test_frame)
        
        # Compresser
        compressed = self.compressor.compress_frame(processed)
        self.assertIsInstance(compressed, bytes)
        
        # Décompresser
        decompressed = self.compressor.decompress_frame(compressed, processed.shape, processed.dtype)
        
        # Vérifier que la décompression fonctionne
        self.assertEqual(decompressed.shape, processed.shape)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV (cv2) library not available")
class TestFrameStackWrapper(unittest.TestCase):
    """Tests pour le wrapper de stacking de frames."""
    
    def setUp(self):
        """Configuration pour les tests."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
            
        # Créer un mock pour l'environnement
        self.mock_env = MagicMock()
        self.mock_env.reset.return_value = np.ones((84, 84, 3), dtype=np.uint8)
        self.mock_env.observation_space = MagicMock()
        
        # Adapter le mock pour qu'il renvoie 4 valeurs comme attendu par le wrapper
        # au lieu des 5 valeurs habituelles de Gymnasium
        self.mock_env.step.return_value = (np.ones((84, 84, 3), dtype=np.uint8), 0, False, {})
        
        self.wrapper = FrameStackWrapper(
            env=self.mock_env,
            n_frames=4,
            compress=True,
            compression_level=5
        )
        
        # Créer une séquence de frames de test
        self.frames = [np.ones((84, 84, 3), dtype=np.uint8) * i for i in range(10)]
        
    def test_init_params(self):
        """Teste que les paramètres sont correctement initialisés."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
            
        self.assertEqual(self.wrapper.n_frames, 4)
        self.assertTrue(self.wrapper.compress)
        self.assertEqual(self.wrapper.compression_level, 5)
        
    def test_add_frames(self):
        """Teste l'ajout de frames au buffer."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
            
        # Réinitialiser pour avoir une frame
        self.wrapper.reset()
        
        # Simuler des steps pour ajouter des frames
        for i in range(3):
            # Adapter pour renvoyer 4 valeurs (observation, reward, done, info)
            self.mock_env.step.return_value = (self.frames[i], 0, False, {})
            result = self.wrapper.step(1)
            
            # Vérifier que le résultat existe
            self.assertIsNotNone(result)
        
    def test_get_stacked_frames(self):
        """Teste la récupération d'un stack de frames."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
            
        # Réinitialiser l'environnement
        self.wrapper.reset()
        
        # Simuler plusieurs steps
        for i in range(6):
            # Adapter pour renvoyer 4 valeurs (observation, reward, done, info)
            self.mock_env.step.return_value = (self.frames[i], 0, False, {})
            observation, reward, done, info = self.wrapper.step(1)
            
        # Vérifier que l'observation n'est pas None
        self.assertIsNotNone(observation)
        
    def test_reset(self):
        """Teste le reset du wrapper."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV (cv2) library not available")
            
        # Reset devrait fonctionner et renvoyer une observation
        observation = self.wrapper.reset()
        
        # Vérifier que l'observation n'est pas None
        self.assertIsNotNone(observation)


if __name__ == "__main__":
    unittest.main()
