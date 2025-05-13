"""
Tests unitaires pour le module de détection de patterns chartistes basé sur CNN.
"""
import unittest
import unittest.mock
import numpy as np
import pandas as pd
import os
import tempfile
from ai_trading.indicators.patterns.cnn_detector import (
    ChartPatternType,
    PatternInstance,
    CNNPatternDetector
)


class TestPatternInstance(unittest.TestCase):
    """Tests pour la classe PatternInstance."""
    
    def test_initialization(self):
        """Teste l'initialisation d'une instance de pattern."""
        pattern = PatternInstance(
            pattern_type=ChartPatternType.DOUBLE_TOP,
            start_idx=10,
            end_idx=50,
            confidence=0.85,
            breakout_target=105.0,
            stop_loss=95.0
        )
        
        # Vérifier les attributs
        self.assertEqual(pattern.pattern_type, ChartPatternType.DOUBLE_TOP)
        self.assertEqual(pattern.start_idx, 10)
        self.assertEqual(pattern.end_idx, 50)
        self.assertEqual(pattern.confidence, 0.85)
        self.assertEqual(pattern.breakout_target, 105.0)
        self.assertEqual(pattern.stop_loss, 95.0)
        
    def test_duration(self):
        """Teste la propriété duration."""
        pattern = PatternInstance(
            pattern_type=ChartPatternType.HEAD_AND_SHOULDERS,
            start_idx=100,
            end_idx=150,
            confidence=0.9
        )
        
        # La durée devrait être (end_idx - start_idx + 1)
        self.assertEqual(pattern.duration, 51)
        
    def test_to_dict(self):
        """Teste la méthode to_dict."""
        pattern = PatternInstance(
            pattern_type=ChartPatternType.ASCENDING_TRIANGLE,
            start_idx=200,
            end_idx=300,
            confidence=0.75,
            breakout_target=150.0,
            stop_loss=140.0
        )
        
        # Convertir en dictionnaire
        pattern_dict = pattern.to_dict()
        
        # Vérifier les clés et les valeurs
        self.assertEqual(pattern_dict["pattern_type"], "triangle_ascendant")
        self.assertEqual(pattern_dict["start_idx"], 200)
        self.assertEqual(pattern_dict["end_idx"], 300)
        self.assertEqual(pattern_dict["confidence"], 0.75)
        self.assertEqual(pattern_dict["duration"], 101)
        self.assertEqual(pattern_dict["breakout_target"], 150.0)
        self.assertEqual(pattern_dict["stop_loss"], 140.0)


class TestCNNPatternDetector(unittest.TestCase):
    """Tests pour la classe CNNPatternDetector."""
    
    def setUp(self):
        """Initialise les données et le détecteur pour les tests."""
        # Créer un détecteur avec une petite fenêtre pour les tests
        self.window_size = 20
        self.detector = CNNPatternDetector(window_size=self.window_size, confidence_threshold=0.7)
        
        # Créer des données OHLCV synthétiques pour les tests
        np.random.seed(42)
        n_points = 100
        
        # Série de prix avec un double sommet évident
        t = np.linspace(0, 4 * np.pi, n_points)
        close = 100 + 10 * np.sin(t) + np.random.normal(0, 1, n_points)
        open_price = close - np.random.normal(0, 0.5, n_points)
        high = np.maximum(open_price, close) + np.random.uniform(0, 1, n_points)
        low = np.minimum(open_price, close) - np.random.uniform(0, 1, n_points)
        volume = 1000 + 500 * np.random.random(n_points)
        
        # Créer le DataFrame
        self.data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
    def test_build_model(self):
        """Teste la construction du modèle CNN."""
        # Construire le modèle avec une fenêtre plus grande pour éviter les erreurs de dimensionnement
        model = self.detector.build_model(input_shape=(32, 5))
        
        # Vérifier que le modèle a été créé
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.detector.model)
        
        # Vérifier que le modèle a la bonne architecture pour l'entrée
        self.assertEqual(model.input_shape, (None, 32, 5))
        self.assertEqual(model.output_shape, (None, len(ChartPatternType) + 1))
        
    def test_preprocess_data(self):
        """Teste le prétraitement des données."""
        # Prétraiter les données
        features = self.detector.preprocess_data(self.data)
        
        # Vérifier les dimensions
        self.assertEqual(features.shape, (len(self.data), 5))
        
        # Vérifier que les prix ont été normalisés correctement
        # Les prix devraient être proches de 1.0 dans la première fenêtre
        first_window = features[:self.window_size, 0:4]
        self.assertAlmostEqual(np.mean(first_window), 1.0, delta=0.5)
        
    def test_create_sliding_windows(self):
        """Teste la création de fenêtres glissantes."""
        # Prétraiter les données
        features = self.detector.preprocess_data(self.data)
        
        # Créer des fenêtres avec un pas de 5
        stride = 5
        windows = self.detector.create_sliding_windows(features, stride)
        
        # Vérifier les dimensions
        expected_n_windows = (len(features) - self.window_size) // stride + 1
        self.assertEqual(windows.shape, (expected_n_windows, self.window_size, 5))
        
        # Vérifier que les fenêtres sont correctement construites
        for i in range(min(3, expected_n_windows)):
            start_idx = i * stride
            end_idx = start_idx + self.window_size
            np.testing.assert_array_equal(windows[i], features[start_idx:end_idx])
            
    def test_merge_overlapping_patterns(self):
        """Teste la fusion des patterns qui se chevauchent."""
        # Créer des patterns qui se chevauchent
        pattern1 = PatternInstance(
            pattern_type=ChartPatternType.DOUBLE_TOP,
            start_idx=10,
            end_idx=30,
            confidence=0.8
        )
        
        pattern2 = PatternInstance(
            pattern_type=ChartPatternType.DOUBLE_TOP,
            start_idx=25,
            end_idx=45,
            confidence=0.9
        )
        
        pattern3 = PatternInstance(
            pattern_type=ChartPatternType.HEAD_AND_SHOULDERS,
            start_idx=50,
            end_idx=70,
            confidence=0.75
        )
        
        # Fusionner les patterns
        merged = self.detector._merge_overlapping_patterns([pattern1, pattern2, pattern3])
        
        # Vérifier le résultat
        self.assertEqual(len(merged), 2)  # pattern1 et pattern2 devraient être fusionnés
        
        # Vérifier le pattern fusionné
        merged_pattern = merged[0]
        self.assertEqual(merged_pattern.pattern_type, ChartPatternType.DOUBLE_TOP)
        self.assertEqual(merged_pattern.start_idx, 10)
        self.assertEqual(merged_pattern.end_idx, 45)
        
        # La confiance devrait être une moyenne pondérée
        expected_confidence = ((0.8 * 21) + (0.9 * 21)) / (21 + 21)  # (conf * durée) / durée_totale
        self.assertAlmostEqual(merged_pattern.confidence, expected_confidence, places=6)
        
        # Le second pattern ne devrait pas être fusionné
        self.assertEqual(merged[1].pattern_type, ChartPatternType.HEAD_AND_SHOULDERS)
        
    def test_add_trading_targets(self):
        """Teste l'ajout des cibles de trading."""
        # Créer un pattern
        pattern = PatternInstance(
            pattern_type=ChartPatternType.DOUBLE_TOP,
            start_idx=10,
            end_idx=30,
            confidence=0.8
        )
        
        # Extraire les données pour ce pattern
        pattern_data = self.data.iloc[10:31]
        
        # Ajouter les cibles de trading
        self.detector._add_trading_targets(pattern, pattern_data)
        
        # Vérifier que les cibles ont été ajoutées
        self.assertIsNotNone(pattern.breakout_target)
        self.assertIsNotNone(pattern.stop_loss)
        
        # Pour un double sommet (bearish), la cible devrait être plus basse que le prix actuel
        self.assertLess(pattern.breakout_target, pattern_data['close'].iloc[-1])
        
        # Le stop devrait être plus haut que le prix actuel
        self.assertGreater(pattern.stop_loss, pattern_data['close'].iloc[-1])
        
    def test_train_and_detect(self):
        """Teste l'entraînement et la détection (test d'intégration simplifié)."""
        # Utiliser un mock pour éviter les problèmes avec TensorFlow
        self.detector.model = unittest.mock.MagicMock()
        
        # Configurer le mock pour predict
        def mock_predict(windows):
            n_windows = len(windows)
            predictions = np.zeros((n_windows, len(ChartPatternType) + 1))
            
            # Faire une prédiction fictive pour le premier pattern
            predictions[0, 0] = 0.8  # HEAD_AND_SHOULDERS avec confiance 0.8
            predictions[0, -1] = 0.2  # Classe "pas de pattern" avec confiance 0.2
            
            # Aucun pattern pour les autres fenêtres
            predictions[1:, -1] = 0.9  # Classe "pas de pattern" avec confiance 0.9
            
            return predictions
        
        self.detector.model.predict = mock_predict
        
        # Configurer un mock pour fit qui retourne un historique d'entraînement
        history = unittest.mock.MagicMock()
        history.history = {'loss': [0.5, 0.3], 'accuracy': [0.7, 0.8]}
        self.detector.model.fit = unittest.mock.MagicMock(return_value=history)
        
        # Test de train
        n_samples = 2
        X_train = np.random.randn(n_samples, self.window_size, 5)
        y_train = np.zeros((n_samples, len(ChartPatternType) + 1))
        
        # Tester l'entraînement
        train_history = self.detector.train(X_train, y_train, epochs=1, batch_size=1)
        self.assertIn('loss', train_history.history)
        
        # Tester la détection
        patterns = self.detector.detect_patterns(self.data, stride=5)
        
        # Vérifier qu'on a détecté au moins un pattern
        self.assertTrue(len(patterns) > 0)
        
        # Vérifier le type du premier pattern
        self.assertEqual(patterns[0].pattern_type, ChartPatternType.HEAD_AND_SHOULDERS)


if __name__ == '__main__':
    unittest.main() 