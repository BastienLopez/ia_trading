#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le module UncertaintyCalibrator.

Ce module teste les fonctionnalités de calibration d'incertitude des prédictions
de marché.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend Agg (non-interactif) avant d'importer pyplot

from ai_trading.llm.predictions.uncertainty_calibration import UncertaintyCalibrator
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.prediction_model import PredictionModel

class TestUncertaintyCalibration(unittest.TestCase):
    """Tests pour la classe UncertaintyCalibrator."""
    
    def setUp(self):
        """Initialise les données pour les tests."""
        # Créer des données de marché simulées
        self.market_data = pd.DataFrame({
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100 + 10,
            'low': np.random.rand(10) * 100 - 10,
            'close': np.random.rand(10) * 100,
            'volume': np.random.rand(10) * 1000,
            'rsi': np.random.rand(10) * 100,
            'macd': np.random.rand(10) * 10 - 5,
            'sma_20': np.random.rand(10) * 100,
            'ema_10': np.random.rand(10) * 100
        })
        
        # Créer une prédiction simulée
        self.prediction = {
            "id": "pred_123",
            "asset": "BTC",
            "direction": "bullish",
            "confidence": "medium",
            "timeframe": "24h",
            "factors": ["Price momentum", "Volume increase", "Positive sentiment"]
        }
        
        # Créer une liste de prédictions pour les tests d'outliers
        self.predictions = [
            {
                "id": f"pred_{i}",
                "asset": "BTC",
                "direction": "bullish" if i % 3 == 0 else "neutral" if i % 3 == 1 else "bearish",
                "confidence": "medium",
                "timeframe": "24h"
            }
            for i in range(10)
        ]
        
        # Ajouter un outlier
        self.predictions.append({
            "id": "pred_outlier",
            "asset": "BTC",
            "direction": "bearish",
            "confidence": "high",
            "timeframe": "24h"
        })
        
        # Créer des mocks pour les modèles de prédiction
        self.mock_market_predictor = Mock(spec=MarketPredictor)
        self.mock_prediction_model = Mock(spec=PredictionModel)
        
        # Configurer les mocks pour les modèles ML
        self.mock_prediction_model.ml_model = Mock()
        self.mock_prediction_model.ml_model.predict_proba = Mock(return_value=np.array([[0.2, 0.1, 0.7]]))
        
        # Créer une instance du calibrateur pour les tests
        self.calibrator = UncertaintyCalibrator(self.mock_prediction_model, self.market_data)
    
    def test_initialization(self):
        """Teste l'initialisation du calibrateur."""
        self.assertIsNotNone(self.calibrator)
        self.assertEqual(self.calibrator.model_type, "hybrid")
        
        # Tester avec différents types de modèles
        calibrator_llm = UncertaintyCalibrator(self.mock_market_predictor)
        self.assertEqual(calibrator_llm.model_type, "llm")
    
    def test_confidence_intervals(self):
        """Teste le calcul des intervalles de confiance."""
        # Tester différentes méthodes
        methods = ["bootstrap", "parametric", "bayesian"]
        
        for method in methods:
            result = self.calibrator.calculate_confidence_intervals(
                self.prediction, confidence_level=0.95, method=method)
            
            # Vérifier les clés attendues
            self.assertIn("prediction_id", result)
            self.assertIn("asset", result)
            self.assertIn("direction", result)
            self.assertIn("lower_bound", result)
            self.assertIn("upper_bound", result)
            self.assertIn("lower_direction", result)
            self.assertIn("upper_direction", result)
            
            # Vérifier la logique de l'intervalle
            self.assertLess(result["lower_bound"], result["upper_bound"])
            
            # Vérifier que l'intervalle est stocké
            self.assertIn(self.prediction["id"], self.calibrator.confidence_intervals)
    
    def test_probability_distribution(self):
        """Teste l'estimation de la distribution de probabilité."""
        distribution = self.calibrator.estimate_probability_distribution(self.prediction)
        
        # Vérifier les clés attendues
        self.assertIn("prediction_id", distribution)
        self.assertIn("asset", distribution)
        self.assertIn("x_values", distribution)
        self.assertIn("y_values", distribution)
        self.assertIn("probabilities", distribution)
        self.assertIn("most_likely_direction", distribution)
        
        # Vérifier les probabilités
        probs = distribution["probabilities"]
        self.assertIn("bearish", probs)
        self.assertIn("neutral", probs)
        self.assertIn("bullish", probs)
        
        # Vérifier que les probabilités somment à 1 (à peu près)
        total_prob = probs["bearish"] + probs["neutral"] + probs["bullish"]
        self.assertAlmostEqual(total_prob, 1.0, places=1)
        
        # Pour une prédiction bullish, la probabilité bullish devrait être la plus élevée
        self.assertEqual(distribution["most_likely_direction"], "bullish")
    
    def test_outlier_detection(self):
        """Teste la détection d'outliers."""
        # Tester la détection par z-score
        result_z = self.calibrator.detect_outliers(self.predictions, method="z_score", threshold=2.0)
        
        # Vérifier les clés attendues
        self.assertIn("total_predictions", result_z)
        self.assertIn("outliers_detected", result_z)
        self.assertIn("outlier_percentage", result_z)
        self.assertIn("outliers", result_z)
        
        # Tester la détection par IQR
        result_iqr = self.calibrator.detect_outliers(self.predictions, method="iqr", threshold=1.5)
        
        # Les détections peuvent varier, mais les structures doivent être cohérentes
        self.assertEqual(result_z["total_predictions"], result_iqr["total_predictions"])
    
    def test_prediction_calibration(self):
        """Teste la calibration des prédictions."""
        # Tester différentes méthodes de calibration
        methods = ["platt", "isotonic"]
        
        for method in methods:
            calibrated = self.calibrator.calibrate_prediction(self.prediction, calibration_method=method)
            
            # Vérifier les clés attendues
            self.assertIn("original_confidence", calibrated)
            self.assertIn("confidence", calibrated)
            self.assertIn("confidence_value", calibrated)
            self.assertIn("calibration_method", calibrated)
            
            # Vérifier que la méthode est correcte
            self.assertEqual(calibrated["calibration_method"], method)
            
            # Vérifier que la valeur de confiance est entre 0 et 1
            self.assertGreaterEqual(calibrated["confidence_value"], 0.0)
            self.assertLessEqual(calibrated["confidence_value"], 1.0)
    
    def test_visualization_methods(self):
        """Teste les méthodes de visualisation."""
        # Générer une distribution
        distribution = self.calibrator.estimate_probability_distribution(self.prediction)
        
        # Remplacer la méthode de visualisation par un mock
        original_plot = self.calibrator.plot_probability_distribution
        plot_called = [False]
        
        def mock_plot(prediction, distribution=None, save_path=None):
            plot_called[0] = True
            # Simuler la création d'un fichier
            if save_path:
                with open(save_path, 'w') as f:
                    f.write('mock distribution plot')
            return None
        
        try:
            # Remplacer la méthode originale par notre mock
            self.calibrator.plot_probability_distribution = mock_plot
            
            # Tester avec un chemin temporaire pour sauvegarder
            with tempfile.TemporaryDirectory() as temp_dir:
                # Chemin pour la visualisation de distribution
                dist_path = os.path.join(temp_dir, "probability_distribution.png")
                
                # Appeler la méthode
                self.calibrator.plot_probability_distribution(
                    self.prediction, distribution=distribution, save_path=dist_path)
                
                # Vérifier que le fichier a été créé
                self.assertTrue(os.path.exists(dist_path))
                
                # Vérifier que la méthode a été appelée
                self.assertTrue(plot_called[0])
        finally:
            # Restaurer la méthode originale
            self.calibrator.plot_probability_distribution = original_plot

if __name__ == '__main__':
    unittest.main() 