#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le module PredictionExplainer.

Ce module teste les fonctionnalités d'interprétabilité et d'explication
des prédictions de marché.
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

from ai_trading.llm.predictions.prediction_explainer import PredictionExplainer
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.prediction_model import PredictionModel
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor

class TestPredictionExplainer(unittest.TestCase):
    """Tests pour la classe PredictionExplainer."""
    
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
            "factors": ["Price momentum", "Volume increase", "Positive sentiment"],
            "contradictions": ["Resistance level"],
            "volatility": "medium",
            "explanation": "The asset shows signs of upward momentum"
        }
        
        # Créer des mocks pour les modèles de prédiction
        self.mock_market_predictor = Mock(spec=MarketPredictor)
        self.mock_prediction_model = Mock(spec=PredictionModel)
        self.mock_multi_horizon = Mock(spec=MultiHorizonPredictor)
        
        # Configurer les mocks pour les modèles ML
        self.mock_prediction_model.ml_model = Mock()
        self.mock_prediction_model.ml_model.predict_proba = Mock(return_value=np.array([[0.2, 0.1, 0.7]]))
        
        # Créer une instance de l'explainer pour les tests
        self.explainer = PredictionExplainer(self.mock_prediction_model, self.market_data)
    
    def test_initialization(self):
        """Teste l'initialisation de l'explainer."""
        self.assertIsNotNone(self.explainer)
        self.assertEqual(self.explainer.model_type, "hybrid")
        
        # Tester avec différents types de modèles
        explainer_llm = PredictionExplainer(self.mock_market_predictor)
        self.assertEqual(explainer_llm.model_type, "llm")
        
        explainer_mh = PredictionExplainer(self.mock_multi_horizon)
        self.assertEqual(explainer_mh.model_type, "multi_horizon")
    
    def test_explain_llm_prediction(self):
        """Teste l'explication des prédictions LLM."""
        explanation = self.explainer._explain_llm_prediction(self.prediction)
        
        # Vérifier les clés attendues
        self.assertIn("factors", explanation)
        self.assertIn("contradictions", explanation)
        self.assertIn("factor_confidence", explanation)
        self.assertIn("prediction_id", explanation)
        self.assertIn("asset", explanation)
        self.assertIn("direction", explanation)
        
        # Vérifier les valeurs
        self.assertEqual(explanation["asset"], "BTC")
        self.assertEqual(explanation["direction"], "bullish")
        self.assertEqual(len(explanation["factors"]), 3)
        self.assertEqual(len(explanation["factor_confidence"]), 3)
    
    @patch('shap.Explainer')
    def test_explain_with_shap(self, mock_shap_explainer):
        """Teste l'explication SHAP."""
        # Configurer le mock SHAP
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.rand(10, 9)
        mock_shap_values.base_values = np.random.rand(10)
        
        mock_explainer_instance = mock_shap_explainer.return_value
        mock_explainer_instance.return_value = mock_shap_values
        
        # Appeler la méthode à tester
        explanation = self.explainer.explain_with_shap(self.prediction, self.market_data)
        
        # Vérifications
        self.assertIn("shap_values", explanation)
        self.assertIn("base_value", explanation)
        self.assertIn("feature_names", explanation)
        self.assertEqual(explanation["asset"], "BTC")
        self.assertEqual(explanation["direction"], "bullish")
        
        # Vérifier que SHAP a été appelé correctement
        mock_shap_explainer.assert_called_once()
    
    @patch('lime.lime_tabular.LimeTabularExplainer')
    def test_explain_with_lime(self, mock_lime_explainer):
        """Teste l'explication LIME."""
        # Configurer le mock LIME
        mock_explanation = MagicMock()
        mock_explanation.as_list.return_value = [
            ('rsi', 0.2),
            ('macd', 0.15),
            ('volume', -0.1)
        ]
        
        mock_explainer_instance = mock_lime_explainer.return_value
        mock_explainer_instance.explain_instance.return_value = mock_explanation
        
        # Appeler la méthode à tester
        explanation = self.explainer.explain_with_lime(self.prediction, self.market_data)
        
        # Vérifications
        self.assertIn("features", explanation)
        self.assertIn("scores", explanation)
        self.assertEqual(len(explanation["features"]), 3)
        self.assertEqual(len(explanation["scores"]), 3)
        self.assertEqual(explanation["asset"], "BTC")
        
        # Vérifier que LIME a été appelé correctement
        mock_lime_explainer.assert_called_once()
    
    def test_generate_report(self):
        """Teste la génération de rapport."""
        # Créer des mocks pour les méthodes d'explication
        self.explainer.explain_with_shap = Mock(return_value={"shap_values": np.random.rand(10, 9)})
        self.explainer.explain_with_lime = Mock(return_value={"features": ["rsi", "macd"], "scores": [0.2, 0.15]})
        self.explainer._explain_llm_prediction = Mock(return_value={"factors": ["Price momentum"]})
        
        # Appeler la méthode à tester
        report = self.explainer.generate_report(self.prediction, ["shap", "lime", "llm"])
        
        # Vérifications
        self.assertIn("prediction", report)
        self.assertIn("explanations", report)
        self.assertIn("shap", report["explanations"])
        self.assertIn("lime", report["explanations"])
        self.assertIn("llm", report["explanations"])
        self.assertIn("comparative_analysis", report)
        
        # Vérifier que les méthodes ont été appelées
        self.explainer.explain_with_shap.assert_called_once()
        self.explainer.explain_with_lime.assert_called_once()
        self.explainer._explain_llm_prediction.assert_called_once()
    
    def test_plotting_methods(self):
        """Teste les méthodes de visualisation."""
        # Configurer les données d'explication factices
        self.explainer.explanations = {
            "shap": {
                "shap_values": np.random.rand(10, 9),
                "feature_names": self.market_data.columns.tolist(),
                "asset": "BTC"
            },
            "lime": {
                "features": ["rsi", "macd", "volume"],
                "scores": [0.2, 0.15, -0.1],
                "asset": "BTC"
            }
        }
        
        # Remplacer les méthodes de visualization par des mocks
        original_plot_shap = self.explainer.plot_shap_summary
        original_plot_lime = self.explainer.plot_lime_explanation
        original_plot_factor = self.explainer.plot_factor_importance
        
        plot_methods_called = []
        
        def mock_plot_shap(save_path=None):
            plot_methods_called.append('shap')
            # Simuler la création d'un fichier
            if save_path:
                with open(save_path, 'w') as f:
                    f.write('mock file')
            return None
            
        def mock_plot_lime(save_path=None):
            plot_methods_called.append('lime')
            # Simuler la création d'un fichier
            if save_path:
                with open(save_path, 'w') as f:
                    f.write('mock file')
            return None
            
        def mock_plot_factor(prediction, save_path=None):
            plot_methods_called.append('factor')
            # Simuler la création d'un fichier
            if save_path:
                with open(save_path, 'w') as f:
                    f.write('mock file')
            return None
        
        try:
            # Remplacer les méthodes originales par nos mocks
            self.explainer.plot_shap_summary = mock_plot_shap
            self.explainer.plot_lime_explanation = mock_plot_lime
            self.explainer.plot_factor_importance = mock_plot_factor
            
            # Tester avec un chemin temporaire pour sauvegarder
            with tempfile.TemporaryDirectory() as temp_dir:
                shap_path = os.path.join(temp_dir, "shap_summary.png")
                lime_path = os.path.join(temp_dir, "lime_explanation.png")
                factor_path = os.path.join(temp_dir, "factor_importance.png")
                
                # Tester les méthodes de visualisation
                self.explainer.plot_shap_summary(save_path=shap_path)
                self.explainer.plot_lime_explanation(save_path=lime_path)
                self.explainer.plot_factor_importance(self.prediction, save_path=factor_path)
                
                # Vérifier que chaque fichier a été créé
                self.assertTrue(os.path.exists(shap_path))
                self.assertTrue(os.path.exists(lime_path))
                self.assertTrue(os.path.exists(factor_path))
                
                # Vérifier que toutes les méthodes ont été appelées
                self.assertEqual(set(plot_methods_called), {'shap', 'lime', 'factor'})
        finally:
            # Restaurer les méthodes originales
            self.explainer.plot_shap_summary = original_plot_shap
            self.explainer.plot_lime_explanation = original_plot_lime
            self.explainer.plot_factor_importance = original_plot_factor

if __name__ == '__main__':
    unittest.main() 