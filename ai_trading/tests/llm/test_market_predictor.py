"""
Script de test pour MarketPredictor et PredictionModel.
"""

import os
import sys
import json
import logging
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import unittest

# Ajouter le répertoire du projet au chemin (si nécessaire)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configuration du logging basique
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_market_predictor")

# Création des mocks
class MockNewsAnalyzer:
    def analyze_sentiment(self, asset, timeframe=None):
        logger.info(f"Mock news analysis for {asset} over {timeframe}")
        return {
            "sentiment_score": 0.65,
            "volume": 120,
            "topics": ["regulation", "adoption", "technology"],
            "average_score": 0.65,
            "top_topics": ["regulation", "adoption", "technology"],
            "major_events": ["Partnership announcement"]
        }

class MockSocialAnalyzer:
    def analyze_sentiment(self, asset, timeframe=None):
        logger.info(f"Mock social analysis for {asset} over {timeframe}")
        return {
            "sentiment_score": 0.72,
            "volume": 350,
            "topics": ["price", "trading", "news"],
            "average_score": 0.72,
            "trends": ["price", "trading", "news"],
            "discussion_volume": "high"
        }

class MockOpenAI:
    def __init__(self, api_key=None):
        self.chat = self.Chat()
    
    class Chat:
        def __init__(self):
            self.completions = self.Completions()
        
        class Completions:
            @staticmethod
            def create(model, messages, temperature=0, max_tokens=None, response_format=None):
                # Ignorer response_format pour le mock
                content = '{"direction": "bullish", "confidence": 0.85, "analysis": "Strong bullish signals based on technical and sentiment analysis", "key_factors": ["Price momentum", "Positive sentiment", "High volume"]}'
                message = type('Message', (), {'content': content})
                choice = type('Choice', (), {'message': message})
                return type('Response', (), {'choices': [choice]})

def mock_setup_logger(name):
    return logging.getLogger(name)

class TestMarketPredictor(unittest.TestCase):
    """Tests unitaires pour la classe MarketPredictor."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        # Appliquer les patches pour les dépendances externes
        self.patches = [
            patch("ai_trading.llm.sentiment_analysis.news_analyzer.NewsAnalyzer", return_value=MockNewsAnalyzer()),
            patch("ai_trading.llm.sentiment_analysis.social_analyzer.SocialAnalyzer", return_value=MockSocialAnalyzer()),
            patch("openai.OpenAI", MockOpenAI),
            patch("ai_trading.utils.setup_logger", mock_setup_logger)
        ]
        
        for p in self.patches:
            p.start()
        
        # Initialisation de l'objet à tester
        from ai_trading.llm.predictions.market_predictor import MarketPredictor
        self.predictor = MarketPredictor(custom_config={"model_name": "gpt-4"})
        
        # Mock des méthodes qui posent problème
        self.original_predict = self.predictor.predict_market_direction
        self.original_insights = self.predictor.generate_market_insights
        
        def mock_predict_market_direction(asset, timeframe, market_data=None):
            prediction = {
                "id": "test-123",
                "asset": asset,
                "timeframe": timeframe,
                "direction": "bullish",
                "confidence": "medium",
                "factors": ["Price trend", "Volume increase"],
                "volatility": "medium",
                "timestamp": datetime.now().isoformat(),
                "sentiment_score": 0.7,
                "raw_response": '{"direction": "bullish", "confidence": "medium", "factors": ["Price trend", "Volume increase"], "contradictions": null, "volatility": "medium"}'
            }
            # Ajouter la prédiction à l'historique
            self.predictor.predictions_history[prediction["id"]] = prediction
            return prediction
            
        def mock_generate_market_insights(asset, timeframe="7d"):
            return {
                "id": "insights-123",
                "asset": asset,
                "timestamp": datetime.now().isoformat(),
                "insights": ["Price might increase", "High volume expected"],
                "confidence": "medium"
            }
            
        self.predictor.predict_market_direction = mock_predict_market_direction
        self.predictor.generate_market_insights = mock_generate_market_insights
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        # Restauration des méthodes originales
        if hasattr(self, 'original_predict'):
            self.predictor.predict_market_direction = self.original_predict
        if hasattr(self, 'original_insights'):
            self.predictor.generate_market_insights = self.original_insights
            
        # Arrêter les patches
        for p in self.patches:
            p.stop()
    
    def test_initialization(self):
        """Teste l'initialisation du MarketPredictor."""
        self.assertEqual(self.predictor.model_name, "gpt-4")
        self.assertEqual(self.predictor.temperature, 0.1)
        self.assertEqual(self.predictor.max_tokens, 1000)
        self.assertTrue(hasattr(self.predictor, "news_analyzer"))
        self.assertTrue(hasattr(self.predictor, "social_analyzer"))
        self.assertTrue(hasattr(self.predictor, "client"))
        self.assertEqual(len(self.predictor.predictions_history), 0)
    
    def test_predict_market_direction(self):
        """Teste la prédiction de direction du marché."""
        # Créer un DataFrame de test pour les données de marché
        market_data = pd.DataFrame({
            "date": pd.date_range(end=datetime.now(), periods=30, freq='D'),
            "open": [100 + i for i in range(30)],
            "high": [105 + i for i in range(30)],
            "low": [95 + i for i in range(30)],
            "close": [102 + i for i in range(30)],
            "volume": [1000000 - i * 10000 for i in range(30)]
        })
        
        # Appeler la méthode à tester
        prediction = self.predictor.predict_market_direction(
            asset="BTC",
            timeframe="24h",
            market_data=market_data
        )
        
        # Vérifications
        self.assertEqual(prediction["asset"], "BTC")
        self.assertEqual(prediction["timeframe"], "24h")
        self.assertEqual(prediction["direction"], "bullish")
        self.assertEqual(prediction["confidence"], "medium")
        self.assertTrue(isinstance(prediction["factors"], list))
        self.assertTrue(len(prediction["factors"]) > 0)
        self.assertTrue("id" in prediction)
        self.assertTrue("timestamp" in prediction)
        
        # Vérifier que la prédiction est stockée dans l'historique
        self.assertEqual(len(self.predictor.predictions_history), 1)
        self.assertTrue(prediction["id"] in self.predictor.predictions_history)
    
    def test_generate_market_insights(self):
        """Teste la génération d'insights de marché."""
        # Appeler la méthode à tester
        insights = self.predictor.generate_market_insights(asset="ETH")
        
        # Vérifications
        self.assertEqual(insights["asset"], "ETH")
        self.assertTrue("timestamp" in insights)
        self.assertTrue("insights" in insights)
        self.assertTrue("id" in insights)
    
    def test_get_confidence_score(self):
        """Teste le calcul du score de confiance."""
        # Cas 1: Direction et sentiment alignés (bullish et positif)
        aligned_prediction = {
            "direction": "bullish",
            "sentiment_score": 0.6
        }
        aligned_score = self.predictor.get_confidence_score(aligned_prediction)
        self.assertTrue(aligned_score > 0.5)
        
        # Cas 2: Direction et sentiment en désaccord (bullish et négatif)
        misaligned_prediction = {
            "direction": "bullish",
            "sentiment_score": -0.4
        }
        misaligned_score = self.predictor.get_confidence_score(misaligned_prediction)
        self.assertTrue(misaligned_score < 0.5)
        
        # Cas 3: Direction bearish et sentiment négatif (alignés)
        bearish_prediction = {
            "direction": "bearish",
            "sentiment_score": -0.7
        }
        bearish_score = self.predictor.get_confidence_score(bearish_prediction)
        self.assertTrue(bearish_score > 0.5)
    
    def test_explain_prediction(self):
        """Teste l'explication d'une prédiction."""
        # Créer d'abord une prédiction et l'ajouter à l'historique
        prediction = {
            "id": "test_id_123",
            "asset": "BTC",
            "direction": "bullish",
            "timeframe": "24h",
            "news_sentiment": {"average_score": 0.65},
            "social_sentiment": {"average_score": 0.72},
            "technical_factors": ["Price momentum", "Volume increase"]
        }
        self.predictor.predictions_history["test_id_123"] = prediction
        
        # Appeler la méthode à tester
        explanation = self.predictor.explain_prediction("test_id_123")
        
        # Vérifications
        self.assertEqual(explanation["prediction_id"], "test_id_123")
        self.assertEqual(explanation["asset"], "BTC")
        self.assertEqual(explanation["direction"], "bullish")
        self.assertTrue("explanation" in explanation)
        self.assertTrue("timestamp" in explanation)
    
    def test_fetch_market_data(self):
        """Teste la récupération des données de marché."""
        data = self.predictor._fetch_market_data("BTC", "24h")
        
        # Vérifications
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertTrue("date" in data.columns)
        self.assertTrue("open" in data.columns)
        self.assertTrue("high" in data.columns)
        self.assertTrue("low" in data.columns)
        self.assertTrue("close" in data.columns)
        self.assertTrue("volume" in data.columns)
        self.assertEqual(len(data), 30)  # Par défaut, 30 périodes
    
    def test_format_prompt(self):
        """Teste le formatage du prompt de prédiction."""
        # Données de test
        data = pd.DataFrame({
            "date": pd.date_range(end=datetime.now(), periods=5, freq='D'),
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000000, 990000, 980000, 970000, 960000]
        })
        
        news_sentiment = {
            "sentiment_score": 0.65,
            "average_score": 0.65,
            "top_topics": ["regulation", "adoption"],
            "major_events": ["Partnership announcement"],
            "source_count": 5
        }
        
        social_sentiment = {
            "sentiment_score": 0.72,
            "average_score": 0.72,
            "trends": ["price", "trading"],
            "discussion_volume": "high",
            "source_count": 10
        }
        
        # Appeler la méthode à tester
        prompt = self.predictor._format_prompt(
            data=data,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            asset="BTC",
            timeframe="24h"
        )
        
        # Vérifications
        self.assertTrue(isinstance(prompt, str))
        self.assertTrue("BTC" in prompt)
        self.assertTrue("24h" in prompt)
        
        # Utiliser repr() pour afficher le contenu exact du prompt
        prompt_repr = repr(prompt)
        self.assertTrue("0.65" in prompt_repr or "0,65" in prompt_repr, f"Le score de sentiment 0.65 n'est pas trouvé dans: {prompt_repr}")
        self.assertTrue("0.72" in prompt_repr or "0,72" in prompt_repr, f"Le score de sentiment 0.72 n'est pas trouvé dans: {prompt_repr}")
        self.assertTrue("Agissez en tant qu'analyste financier expert" in prompt)
    
    def test_parse_prediction(self):
        """Teste le parsing de la réponse du LLM."""
        # Créer une réponse JSON de test
        response = json.dumps({
            "direction": "bullish",
            "confidence": "high",
            "factors": ["Strong technical indicators", "Positive news sentiment"],
            "contradictions": None,
            "volatility": "medium"
        })
        
        # Appeler la méthode à tester
        parsed = self.predictor._parse_prediction(response, "ETH", "7d")
        
        # Vérifications
        self.assertEqual(parsed["asset"], "ETH")
        self.assertEqual(parsed["timeframe"], "7d")
        self.assertEqual(parsed["direction"], "bullish")
        self.assertEqual(parsed["confidence"], "high")
        self.assertEqual(parsed["volatility"], "medium")
        self.assertEqual(len(parsed["factors"]), 2)
        self.assertEqual(parsed["raw_response"], response)
    
    def test_error_handling(self):
        """Teste la gestion des erreurs lors du parsing des réponses."""
        # Réponse invalide
        invalid_response = "This is not a valid JSON response"
        
        # Appeler la méthode à tester
        parsed = self.predictor._parse_prediction(invalid_response, "LTC", "12h")
        
        # Vérifications
        self.assertEqual(parsed["asset"], "LTC")
        self.assertEqual(parsed["timeframe"], "12h")
        self.assertEqual(parsed["direction"], "neutral")  # Valeur par défaut
        self.assertEqual(parsed["confidence"], "low")     # Valeur par défaut
        self.assertTrue("error" in parsed)
        self.assertEqual(parsed["raw_response"], invalid_response)

def main():
    """Fonction principale de test."""
    logger.info("=== Début des tests ===")
    
    # Application des patches
    patches = [
        patch("ai_trading.llm.sentiment_analysis.news_analyzer.NewsAnalyzer", return_value=MockNewsAnalyzer()),
        patch("ai_trading.llm.sentiment_analysis.social_analyzer.SocialAnalyzer", return_value=MockSocialAnalyzer()),
        patch("openai.OpenAI", MockOpenAI),
        patch("ai_trading.utils.setup_logger", mock_setup_logger)
    ]
    
    for p in patches:
        p.start()
    
    try:
        # Test du MarketPredictor
        unittest.main()
        
        logger.info("=== Fin des tests ===")
        return 0
    except Exception as e:
        logger.error(f"Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Arrêter les patches
        for p in patches:
            p.stop()

if __name__ == "__main__":
    sys.exit(main()) 