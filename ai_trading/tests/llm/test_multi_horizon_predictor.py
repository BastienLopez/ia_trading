#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

# Configuration du logging basique
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_multi_horizon_predictor")


# Création des mocks
class MockNewsAnalyzer:
    def analyze_sentiment(self, asset, timeframe):
        logger.info(f"Mock news analysis for {asset} over {timeframe}")
        return {
            "sentiment_score": 0.65,
            "volume": 120,
            "topics": ["regulation", "adoption", "technology"],
            "average_score": 0.65,
            "top_topics": ["regulation", "adoption", "technology"],
            "major_events": ["Partnership announcement"],
        }


class MockSocialAnalyzer:
    def analyze_sentiment(self, asset, timeframe):
        logger.info(f"Mock social analysis for {asset} over {timeframe}")
        return {
            "sentiment_score": 0.72,
            "volume": 350,
            "topics": ["price", "trading", "news"],
            "average_score": 0.72,
            "trends": ["price", "trading", "news"],
            "discussion_volume": "high",
        }


class MockOpenAI:
    class ChatCompletion:
        @staticmethod
        def create(model, messages, temperature=0, max_tokens=None):
            content = json.dumps(
                {
                    "direction": "bullish",
                    "confidence": "medium",
                    "factors": ["Price trend", "Volume increase", "Positive sentiment"],
                    "contradictions": None,
                    "volatility": "medium",
                    "explanation": "The asset shows signs of increasing momentum with positive sentiment across news and social media.",
                }
            )
            return type(
                "obj",
                (object,),
                {
                    "choices": [
                        type(
                            "obj",
                            (object,),
                            {"message": type("obj", (object,), {"content": content})},
                        )
                    ]
                },
            )

    def __init__(self, api_key=None):
        pass


def mock_setup_logger(name):
    return logging.getLogger(name)


def generate_mock_market_data(days=100):
    """Génère des données de marché fictives pour les tests."""
    np.random.seed(42)  # Pour la reproductibilité

    date_range = [datetime.now() - timedelta(days=i) for i in range(days)]
    date_range.reverse()

    # Simuler un prix avec tendance et volatilité
    initial_price = 50000
    trend = np.cumsum(np.random.normal(0.001, 0.02, days))
    prices = initial_price * (1 + trend)

    # Créer un DataFrame avec des données OHLCV
    data = pd.DataFrame(
        {
            "date": date_range,
            "open": prices * (1 + np.random.normal(0, 0.01, days)),
            "high": prices * (1 + np.random.normal(0.02, 0.01, days)),
            "low": prices * (1 - np.random.normal(0.02, 0.01, days)),
            "close": prices,
            "volume": np.random.normal(1000000, 200000, days),
            "future_return": np.random.normal(0.001, 0.02, days),
            "direction_code": np.random.choice(
                [0, 1, 2], days
            ),  # 0=bearish, 1=neutral, 2=bullish
        }
    )

    return data


def generate_mock_sentiment_data(days=100):
    """Génère des données de sentiment fictives pour les tests."""
    np.random.seed(43)  # Pour la reproductibilité

    date_range = [datetime.now() - timedelta(days=i) for i in range(days)]
    date_range.reverse()

    data = pd.DataFrame(
        {
            "date": date_range,
            "news_sentiment": np.random.normal(0.2, 0.3, days),
            "social_sentiment": np.random.normal(0.1, 0.4, days),
            "news_volume": np.random.normal(100, 30, days),
            "social_volume": np.random.normal(500, 100, days),
        }
    )

    return data


import types


class TestMultiHorizonPredictor:
    """Tests unitaires pour la classe MultiHorizonPredictor."""

    def setup_method(self):
        """Configuration initiale pour chaque test."""
        # Appliquer les patches pour les dépendances externes
        self.patches = [
            patch(
                "ai_trading.llm.sentiment_analysis.news_analyzer.NewsAnalyzer",
                return_value=MockNewsAnalyzer(),
            ),
            patch(
                "ai_trading.llm.sentiment_analysis.social_analyzer.SocialAnalyzer",
                return_value=MockSocialAnalyzer(),
            ),
            patch("openai.OpenAI", MockOpenAI),
            patch("ai_trading.utils.setup_logger", mock_setup_logger),
        ]

        for p in self.patches:
            p.start()

        # Création de répertoire temporaire pour les tests
        import tempfile

        self.temp_dir = tempfile.mkdtemp()

        # Initialisation de l'objet à tester
        from ai_trading.llm.predictions.multi_horizon_predictor import (
            MultiHorizonPredictor,
        )

        self.predictor = MultiHorizonPredictor(
            llm_model="gpt-4", model_save_dir=self.temp_dir, use_hybrid=True
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        # Arrêter les patches
        for p in self.patches:
            p.stop()

        # Suppression des fichiers temporaires
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Teste l'initialisation du MultiHorizonPredictor."""
        assert self.predictor is not None
        assert self.predictor.llm_model == "gpt-4"
        assert self.predictor.model_save_dir == self.temp_dir
        assert self.predictor.use_hybrid is True
        assert self.predictor.market_predictor is not None
        assert isinstance(self.predictor.prediction_models, dict)

    def test_short_term_predictions(self):
        """Teste les prédictions à court terme."""

        # Application d'un mock simplifié pour la méthode predict du PredictionModel
        def mock_predict(self, asset_data, timeframe):
            return {
                "asset": asset_data.get("asset", "BTC"),
                "timeframe": timeframe,
                "direction": "bullish",
                "confidence": 0.75,
                "probability": 0.75,
                "factors": ["Prix en hausse", "Volume en augmentation"],
                "timestamp": datetime.now().isoformat(),
            }

        # Désactivation de la méthode d'initialisation des modèles
        def mock_initialize(self, timeframes):
            # Ne rien faire pour éviter les appels au PredictionModel
            return {}

        # Appliquer les mocks
        self.predictor.initialize_prediction_models = types.MethodType(
            mock_initialize, self.predictor
        )

        # Test des prédictions court terme
        short_term_predictions = self.predictor.predict_all_horizons(
            asset="BTC", short_term=True, medium_term=False, long_term=False
        )

        # Vérifications
        assert len(short_term_predictions) == len(self.predictor.SHORT_TERM)
        for timeframe, prediction in short_term_predictions.items():
            assert timeframe in self.predictor.SHORT_TERM
            assert prediction["asset"] == "BTC"
            assert prediction["timeframe"] == timeframe
            assert prediction["direction"] in ["bullish", "bearish", "neutral"]
            assert isinstance(prediction["confidence"], (float, int))
            assert 0 <= prediction["confidence"] <= 1
            assert "factors" in prediction

    def test_all_time_horizons(self):
        """Teste la prédiction sur tous les horizons temporels."""

        # Désactivation de la méthode d'initialisation des modèles
        def mock_initialize(self, timeframes):
            # Ne rien faire
            return {}

        # Appliquer le mock
        self.predictor.initialize_prediction_models = types.MethodType(
            mock_initialize, self.predictor
        )

        # Test des prédictions sur tous les horizons
        all_predictions = self.predictor.predict_all_horizons(
            asset="ETH", short_term=True, medium_term=True, long_term=True
        )

        # Vérifications
        all_timeframes = (
            self.predictor.SHORT_TERM
            + self.predictor.MEDIUM_TERM
            + self.predictor.LONG_TERM
        )
        assert len(all_predictions) == len(all_timeframes)
        for timeframe in all_timeframes:
            assert timeframe in all_predictions
            assert all_predictions[timeframe]["asset"] == "ETH"
            assert all_predictions[timeframe]["timeframe"] == timeframe

    def test_consistency_analysis(self):
        """Teste l'analyse de cohérence des prédictions."""
        # Générer des prédictions fictives
        all_predictions = {}

        # Prédictions court terme (toutes bullish)
        for timeframe in self.predictor.SHORT_TERM:
            all_predictions[timeframe] = {
                "direction": "bullish",
                "confidence": "high",
                "probability": 0.8,
                "factors": ["Prix en hausse", "Volume en augmentation"],
            }

        # Prédictions moyen terme (mixtes)
        for i, timeframe in enumerate(self.predictor.MEDIUM_TERM):
            direction = (
                "bullish" if i < len(self.predictor.MEDIUM_TERM) // 2 else "neutral"
            )
            confidence = "medium"
            all_predictions[timeframe] = {
                "direction": direction,
                "confidence": confidence,
                "probability": 0.6,
                "factors": ["Résistance", "Support"],
            }

        # Prédictions long terme (bearish)
        for timeframe in self.predictor.LONG_TERM:
            all_predictions[timeframe] = {
                "direction": "bearish",
                "confidence": "low",
                "probability": 0.55,
                "factors": ["Tendance baissière longue durée"],
            }

        # Analyser la cohérence
        consistency_analysis = self.predictor.analyze_consistency(all_predictions)

        # Vérifications
        assert "horizon_analysis" in consistency_analysis
        assert "trading_signals" in consistency_analysis
        assert "timestamp" in consistency_analysis

        # Vérifier l'analyse par horizon
        assert "short_term" in consistency_analysis["horizon_analysis"]
        assert "medium_term" in consistency_analysis["horizon_analysis"]
        assert "long_term" in consistency_analysis["horizon_analysis"]

        # Vérifier la direction globale par horizon
        assert (
            consistency_analysis["horizon_analysis"]["short_term"]["overall_direction"]
            == "bullish"
        )
        assert (
            consistency_analysis["horizon_analysis"]["long_term"]["overall_direction"]
            == "bearish"
        )

        # Vérifier le signal de trading
        assert "signal" in consistency_analysis["trading_signals"]
        assert "description" in consistency_analysis["trading_signals"]
        assert consistency_analysis["trading_signals"]["short_term_dir"] == "bullish"
        assert consistency_analysis["trading_signals"]["long_term_dir"] == "bearish"

    def test_trading_signals(self):
        """Teste la génération de signaux de trading basés sur l'analyse de cohérence."""
        # Cas 1: Signal d'achat fort (bullish sur tous les horizons)
        bullish_analysis = {
            "short_term": {
                "overall_direction": "bullish",
                "consistency": 0.9,
                "confidence": "high",
            },
            "medium_term": {
                "overall_direction": "bullish",
                "consistency": 0.8,
                "confidence": "medium",
            },
            "long_term": {
                "overall_direction": "bullish",
                "consistency": 0.75,
                "confidence": "medium",
            },
        }

        bullish_signal = self.predictor._generate_trading_signals(bullish_analysis)
        assert bullish_signal["signal"] == "strong_buy"

        # Cas 2: Signal de vente (bearish sur court et moyen terme)
        bearish_analysis = {
            "short_term": {
                "overall_direction": "bearish",
                "consistency": 0.8,
                "confidence": "medium",
            },
            "medium_term": {
                "overall_direction": "bearish",
                "consistency": 0.7,
                "confidence": "low",
            },
            "long_term": {
                "overall_direction": "neutral",
                "consistency": 0.6,
                "confidence": "low",
            },
        }

        bearish_signal = self.predictor._generate_trading_signals(bearish_analysis)
        assert bearish_signal["signal"] == "sell"

        # Cas 3: Signal de renversement potentiel
        reversal_analysis = {
            "short_term": {
                "overall_direction": "bearish",
                "consistency": 0.7,
                "confidence": "medium",
            },
            "medium_term": {
                "overall_direction": "neutral",
                "consistency": 0.5,
                "confidence": "low",
            },
            "long_term": {
                "overall_direction": "bullish",
                "consistency": 0.8,
                "confidence": "medium",
            },
        }

        reversal_signal = self.predictor._generate_trading_signals(reversal_analysis)
        assert reversal_signal["signal"] == "potential_reversal_bullish"


if __name__ == "__main__":
    # Exécution manuelle des tests
    test = TestMultiHorizonPredictor()
    try:
        test.setup_method()
        print("=== Test d'initialisation ===")
        test.test_initialization()
        print("=== Test de prédictions à court terme ===")
        test.test_short_term_predictions()
        print("=== Test de prédictions sur tous les horizons ===")
        test.test_all_time_horizons()
        print("=== Test d'analyse de cohérence ===")
        test.test_consistency_analysis()
        print("=== Test de génération de signaux de trading ===")
        test.test_trading_signals()
        print("=== Tous les tests ont réussi ===")
    except Exception as e:
        print(f"Erreur lors des tests: {e}")
        import traceback

        traceback.print_exc()
    finally:
        test.teardown_method()
