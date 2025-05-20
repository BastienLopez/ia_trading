"""
Tests unitaires pour le module enhanced_news_analyzer.
"""

import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd

from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer


class TestEnhancedNewsAnalyzer(unittest.TestCase):
    """Tests pour la classe EnhancedNewsAnalyzer."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = EnhancedNewsAnalyzer(
            enable_cache=True,
            cache_dir=self.temp_dir,
        )
        
        # Données de test
        self.test_news = [
            {
                "title": "Bitcoin hits new all-time high!",
                "body": "Bitcoin (BTC) reached $100,000, up 15% today.",
                "published_at": "2024-03-20T12:00:00Z",
            },
            {
                "title": "Market crash: Ethereum drops",
                "body": "ETH price falls 20% to $2,000.",
                "published_at": "2024-03-20T13:00:00Z",
            },
        ]

    def test_analyzer_initialization(self):
        """Teste l'initialisation de l'analyseur."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.cache)
        self.assertTrue(os.path.exists(self.temp_dir))

    @patch("ai_trading.llm.sentiment_analysis.sentiment_utils.pipeline")
    def test_analyze_news(self, mock_pipeline):
        """Teste l'analyse d'un batch d'actualités."""
        # Configuration du mock
        mock_pipeline.return_value = Mock(return_value=[{"label": "positive", "score": 0.8}])
        
        # Analyse des actualités
        results = self.analyzer.analyze_news(self.test_news)
        
        # Vérifications
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
        self.assertIn("title_sentiment", results.columns)
        self.assertIn("body_sentiment", results.columns)
        self.assertIn("global_sentiment", results.columns)
        self.assertIn("entities", results.columns)

    def test_extract_entities(self):
        """Teste l'extraction des entités."""
        text = "Bitcoin (BTC) price reached $50,000, up 10% today!"
        entities = self.analyzer._extract_entities(text)
        
        self.assertIn("bitcoin", entities["crypto_entities"])
        self.assertIn("btc", entities["crypto_entities"])
        self.assertIn("$50,000", entities["money_entities"])
        self.assertIn("10%", entities["percentage_entities"])

    def test_calculate_global_sentiment(self):
        """Teste le calcul du sentiment global."""
        title_sentiment = {"label": "positive", "score": 0.8}
        body_sentiment = {"label": "negative", "score": -0.4}
        
        global_sentiment = self.analyzer._calculate_global_sentiment(
            title_sentiment, body_sentiment
        )
        
        self.assertIn("label", global_sentiment)
        self.assertIn("score", global_sentiment)
        # 0.8 * 0.6 + (-0.4) * 0.4 = 0.32
        self.assertAlmostEqual(global_sentiment["score"], 0.32)
        self.assertEqual(global_sentiment["label"], "positive")

    def test_generate_report(self):
        """Teste la génération du rapport."""
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            "title": ["Test 1", "Test 2"],
            "published_at": [
                datetime(2024, 3, 20, 12, 0),
                datetime(2024, 3, 20, 13, 0),
            ],
            "global_sentiment": [
                {"label": "positive", "score": 0.8},
                {"label": "negative", "score": -0.6},
            ],
            "entities": [
                {
                    "crypto_entities": ["bitcoin", "btc"],
                    "money_entities": ["$50,000"],
                    "percentage_entities": ["10%"],
                },
                {
                    "crypto_entities": ["ethereum", "eth"],
                    "money_entities": ["$2,000"],
                    "percentage_entities": ["20%"],
                },
            ],
        })
        
        report = self.analyzer.generate_report(df)
        
        self.assertEqual(report["total_articles"], 2)
        self.assertIn("average_sentiment", report)
        self.assertIn("sentiment_distribution", report)
        self.assertIn("entities", report)
        self.assertIn("sentiment_trend", report)

    def test_empty_report(self):
        """Teste la génération d'un rapport vide."""
        report = self.analyzer.generate_report(pd.DataFrame())
        
        self.assertEqual(report["total_articles"], 0)
        self.assertEqual(report["average_sentiment"], 0.0)
        self.assertEqual(report["sentiment_distribution"], {})
        self.assertEqual(report["entities"], {})
        self.assertEqual(report["sentiment_trend"], {})

    @patch("matplotlib.pyplot.savefig")
    def test_plot_trends(self, mock_savefig):
        """Teste la génération du graphique des tendances."""
        df = pd.DataFrame({
            "published_at": [
                datetime(2024, 3, 20, 12, 0),
                datetime(2024, 3, 20, 13, 0),
            ],
            "global_sentiment": [
                {"label": "positive", "score": 0.8},
                {"label": "negative", "score": -0.6},
            ],
        })
        
        self.analyzer.plot_trends(df)
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_distribution(self, mock_savefig):
        """Teste la génération du graphique de distribution."""
        df = pd.DataFrame({
            "global_sentiment": [
                {"label": "positive", "score": 0.8},
                {"label": "negative", "score": -0.6},
                {"label": "positive", "score": 0.7},
            ],
        })
        
        self.analyzer.plot_distribution(df)
        mock_savefig.assert_called_once()


if __name__ == "__main__":
    unittest.main()
