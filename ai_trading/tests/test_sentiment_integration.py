"""Tests d'intégration pour le pipeline complet de sentiment."""

import os
import sys
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.llm.sentiment_analysis import EnhancedNewsAnalyzer, SocialAnalyzer
from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.config import VISUALIZATION_DIR


class TestSentimentIntegration(unittest.TestCase):
    def setUp(self):
        self.news_analyzer = EnhancedNewsAnalyzer()
        self.social_analyzer = SocialAnalyzer(platform="twitter")

        self.sample_news = [
            {
                "title": "Bitcoin New High",
                "body": "Bitcoin reaches $100k",
                "published_at": datetime.now().isoformat(),
            }
        ]

        self.sample_tweets = [
            {
                "text": "BTC to the moon!",
                "full_text": "BTC to the moon!",
                "retweet_count": 1000,
                "favorite_count": 2500,
                "reply_count": 300,
                "user": {"followers_count": 15000},
                "created_at": datetime.now().isoformat(),
            }
        ]

        # Créer des données de marché synthétiques
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        self.market_data = pd.DataFrame(
            {
                "open": np.linspace(100, 150, 30) + np.random.normal(0, 5, 30),
                "high": np.linspace(105, 155, 30) + np.random.normal(0, 5, 30),
                "low": np.linspace(95, 145, 30) + np.random.normal(0, 5, 30),
                "close": np.linspace(102, 152, 30) + np.random.normal(0, 5, 30),
                "volume": np.random.uniform(1000, 5000, 30),
            },
            index=dates,
        )

        # Créer des données de sentiment synthétiques
        self.sentiment_data = pd.DataFrame(
            {
                "polarity": np.random.uniform(-1, 1, 30),
                "subjectivity": np.random.uniform(0, 1, 30),
                "compound_score": np.random.uniform(-1, 1, 30),
            },
            index=dates,
        )

        # Initialiser l'intégrateur de données
        self.integrator = RLDataIntegrator()

    def test_full_news_analysis(self):
        df = self.news_analyzer.analyze_news(self.sample_news)
        self.assertIn("global_sentiment", df.columns)
        self.assertFalse(df.empty)

        report = self.news_analyzer.generate_report(df)
        self.assertIn("top_cryptos", report)

        # Générer le graphique dans le dossier visualizations
        visualization_path = os.path.join(VISUALIZATION_DIR, "sentiment_analysis.png")
        self.news_analyzer.plot_trends(df, visualization_path)

        # Vérifier que le fichier existe dans le bon dossier
        self.assertTrue(
            os.path.exists(visualization_path),
            f"Le fichier {visualization_path} n'existe pas",
        )

    def test_social_analysis_integration(self):
        df = self.social_analyzer.analyze_social_posts(self.sample_tweets)
        # Ajouter manuellement la colonne engagement pour le test
        df["engagement"] = 0.5
        enhanced_df = self.social_analyzer.calculate_virality(df)

        self.assertIn("viral_risk", enhanced_df.columns)
        self.assertTrue(enhanced_df["engagement"].between(0, 1).all())

    def test_integrate_sentiment_data(self):
        """Teste l'intégration des données de sentiment."""
        # Intégrer les données de sentiment
        integrated_data = self.integrator.integrate_sentiment_data(
            self.market_data, self.sentiment_data
        )

        # Vérifier que les données sont correctement intégrées
        self.assertIsNotNone(integrated_data)
        self.assertEqual(len(integrated_data), len(self.market_data))

        # Vérifier que les colonnes de sentiment sont présentes
        self.assertIn("polarity", integrated_data.columns)
        self.assertIn("subjectivity", integrated_data.columns)
        self.assertIn("compound_score", integrated_data.columns)

        # Vérifier qu'il n'y a pas de valeurs NaN
        self.assertFalse(integrated_data["polarity"].isna().any())
        self.assertFalse(integrated_data["subjectivity"].isna().any())
        self.assertFalse(integrated_data["compound_score"].isna().any())

    def test_integrate_sentiment_data_with_missing_values(self):
        """Teste l'intégration des données de sentiment avec des valeurs manquantes."""
        # Créer des données de sentiment avec des valeurs manquantes
        sentiment_with_missing = self.sentiment_data.copy()
        sentiment_with_missing.iloc[5:10, :] = np.nan

        # Intégrer les données de sentiment
        integrated_data = self.integrator.integrate_sentiment_data(
            self.market_data, sentiment_with_missing
        )

        # Vérifier que les valeurs manquantes sont remplies
        self.assertFalse(integrated_data["polarity"].isna().any())
        self.assertFalse(integrated_data["subjectivity"].isna().any())
        self.assertFalse(integrated_data["compound_score"].isna().any())

    def test_integrate_sentiment_data_with_different_frequencies(self):
        """Teste l'intégration des données de sentiment avec des fréquences différentes."""
        # Créer des données de sentiment avec une fréquence différente (hebdomadaire)
        weekly_dates = pd.date_range(start="2023-01-01", periods=5, freq="W")
        weekly_sentiment = pd.DataFrame(
            {
                "polarity": np.random.uniform(-1, 1, 5),
                "subjectivity": np.random.uniform(0, 1, 5),
                "compound_score": np.random.uniform(-1, 1, 5),
            },
            index=weekly_dates,
        )

        # Intégrer les données de sentiment
        integrated_data = self.integrator.integrate_sentiment_data(
            self.market_data, weekly_sentiment
        )

        # Vérifier que les données sont correctement intégrées
        self.assertIsNotNone(integrated_data)
        self.assertEqual(len(integrated_data), len(self.market_data))

        # Vérifier que les colonnes de sentiment sont présentes
        self.assertIn("polarity", integrated_data.columns)
        self.assertIn("subjectivity", integrated_data.columns)
        self.assertIn("compound_score", integrated_data.columns)

        # Vérifier qu'il n'y a pas de valeurs NaN
        self.assertFalse(integrated_data["polarity"].isna().any())
        self.assertFalse(integrated_data["subjectivity"].isna().any())
        self.assertFalse(integrated_data["compound_score"].isna().any())

    def test_integrate_sentiment_data_with_empty_sentiment(self):
        """Teste l'intégration des données de sentiment avec des données de sentiment vides."""
        # Créer des données de sentiment vides
        empty_sentiment = pd.DataFrame()

        # Intégrer les données de sentiment
        integrated_data = self.integrator.integrate_sentiment_data(
            self.market_data, empty_sentiment
        )

        # Vérifier que les données de marché sont retournées telles quelles
        self.assertIsNotNone(integrated_data)
        self.assertEqual(len(integrated_data), len(self.market_data))
        self.assertEqual(list(integrated_data.columns), list(self.market_data.columns))


if __name__ == "__main__":
    unittest.main()
