"""
Tests unitaires pour le module sentiment_utils.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ai_trading.llm.sentiment_analysis.sentiment_utils import (
    SentimentCache,
    SentimentVisualizer,
    calculate_sentiment_metrics,
    get_llm_client,
)


class TestSentimentCache(unittest.TestCase):
    """Tests pour la classe SentimentCache."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SentimentCache(cache_dir=self.temp_dir)

    def test_cache_initialization(self):
        """Teste l'initialisation du cache."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(Path(self.temp_dir).is_dir())

    def test_save_and_load(self):
        """Teste la sauvegarde et le chargement depuis le cache."""
        test_key = "test_key"
        test_data = {"sentiment": "positive", "score": 0.9}

        # Sauvegarde dans le cache
        self.cache.save(test_key, test_data)

        # Chargement depuis le cache
        loaded_data = self.cache.load(test_key)
        self.assertEqual(loaded_data, test_data)

    def test_load_nonexistent(self):
        """Teste le chargement d'une clé inexistante."""
        self.assertIsNone(self.cache.load("nonexistent_key"))


class TestSentimentVisualizer(unittest.TestCase):
    """Tests pour la classe SentimentVisualizer."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = SentimentVisualizer(output_dir=self.temp_dir)
        
        # Données de test
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        self.test_data = pd.DataFrame({
            "date": dates,
            "score": np.random.uniform(-1, 1, size=10),
            "sentiment": np.random.choice(["positive", "neutral", "negative"], size=10)
        })

    def test_visualizer_initialization(self):
        """Teste l'initialisation du visualiseur."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(Path(self.temp_dir).is_dir())

    def test_plot_sentiment_trends(self):
        """Teste la génération du graphique des tendances."""
        self.visualizer.plot_sentiment_trends(self.test_data)
        self.assertTrue(any(Path(self.temp_dir).glob("sentiment_trends*.png")))

    def test_plot_sentiment_distribution(self):
        """Teste la génération du graphique de distribution."""
        self.visualizer.plot_sentiment_distribution(self.test_data)
        self.assertTrue(any(Path(self.temp_dir).glob("sentiment_distribution*.png")))


class TestSentimentUtils(unittest.TestCase):
    """Tests pour les fonctions utilitaires."""

    @patch("ai_trading.llm.sentiment_analysis.sentiment_utils.pipeline")
    def test_get_llm_client(self, mock_pipeline):
        """Teste la création du client LLM."""
        # Configuration du mock
        mock_pipeline.return_value = Mock()

        # Test avec les paramètres par défaut
        client = get_llm_client()
        self.assertIsNotNone(client)
        mock_pipeline.assert_called_once_with(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            top_k=None,  # Nouveau paramètre à la place de return_all_scores
        )

        # Test avec une erreur
        mock_pipeline.side_effect = Exception("Test error")
        client = get_llm_client()
        self.assertIsNone(client)

    def test_calculate_sentiment_metrics(self):
        """Teste le calcul des métriques de sentiment."""
        test_sentiments = [
            {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            {"positive": 0.2, "negative": 0.7, "neutral": 0.1},
            {"positive": 0.3, "negative": 0.3, "neutral": 0.4},
        ]

        metrics = calculate_sentiment_metrics(test_sentiments)

        self.assertIsInstance(metrics, dict)
        self.assertIn("average_score", metrics)
        self.assertIn("positive_ratio", metrics)
        self.assertIn("negative_ratio", metrics)
        self.assertIn("neutral_ratio", metrics)


def test_sentiment_visualization():
    # Test de la visualisation des sentiments
    pass

def test_sentiment_caching():
    # Test du cache des sentiments
    pass

def test_sentiment_tools():
    # Test des outils de sentiment
    pass

if __name__ == '__main__':
    pytest.main([__file__]) 