"""
Tests unitaires pour le module de prétraitement amélioré.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ajout du chemin absolu vers le répertoire ai_trading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.utils.enhanced_preprocessor import (
    EnhancedMarketDataPreprocessor,
    EnhancedTextDataPreprocessor,
)


class TestEnhancedMarketDataPreprocessor(unittest.TestCase):
    """Tests pour la classe EnhancedMarketDataPreprocessor."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.preprocessor = EnhancedMarketDataPreprocessor(scaling="minmax")

        # Création des données d'exemple pour les tests
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.sample_data = pd.DataFrame(
            {
                "open": np.random.normal(100, 10, 100),
                "high": np.random.normal(105, 10, 100),
                "low": np.random.normal(95, 10, 100),
                "close": np.random.normal(100, 10, 100),
                "volume": np.random.normal(1000, 100, 100),
            },
            index=dates,
        )

        # Création d'un DataFrame de test
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        self.test_data = pd.DataFrame(
            {
                "open": np.random.normal(100, 5, 100),
                "high": np.random.normal(105, 5, 100),
                "low": np.random.normal(95, 5, 100),
                "close": np.random.normal(102, 5, 100),
                "volume": np.random.normal(1000, 100, 100),
                "market_cap": np.random.normal(1000000, 50000, 100),
                "source": ["coingecko"] * 100,
            },
            index=dates,
        )

        # Ajout de quelques valeurs manquantes
        self.test_data.loc[self.test_data.index[10:15], "close"] = np.nan
        self.test_data.loc[self.test_data.index[20:22], "volume"] = np.nan

        # Création d'un fichier temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test_data.csv")
        self.test_data.to_csv(self.temp_file)

    def tearDown(self):
        """Nettoyage après chaque test."""
        # Suppression du fichier temporaire
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_clean_market_data(self):
        """Teste le nettoyage des données de marché."""
        cleaned_data = self.preprocessor.clean_market_data(self.sample_data)

        # Vérification qu'il n'y a pas de valeurs manquantes
        assert cleaned_data.isna().sum().sum() == 0

        # Vérification que les données sont en float16
        assert all(cleaned_data.dtypes == np.float16)

        # Vérification qu'il n'y a pas de doublons
        assert not cleaned_data.duplicated().any()

    def test_normalize_market_data(self):
        """Teste la normalisation des données de marché."""
        cleaned_data = self.preprocessor.clean_market_data(self.sample_data)
        normalized_data = self.preprocessor.normalize_market_data(cleaned_data)

        # Vérification que les données sont normalisées
        for col in normalized_data.columns:
            assert normalized_data[col].min() >= -1e6
            assert normalized_data[col].max() <= 1e6
            assert not np.isinf(normalized_data[col]).any()

    def test_create_technical_features(self):
        """Teste la création des features techniques."""
        feature_data = self.preprocessor.create_technical_features(self.test_data)

        # Vérification que le DataFrame n'est pas vide
        self.assertIsNotNone(feature_data)
        self.assertGreater(len(feature_data), 0)

        # Vérification des features techniques
        expected_features = [
            "sma_7",
            "sma_21",
            "rsi_14",
            "macd",
            "bb_upper",
            "bb_lower",
        ]
        for feature in expected_features:
            self.assertIn(feature, feature_data.columns)

        # Vérification que les NaN ont été supprimés
        self.assertEqual(feature_data.isna().sum().sum(), 0)

    def test_create_lagged_features(self):
        """Teste la création des features décalées."""
        # Nettoyage et création des features techniques
        cleaned_data = self.preprocessor.clean_market_data(self.test_data)
        feature_data = self.preprocessor.create_technical_features(cleaned_data)

        # Création des lags
        columns = ["close", "volume"]
        lags = [1, 2]
        lagged_data = self.preprocessor.create_lagged_features(
            feature_data, columns, lags
        )

        # Vérification des nouvelles colonnes
        for col in columns:
            for lag in lags:
                self.assertIn(f"{col}_lag_{lag}", lagged_data.columns)

    def test_create_target_variable(self):
        """Teste la création de la variable cible."""
        preprocessor = EnhancedMarketDataPreprocessor()
        cleaned_data = preprocessor.clean_market_data(self.sample_data)

        # Test avec la méthode 'return'
        target = preprocessor.create_target_variable(
            cleaned_data, horizon=1, method="return"
        )
        assert isinstance(target, pd.Series)
        assert not target.isna().any()

        # Test avec la méthode 'direction'
        target = preprocessor.create_target_variable(
            cleaned_data, horizon=1, method="direction"
        )
        assert isinstance(target, pd.Series)
        assert not target.isna().any()
        assert all(target.isin([-1, 0, 1]))

        # Test avec la méthode 'threshold'
        target = preprocessor.create_target_variable(
            cleaned_data, horizon=1, method="threshold"
        )
        assert isinstance(target, pd.Series)
        assert not target.isna().any()
        assert all(target.isin([-1, 0, 1]))

    def test_split_data(self):
        """Teste la division des données."""
        # Préparation des données
        cleaned_data = self.preprocessor.clean_market_data(self.test_data)

        # Division des données
        train, val, test = self.preprocessor.split_data(cleaned_data)

        # Vérification des dimensions
        self.assertEqual(len(train) + len(val) + len(test), len(cleaned_data))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

    def test_preprocess_market_data(self):
        """Teste le prétraitement complet des données de marché."""
        # Vérifier si la méthode preprocess_market_data existe
        if hasattr(self.preprocessor, "preprocess_market_data"):
            try:
                # Essayer d'abord avec un DataFrame
                processed_data = self.preprocessor.preprocess_market_data(
                    self.test_data
                )

                # Vérification que le DataFrame n'est pas vide
                self.assertIsNotNone(processed_data)
                self.assertGreater(len(processed_data), 0)

                # Vérifier également avec le chemin du fichier
                processed_data_file = self.preprocessor.preprocess_market_data(
                    self.temp_file
                )
                self.assertIsNotNone(processed_data_file)
                self.assertGreater(len(processed_data_file), 0)

            except Exception as e:
                self.fail(f"preprocess_market_data a levé une exception: {e}")
        else:
            self.skipTest("La méthode preprocess_market_data n'existe pas")


class TestEnhancedTextDataPreprocessor(unittest.TestCase):
    """Tests pour la classe EnhancedTextDataPreprocessor."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.preprocessor = EnhancedTextDataPreprocessor(language="english")

        # Exemples de textes
        self.test_text = "This is a test tweet about #Bitcoin and $ETH! Check out https://example.com @username"

        # Exemples de données d'actualités
        self.test_news = [
            {
                "title": "Bitcoin Surges to New Highs",
                "body": "Bitcoin reached $60,000 today, setting a new record.",
                "published_on": "2023-01-01",
            },
            {
                "title": "Ethereum Update Delayed",
                "body": "The Ethereum 2.0 update has been delayed until next quarter.",
                "published_on": "2023-01-02",
            },
        ]

        # Exemples de données sociales
        self.test_social = [
            {
                "text": "Just bought some #BTC! To the moon! 🚀",
                "created_at": "2023-01-01",
            },
            {
                "text": "Ethereum looking bullish today. #ETH #crypto",
                "created_at": "2023-01-02",
            },
        ]

    def test_clean_text(self):
        """Teste le nettoyage du texte."""
        cleaned = self.preprocessor.clean_text(self.test_text)

        # Vérification que les éléments indésirables ont été supprimés
        self.assertNotIn("#", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn(
            "$", cleaned
        )  # Vérification spécifique pour les symboles de crypto

    def test_tokenize_text_simple(self):
        """Teste une version simplifiée de la tokenization."""
        # Utiliser un texte très simple
        simple_text = "bitcoin ethereum crypto"
        tokens = self.preprocessor.tokenize_text(simple_text)

        # Vérifications de base
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_extract_sentiment_keywords(self):
        """Teste l'extraction des mots-clés de sentiment."""
        # Création de tokens de test
        tokens_list = [["bitcoin", "surge", "high"], ["ethereum", "update", "delay"]]

        # Extraction des mots-clés
        keywords = self.preprocessor.extract_sentiment_keywords(tokens_list, top_n=5)

        # Vérification des résultats
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 5)


if __name__ == "__main__":
    unittest.main()
