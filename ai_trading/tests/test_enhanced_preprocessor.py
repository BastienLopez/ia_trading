"""
Tests unitaires pour le module de prétraitement amélioré.
"""

import os
import sys
import tempfile
import unittest
import shutil

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
        self.preprocessor = EnhancedMarketDataPreprocessor(scaling_method="minmax")

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
        cleaned_data = self.preprocessor.clean_market_data(self.test_data)

        # Vérification que les valeurs manquantes ont été traitées
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)

        # Vérification que les dimensions sont correctes
        self.assertEqual(cleaned_data.shape, self.test_data.shape)

    def test_normalize_market_data(self):
        """Teste la normalisation des données de marché."""
        # Nettoyage préalable
        cleaned_data = self.preprocessor.clean_market_data(self.test_data)

        # Normalisation
        normalized_data = self.preprocessor.normalize_market_data(cleaned_data)

        # Vérification que les valeurs sont dans [0, 1] pour MinMaxScaler
        for col in ["open", "high", "low", "close", "volume", "market_cap"]:
            self.assertGreaterEqual(normalized_data[col].min(), 0)
            self.assertLessEqual(normalized_data[col].max(), 1.0001)

    def test_create_technical_features(self):
        """Teste la création des features techniques."""
        # Liste des features techniques attendues
        expected_features = [
            'returns', 'sma_7', 'sma_21', 'sma_50', 'volatility_7',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower',
            'sma_20', 'ema_50', 'atr'
        ]
        
        # Création des features techniques
        feature_data = self.preprocessor.create_technical_features(self.test_data)
        
        # Vérification des features
        for feature in expected_features:
            self.assertIn(feature, feature_data.columns)

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
        # Préparation des données
        cleaned_data = self.preprocessor.clean_market_data(self.test_data)

        # Test avec différentes méthodes
        for method in ["return", "direction", "threshold"]:
            target_data = self.preprocessor.create_target_variable(
                cleaned_data, horizon=1, method=method
            )
            self.assertIn("target", target_data.columns)

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
        # Créer un DataFrame avec suffisamment de données (100 périodes)
        test_data = pd.DataFrame({
            'open': np.linspace(100, 200, 100),
            'high': np.linspace(105, 205, 100),
            'low': np.linspace(95, 195, 100),
            'close': np.linspace(102, 202, 100),
            'volume': np.linspace(1000, 10000, 100),
            'timestamp': pd.date_range(start="2023-01-01", periods=100, freq="H")
        }).set_index('timestamp')

        # Test avec DataFrame
        processed_df = self.preprocessor.preprocess_market_data(test_data)
        
        # Vérifications
        self.assertIsNotNone(processed_df)
        self.assertGreater(len(processed_df), 50)  # Garder au moins 50% des données après nettoyage
        
        # Vérifier la présence des features clés
        expected_features = ['rsi', 'macd', 'macd_signal', 'atr']
        for feature in expected_features:
            self.assertIn(feature, processed_df.columns)


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
