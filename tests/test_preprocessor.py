"""
Tests unitaires pour le module de prétraitement des données.
"""

import unittest
from datetime import datetime, timedelta
import os
import shutil

import pandas as pd
import numpy as np

from ai_trading.utils.preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """Tests pour la classe DataPreprocessor."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.preprocessor = DataPreprocessor(cache_dir="test_cache")
        
        # Création de données de test
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 46000, len(dates)),
            'high': np.random.uniform(46000, 47000, len(dates)),
            'low': np.random.uniform(44000, 45000, len(dates)),
            'close': np.random.uniform(45000, 46000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }).set_index('timestamp')
        
        # Ajout de quelques valeurs manquantes et aberrantes
        self.test_data.loc[dates[5], 'close'] = np.nan
        self.test_data.loc[dates[10], 'volume'] = np.inf
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")
    
    def test_clean_market_data(self):
        """Test du nettoyage des données de marché."""
        # Test avec méthode ffill
        cleaned_df = self.preprocessor.clean_market_data(self.test_data, fill_method='ffill')
        
        # Vérifications
        self.assertFalse(cleaned_df.isnull().any().any())
        self.assertFalse(np.isinf(cleaned_df.values).any())
        self.assertEqual(len(cleaned_df), len(self.test_data))
        
        # Test avec méthode interpolate
        cleaned_df = self.preprocessor.clean_market_data(self.test_data, fill_method='interpolate')
        self.assertFalse(cleaned_df.isnull().any().any())
    
    def test_normalize_data(self):
        """Test de la normalisation des données."""
        # Test de la normalisation des prix
        normalized_df = self.preprocessor.normalize_data(
            self.test_data,
            price_cols=['open', 'high', 'low', 'close'],
            volume_cols=['volume']
        )
        
        # Vérifications
        self.assertTrue((normalized_df['open'] >= 0).all())
        self.assertTrue((normalized_df['open'] <= 1).all())
        self.assertTrue((normalized_df['volume'].std() - 1.0) < 0.1)
    
    def test_create_features(self):
        """Test de la création des features techniques."""
        # Test avec tous les types de features
        features_df = self.preprocessor.create_features(
            self.test_data,
            feature_types=['momentum', 'trend', 'volatility', 'volume']
        )
        
        # Vérification de la présence des indicateurs
        self.assertTrue('rsi' in features_df.columns)
        self.assertTrue('macd' in features_df.columns)
        self.assertTrue('bb_high' in features_df.columns)
        self.assertTrue('volume_ema' in features_df.columns)
        
        # Test avec un sous-ensemble de features
        features_df = self.preprocessor.create_features(
            self.test_data,
            feature_types=['momentum']
        )
        
        self.assertTrue('rsi' in features_df.columns)
        self.assertFalse('bb_high' in features_df.columns)
    
    def test_prepare_data(self):
        """Test de la préparation complète des données."""
        # Test avec toutes les étapes
        prepared_df = self.preprocessor.prepare_data(
            self.test_data,
            clean=True,
            normalize=True,
            add_features=True
        )
        
        # Vérifications
        self.assertFalse(prepared_df.isnull().any().any())
        self.assertTrue('rsi' in prepared_df.columns)
        self.assertTrue((prepared_df['open'] >= 0).all())
        self.assertTrue((prepared_df['open'] <= 1).all())
        
        # Test sans normalisation
        prepared_df = self.preprocessor.prepare_data(
            self.test_data,
            clean=True,
            normalize=False,
            add_features=True
        )
        
        self.assertTrue((prepared_df['open'] > 1000).any())
    
    def test_prepare_sequence_data(self):
        """Test de la préparation des séquences."""
        sequence_length = 5
        X, y = self.preprocessor.prepare_sequence_data(
            self.test_data,
            sequence_length=sequence_length,
            target_column='close'
        )
        
        # Vérifications
        expected_samples = len(self.test_data) - sequence_length
        expected_features = len(self.test_data.columns)
        
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], sequence_length)
        self.assertEqual(X.shape[2], expected_features)
        self.assertEqual(len(y), expected_samples)
        
        # Test avec features spécifiques
        feature_columns = ['close', 'volume']
        X, y = self.preprocessor.prepare_sequence_data(
            self.test_data,
            sequence_length=sequence_length,
            target_column='close',
            feature_columns=feature_columns
        )
        
        self.assertEqual(X.shape[2], len(feature_columns))

if __name__ == '__main__':
    unittest.main() 