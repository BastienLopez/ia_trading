import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.adaptive_normalization import AdaptiveNormalizer

class TestAdaptiveNormalization(unittest.TestCase):
    """Tests pour la normalisation adaptative."""

    def setUp(self):
        """Prépare les données pour les tests."""
        # Créer des données synthétiques
        np.random.seed(42)
        self.data = pd.DataFrame({
            "feature1": np.random.normal(100, 15, 100),
            "feature2": np.random.uniform(0, 1000, 100),
            "feature3": np.random.exponential(5, 100),
            "non_numeric": ["A", "B", "C"] * 33 + ["A"]
        })
        
        # Initialiser le normalisateur
        self.normalizer = AdaptiveNormalizer()
    
    def test_normalize_features_minmax(self):
        """Teste la normalisation minmax des features."""
        # Normaliser les features avec la méthode minmax
        normalized_df = self.normalizer.normalize_features(self.data, method="minmax")
        
        # Vérifier que les données sont correctement normalisées
        self.assertIsNotNone(normalized_df)
        self.assertEqual(len(normalized_df), len(self.data))
        
        # Vérifier que les valeurs sont entre 0 et 1 pour les colonnes numériques
        for col in ["feature1", "feature2", "feature3"]:
            self.assertTrue((normalized_df[col] >= 0).all())
            self.assertTrue((normalized_df[col] <= 1).all())
        
        # Vérifier que la colonne non numérique est inchangée
        pd.testing.assert_series_equal(normalized_df["non_numeric"], self.data["non_numeric"])
    
    def test_normalize_features_zscore(self):
        """Teste la normalisation zscore des features."""
        # Normaliser les features avec la méthode zscore
        normalized_df = self.normalizer.normalize_features(self.data, method="zscore")
        
        # Vérifier que les données sont correctement normalisées
        self.assertIsNotNone(normalized_df)
        self.assertEqual(len(normalized_df), len(self.data))
        
        # Vérifier que la moyenne est proche de 0 et l'écart-type proche de 1 pour les colonnes numériques
        for col in ["feature1", "feature2", "feature3"]:
            self.assertAlmostEqual(normalized_df[col].mean(), 0, delta=0.1)
            self.assertAlmostEqual(normalized_df[col].std(), 1, delta=0.1)
        
        # Vérifier que la colonne non numérique est inchangée
        pd.testing.assert_series_equal(normalized_df["non_numeric"], self.data["non_numeric"])
    
    def test_normalize_features_adaptive(self):
        """Teste la normalisation adaptative des features."""
        # Normaliser les features avec la méthode adaptive
        normalized_df = self.normalizer.normalize_features(self.data, method="adaptive", window_size=20)
        
        # Vérifier que les données sont correctement normalisées
        self.assertIsNotNone(normalized_df)
        self.assertEqual(len(normalized_df), len(self.data))
        
        # Vérifier que les valeurs sont entre 0 et 1 pour les colonnes numériques
        for col in ["feature1", "feature2", "feature3"]:
            self.assertTrue((normalized_df[col] >= 0).all())
            self.assertTrue((normalized_df[col] <= 1).all())
        
        # Vérifier que la colonne non numérique est inchangée
        pd.testing.assert_series_equal(normalized_df["non_numeric"], self.data["non_numeric"])
    
    def test_normalize_features_with_missing_values(self):
        """Teste la normalisation des features avec des valeurs manquantes."""
        # Créer des données avec des valeurs manquantes
        data_with_missing = self.data.copy()
        data_with_missing.iloc[10:20, 0] = np.nan
        data_with_missing.iloc[30:40, 1] = np.nan
        
        # Normaliser les features
        normalized_df = self.normalizer.normalize_features(data_with_missing, method="adaptive")
        
        # Vérifier que les données sont correctement normalisées
        self.assertIsNotNone(normalized_df)
        self.assertEqual(len(normalized_df), len(data_with_missing))
        
        # Vérifier que les valeurs NaN sont toujours NaN
        self.assertTrue(normalized_df.iloc[10:20, 0].isna().all())
        self.assertTrue(normalized_df.iloc[30:40, 1].isna().all())
    
    def test_normalize_features_with_empty_dataframe(self):
        """Teste la normalisation des features avec un DataFrame vide."""
        # Créer un DataFrame vide
        empty_df = pd.DataFrame()
        
        # Normaliser les features
        normalized_df = self.normalizer.normalize_features(empty_df)
        
        # Vérifier que le résultat est un DataFrame vide
        self.assertTrue(normalized_df.empty)
    
    def test_normalize_features_with_constant_values(self):
        """Teste la normalisation des features avec des valeurs constantes."""
        # Créer des données avec des valeurs constantes
        constant_data = pd.DataFrame({
            "constant1": [5] * 100,
            "constant2": [10] * 100
        })
        
        # Normaliser les features
        normalized_df = self.normalizer.normalize_features(constant_data, method="adaptive")
        
        # Vérifier que les données sont correctement normalisées
        self.assertIsNotNone(normalized_df)
        self.assertEqual(len(normalized_df), len(constant_data))
        
        # Pour des valeurs constantes, la normalisation devrait donner 0.5
        self.assertTrue((normalized_df["constant1"] == 0.5).all())
        self.assertTrue((normalized_df["constant2"] == 0.5).all())

if __name__ == "__main__":
    unittest.main() 