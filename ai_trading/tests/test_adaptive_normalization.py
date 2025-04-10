import unittest
import numpy as np
import pandas as pd
from ai_trading.rl.adaptive_normalization import AdaptiveNormalizer

class TestAdaptiveNormalizer(unittest.TestCase):
    """Tests pour le normalisateur adaptatif."""
    
    def setUp(self):
        """Initialise le normalisateur pour les tests."""
        self.feature_names = ['price', 'volume', 'rsi']
        self.normalizer_minmax = AdaptiveNormalizer(
            window_size=100,
            method='minmax',
            clip_values=True,
            feature_names=self.feature_names
        )
        
        self.normalizer_zscore = AdaptiveNormalizer(
            window_size=100,
            method='zscore',
            clip_values=True,
            feature_names=self.feature_names
        )
    
    def test_normalize_minmax(self):
        """Teste la normalisation minmax."""
        # Créer des données de test
        features = {
            'price': 100,
            'volume': 1000,
            'rsi': 50
        }
        
        # Première normalisation (devrait donner 0.5 car min=max)
        normalized = self.normalizer_minmax.normalize(features)
        
        for key in normalized:
            self.assertEqual(normalized[key], 0.5)
        
        # Ajouter des valeurs extrêmes
        self.normalizer_minmax.update({'price': 200, 'volume': 2000, 'rsi': 100})
        self.normalizer_minmax.update({'price': 50, 'volume': 500, 'rsi': 0})
        
        # Normaliser à nouveau
        normalized = self.normalizer_minmax.normalize(features)
        
        # Vérifier que les valeurs sont correctement normalisées
        self.assertAlmostEqual(normalized['price'], 0.33, delta=0.01)
        self.assertAlmostEqual(normalized['volume'], 0.33, delta=0.01)
        self.assertAlmostEqual(normalized['rsi'], 0.5, delta=0.01)
    
    def test_normalize_zscore(self):
        """Teste la normalisation zscore."""
        # Créer des données de test
        features = {
            'price': 100,
            'volume': 1000,
            'rsi': 50
        }
        
        # Première normalisation (devrait donner 0 car mean=value)
        normalized = self.normalizer_zscore.normalize(features)
        
        for key in normalized:
            self.assertEqual(normalized[key], 0)
        
        # Ajouter des valeurs pour créer une distribution
        for i in range(10):
            self.normalizer_zscore.update({
                'price': 100 + 10 * np.random.randn(),
                'volume': 1000 + 100 * np.random.randn(),
                'rsi': 50 + 5 * np.random.randn()
            })
        
        # Normaliser à nouveau
        normalized = self.normalizer_zscore.normalize(features)
        
        # Vérifier que les valeurs sont dans une plage raisonnable
        for key in normalized:
            self.assertTrue(-3 <= normalized[key] <= 3)
    
    def test_normalize_array(self):
        """Teste la normalisation d'un tableau."""
        # Créer un tableau de test
        feature_array = np.array([100, 1000, 50])
        
        # Normaliser
        normalized = self.normalizer_minmax.normalize_array(feature_array)
        
        # Vérifier que le tableau a la bonne taille
        self.assertEqual(len(normalized), len(feature_array))
        
        # Vérifier que toutes les valeurs sont entre 0 et 1
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))
    
    def test_adaptive_behavior(self):
        """Teste le comportement adaptatif du normalisateur."""
        # Initialiser avec des valeurs
        for i in range(50):
            self.normalizer_minmax.update({
                'price': 100 + i,
                'volume': 1000 + i * 10,
                'rsi': 50
            })
        
        # Normaliser une valeur
        normalized1 = self.normalizer_minmax.normalize({'price': 125})
        
        # Ajouter des valeurs plus élevées
        for i in range(50):
            self.normalizer_minmax.update({
                'price': 200 + i,
                'volume': 2000 + i * 10,
                'rsi': 50
            })
        
        # Normaliser la même valeur
        normalized2 = self.normalizer_minmax.normalize({'price': 125})
        
        # La valeur normalisée devrait être plus petite après l'ajout de valeurs plus élevées
        self.assertLess(normalized2['price'], normalized1['price'])

if __name__ == "__main__":
    unittest.main() 