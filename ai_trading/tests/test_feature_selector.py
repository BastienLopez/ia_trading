import unittest
import pandas as pd
import numpy as np
from ..utils.feature_selector import FeatureSelector

class TestFeatureSelector(unittest.TestCase):
    def setUp(self):
        """Prépare les données de test."""
        # Création d'un DataFrame de test
        np.random.seed(42)
        n_samples = 1000
        
        # Création de caractéristiques avec différents niveaux d'importance
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples),
            'feature5': np.random.normal(0, 1, n_samples)
        })
        
        # Création d'une variable cible avec des dépendances connues
        self.data['target'] = (
            2 * self.data['feature1'] +  # forte importance
            1 * self.data['feature2'] +  # importance moyenne
            0.5 * self.data['feature3'] + # faible importance
            0.1 * self.data['feature4'] + # très faible importance
            0 * self.data['feature5'] +   # aucune importance
            np.random.normal(0, 0.1, n_samples)  # bruit
        )
        
        self.selector = FeatureSelector(n_features=3)

    def test_select_features(self):
        """Teste la sélection des caractéristiques."""
        selected_features = self.selector.select_features(self.data, 'target')
        
        # Vérifie que le bon nombre de caractéristiques est sélectionné
        self.assertEqual(len(selected_features), 3)
        
        # Vérifie que feature1 et feature2 sont parmi les caractéristiques sélectionnées
        # car elles ont la plus forte influence sur la target
        self.assertIn('feature1', selected_features)
        self.assertIn('feature2', selected_features)

    def test_get_feature_importance(self):
        """Teste la récupération des scores d'importance."""
        # Sélectionne d'abord les caractéristiques
        self.selector.select_features(self.data, 'target')
        
        # Récupère les scores d'importance
        importance_scores = self.selector.get_feature_importance()
        
        # Vérifie que toutes les caractéristiques ont un score
        self.assertEqual(len(importance_scores), len(self.data.columns) - 1)
        
        # Vérifie que feature1 a un score plus élevé que feature5
        self.assertGreater(
            importance_scores['feature1'],
            importance_scores['feature5']
        )

    def test_transform(self):
        """Teste la transformation des données."""
        # Sélectionne d'abord les caractéristiques
        selected_features = self.selector.select_features(self.data, 'target')
        
        # Transforme les données
        transformed_data = self.selector.transform(self.data)
        
        # Vérifie que seules les caractéristiques sélectionnées sont présentes
        self.assertEqual(len(transformed_data.columns), 3)
        for feature in selected_features:
            self.assertIn(feature, transformed_data.columns)

    def test_error_without_selection(self):
        """Teste les erreurs quand select_features n'a pas été appelé."""
        selector = FeatureSelector()
        
        # Teste get_feature_importance sans sélection préalable
        with self.assertRaises(ValueError):
            selector.get_feature_importance()
            
        # Teste transform sans sélection préalable
        with self.assertRaises(ValueError):
            selector.transform(self.data)

if __name__ == '__main__':
    unittest.main() 