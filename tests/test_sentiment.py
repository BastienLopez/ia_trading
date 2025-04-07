"""
Tests unitaires pour le module d'analyse de sentiment.
"""

import unittest
from unittest.mock import Mock, patch
import os
import shutil

import numpy as np
import torch

from ai_trading.llm.sentiment_analysis.sentiment_model import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    """Tests pour la classe SentimentAnalyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        cls.test_cache_dir = "test_sentiment_cache"
        cls.analyzer = SentimentAnalyzer(cache_dir=cls.test_cache_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        if os.path.exists(cls.test_cache_dir):
            shutil.rmtree(cls.test_cache_dir)
    
    def test_initialization(self):
        """Test de l'initialisation de l'analyseur."""
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.tokenizer)
        self.assertIsNotNone(self.analyzer.sentiment_pipeline)
    
    def test_analyze_quick_single(self):
        """Test de l'analyse rapide d'un seul texte."""
        text = "Bitcoin hits new all-time high!"
        result = self.analyzer.analyze_quick(text)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn('normalized_score', result[0])
        self.assertIn('label', result[0])
        self.assertIn('confidence', result[0])
        self.assertGreater(result[0]['normalized_score'], -1)
        self.assertLess(result[0]['normalized_score'], 1)
    
    def test_analyze_quick_multiple(self):
        """Test de l'analyse rapide de plusieurs textes."""
        texts = [
            "Bitcoin crashes 20%",
            "New crypto regulation approved",
            "Major adoption of blockchain technology"
        ]
        results = self.analyzer.analyze_quick(texts)
        
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIn('normalized_score', result)
            self.assertIn('label', result)
            self.assertIn('confidence', result)
    
    def test_analyze_detailed(self):
        """Test de l'analyse détaillée."""
        text = "Bitcoin adoption grows as technology improves, but regulatory concerns remain."
        result = self.analyzer.analyze_detailed(text)
        
        self.assertIn('general', result)
        self.assertIn('aspects', result)
        self.assertIn('price', result['aspects'])
        self.assertIn('technology', result['aspects'])
        self.assertIn('adoption', result['aspects'])
        self.assertIn('regulation', result['aspects'])
    
    def test_analyze_batch(self):
        """Test de l'analyse par lots."""
        texts = ["Positive news"] * 50
        results = self.analyzer.analyze_batch(texts, batch_size=16)
        
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIn('normalized_score', result)
            self.assertIn('label', result)
            self.assertIn('confidence', result)
    
    def test_analyze_batch_detailed(self):
        """Test de l'analyse par lots avec analyse détaillée."""
        texts = ["Market news"] * 3
        results = self.analyzer.analyze_batch(texts, detailed=True)
        
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIn('general', result)
            self.assertIn('aspects', result)
    
    def test_aggregate_sentiment(self):
        """Test de l'agrégation des sentiments."""
        sentiments = [
            {'normalized_score': 0.8, 'label': 'POSITIVE', 'confidence': 0.9},
            {'normalized_score': -0.3, 'label': 'NEGATIVE', 'confidence': 0.7},
            {'normalized_score': 0.5, 'label': 'POSITIVE', 'confidence': 0.8}
        ]
        
        # Test sans poids
        result = self.analyzer.aggregate_sentiment(sentiments)
        self.assertIn('normalized_score', result)
        self.assertIn('label', result)
        self.assertIn('confidence', result)
        
        # Test avec poids
        weights = [0.5, 0.3, 0.2]
        result = self.analyzer.aggregate_sentiment(sentiments, weights=weights)
        self.assertGreater(result['normalized_score'], 0)  # Devrait être positif vu les poids
    
    def test_error_handling(self):
        """Test de la gestion des erreurs."""
        # Test avec texte vide
        result = self.analyzer.analyze_quick("")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'NEUTRAL')
        
        # Test avec liste vide
        result = self.analyzer.aggregate_sentiment([])
        self.assertEqual(result['label'], 'NEUTRAL')
        self.assertEqual(result['normalized_score'], 0.0)
        
        # Test avec texte invalide
        result = self.analyzer.analyze_detailed(None)
        self.assertEqual(result['general']['label'], 'NEUTRAL')

if __name__ == '__main__':
    unittest.main() 