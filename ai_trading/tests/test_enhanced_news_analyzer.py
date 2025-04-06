"""
Tests unitaires pour le module d'analyse de sentiment amélioré.
"""

import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import tempfile
import json

# Ajout du chemin absolu vers le répertoire ai_trading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer

class TestEnhancedNewsAnalyzer(unittest.TestCase):
    """Tests pour la classe EnhancedNewsAnalyzer."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.analyzer = EnhancedNewsAnalyzer()
        
        # Exemples d'actualités pour les tests
        self.test_news = [
            {
                "title": "Bitcoin Surges to $60,000 as Institutional Adoption Grows",
                "body": "Bitcoin reached a new all-time high of $60,000 today as more institutional investors are entering the cryptocurrency market. Major companies like Tesla and MicroStrategy have added BTC to their balance sheets.",
                "published_at": "2023-03-15T12:30:00Z"
            },
            {
                "title": "Ethereum Price Drops 10% Following Network Congestion",
                "body": "Ethereum (ETH) experienced a significant price drop of 10% in the last 24 hours due to network congestion and high gas fees. Developers are working on solutions to address these scaling issues.",
                "published_at": "2023-03-14T09:15:00Z"
            }
        ]
    
    def test_analyze_sentiment(self):
        """Teste l'analyse de sentiment."""
        # Test avec un texte positif
        positive_text = "Bitcoin surges to new all-time high as adoption grows"
        positive_result = self.analyzer.analyze_sentiment(positive_text)
        
        self.assertIsInstance(positive_result, dict)
        self.assertIn("label", positive_result)
        self.assertIn("score", positive_result)
        
        # Test avec un texte négatif
        negative_text = "Bitcoin crashes 20% as market fears regulatory crackdown"
        negative_result = self.analyzer.analyze_sentiment(negative_text)
        
        self.assertIsInstance(negative_result, dict)
        self.assertIn("label", negative_result)
        self.assertIn("score", negative_result)
        
        # Vérification que les scores sont différents
        self.assertNotEqual(positive_result["score"], negative_result["score"])
    
    def test_extract_entities(self):
        """Teste l'extraction d'entités."""
        # Test avec un texte contenant des entités crypto
        text = "Bitcoin and Ethereum are the largest cryptocurrencies by market cap"
        entities = self.analyzer.extract_entities(text)
        
        self.assertIsInstance(entities, list)
        
        # Vérification que des entités sont détectées
        # Note: Même si le modèle n'est pas disponible, la méthode de repli devrait fonctionner
        self.assertGreater(len(entities), 0)
    
    def test_analyze_news(self):
        """Teste l'analyse complète des actualités."""
        enriched_news = self.analyzer.analyze_news(self.test_news)
        
        self.assertIsInstance(enriched_news, list)
        self.assertEqual(len(enriched_news), len(self.test_news))
        
        # Vérification des champs ajoutés
        for news in enriched_news:
            self.assertIn("title_sentiment", news)
            self.assertIn("body_sentiment", news)
            self.assertIn("global_sentiment", news)
    
    def test_analyze_news_dataframe(self):
        """Teste l'analyse des actualités à partir d'un DataFrame."""
        # Création d'un DataFrame
        df = pd.DataFrame(self.test_news)
        
        # Analyse
        enriched_df = self.analyzer.analyze_news_dataframe(df)
        
        self.assertIsInstance(enriched_df, pd.DataFrame)
        self.assertEqual(len(enriched_df), len(df))
        
        # Vérification des colonnes ajoutées
        expected_columns = [
            "title_sentiment_label", 
            "title_sentiment_score", 
            "body_sentiment_label", 
            "body_sentiment_score", 
            "global_sentiment_label", 
            "global_sentiment_score"
        ]
        
        for col in expected_columns:
            self.assertIn(col, enriched_df.columns)
    
    def test_generate_sentiment_report(self):
        """Teste la génération du rapport de sentiment."""
        # Création d'un DataFrame enrichi
        df = pd.DataFrame(self.test_news)
        enriched_df = self.analyzer.analyze_news_dataframe(df)
        
        # Génération du rapport
        report = self.analyzer.generate_sentiment_report(enriched_df)
        
        self.assertIsInstance(report, dict)
        self.assertIn("total_articles", report)
        self.assertIn("sentiment_distribution", report)
        
        # Vérification des valeurs
        self.assertEqual(report["total_articles"], len(df))
    
    def test_clean_text(self):
        """Teste le nettoyage de texte."""
        # Test avec un texte contenant des URLs et des balises HTML
        text = "Check out this link: https://example.com and <b>bold text</b>"
        cleaned_text = self.analyzer._clean_text(text)
        
        # Vérification que les URLs et balises HTML sont supprimées
        self.assertNotIn("https://", cleaned_text)
        self.assertNotIn("<b>", cleaned_text)
        self.assertNotIn("</b>", cleaned_text)
    
    def test_hash_text(self):
        """Teste la génération de hash pour un texte."""
        text = "This is a test text"
        hash_value = self.analyzer._hash_text(text)
        
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 32)  # MD5 hash length
        
        # Vérification que le même texte donne le même hash
        hash_value2 = self.analyzer._hash_text(text)
        self.assertEqual(hash_value, hash_value2)
        
        # Vérification que des textes différents donnent des hashs différents
        hash_value3 = self.analyzer._hash_text("Different text")
        self.assertNotEqual(hash_value, hash_value3)
    
    def test_cache_functionality(self):
        """Teste la fonctionnalité de cache."""
        # Création d'un répertoire temporaire pour le cache
        with tempfile.TemporaryDirectory() as temp_dir:
            # Création d'un analyseur avec cache
            analyzer_with_cache = EnhancedNewsAnalyzer(enable_cache=True, cache_dir=temp_dir)
            
            # Création d'un cache fictif
            test_cache = {"test_key": "test_value"}
            cache_file = os.path.join(temp_dir, "test_cache.pkl")
            
            # Sauvegarde du cache
            analyzer_with_cache._save_cache(test_cache, cache_file)
            
            # Vérification que le fichier de cache a été créé
            self.assertTrue(os.path.exists(cache_file))
            
            # Chargement du cache
            loaded_cache = analyzer_with_cache._load_cache(cache_file)
            
            # Vérification que le cache chargé est correct
            self.assertEqual(loaded_cache, test_cache)
    
    def test_fallback_sentiment_analysis(self):
        """Teste l'analyse de sentiment de repli."""
        # Test avec un texte positif
        positive_text = "Bitcoin is showing bullish signals with strong growth and potential"
        positive_result = self.analyzer._fallback_sentiment_analysis(positive_text)
        
        self.assertIsInstance(positive_result, dict)
        self.assertIn("label", positive_result)
        self.assertIn("score", positive_result)
        self.assertEqual(positive_result["label"], "positive")
        
        # Test avec un texte négatif
        negative_text = "Bitcoin crashes with significant losses and high risk of further decline"
        negative_result = self.analyzer._fallback_sentiment_analysis(negative_text)
        
        self.assertIsInstance(negative_result, dict)
        self.assertIn("label", negative_result)
        self.assertIn("score", negative_result)
        self.assertEqual(negative_result["label"], "negative")
        
        # Test avec un texte neutre
        neutral_text = "Bitcoin price remains stable with some positive and negative signals"
        neutral_result = self.analyzer._fallback_sentiment_analysis(neutral_text)
        
        self.assertIsInstance(neutral_result, dict)
        self.assertIn("label", neutral_result)
        self.assertIn("score", neutral_result)
        self.assertEqual(neutral_result["label"], "neutral")
    
    def test_fallback_entity_extraction(self):
        """Teste l'extraction d'entités de repli."""
        # Test avec un texte contenant des entités crypto et des montants
        text = "Bitcoin reached $50,000 while Ethereum increased by 15%"
        entities = self.analyzer._fallback_entity_extraction(text)
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        # Vérification des types d'entités
        entity_types = [entity["type"] for entity in entities]
        self.assertIn("CRYPTO", entity_types)
        self.assertIn("MONEY", entity_types)
        self.assertIn("PERCENTAGE", entity_types)

if __name__ == "__main__":
    unittest.main() 