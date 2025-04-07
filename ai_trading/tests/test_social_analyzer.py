"""
Tests pour l'analyseur de réseaux sociaux.
"""

import unittest
import pandas as pd
from datetime import datetime
from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer

class TestSocialAnalyzer(unittest.TestCase):
    def setUp(self):
        self.twitter_analyzer = SocialAnalyzer(platform='twitter')
        self.reddit_analyzer = SocialAnalyzer(platform='reddit')
        
        self.sample_tweet = {
            "full_text": "Bitcoin breaking $60k! 🎉 #BTC #bullish",
            "retweet_count": 100,
            "favorite_count": 250
        }
        
        self.sample_reddit_post = {
            "body": "Ethereum 2.0 upgrade is a game changer!",
            "score": 500,
            "num_comments": 150
        }

    def test_twitter_analysis(self):
        df = self.twitter_analyzer.analyze_social_posts([self.sample_tweet])
        self.assertIn('engagement_score', df.columns)
        self.assertGreater(df.iloc[0]['engagement_score'], 0)

    def test_reddit_analysis(self):
        df = self.reddit_analyzer.analyze_social_posts([self.sample_reddit_post])
        self.assertIn('sentiment_label', df.columns)
        
    def test_hashtag_extraction(self):
        hashtags = self.twitter_analyzer._extract_hashtags("#BTC and #ETH are pumping!")
        self.assertCountEqual(hashtags, ['btc', 'eth'])

    def test_empty_data(self):
        """Test avec des données vides."""
        empty_df = self.twitter_analyzer.analyze_social_posts([])
        self.assertTrue(empty_df.empty)

    def test_metric_normalization(self):
        """Vérifie la normalisation des métriques."""
        df = self.twitter_analyzer.analyze_social_posts([self.sample_tweet])
        self.assertBetween(df.iloc[0]['retweet_count_norm'], 0.0, 1.0)

    def assertBetween(self, value, min_val, max_val):
        """Helper pour vérifier les plages de valeurs."""
        self.assertTrue(min_val <= value <= max_val)

if __name__ == '__main__':
    unittest.main() 