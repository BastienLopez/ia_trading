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
            'full_text': "Bitcoin to the moon! 🚀",
            'retweet_count': 150,
            'favorite_count': 300,
            'reply_count': 45,
            'created_at': "2023-03-15T12:00:00Z",
            'user': {
                'screen_name': 'CryptoExpert',
                'followers_count': 15000,
                'verified': True
            }
        }
        
        self.sample_reddit_post = {
            'title': "Ethereum Merge Update",
            'selftext': "The merge was successfully completed...",
            'score': 2500,
            'upvote_ratio': 0.95,
            'num_comments': 450,
            'created_utc': 1663257600,
            'author': {
                'name': 'EthDev',
                'comment_karma': 25000,
                'link_karma': 45000
            }
        }

    def test_twitter_analysis(self):
        df = self.twitter_analyzer.analyze_social_posts([self.sample_tweet])
        self.assertIn('engagement_score', df.columns)
        self.assertFalse(df['engagement_score'].isnull().any())
        self.assertGreaterEqual(df.iloc[0]['engagement_score'], 0)

    def test_reddit_analysis(self):
        df = self.reddit_analyzer.analyze_social_posts([self.sample_reddit_post])
        required_columns = {'sentiment_label', 'sentiment_score', 'engagement_score'}
        self.assertTrue(required_columns.issubset(df.columns))
        self.assertIn(df.iloc[0]['sentiment_label'], ['positive', 'neutral', 'negative'])
        self.assertBetween(df.iloc[0]['sentiment_score'], -1.0, 1.0)

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
        self.assertFalse(df['engagement_score'].isnull().any())

    def assertBetween(self, value, min_val, max_val):
        """Helper pour vérifier les plages de valeurs."""
        self.assertTrue(min_val <= value <= max_val)

if __name__ == '__main__':
    unittest.main() 