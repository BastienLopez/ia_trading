"""Tests d'intégration pour le pipeline complet de sentiment."""

import unittest
import pandas as pd
from datetime import datetime, timedelta

from ai_trading.llm.sentiment_analysis import (
    EnhancedNewsAnalyzer,
    SocialAnalyzer
)

class TestSentimentIntegration(unittest.TestCase):
    def setUp(self):
        self.news_analyzer = EnhancedNewsAnalyzer()
        self.social_analyzer = SocialAnalyzer(platform='twitter')
        
        self.sample_news = [
            {
                "title": "Bitcoin New High",
                "body": "Bitcoin reaches $100k",
                "published_at": datetime.now().isoformat()
            }
        ]
        
        self.sample_tweets = [
            {
                "text": "BTC to the moon!",
                "retweet_count": 1000,
                "created_at": datetime.now().isoformat()
            }
        ]
    
    def test_full_news_analysis(self):
        df = self.news_analyzer.analyze_news(self.sample_news)
        self.assertIn('sentiment_score', df.columns)
        self.assertFalse(df.empty)
        
        report = self.news_analyzer.generate_report(df)
        self.assertIn('top_cryptos', report)
        
        self.news_analyzer.plot_trends(df, "test_plot.png")
    
    def test_social_analysis_integration(self):
        df = self.social_analyzer.analyze_posts(self.sample_tweets)
        enhanced_df = self.social_analyzer.calculate_virality(df)
        
        self.assertIn('viral_risk', enhanced_df.columns)
        self.assertTrue(enhanced_df['engagement'].between(0, 1).all())

if __name__ == '__main__':
    unittest.main() 