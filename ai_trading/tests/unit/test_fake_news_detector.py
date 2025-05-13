import unittest
from datetime import datetime

from ai_trading.llm.sentiment_analysis.fake_news_detector import FakeNewsDetector


class TestFakeNewsDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FakeNewsDetector()
        self.sample_news_content = (
            "Bitcoin atteint un nouveau record historique selon des experts."
        )
        self.sample_source_url = "https://crypto-news.example.com/article123"
        self.sample_sharing_data = [
            {"from_user": "user1", "to_user": "user2", "timestamp": datetime.now()},
            {"from_user": "user2", "to_user": "user3", "timestamp": datetime.now()},
        ]
        self.sample_user_behaviors = [
            {
                "user_id": "user1",
                "post_frequency": 10,
                "avg_sharing_delay": 60,
                "content_similarity": 0.8,
            },
            {
                "user_id": "user2",
                "post_frequency": 1000,
                "avg_sharing_delay": 1,
                "content_similarity": 0.95,
            },
        ]

    def test_verify_source_credibility(self):
        credibility = self.detector.verify_source_credibility(
            self.sample_source_url, self.sample_news_content
        )
        self.assertIsInstance(credibility, float)
        self.assertTrue(0 <= credibility <= 1)

    def test_analyze_propagation(self):
        propagation = self.detector.analyze_propagation(
            self.sample_news_content, self.sample_sharing_data
        )
        self.assertIn("velocity", propagation)
        self.assertIn("centrality", propagation)
        self.assertIn("suspicious_patterns", propagation)

    def test_detect_bots(self):
        bot_predictions = self.detector.detect_bots(self.sample_user_behaviors)
        self.assertEqual(len(bot_predictions), len(self.sample_user_behaviors))
        self.assertTrue(all(isinstance(pred, bool) for pred in bot_predictions))

    def test_calculate_credibility_score(self):
        score = self.detector.calculate_credibility_score(
            self.sample_news_content,
            self.sample_source_url,
            self.sample_sharing_data,
            self.sample_user_behaviors,
        )
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)

    def test_credibility_score_with_obvious_fake(self):
        fake_news_content = (
            "URGENT: Tous les exchanges crypto ferment définitivement!!!"
        )
        fake_source_url = "https://crypto-scam-news.example.com/urgent"
        fake_sharing_data = [
            {
                "from_user": f"bot{i}",
                "to_user": f"bot{i+1}",
                "timestamp": datetime.now(),
            }
            for i in range(10)
        ]
        fake_user_behaviors = [
            {
                "user_id": f"bot{i}",
                "post_frequency": 1000,
                "avg_sharing_delay": 1,
                "content_similarity": 0.99,
            }
            for i in range(10)
        ]

        score = self.detector.calculate_credibility_score(
            fake_news_content, fake_source_url, fake_sharing_data, fake_user_behaviors
        )
        self.assertLess(
            score, 0.3
        )  # Score devrait être bas pour une fake news évidente


if __name__ == "__main__":
    unittest.main()
