import unittest
from datetime import datetime, timedelta

from ai_trading.llm.sentiment_analysis.contextual_analyzer import ContextualAnalyzer


class TestContextualAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ContextualAnalyzer()
        self.sample_text = "Le prix du Bitcoin montre une forte tendance haussière."
        self.sample_market_data = {
            "price": 50000,
            "volume": 1000000,
            "volatility": 0.15,
            "trend": "bullish",
        }
        self.sample_entities = [
            {
                "name": "Bitcoin",
                "type": "cryptocurrency",
                "attributes": {"market_cap": 1000000000000},
            },
            {
                "name": "Ethereum",
                "type": "cryptocurrency",
                "attributes": {"market_cap": 500000000000},
            },
        ]
        self.sample_texts = [
            {
                "text": "Le marché crypto est très optimiste aujourd'hui!",
                "timestamp": datetime.now() - timedelta(hours=3),
                "sentiment": 0.8,
            },
            {
                "text": "Les investisseurs restent prudents malgré la hausse.",
                "timestamp": datetime.now() - timedelta(hours=2),
                "sentiment": 0.3,
            },
            {
                "text": "Nouvelle ATH pour Bitcoin!",
                "timestamp": datetime.now() - timedelta(hours=1),
                "sentiment": 0.9,
            },
        ]

    def test_analyze_market_context(self):
        context = self.analyzer.analyze_market_context(
            self.sample_text, self.sample_market_data, datetime.now()
        )
        self.assertIn("market_phase", context)
        self.assertIn("volatility_impact", context)
        self.assertIn("trend_alignment", context)
        self.assertIn("sentiment_strength", context)

    def test_analyze_entity_relations(self):
        relations = self.analyzer.analyze_entity_relations(
            self.sample_text, self.sample_entities
        )
        self.assertIn("entity_importance", relations)
        self.assertIn("entity_communities", relations)
        self.assertIn("relation_strength", relations)
        self.assertIn("influence_flow", relations)

    def test_detect_sarcasm(self):
        sarcastic_text = "Bien sûr, Bitcoin va atteindre 1 million demain... 🙄"
        context = {"market_trend": "bearish", "recent_events": ["crash", "fud"]}

        is_sarcastic, confidence = self.analyzer.detect_sarcasm(sarcastic_text, context)
        self.assertIsInstance(is_sarcastic, bool)
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)

    def test_analyze_temporal_sentiment(self):
        time_window = timedelta(hours=4)
        temporal_analysis = self.analyzer.analyze_temporal_sentiment(
            self.sample_texts, time_window
        )

        self.assertIn("overall_trend", temporal_analysis)
        self.assertIn("volatility", temporal_analysis)
        self.assertIn("momentum", temporal_analysis)
        self.assertIn("breakpoints", temporal_analysis)

    def test_sentiment_trend_consistency(self):
        # Test avec une série de sentiments clairement positifs
        positive_texts = [
            {
                "text": f"Très positif {i}",
                "timestamp": datetime.now() - timedelta(hours=3 - i),
                "sentiment": 0.8 + i / 10,
            }
            for i in range(3)
        ]

        analysis = self.analyzer.analyze_temporal_sentiment(
            positive_texts, timedelta(hours=4)
        )
        self.assertGreater(analysis["overall_trend"]["slope"], 0)

    def test_sarcasm_detection_with_context(self):
        # Test de détection de sarcasme avec contexte contradictoire
        bullish_context = {
            "market_trend": "bullish",
            "recent_events": ["ath", "institutional_adoption"],
        }

        bearish_text = "Oh oui, c'est vraiment le meilleur moment pour acheter... 🙄"
        is_sarcastic, confidence = self.analyzer.detect_sarcasm(
            bearish_text, bullish_context
        )
        self.assertTrue(is_sarcastic)
        self.assertGreater(confidence, 0.7)


if __name__ == "__main__":
    unittest.main()
