from datetime import datetime

import pandas as pd

from ai_trading.llm.sentiment_analysis.sentiment_pipeline import SentimentPipeline


class StubNewsAnalyzer:
    def analyze_news(self, items):
        return pd.DataFrame(
            {
                "title": [item["title"] for item in items],
                "global_sentiment": [{"label": "positive", "score": 0.7} for _ in items],
                "published_at": [pd.Timestamp(item["published_at"]) for item in items],
            }
        )


class StubFakeNewsDetector:
    def calculate_credibility_score(self, *_args):
        return 0.8

    def analyze_propagation(self, _news_id, _sharing_data):
        return {"velocity": 0.2, "suspicious_patterns": 0.1}


class StubContextualAnalyzer:
    def analyze_market_context(self, _text, _market_data, _timestamp):
        return {"market_phase": "bull"}

    def detect_sarcasm(self, _text, _market_data):
        return True, 0.75


def test_pipeline_enriches_news_with_credibility_and_context():
    pipeline = SentimentPipeline(
        news_analyzer=StubNewsAnalyzer(),
        fake_news_detector=StubFakeNewsDetector(),
        contextual_analyzer=StubContextualAnalyzer(),
    )
    items = [
        {
            "id": "article-1",
            "title": "Bitcoin adoption grows",
            "body": "Institutional demand rises.",
            "source_url": "https://example.com/article",
            "published_at": datetime(2026, 1, 1),
        }
    ]

    result = pipeline.analyze_news(
        items,
        market_data={"trend": "bullish", "volatility": 0.2},
        sharing_by_news_id={"article-1": []},
        user_behaviors_by_news_id={"article-1": []},
    )

    assert result.loc[0, "credibility_score"] == 0.8
    assert result.loc[0, "propagation"]["suspicious_patterns"] == 0.1
    assert result.loc[0, "context_analysis"]["market_phase"] == "bull"
    assert bool(result.loc[0, "is_sarcastic"])
    assert result.loc[0, "sarcasm_score"] == 0.75
