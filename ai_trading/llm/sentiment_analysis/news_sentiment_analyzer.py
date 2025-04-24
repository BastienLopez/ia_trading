import os

from ai_trading.config import SENTIMENT_CACHE_DIR, VISUALIZATION_DIR


class NewsSentimentAnalyzer:
    def analyze_sentiment(self, text):
        # ... existing code ...
        cache_path = os.path.join(
            SENTIMENT_CACHE_DIR, f"{self.symbol}_sentiment_cache.json"
        )
        # ... existing code ...

    def plot_sentiment_trends(self):
        # ... existing code ...
        output_path = os.path.join(
            VISUALIZATION_DIR, f"{self.symbol}_sentiment_trends.png"
        )
        # ... existing code ...
