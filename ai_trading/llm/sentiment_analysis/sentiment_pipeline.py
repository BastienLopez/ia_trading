"""Pipeline unifié pour enrichir le sentiment par crédibilité et contexte marché."""

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .contextual_analyzer import ContextualAnalyzer
from .enhanced_news_analyzer import EnhancedNewsAnalyzer
from .fake_news_detector import FakeNewsDetector
from .social_analyzer import SocialAnalyzer


class SentimentPipeline:
    """Compose les analyseurs P2 dans une sortie stable et exploitable par le RL."""

    def __init__(
        self,
        news_analyzer: Optional[EnhancedNewsAnalyzer] = None,
        fake_news_detector: Optional[FakeNewsDetector] = None,
        contextual_analyzer: Optional[ContextualAnalyzer] = None,
    ):
        self.news_analyzer = news_analyzer or EnhancedNewsAnalyzer()
        self.fake_news_detector = fake_news_detector or FakeNewsDetector()
        self.contextual_analyzer = contextual_analyzer or ContextualAnalyzer()

    def analyze_news(
        self,
        news_items: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
        sharing_by_news_id: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        user_behaviors_by_news_id: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> pd.DataFrame:
        """Analyse des actualités et ajoute crédibilité, propagation et contexte."""
        results = self.news_analyzer.analyze_news(news_items).copy()
        if results.empty:
            return results

        sharing_by_news_id = sharing_by_news_id or {}
        user_behaviors_by_news_id = user_behaviors_by_news_id or {}
        enrichments = [
            self._enrich_news(
                item,
                market_data,
                sharing_by_news_id.get(self._news_id(item), []),
                user_behaviors_by_news_id.get(self._news_id(item), []),
            )
            for item in news_items
        ]
        return pd.concat([results.reset_index(drop=True), pd.DataFrame(enrichments)], axis=1)

    def analyze_social_posts(
        self,
        posts: List[Dict[str, Any]],
        platform: str = "twitter",
        market_data: Optional[Dict[str, Any]] = None,
        analyzer: Optional[SocialAnalyzer] = None,
    ) -> pd.DataFrame:
        """Analyse les posts sociaux et ajoute le contexte/sarcasme à chaque sortie."""
        social_analyzer = analyzer or SocialAnalyzer(platform=platform)
        results = social_analyzer.analyze_social_posts(posts).copy()
        if results.empty or market_data is None:
            return results

        context = [
            self._context_for_text(
                str(post.get("text") or post.get("full_text") or post.get("selftext") or ""),
                post.get("created_at") or post.get("created_utc"),
                market_data,
            )
            for post in posts
        ]
        return pd.concat([results.reset_index(drop=True), pd.DataFrame(context)], axis=1)

    def _enrich_news(
        self,
        item: Dict[str, Any],
        market_data: Optional[Dict[str, Any]],
        sharing_data: List[Dict[str, Any]],
        user_behaviors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        content = " ".join(
            str(item.get(field, "")) for field in ("title", "body") if item.get(field)
        )
        source_url = str(item.get("source_url") or item.get("url") or "")
        credibility = self.fake_news_detector.calculate_credibility_score(
            content, source_url, sharing_data, user_behaviors
        )
        propagation = self.fake_news_detector.analyze_propagation(
            self._news_id(item), sharing_data
        )
        context = self._context_for_text(content, item.get("published_at"), market_data)
        return {
            "credibility_score": credibility,
            "propagation": propagation,
            **context,
        }

    def _context_for_text(
        self, text: str, timestamp: Any, market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if market_data is None:
            return {"context_analysis": None, "is_sarcastic": False, "sarcasm_score": 0.0}
        parsed_timestamp = pd.to_datetime(timestamp, errors="coerce")
        timestamp_value = (
            parsed_timestamp.to_pydatetime() if not pd.isna(parsed_timestamp) else pd.Timestamp.now().to_pydatetime()
        )
        context = self.contextual_analyzer.analyze_market_context(
            text, market_data, timestamp_value
        )
        is_sarcastic, sarcasm_score = self.contextual_analyzer.detect_sarcasm(
            text, market_data
        )
        return {
            "context_analysis": context,
            "is_sarcastic": is_sarcastic,
            "sarcasm_score": sarcasm_score,
        }

    @staticmethod
    def _news_id(item: Dict[str, Any]) -> str:
        return str(item.get("id") or item.get("url") or item.get("title") or "unknown")
