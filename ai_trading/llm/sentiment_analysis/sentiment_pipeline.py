"""Pipeline unifié pour enrichir le sentiment par crédibilité et contexte marché."""

from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path

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

    @staticmethod
    def to_p4_observations(analyses: pd.DataFrame, asset: str, timeframe: str,
                           source: str, timestamp_column: str = "timestamp") -> pd.DataFrame:
        """Convertit une sortie P2 en observations causales persistables pour P4."""
        if analyses.empty:
            raise ValueError("P2: aucune analyse à convertir")
        if not asset or not timeframe or not source:
            raise ValueError("P2: actif, timeframe et source sont obligatoires")
        timestamp_col = next((column for column in (timestamp_column, "published_at", "created_at", "date")
                              if column in analyses.columns), None)
        score_col = next((column for column in ("sentiment_score", "global_sentiment_score", "compound_score", "score")
                          if column in analyses.columns), None)
        if timestamp_col is None or score_col is None:
            raise ValueError("P2: timestamp et score continu obligatoires pour P4")
        timestamps = pd.to_datetime(analyses[timestamp_col], utc=True, errors="coerce")
        scores = pd.to_numeric(analyses[score_col], errors="coerce")
        if timestamps.isna().any() or scores.isna().any():
            raise ValueError("P2: timestamp ou score invalide pour P4")
        quality = analyses.get("quality", analyses.get("credibility_score", pd.Series(1.0, index=analyses.index)))
        observations = pd.DataFrame({
            "timestamp": timestamps,
            "asset": asset.upper(),
            "timeframe": timeframe,
            "sentiment_score": scores.clip(-1.0, 1.0),
            "quality": pd.to_numeric(quality, errors="coerce").fillna(0.0).clip(0.0, 1.0),
            "source": source,
        })
        return observations.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def persist_p4_observations(observations: pd.DataFrame, path: Any) -> Path:
        """Persiste atomiquement les observations P2 consommées par P4."""
        target = Path(path)
        if target.suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("P2: sortie P4 attendue en .csv ou .parquet")
        required = {"timestamp", "asset", "timeframe", "sentiment_score", "quality", "source"}
        if not required.issubset(observations.columns):
            raise ValueError("P2: contrat d'observations P4 incomplet")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = pd.read_parquet(target) if target.suffix.lower() == ".parquet" else pd.read_csv(target)
            combined = pd.concat([existing, observations], ignore_index=True)
        else:
            combined = observations.copy()
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="raise")
        combined = combined.drop_duplicates(["timestamp", "asset", "timeframe", "source"], keep="last")
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        if target.suffix.lower() == ".parquet":
            combined.to_parquet(temporary, index=False)
        else:
            combined.to_csv(temporary, index=False)
        temporary.replace(target)
        return target

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
