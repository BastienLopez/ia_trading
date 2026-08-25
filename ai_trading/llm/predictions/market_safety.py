"""Gardes causales P4 pour régimes instables et données de flux dégradées."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class MarketSafetyGuard:
    """Dégrade une prédiction, sans jamais générer un ordre ou une allocation."""

    def __init__(self, jump_threshold: float = 0.08, volume_z_threshold: float = 4.0,
                 max_lag_seconds: float = 300.0):
        self.jump_threshold = float(jump_threshold)
        self.volume_z_threshold = float(volume_z_threshold)
        self.max_lag_seconds = float(max_lag_seconds)

    def assess(self, market_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None,
               as_of: Optional[Any] = None) -> Dict[str, Any]:
        if market_data is None or len(market_data) < 2:
            return {"unstable": True, "confidence_multiplier": 0.0, "reasons": ["insufficient_market_data"]}
        timestamp_column = next((name for name in ("timestamp", "date", "datetime") if name in market_data), None)
        if timestamp_column is None and not isinstance(market_data.index, pd.DatetimeIndex):
            return {"unstable": True, "confidence_multiplier": 0.0, "reasons": ["missing_market_timestamp"]}
        timestamps = (pd.to_datetime(market_data[timestamp_column], utc=True, errors="coerce")
                      if timestamp_column else pd.Series(pd.to_datetime(market_data.index, utc=True), index=market_data.index))
        if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            return {"unstable": True, "confidence_multiplier": 0.0, "reasons": ["invalid_market_timestamps"]}
        cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        if timestamps.iloc[-1] > cutoff:
            return {"unstable": True, "confidence_multiplier": 0.0, "reasons": ["future_market_observation"]}
        reasons = []
        close = pd.to_numeric(market_data["close"], errors="coerce")
        if close.isna().any() or (close <= 0).any():
            reasons.append("invalid_price")
        else:
            jump = float(abs(close.pct_change().iloc[-1]))
            if jump >= self.jump_threshold:
                reasons.append("price_jump")
        if "volume" in market_data:
            volume = pd.to_numeric(market_data["volume"], errors="coerce")
            baseline = volume.iloc[:-1]
            if baseline.notna().sum() >= 5 and float(baseline.std(ddof=0)) > 0:
                z_score = abs(float((volume.iloc[-1] - baseline.mean()) / baseline.std(ddof=0)))
                if z_score >= self.volume_z_threshold:
                    reasons.append("abnormal_volume")
            elif baseline.notna().sum() >= 5 and float(volume.iloc[-1]) > float(baseline.mean()) * 3:
                reasons.append("abnormal_volume")
        if sentiment_data is not None and not sentiment_data.empty and "sentiment_score" in sentiment_data:
            sentiment = pd.to_numeric(sentiment_data["sentiment_score"], errors="coerce").dropna()
            if len(sentiment) >= 2 and close.notna().sum() >= 2:
                if np.sign(close.iloc[-1] - close.iloc[-2]) * np.sign(sentiment.iloc[-1] - sentiment.iloc[-2]) < 0:
                    reasons.append("sentiment_price_divergence")
        multiplier = 0.0 if any(reason in reasons for reason in ("invalid_price", "future_market_observation")) else (0.35 if reasons else 1.0)
        return {"unstable": bool(reasons), "confidence_multiplier": multiplier, "reasons": reasons,
                "trading_enabled": False}

    def apply(self, prediction: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(prediction)
        result["market_safety"] = assessment
        result["confidence"] = float(np.clip(float(result.get("confidence", 0.0)) * assessment["confidence_multiplier"], 0, 1))
        if assessment["unstable"]:
            result.update({"abstain": True, "direction": "neutral"})
        result["trading_enabled"] = False
        return result
