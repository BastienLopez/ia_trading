"""Contrats causaux partagés par les prédictions P4.

Ce module est volontairement indépendant des fournisseurs réseau : les données
P1/P2 peuvent être injectées dans les tests et dans les jobs batch, puis sont
validées avant toute inférence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class PredictionInputError(ValueError):
    """Entrée P4 invalide ou non causale."""


def _as_utc_timestamp(value: Optional[Any]) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _timestamp_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise PredictionInputError(f"{label}: aucune donnée disponible")
    result = frame.copy()
    timestamp_column = next(
        (column for column in ("timestamp", "date", "datetime", "as_of") if column in result.columns),
        None,
    )
    if timestamp_column:
        timestamps = pd.to_datetime(result.pop(timestamp_column), utc=True, errors="coerce")
    elif isinstance(result.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(result.index, utc=True, errors="coerce")
    else:
        raise PredictionInputError(f"{label}: horodatage obligatoire")
    if timestamps.isna().any():
        raise PredictionInputError(f"{label}: horodatage invalide")
    result.index = pd.DatetimeIndex(timestamps, name="timestamp")
    if result.index.has_duplicates:
        raise PredictionInputError(f"{label}: horodatages dupliqués")
    return result.sort_index()


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convertit un horizon P4 en durée, sans valeur implicite non traçable."""
    try:
        value, unit = int(timeframe[:-1]), timeframe[-1].lower()
    except (TypeError, ValueError, IndexError) as error:
        raise PredictionInputError("timeframe P4 invalide") from error
    mapping = {"m": "min", "h": "h", "d": "d"}
    if value <= 0 or unit not in mapping:
        raise PredictionInputError("timeframe P4 invalide")
    return pd.to_timedelta(value, unit=mapping[unit])


def _validate_identity(frame: pd.DataFrame, label: str, asset: Optional[str], timeframe: Optional[str]) -> pd.DataFrame:
    result = frame.copy()
    if asset is not None:
        expected_asset = asset.upper()
        if "asset" in result.columns and not result["asset"].dropna().astype(str).str.upper().eq(expected_asset).all():
            raise PredictionInputError(f"{label}: actif incompatible avec la requête")
        result["asset"] = expected_asset
    elif "asset" not in result.columns or result["asset"].replace("", np.nan).isna().any():
        raise PredictionInputError(f"{label}: actif obligatoire")
    if timeframe is not None:
        if "timeframe" in result.columns and not result["timeframe"].dropna().astype(str).eq(timeframe).all():
            raise PredictionInputError(f"{label}: timeframe incompatible avec la requête")
        result["timeframe"] = timeframe
    elif "timeframe" not in result.columns or result["timeframe"].replace("", np.nan).isna().any():
        raise PredictionInputError(f"{label}: timeframe obligatoire")
    return result


def normalize_market_data(market_data: pd.DataFrame, as_of: Optional[Any] = None,
                          asset: Optional[str] = None, timeframe: Optional[str] = None) -> pd.DataFrame:
    """Normalise une série P1 et interdit toute bougie postérieure au cutoff."""
    cutoff = _as_utc_timestamp(as_of)
    market = _validate_identity(_timestamp_index(market_data, "marché"), "marché", asset, timeframe)
    if (market.index > cutoff).any():
        raise PredictionInputError("marché: données futures par rapport à as_of")
    if "close" not in market.columns:
        if "price" not in market.columns:
            raise PredictionInputError("marché: colonne close ou price obligatoire")
        market["close"] = market["price"]
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    if market["close"].isna().any() or (market["close"] <= 0).any():
        raise PredictionInputError("marché: close doit être numérique et positif")
    market["open"] = pd.to_numeric(market.get("open", market["close"].shift(1)), errors="coerce").fillna(market["close"])
    market["high"] = pd.to_numeric(market.get("high", market[["open", "close"]].max(axis=1)), errors="coerce")
    market["low"] = pd.to_numeric(market.get("low", market[["open", "close"]].min(axis=1)), errors="coerce")
    market["volume"] = pd.to_numeric(market.get("volume", 0.0), errors="coerce").fillna(0.0)
    if (market["high"] < market[["open", "close"]].max(axis=1)).any() or (market["low"] > market[["open", "close"]].min(axis=1)).any():
        raise PredictionInputError("marché: OHLC incohérent")
    if "source" not in market.columns or market["source"].replace("", np.nan).isna().any():
        raise PredictionInputError("marché: provenance source obligatoire")
    market["source"] = market["source"].astype(str)
    market["as_of"] = cutoff
    return market.loc[market.index <= cutoff].copy()


def aggregate_sentiment_data(
    sentiment_data: pd.DataFrame, as_of: Optional[Any] = None, asset: Optional[str] = None,
    timeframe: Optional[str] = None, max_age: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Normalise P2 en score continu, provenance et qualité, sans valeur future."""
    cutoff = _as_utc_timestamp(as_of)
    sentiment = _validate_identity(_timestamp_index(sentiment_data, "sentiment"), "sentiment", asset, timeframe)
    if (sentiment.index > cutoff).any():
        raise PredictionInputError("sentiment: données futures par rapport à as_of")
    score_columns = [
        column for column in ("sentiment_score", "compound_score", "global_sentiment_score", "news_sentiment", "social_sentiment")
        if column in sentiment.columns
    ]
    if not score_columns:
        raise PredictionInputError("sentiment: score continu obligatoire")
    scores = sentiment[score_columns].apply(pd.to_numeric, errors="coerce")
    sentiment["sentiment_score"] = scores.mean(axis=1).clip(-1.0, 1.0)
    if sentiment["sentiment_score"].isna().any():
        raise PredictionInputError("sentiment: score invalide")
    if "source" not in sentiment.columns or sentiment["source"].replace("", np.nan).isna().any():
        raise PredictionInputError("sentiment: provenance source obligatoire")
    sentiment["source"] = sentiment["source"].astype(str)
    quality = sentiment["quality"] if "quality" in sentiment.columns else pd.Series(1.0, index=sentiment.index)
    sentiment["quality"] = pd.to_numeric(quality, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    sentiment["as_of"] = cutoff
    latest_timestamp = sentiment.index.max()
    freshness = cutoff - latest_timestamp
    if max_age is not None and freshness > max_age:
        raise PredictionInputError(f"sentiment: observation périmée ({freshness.total_seconds():.0f}s)")
    sentiment["freshness_seconds"] = float(freshness.total_seconds())
    return sentiment[["asset", "timeframe", "sentiment_score", "source", "quality", "freshness_seconds", "as_of"]].copy()


def align_market_and_sentiment(
    market_data: pd.DataFrame,
    sentiment_data: Optional[pd.DataFrame],
    as_of: Optional[Any] = None,
    asset: Optional[str] = None,
    timeframe: Optional[str] = None,
    require_sentiment: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aligne P2 sur P1 avec la dernière observation passée uniquement."""
    market = normalize_market_data(market_data, as_of, asset, timeframe)
    cutoff = _as_utc_timestamp(as_of)
    metadata: Dict[str, Any] = {"as_of": cutoff.isoformat(), "sentiment_available": sentiment_data is not None}
    if sentiment_data is None or sentiment_data.empty:
        if require_sentiment:
            raise PredictionInputError("sentiment: observations P2 horodatées requises")
        market["sentiment_score"] = 0.0
        market["sentiment_quality"] = 0.0
        market["sentiment_freshness_seconds"] = float("inf")
        market["sentiment_source"] = "unavailable"
        return market, metadata
    max_age = min(timeframe_to_timedelta(timeframe or "24h") * 4, pd.Timedelta(days=7))
    sentiment = aggregate_sentiment_data(sentiment_data, cutoff, asset, timeframe, max_age=max_age).rename(
        columns={"source": "sentiment_source", "quality": "sentiment_quality", "freshness_seconds": "sentiment_freshness_seconds"}
    )
    aligned = pd.merge_asof(
        market.reset_index().sort_values("timestamp"),
        sentiment.reset_index().sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("timestamp")
    aligned["sentiment_score"] = aligned["sentiment_score"].fillna(0.0)
    aligned["sentiment_quality"] = aligned["sentiment_quality"].fillna(0.0)
    aligned["sentiment_freshness_seconds"] = aligned["sentiment_freshness_seconds"].fillna(float("inf"))
    aligned["sentiment_source"] = aligned["sentiment_source"].fillna("unavailable")
    metadata["sentiment_observations"] = int(len(sentiment))
    metadata["sentiment_freshness_seconds"] = float(sentiment["sentiment_freshness_seconds"].iloc[-1])
    return aligned, metadata


def input_fingerprint(asset: str, timeframe: str, aligned_data: pd.DataFrame, as_of: Any) -> str:
    """Empreinte stable pour une clé de cache dépendante des entrées réelles."""
    numeric = aligned_data.select_dtypes(include=[np.number]).round(10)
    payload = {
        "asset": asset.upper(),
        "timeframe": timeframe,
        "as_of": _as_utc_timestamp(as_of).isoformat(),
        "index": [timestamp.isoformat() for timestamp in aligned_data.index],
        "values": pd.util.hash_pandas_object(numeric, index=True).astype(str).tolist(),
        "sources": aligned_data.get("source", pd.Series(dtype=str)).astype(str).tolist(),
        "sentiment_sources": aligned_data.get("sentiment_source", pd.Series(dtype=str)).astype(str).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PredictionContext:
    asset: str
    timeframe: str
    as_of: str
    fingerprint: str
    data_sources: Tuple[str, ...]
    sentiment_sources: Tuple[str, ...]
    sentiment_freshness_seconds: float
