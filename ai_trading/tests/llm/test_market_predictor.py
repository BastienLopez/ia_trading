"""Contrats P4 de MarketPredictor : données causales et LLM injectable."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from ai_trading.llm.predictions.market_predictor import MarketPredictor, P2SentimentDataProvider
from ai_trading.llm.predictions.prediction_contract import PredictionInputError, align_market_and_sentiment
from ai_trading.llm.sentiment_analysis.sentiment_pipeline import SentimentPipeline


def market_frame(periods=48):
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    close = 100 + np.linspace(0, 6, periods)
    return pd.DataFrame({"timestamp": timestamps, "open": close - .2, "high": close + .5,
                         "low": close - .5, "close": close, "volume": 1000,
                         "source": "p1_fixture", "asset": "BTC", "timeframe": "1h"})


def sentiment_frame(periods=48):
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "sentiment_score": np.linspace(-.1, .3, periods),
                         "source": "p2_fixture", "quality": .9, "asset": "BTC"})


class InvalidClient:
    def complete(self, prompt, context):
        del prompt, context
        return "not-json"


class DeterministicClient:
    mode = "test_injected_client"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, context):
        del prompt, context
        self.calls += 1
        return json.dumps({"direction": "bullish", "confidence": 0.7, "factors": [], "contradictions": []})


class SlowClient:
    mode = "slow_test_client"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, context):
        del prompt, context
        self.calls += 1
        time.sleep(.05)
        return '{}'


def predictor(tmp_path, client=None):
    return MarketPredictor({"llm_client": client or DeterministicClient(), "cache_dir": str(tmp_path), "use_gpu": False})


def test_offline_prediction_is_timestamped_and_cached(tmp_path):
    market, sentiment = market_frame(), sentiment_frame()
    service = predictor(tmp_path)
    first = service.predict_market_direction("BTC", "1h", market, sentiment, market.timestamp.iloc[-1])
    second = service.predict_market_direction("BTC", "1h", market, sentiment, market.timestamp.iloc[-1])
    assert first["mode"] == "test_injected_client"
    assert first["as_of"] == market.timestamp.iloc[-1].isoformat()
    assert 0 <= first["confidence"] <= 1
    assert first["sentiment_sources"] == ["p2_fixture"]
    assert first["sentiment_freshness_seconds"] == 0.0
    assert first["data_version"] == first["input_fingerprint"]
    assert second["id"] == first["id"]
    assert second["input_fingerprint"] == first["input_fingerprint"]


def test_invalid_llm_response_is_degraded_after_bounded_retries(tmp_path):
    result = predictor(tmp_path, InvalidClient()).predict_market_direction(
        "BTC", "1h", market_frame(), sentiment_frame(), "2025-01-02T23:00:00Z"
    )
    assert result["mode"] == "degraded"
    assert result["abstain"] is True
    assert result["error"]["type"] == "RuntimeError"


def test_future_observation_is_rejected_without_silent_alignment(tmp_path):
    market, sentiment = market_frame(), sentiment_frame()
    market.loc[market.index[-1], "timestamp"] = pd.Timestamp("2025-02-01", tz="UTC")
    result = predictor(tmp_path).predict_market_direction("BTC", "1h", market, sentiment, "2025-01-02T23:00:00Z")
    assert result["mode"] == "degraded"
    assert result["abstain"] is True
    assert "future" in result["error"]["message"].lower()


def test_p2_requires_identity_source_and_freshness():
    market, sentiment = market_frame(), sentiment_frame()
    with pytest.raises(PredictionInputError, match="observation périmée"):
        align_market_and_sentiment(market, sentiment.iloc[:1], market.timestamp.iloc[-1], asset="BTC", timeframe="1h")
    without_source = sentiment.drop(columns="source")
    with pytest.raises(PredictionInputError, match="provenance source obligatoire"):
        align_market_and_sentiment(market, without_source, market.timestamp.iloc[-1], asset="BTC", timeframe="1h")
    wrong_asset = sentiment.assign(asset="ETH")
    with pytest.raises(PredictionInputError, match="actif incompatible"):
        align_market_and_sentiment(market, wrong_asset, market.timestamp.iloc[-1], asset="BTC", timeframe="1h")


def test_llm_parser_rejects_text_confidence_and_disabled_client(tmp_path):
    market, sentiment = market_frame(), sentiment_frame()
    service = predictor(tmp_path)
    parsed = service._parse_prediction('{"direction":"bullish","confidence":"high","factors":[]}', "BTC", "1h")
    assert "error" in parsed
    disabled = MarketPredictor({"cache_dir": str(tmp_path / "disabled"), "use_gpu": False})
    result = disabled.predict_market_direction("BTC", "1h", market, sentiment, market.timestamp.iloc[-1])
    assert result["mode"] == "degraded"
    assert result["abstain"] is True


def test_p2_persisted_provider_filters_asset_timeframe_and_cutoff(tmp_path):
    market, sentiment = market_frame(), sentiment_frame()
    observations_path = tmp_path / "p2.csv"
    sentiment.assign(timeframe="1h").to_csv(observations_path, index=False)
    provider = P2SentimentDataProvider(observations_path)
    loaded = provider.fetch("BTC", "1h", market.timestamp.iloc[-1])
    assert len(loaded) == len(sentiment)
    service = MarketPredictor({"llm_client": DeterministicClient(), "p2_observations_path": str(observations_path),
                               "cache_dir": str(tmp_path / "cache"), "use_gpu": False})
    result = service.predict_market_direction("BTC", "1h", market, as_of=market.timestamp.iloc[-1])
    assert result["sentiment_sources"] == ["p2_fixture"]


def test_p2_pipeline_persists_the_contract_consumed_by_p4(tmp_path):
    market, _ = market_frame(), sentiment_frame()
    analyses = pd.DataFrame({"published_at": market.timestamp.iloc[-2:], "compound_score": [.2, .4],
                             "credibility_score": [.7, .9]})
    observations = SentimentPipeline.to_p4_observations(analyses, "BTC", "1h", "p2_news")
    path = SentimentPipeline.persist_p4_observations(observations, tmp_path / "observations.csv")
    loaded = P2SentimentDataProvider(path).fetch("BTC", "1h", market.timestamp.iloc[-1])
    assert set(loaded.columns) >= {"timestamp", "asset", "timeframe", "sentiment_score", "quality", "source"}
    assert len(loaded) == 2


def test_injected_llm_timeout_is_bounded_and_traceable(tmp_path):
    market, sentiment = market_frame(), sentiment_frame()
    slow = SlowClient()
    service = MarketPredictor({"llm_client": slow, "llm_timeout_seconds": .01, "cache_dir": str(tmp_path), "use_gpu": False})
    result = service.predict_market_direction("BTC", "1h", market, sentiment, market.timestamp.iloc[-1])
    assert result["mode"] == "degraded"
    assert result["error"]["type"] == "RuntimeError"
    assert slow.calls == 3


def test_concurrent_identical_requests_keep_one_contract_fingerprint(tmp_path):
    market, sentiment = market_frame(), sentiment_frame()
    client = DeterministicClient()
    service = predictor(tmp_path, client)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _: service.predict_market_direction("BTC", "1h", market, sentiment, "2025-01-02T23:00:00Z"), range(4)
        ))
    assert len({item["input_fingerprint"] for item in results}) == 1
    assert all(item["mode"] == "test_injected_client" for item in results)
    assert client.calls == 1


def test_input_change_and_stream_error_invalidate_asset_cache(tmp_path):
    market, sentiment = market_frame(), sentiment_frame()
    service = predictor(tmp_path)
    first = service.predict_market_direction("BTC", "1h", market, sentiment, market.timestamp.iloc[-1])
    changed = market.copy()
    changed.loc[changed.index[-1], "close"] += 5
    changed.loc[changed.index[-1], "high"] += 5
    second = service.predict_market_direction("BTC", "1h", changed, sentiment, changed.timestamp.iloc[-1])
    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert list(service.cache.memory_cache) == [
        f"p4:v2:BTC:1h:{second['input_fingerprint']}"
    ]

    future = changed.copy()
    future.loc[future.index[-1], "timestamp"] = pd.Timestamp("2025-02-01", tz="UTC")
    failed = service.predict_market_direction("BTC", "1h", future, sentiment, "2025-01-02T23:00:00Z")
    assert failed["mode"] == "degraded"
    assert not service.cache.memory_cache
