"""Régressions P4 cache/latence, sans données synthétiques dans le runtime."""

import numpy as np
import pandas as pd

from ai_trading.llm.predictions.market_predictor import MarketPredictor


class DeterministicClient:
    mode = "test_injected_client"

    def complete(self, prompt, context):
        del prompt, context
        return '{"direction":"bullish","confidence":0.7,"factors":[],"contradictions":[]}'


def inputs():
    timestamp = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    close = 100 + np.arange(24) * .2
    return (
        pd.DataFrame({"timestamp": timestamp, "open": close - .1, "high": close + .3, "low": close - .3,
                      "close": close, "volume": 1000, "source": "p1_fixture"}),
        pd.DataFrame({"timestamp": timestamp, "sentiment_score": .1, "quality": .8, "source": "p2_fixture"}),
    )


def test_ttl_is_monotonic_and_cache_invalidation_uses_input_version(tmp_path):
    market, sentiment = inputs()
    predictor = MarketPredictor({"use_gpu": False, "cache_dir": str(tmp_path), "llm_client": DeterministicClient()})
    first = predictor.predict_market_direction("BTC", "1h", market, sentiment, market.timestamp.iloc[-1])
    changed = market.copy()
    changed.loc[changed.index[-1], "close"] += 5
    changed.loc[changed.index[-1], "high"] += 5
    second = predictor.predict_market_direction("BTC", "1h", changed, sentiment, changed.timestamp.iloc[-1])
    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert predictor._get_ttl_for_timeframe("1h") < predictor._get_ttl_for_timeframe("24h")
    assert predictor._get_ttl_for_timeframe("24h") < predictor._get_ttl_for_timeframe("7d")


def test_gpu_is_optional_and_tensor_rt_is_never_claimed_when_disabled(tmp_path):
    predictor = MarketPredictor({"use_gpu": False, "enable_tensorrt": False, "cache_dir": str(tmp_path)})
    assert predictor.rtx_optimizer is None
