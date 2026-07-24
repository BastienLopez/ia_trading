"""Multi-horizons P4 sans collecte fictive ni appel réseau."""

import numpy as np
import pandas as pd

from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor


class DeterministicClient:
    mode = "test_injected_client"

    def complete(self, prompt, context):
        del prompt, context
        return '{"direction":"bullish","confidence":0.7,"factors":[],"contradictions":[]}'


def frames():
    timestamp = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    close = 100 + np.linspace(0, 4, len(timestamp))
    return (
        pd.DataFrame({"timestamp": timestamp, "open": close - .2, "high": close + .5, "low": close - .5,
                      "close": close, "volume": 1000, "source": "p1_fixture"}),
        pd.DataFrame({"timestamp": timestamp, "sentiment_score": .2, "quality": .9, "source": "p2_fixture"}),
    )


def test_horizon_predictions_are_timestamped_with_injected_client(tmp_path):
    market, sentiment = frames()
    service = MultiHorizonPredictor(model_save_dir=str(tmp_path), use_hybrid=False,
                                    custom_config={"use_gpu": False, "cache_dir": str(tmp_path / "cache"),
                                                   "llm_client": DeterministicClient()})
    result = service.predict_all_horizons("BTC", short_term=True, medium_term=False, long_term=False,
                                          market_data=market, sentiment_data=sentiment,
                                          as_of=market.timestamp.iloc[-1])
    assert set(result) == set(service.SHORT_TERM)
    assert all(item["mode"] == "test_injected_client" for item in result.values())
    assert all(0 <= item["confidence"] <= 1 for item in result.values())


def test_consistency_is_numeric_and_signal_never_enables_trading(tmp_path):
    service = MultiHorizonPredictor(model_save_dir=str(tmp_path), use_hybrid=False,
                                    custom_config={"use_gpu": False, "cache_dir": str(tmp_path / "cache")})
    predictions = {
        **{timeframe: {"direction": "bullish", "confidence": .75} for timeframe in service.SHORT_TERM},
        **{timeframe: {"direction": "neutral", "confidence": .50} for timeframe in service.MEDIUM_TERM},
        **{timeframe: {"direction": "bearish", "confidence": .60} for timeframe in service.LONG_TERM},
    }
    analysis = service.analyze_consistency(predictions)
    assert isinstance(analysis["horizon_analysis"]["short_term"]["confidence"], float)
    assert analysis["trading_signals"]["trading_enabled"] is False
