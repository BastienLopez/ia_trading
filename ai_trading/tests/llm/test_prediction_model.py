"""P0 : entraînement ML-only, calibration temporelle, persistance et cache frais."""

import numpy as np
import pandas as pd
import time

from ai_trading.llm.predictions.cache_manager import CacheManager
from ai_trading.llm.predictions.prediction_model import PredictionModel


def frames(periods=96):
    timestamp = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    close = 100 + np.sin(np.arange(periods) / 2) * 2 + np.arange(periods) * .03
    market = pd.DataFrame({"timestamp": timestamp, "open": close - .2, "high": close + .4,
                           "low": close - .4, "close": close, "volume": 1000 + np.arange(periods),
                           "source": "p1_fixture", "asset": "BTC", "timeframe": "1h"})
    sentiment = pd.DataFrame({"timestamp": timestamp, "sentiment_score": np.sin(np.arange(periods) / 3),
                              "quality": .9, "source": "p2_fixture", "asset": "BTC"})
    return market, sentiment


def test_ml_only_training_inference_and_persistence(tmp_path):
    market, sentiment = frames()
    configuration = {"use_gpu": False, "prediction_mode": "ml_only", "min_prediction_confidence": 0.0,
                     "model_dir": str(tmp_path / "models"), "cache_dir": str(tmp_path / "cache")}
    model = PredictionModel(configuration)
    metrics = model.train(market, sentiment, market.timestamp.iloc[-1])
    result = model.predict("BTC", "1h", market, sentiment, market.timestamp.iloc[-1])
    path = tmp_path / "hybrid.joblib"
    model.save_model(str(path))
    loaded = PredictionModel(configuration)
    loaded.load_model(str(path))
    assert metrics["validation_type"] == "timeseries_last_fold"
    assert metrics["n_models"] == 2
    assert result["hybrid_status"] == "ml_only_ready"
    assert result["mode"] == "ml_only"
    assert result["trading_enabled"] is False
    assert set(result["probabilities"]) == {"bearish", "neutral", "bullish"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-9
    assert loaded.feature_columns == model.feature_columns
    assert set(metrics["calibration"]) == {"validation", "test", "calibration_source"}
    assert "abstention_rate" in metrics["calibration"]["test"]
    assert "mean_interval_width" in metrics["calibration"]["test"]


def test_cache_survives_restart_and_prefix_invalidation(tmp_path):
    location = str(tmp_path / "cache")
    first = CacheManager(capacity=4, ttl=60, persist_path=location, enable_disk_cache=True,
                         enable_predictive_loading=False)
    first.set("p4:v2:BTC:1h:abc", {"value": 1})
    restarted = CacheManager(capacity=4, ttl=60, persist_path=location, enable_disk_cache=True,
                             enable_predictive_loading=False)
    assert restarted.get("p4:v2:BTC:1h:abc") == {"value": 1}
    assert restarted.invalidate_prefix("p4:v2:BTC") == 1
    assert restarted.get("p4:v2:BTC:1h:abc") is None
    after_invalidation = CacheManager(capacity=4, ttl=60, persist_path=location, enable_disk_cache=True,
                                      enable_predictive_loading=False)
    assert after_invalidation.get("p4:v2:BTC:1h:abc") is None


def test_cache_ttl_expires_from_memory_and_disk(tmp_path):
    location = str(tmp_path / "ttl-cache")
    cache = CacheManager(capacity=4, ttl=.01, persist_path=location, enable_disk_cache=True,
                         enable_predictive_loading=False)
    cache.set("p4:v2:BTC:1h:ttl", {"value": 1}, ttl=.01)
    time.sleep(.03)
    assert cache.get("p4:v2:BTC:1h:ttl") is None
    restarted = CacheManager(capacity=4, ttl=60, persist_path=location, enable_disk_cache=True,
                             enable_predictive_loading=False)
    assert restarted.get("p4:v2:BTC:1h:ttl") is None
