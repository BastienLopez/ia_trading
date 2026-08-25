"""Contrats P4 P1 : horizons, flux, garde, artefacts, API et feature RL."""

import asyncio
import threading
import time

import numpy as np
import pandas as pd

from ai_trading.api import get_app
from ai_trading.dashboard.p4_readonly import p4_snapshot
from ai_trading.llm.predictions.market_safety import MarketSafetyGuard
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor
from ai_trading.llm.predictions.p4_registry import registry
from ai_trading.llm.predictions.prediction_explainer import PredictionExplainer
from ai_trading.llm.predictions.prediction_contract import aggregate_sentiment_data
from ai_trading.llm.predictions.realtime_stream import P1PollingSource, P1StreamConsumer
from ai_trading.rl.data_integration import RLDataIntegrator


def market(periods=48):
    timestamp = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    close = 100 + np.arange(periods) * .1
    return pd.DataFrame({"timestamp": timestamp, "open": close - .1, "high": close + .2,
                         "low": close - .2, "close": close, "volume": 1000., "source": "p1_test"})


def test_market_guard_abstains_on_jump_volume_and_divergence():
    frame = market()
    frame.loc[frame.index[-1], ["close", "high", "volume"]] = [130., 131., 100000.]
    sentiment = pd.DataFrame({"sentiment_score": [.4, -.4]})
    guard = MarketSafetyGuard(jump_threshold=.05, volume_z_threshold=3.)
    assessment = guard.assess(frame, sentiment, frame.timestamp.iloc[-1])
    result = guard.apply({"direction": "bullish", "confidence": .8}, assessment)
    assert assessment["unstable"] is True
    assert {"price_jump", "abnormal_volume"}.issubset(assessment["reasons"])
    assert result["abstain"] is True and result["direction"] == "neutral"
    assert result["trading_enabled"] is False


def test_p1_stream_has_backpressure_reconnect_and_metrics():
    calls, handled = {"count": 0}, []
    processed = threading.Event()

    class BrokenSource:
        def receive(self, timeout):
            del timeout
            raise RuntimeError("temporary network failure")

    class Source:
        def __init__(self):
            self.items = [{"asset": "BTC", "price": 100., "ohlcv_ready": True}]
        def receive(self, timeout):
            del timeout
            return self.items.pop(0) if self.items else None

    def factory():
        calls["count"] += 1
        return BrokenSource() if calls["count"] == 1 else Source()

    stream = P1StreamConsumer(factory, lambda item: (handled.append(item), processed.set()) and {"cache_hit": True},
                              max_queue_size=1, receive_timeout=.01, max_reconnects=2)
    stream.start()
    assert processed.wait(.5)
    stream.stop()
    metrics = stream.snapshot()
    assert handled and metrics["reconnects"] >= 1
    assert metrics["processed"] >= 1 and metrics["queue_max"] <= 1
    assert metrics["cache_hits"] >= 1 and metrics["memory_queue_capacity"] == 1


def test_p1_polling_source_reads_timestamped_provider_without_fallback():
    class Provider:
        def fetch(self, asset, timeframe):
            assert (asset, timeframe) == ("BTC", "1m")
            return market(2).assign(asset="BTC", timeframe="1m")
    source = P1PollingSource(Provider(), "BTC", "1m")
    item = source.receive(.5)
    assert item["source"] == "p1_test" and item["timestamp"] and item["ohlcv_ready"] is True
    assert source.receive(.5) is None


def test_stream_degrades_instead_of_predicting_from_incomplete_p1_tick():
    class Source:
        def __init__(self):
            self.item = {"asset": "BTC", "close": 100., "ohlcv_ready": False}
        def receive(self, timeout):
            del timeout
            item, self.item = self.item, None
            return item
    handled = []
    stream = P1StreamConsumer(Source, handled.append, receive_timeout=.01)
    stream.start()
    deadline = time.time() + .5
    while time.time() < deadline and stream.snapshot()["state"] != "degraded":
        time.sleep(.01)
    assert stream.snapshot()["state"] == "degraded"
    stream.stop()
    assert not handled and stream.snapshot()["state"] == "stopped"
    assert stream.snapshot()["errors"] >= 1


def test_explainer_writes_real_html_and_pdf(tmp_path):
    prediction = {"id": "p1-artifact", "asset": "BTC", "direction": "bullish", "confidence": .7,
                  "timeframe": "1h", "factors": ["momentum"], "contradictions": []}
    explainer = PredictionExplainer(output_dir=str(tmp_path))
    html = explainer.generate_report(prediction, ["llm"], "html")
    pdf = explainer.generate_report(prediction, ["llm"], "pdf")
    explainer.plot_factor_importance(prediction, save_path=str(tmp_path / "prediction_p1-artifact.png"))
    assert (tmp_path / "prediction_p1-artifact.html").read_text(encoding="utf-8").startswith("\n        <html>")
    assert (tmp_path / "prediction_p1-artifact.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "prediction_p1-artifact.png").read_bytes().startswith(b"\x89PNG")
    assert html["html_path"] and pdf["pdf_path"]


def test_read_only_p4_api_exposes_prediction_metrics_without_trading():
    registry.record_prediction({"id": "api-p4", "asset": "BTC", "timeframe": "1h", "confidence": .6,
                                "as_of": "2025-01-01T01:00:00Z", "data_version": "v1", "abstain": False,
                                "trading_enabled": True})
    app = get_app()
    endpoints = {route.path: route.endpoint for route in app.routes}
    prediction = asyncio.run(endpoints["/api/v4/predictions/{asset}/{timeframe}"]("BTC", "1h"))
    metrics = asyncio.run(endpoints["/api/v4/predictions/{asset}/{timeframe}/metrics"]("BTC", "1h"))
    assert prediction["trading_enabled"] is False
    assert metrics["trading_enabled"] is False and metrics["data_version"] == "v1"
    dashboard = p4_snapshot("BTC", "1h")
    assert dashboard["available"] is True and dashboard["trading_enabled"] is False


def test_p4_rl_feature_is_causal_and_never_is_an_order():
    frame = market(4)
    p4 = pd.DataFrame({"as_of": [frame.timestamp.iloc[2]], "direction": ["bullish"], "confidence": [.8],
                       "abstain": [False], "trading_enabled": [False]})
    integrated = RLDataIntegrator().integrate_p4_prediction_feature(frame, p4)
    assert integrated.loc[0, "p4_confidence"] == 0.0
    assert integrated.loc[2, "p4_direction_score"] == 1.0
    assert integrated.loc[2, "p4_abstain"] == 0.0


def test_p2_sources_at_the_same_timestamp_are_quality_weighted_without_losing_provenance():
    timestamp = pd.Timestamp("2025-01-01T00:00:00Z")
    sentiment = pd.DataFrame({"timestamp": [timestamp, timestamp], "asset": ["BTC", "BTC"],
                              "timeframe": ["1h", "1h"], "sentiment_score": [-1.0, 1.0],
                              "quality": [.25, .75], "source": ["news", "social"]})
    aggregated = aggregate_sentiment_data(sentiment, timestamp, asset="BTC", timeframe="1h")
    assert len(aggregated) == 1
    assert aggregated["sentiment_score"].iloc[0] == .5
    assert aggregated["source"].iloc[0] == "news|social"


def test_multi_horizon_models_train_persist_and_predict_separately(tmp_path):
    horizons = ["5m", "1h", "1d"]
    market_by_horizon, sentiment_by_horizon = {}, {}
    for horizon in horizons:
        frame = market(72)
        close = 100 + np.sin(np.arange(len(frame)) / 2) * 3
        frame[["open", "high", "low", "close"]] = np.column_stack((close - .1, close + .2, close - .2, close))
        frame["asset"], frame["timeframe"] = "BTC", horizon
        market_by_horizon[horizon] = frame
        sentiment_by_horizon[horizon] = pd.DataFrame({"timestamp": frame["timestamp"], "asset": "BTC",
                                                       "timeframe": horizon, "sentiment_score": np.sin(np.arange(len(frame))),
                                                       "quality": .9, "source": "p2_test"})
    predictor = MultiHorizonPredictor(model_save_dir=str(tmp_path), use_hybrid=True,
                                      custom_config={"use_gpu": False, "cache_dir": str(tmp_path / "cache")})
    trained = predictor.train_models("BTC", horizons, market_data=market_by_horizon,
                                     sentiment_data=sentiment_by_horizon, as_of=market_by_horizon["5m"].timestamp.iloc[-1])
    predictions = predictor.predict_all_horizons("BTC", market_data=market_by_horizon,
                                                  sentiment_data=sentiment_by_horizon,
                                                  as_of=market_by_horizon["5m"].timestamp.iloc[-1],
                                                  timeframes=horizons)
    assert all("error" not in trained[horizon] for horizon in horizons)
    assert all((tmp_path / f"prediction_model_{horizon}.pkl").is_file() for horizon in horizons)
    assert {"5m", "1h", "1d"}.issubset(predictions)
    assert all(predictions[horizon]["mode"] == "ml_only" for horizon in horizons)
