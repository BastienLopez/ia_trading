"""P4 réellement injectée dans l'observation du walk-forward P3, sans look-ahead."""

import numpy as np
import pandas as pd
import pytest
import sys
import json

from ai_trading.llm.predictions.walk_forward_features import build_causal_p4_features
from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment
from ai_trading.scripts import run_real_multi_asset_walk_forward as runner


def _market(rows=70):
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    close = 100 + np.sin(np.arange(rows) / 2) * 3
    return pd.DataFrame({"open": close - .1, "high": close + .2, "low": close - .2,
                         "close": close, "volume": 1000 + np.arange(rows)}, index=index)


def test_p4_causal_features_are_in_rl_observation_and_ignore_future(tmp_path):
    frame = _market()
    observations = pd.DataFrame({"timestamp": frame.index, "asset": "BTC", "timeframe": "1h",
                                 "sentiment_score": np.sin(np.arange(len(frame)) / 2), "quality": .9,
                                 "source": "p2_integration"})
    path = tmp_path / "p2.csv"
    observations.to_csv(path, index=False)
    base = {"BTC/USDT": frame}
    augmented, manifest = build_causal_p4_features(
        base, timeframe="1h", observations_path=path, horizons=["1h", "4h"],
        output_dir=tmp_path / "artifacts", min_train_candles=30, retrain_interval=100,
    )
    changed = frame.copy()
    changed.iloc[50:, changed.columns.get_loc("close")] *= 1.5
    changed.iloc[50:, changed.columns.get_loc("high")] *= 1.5
    altered, _ = build_causal_p4_features(
        {"BTC/USDT": changed}, timeframe="1h", observations_path=path, horizons=["1h", "4h"],
        output_dir=tmp_path / "altered", min_train_candles=30, retrain_interval=100,
    )
    columns = ["p4_confidence", "p4_direction_score", "p4_abstain", "p4_available"]
    assert set(columns).issubset(augmented["BTC/USDT"].columns)
    pd.testing.assert_frame_equal(augmented["BTC/USDT"][columns].iloc[:50], altered["BTC/USDT"][columns].iloc[:50])
    # La fixture courte ne possède pas nécessairement trois plis temporels
    # valides : P4 doit alors s'abstenir et laisser une preuve, jamais replier
    # silencieusement sur un modèle non calibré.
    assert manifest["assets"]["BTC"]["1h"]["calibration_rejections"] >= 1
    assert (tmp_path / "artifacts" / "p4_training_reports.json").is_file()
    assert (tmp_path / "artifacts" / "p4_predictions.parquet").is_file()

    plain_env = MultiAssetTradingEnvironment(base, window_size=20, include_technical_indicators=True)
    p4_env = MultiAssetTradingEnvironment(augmented, window_size=20, include_technical_indicators=True)
    plain_state, _ = plain_env.reset(seed=1)
    p4_state, _ = p4_env.reset(seed=1)
    assert p4_state.size > plain_state.size


def test_real_multi_asset_runner_records_enabled_p4_features(tmp_path, monkeypatch):
    frame = _market(70)
    observations = pd.DataFrame({"timestamp": frame.index, "asset": "BTC", "timeframe": "1h",
                                 "sentiment_score": np.sin(np.arange(len(frame)) / 2), "quality": .9,
                                 "source": "p2_runner"})
    p2_path = tmp_path / "p2.csv"
    observations.to_csv(p2_path, index=False)
    monkeypatch.setattr(runner, "_fetch_ohlcv", lambda asset, args: frame.copy())
    monkeypatch.setattr(sys, "argv", ["runner", "--assets", "BTC/USDT", "--timeframe", "1h", "--days", "70",
                                       "--train-candles", "40", "--validation-candles", "15", "--test-candles", "10",
                                       "--step-candles", "10", "--max-windows", "1", "--window-start", "0",
                                       "--validation-seeds", "73", "--episodes", "1", "--max-training-steps", "1",
                                       "--max-optimization-steps", "1", "--batch-size", "8", "--window-size", "10",
                                       "--candidate-config", "/app/ai_trading/configs/p3_locked_candidate.json",
                                       "--candidate-id", "p3-bf72563778749e59", "--evaluation-mode", "validation",
                                       "--enable-p4-features", "--p4-sentiment-observations", str(p2_path),
                                       "--p4-horizons", "1h", "--p4-min-train-candles", "30",
                                       "--p4-retrain-interval", "100", "--output-dir", str(tmp_path / "runs")])
    summary_path = runner._run_unlocked(runner.parse_args())
    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    run_manifest = __import__("json").loads((summary_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary[0]["p4"]["enabled"] is True
    assert summary[0]["p4"]["assets"]["BTC"]["1h"]["calibration_rejections"] >= 1
    assert run_manifest["p4"]["enabled"] is True
    assert run_manifest["resolved_windows"] == [run_manifest["resolved_windows"][0]]


def test_p4_abstains_until_the_first_fresh_p2_observation(tmp_path):
    frame = _market()
    observations = pd.DataFrame({"timestamp": frame.index[35:], "asset": "BTC", "timeframe": "1h",
                                 "sentiment_score": np.sin(np.arange(len(frame) - 35) / 2), "quality": .9,
                                 "source": "p2_delayed"})
    path = tmp_path / "delayed_p2.csv"
    observations.to_csv(path, index=False)
    augmented, manifest = build_causal_p4_features(
        {"BTC/USDT": frame}, timeframe="1h", observations_path=path, horizons=["1h"],
        output_dir=tmp_path / "artifacts", min_train_candles=30, retrain_interval=100,
    )
    result = augmented["BTC/USDT"]
    assert result["p4_abstain"].iloc[:35].eq(1.0).all()
    assert manifest["assets"]["BTC"]["1h"]["p2_unavailable_or_stale_points"] >= 35


def test_p4_persists_only_temporally_calibrated_predictions_and_metrics(tmp_path):
    """Une calibration temporelle valide est nécessaire avant toute prédiction P4."""
    frame = _market(480)
    observations = pd.DataFrame({"timestamp": frame.index, "asset": "BTC", "timeframe": "1h",
                                 "sentiment_score": np.sin(np.arange(len(frame)) / 7), "quality": .9,
                                 "source": "p2_temporal_calibration"})
    path = tmp_path / "p2.csv"
    observations.to_csv(path, index=False)
    artifacts = tmp_path / "artifacts"
    _, manifest = build_causal_p4_features(
        {"BTC/USDT": frame}, timeframe="1h", observations_path=path, horizons=["1h"],
        output_dir=artifacts, min_train_candles=180, retrain_interval=60,
    )
    status = manifest["assets"]["BTC"]["1h"]
    assert status["retrain_count"] >= 1
    assert status["calibration_rejections"] == 0

    predictions = pd.read_parquet(artifacts / "p4_predictions.parquet")
    assert not predictions.empty
    assert predictions["calibrated"].eq(True).all()
    assert {"abstain", "probabilities_json", "confidence", "direction"}.issubset(predictions.columns)

    quality = json.loads((artifacts / "p4_quality_metrics.json").read_text(encoding="utf-8"))
    assert quality
    validation = quality[-1]["calibration"]["validation"]
    test = quality[-1]["calibration"]["test"]
    for metrics in (validation, test):
        assert {"brier_score", "ece", "coverage", "mean_interval_width"}.issubset(metrics)
        assert "probabilities" not in metrics and "targets" not in metrics
    assert quality[-1]["calibration"]["calibration_source"] == "validation_oos"


def test_runner_rejects_xau_intraday_before_any_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_fetch_ohlcv", lambda *_args: pytest.fail("téléchargement P1 inattendu"))
    monkeypatch.setattr(sys, "argv", ["runner", "--timeframe", "1h", "--enable-p4-features",
                                       "--p4-sentiment-observations", str(tmp_path / "missing.csv")])
    with pytest.raises(ValueError, match="XAU/USD ne fournit que 1d"):
        runner._run_unlocked(runner.parse_args())
