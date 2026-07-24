"""Tests de calibration P4 sur probabilités et labels hors échantillon."""

import numpy as np

from ai_trading.llm.predictions.uncertainty_calibration import UncertaintyCalibrator


def oos_probabilities():
    return np.asarray([
        [.75, .15, .10], [.15, .70, .15], [.10, .20, .70], [.55, .30, .15],
        [.20, .60, .20], [.15, .20, .65], [.60, .25, .15], [.15, .65, .20],
    ])


def test_intervals_require_out_of_sample_calibration():
    result = UncertaintyCalibrator().calculate_confidence_intervals({"id": "p", "confidence": .7})
    assert "error" in result


def test_empirical_calibration_metrics_distribution_and_interval(tmp_path):
    calibrator = UncertaintyCalibrator()
    metrics = calibrator.fit_calibration(oos_probabilities(), np.asarray([0, 1, 2, 1, 1, 2, 0, 1]), n_bins=4)
    assert metrics["brier_score"] >= 0
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["coverage"] <= 1
    assert 0 <= metrics["abstention_rate"] <= 1
    assert metrics["mean_interval_width"] >= 0
    prediction = {"id": "p", "asset": "BTC", "direction": "bullish", "confidence": .70,
                  "probabilities": {"bearish": .1, "neutral": .2, "bullish": .7}}
    interval = calibrator.calculate_confidence_intervals(prediction)
    distribution = calibrator.estimate_probability_distribution(prediction)
    assert 0 <= interval["lower_bound"] <= interval["upper_bound"] <= 1
    assert distribution["distribution_source"] == "model_probabilities"
    assert sum(distribution["probabilities"].values()) == 1
    output = tmp_path / "calibration.png"
    calibrator.plot_calibration_curve(str(output))
    assert output.exists()


def test_validation_and_test_reports_are_strictly_separate():
    calibrator = UncertaintyCalibrator()
    report = calibrator.evaluate_validation_and_test(
        oos_probabilities()[:4], np.asarray([0, 1, 2, 1]),
        oos_probabilities()[4:], np.asarray([1, 2, 0, 1]),
    )
    assert report["calibration_source"] == "validation_oos"
    assert report["validation"]["samples"] == 4
    assert report["test"]["samples"] == 4
    assert "mean_interval_width" in report["test"]


def test_calibration_never_replaces_numeric_confidence_by_label():
    calibrator = UncertaintyCalibrator()
    calibrator.fit_calibration(oos_probabilities(), np.asarray([0, 1, 2, 1, 1, 2, 0, 1]), n_bins=4)
    result = calibrator.calibrate_prediction({"confidence": .7, "direction": "bullish"})
    assert isinstance(result["confidence"], float)
    assert result["confidence_label"] in {"low", "medium", "high"}
