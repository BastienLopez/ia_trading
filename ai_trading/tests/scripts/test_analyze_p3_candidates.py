import json
import shutil
from copy import deepcopy

import pandas as pd
import pytest

from ai_trading.scripts.analyze_p3_candidates import AnalysisError, analyze_runs, parameter_fingerprint, top5_manifest


def _parameters(index):
    return {
        "agent_type": "ppo", "hidden_size": 64 + index, "learning_rate": 0.0003,
        "base_slippage": 0.001, "max_active_positions": 3, "rebalance_frequency": 5,
        "min_trade_fraction": 0.03, "min_action_magnitude": 0.1, "max_asset_exposure": 0.4,
    }


def _metrics(return_value=0.10, benchmark=0.05, *, closed=25, profit_factor=2.0, diversity=True):
    return {
        "strategy": {"total_return": return_value, "annual_return": return_value, "volatility": 0.1, "sharpe_ratio": 1.0, "max_drawdown": 0.05, "sortino_ratio": 1.2, "calmar_ratio": 2.0},
        "buy_and_hold": {"total_return": benchmark, "annual_return": benchmark, "volatility": 0.1, "sharpe_ratio": 0.5, "max_drawdown": 0.1, "sortino_ratio": 0.6, "calmar_ratio": 0.5},
        "closed_trade_metrics": {"closed_trade_count": closed, "profit_factor": profit_factor, "expectancy": 1.0, "win_rate": 0.6},
        "action_diversity": {"passed": diversity, "directional_passed": diversity, "direction_counts": {"buy": 12 if diversity else 0, "hold": 3, "sell": 12 if diversity else 0}},
        "executed_trade_count": closed * 2, "trade_decision_count": 10, "test_candles": 100,
        "decision_steps": 50, "evaluation_bar_count": 99, "action_component_count": 150,
    }


def _write_seed(directory, metrics, *, reconciled=True):
    directory.mkdir(parents=True)
    initial = 10_000.0
    ending = initial * (1 + metrics["strategy"]["total_return"])
    metrics = deepcopy(metrics)
    metrics["trade_audit"] = {
        "initial_balance": initial, "ending_equity": ending, "equity_pnl": ending - initial,
        "ledger_pnl": ending - initial, "reconciliation_error": 0.0 if reconciled else 1.0,
        "reconciled": reconciled, "closed_trades": metrics["closed_trade_metrics"]["closed_trade_count"],
    }
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (directory / "trade_audit.json").write_text(json.dumps(metrics["trade_audit"]), encoding="utf-8")
    pd.DataFrame({"timestamp": ["2026-01-01"], "agent_rl": [ending], "passif_diversifie": [initial]}).to_csv(directory / "equity_curve.csv", index=False)
    pd.DataFrame({"timestamp": ["2026-01-01"], "decision_available": [True], "action_BTC": [0.2]}).to_csv(directory / "actions.csv", index=False)
    pd.DataFrame(columns=["timestamp", "side"]).to_csv(directory / "trades.csv", index=False)
    pd.DataFrame({"pnl_net": [1.0] * metrics["closed_trade_metrics"]["closed_trade_count"]}).to_csv(directory / "closed_trades.csv", index=False)


def _write_run(tmp_path, *, run_name="run", window=0, seeds=(42,), parameter_overrides=None, metrics_overrides=None, reconciled=True):
    root = tmp_path / run_name / "20260101T000000Z"
    candidates = []
    for index in range(10):
        params = _parameters(index)
        if parameter_overrides and index in parameter_overrides:
            params = parameter_overrides[index]
        seed_metrics = []
        for seed in seeds:
            metrics = _metrics()
            if metrics_overrides and (index, seed) in metrics_overrides:
                metrics = metrics_overrides[index, seed]
            _write_seed(root / f"window_{window:02d}" / "validation" / f"candidate_{index:02d}" / f"seed_{seed}", metrics, reconciled=reconciled)
            seed_metrics.append(metrics)
        candidates.append({"parameters": params, "metrics": seed_metrics[0], "eligible": True, "reason": None})
    (root / "walk_forward_summary.json").write_text(json.dumps([{"window": window, "validation_candidates": candidates}]), encoding="utf-8")
    (root.parent / "command.log").write_text("completed\n", encoding="utf-8")
    return root.parent


def test_complete_candidate_and_manifest_are_reproducible(tmp_path):
    run = _write_run(tmp_path)
    analysis = analyze_runs([run], expected_seeds=(42,))

    assert not [issue for issue in analysis["anomalies"] if issue["severity"] == "error"]
    assert len(analysis["candidates"]) == 10
    assert len(top5_manifest(analysis)["candidates"]) == 5


def test_single_candidate_manifest_run_is_complete_when_summary_matches(tmp_path):
    run = _write_run(tmp_path)
    root = run / "20260101T000000Z"
    for index in range(1, 10):
        shutil.rmtree(root / "window_00" / "validation" / f"candidate_{index:02d}")
    summary_path = root / "walk_forward_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[0]["validation_candidates"] = summary[0]["validation_candidates"][:1]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    analysis = analyze_runs([run], expected_seeds=(42,))

    assert len(analysis["candidates"]) == 1
    assert not [issue for issue in analysis["anomalies"] if issue["severity"] == "error"]


def test_missing_seed_is_reported(tmp_path):
    analysis = analyze_runs([_write_run(tmp_path)], expected_seeds=(42, 314))

    assert any("seeds trouvées" in issue["message"] for issue in analysis["anomalies"])


def test_nan_metric_is_reported(tmp_path):
    invalid = _metrics()
    invalid["strategy"]["total_return"] = float("nan")
    analysis = analyze_runs([_write_run(tmp_path, metrics_overrides={(0, 42): invalid})], expected_seeds=(42,))

    assert any("NaN" in issue["message"] for issue in analysis["anomalies"])


def test_infinite_profit_factor_is_neutralized(tmp_path):
    metrics = _metrics(profit_factor=float("inf"))
    analysis = analyze_runs([_write_run(tmp_path, metrics_overrides={(0, 42): metrics})], expected_seeds=(42,))
    candidate = next(item for item in analysis["candidates"] if 0 in item["original_indices"])

    assert candidate["profit_factor_robust"] == 0.0
    assert any("profit_factor infini" in issue["message"] for issue in analysis["anomalies"])


def test_unreconciled_ledger_is_reported(tmp_path):
    with pytest.raises(AnalysisError, match="aucun candidat analysable"):
        analyze_runs([_write_run(tmp_path, reconciled=False)], expected_seeds=(42,))


def test_mono_action_policy_fails_runner_guard(tmp_path):
    analysis = analyze_runs([_write_run(tmp_path, metrics_overrides={(0, 42): _metrics(diversity=False)})], expected_seeds=(42,))
    candidate = next(item for item in analysis["candidates"] if 0 in item["original_indices"])

    assert not candidate["windows"][0]["eligible"]
    assert candidate["windows"][0]["reason"] == "diversité d'actions insuffisante"


def test_identical_parameters_under_two_indices_are_one_configuration(tmp_path):
    shared = _parameters(0)
    analysis = analyze_runs([_write_run(tmp_path, parameter_overrides={1: shared})], expected_seeds=(42,))

    assert len(analysis["candidates"]) == 9
    assert any(item["original_indices"] == [0, 1] for item in analysis["candidates"])


def test_different_parameters_under_same_index_are_flagged(tmp_path):
    first = _write_run(tmp_path, run_name="first", window=0)
    changed = _parameters(0)
    changed["hidden_size"] = 999
    second = _write_run(tmp_path, run_name="second", window=1, parameter_overrides={0: changed})
    analysis = analyze_runs([first, second], expected_seeds=(42,))

    assert any("index 0 désigne plusieurs empreintes" in issue["message"] for issue in analysis["anomalies"])


def test_ranking_is_deterministic(tmp_path):
    run = _write_run(tmp_path)
    first = analyze_runs([run], expected_seeds=(42,))
    second = analyze_runs([run], expected_seeds=(42,))

    assert [item["candidate_id"] for item in first["candidates"]] == [item["candidate_id"] for item in second["candidates"]]
    assert [item["robust_score"] for item in first["candidates"]] == [item["robust_score"] for item in second["candidates"]]
