import numpy as np
import pandas as pd
import pytest

from ai_trading.rl.walk_forward import (
    build_walk_forward_windows,
    aggregate_validation_metrics,
    phase4_gate,
    select_candidate_on_validation,
)
from ai_trading.scripts.run_real_market_walk_forward import _evaluation_with_context


def _ohlcv(rows=50):
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 130.0, rows)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0}, index=index
    )


def _metrics(strategy_return, benchmark_return=0.01):
    return {
        "strategy": {"total_return": strategy_return, "max_drawdown": -0.05, "sharpe_ratio": 1.2, "sortino_ratio": 1.4},
        "buy_and_hold": {"total_return": benchmark_return},
        "closed_trade_metrics": {"closed_trade_count": 8, "profit_factor": 1.4, "minimum_trade_count_met": True},
        "action_diversity": {"passed": True},
    }


def test_walk_forward_partitions_are_chronological_and_tests_do_not_overlap():
    windows = build_walk_forward_windows(_ohlcv(), 20, 10, 10, 10)
    assert len(windows) == 2
    for window in windows:
        assert window.train.index.max() < window.validation.index.min() < window.test.index.min()
    assert windows[0].test.index.max() < windows[1].test.index.min()


def test_candidate_selection_refuses_test_metrics_and_uses_validation_only():
    selected = select_candidate_on_validation([
        {"name": "weak", "validation": _metrics(0.01)},
        {"name": "strong", "validation": _metrics(0.04)},
    ])
    assert selected["name"] == "strong"
    with pytest.raises(ValueError, match="uniquement sa validation"):
        select_candidate_on_validation([{"validation": _metrics(0.02), "test": _metrics(0.50)}])
    inactive = _metrics(0.90)
    inactive["action_diversity"] = {"passed": False}
    with pytest.raises(ValueError, match="garde-fous de validation"):
        select_candidate_on_validation([{"validation": inactive}])
    buy_only = _metrics(0.90)
    buy_only["action_diversity"] = {"passed": True, "directional_passed": False}
    with pytest.raises(ValueError, match="garde-fous de validation"):
        select_candidate_on_validation([{"validation": buy_only}])
    with pytest.raises(ValueError, match="garde-fous de validation"):
        select_candidate_on_validation([{"validation": _metrics(0.90)}], min_closed_trades=30)


def test_validation_seed_aggregation_uses_median_and_requires_every_diversity_guard():
    first, second, third = _metrics(0.01), _metrics(0.90), _metrics(0.04)
    aggregated = aggregate_validation_metrics([first, second, third])
    assert aggregated["strategy"]["total_return"] == pytest.approx(0.04)
    assert aggregated["validation_seed_count"] == 3
    third["action_diversity"] = {"passed": False, "directional_passed": False}
    assert not aggregate_validation_metrics([first, second, third])["action_diversity"]["passed"]


def test_validation_rejects_churn_and_non_profitable_trade_distribution():
    churn = _metrics(0.90)
    churn["test_candles"] = 20
    churn["execution_reason_counts"] = {"agent_order": 10}
    with pytest.raises(ValueError, match="garde-fous de validation"):
        select_candidate_on_validation([{"validation": churn}])
    losing = _metrics(0.90)
    losing["closed_trade_metrics"]["expectancy"] = -1.0
    with pytest.raises(ValueError, match="garde-fous de validation"):
        select_candidate_on_validation([{"validation": losing}])


def test_validation_rejects_a_candidate_below_its_risk_matched_benchmark():
    below_benchmark = _metrics(0.01, benchmark_return=0.02)
    with pytest.raises(ValueError, match="garde-fous de validation"):
        select_candidate_on_validation([{"validation": below_benchmark}])


def test_phase4_gate_requires_multiple_real_out_of_sample_criteria():
    rows = [{"window": index, "test": _metrics(0.04)} for index in range(3)]
    assert phase4_gate(rows)["passed"]
    rows[0]["test"]["action_diversity"] = {"passed": False}
    result = phase4_gate(rows)
    assert not result["passed"]
    assert any("action_diversity" in reason for reason in result["reasons"])


def test_phase4_gate_compares_to_risk_matched_benchmark_when_available():
    rows = [{"window": index, "test": _metrics(0.04, benchmark_return=0.80)} for index in range(3)]
    for row in rows:
        row["test"]["risk_matched_buy_and_hold"] = {"total_return": 0.02}
    assert phase4_gate(rows)["passed"]


def test_phase4_gate_rejects_a_window_without_validation_eligible_candidate():
    rows = [{"window": index, "test": _metrics(0.04)} for index in range(3)]
    rows[1].pop("test")
    result = phase4_gate(rows)
    assert not result["passed"]
    assert result["checks"][1]["validation_rejected"]
    assert any("Aucun candidat validé" in reason for reason in result["reasons"])


def test_evaluation_context_uses_only_past_bars_and_keeps_every_test_bar():
    data = _ohlcv()
    history, target = data.iloc[:30], data.iloc[30:40]
    evaluation = _evaluation_with_context(history, target, window_size=5)
    assert len(evaluation) == 15
    assert evaluation.index[:5].max() < target.index.min()
    assert evaluation.index[5:].equals(target.index)
