import json
from pathlib import Path

import pandas as pd
from types import SimpleNamespace

import pytest

from ai_trading.scripts.run_real_multi_asset_walk_forward import (
    CANDIDATES,
    _aggregate_validation,
    _assert_test_available,
    _eligible,
    _environment,
    _metrics,
    _load_candidate_config,
    _parameter_fingerprint,
    _selected_candidate_entries,
    _passive_equity,
    _record_test_consumed,
    _save,
    _slice_with_context,
    _write_trade_audit,
)
from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment


def test_candidate_protocol_does_not_override_the_cli_training_budget():
    for candidate in CANDIDATES:
        assert "episodes" not in candidate
        assert "max_training_steps" not in candidate
        assert "max_optimization_steps" not in candidate
    trend_candidate = CANDIDATES[1]
    assert trend_candidate["turnover_reward_penalty"] < 0.002
    assert trend_candidate["bull_underexposure_penalty"] > 0


def test_candidate_config_requires_a_complete_matching_fingerprint(tmp_path):
    parameters = dict(CANDIDATES[0])
    manifest = {
        "schema_version": 1,
        "candidates": [{
            "candidate_id": "p3-test",
            "original_candidate_index": 0,
            "parameter_fingerprint": _parameter_fingerprint(parameters),
            "parameters": parameters,
            "source_runs": ["p3_validation_w01"],
            "original_seeds": [42, 314, 2024],
        }],
    }
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _load_candidate_config(path) == manifest["candidates"]
    args = SimpleNamespace(candidate_config=str(path), candidate_indices=None, candidate_ids=["p3-test"])
    assert _selected_candidate_entries(args) == manifest["candidates"]

    manifest["candidates"][0]["parameter_fingerprint"] = "wrong"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="empreinte invalide"):
        _load_candidate_config(path)


def test_research_candidates_are_versioned_and_can_override_short_risk_limits():
    candidates = _load_candidate_config(
        Path("/app/ai_trading/configs/p3_p4_research_candidates.json")
    )
    assert len(candidates) == 4
    long_only = next(item["parameters"] for item in candidates if "long-only" in item["candidate_id"])
    conservative = next(item["parameters"] for item in candidates if "low-blend" in item["candidate_id"])
    assert long_only["allow_short"] is False
    assert conservative["max_short_exposure"] == pytest.approx(0.10)


def test_multi_seed_selection_rejects_a_candidate_with_one_losing_seed():
    base = {
        "strategy": {"total_return": 0.10},
        "buy_and_hold": {"total_return": 0.05},
        "closed_trade_metrics": {"closed_trade_count": 10, "profit_factor": 2.0, "expectancy": 1.0},
        "action_diversity": {"passed": True, "directional_passed": True, "direction_counts": {"buy": 2, "sell": 2}},
        "trade_decision_count": 2,
        "evaluation_bar_count": 20,
        "executed_trade_count": 2,
        "test_candles": 20,
        "decision_steps": 2,
        "action_component_count": 6,
    }
    losing = {**base, "strategy": {"total_return": 0.04}}
    aggregate = _aggregate_validation([base, losing])

    assert aggregate["seed_stability"]["outperforming_seed_count"] == 1
    assert _eligible(aggregate, min_trades=5, max_order_rate=0.35) == (
        False, "candidat instable selon les seeds d'entraînement"
    )


def test_validation_context_preserves_indicator_history_without_trading_it():
    index = pd.date_range("2025-01-01", periods=200)
    data = {"BTC": pd.DataFrame({"close": range(200)}, index=index)}
    sliced, start_step = _slice_with_context(data, index[120:150], context_candles=80)

    assert start_step == 80
    assert sliced["BTC"].index[0] == index[40]
    assert sliced["BTC"].index[start_step] == index[120]


def test_short_fills_are_ledgered_and_reported_separately(tmp_path):
    timestamp = pd.Timestamp("2026-01-01")
    events = [
        {"timestamp": timestamp, "symbol": "BTC", "side": "short_open", "position_side": "short", "quantity": 1.0, "price": 100.0, "fee": 1.0, "reason": "agent_order"},
        {"timestamp": timestamp + pd.Timedelta(days=1), "symbol": "BTC", "side": "borrow_fee", "position_side": "short", "quantity": 0.0, "price": 90.0, "fee": 0.1, "reason": "short_borrow_fee"},
        {"timestamp": timestamp + pd.Timedelta(days=2), "symbol": "BTC", "side": "short_cover", "position_side": "short", "quantity": 1.0, "price": 80.0, "fee": 1.0, "reason": "take_profit"},
    ]
    data = {"BTC": pd.DataFrame({"close": [100.0, 90.0, 80.0]}, index=pd.date_range(timestamp, periods=3))}

    audit = _write_trade_audit(tmp_path, events, data)
    closed = pd.read_csv(tmp_path / "closed_trades.csv")
    fills = pd.read_csv(tmp_path / "order_fills.csv")

    assert audit["short_open_orders"] == 1
    assert audit["short_cover_orders"] == 1
    assert audit["borrow_fees"] == 0.1
    assert closed.loc[0, "position_side"] == "short"
    assert closed.loc[0, "borrow_fee"] == 0.1
    assert closed.loc[0, "pnl_net"] == pytest.approx(17.9)
    assert fills["side"].tolist() == ["short_open", "borrow_fee", "short_cover"]


def test_trade_audit_reconciles_equity_with_realized_short_pnl(tmp_path):
    timestamp = pd.Timestamp("2026-01-01")
    events = [
        {"timestamp": timestamp, "symbol": "BTC", "side": "short_open", "position_side": "short", "quantity": 1.0, "price": 100.0, "fee": 1.0, "reason": "agent_order"},
        {"timestamp": timestamp + pd.Timedelta(days=1), "symbol": "BTC", "side": "borrow_fee", "position_side": "short", "quantity": 0.0, "price": 90.0, "fee": 0.1, "reason": "short_borrow_fee"},
        {"timestamp": timestamp + pd.Timedelta(days=2), "symbol": "BTC", "side": "short_cover", "position_side": "short", "quantity": 1.0, "price": 80.0, "fee": 1.0, "reason": "take_profit"},
    ]
    data = {"BTC": pd.DataFrame({"close": [100.0, 90.0, 80.0]}, index=pd.date_range(timestamp, periods=3))}

    audit = _write_trade_audit(
        tmp_path, events, data, initial_balance=10_000.0, ending_equity=10_017.9
    )

    assert audit["ledger_pnl"] == pytest.approx(17.9)
    assert audit["reconciliation_error"] == pytest.approx(0.0)
    assert audit["reconciled"]


def test_buy_and_hold_uses_agent_horizon_and_entry_exit_fees():
    dates = pd.date_range("2026-01-01", periods=3)
    data = {
        "BTC": pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=dates),
        "ETH": pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=dates),
    }

    equity = _passive_equity(data, initial=10_000.0, fee=0.01, window_size=0)

    assert len(equity) == 3
    assert equity[0] == 10_000.0
    assert equity[-1] == pytest.approx(10_000.0 * 0.99**2 * 1.2)


def test_save_renders_signed_short_allocations(tmp_path):
    dates = pd.date_range("2026-01-01", periods=3)
    data = {"BTC": pd.DataFrame({"close": [100.0, 95.0, 90.0]}, index=dates)}
    result = {
        "equity": [10_000.0, 10_100.0, 10_250.0],
        "actions": [[-0.2], [0.2]],
        "raw_actions": [[-0.2], [0.2]],
        "decision_available": [True, True],
        "events": [],
        "allocations": [{"BTC": -0.2}, {"BTC": 0.2}],
    }

    _save(tmp_path, data, result, [10_000.0, 9_500.0, 9_000.0], {}, "validation")

    assert (tmp_path / "allocations.png").is_file()
    assert (tmp_path / "equity_comparison.png").is_file()
    actions = pd.read_csv(tmp_path / "actions.csv")
    equity = pd.read_csv(tmp_path / "equity_curve.csv")
    assert actions.loc[0, "timestamp"] == "2026-01-01"
    assert equity.loc[0, "timestamp"] == "2026-01-02"


def test_turnover_counts_rebalance_decisions_not_each_multi_asset_fill():
    timestamp = pd.Timestamp("2026-01-01")
    result = {
        "equity": [10_000.0, 10_100.0, 10_050.0],
        "actions": [[0.2, -0.2, 0.0], [0.0, 0.0, 0.0]],
        "decision_available": [True, True],
        "events": [
            {"timestamp": timestamp, "symbol": "BTC", "side": "buy", "quantity": 1.0, "price": 100.0, "fee": 0.1, "reason": "agent_order"},
            {"timestamp": timestamp, "symbol": "ETH", "side": "short_open", "quantity": 1.0, "price": 100.0, "fee": 0.1, "reason": "agent_order"},
            {"timestamp": timestamp, "symbol": "XAU", "side": "borrow_fee", "quantity": 0.0, "price": 100.0, "fee": 0.1, "reason": "short_borrow_fee"},
        ],
    }

    metrics = _metrics(result, [10_000.0, 9_900.0, 9_800.0], candles=3)

    assert metrics["executed_trade_count"] == 2
    assert metrics["trade_decision_count"] == 1
    assert metrics["decision_steps"] == 2
    assert metrics["evaluation_bar_count"] == 2


def test_oos_registry_refuses_a_consumed_holdout(tmp_path):
    args = SimpleNamespace(
        assets=["BTC/USDT", "ETH/USDT", "XAU/USD"], timeframe="1d",
        oos_registry=str(tmp_path / "oos_registry.json"),
    )
    window = SimpleNamespace(test=pd.DataFrame(index=pd.date_range("2025-07-01", periods=3)))

    registry_path, fingerprint = _assert_test_available(args, window)
    _record_test_consumed(registry_path, fingerprint, args, window)

    with pytest.raises(RuntimeError, match="déjà été consommée"):
        _assert_test_available(args, window)
