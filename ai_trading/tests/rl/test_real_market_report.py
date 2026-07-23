import numpy as np
import pandas as pd
import pytest
import json

from ai_trading.rl.real_market_report import (
    EvaluationRun,
    buy_and_hold_equity,
    calculate_metrics,
    chronological_split,
    discrete_action_label,
    save_report,
    validate_ohlcv,
)
from ai_trading.rl.policy_validation import (
    PolicyDiversityError,
    action_diversity_metrics,
    require_action_diversity,
)


def _ohlcv(rows: int = 150) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = np.linspace(100.0, 150.0, rows)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 10.0},
        index=index,
    )


def test_chronological_split_is_disjoint_and_ordered():
    train, test = chronological_split(_ohlcv(), train_ratio=0.6)
    assert train.index.max() < test.index.min()
    assert len(train) == 90
    assert len(test) == 60


def test_buy_and_hold_is_marked_to_market_not_flat():
    curve = buy_and_hold_equity(np.array([100.0, 80.0, 120.0]), initial_balance=1_000.0, fee=0.001)
    assert curve[1] < curve[0]
    assert curve[2] > curve[0]
    risk_matched = buy_and_hold_equity(
        np.array([100.0, 80.0, 120.0]), initial_balance=1_000.0, fee=0.001, exposure_fraction=0.2
    )
    assert risk_matched[0] < 1_000.0
    assert risk_matched[2] < curve[2]


def test_discrete_action_labels_are_auditable():
    assert discrete_action_label(0) == "HOLD"
    assert discrete_action_label(3) == "BUY 60%"
    assert discrete_action_label(8) == "SELL 60%"


def test_validation_rejects_non_positive_prices():
    invalid = _ohlcv()
    invalid.loc[invalid.index[0], "close"] = 0
    with pytest.raises(ValueError, match="strictement positifs"):
        validate_ohlcv(invalid)


def test_save_report_writes_auditable_real_market_artifacts(tmp_path):
    data = _ohlcv()
    dates = data.index[-10:]
    equity = np.array([1_000, 980, 1_020, 990, 1_030, 1_010, 1_050, 1_040, 1_080, 1_070], dtype=float)
    benchmark = buy_and_hold_equity(data.loc[dates, "close"], initial_balance=1_000, fee=0.001)
    paths = save_report(
        tmp_path,
        "BTC/USDT",
        "1h",
        data,
        EvaluationRun(
            dates,
            equity,
            benchmark,
            rewards=[0.1] * 9,
            actions=[0] * 9,
            executions=[False] * 9,
            execution_reasons=[None] * 9,
            fills=[],
        ),
        transaction_fee=0.001,
        training_metadata={"device": "cpu"},
        data_source="Yahoo Finance GC=F (proxy XAU/USD)",
    )
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert '"executed_trade_count": 0' in paths["metrics"].read_text(encoding="utf-8")
    assert '"source": "Yahoo Finance GC=F (proxy XAU/USD)"' in paths["metrics"].read_text(encoding="utf-8")
    equity_curve = pd.read_csv(paths["equity_curve"])
    assert {"buy_hold_equity", "buy_hold_full_equity", "buy_hold_risk_matched_equity", "action_label"}.issubset(equity_curve.columns)
    payload = paths["metrics"].read_text(encoding="utf-8")
    assert '"3": "BUY 60%"' in payload
    assert '"8": "SELL 60%"' in payload
    metrics = calculate_metrics(equity, periods_per_year=8760)
    assert metrics["max_drawdown"] < 0


def test_report_records_causal_baselines_and_trade_pnl_by_regime(tmp_path):
    data = _ohlcv()
    dates = data.index[-6:]
    paths = save_report(
        tmp_path,
        "BTC/USDT",
        "1h",
        data,
        EvaluationRun(
            dates=dates,
            equity=np.array([1000, 1005, 1010, 1008, 1015, 1020], dtype=float),
            benchmark_equity=np.array([1000, 1002, 1004, 1006, 1008, 1010], dtype=float),
            rewards=[0.0] * 5,
            actions=[1, 0, 0, 6, 0],
            executions=[True, False, False, True, False],
            execution_reasons=["agent_order", None, None, "agent_order", None],
            fills=[
                {"timestamp": dates[1], "side": "buy", "quantity": 1.0, "price": 100.0, "fee": 0.1, "reason": "agent_order"},
                {"timestamp": dates[4], "side": "sell", "quantity": 1.0, "price": 105.0, "fee": 0.1, "reason": "take_profit"},
            ],
            baseline_equity={"ema_trend": np.array([1000, 1001, 1003, 1004, 1007, 1009], dtype=float)},
            market_regimes=["range", "bull", "bull", "bear", "bear", "range"],
        ),
        transaction_fee=0.001,
        training_metadata={"device": "cpu"},
    )
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    curve = pd.read_csv(paths["equity_curve"])
    trades = pd.read_csv(paths["trades"])
    assert "ema_trend" in payload["causal_baselines"]
    assert payload["closed_trade_pnl_by_exit_regime"]["bear"] > 0.0
    assert "baseline_ema_trend_equity" in curve and "market_regime" in curve
    assert {"entry_regime", "exit_regime"}.issubset(trades.columns)


def test_policy_validation_refuses_a_mono_action_policy():
    with pytest.raises(PolicyDiversityError, match="Politique dégénérée"):
        require_action_diversity([2] * 30)
    with pytest.raises(PolicyDiversityError, match="Politique dégénérée"):
        require_action_diversity([1, 3, 5, 3, 1])
    assert require_action_diversity([0, 1, 0, 2, 1, 0], max_dominant_share=0.7)["passed"]
    buy_only = action_diversity_metrics([1, 3, 5, 3, 1])
    assert buy_only["passed"]
    assert not buy_only["directional_passed"]
