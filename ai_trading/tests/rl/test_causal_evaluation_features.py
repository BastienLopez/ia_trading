import numpy as np
import pandas as pd
import pytest

from ai_trading.rl.causal_baselines import causal_single_asset_baselines, diversified_passive_equity
from ai_trading.rl.causal_features import merge_causal_external_features
from ai_trading.rl.market_regime import add_causal_regime_features, regime_summary


def _ohlcv(rows=160):
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = 100.0 + np.linspace(0, 20, rows) + np.sin(np.arange(rows) / 5)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=index,
    )


def test_regime_features_are_causal_and_auditable():
    source = _ohlcv()
    original = add_causal_regime_features(source)
    changed = source.copy()
    changed.loc[changed.index[120]:, "close"] *= 10.0
    altered = add_causal_regime_features(changed)

    assert {"regime_trend", "regime_volatility", "market_regime"}.issubset(original.columns)
    pd.testing.assert_frame_equal(
        original.loc[: original.index[119], ["regime_trend", "regime_volatility", "market_regime"]],
        altered.loc[: altered.index[119], ["regime_trend", "regime_volatility", "market_regime"]],
    )
    assert sum(regime_summary(original["market_regime"]).values()) == len(original)


def test_external_features_are_backward_only_and_never_overwrite_ohlcv():
    market = _ohlcv(8)
    external = pd.DataFrame(
        {"sentiment": [0.2, -0.7]},
        index=pd.DatetimeIndex([market.index[2], market.index[6]]),
    )
    merged = merge_causal_external_features(market, external)

    assert np.isnan(merged.loc[market.index[1], "external_sentiment"])
    assert merged.loc[market.index[2], "external_sentiment"] == pytest.approx(0.2)
    assert merged.loc[market.index[5], "external_sentiment"] == pytest.approx(0.2)
    assert merged.loc[market.index[5], "external_feature_age_seconds"] > 0.0
    assert merged.loc[market.index[5], "close"] == market.loc[market.index[5], "close"]
    with pytest.raises(ValueError, match="OHLCV"):
        merge_causal_external_features(market, external.assign(close=1.0))


def test_causal_baselines_have_no_future_dependency_and_include_costs():
    source = _ohlcv()
    original = causal_single_asset_baselines(source, initial_balance=10_000, fee=0.001, slippage=0.001)
    changed = source.copy()
    changed.loc[changed.index[120]:, "close"] *= 5.0
    altered = causal_single_asset_baselines(changed, initial_balance=10_000, fee=0.001, slippage=0.001)

    assert set(original) == {"ema_trend", "rsi_reversion", "atr_trend"}
    for name, result in original.items():
        assert len(result.equity) == len(source)
        assert np.isfinite(result.equity).all()
        np.testing.assert_allclose(result.equity[:120], altered[name].equity[:120])


def test_diversified_passive_baseline_aligns_assets_and_charges_entry_costs():
    btc = _ohlcv(30)
    eth = _ohlcv(30).assign(close=lambda frame: frame["close"] * 0.1)
    dates, equity = diversified_passive_equity(
        {"BTC": btc, "ETH": eth}, initial_balance=10_000, fee=0.001, slippage=0.001
    )
    assert dates.equals(btc.index)
    assert len(equity) == len(dates)
    assert equity[0] < 10_000
