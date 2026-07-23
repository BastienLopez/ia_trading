import numpy as np
import pandas as pd

from ai_trading.rl.indicator_fusion import add_causal_indicator_fusion
from ai_trading.rl.trading_environment import TradingEnvironment


def _ohlcv(rows=160):
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = 100 + np.linspace(0, 20, rows) + np.sin(np.arange(rows) / 3)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100 + np.arange(rows)}, index=index)


def test_indicator_fusion_is_causal_and_bounded():
    source = _ohlcv()
    original = add_causal_indicator_fusion(source)
    changed = source.copy()
    changed.loc[changed.index[120]:, "close"] *= 3
    altered = add_causal_indicator_fusion(changed)
    columns = ["signal_trend", "signal_momentum", "signal_mean_reversion", "signal_confidence", "signal_direction"]
    np.testing.assert_allclose(original.loc[: original.index[119], columns], altered.loc[: altered.index[119], columns])
    assert original["signal_confidence"].between(0.0, 1.0).all()
    assert original["signal_direction"].between(-1.0, 1.0).all()


def test_environment_exposes_raw_indicators_and_fused_signals():
    env = TradingEnvironment(_ohlcv(), window_size=30)
    assert {"rsi", "macd", "atr", "adx", "signal_direction", "signal_confidence"}.issubset(env.feature_columns)
