"""Fusion causale des indicateurs techniques pour les politiques RL.

Les indicateurs bruts restent dans l'observation. Ces colonnes supplémentaires
expriment leurs accords (tendance, momentum, volume, volatilité) sans décider
à la place de l'agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(frame: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(default, index=frame.index, dtype=float)


def add_causal_indicator_fusion(data: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des signaux bornés, calculés uniquement à partir de t et du passé."""
    if "close" not in data:
        raise ValueError("close est requis pour fusionner les indicateurs")
    frame = data.copy()
    close = _series(frame, "close").replace(0.0, np.nan)
    ema_fast = _series(frame, "ema_9", "ema_12", "ema")
    ema_slow = _series(frame, "ema_21", "ema_26", "sma_20")
    macd_hist = _series(frame, "macd_hist")
    rsi = _series(frame, "rsi", default=50.0).fillna(50.0)
    stoch = _series(frame, "stoch_k", default=50.0).fillna(50.0)
    adx = _series(frame, "adx").fillna(0.0)
    atr = _series(frame, "atr").fillna(0.0)
    volume = _series(frame, "volume").fillna(0.0)
    obv = _series(frame, "obv").fillna(0.0)

    trend = np.tanh((ema_fast - ema_slow).div(close).fillna(0.0) * 40.0)
    momentum = np.tanh(macd_hist.div(close).fillna(0.0) * 120.0)
    oscillator = ((rsi - 50.0) / 50.0 + (stoch - 50.0) / 50.0) / 2.0
    oscillator = oscillator.clip(-1.0, 1.0)
    trend_strength = (adx / 50.0).clip(0.0, 1.0)
    volume_ratio = volume.div(volume.rolling(20, min_periods=5).mean().replace(0.0, np.nan)).fillna(1.0)
    volume_confirmation = np.tanh((volume_ratio - 1.0) * 2.0)
    obv_confirmation = np.tanh(obv.diff().rolling(5, min_periods=2).mean().div(obv.abs().rolling(20, min_periods=5).mean().replace(0.0, np.nan)).fillna(0.0) * 20.0)
    volatility = atr.div(close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volatility_penalty = (volatility / volatility.rolling(60, min_periods=20).median().replace(0.0, np.nan)).fillna(1.0).clip(0.5, 3.0)

    # La tendance pèse davantage quand ADX confirme ; RSI/Stoch sont réduits
    # dans un trend fort afin d'éviter de vendre mécaniquement un marché haussier.
    directional_score = (
        0.40 * trend * (0.5 + 0.5 * trend_strength)
        + 0.25 * momentum
        + 0.20 * oscillator * (1.0 - 0.5 * trend_strength)
        + 0.10 * volume_confirmation
        + 0.05 * obv_confirmation
    )
    frame["signal_trend"] = trend.astype(float)
    frame["signal_momentum"] = momentum.astype(float)
    frame["signal_mean_reversion"] = oscillator.astype(float)
    frame["signal_volume_confirmation"] = ((volume_confirmation + obv_confirmation) / 2.0).astype(float)
    frame["signal_volatility_penalty"] = volatility_penalty.astype(float)
    frame["signal_confidence"] = (np.abs(directional_score) * trend_strength * (1.0 / volatility_penalty)).clip(0.0, 1.0).astype(float)
    frame["signal_direction"] = np.tanh(directional_score * 2.0).astype(float)
    return frame
