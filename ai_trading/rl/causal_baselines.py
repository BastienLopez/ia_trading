"""Baselines de trading causales pour évaluer honnêtement le RL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineResult:
    equity: np.ndarray
    target_exposure: np.ndarray
    trade_count: int


def _simulate_target_exposure(
    prices: pd.Series,
    target_exposure: pd.Series,
    initial_balance: float,
    fee: float,
    slippage: float,
) -> BaselineResult:
    """Exécute une cible d'exposition au close t puis marque t+1 au marché."""
    values = pd.to_numeric(prices, errors="coerce").to_numpy(dtype=float)
    targets = target_exposure.clip(0.0, 1.0).fillna(0.0).to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Les prix de baseline doivent être positifs et finis")
    if initial_balance <= 0 or not 0 <= fee < 1 or not 0 <= slippage < 1:
        raise ValueError("Paramètres de simulation baseline invalides")

    cash, quantity, trades = float(initial_balance), 0.0, 0
    equity = [float(initial_balance)]
    for index, price in enumerate(values[:-1]):
        portfolio_value = cash + quantity * price
        desired_asset_value = portfolio_value * targets[index]
        current_asset_value = quantity * price
        delta = desired_asset_value - current_asset_value
        if delta > max(portfolio_value * 1e-5, 1e-8):
            execution = price * (1.0 + slippage)
            spend = min(delta, cash)
            quantity += spend / (execution * (1.0 + fee))
            cash -= spend
            trades += 1
        elif delta < -max(portfolio_value * 1e-5, 1e-8):
            execution = price * (1.0 - slippage)
            quantity_to_sell = min(quantity, -delta / price)
            cash += quantity_to_sell * execution * (1.0 - fee)
            quantity -= quantity_to_sell
            trades += 1
        equity.append(cash + quantity * values[index + 1])
    return BaselineResult(np.asarray(equity, dtype=float), targets, trades)


def causal_single_asset_baselines(
    data: pd.DataFrame,
    *,
    initial_balance: float,
    fee: float,
    slippage: float,
) -> dict[str, BaselineResult]:
    """Construit EMA trend et RSI reversion à partir des informations à t."""
    if "close" not in data:
        raise ValueError("close est requis pour les baselines")
    close = pd.to_numeric(data["close"], errors="coerce")
    fast = close.ewm(span=20, adjust=False, min_periods=20).mean()
    slow = close.ewm(span=60, adjust=False, min_periods=60).mean()
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    losses = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rsi = 100.0 - 100.0 / (1.0 + gains.div(losses.replace(0.0, np.nan)))

    trend_target = (fast > slow).astype(float).fillna(0.0)
    # La position RSI est persistante : achat sous 35, sortie au-dessus de 55.
    rsi_target = pd.Series(0.0, index=data.index)
    in_position = False
    for index, value in rsi.items():
        if np.isfinite(value):
            if value < 35.0:
                in_position = True
            elif value > 55.0:
                in_position = False
        rsi_target.loc[index] = float(in_position)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (pd.to_numeric(data.get("high", close), errors="coerce") - pd.to_numeric(data.get("low", close), errors="coerce")).abs(),
            (pd.to_numeric(data.get("high", close), errors="coerce") - previous_close).abs(),
            (pd.to_numeric(data.get("low", close), errors="coerce") - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    # Trend-following causal : n'expose le portefeuille que si la tendance est
    # haussière, et réduit de moitié lorsque l'ATR relatif est exceptionnel.
    atr_ratio = atr.div(close).replace([np.inf, -np.inf], np.nan)
    median_atr_ratio = atr_ratio.rolling(60, min_periods=60).median()
    atr_trend_target = (fast > slow).astype(float)
    atr_trend_target = atr_trend_target.where(
        atr_ratio <= median_atr_ratio * 1.75, 0.5
    ).fillna(0.0)
    return {
        "ema_trend": _simulate_target_exposure(close, trend_target, initial_balance, fee, slippage),
        "rsi_reversion": _simulate_target_exposure(close, rsi_target, initial_balance, fee, slippage),
        "atr_trend": _simulate_target_exposure(close, atr_trend_target, initial_balance, fee, slippage),
    }


def diversified_passive_equity(
    data_by_asset: dict[str, pd.DataFrame],
    *,
    initial_balance: float,
    fee: float,
    slippage: float,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Buy & Hold égal-pondéré, frais/slippage d'entrée inclus, sans prix inventé."""
    if not data_by_asset:
        raise ValueError("Au moins un actif est requis")
    common_index: pd.DatetimeIndex | None = None
    closes: dict[str, pd.Series] = {}
    for symbol, frame in data_by_asset.items():
        if "close" not in frame:
            raise ValueError(f"close manquant pour {symbol}")
        series = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if (series <= 0).any():
            raise ValueError(f"Prix invalide pour {symbol}")
        closes[symbol] = series
        common_index = series.index if common_index is None else common_index.intersection(series.index)
    if common_index is None or len(common_index) < 2:
        raise ValueError("Les actifs doivent partager au moins deux bougies")
    common_index = common_index.sort_values()
    allocation = initial_balance / len(closes)
    quantities = {
        symbol: allocation / (float(series.loc[common_index[0]]) * (1.0 + fee + slippage))
        for symbol, series in closes.items()
    }
    equity = np.zeros(len(common_index), dtype=float)
    for index, timestamp in enumerate(common_index):
        equity[index] = sum(quantities[symbol] * float(series.loc[timestamp]) for symbol, series in closes.items())
    return common_index, equity
