"""Features de régime de marché strictement causales pour le RL.

Les labels servent au reporting et les deux colonnes numériques sont intégrées à
l'observation. Chaque valeur à l'instant ``t`` dépend exclusivement de bougies
dont l'horodatage est inférieur ou égal à ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_COLUMNS = ("regime_trend", "regime_volatility", "market_regime")


def add_causal_regime_features(
    data: pd.DataFrame,
    *,
    fast_window: int = 20,
    slow_window: int = 60,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Ajoute tendance, volatilité relative et label de régime sans look-ahead."""
    if "close" not in data.columns:
        raise ValueError("La colonne close est requise pour calculer les régimes")
    if min(fast_window, slow_window, volatility_window) < 2:
        raise ValueError("Les fenêtres de régime doivent être >= 2")

    frame = data.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("close doit être fini et strictement positif")

    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    trend = fast.div(slow).sub(1.0)
    returns = close.pct_change(fill_method=None)
    volatility = returns.rolling(volatility_window, min_periods=volatility_window).std(ddof=0)
    volatility_reference = volatility.rolling(slow_window, min_periods=slow_window).median()
    relative_volatility = volatility.div(volatility_reference.replace(0.0, np.nan))

    frame["regime_trend"] = trend.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["regime_volatility"] = (
        relative_volatility.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)
    )
    frame["market_regime"] = np.select(
        [frame["regime_trend"] > 0.01, frame["regime_trend"] < -0.01],
        ["bull", "bear"],
        default="range",
    )
    return frame


def regime_summary(labels: pd.Series | list[str]) -> dict[str, int]:
    """Compte les régimes réellement évalués pour l'audit des trades."""
    values = pd.Series(labels, dtype="object").dropna()
    return {str(name): int(count) for name, count in values.value_counts().sort_index().items()}
