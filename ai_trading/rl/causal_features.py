"""Contrat d'intégration des signaux externes horodatés.

P4/P5 pourront fournir des prédictions ou sentiments à ce module. Le merge est
uniquement ``backward`` : une bougie ne voit jamais une publication future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def merge_causal_external_features(
    market_data: pd.DataFrame,
    external_features: pd.DataFrame,
    *,
    prefix: str = "external_",
) -> pd.DataFrame:
    """Fusionne des features externes numériques disponibles au plus tard à t.

    Les index doivent être des ``DatetimeIndex``. Les colonnes OHLCV sont
    réservées au marché afin de ne jamais écraser les données de prix.
    """
    if not isinstance(market_data.index, pd.DatetimeIndex):
        raise ValueError("market_data doit utiliser un DatetimeIndex")
    if not isinstance(external_features.index, pd.DatetimeIndex):
        raise ValueError("external_features doit utiliser un DatetimeIndex")
    if external_features.index.has_duplicates:
        raise ValueError("external_features contient des horodatages dupliqués")
    forbidden = {"open", "high", "low", "close", "volume"}.intersection(external_features.columns)
    if forbidden:
        raise ValueError(f"Les features externes ne doivent pas écraser OHLCV: {sorted(forbidden)}")

    numeric = external_features.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("external_features doit contenir au moins une colonne numérique")
    renamed = numeric.rename(columns=lambda name: f"{prefix}{name}")
    market = market_data.sort_index().copy()
    external = renamed.sort_index().copy()
    market["__market_timestamp"] = market.index
    external["__external_timestamp"] = external.index
    merged = pd.merge_asof(
        market.reset_index(drop=True),
        external.reset_index(drop=True),
        left_on="__market_timestamp",
        right_on="__external_timestamp",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("__market_timestamp")
    merged.index.name = market_data.index.name
    if (merged["__external_timestamp"].dropna() > merged.index[merged["__external_timestamp"].notna()]).any():
        raise RuntimeError("Fusion causale invalide: une feature future a été détectée")
    merged["external_feature_age_seconds"] = (
        (merged.index.to_series() - merged["__external_timestamp"]).dt.total_seconds().clip(lower=0.0)
    )
    return merged.drop(columns="__external_timestamp")
