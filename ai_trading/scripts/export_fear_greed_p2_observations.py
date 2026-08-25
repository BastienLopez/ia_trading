"""Exporte un historique P2 réel Fear & Greed, consommable par le walk-forward P4.

La source est un indicateur de sentiment crypto global, pas un sentiment propre
à un actif. Les lignes BTC et ETH partagent donc la même provenance explicite.
Elle ne doit pas être interpolée vers des bougies intraday.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from ai_trading.llm.sentiment_analysis.sentiment_pipeline import SentimentPipeline


FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"
SOURCE = "alternative_me_fear_greed_global_crypto"


def _asset_code(asset: str) -> str:
    return asset.split("/")[0].upper()


def observations_from_payload(payload: dict[str, Any], assets: Iterable[str], timeframe: str) -> pd.DataFrame:
    """Valide la réponse source puis construit le contrat P2 sans interpolation."""
    if timeframe != "1d":
        raise ValueError("Fear & Greed est quotidien : --timeframe 1d est requis")
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError("P2 Fear & Greed: réponse sans historique exploitable")
    source = pd.DataFrame(records)
    if not {"timestamp", "value"}.issubset(source.columns):
        raise ValueError("P2 Fear & Greed: timestamp et value obligatoires")
    unix_timestamps = pd.to_numeric(source["timestamp"], errors="coerce")
    timestamps = pd.to_datetime(unix_timestamps, unit="s", utc=True, errors="coerce")
    values = pd.to_numeric(source["value"], errors="coerce")
    if timestamps.isna().any() or values.isna().any() or not values.between(0, 100).all():
        raise ValueError("P2 Fear & Greed: historique invalide")
    base = pd.DataFrame({
        "timestamp": timestamps,
        "sentiment_score": ((values - 50.0) / 50.0).clip(-1.0, 1.0),
        "quality": 0.65,
        "source": SOURCE,
    }).drop_duplicates("timestamp", keep="last")
    observations = pd.concat(
        [base.assign(asset=_asset_code(asset), timeframe=timeframe) for asset in assets],
        ignore_index=True,
    )
    return observations.sort_values(["asset", "timestamp"]).reset_index(drop=True)


def fetch_payload(timeout: float, retries: int) -> dict[str, Any]:
    error: Exception | None = None
    for _ in range(retries):
        try:
            response = requests.get(FEAR_GREED_URL, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            error = exc
    raise RuntimeError(f"P2 Fear & Greed indisponible après {retries} tentative(s): {error}") from error


def export(output: str | Path, assets: Iterable[str], timeframe: str, timeout: float = 15.0, retries: int = 3) -> Path:
    observations = observations_from_payload(fetch_payload(timeout, retries), assets, timeframe)
    target = SentimentPipeline.persist_p4_observations(observations, output)
    manifest = {
        "source": SOURCE,
        "source_url": FEAR_GREED_URL,
        "scope": "global_crypto_sentiment",
        "assets": sorted({_asset_code(asset) for asset in assets}),
        "timeframe": timeframe,
        "rows": int(len(observations)),
        "min_timestamp": observations["timestamp"].min().isoformat(),
        "max_timestamp": observations["timestamp"].max().isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    target.with_suffix(f"{target.suffix}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--timeframe", default="1d", choices=("1d",))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(export(args.output, args.assets, args.timeframe, args.timeout, args.retries))
