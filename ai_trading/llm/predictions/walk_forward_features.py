"""Construction causale de features P4 pour un walk-forward RL multi-actifs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from ai_trading.llm.predictions.prediction_model import PredictionModel
from ai_trading.llm.predictions.prediction_contract import timeframe_to_timedelta


P2_OBSERVATION_COLUMNS = {"timestamp", "asset", "timeframe", "sentiment_score", "quality", "source"}


def _compact_calibration_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Conserve les métriques auditables sans dupliquer les séries OOS brutes.

    Les probabilités réellement émises sont déjà persistées, ligne par ligne,
    dans ``p4_predictions.parquet``. Garder ici les tableaux complets de chaque
    re-train faisait grossir les artefacts sans apporter de nouvelle preuve.
    """
    calibration = report.get("calibration", {})
    result: Dict[str, Any] = {}
    for segment, values in calibration.items():
        if isinstance(values, dict):
            result[segment] = {
                key: value for key, value in values.items()
                if key not in {"probabilities", "targets"}
            }
        else:
            result[segment] = values
    return result


def _asset_code(symbol: str) -> str:
    return symbol.split("/")[0].upper()


def _load_observations(path: str | Path) -> pd.DataFrame:
    location = Path(path)
    if not location.is_file():
        raise ValueError(f"P4: observations P2 introuvables: {location}")
    observations = pd.read_parquet(location) if location.suffix.lower() == ".parquet" else pd.read_csv(location)
    missing = P2_OBSERVATION_COLUMNS - set(observations.columns)
    if missing:
        raise ValueError(f"P4: contrat P2 incomplet, colonnes absentes: {sorted(missing)}")
    observations = observations.copy()
    observations["timestamp"] = pd.to_datetime(observations["timestamp"], utc=True, errors="coerce")
    if observations["timestamp"].isna().any():
        raise ValueError("P4: observations P2 avec timestamps invalides")
    observations["sentiment_score"] = pd.to_numeric(observations["sentiment_score"], errors="coerce")
    observations["quality"] = pd.to_numeric(observations["quality"], errors="coerce")
    if observations[["sentiment_score", "quality"]].isna().any().any():
        raise ValueError("P4: score ou qualité P2 non numérique")
    if not observations["sentiment_score"].between(-1.0, 1.0).all():
        raise ValueError("P4: sentiment_score P2 doit être borné dans [-1, 1]")
    if not observations["quality"].between(0.0, 1.0).all():
        raise ValueError("P4: quality P2 doit être bornée dans [0, 1]")
    identity = observations[["asset", "timeframe", "source"]]
    if identity.isna().any().any() or identity.astype(str).apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("P4: actif, timeframe et source P2 sont obligatoires")
    return observations


def validate_p2_observations(path: str | Path, assets: Iterable[str], timeframe: str) -> None:
    """Échoue avant tout téléchargement P1 si l'entrée P2 ne couvre pas le run."""
    observations = _load_observations(path)
    available = set(observations.loc[observations["timeframe"] == timeframe, "asset"].str.upper())
    missing = sorted({_asset_code(asset) for asset in assets} - available)
    if missing:
        raise ValueError(f"P4: observations P2 absentes pour {', '.join(missing)}/{timeframe}; génération refusée")


def _market_frame(frame: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
    result = frame.copy().reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["asset"], result["timeframe"] = asset, timeframe
    result["source"] = result.get("source", "p1_walk_forward")
    return result


def _steps_for_horizon(base_timeframe: str, horizon: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440}
    def minutes(value: str) -> int:
        return int(value[:-1]) * units[value[-1].lower()]
    base, target = minutes(base_timeframe), minutes(horizon)
    if target < base or target % base:
        raise ValueError(f"P4: horizon {horizon} incompatible avec timeframe source {base_timeframe}")
    return target // base


def build_causal_p4_features(
    raw: Dict[str, pd.DataFrame], *, timeframe: str, observations_path: str | Path,
    horizons: Iterable[str], output_dir: str | Path, min_train_candles: int = 120,
    retrain_interval: int = 100,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Prédit seulement avec les bougies et sentiments disponibles à chaque instant.

    Les premières bougies, avant apprentissage, reçoivent l'état neutre explicite
    ``abstain=1``. Aucune prédiction ni sentiment futur ne remplit ces lignes.
    """
    if min_train_candles < 20 or retrain_interval < 1:
        raise ValueError("P4: min_train_candles >=20 et retrain_interval >=1 requis")
    observations, target_dir = _load_observations(observations_path), Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(dict.fromkeys(horizons))
    if not horizons:
        raise ValueError("P4: au moins un horizon est requis")
    output, manifest = {}, {"timeframe": timeframe, "horizons": list(horizons), "assets": {}, "trading_enabled": False}
    prediction_rows: list[dict[str, Any]] = []
    training_reports: list[dict[str, Any]] = []
    quality_reports: list[dict[str, Any]] = []
    max_sentiment_age = min(timeframe_to_timedelta(timeframe) * 4, pd.Timedelta(days=7))
    for symbol, original in raw.items():
        asset, market = _asset_code(symbol), _market_frame(original, _asset_code(symbol), timeframe)
        sentiment = observations[(observations["asset"].str.upper() == asset) & (observations["timeframe"] == timeframe)].copy()
        if sentiment.empty:
            raise ValueError(f"P4: aucune observation P2 {asset}/{timeframe}; génération refusée")
        per_horizon: Dict[str, Dict[str, np.ndarray]] = {}
        for horizon in horizons:
            steps = _steps_for_horizon(timeframe, horizon)
            model = None
            confidence = np.zeros(len(market), dtype=float)
            direction = np.zeros(len(market), dtype=float)
            abstain = np.ones(len(market), dtype=float)
            trained = 0
            unavailable_or_stale = 0
            calibration_rejections = 0
            last_training_attempt: int | None = None
            for position, timestamp in enumerate(market["timestamp"]):
                available_sentiment = sentiment[sentiment["timestamp"] <= timestamp]
                fresh_sentiment = not available_sentiment.empty and (
                    timestamp - available_sentiment["timestamp"].max() <= max_sentiment_age
                )
                if not fresh_sentiment:
                    unavailable_or_stale += 1
                    continue
                if position >= min_train_candles and (
                    last_training_attempt is None or position - last_training_attempt >= retrain_interval
                ):
                    last_training_attempt = position
                    candidate = PredictionModel({"prediction_mode": "ml_only", "use_gpu": False,
                                                  "prediction_horizon_steps": steps, "min_prediction_confidence": 0.45,
                                                  "require_temporal_calibration": True,
                                                  "model_dir": str(target_dir / "models" / asset / horizon),
                                                  "cache_dir": str(target_dir / "cache" / asset / horizon)})
                    history = market.iloc[: position + 1]
                    try:
                        report = candidate.train(history, available_sentiment, timestamp)
                    except Exception as error:
                        calibration_rejections += 1
                        training_reports.append({"asset": asset, "horizon": horizon, "as_of": timestamp.isoformat(),
                                                 "accepted": False, "error": str(error)})
                        model = None
                    else:
                        model = candidate
                        trained += 1
                        training_reports.append({"asset": asset, "horizon": horizon, "as_of": timestamp.isoformat(),
                                                 "accepted": True, "metrics": report})
                        quality_reports.append({"asset": asset, "horizon": horizon, "as_of": timestamp.isoformat(),
                                                "calibration": _compact_calibration_report(report),
                                                "accuracy": report.get("accuracy"), "f1": report.get("f1")})
                if model is not None:
                    prediction = model.predict(asset, timeframe, market.iloc[: position + 1], available_sentiment, timestamp)
                    confidence[position] = float(prediction.get("confidence", 0.0))
                    abstain[position] = float(bool(prediction.get("abstain", True)))
                    direction[position] = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}.get(prediction.get("direction"), 0.0)
                    prediction_rows.append({"timestamp": timestamp, "asset": asset, "timeframe": timeframe,
                                            "horizon": horizon, "confidence": confidence[position],
                                            "direction": prediction.get("direction", "neutral"),
                                            "abstain": bool(abstain[position]),
                                            "probabilities_json": json.dumps(prediction.get("probabilities", {}), sort_keys=True),
                                            "data_version": prediction.get("data_version"), "calibrated": True})
            per_horizon[horizon] = {"confidence": confidence, "direction": direction, "abstain": abstain}
            if model is not None:
                model.save_model(str(target_dir / "models" / f"{asset}_{horizon}.joblib"))
            manifest["assets"].setdefault(asset, {})[horizon] = {
                "horizon_steps": steps,
                "retrain_count": trained,
                "available_ratio": float((1.0 - abstain).mean()),
                "p2_unavailable_or_stale_points": unavailable_or_stale,
                "calibration_rejections": calibration_rejections,
                "max_sentiment_age_seconds": float(max_sentiment_age.total_seconds()),
            }
        augmented = original.copy()
        augmented["p4_confidence"] = np.mean([item["confidence"] for item in per_horizon.values()], axis=0)
        augmented["p4_direction_score"] = np.mean([item["direction"] for item in per_horizon.values()], axis=0)
        augmented["p4_abstain"] = np.max([item["abstain"] for item in per_horizon.values()], axis=0)
        augmented["p4_available"] = 1.0 - augmented["p4_abstain"]
        augmented.reset_index().rename(columns={augmented.index.name or "index": "timestamp"}).to_parquet(
            target_dir / f"p4_features_{asset}.parquet", index=False
        )
        output[symbol] = augmented
    (target_dir / "p4_feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows, columns=["timestamp", "asset", "timeframe", "horizon", "confidence", "direction",
                                           "abstain", "probabilities_json", "data_version", "calibrated"]).to_parquet(
        target_dir / "p4_predictions.parquet", index=False
    )
    (target_dir / "p4_training_reports.json").write_text(json.dumps(training_reports, indent=2, default=str), encoding="utf-8")
    (target_dir / "p4_quality_metrics.json").write_text(json.dumps(quality_reports, indent=2, default=str), encoding="utf-8")
    return output, manifest
