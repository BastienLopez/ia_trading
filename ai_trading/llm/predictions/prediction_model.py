"""Modèle hybride P4 : ML tabulaire calibré + prédiction contextuelle injectable."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import ai_trading.config as config
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.market_safety import MarketSafetyGuard
from ai_trading.llm.predictions.model_ensemble import ModelEnsemble
from ai_trading.llm.predictions.prediction_contract import PredictionInputError, align_market_and_sentiment
from ai_trading.llm.predictions.rtx_optimizer import RTXOptimizer, detect_rtx_gpu
from ai_trading.utils import setup_logger

logger = setup_logger("prediction_model")


class TemporalProbabilityCalibrator:
    """Calibration post-fit sur un segment strictement ultérieur.

    ``CalibratedClassifierCV(FrozenEstimator(...))`` applique encore une CV
    stratifiée par défaut dans scikit-learn 1.7. Cette classe évite ce chemin :
    le modèle est appris sur le passé, les transformateurs de probabilités sur
    le segment suivant, et aucune permutation/stratification n'est effectuée.
    """

    def __init__(self, estimator: Any, method: str = "sigmoid"):
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("calibration_method doit être sigmoid ou isotonic")
        self.estimator, self.method = estimator, method
        self.classes_ = np.asarray(estimator.classes_)
        self.calibrators: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalProbabilityCalibrator":
        probabilities, targets = self.estimator.predict_proba(X), np.asarray(y)
        self.calibrators = []
        for position, class_id in enumerate(self.classes_):
            binary = (targets == class_id).astype(int)
            if binary.min() == binary.max():
                raise PredictionInputError("calibration temporelle: support insuffisant pour une classe")
            if self.method == "sigmoid":
                calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
                calibrator.fit(probabilities[:, [position]], binary)
            else:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(probabilities[:, position], binary)
            self.calibrators.append(calibrator)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.estimator.predict_proba(X)
        calibrated = np.column_stack([
            calibrator.predict_proba(probabilities[:, [position]])[:, 1]
            if self.method == "sigmoid" else calibrator.predict(probabilities[:, position])
            for position, calibrator in enumerate(self.calibrators)
        ])
        calibrated = np.clip(calibrated, 1e-8, 1.0)
        return calibrated / calibrated.sum(axis=1, keepdims=True)


class PredictionNN(nn.Module):
    """Conservé pour compatibilité GPU ; P0 utilise l'ensemble tabulaire calibré."""

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 3):
        super().__init__()
        self.input_shape = (1, input_size)
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return torch.softmax(self.network(x), dim=1)


class PredictionModel:
    direction_mapping = {"bearish": 0, "neutral": 1, "bullish": 2}
    inverse_direction_mapping = {0: "bearish", 1: "neutral", 2: "bullish"}

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        self.config = custom_config or {}
        self.prediction_mode = self.config.get("prediction_mode", "ml_only")
        if self.prediction_mode not in {"ml_only", "hybrid"}:
            raise ValueError("prediction_mode doit être ml_only ou hybrid")
        default_llm_weight = 0.0 if self.prediction_mode == "ml_only" else 0.4
        default_ml_weight = 1.0 if self.prediction_mode == "ml_only" else 0.6
        self.llm_weight = float(self.config.get("llm_weight", default_llm_weight))
        self.ml_weight = float(self.config.get("ml_weight", default_ml_weight))
        if self.llm_weight < 0 or self.ml_weight < 0 or self.llm_weight + self.ml_weight == 0:
            raise ValueError("Les poids LLM/ML doivent être positifs")
        total_weight = self.llm_weight + self.ml_weight
        self.llm_weight, self.ml_weight = self.llm_weight / total_weight, self.ml_weight / total_weight
        self.calibration_method = self.config.get("calibration_method", "sigmoid")
        self.model_dir = self.config.get("model_dir", str(config.DATA_DIR / "models" / "predictions"))
        os.makedirs(self.model_dir, exist_ok=True)
        predictor_config = dict(self.config.get("market_predictor_config", {}))
        for key in ("market_data_provider", "sentiment_provider", "data_collector", "llm_client", "cache_dir", "use_gpu"):
            if key in self.config:
                predictor_config[key] = self.config[key]
        self.market_predictor = self.config.get("market_predictor") or MarketPredictor(predictor_config)
        self.market_safety_guard = self.config.get("market_safety_guard") or MarketSafetyGuard()
        self.ml_model: List[Any] = []
        self.calibration_model_templates: List[Any] = []
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.target_column = "direction"
        self.ensemble = ModelEnsemble(fusion_strategy="confidence", adjust_weights=False,
                                      min_consensus_ratio=float(self.config.get("min_consensus_ratio", 0.6)))
        self.torch_models: List[nn.Module] = []
        self.has_torch_models = False
        self.rtx_optimizer = None
        if self.config.get("use_gpu", True) and torch.cuda.is_available() and detect_rtx_gpu():
            self.rtx_optimizer = RTXOptimizer(enable_tensorrt=self.config.get("enable_tensorrt", False))

    def _prepare_pytorch_models(self, input_size: int) -> List[nn.Module]:
        models = [PredictionNN(input_size), PredictionNN(input_size, hidden_size=128)]
        if self.rtx_optimizer:
            return [self.rtx_optimizer.to_device(model) for model in models]
        return models

    def _optimize_pytorch_models(self, models: List[nn.Module]) -> List[nn.Module]:
        return [self.rtx_optimizer.optimize_for_inference(model) if self.rtx_optimizer else model.eval() for model in models]

    def _prepare_data(
        self, market_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame], for_training: bool = True,
        as_of: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        asset = str(market_data["asset"].iloc[-1]) if "asset" in market_data.columns else None
        timeframe = str(market_data["timeframe"].iloc[-1]) if "timeframe" in market_data.columns else None
        aligned, _ = align_market_and_sentiment(
            market_data, sentiment_data, as_of, asset=asset, timeframe=timeframe, require_sentiment=True,
        )
        aligned["return_1"] = aligned["close"].pct_change()
        aligned["volume_change"] = aligned["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        aligned["range_ratio"] = (aligned["high"] - aligned["low"]) / aligned["close"]
        labels: Optional[pd.Series] = None
        if for_training:
            if "direction" in market_data.columns:
                label_source = market_data.copy()
                timestamp_col = next((column for column in ("timestamp", "date", "datetime") if column in label_source.columns), None)
                if timestamp_col:
                    label_source.index = pd.to_datetime(label_source[timestamp_col], utc=True)
                labels = label_source["direction"].reindex(aligned.index).map(self.direction_mapping)
            else:
                horizon_steps = int(self.config.get("prediction_horizon_steps", 1))
                if horizon_steps < 1:
                    raise PredictionInputError("prediction_horizon_steps doit être >= 1")
                future_return = aligned["close"].shift(-horizon_steps) / aligned["close"] - 1
                labels = pd.Series(1, index=aligned.index, dtype=float)
                labels[future_return > float(self.config.get("bullish_threshold", 0.002))] = 2
                labels[future_return < -float(self.config.get("bearish_threshold", 0.002))] = 0
                labels.iloc[-horizon_steps:] = np.nan
        excluded = {"asset", "timeframe", "source", "sentiment_source", "as_of", "direction", "future_return", "direction_code"}
        features = aligned[[column for column in aligned.columns if column not in excluded and pd.api.types.is_numeric_dtype(aligned[column])]].copy()
        features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        if for_training:
            valid = labels.notna()
            features, labels = features.loc[valid], labels.loc[valid].astype(int)
            if len(features) < 20 or labels.nunique() < 2:
                raise PredictionInputError("données insuffisantes pour entraîner P4")
            self.feature_columns = list(features.columns)
        elif self.feature_columns:
            features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        return features, labels

    def _calibrate(self, estimator: Any, X: np.ndarray, y: pd.Series) -> Any:
        if self.config.get("require_temporal_calibration", False):
            return self._temporal_prefit_calibration(estimator, X, y)
        splits = min(3, max(2, len(X) // 20))
        try:
            cv = TimeSeriesSplit(n_splits=splits)
            try:
                calibrated = CalibratedClassifierCV(estimator=clone(estimator), method=self.calibration_method, cv=cv)
            except TypeError:  # scikit-learn < 1.2
                calibrated = CalibratedClassifierCV(base_estimator=clone(estimator), method=self.calibration_method, cv=cv)
            return calibrated.fit(X, y)
        except ValueError as error:
            if self.config.get("require_temporal_calibration", False):
                raise PredictionInputError(f"calibration temporelle invalide: {error}") from error
            logger.warning("Calibration indisponible, modèle brut conservé: %s", error)
            return estimator.fit(X, y)

    def _temporal_prefit_calibration(self, estimator: Any, X: np.ndarray, y: pd.Series) -> Any:
        """Calibre sur le segment chronologiquement postérieur au fit.

        ``CalibratedClassifierCV(TimeSeriesSplit)`` échoue dès qu'un pli ancien
        contient une classe rare. Cette partition explicite reste causale tout
        en refusant les historiques qui ne contiennent pas deux classes dans
        chacun des segments fit/calibration.
        """
        values = np.asarray(X)
        targets = np.asarray(y, dtype=int)
        calibration_size = max(20, int(np.ceil(len(values) * 0.20)))
        fit_end = len(values) - calibration_size
        if fit_end < 20 or len(np.unique(targets[:fit_end])) < 2 or len(np.unique(targets[fit_end:])) < 2:
            raise PredictionInputError("calibration temporelle: classes insuffisantes dans les segments causaux")
        fitted = clone(estimator).fit(values[:fit_end], targets[:fit_end])
        calibrated = TemporalProbabilityCalibrator(fitted, method=self.calibration_method)
        try:
            return calibrated.fit(values[fit_end:], targets[fit_end:])
        except ValueError as error:
            raise PredictionInputError(f"calibration temporelle invalide: {error}") from error

    def train(self, market_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame], as_of: Optional[Any] = None) -> Dict[str, Any]:
        X, y = self._prepare_data(market_data, sentiment_data, True, as_of)
        split = TimeSeriesSplit(n_splits=5)
        train_index, test_index = list(split.split(X))[-1]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        self.scaler = StandardScaler().fit(X_train)
        train_scaled, test_scaled = self.scaler.transform(X_train), self.scaler.transform(X_test)
        candidates = [
            RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        ]
        self.calibration_model_templates = candidates
        self.ml_model = [self._calibrate(model, train_scaled, y_train) for model in candidates]
        probabilities = self._predict_proba(test_scaled)
        predictions = probabilities.argmax(axis=1)
        # Calibration sur prédictions walk-forward du train, puis rapport séparé sur le test final.
        from ai_trading.llm.predictions.uncertainty_calibration import UncertaintyCalibrator
        calibrator = UncertaintyCalibrator(self)
        try:
            validation = calibrator.perform_cross_validation(
                train_scaled, y_train.to_numpy(), n_splits=3,
                min_valid_folds=int(self.config.get("min_calibrated_oos_folds", 2)),
            )
        except ValueError as error:
            if self.config.get("require_temporal_calibration", False):
                raise PredictionInputError(f"validation temporelle invalide: {error}") from error
            logger.warning("Validation walk-forward indisponible: %s", error)
            validation = {"error": str(error), "validation_type": "timeseries_walk_forward"}
        if validation.get("error"):
            if self.config.get("require_temporal_calibration", False):
                raise PredictionInputError(f"validation temporelle invalide: {validation['error']}")
            calibration_report = {"validation": validation, "test": {"error": "validation_oos_indisponible"}}
        else:
            validation_probabilities = np.asarray(validation["probabilities"], dtype=float)
            validation_targets = np.asarray(validation["targets"], dtype=int)
            calibration_report = calibrator.evaluate_validation_and_test(
                validation_probabilities, validation_targets, probabilities, y_test.to_numpy(),
                abstention_threshold=float(self.config.get("min_prediction_confidence", 0.45)),
            )
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "train_samples": int(len(X_train)), "test_samples": int(len(X_test)), "features": self.feature_columns,
            "n_models": len(self.ml_model), "validation_type": "timeseries_last_fold",
            "calibration": calibration_report,
        }
        self._save_models()
        return metrics

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.ml_model:
            return np.full((len(X), 3), 1 / 3)
        probabilities = []
        for model in self.ml_model:
            raw = model.predict_proba(X)
            expanded = np.zeros((len(X), 3), dtype=float)
            for position, class_id in enumerate(model.classes_):
                if int(class_id) in self.inverse_direction_mapping:
                    expanded[:, int(class_id)] = raw[:, position]
            probabilities.append(expanded)
        average = np.mean(probabilities, axis=0)
        return average / average.sum(axis=1, keepdims=True)

    def _get_ml_prediction(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, pd.DataFrame):
            features = data
        else:
            features = pd.DataFrame(data)
        if self.scaler is None or not self.ml_model:
            return {"direction": "neutral", "confidence": 0.0, "probabilities": {direction: 1 / 3 for direction in self.direction_mapping}, "abstain": True}
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        probability = self._predict_proba(self.scaler.transform(features))[-1]
        direction_id = int(np.argmax(probability))
        return {"direction": self.inverse_direction_mapping[direction_id], "confidence": float(probability[direction_id]),
                "probabilities": {self.inverse_direction_mapping[index]: float(value) for index, value in enumerate(probability)}}

    def _combine_predictions(self, llm_prediction: Dict[str, Any], ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        llm_probabilities = np.zeros(3, dtype=float)
        llm_confidence = float(llm_prediction.get("confidence", 0.0))
        for direction, index in self.direction_mapping.items():
            llm_probability = (1.0 - llm_confidence) / 2
            if llm_prediction.get("direction") == direction:
                llm_probability = llm_confidence
            llm_probabilities[index] = llm_probability
        ml_probabilities = np.asarray([float(ml_prediction["probabilities"].get(direction, 0.0))
                                       for direction in self.direction_mapping], dtype=float)
        distributions, weights = [ml_probabilities], [self.ml_weight]
        if self.llm_weight > 0:
            distributions.insert(0, llm_probabilities)
            weights.insert(0, self.llm_weight)
        fused = self.ensemble.fuse_probabilities(distributions, weights=weights)
        confidence = float(fused["confidence"])
        disagreement = self.llm_weight > 0 and llm_prediction.get("direction") != ml_prediction.get("direction")
        abstain = (confidence < float(self.config.get("min_prediction_confidence", 0.45))
                   or not fused["is_consensus_sufficient"]
                   or (disagreement and confidence < 0.60))
        return {**fused, "direction": "neutral" if abstain else fused["direction"],
                "abstain": abstain, "consensus": not disagreement}

    def predict(self, asset: str, timeframe: str, market_data: Optional[pd.DataFrame] = None,
                sentiment_data: Optional[pd.DataFrame] = None, as_of: Optional[Any] = None) -> Dict[str, Any]:
        if self.prediction_mode == "ml_only":
            raw_market = market_data if market_data is not None else self.market_predictor._fetch_market_data(asset, timeframe, as_of)
            raw_sentiment = sentiment_data
            if raw_sentiment is None and self.market_predictor.sentiment_provider is not None:
                raw_sentiment = self.market_predictor.sentiment_provider(asset=asset, timeframe=timeframe, as_of=as_of)
            if not self.ml_model or self.scaler is None:
                return {
                    "asset": asset.upper(), "timeframe": timeframe, "as_of": str(as_of) if as_of is not None else None,
                    "direction": "neutral", "confidence": 0.0,
                    "probabilities": {direction: 1 / 3 for direction in self.direction_mapping},
                    "abstain": True, "hybrid_status": "ml_only_untrained", "mode": "ml_only",
                    "trading_enabled": False,
                }
            try:
                features, _ = self._prepare_data(raw_market, raw_sentiment, False, as_of)
                ml_prediction = self._get_ml_prediction(features)
                confidence = float(ml_prediction["confidence"])
                abstain = confidence < float(self.config.get("min_prediction_confidence", 0.45))
                result = {
                    **ml_prediction, "direction": "neutral" if abstain else ml_prediction["direction"],
                    "abstain": abstain, "asset": asset.upper(), "timeframe": timeframe,
                    "as_of": str(as_of) if as_of is not None else None, "hybrid_status": "ml_only_ready",
                    "mode": "ml_only", "trading_enabled": False,
                    "data_version": self.market_predictor._resolve_inputs(asset, timeframe, raw_market, raw_sentiment, as_of)[1].fingerprint,
                }
                safety_sentiment = raw_sentiment
                if raw_sentiment is not None and as_of is not None:
                    timestamp_column = next((column for column in ("timestamp", "date", "datetime") if column in raw_sentiment), None)
                    if timestamp_column is not None:
                        cutoff = pd.Timestamp(as_of)
                        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
                        safety_sentiment = raw_sentiment[
                            pd.to_datetime(raw_sentiment[timestamp_column], utc=True, errors="coerce") <= cutoff
                        ]
                return self.market_safety_guard.apply(
                    result, self.market_safety_guard.assess(raw_market, safety_sentiment, as_of)
                )
            except Exception as error:
                return {
                    "asset": asset.upper(), "timeframe": timeframe, "direction": "neutral", "confidence": 0.0,
                    "abstain": True, "hybrid_status": "ml_only_degraded", "mode": "degraded",
                    "trading_enabled": False, "error": {"type": type(error).__name__, "message": str(error)},
                }
        llm_prediction = self.market_predictor.predict_market_direction(asset, timeframe, market_data, sentiment_data, as_of)
        if llm_prediction.get("mode") == "degraded" or not self.ml_model or self.scaler is None:
            result = dict(llm_prediction)
            result["hybrid_status"] = "llm_only_untrained" if self.ml_model == [] else "degraded"
            return result
        raw_market = market_data if market_data is not None else self.market_predictor._fetch_market_data(asset, timeframe, as_of)
        features, _ = self._prepare_data(raw_market, sentiment_data, False, as_of)
        ml_prediction = self._get_ml_prediction(features)
        combined = self._combine_predictions(llm_prediction, ml_prediction)
        combined.update({"asset": asset.upper(), "timeframe": timeframe, "as_of": llm_prediction.get("as_of"),
                         "llm_prediction": llm_prediction, "ml_prediction": ml_prediction,
                         "hybrid_status": "ready", "timestamp": datetime.now().isoformat()})
        return combined

    def batch_predict(self, assets: List[str], timeframe: str = "24h", **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        return {asset: self.predict(asset, timeframe, **kwargs) for asset in assets}

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_proba(X).argmax(axis=1)

    def _save_models(self, path: Optional[str] = None) -> str:
        target = path or os.path.join(self.model_dir, "prediction_model.joblib")
        joblib.dump({"models": self.ml_model, "scaler": self.scaler, "feature_columns": self.feature_columns,
                     "config": {"llm_weight": self.llm_weight, "ml_weight": self.ml_weight,
                                "prediction_mode": self.prediction_mode}}, target)
        return target

    def save_model(self, path: str) -> str:
        return self._save_models(path)

    def load_model(self, path: str) -> None:
        payload = joblib.load(path)
        self.ml_model, self.scaler = payload["models"], payload["scaler"]
        self.feature_columns = list(payload["feature_columns"])
        saved_config = payload.get("config", {})
        self.prediction_mode = saved_config.get("prediction_mode", self.prediction_mode)

    def _fetch_recent_data(self, asset: str, timeframe: str, as_of: Optional[Any] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        sentiment = (self.market_predictor.sentiment_provider(asset=asset, timeframe=timeframe, as_of=as_of)
                     if self.market_predictor.sentiment_provider else None)
        return self.market_predictor._fetch_market_data(asset, timeframe, as_of), sentiment

    def get_cache_stats(self) -> Dict[str, Any]:
        return {"market_predictor_cache": self.market_predictor.get_cache_stats()}

    def cleanup_resources(self) -> None:
        self.market_predictor.cleanup_resources()
        if self.rtx_optimizer:
            self.rtx_optimizer.clear_cache()
