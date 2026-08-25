"""Prédictions P4 causales à partir de données P1/P2 horodatées.

Le fournisseur LLM est injectable. Sans fournisseur externe, le prédicteur
retourne un état dégradé traçable ; les doubles de test restent dans les tests
et ne constituent jamais un chemin runtime de production.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import ai_trading.config as config
from ai_trading.llm.predictions.cache_manager import CacheManager
from ai_trading.llm.predictions.market_safety import MarketSafetyGuard
from ai_trading.llm.predictions.prediction_contract import (
    PredictionContext,
    PredictionInputError,
    align_market_and_sentiment,
    input_fingerprint,
)
from ai_trading.llm.predictions.performance_analysis import PerformanceProfiler, profile
from ai_trading.llm.predictions.rtx_optimizer import RTXOptimizer, detect_rtx_gpu
from ai_trading.utils import setup_logger

logger = setup_logger("market_predictor")


class P1MarketDataProvider:
    """Adaptateur P1 explicite vers les séries de prix persistables/traçables."""

    COIN_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano", "XRP": "ripple"}

    def __init__(self, collector: Optional[Any] = None, days: int = 30):
        self.collector = collector
        self.days = days

    def fetch(self, asset: str, timeframe: str, as_of: Optional[Any] = None) -> pd.DataFrame:
        from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector

        coin_id = self.COIN_IDS.get(asset.upper(), asset.lower())
        collector = self.collector or EnhancedDataCollector()
        frame = collector.get_merged_price_data(coin_id, days=self.days, include_fear_greed=True, mock_data=False)
        if frame.empty:
            raise PredictionInputError(f"P1: aucune donnée réelle disponible pour {asset}")
        result = frame.copy().reset_index().rename(columns={frame.index.name or "index": "timestamp"})
        result["source"] = result["source"].astype(str) if "source" in result.columns else "p1"
        result["asset"] = asset.upper()
        result["timeframe"] = timeframe
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            result = result[pd.to_datetime(result["timestamp"], utc=True) <= cutoff]
        return result


class P2SentimentDataProvider:
    """Lit les observations P2 persistées ; aucune requête textuelle n'est exécutée par P4."""

    def __init__(self, observations_path: Optional[Any] = None):
        self.observations_path = Path(observations_path or config.DATA_DIR / "sentiment" / "observations.parquet")

    def fetch(self, asset: str, timeframe: str, as_of: Optional[Any] = None) -> pd.DataFrame:
        if not self.observations_path.exists():
            raise PredictionInputError(f"P2: observations introuvables: {self.observations_path}")
        suffix = self.observations_path.suffix.lower()
        if suffix == ".parquet":
            observations = pd.read_parquet(self.observations_path)
        elif suffix == ".csv":
            observations = pd.read_csv(self.observations_path)
        else:
            raise PredictionInputError("P2: format d'observations attendu .parquet ou .csv")
        if observations.empty:
            raise PredictionInputError("P2: aucune observation disponible")
        if "asset" not in observations.columns or "timeframe" not in observations.columns:
            raise PredictionInputError("P2: actif et timeframe obligatoires dans les observations")
        selected = observations[
            observations["asset"].astype(str).str.upper().eq(asset.upper())
            & observations["timeframe"].astype(str).eq(timeframe)
        ].copy()
        timestamp_column = next((column for column in ("timestamp", "date", "datetime") if column in selected.columns), None)
        if timestamp_column is None:
            raise PredictionInputError("P2: horodatage obligatoire dans les observations")
        cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        selected = selected[pd.to_datetime(selected[timestamp_column], utc=True, errors="coerce") <= cutoff]
        if selected.empty:
            raise PredictionInputError(f"P2: aucune observation pour {asset}/{timeframe} avant as_of")
        return selected


class MarketPredictor:
    """Prédicteur directionnel avec contrat P1/P2, cache frais et fallback local."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        self.config = custom_config or {}
        self.model_name = self.config.get("model_name", "offline-heuristic")
        self.temperature = float(self.config.get("temperature", 0.1))
        self.max_tokens = int(self.config.get("max_tokens", 1000))
        self.client = self.config.get("llm_client")
        self.require_sentiment = bool(self.config.get("require_sentiment", True))
        self.market_safety_guard = self.config.get("market_safety_guard") or MarketSafetyGuard()
        self.llm_timeout_seconds = float(self.config.get("llm_timeout_seconds", 10.0))
        self.market_data_provider = self.config.get("market_data_provider") or P1MarketDataProvider(
            collector=self.config.get("data_collector"), days=int(self.config.get("market_history_days", 30))
        )
        self.sentiment_provider: Callable[..., pd.DataFrame] = self.config.get("sentiment_provider") or P2SentimentDataProvider(
            self.config.get("p2_observations_path")
        ).fetch
        # Les attributs restent disponibles pour compatibilité, mais P4 ne leur envoie plus de requête texte.
        self.news_analyzer = self.config.get("news_analyzer")
        self.social_analyzer = self.config.get("social_analyzer")
        self.predictions_history: Dict[str, Dict[str, Any]] = {}
        cache_dir = self.config.get("cache_dir", str(config.DATA_DIR / "cache" / "predictions"))
        self.cache = CacheManager(
            capacity=int(self.config.get("cache_capacity", 100)),
            ttl=int(self.config.get("cache_ttl", 3600)),
            persist_path=cache_dir,
            enable_disk_cache=bool(self.config.get("enable_disk_cache", True)),
        )
        self._cache_version_lock = threading.RLock()
        self._input_versions: Dict[Tuple[str, str], str] = {}
        self._scope_prediction_locks: Dict[Tuple[str, str], Any] = {}
        self.profiler = PerformanceProfiler()
        self.rtx_optimizer = None
        if self.config.get("use_gpu", True) and torch.cuda.is_available() and detect_rtx_gpu():
            self.rtx_optimizer = RTXOptimizer(
                device_id=self.config.get("gpu_device_id"),
                enable_tensor_cores=self.config.get("enable_tensor_cores", True),
                enable_half_precision=self.config.get("enable_half_precision", True),
                optimize_memory=self.config.get("optimize_memory", True),
                enable_tensorrt=self.config.get("enable_tensorrt", False),
            )

    def _get_ttl_for_timeframe(self, timeframe: str) -> int:
        try:
            value, unit = int(timeframe[:-1]), timeframe[-1].lower()
        except (ValueError, IndexError):
            return 300
        if unit == "m":
            return max(30, min(value * 12, 300))
        if unit == "h":
            return max(60, min(value * 60, 1800))
        if unit == "d":
            return max(300, min(value * 600, 3600))
        return 300

    def _sync_cache_version(self, context: PredictionContext) -> None:
        """Invalide les résultats d'un actif/horizon dès qu'une entrée P1/P2 change."""
        scope = (context.asset, context.timeframe)
        prefix = f"p4:v2:{context.asset}:{context.timeframe}:"
        with self._cache_version_lock:
            previous = self._input_versions.get(scope)
            if previous is not None and previous != context.fingerprint:
                self.cache.invalidate_prefix(prefix)
            self._input_versions[scope] = context.fingerprint

    def _get_scope_prediction_lock(self, context: PredictionContext) -> Any:
        """Empêche qu'une ancienne version soit réinsérée pendant son invalidation."""
        scope = (context.asset, context.timeframe)
        with self._cache_version_lock:
            return self._scope_prediction_locks.setdefault(scope, threading.RLock())

    def _invalidate_cache_after_input_error(self, asset: str, timeframe: str) -> None:
        """Une erreur de flux P1/P2 interdit de réutiliser un résultat périmé."""
        prefix = f"p4:v2:{asset.upper()}:{timeframe}:"
        with self._cache_version_lock:
            self.cache.invalidate_prefix(prefix)
            self._input_versions.pop((asset.upper(), timeframe), None)

    def _resolve_inputs(
        self,
        asset: str,
        timeframe: str,
        market_data: Optional[pd.DataFrame],
        sentiment_data: Optional[pd.DataFrame],
        as_of: Optional[Any],
    ) -> tuple[pd.DataFrame, PredictionContext]:
        raw_market = market_data if market_data is not None else self.market_data_provider.fetch(asset, timeframe, as_of)
        raw_sentiment = sentiment_data
        if raw_sentiment is None and self.sentiment_provider is not None:
            raw_sentiment = self.sentiment_provider(asset=asset, timeframe=timeframe, as_of=as_of)
        aligned, metadata = align_market_and_sentiment(
            raw_market, raw_sentiment, as_of, asset=asset, timeframe=timeframe,
            require_sentiment=self.require_sentiment,
        )
        if len(aligned) < 2:
            raise PredictionInputError("P4: au moins deux observations marché sont requises")
        cutoff = metadata["as_of"]
        fingerprint = input_fingerprint(asset, timeframe, aligned, cutoff)
        return aligned, PredictionContext(
            asset=asset.upper(), timeframe=timeframe, as_of=cutoff, fingerprint=fingerprint,
            data_sources=tuple(sorted(aligned["source"].unique())),
            sentiment_sources=tuple(sorted(aligned["sentiment_source"].unique())),
            sentiment_freshness_seconds=float(metadata.get("sentiment_freshness_seconds", float("inf"))),
        )

    def _build_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        closes = data["close"].astype(float)
        momentum = float(closes.iloc[-1] / closes.iloc[-2] - 1.0)
        returns = closes.pct_change().dropna()
        volatility_value = float(returns.tail(min(20, len(returns))).std(ddof=0)) if not returns.empty else 0.0
        return {
            "momentum": momentum,
            "sentiment_score": float(data["sentiment_score"].iloc[-1]),
            "volatility": "high" if volatility_value > 0.04 else "medium" if volatility_value > 0.015 else "low",
            "volatility_value": volatility_value,
        }

    @profile(output_dir=str(config.DATA_DIR / "profiling" / "market_predictor"))
    def predict_market_direction(
        self,
        asset: str,
        timeframe: str = "24h",
        market_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
        as_of: Optional[Any] = None,
    ) -> Dict[str, Any]:
        self.profiler.start_profiling()
        context: Optional[PredictionContext] = None
        try:
            data, context = self._resolve_inputs(asset, timeframe, market_data, sentiment_data, as_of)
            with self._get_scope_prediction_lock(context):
                self._sync_cache_version(context)
                cache_key = f"p4:v2:{context.asset}:{timeframe}:{context.fingerprint}"

                def compute_prediction() -> Dict[str, Any]:
                    prompt = self._format_prompt(
                        data, {"sentiment_score": float(data["sentiment_score"].iloc[-1]), "as_of": context.as_of},
                        {}, context.asset, timeframe,
                    )
                    prediction = self._request_valid_prediction(prompt, self._build_context(data), context.asset, timeframe)
                    prediction.update(
                        {
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.now().isoformat(),
                            "as_of": context.as_of,
                            "input_fingerprint": context.fingerprint,
                            "data_sources": list(context.data_sources),
                            "sentiment_sources": list(context.sentiment_sources),
                            "sentiment_score": float(data["sentiment_score"].iloc[-1]),
                            "sentiment_freshness_seconds": context.sentiment_freshness_seconds,
                            "mode": getattr(self.client, "mode", "external"),
                            "data_version": context.fingerprint,
                            "trading_enabled": False,
                        }
                    )
                    prediction["confidence"] = self.get_confidence_score(prediction)
                    prediction["confidence_label"] = self._confidence_label(prediction["confidence"])
                    prediction["abstain"] = False
                    prediction = self.market_safety_guard.apply(
                        prediction, self.market_safety_guard.assess(data, as_of=context.as_of)
                    )
                    if self.rtx_optimizer:
                        prediction["gpu_info"] = self.rtx_optimizer.get_optimization_info()
                    self.predictions_history[prediction["id"]] = prediction
                    self.profiler.record_metrics()
                    prediction["performance_metrics"] = self.profiler.get_summary()
                    return prediction

                return self.cache.get_or_compute(
                    cache_key, compute_prediction, self._get_ttl_for_timeframe(timeframe)
                )
        except Exception as error:
            if context is None:
                self._invalidate_cache_after_input_error(asset, timeframe)
            logger.error("P4 prediction failed for %s/%s: %s", asset, timeframe, error)
            return self._degraded_prediction(asset, timeframe, as_of, error)
        finally:
            self.profiler.record_metrics()

    def _degraded_prediction(self, asset: str, timeframe: str, as_of: Optional[Any], error: Exception) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()), "asset": asset.upper(), "timeframe": timeframe,
            "direction": "neutral", "confidence": 0.0, "confidence_label": "low",
            "abstain": True, "factors": [], "contradictions": ["data_or_provider_failure"],
            "volatility": "unknown", "as_of": str(as_of) if as_of is not None else None,
            "error": {"type": type(error).__name__, "message": str(error)}, "mode": "degraded", "trading_enabled": False,
            "timestamp": datetime.now().isoformat(),
        }

    def _call_llm_with_retry(self, prompt: str, context: Dict[str, Any], max_retries: int = 3) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self._call_llm(prompt, context)
            except Exception as error:
                last_error = error
                if attempt + 1 < max_retries:
                    time.sleep(min(0.25 * (2 ** attempt), 1.0))
        raise RuntimeError("LLM provider unavailable") from last_error

    def _request_valid_prediction(self, prompt: str, context: Dict[str, Any], asset: str,
                                  timeframe: str, max_retries: int = 3) -> Dict[str, Any]:
        """Répète également les réponses JSON invalides, puis laisse le mode dégradé prendre le relais."""
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self._call_llm(prompt, context)
                parsed = self._parse_prediction(response, asset, timeframe)
                if parsed.get("error"):
                    raise ValueError(f"LLM payload invalide: {parsed['error']}")
                return parsed
            except Exception as error:
                last_error = error
                if attempt + 1 < max_retries:
                    time.sleep(min(0.25 * (2 ** attempt), 1.0))
        raise RuntimeError("LLM provider unavailable or invalid payload") from last_error

    def _call_llm(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        if self.client is None:
            raise RuntimeError("LLM client disabled; utilisez PredictionModel en mode ml_only")

        def request() -> str:
            if hasattr(self.client, "complete"):
                return self.client.complete(prompt, context or {})
            if hasattr(self.client, "chat"):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            raise TypeError("LLM client must implement complete() or chat.completions.create()")

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p4_llm")
        future = executor.submit(request)
        try:
            return future.result(timeout=self.llm_timeout_seconds)
        except FuturesTimeoutError as error:
            future.cancel()
            raise TimeoutError(f"LLM timeout après {self.llm_timeout_seconds}s") from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _format_prompt(self, data: pd.DataFrame, news_sentiment: Dict[str, Any], social_sentiment: Dict[str, Any], asset: str, timeframe: str) -> str:
        del social_sentiment
        latest = data.iloc[-1]
        return (
            "Agissez en tant qu'analyste financier expert. Répondez JSON uniquement. "
            f"asset={asset}; timeframe={timeframe}; as_of={news_sentiment.get('as_of', latest.name)}; close={latest['close']}; "
            f"volume={latest['volume']}; sentiment={news_sentiment.get('sentiment_score', 0)}. "
            "Schéma: direction(bullish|bearish|neutral), confidence[0,1], factors[], contradictions[], volatility."
        )

    def _parse_prediction(self, response: str, asset: str, timeframe: str) -> Dict[str, Any]:
        try:
            payload = json.loads(response)
            direction = str(payload.get("direction", "neutral")).lower()
            if direction not in {"bullish", "bearish", "neutral"}:
                raise ValueError("direction invalide")
            raw_confidence = payload.get("confidence", 0.0)
            if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
                raise ValueError("confidence numérique obligatoire")
            confidence = float(raw_confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence hors [0,1]")
            factors = payload.get("factors", payload.get("key_factors", []))
            if not isinstance(factors, list):
                raise ValueError("factors doit être une liste")
            contradictions = payload.get("contradictions", []) or []
            if not isinstance(contradictions, list):
                raise ValueError("contradictions doit être une liste")
            return {"asset": asset, "timeframe": timeframe, "direction": direction, "confidence": confidence,
                    "factors": [str(value) for value in factors], "contradictions": [str(value) for value in contradictions],
                    "volatility": str(payload.get("volatility", "unknown")), "raw_response": response}
        except Exception as error:
            return {"asset": asset, "timeframe": timeframe, "direction": "neutral", "confidence": 0.0,
                    "factors": [], "contradictions": ["invalid_llm_payload"], "volatility": "unknown",
                    "error": str(error), "raw_response": response, "abstain": True}

    @staticmethod
    def _confidence_label(value: float) -> str:
        return "high" if value >= 0.7 else "medium" if value >= 0.4 else "low"

    def get_confidence_score(self, prediction: Dict[str, Any]) -> float:
        base = float(prediction.get("confidence", 0.0))
        sentiment = float(prediction.get("sentiment_score", 0.0))
        direction = prediction.get("direction")
        aligned = (direction == "bullish" and sentiment >= 0) or (direction == "bearish" and sentiment <= 0) or direction == "neutral"
        penalty = 0.15 if prediction.get("contradictions") else 0.0
        return float(np.clip(base + (0.05 if aligned else -0.10) - penalty, 0.0, 1.0))

    def _fetch_market_data(self, asset: str, timeframe: str, as_of: Optional[Any] = None) -> pd.DataFrame:
        return self.market_data_provider.fetch(asset, timeframe, as_of)

    def batch_predict_directions(self, assets: Iterable[str], timeframe: str = "24h", **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        return {asset: self.predict_market_direction(asset, timeframe, **kwargs) for asset in assets}

    def generate_market_insights(self, asset: str, timeframe: str = "7d", **kwargs: Any) -> Dict[str, Any]:
        prediction = self.predict_market_direction(asset, timeframe, **kwargs)
        return {"id": str(uuid.uuid4()), "asset": asset.upper(), "timestamp": datetime.now().isoformat(),
                "as_of": prediction.get("as_of"), "insights": prediction.get("factors", []),
                "confidence": prediction.get("confidence"), "mode": prediction.get("mode")}

    def explain_prediction(self, prediction_id: str) -> Dict[str, Any]:
        prediction = self.predictions_history.get(prediction_id)
        if prediction is None:
            return {"error": "Prédiction non trouvée"}
        return {"prediction_id": prediction_id, "asset": prediction["asset"], "direction": prediction["direction"],
                "explanation": {"factors": prediction.get("factors", []), "contradictions": prediction.get("contradictions", []),
                                "mode": prediction.get("mode")}, "timestamp": datetime.now().isoformat()}

    def purge_cache(self) -> None:
        self.cache.purge_expired()

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache.get_stats()

    def cleanup_resources(self) -> None:
        if self.rtx_optimizer:
            self.rtx_optimizer.clear_cache()
