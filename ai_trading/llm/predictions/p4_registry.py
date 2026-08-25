"""Registre en mémoire, lecture seule côté API, pour les artefacts P4."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple


class P4Registry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._predictions: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._explanations: Dict[str, Dict[str, Any]] = {}

    def record_prediction(self, prediction: Dict[str, Any]) -> None:
        asset, timeframe = str(prediction["asset"]).upper(), str(prediction["timeframe"])
        snapshot = dict(prediction)
        snapshot["trading_enabled"] = False
        with self._lock:
            self._predictions[(asset, timeframe)] = snapshot

    def record_explanation(self, prediction_id: str, explanation: Dict[str, Any]) -> None:
        with self._lock:
            self._explanations[prediction_id] = dict(explanation)

    def prediction(self, asset: str, timeframe: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            value = self._predictions.get((asset.upper(), timeframe))
            return dict(value) if value else None

    def explanation(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            value = self._explanations.get(prediction_id)
            return dict(value) if value else None


registry = P4Registry()
