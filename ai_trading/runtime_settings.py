"""Parametres runtime communs aux points d'entree du projet.

Ce module est volontairement independant des dependances ML et web afin que les
gardes-fous soient testables sans charger le pipeline de trading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


class ConfigurationError(ValueError):
    """Le processus ne peut pas demarrer avec cette configuration."""


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"Valeur booleenne invalide: {value!r}")


@dataclass(frozen=True)
class RuntimeSettings:
    trading_mode: str
    live_trading_enabled: bool
    allow_synthetic_data: bool
    api_bind_host: str
    api_port: int
    debug: bool
    binance_api_key: str | None
    binance_api_secret: str | None

    @property
    def live_orders_enabled(self) -> bool:
        return (
            self.trading_mode == "live"
            and self.live_trading_enabled
            and bool(self.binance_api_key)
            and bool(self.binance_api_secret)
        )


def get_runtime_settings(environ: Mapping[str, str] | None = None) -> RuntimeSettings:
    environment = os.environ if environ is None else environ
    trading_mode = environment.get("TRADING_MODE", "paper").strip().lower()
    if trading_mode not in {"paper", "live"}:
        raise ConfigurationError("TRADING_MODE doit etre 'paper' ou 'live'.")
    try:
        api_port = int(environment.get("API_PORT", "8000"))
    except ValueError as exc:
        raise ConfigurationError("API_PORT doit etre un entier.") from exc
    if not 1 <= api_port <= 65535:
        raise ConfigurationError("API_PORT doit etre compris entre 1 et 65535.")
    return RuntimeSettings(
        trading_mode=trading_mode,
        live_trading_enabled=_as_bool(environment.get("LIVE_TRADING_ENABLED"), False),
        allow_synthetic_data=_as_bool(environment.get("ALLOW_SYNTHETIC_DATA"), False),
        api_bind_host=environment.get("API_BIND_HOST", "127.0.0.1"),
        api_port=api_port,
        debug=_as_bool(environment.get("APP_DEBUG"), False),
        binance_api_key=environment.get("BINANCE_API_KEY") or None,
        binance_api_secret=environment.get("BINANCE_API_SECRET") or None,
    )
