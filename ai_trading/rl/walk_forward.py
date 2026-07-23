"""Contrats stricts de validation walk-forward, sans sélection sur le test."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import median
from typing import Any, Iterable

import pandas as pd

from ai_trading.rl.real_market_report import validate_ohlcv


@dataclass(frozen=True)
class WalkForwardWindow:
    """Une fenêtre chronologique train / validation / test disjointe."""

    index: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def build_walk_forward_windows(
    data: pd.DataFrame,
    train_candles: int,
    validation_candles: int,
    test_candles: int,
    step_candles: int | None = None,
) -> list[WalkForwardWindow]:
    """Découpe des fenêtres sans mélange temporel ni test dupliqué."""
    if min(train_candles, validation_candles, test_candles) < 2:
        raise ValueError("Chaque partition doit contenir au moins deux bougies")
    step = test_candles if step_candles is None else step_candles
    if step < test_candles:
        raise ValueError("Le pas doit être >= au test pour ne pas réévaluer les mêmes bougies")
    frame = validate_ohlcv(data)
    width = train_candles + validation_candles + test_candles
    windows: list[WalkForwardWindow] = []
    start = 0
    while start + width <= len(frame):
        train_end = start + train_candles
        validation_end = train_end + validation_candles
        test_end = validation_end + test_candles
        windows.append(
            WalkForwardWindow(
                index=len(windows),
                train=frame.iloc[start:train_end].copy(),
                validation=frame.iloc[train_end:validation_end].copy(),
                test=frame.iloc[validation_end:test_end].copy(),
            )
        )
        start += step
    if not windows:
        raise ValueError("Pas assez de bougies pour créer une fenêtre walk-forward")
    return windows


def validation_objective(metrics: dict[str, Any]) -> float:
    """Score déterministe utilisé exclusivement sur validation.

    Les performances de test ne sont volontairement pas acceptées ici afin de
    bloquer toute sélection a posteriori.
    """
    strategy = metrics["strategy"]
    benchmark = metrics.get("risk_matched_buy_and_hold", metrics["buy_and_hold"])
    return float(
        strategy["total_return"]
        - benchmark["total_return"]
        + 0.01 * strategy.get("sharpe_ratio", 0.0)
        + 0.005 * strategy.get("sortino_ratio", 0.0)
        + 0.10 * strategy["max_drawdown"]
    )


def validation_eligibility(
    metrics: dict[str, Any], min_closed_trades: int = 5, max_agent_order_rate: float = 0.35,
) -> tuple[bool, str | None]:
    """Écarte avant le test les politiques inactives ou dégénérées."""
    diversity = metrics.get("action_diversity", {})
    trades = metrics.get("closed_trade_metrics", {})
    strategy = metrics.get("strategy", {})
    benchmark = metrics.get("risk_matched_buy_and_hold", metrics.get("buy_and_hold", {}))
    if float(strategy.get("total_return", float("-inf"))) <= float(benchmark.get("total_return", float("inf"))):
        return False, "rendement net inférieur ou égal au benchmark sur validation"
    if not diversity.get("passed") or not diversity.get("directional_passed", diversity.get("passed")):
        return False, "diversité d'actions insuffisante sur validation"
    if int(trades.get("closed_trade_count", 0)) < min_closed_trades:
        return False, f"moins de {min_closed_trades} trades clôturés sur validation"
    if "profit_factor" in trades and float(trades["profit_factor"]) <= 1.0:
        return False, "profit factor non rentable sur validation"
    if "expectancy" in trades and float(trades["expectancy"]) <= 0.0:
        return False, "expectancy non positive sur validation"
    executed = int(metrics.get("execution_reason_counts", {}).get("agent_order", 0))
    candles = int(metrics.get("test_candles", 0))
    if candles and executed / candles > max_agent_order_rate:
        return False, "turnover agent excessif sur validation"
    return True, None


def aggregate_validation_metrics(metrics_by_seed: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Agrège des validations indépendantes, sans jamais lire le test.

    Les rendements sont médians (robustes aux seeds extrêmes) et les garde-fous
    booléens exigent un succès sur chaque seed.
    """
    entries = list(metrics_by_seed)
    if not entries:
        raise ValueError("Au moins une validation par seed est requise")

    def median_metrics(path: str, fallback: str | None = None) -> dict[str, float]:
        keys = entries[0].get(path, entries[0].get(fallback or "", {}))
        return {
            key: float(median(entry.get(path, entry.get(fallback or "", {}))[key] for entry in entries))
            for key, value in keys.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    trade_keys = entries[0].get("closed_trade_metrics", {})
    diversity_keys = entries[0].get("action_diversity", {})
    trades = {
        key: (
            bool(all(entry.get("closed_trade_metrics", {}).get(key) for entry in entries))
            if isinstance(value, bool)
            else float(median(entry.get("closed_trade_metrics", {}).get(key, 0.0) or 0.0 for entry in entries))
        )
        for key, value in trade_keys.items()
    }
    diversity = {
        key: bool(all(entry.get("action_diversity", {}).get(key) for entry in entries))
        if isinstance(value, bool)
        else value
        for key, value in diversity_keys.items()
    }
    return {
        "strategy": median_metrics("strategy"),
        "buy_and_hold": median_metrics("buy_and_hold"),
        "risk_matched_buy_and_hold": median_metrics("risk_matched_buy_and_hold", "buy_and_hold"),
        "closed_trade_metrics": trades,
        "action_diversity": diversity,
        "validation_seed_count": len(entries),
    }


def select_candidate_on_validation(
    candidates: Iterable[dict[str, Any]], min_closed_trades: int = 5, max_agent_order_rate: float = 0.35,
) -> dict[str, Any]:
    """Sélectionne un candidat à partir de ses seuls résultats validation."""
    entries = list(candidates)
    if not entries:
        raise ValueError("Aucun candidat à sélectionner")
    eligible = []
    for candidate in entries:
        if "validation" not in candidate or "test" in candidate:
            raise ValueError("Un candidat doit contenir uniquement sa validation avant le test")
        is_eligible, reason = validation_eligibility(
            candidate["validation"], min_closed_trades=min_closed_trades,
            max_agent_order_rate=max_agent_order_rate,
        )
        candidate["validation_eligible"] = is_eligible
        candidate["validation_rejection_reason"] = reason
        if is_eligible:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("Aucun candidat ne satisfait les garde-fous de validation")
    return max(eligible, key=lambda candidate: validation_objective(candidate["validation"]))


def phase4_gate(window_results: Iterable[dict[str, Any]], min_windows: int = 3) -> dict[str, Any]:
    """Calcule les critères de passage P4 sur des tests hors-échantillon figés."""
    rows = list(window_results)
    if len(rows) < min_windows:
        return {
            "passed": False,
            "window_count": len(rows),
            "required_windows": min_windows,
            "reasons": ["Nombre de fenêtres hors-échantillon insuffisant"],
        }
    checks = []
    for row in rows:
        if not row.get("test"):
            checks.append({
                "window": row["window"],
                "validation_rejected": True,
                "beats_buy_and_hold": False,
                "acceptable_drawdown": False,
                "profit_factor_above_one": False,
                "minimum_closed_trades": False,
                "action_diversity": False,
                "positive_sharpe": False,
                "positive_sortino": False,
            })
            continue
        strategy = row["test"]["strategy"]
        benchmark = row["test"].get("risk_matched_buy_and_hold", row["test"]["buy_and_hold"])
        trades = row["test"].get("closed_trade_metrics", {})
        diversity = row["test"].get("action_diversity", {})
        checks.append({
            "window": row["window"],
            "beats_buy_and_hold": strategy["total_return"] > benchmark["total_return"],
            "acceptable_drawdown": strategy["max_drawdown"] >= -0.15,
            "profit_factor_above_one": (trades.get("profit_factor") or 0.0) > 1.0,
            "minimum_closed_trades": bool(trades.get("minimum_trade_count_met")),
            "action_diversity": bool(diversity.get("directional_passed", diversity.get("passed"))),
            "positive_sharpe": strategy.get("sharpe_ratio", 0.0) > 0.0,
            "positive_sortino": strategy.get("sortino_ratio", 0.0) > 0.0,
        })
    beat_count = sum(check["beats_buy_and_hold"] for check in checks)
    required_beats = ceil(len(checks) * 2 / 3)
    hard_checks = ("acceptable_drawdown", "profit_factor_above_one", "minimum_closed_trades", "action_diversity")
    reasons = []
    rejected_windows = [check["window"] for check in checks if check.get("validation_rejected")]
    if rejected_windows:
        reasons.append(
            f"Aucun candidat validé avant test sur fenêtres {rejected_windows}"
        )
    if beat_count < required_beats:
        reasons.append(f"Rendement net supérieur au Buy & Hold dans {beat_count}/{len(checks)} fenêtres, requis {required_beats}")
    for check_name in hard_checks:
        failures = [check["window"] for check in checks if not check[check_name]]
        if failures:
            reasons.append(f"{check_name} en échec sur fenêtres {failures}")
    stability = sum(check["positive_sharpe"] and check["positive_sortino"] for check in checks) >= required_beats
    if not stability:
        reasons.append("Sharpe et Sortino positifs insuffisamment stables")
    return {
        "passed": not reasons,
        "window_count": len(checks),
        "required_windows": min_windows,
        "required_beating_windows": required_beats,
        "checks": checks,
        "reasons": reasons,
    }
