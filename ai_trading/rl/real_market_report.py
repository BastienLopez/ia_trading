"""Rapports reproductibles d'évaluation RL sur données de marché réelles.

Ce module ne génère aucune donnée synthétique : le script associé sauvegarde les
bougies téléchargées, la configuration et les résultats afin qu'un rapport puisse
être audité ou régénéré ultérieurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from ai_trading.rl.trade_ledger import FifoTradeLedger
from ai_trading.rl.policy_validation import action_diversity_metrics
from ai_trading.rl.market_regime import regime_summary

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERIODS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "1d": 365,
}


@dataclass(frozen=True)
class EvaluationRun:
    """Séries produites lors d'une évaluation hors-échantillon."""

    dates: pd.DatetimeIndex
    equity: np.ndarray
    benchmark_equity: np.ndarray
    rewards: list[float]
    actions: list[float | int]
    executions: list[bool]
    execution_reasons: list[str | None]
    fills: list[dict]
    action_type: str = "discrete"
    n_discrete_actions: int = 5
    risk_matched_benchmark_equity: np.ndarray | None = None
    baseline_equity: dict[str, np.ndarray] | None = None
    market_regimes: list[str] | None = None


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Retourne une série OHLCV triée et valide, sans prix inventé."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Colonnes OHLCV manquantes: " + ", ".join(sorted(missing)))

    frame = data.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Les données OHLCV doivent avoir un index DatetimeIndex")
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    if frame.empty or (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLCV invalide: les prix doivent être finis et strictement positifs")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Les bougies doivent être ordonnées chronologiquement")
    return frame


def chronological_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les bougies sans fuite temporelle entre entraînement et test."""
    if not 0.5 <= train_ratio < 1:
        raise ValueError("train_ratio doit être compris entre 0.5 et 1")
    frame = validate_ohlcv(data)
    split_at = int(len(frame) * train_ratio)
    if split_at < 60 or len(frame) - split_at < 60:
        raise ValueError("Au moins 60 bougies sont requises dans train et test")
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def buy_and_hold_equity(
    prices: pd.Series | np.ndarray,
    initial_balance: float,
    fee: float,
    exposure_fraction: float = 1.0,
) -> np.ndarray:
    """Valorise un Buy & Hold avec frais d'entrée et exposition explicite."""
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("La série de prix du benchmark doit être finie, positive et contenir au moins deux points")
    if initial_balance <= 0 or not 0 <= fee < 1 or not 0 < exposure_fraction <= 1:
        raise ValueError("Solde initial ou frais invalides")
    invested_cash = initial_balance * exposure_fraction
    quantity = invested_cash / (values[0] * (1 + fee))
    return initial_balance - invested_cash + quantity * values


def discrete_action_label(action: int | float, n_discrete_actions: int = 5) -> str:
    """Retourne un libellé auditable HOLD / BUY / SELL pour un indice discret."""
    value = int(action)
    if value == 0:
        return "HOLD"
    if 1 <= value <= n_discrete_actions:
        return f"BUY {value / n_discrete_actions:.0%}"
    if n_discrete_actions < value <= 2 * n_discrete_actions:
        return f"SELL {(value - n_discrete_actions) / n_discrete_actions:.0%}"
    return f"INVALID {value}"


def calculate_metrics(equity: np.ndarray, periods_per_year: int) -> dict[str, float]:
    """Calcule des métriques à partir d'une courbe de valeur marquée au marché."""
    values = np.asarray(equity, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Courbe de portefeuille invalide")
    returns = np.diff(values) / values[:-1]
    drawdowns = values / np.maximum.accumulate(values) - 1.0
    volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year)) if returns.std(ddof=0) > 0 else 0.0
    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std(ddof=0) * np.sqrt(periods_per_year))
        if len(downside) and downside.std(ddof=0) > 0
        else 0.0
    )
    return {
        "initial_value": float(values[0]),
        "final_value": float(values[-1]),
        "total_return": float(values[-1] / values[0] - 1.0),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(drawdowns.min()),
    }


def save_report(
    output_dir: Path,
    symbol: str,
    timeframe: str,
    raw_data: pd.DataFrame,
    run: EvaluationRun,
    transaction_fee: float,
    training_metadata: dict[str, Any],
    data_source: str = "CCXT public OHLCV",
) -> dict[str, Path]:
    """Sauvegarde les données, métriques et graphes d'un run vérifiable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_matched_benchmark = (
        run.risk_matched_benchmark_equity
        if run.risk_matched_benchmark_equity is not None
        else run.benchmark_equity
    )
    if (
        len(run.dates) != len(run.equity)
        or len(run.equity) != len(run.benchmark_equity)
        or len(run.equity) != len(risk_matched_benchmark)
    ):
        raise ValueError("Dates, portefeuille et benchmark doivent avoir la même longueur")

    periods = PERIODS_PER_YEAR.get(timeframe)
    if periods is None:
        raise ValueError(f"Timeframe non supporté pour les métriques: {timeframe}")
    strategy_metrics = calculate_metrics(run.equity, periods)
    benchmark_metrics = calculate_metrics(run.benchmark_equity, periods)
    risk_matched_metrics = calculate_metrics(risk_matched_benchmark, periods)
    baseline_equity = run.baseline_equity or {}
    if any(len(values) != len(run.equity) for values in baseline_equity.values()):
        raise ValueError("Chaque baseline doit couvrir exactement la période évaluée")
    baseline_metrics = {
        name: calculate_metrics(np.asarray(values, dtype=float), periods)
        for name, values in baseline_equity.items()
    }
    action_labels = [
        discrete_action_label(action, run.n_discrete_actions)
        if run.action_type == "discrete"
        else ("SELL" if float(action) < -0.05 else "BUY" if float(action) > 0.05 else "HOLD")
        for action in run.actions
    ]
    equity_frame = pd.DataFrame(
        {
            "strategy_equity": run.equity,
            # Compatibilite des consommateurs existants; il s'agit du benchmark
            # Buy & Hold a 100 % d'exposition, desormais explicite ci-dessous.
            "buy_hold_equity": run.benchmark_equity,
            "buy_hold_full_equity": run.benchmark_equity,
            "buy_hold_risk_matched_equity": risk_matched_benchmark,
            "reward": [np.nan, *run.rewards],
            "action": [np.nan, *run.actions],
            "action_label": [None, *action_labels],
            "trade_executed": [False, *run.executions],
            "execution_reason": [None, *run.execution_reasons],
        },
        index=run.dates,
    )
    for name, values in baseline_equity.items():
        equity_frame[f"baseline_{name}_equity"] = np.asarray(values, dtype=float)
    if run.market_regimes is not None:
        if len(run.market_regimes) != len(run.equity):
            raise ValueError("Les régimes doivent couvrir exactement la période évaluée")
        equity_frame["market_regime"] = list(run.market_regimes)
    equity_frame.index.name = "timestamp"
    raw_path = output_dir / "ohlcv_raw.csv"
    equity_path = output_dir / "equity_curve.csv"
    metrics_path = output_dir / "metrics.json"
    comparison_path = output_dir / "equity_comparison.png"
    baselines_path = output_dir / "baseline_diagnostics.png"
    rewards_path = output_dir / "rewards_and_actions.png"
    trades_path = output_dir / "trades.csv"
    raw_data.to_csv(raw_path)
    equity_frame.to_csv(equity_path)
    ledger = FifoTradeLedger()
    for fill in run.fills:
        if fill["side"] == "buy":
            ledger.buy(fill["timestamp"], fill["quantity"], fill["price"], fill["fee"])
        else:
            ledger.sell(fill["timestamp"], fill["quantity"], fill["price"], fill["fee"], fill["reason"])
    trades = ledger.dataframe()
    regime_pnl: dict[str, float] = {}
    if not trades.empty and run.market_regimes is not None:
        regime_by_time = pd.Series(list(run.market_regimes), index=run.dates)
        trades["entry_regime"] = pd.to_datetime(trades["entry_time"]).map(regime_by_time).fillna("unknown")
        trades["exit_regime"] = pd.to_datetime(trades["exit_time"]).map(regime_by_time).fillna("unknown")
        regime_pnl = {
            str(name): float(value)
            for name, value in trades.groupby("exit_regime")["pnl_net"].sum().sort_index().items()
        }
    trades.to_csv(trades_path, index=False)

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "source": data_source,
        "transaction_fee": transaction_fee,
        "strategy": strategy_metrics,
        "buy_and_hold": benchmark_metrics,
        "risk_matched_buy_and_hold": risk_matched_metrics,
        "causal_baselines": baseline_metrics,
        "training": training_metadata,
        "test_start": run.dates[0].isoformat(),
        "test_end": run.dates[-1].isoformat(),
        "test_candles": len(run.equity),
        "executed_trade_count": int(sum(run.executions)),
        "execution_reason_counts": pd.Series(run.execution_reasons).dropna().value_counts().to_dict(),
        "closed_trade_metrics": ledger.metrics(),
        "closed_trade_pnl_by_exit_regime": regime_pnl,
        "evaluated_regimes": regime_summary(run.market_regimes or []),
        "action_diversity": action_diversity_metrics(
            run.actions,
            action_type=run.action_type,
            n_discrete_actions=run.n_discrete_actions,
        ),
        "selected_action_counts": {
            str(action): int(run.actions.count(action)) for action in sorted(set(run.actions))
        },
        "action_mapping": (
            {str(action): discrete_action_label(action, run.n_discrete_actions) for action in range(2 * run.n_discrete_actions + 1)}
            if run.action_type == "discrete"
            else {"negative": "SELL", "near_zero": "HOLD", "positive": "BUY"}
        ),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(14, 7))
    axis.plot(run.dates, run.equity, label="Agent RL (hors-échantillon)", linewidth=1.5)
    axis.plot(run.dates, run.benchmark_equity, label="Buy & Hold (100 % exposition, frais d'entrée)", linestyle="--", linewidth=1.3)
    axis.set_title(f"{symbol} — agent RL vs Buy & Hold (hors-échantillon)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Valeur du portefeuille")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)

    if baseline_equity:
        fig, axis = plt.subplots(figsize=(14, 7))
        axis.plot(run.dates, run.benchmark_equity, label="Buy & Hold (100 % exposition)", linestyle="--", linewidth=1.3)
        for name, values in baseline_equity.items():
            axis.plot(run.dates, values, label=f"Baseline causale: {name}", linestyle="-.", linewidth=1.0)
        if not np.allclose(risk_matched_benchmark, run.benchmark_equity):
            axis.plot(run.dates, risk_matched_benchmark, label="Buy & Hold (exposition risque identique)", linestyle=":", linewidth=1.3)
        axis.set_title(f"{symbol} — diagnostics des baselines (hors sélection RL)")
        axis.set_xlabel("Date")
        axis.set_ylabel("Valeur du portefeuille")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(baselines_path, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(run.dates[1:], run.rewards, color="tab:purple", linewidth=0.9)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_title("Récompense par bougie (évaluation hors-échantillon)")
    axes[0].set_ylabel("Récompense")
    axes[0].grid(alpha=0.25)
    axes[1].step(run.dates[1:], run.actions, where="post", color="tab:orange", label="Action demandée")
    executed_dates = run.dates[1:][np.asarray(run.executions, dtype=bool)]
    executed_actions = np.asarray(run.actions)[np.asarray(run.executions, dtype=bool)]
    if len(executed_dates):
        reasons = np.asarray(run.execution_reasons, dtype=object)[np.asarray(run.executions, dtype=bool)]
        agent_orders = reasons == "agent_order"
        buy_orders = agent_orders & (executed_actions > 0) & (executed_actions <= run.n_discrete_actions)
        sell_orders = agent_orders & ~buy_orders
        if buy_orders.any():
            axes[1].scatter(executed_dates[buy_orders], executed_actions[buy_orders], color="tab:green", marker="^", zorder=3, label="Achat agent")
        if sell_orders.any():
            axes[1].scatter(executed_dates[sell_orders], executed_actions[sell_orders], color="tab:red", marker="v", zorder=3, label="Vente agent")
        if (~agent_orders).any():
            axes[1].scatter(executed_dates[~agent_orders], executed_actions[~agent_orders], color="tab:purple", marker="x", zorder=3, label="Sortie risque")
    axes[1].set_title(f"Actions de l'agent — ordres exécutés : {sum(run.executions)}")
    axes[1].set_xlabel("Date")
    if run.action_type == "discrete":
        ticks = np.arange(0, 2 * run.n_discrete_actions + 1)
        axes[1].set_yticks(ticks, [discrete_action_label(value, run.n_discrete_actions) for value in ticks])
        axes[1].set_ylabel("Action demandée")
    else:
        axes[1].set_ylabel("Action continue (-1 vente, 0 hold, +1 achat)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(rewards_path, dpi=150)
    plt.close(fig)
    paths = {
        "raw_ohlcv": raw_path,
        "equity_curve": equity_path,
        "metrics": metrics_path,
        "equity_comparison": comparison_path,
        "rewards_and_actions": rewards_path,
        "trades": trades_path,
    }
    if baseline_equity:
        paths["baseline_diagnostics"] = baselines_path
    return paths
