"""Walk-forward réel GPU pour un portefeuille BTC + ETH + or.

La sélection ne lit que la validation. Le test n'est exécuté qu'une fois pour
le candidat retenu et le rapport compare le portefeuille RL au passif 1/N.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ai_trading.rl.agents.ppo_agent import PPOAgent
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment
from ai_trading.rl.performance_evaluation import calculate_metrics
from ai_trading.rl.policy_validation import action_diversity_metrics
from ai_trading.rl.trade_ledger import FifoTradeLedger
from ai_trading.rl.trading_system import RLTradingSystem
from ai_trading.rl.walk_forward import build_walk_forward_windows
from ai_trading.scripts.run_real_market_walk_forward import _fetch_ohlcv, _seed


CANDIDATES = (
    {"agent_type": "ppo", "hidden_size": 128, "learning_rate": 3e-4, "base_slippage": 0.001, "max_active_positions": 3, "rebalance_frequency": 5, "min_trade_fraction": 0.03, "min_action_magnitude": 0.12, "max_asset_exposure": 0.45},
    # Politique tendance : moins de turnover et pénalité explicite seulement
    # lorsqu'elle sous-expose le portefeuille dans un régime haussier causal.
    {"agent_type": "ppo", "hidden_size": 256, "learning_rate": 2e-4, "base_slippage": 0.001, "max_active_positions": 3, "rebalance_frequency": 10, "min_trade_fraction": 0.03, "min_action_magnitude": 0.16, "max_asset_exposure": 0.45, "turnover_reward_penalty": 0.0005, "bull_underexposure_penalty": 0.30, "bull_exposure_target": 0.65},
    # PPO sparse et SAC Transformer : deux politiques réellement distinctes.
    # Le budget vient exclusivement de la ligne de commande afin de comparer
    # les agents à données, frais et coût de calcul identiques.
    {"agent_type": "ppo", "hidden_size": 64, "learning_rate": 5e-4, "base_slippage": 0.001, "max_active_positions": 3, "rebalance_frequency": 7, "min_trade_fraction": 0.03, "min_action_magnitude": 0.03, "max_asset_exposure": 0.35, "signal_action_blend": 0.45, "strict_short_entry": True, "short_entry_min_confidence": 0.30, "turnover_reward_penalty": 0.0005},
    {"agent_type": "ppo", "hidden_size": 128, "learning_rate": 1e-4, "base_slippage": 0.0008, "max_active_positions": 3, "rebalance_frequency": 8, "min_trade_fraction": 0.04, "min_action_magnitude": 0.10, "max_asset_exposure": 0.45, "strict_short_entry": True, "short_entry_min_confidence": 0.30, "turnover_reward_penalty": 0.0005},
    {"agent_type": "ppo", "hidden_size": 128, "learning_rate": 2e-4, "base_slippage": 0.001, "max_active_positions": 2, "rebalance_frequency": 5, "min_trade_fraction": 0.03, "min_action_magnitude": 0.08, "max_asset_exposure": 0.40, "signal_action_blend": 0.25, "strict_short_entry": True, "short_entry_min_confidence": 0.35, "turnover_reward_penalty": 0.001},
    {"agent_type": "ppo", "hidden_size": 256, "learning_rate": 1e-4, "base_slippage": 0.001, "max_active_positions": 3, "rebalance_frequency": 12, "min_trade_fraction": 0.05, "min_action_magnitude": 0.18, "max_asset_exposure": 0.45, "signal_action_blend": 0.20, "strict_short_entry": True, "short_entry_min_confidence": 0.35, "turnover_reward_penalty": 0.00025, "bull_underexposure_penalty": 0.15, "bull_exposure_target": 0.60},
    {"agent_type": "ppo", "hidden_size": 64, "learning_rate": 3e-4, "base_slippage": 0.0012, "max_active_positions": 3, "rebalance_frequency": 4, "min_trade_fraction": 0.02, "min_action_magnitude": 0.06, "max_asset_exposure": 0.35, "signal_action_blend": 0.30, "strict_short_entry": True, "short_entry_min_confidence": 0.40, "turnover_reward_penalty": 0.0015},
    {"agent_type": "ppo", "hidden_size": 128, "learning_rate": 5e-4, "base_slippage": 0.001, "max_active_positions": 2, "rebalance_frequency": 9, "min_trade_fraction": 0.04, "min_action_magnitude": 0.14, "max_asset_exposure": 0.45, "signal_action_blend": 0.55, "strict_short_entry": True, "short_entry_min_confidence": 0.35, "turnover_reward_penalty": 0.0005},
    {"agent_type": "ppo", "hidden_size": 256, "learning_rate": 2e-4, "base_slippage": 0.0008, "max_active_positions": 3, "rebalance_frequency": 6, "min_trade_fraction": 0.03, "min_action_magnitude": 0.10, "max_asset_exposure": 0.45, "signal_action_blend": 0.15, "strict_short_entry": True, "short_entry_min_confidence": 0.30, "turnover_reward_penalty": 0.00075},
    {"agent_type": "sac", "d_model": 128, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128, "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005, "entropy_regularization": 0.12, "base_slippage": 0.0015, "max_active_positions": 2, "rebalance_frequency": 3, "min_trade_fraction": 0.01, "min_action_magnitude": 0.01, "max_asset_exposure": 0.35},
)


def _common_data(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common = sorted(set.intersection(*(set(frame.index) for frame in raw.values())))
    if not common:
        raise ValueError("Aucune bougie commune entre les actifs demandés")
    return {symbol: frame.loc[common].copy() for symbol, frame in raw.items()}


def _slice(data: dict[str, pd.DataFrame], index: pd.Index) -> dict[str, pd.DataFrame]:
    return {symbol: frame.loc[index].copy() for symbol, frame in data.items()}


def _slice_with_context(data: dict[str, pd.DataFrame], index: pd.Index, context_candles: int) -> tuple[dict[str, pd.DataFrame], int]:
    """Ajoute un historique causal pour les indicateurs, sans trader ce warmup."""
    frame = next(iter(data.values()))
    start = frame.index.get_loc(index[0])
    context_start = max(0, start - context_candles)
    combined_index = frame.index[context_start : start + len(index)]
    return _slice(data, combined_index), start - context_start


def _environment(data: dict[str, pd.DataFrame], args: argparse.Namespace, params: dict[str, Any], start_step: int | None = None) -> MultiAssetTradingEnvironment:
    return MultiAssetTradingEnvironment(
        data, initial_balance=args.initial_balance, transaction_fee=args.transaction_fee,
        window_size=args.window_size, reward_function="risk_adjusted_excess", risk_management=True,
        base_slippage=params["base_slippage"], max_active_positions=params["max_active_positions"],
        rebalance_frequency=params["rebalance_frequency"],
        min_trade_fraction=params["min_trade_fraction"],
        min_action_magnitude=params["min_action_magnitude"],
        max_asset_exposure=params["max_asset_exposure"],
        allow_short=args.allow_short,
        max_short_exposure=args.max_short_exposure,
        max_total_short_exposure=args.max_total_short_exposure,
        short_initial_margin=args.short_initial_margin,
        short_maintenance_margin=args.short_maintenance_margin,
        short_borrow_fee_rate=args.short_borrow_fee_rate,
        max_short_loss_pct=args.max_short_loss_pct,
        max_short_trailing_drawdown_pct=args.max_short_trailing_drawdown_pct,
        strict_short_entry=params.get("strict_short_entry", True),
        short_entry_min_confidence=params.get("short_entry_min_confidence", 0.25),
        signal_action_blend=params.get("signal_action_blend", 0.0),
        regime_action_guard=params.get("regime_action_guard", not args.disable_regime_action_guard),
        allocation_method="smart", normalize_observation=True, start_step=start_step,
        turnover_reward_penalty=params.get("turnover_reward_penalty", 0.002),
        bull_underexposure_penalty=params.get("bull_underexposure_penalty", 0.0),
        bull_exposure_target=params.get("bull_exposure_target", 0.75),
    )


def _agent(env: MultiAssetTradingEnvironment, params: dict[str, Any], args: argparse.Namespace) -> PPOAgent | SACAgent:
    system = RLTradingSystem()
    system._env = env
    if params["agent_type"] == "ppo":
        agent = system.create_agent(
            "ppo", state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0],
            hidden_size=params["hidden_size"], learning_rate=params["learning_rate"],
            update_epochs=args.ppo_update_epochs, mini_batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        agent = system.create_agent(
            "sac", state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0],
            d_model=params["d_model"], n_heads=params["n_heads"], num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"], gamma=params["gamma"], tau=params["tau"],
            entropy_regularization=params["entropy_regularization"],
            learning_rate=params["learning_rate"], batch_size=args.batch_size,
            buffer_size=args.buffer_size, sequence_length=args.sequence_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    budget_scale = args.candidate_budget_scale
    system.train(agent=agent, episodes=max(1, round(params.get("episodes", args.episodes) * budget_scale)), batch_size=args.batch_size,
                 max_total_steps=max(1, round(params.get("max_training_steps", args.max_training_steps) * budget_scale)),
                 train_every=args.train_every,
                 max_optimization_steps=max(1, round(params.get("max_optimization_steps", args.max_optimization_steps) * budget_scale)))
    return agent


def _evaluate(agent: PPOAgent | SACAgent, env: MultiAssetTradingEnvironment) -> dict[str, Any]:
    state, _ = env.reset()
    values, actions, raw_actions, decision_available = [env.initial_balance], [], [], []
    done = False
    while not done:
        action_mask = env.get_action_mask()
        if isinstance(agent, PPOAgent):
            action, _ = agent.get_action(state, deterministic=True, action_mask=action_mask)
        else:
            action = agent.select_action(state, deterministic=True, action_mask=action_mask)
        raw_actions.append(np.asarray(action, dtype=float))
        action = env.project_action(action)
        state, _, terminated, truncated, _ = env.step(action)
        actions.append(np.asarray(action, dtype=float))
        decision_available.append(bool(np.any(np.abs(action_mask["low"]) > 1e-6) or np.any(action_mask["high"] > 1e-6)))
        values.append(float(env.get_portfolio_value()))
        done = terminated or truncated
    # Une évaluation est marquée au marché sur la courbe, mais le ledger doit
    # aussi matérialiser ce PnL à la dernière bougie pour mesurer PF/win-rate.
    env.close_all_positions()
    values[-1] = float(env.get_portfolio_value())
    return {
        "equity": values,
        "actions": actions,
        "raw_actions": raw_actions,
        "decision_available": decision_available,
        "events": list(env.trade_events),
        "allocations": list(env.allocation_history),
    }


def _passive_equity(data: dict[str, pd.DataFrame], initial: float, fee: float, window_size: int) -> list[float]:
    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in data.items()}).iloc[window_size:]
    normalized = closes.div(closes.iloc[0])
    # L'agent part de ``initial`` puis sa courbe inclut son entree et sa
    # liquidation finale. Le benchmark doit garder ce meme point de depart et
    # supporter les memes deux frais, sinon son rendement net est surestime.
    net_equity = initial * (1 - fee) ** 2 * normalized.mean(axis=1)
    return [initial, *net_equity.iloc[1:].tolist()]


def _record_ledger_event(ledger: FifoTradeLedger, event: dict[str, Any]) -> None:
    """Transcrit uniquement les fills de position; les frais d'emprunt restent séparés."""
    side = event["side"]
    if side == "buy":
        ledger.buy(event["timestamp"], event["quantity"], event["price"], event["fee"])
    elif side == "sell":
        ledger.sell(event["timestamp"], event["quantity"], event["price"], event["fee"], event["reason"])
    elif side == "short_open":
        ledger.short_sell(event["timestamp"], event["quantity"], event["price"], event["fee"])
    elif side == "short_cover":
        ledger.cover(event["timestamp"], event["quantity"], event["price"], event["fee"], event["reason"])


def _build_ledgers(events: list[dict[str, Any]]) -> dict[str, FifoTradeLedger]:
    ledgers: dict[str, FifoTradeLedger] = {}
    for event in events:
        if event["side"] == "borrow_fee":
            ledger = ledgers.get(event["symbol"])
            if ledger is not None:
                ledger.charge_short_borrow_fee(float(event["fee"]))
            continue
        ledger = ledgers.setdefault(event["symbol"], FifoTradeLedger())
        _record_ledger_event(ledger, event)
    return ledgers


def _ledger_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    ledgers = _build_ledgers(events)
    closed = [trade for ledger in ledgers.values() for trade in ledger.closed]
    wins = [trade.pnl_net for trade in closed if trade.pnl_net > 0]
    losses = [trade.pnl_net for trade in closed if trade.pnl_net < 0]
    return {
        "closed_trade_count": len(closed),
        "profit_factor": float(sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0),
        "expectancy": float(np.mean([trade.pnl_net for trade in closed])) if closed else 0.0,
        "win_rate": float(len(wins) / len(closed)) if closed else 0.0,
    }


def _write_trade_audit(
    directory: Path,
    events: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    *,
    initial_balance: float | None = None,
    ending_equity: float | None = None,
) -> dict[str, Any]:
    """Écrit les fills, trades FIFO clôturés et positions ouvertes d'un run."""
    fill_columns = ["timestamp", "symbol", "side", "position_side", "quantity", "price", "fee", "reason"]
    fills = pd.DataFrame(events, columns=fill_columns)
    fills.to_csv(directory / "order_fills.csv", index=False)

    ledgers = _build_ledgers(events)

    closed_frames, open_frames = [], []
    for symbol, ledger in ledgers.items():
        closed = ledger.dataframe()
        if not closed.empty:
            closed.insert(0, "symbol", symbol)
            closed["duration_days"] = closed["duration_seconds"] / 86_400.0
            closed_frames.append(closed)
        open_lots = ledger.open_lots_dataframe()
        if not open_lots.empty:
            mark_price = float(data[symbol]["close"].iloc[-1])
            open_lots.insert(0, "symbol", symbol)
            open_lots = open_lots.rename(columns={"time": "entry_time", "price": "entry_price", "fee": "entry_fee"})
            open_lots["mark_price"] = mark_price
            long_mask = open_lots["position_side"] == "long"
            open_lots["entry_cost"] = open_lots["quantity"] * open_lots["entry_price"] + open_lots["entry_fee"]
            open_lots["mark_value"] = open_lots["quantity"] * mark_price
            open_lots["unrealized_pnl_net"] = np.where(
                long_mask,
                open_lots["mark_value"] - open_lots["entry_cost"],
                (open_lots["quantity"] * open_lots["entry_price"] - open_lots["entry_fee"]) - open_lots["mark_value"],
            ) - open_lots["borrow_fee"]
            open_lots["unrealized_return_pct"] = open_lots["unrealized_pnl_net"] / open_lots["entry_cost"].replace(0, np.nan)
            open_frames.append(open_lots)

    closed_columns = ["symbol", "position_side", "entry_time", "exit_time", "quantity", "entry_price", "exit_price", "entry_fee", "exit_fee", "pnl_net", "return_pct", "duration_seconds", "duration_days", "exit_reason", "outcome"]
    open_columns = ["symbol", "position_side", "entry_time", "quantity", "entry_price", "entry_fee", "mark_price", "entry_cost", "mark_value", "unrealized_pnl_net", "unrealized_return_pct"]
    closed_trades = pd.concat(closed_frames, ignore_index=True) if closed_frames else pd.DataFrame(columns=closed_columns)
    open_positions = pd.concat(open_frames, ignore_index=True) if open_frames else pd.DataFrame(columns=open_columns)
    closed_trades.to_csv(directory / "closed_trades.csv", index=False)
    open_positions.to_csv(directory / "open_positions.csv", index=False)

    summary = {
        "orders_total": int(len(fills)),
        "buy_orders": int((fills["side"] == "buy").sum()) if not fills.empty else 0,
        "sell_orders": int((fills["side"] == "sell").sum()) if not fills.empty else 0,
        "short_open_orders": int((fills["side"] == "short_open").sum()) if not fills.empty else 0,
        "short_cover_orders": int((fills["side"] == "short_cover").sum()) if not fills.empty else 0,
        "borrow_fees": float(fills.loc[fills["side"] == "borrow_fee", "fee"].sum()) if not fills.empty else 0.0,
        "closed_trades": int(len(closed_trades)),
        "open_lots": int(len(open_positions)),
        "realized_pnl_net": float(closed_trades["pnl_net"].sum()) if not closed_trades.empty else 0.0,
        "unrealized_pnl_net": float(open_positions["unrealized_pnl_net"].sum()) if not open_positions.empty else 0.0,
    }
    if initial_balance is not None and ending_equity is not None:
        ledger_pnl = summary["realized_pnl_net"] + summary["unrealized_pnl_net"]
        equity_pnl = float(ending_equity) - float(initial_balance)
        reconciliation_error = equity_pnl - ledger_pnl
        summary.update({
            "initial_balance": float(initial_balance),
            "ending_equity": float(ending_equity),
            "equity_pnl": equity_pnl,
            "ledger_pnl": ledger_pnl,
            "reconciliation_error": reconciliation_error,
            "reconciled": bool(np.isclose(equity_pnl, ledger_pnl, atol=1e-6, rtol=1e-9)),
        })
    (directory / "trade_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _metrics(result: dict[str, Any], passive: list[float], candles: int) -> dict[str, Any]:
    # Une moyenne BTC/ETH/or peut annuler un achat et une vente simultanés et
    # faire croire à un hold. La diversité est donc mesurée par actif, sur les
    # seules étapes où le cooldown autorisait réellement une décision.
    decision_available = result.get("decision_available", [True] * len(result["actions"]))
    decision_actions = [
        action for action, available in zip(result["actions"], decision_available) if available
    ]
    actions = [float(value) for action in decision_actions for value in np.asarray(action).reshape(-1)]
    fills = [event for event in result["events"] if event["side"] != "borrow_fee"]
    trade_decision_count = len({pd.Timestamp(event["timestamp"]) for event in fills})
    return {
        "strategy": calculate_metrics(result["equity"]),
        "buy_and_hold": calculate_metrics(passive),
        "closed_trade_metrics": _ledger_metrics(result["events"]),
        "action_diversity": action_diversity_metrics(actions, action_type="continuous"),
        "executed_trade_count": len(fills),
        "trade_decision_count": trade_decision_count,
        "test_candles": candles,
        # Ce sont les seules étapes où le cooldown autorise une décision.
        # L'évaluation complète est donnée séparément par evaluation_bar_count.
        "decision_steps": len(decision_actions),
        "evaluation_bar_count": len(result["actions"]),
        "action_component_count": len(actions),
    }


def _eligible(metrics: dict[str, Any], min_trades: int, max_order_rate: float) -> tuple[bool, str | None]:
    stability = metrics.get("seed_stability")
    if stability and stability["seed_count"] > 1 and stability["outperforming_seed_count"] != stability["seed_count"]:
        return False, "candidat instable selon les seeds d'entraînement"
    if metrics["strategy"]["total_return"] <= metrics["buy_and_hold"]["total_return"]:
        return False, "rendement net inférieur au portefeuille passif"
    if not metrics["action_diversity"]["passed"] or not metrics["action_diversity"]["directional_passed"]:
        return False, "diversité d'actions insuffisante"
    directions = metrics["action_diversity"]["direction_counts"]
    if directions.get("buy", 0) == 0 or directions.get("sell", 0) == 0:
        return False, "politique sans alternance achat/vente"
    trades = metrics["closed_trade_metrics"]
    if trades["closed_trade_count"] < min_trades or trades["profit_factor"] <= 1 or trades["expectancy"] <= 0:
        return False, "ledger non rentable ou insuffisant"
    if metrics["trade_decision_count"] / max(1, metrics["evaluation_bar_count"]) > max_order_rate:
        return False, "turnover excessif"
    return True, None


def _aggregate_validation(metrics_by_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Médianes robustes ; les garde-fous booléens doivent passer sur chaque seed."""
    if not metrics_by_seed:
        raise ValueError("Validation multi-seed vide")
    first = metrics_by_seed[0]
    aggregate: dict[str, Any] = {"validation_seed_count": len(metrics_by_seed)}
    for section in ("strategy", "buy_and_hold", "closed_trade_metrics"):
        aggregate[section] = {
            key: float(np.median([entry[section][key] for entry in metrics_by_seed]))
            for key in first[section]
        }
    diversity = first["action_diversity"].copy()
    for key in ("passed", "directional_passed"):
        diversity[key] = bool(all(entry["action_diversity"][key] for entry in metrics_by_seed))
    aggregate["action_diversity"] = diversity
    all_directions = {direction for entry in metrics_by_seed for direction in entry["action_diversity"]["direction_counts"]}
    diversity["direction_counts"] = {
        direction: min(entry["action_diversity"]["direction_counts"].get(direction, 0) for entry in metrics_by_seed)
        for direction in sorted(all_directions)
    }
    aggregate["executed_trade_count"] = float(np.median([entry["executed_trade_count"] for entry in metrics_by_seed]))
    aggregate["trade_decision_count"] = float(np.median([entry["trade_decision_count"] for entry in metrics_by_seed]))
    aggregate["test_candles"] = first["test_candles"]
    aggregate["decision_steps"] = first["decision_steps"]
    aggregate["evaluation_bar_count"] = first["evaluation_bar_count"]
    aggregate["action_component_count"] = first["action_component_count"]
    excess_returns = np.asarray([
        entry["strategy"]["total_return"] - entry["buy_and_hold"]["total_return"]
        for entry in metrics_by_seed
    ], dtype=float)
    aggregate["seed_stability"] = {
        "min_excess_return": float(excess_returns.min()),
        "median_excess_return": float(np.median(excess_returns)),
        "max_excess_return": float(excess_returns.max()),
        "outperforming_seed_count": int(np.count_nonzero(excess_returns > 0)),
        "seed_count": int(len(excess_returns)),
    }
    return aggregate


def _save(directory: Path, data: dict[str, pd.DataFrame], result: dict[str, Any], passive: list[float], metrics: dict[str, Any], stage: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = next(iter(data.values())).index
    # L'equity est valorisée après chaque step, tandis que l'action/fill est
    # décidé sur la bougie précédente. Les deux séries ont donc des dates
    # distinctes et ne doivent pas être décalées dans les exports d'audit.
    dates = index[-len(result["equity"]) + 1 :]
    decision_dates = index[-len(result["actions"]) - 1 : -1] if result["actions"] else index[:0]
    agent_equity, passive_equity = result["equity"][1:], passive[1:]
    pd.DataFrame({"timestamp": dates, "agent_rl": agent_equity, "passif_diversifie": passive_equity}).to_csv(directory / "equity_curve.csv", index=False)
    pd.DataFrame(result["events"]).to_csv(directory / "trades.csv", index=False)
    if result["actions"]:
        action_frame = pd.DataFrame(result["actions"], columns=[f"action_{symbol}" for symbol in data])
        action_frame.insert(0, "timestamp", decision_dates)
        action_frame.insert(1, "decision_available", result.get("decision_available", [True] * len(action_frame)))
        if result.get("raw_actions"):
            raw_frame = pd.DataFrame(result["raw_actions"], columns=[f"raw_action_{symbol}" for symbol in data])
            action_frame = pd.concat([action_frame, raw_frame], axis=1)
        action_frame.to_csv(directory / "actions.csv", index=False)
    metrics["trade_audit"] = _write_trade_audit(
        directory,
        result["events"],
        data,
        initial_balance=float(result["equity"][0]),
        ending_equity=float(result["equity"][-1]),
    )
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, agent_equity, label="Agent RL multi-actifs", color="#1f77b4")
    ax.plot(dates, passive_equity, label="Buy & Hold diversifié égalitaire", color="#ff7f0e", linestyle="--")
    ax.set_title(f"{' + '.join(data)} — portefeuille réel {stage}")
    ax.set_ylabel("Valeur du portefeuille")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(directory / "equity_comparison.png", dpi=160); plt.close(fig)
    if result["allocations"]:
        allocation = pd.DataFrame(result["allocations"], index=dates[:len(result["allocations"])])
        # Les shorts portent un poids négatif. Un graphique area empilé refuse
        # de mélanger signes positifs et négatifs dans une même colonne ; les
        # courbes signées exposent au contraire clairement long, cash et short.
        fig, ax = plt.subplots(figsize=(14, 5))
        for symbol in allocation.columns:
            ax.plot(allocation.index, allocation[symbol], label=symbol)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Expositions RL nettes (positif = long, négatif = short)")
        ax.set_ylabel("Part de l'equity")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "allocations.png", dpi=160)
        plt.close(fig)


def _acquire_run_lock(output_dir: Path) -> Path:
    """Crée un verrou PID atomique et récupère un verrou orphelin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".run.lock"
    for _ in range(2):
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
                lock_file.write(str(os.getpid()))
            return lock_path
        except FileExistsError:
            try:
                owner_pid = int(lock_path.read_text(encoding="utf-8").strip())
                os.kill(owner_pid, 0)
            except (ValueError, OSError):
                # Le processus n'existe plus : un timeout ou un crash ne doit
                # jamais bloquer la campagne suivante.
                lock_path.unlink(missing_ok=True)
                continue
            raise RuntimeError(
                f"Une campagne est déjà en cours dans {output_dir}. "
                "Attends sa fin ou utilise un autre --output-dir."
            )
    else:
        raise RuntimeError(f"Impossible d'acquérir le verrou de {output_dir}.")


def run(args: argparse.Namespace) -> Path:
    """Exécute une campagne en excluant tout second run sur le même dossier."""
    lock_path = _acquire_run_lock(Path(args.output_dir))
    try:
        return _run_unlocked(args)
    finally:
        lock_path.unlink(missing_ok=True)


def _test_fingerprint(args: argparse.Namespace, window) -> str:
    """Identifiant stable d'un holdout : un même intervalle ne doit pas être rejoué."""
    payload = {
        "assets": list(args.assets), "timeframe": args.timeframe,
        "test_start": str(window.test.index[0]), "test_end": str(window.test.index[-1]),
        "test_candles": len(window.test),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_oos_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Registre hors-échantillon illisible : {path}") from exc


def _assert_test_available(args: argparse.Namespace, window) -> tuple[Path, str]:
    registry_path = Path(args.oos_registry)
    fingerprint = _test_fingerprint(args, window)
    if fingerprint in _load_oos_registry(registry_path):
        raise RuntimeError(
            "Cette fenêtre hors-échantillon a déjà été consommée. "
            "Lance uniquement la validation ou choisis une nouvelle fenêtre."
        )
    return registry_path, fingerprint


def _record_test_consumed(registry_path: Path, fingerprint: str, args: argparse.Namespace, window) -> None:
    registry = _load_oos_registry(registry_path)
    registry[fingerprint] = {
        "assets": list(args.assets), "timeframe": args.timeframe,
        "test_start": str(window.test.index[0]), "test_end": str(window.test.index[-1]),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _run_unlocked(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    raw = _common_data({asset: _fetch_ohlcv(asset, args) for asset in args.assets})
    anchor = next(iter(raw.values()))
    all_windows = build_walk_forward_windows(
        anchor, args.train_candles, args.validation_candles, args.test_candles, args.step_candles
    )
    windows = all_windows[args.window_start : args.window_start + args.max_windows]
    if not windows:
        raise ValueError("window-start est hors des fenêtres walk-forward disponibles")
    root = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    feature_context = max(args.window_size, 80)
    rows = []
    for window in windows:
        candidates = []
        selected_indices = args.candidate_indices if args.candidate_indices is not None else range(len(CANDIDATES))
        if any(index < 0 or index >= len(CANDIDATES) for index in selected_indices):
            raise ValueError(f"candidate-indices doit être compris entre 0 et {len(CANDIDATES) - 1}")
        for candidate_index in selected_indices:
            params = CANDIDATES[candidate_index]
            seed_metrics = []
            for seed in args.validation_seeds:
                _seed(seed + window.index * 10_000 + candidate_index * 100)
                agent = _agent(_environment(_slice(raw, window.train.index), args, params), params, args)
                validation_data, validation_start = _slice_with_context(
                    raw, window.validation.index, feature_context
                )
                validation_result = _evaluate(
                    agent, _environment(validation_data, args, params, start_step=validation_start)
                )
                validation_passive = _passive_equity(
                    validation_data, args.initial_balance, args.transaction_fee, validation_start
                )
                validation_metrics = _metrics(validation_result, validation_passive, len(window.validation))
                candidate_dir = root / f"window_{window.index:02d}" / "validation" / f"candidate_{candidate_index:02d}" / f"seed_{seed}"
                _save(candidate_dir, validation_data, validation_result, validation_passive, validation_metrics, "validation")
                seed_metrics.append(validation_metrics)
            validation_metrics = _aggregate_validation(seed_metrics)
            eligible, reason = _eligible(validation_metrics, args.min_closed_trades, args.max_agent_order_rate)
            candidates.append({
                "parameters": params, "metrics": validation_metrics, "eligible": eligible,
                "reason": reason,
            })
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        row: dict[str, Any] = {"window": window.index, "validation_candidates": candidates}
        if eligible:
            selected = max(eligible, key=lambda item: item["metrics"]["strategy"]["total_return"])
            row["selected_parameters"] = selected["parameters"]
            if args.evaluation_mode == "validation":
                row["test_deferred"] = True
                rows.append(row)
                continue
            registry_path, fingerprint = _assert_test_available(args, window)
            train_validation = _slice(raw, window.train.index.append(window.validation.index))
            agent = _agent(_environment(train_validation, args, selected["parameters"]), selected["parameters"], args)
            test_data, test_start = _slice_with_context(raw, window.test.index, feature_context)
            result = _evaluate(agent, _environment(test_data, args, selected["parameters"], start_step=test_start))
            passive = _passive_equity(test_data, args.initial_balance, args.transaction_fee, test_start)
            metrics = _metrics(result, passive, len(window.test))
            directory = root / f"window_{window.index:02d}" / "test"
            _save(directory, test_data, result, passive, metrics, "hors-échantillon")
            test_eligible, test_reason = _eligible(
                metrics, args.min_closed_trades, args.max_agent_order_rate
            )
            row.update({
                "test": metrics,
                "test_eligible": test_eligible,
                "test_rejection_reason": test_reason,
                "report_dir": str(directory),
            })
            _record_test_consumed(registry_path, fingerprint, args, window)
        else:
            row["validation_rejected"] = True
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "walk_forward_summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return root / "walk_forward_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=["BTC/USDT", "ETH/USDT", "XAU/USD"])
    parser.add_argument("--exchange", default="binance"); parser.add_argument("--timeframe", default="1d", choices=("1d", "4h", "1h"))
    parser.add_argument("--days", type=int, default=1800); parser.add_argument("--train-candles", type=int, default=500)
    parser.add_argument("--validation-candles", type=int, default=150); parser.add_argument("--test-candles", type=int, default=150)
    parser.add_argument("--step-candles", type=int, default=150); parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--window-start", type=int, default=0, help="Indice de la première fenêtre walk-forward à évaluer.")
    parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--max-training-steps", type=int, default=10000)
    parser.add_argument("--max-optimization-steps", type=int, default=1000); parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=8); parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--initial-balance", type=float, default=10000.0); parser.add_argument("--transaction-fee", type=float, default=0.001)
    parser.add_argument("--min-closed-trades", type=int, default=20); parser.add_argument("--max-agent-order-rate", type=float, default=0.35)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[42, 314, 2024])
    parser.add_argument("--ppo-update-epochs", type=int, default=5)
    parser.add_argument("--allow-short", action="store_true", help="Active les shorts avec marge et couvertures obligatoires.")
    parser.add_argument("--max-short-exposure", type=float, default=0.25)
    parser.add_argument("--max-total-short-exposure", type=float, default=0.45)
    parser.add_argument("--short-initial-margin", type=float, default=0.50)
    parser.add_argument("--short-maintenance-margin", type=float, default=0.35)
    parser.add_argument("--short-borrow-fee-rate", type=float, default=0.0001)
    parser.add_argument("--max-short-loss-pct", type=float, default=0.08)
    parser.add_argument("--max-short-trailing-drawdown-pct", type=float, default=0.10)
    parser.add_argument("--disable-regime-action-guard", action="store_true", help="Désactive le blocage causal des positions contre-tendance.")
    parser.add_argument("--candidate-budget-scale", type=float, default=1.0, help="Réduit tous les budgets candidat pour un smoke test.")
    parser.add_argument("--candidate-indices", nargs="+", type=int, default=None, help="Sous-ensemble de candidats à valider, par indice zéro-based.")
    parser.add_argument("--evaluation-mode", choices=("validation", "final-test"), default="validation", help="Validation only par défaut; le holdout ne s'exécute qu'en final-test.")
    parser.add_argument("--oos-registry", default="ai_trading/info_retour/p3_oos_registry.json", help="Registre persistant des fenêtres hors-échantillon consommées.")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output-dir", default="ai_trading/info_retour/p3_multi_asset_real")
    return parser.parse_args()


if __name__ == "__main__":
    print(f"summary: {run(parse_args())}")
