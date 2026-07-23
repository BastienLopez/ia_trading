"""Walk-forward réel GPU pour un portefeuille BTC + ETH + or.

La sélection ne lit que la validation. Le test n'est exécuté qu'une fois pour
le candidat retenu et le rapport compare le portefeuille RL au passif 1/N.
"""

from __future__ import annotations

import argparse
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
    {"agent_type": "ppo", "hidden_size": 256, "learning_rate": 1e-4, "base_slippage": 0.0015, "max_active_positions": 2, "rebalance_frequency": 3, "min_trade_fraction": 0.02, "min_action_magnitude": 0.08, "max_asset_exposure": 0.45},
    # PPO sparse et SAC Transformer : deux politiques réellement distinctes
    # pour le smoke, sans conserver un SAC qui se repliait systématiquement
    # sur cash avec ce budget court.
    {"agent_type": "ppo", "hidden_size": 64, "learning_rate": 5e-4, "base_slippage": 0.001, "max_active_positions": 3, "rebalance_frequency": 7, "min_trade_fraction": 0.03, "min_action_magnitude": 0.03, "max_asset_exposure": 0.35},
    {"agent_type": "sac", "d_model": 128, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128, "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005, "entropy_regularization": 0.12, "base_slippage": 0.0015, "max_active_positions": 2, "rebalance_frequency": 3, "min_trade_fraction": 0.01, "min_action_magnitude": 0.01, "max_asset_exposure": 0.35},
)


def _common_data(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common = sorted(set.intersection(*(set(frame.index) for frame in raw.values())))
    if not common:
        raise ValueError("Aucune bougie commune entre BTC, ETH et or")
    return {symbol: frame.loc[common].copy() for symbol, frame in raw.items()}


def _slice(data: dict[str, pd.DataFrame], index: pd.Index) -> dict[str, pd.DataFrame]:
    return {symbol: frame.loc[index].copy() for symbol, frame in data.items()}


def _environment(data: dict[str, pd.DataFrame], args: argparse.Namespace, params: dict[str, Any]) -> MultiAssetTradingEnvironment:
    return MultiAssetTradingEnvironment(
        data, initial_balance=args.initial_balance, transaction_fee=args.transaction_fee,
        window_size=args.window_size, reward_function="risk_adjusted_excess", risk_management=True,
        base_slippage=params["base_slippage"], max_active_positions=params["max_active_positions"],
        rebalance_frequency=params["rebalance_frequency"],
        min_trade_fraction=params["min_trade_fraction"],
        min_action_magnitude=params["min_action_magnitude"],
        max_asset_exposure=params["max_asset_exposure"],
        allocation_method="smart", normalize_observation=True,
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
    system.train(agent=agent, episodes=args.episodes, batch_size=args.batch_size,
                 max_total_steps=args.max_training_steps, train_every=args.train_every,
                 max_optimization_steps=args.max_optimization_steps)
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
    return (initial * (1 - fee) * normalized.mean(axis=1)).tolist()


def _ledger_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    ledgers: dict[str, FifoTradeLedger] = {}
    for event in events:
        ledger = ledgers.setdefault(event["symbol"], FifoTradeLedger())
        if event["side"] == "buy":
            ledger.buy(event["timestamp"], event["quantity"], event["price"], event["fee"])
        else:
            ledger.sell(event["timestamp"], event["quantity"], event["price"], event["fee"], event["reason"])
    closed = [trade for ledger in ledgers.values() for trade in ledger.closed]
    wins = [trade.pnl_net for trade in closed if trade.pnl_net > 0]
    losses = [trade.pnl_net for trade in closed if trade.pnl_net < 0]
    return {
        "closed_trade_count": len(closed),
        "profit_factor": float(sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0),
        "expectancy": float(np.mean([trade.pnl_net for trade in closed])) if closed else 0.0,
        "win_rate": float(len(wins) / len(closed)) if closed else 0.0,
    }


def _write_trade_audit(directory: Path, events: list[dict[str, Any]], data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Écrit les fills, trades FIFO clôturés et positions ouvertes d'un run."""
    fill_columns = ["timestamp", "symbol", "side", "quantity", "price", "fee", "reason"]
    fills = pd.DataFrame(events, columns=fill_columns)
    fills.to_csv(directory / "order_fills.csv", index=False)

    ledgers: dict[str, FifoTradeLedger] = {}
    for event in events:
        ledger = ledgers.setdefault(event["symbol"], FifoTradeLedger())
        if event["side"] == "buy":
            ledger.buy(event["timestamp"], event["quantity"], event["price"], event["fee"])
        else:
            ledger.sell(event["timestamp"], event["quantity"], event["price"], event["fee"], event["reason"])

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
            open_lots["entry_cost"] = open_lots["quantity"] * open_lots["entry_price"] + open_lots["entry_fee"]
            open_lots["mark_value"] = open_lots["quantity"] * mark_price
            open_lots["unrealized_pnl_net"] = open_lots["mark_value"] - open_lots["entry_cost"]
            open_lots["unrealized_return_pct"] = open_lots["mark_value"] / open_lots["entry_cost"] - 1.0
            open_frames.append(open_lots)

    closed_columns = ["symbol", "entry_time", "exit_time", "quantity", "entry_price", "exit_price", "entry_fee", "exit_fee", "pnl_net", "return_pct", "duration_seconds", "duration_days", "exit_reason", "outcome"]
    open_columns = ["symbol", "entry_time", "quantity", "entry_price", "entry_fee", "mark_price", "entry_cost", "mark_value", "unrealized_pnl_net", "unrealized_return_pct"]
    closed_trades = pd.concat(closed_frames, ignore_index=True) if closed_frames else pd.DataFrame(columns=closed_columns)
    open_positions = pd.concat(open_frames, ignore_index=True) if open_frames else pd.DataFrame(columns=open_columns)
    closed_trades.to_csv(directory / "closed_trades.csv", index=False)
    open_positions.to_csv(directory / "open_positions.csv", index=False)

    summary = {
        "orders_total": int(len(fills)),
        "buy_orders": int((fills["side"] == "buy").sum()) if not fills.empty else 0,
        "sell_orders": int((fills["side"] == "sell").sum()) if not fills.empty else 0,
        "closed_trades": int(len(closed_trades)),
        "open_lots": int(len(open_positions)),
        "realized_pnl_net": float(closed_trades["pnl_net"].sum()) if not closed_trades.empty else 0.0,
        "unrealized_pnl_net": float(open_positions["unrealized_pnl_net"].sum()) if not open_positions.empty else 0.0,
    }
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
    return {
        "strategy": calculate_metrics(result["equity"]),
        "buy_and_hold": calculate_metrics(passive),
        "closed_trade_metrics": _ledger_metrics(result["events"]),
        "action_diversity": action_diversity_metrics(actions, action_type="continuous"),
        "executed_trade_count": len(result["events"]),
        "test_candles": candles,
        "evaluation_steps": len(decision_actions),
        "action_component_count": len(actions),
    }


def _eligible(metrics: dict[str, Any], min_trades: int, max_order_rate: float) -> tuple[bool, str | None]:
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
    if metrics["executed_trade_count"] / max(1, metrics["evaluation_steps"]) > max_order_rate:
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
    aggregate["test_candles"] = first["test_candles"]
    aggregate["evaluation_steps"] = first["evaluation_steps"]
    aggregate["action_component_count"] = first["action_component_count"]
    return aggregate


def _save(directory: Path, data: dict[str, pd.DataFrame], result: dict[str, Any], passive: list[float], metrics: dict[str, Any], stage: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dates = next(iter(data.values())).index[-len(result["equity"]) + 1 :]
    agent_equity, passive_equity = result["equity"][1:], passive[1:]
    pd.DataFrame({"timestamp": dates, "agent_rl": agent_equity, "passif_diversifie": passive_equity}).to_csv(directory / "equity_curve.csv", index=False)
    pd.DataFrame(result["events"]).to_csv(directory / "trades.csv", index=False)
    if result["actions"]:
        action_frame = pd.DataFrame(result["actions"], columns=[f"action_{symbol}" for symbol in data])
        action_frame.insert(0, "timestamp", dates[:len(action_frame)])
        action_frame.insert(1, "decision_available", result.get("decision_available", [True] * len(action_frame)))
        if result.get("raw_actions"):
            raw_frame = pd.DataFrame(result["raw_actions"], columns=[f"raw_action_{symbol}" for symbol in data])
            action_frame = pd.concat([action_frame, raw_frame], axis=1)
        action_frame.to_csv(directory / "actions.csv", index=False)
    metrics["trade_audit"] = _write_trade_audit(directory, result["events"], data)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, agent_equity, label="Agent RL multi-actifs", color="#1f77b4")
    ax.plot(dates, passive_equity, label="Buy & Hold diversifié égalitaire", color="#ff7f0e", linestyle="--")
    ax.set_title(f"BTC + ETH + or — portefeuille réel {stage}")
    ax.set_ylabel("Valeur du portefeuille")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(directory / "equity_comparison.png", dpi=160); plt.close(fig)
    if result["allocations"]:
        allocation = pd.DataFrame(result["allocations"], index=dates[:len(result["allocations"])])
        ax = allocation.plot.area(figsize=(14, 5), alpha=0.75, title="Allocations RL multi-actifs")
        ax.set_ylabel("Poids relatif investi"); ax.figure.tight_layout(); ax.figure.savefig(directory / "allocations.png", dpi=160); plt.close(ax.figure)


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


def _run_unlocked(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    raw = _common_data({asset: _fetch_ohlcv(asset, args) for asset in args.assets})
    anchor = next(iter(raw.values()))
    windows = build_walk_forward_windows(
        anchor, args.train_candles, args.validation_candles, args.test_candles, args.step_candles
    )[:args.max_windows]
    root = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for window in windows:
        candidates = []
        for candidate_index, params in enumerate(CANDIDATES):
            seed_metrics = []
            for seed in args.validation_seeds:
                _seed(seed + window.index * 10_000 + candidate_index * 100)
                agent = _agent(_environment(_slice(raw, window.train.index), args, params), params, args)
                validation_data = _slice(raw, window.validation.index)
                validation_result = _evaluate(agent, _environment(validation_data, args, params))
                validation_passive = _passive_equity(
                    validation_data, args.initial_balance, args.transaction_fee, args.window_size
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
            train_validation = _slice(raw, window.train.index.append(window.validation.index))
            agent = _agent(_environment(train_validation, args, selected["parameters"]), selected["parameters"], args)
            test_data = _slice(raw, window.test.index)
            result = _evaluate(agent, _environment(test_data, args, selected["parameters"]))
            passive = _passive_equity(test_data, args.initial_balance, args.transaction_fee, args.window_size)
            metrics = _metrics(result, passive, len(window.test))
            directory = root / f"window_{window.index:02d}" / "test"
            _save(directory, test_data, result, passive, metrics, "hors-échantillon")
            test_eligible, test_reason = _eligible(
                metrics, args.min_closed_trades, args.max_agent_order_rate
            )
            row.update({
                "selected_parameters": selected["parameters"],
                "test": metrics,
                "test_eligible": test_eligible,
                "test_rejection_reason": test_reason,
                "report_dir": str(directory),
            })
        else:
            row["validation_rejected"] = True
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "walk_forward_summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return root / "walk_forward_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs=3, default=["BTC/USDT", "ETH/USDT", "XAU/USD"])
    parser.add_argument("--exchange", default="binance"); parser.add_argument("--timeframe", default="1d", choices=("1d",))
    parser.add_argument("--days", type=int, default=1800); parser.add_argument("--train-candles", type=int, default=500)
    parser.add_argument("--validation-candles", type=int, default=150); parser.add_argument("--test-candles", type=int, default=150)
    parser.add_argument("--step-candles", type=int, default=150); parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--max-training-steps", type=int, default=10000)
    parser.add_argument("--max-optimization-steps", type=int, default=1000); parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=8); parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--initial-balance", type=float, default=10000.0); parser.add_argument("--transaction-fee", type=float, default=0.001)
    parser.add_argument("--min-closed-trades", type=int, default=20); parser.add_argument("--max-agent-order-rate", type=float, default=0.35)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[42, 314, 2024])
    parser.add_argument("--ppo-update-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output-dir", default="ai_trading/info_retour/p3_multi_asset_real")
    return parser.parse_args()


if __name__ == "__main__":
    print(f"summary: {run(parse_args())}")
