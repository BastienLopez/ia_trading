"""Validation walk-forward réelle BTC, ETH et or, sans sélection sur le test.

Exemple Docker (GPU, séquentiel) :
python3 -m ai_trading.scripts.run_real_market_walk_forward --assets BTC/USDT ETH/USDT XAU/USD --timeframe 1d
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ai_trading.data_processor import DataProcessor
from ai_trading.rl.multi_asset_trading import MultiAssetTradingSystem
from ai_trading.rl.real_market_report import calculate_metrics, save_report, validate_ohlcv
from ai_trading.rl.trade_ledger import FifoTradeLedger
from ai_trading.rl.policy_validation import action_diversity_metrics
from ai_trading.rl.causal_features import merge_causal_external_features
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.trading_system import RLTradingSystem
from ai_trading.rl.walk_forward import (
    build_walk_forward_windows,
    phase4_gate,
    select_candidate_on_validation,
    validation_objective,
    aggregate_validation_metrics,
)
from ai_trading.scripts.run_real_market_rl_report import evaluate_out_of_sample


DEFAULT_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "dqn": (
        {"agent": {"hidden_size": 64, "learning_rate": 3e-4}, "risk": {"stop_loss_atr_factor": 1.5, "take_profit_atr_factor": 3.0, "trailing_stop_atr_factor": 1.2, "max_trade_fraction": 0.20, "min_trade_interval": 3, "min_signal_confidence": 0.05}},
        {"agent": {"hidden_size": 128, "learning_rate": 1e-3}, "risk": {"stop_loss_atr_factor": 2.5, "take_profit_atr_factor": 5.0, "trailing_stop_atr_factor": 2.0, "max_trade_fraction": 0.15, "min_trade_interval": 5, "min_signal_confidence": 0.15}},
    ),
    "ppo": (
        {"agent": {"hidden_size": 64, "learning_rate": 3e-4}, "risk": {"stop_loss_atr_factor": 1.5, "take_profit_atr_factor": 3.0, "trailing_stop_atr_factor": 1.2, "max_trade_fraction": 0.20, "min_trade_interval": 3, "min_signal_confidence": 0.05}},
        {"agent": {"hidden_size": 128, "learning_rate": 1e-4}, "risk": {"stop_loss_atr_factor": 2.5, "take_profit_atr_factor": 5.0, "trailing_stop_atr_factor": 2.0, "max_trade_fraction": 0.15, "min_trade_interval": 5, "min_signal_confidence": 0.15}},
    ),
    "sac": (
        {"agent": {"d_model": 32, "learning_rate": 3e-4}, "risk": {"stop_loss_atr_factor": 1.5, "take_profit_atr_factor": 3.0, "trailing_stop_atr_factor": 1.2, "max_trade_fraction": 0.20, "min_trade_interval": 3, "min_signal_confidence": 0.05}},
        {"agent": {"d_model": 64, "learning_rate": 1e-4}, "risk": {"stop_loss_atr_factor": 2.5, "take_profit_atr_factor": 5.0, "trailing_stop_atr_factor": 2.0, "max_trade_fraction": 0.15, "min_trade_interval": 5, "min_signal_confidence": 0.15}},
    ),
}


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _fetch_ohlcv(asset: str, args: argparse.Namespace) -> pd.DataFrame:
    """Télécharge exclusivement des bougies publiques réelles et closes."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    if asset == "XAU/USD":
        if args.timeframe != "1d":
            raise ValueError("XAU/USD est volontairement limité à 1d (source Yahoo Finance GC=F)")
        raw = MultiAssetTradingSystem._collect_traditional_market_data(
            asset, start.date().isoformat(), (end.date() + timedelta(days=1)).isoformat()
        )
        raw.index = pd.to_datetime(raw.index, utc=True).tz_localize(None)
        return validate_ohlcv(raw)
    processor = DataProcessor(use_float16=False, use_cache=False)
    raw = processor.download_historical_data(
        exchange_id=args.exchange,
        symbol=asset,
        timeframe=args.timeframe,
        start_date=start,
        end_date=end,
        save=False,
    )
    # La dernière bougie peut être ouverte ; elle ne participe à aucune phase.
    closed_until = pd.Timestamp.now(tz="UTC").floor(args.timeframe).tz_localize(None)
    raw = raw.loc[raw.index < closed_until]
    return validate_ohlcv(raw)


def _environment(data: pd.DataFrame, args: argparse.Namespace, agent_type: str, params: dict[str, Any] | None = None) -> TradingEnvironment:
    if not 0 < args.max_position_fraction <= 1:
        raise ValueError("max_position_fraction doit être dans ]0, 1]")
    profile = dict((params or {}).get("risk", {}))
    execution_keys = {"max_trade_fraction", "min_trade_interval", "min_signal_confidence"}
    risk_config = {"max_position_size": args.max_position_fraction, **{key: value for key, value in profile.items() if key not in execution_keys}}
    return TradingEnvironment(
        data,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        slippage_value=args.slippage,
        max_trade_fraction=profile.get("max_trade_fraction", args.max_trade_fraction),
        min_trade_interval=profile.get("min_trade_interval", args.min_trade_interval),
        min_signal_confidence=profile.get("min_signal_confidence", 0.0),
        risk_config=risk_config,
        benchmark_exposure=args.max_position_fraction,
        invalid_action_penalty=args.invalid_action_penalty,
        reward_function=args.reward_function,
        reward_drawdown_weight=args.reward_drawdown_weight,
        reward_downside_weight=args.reward_downside_weight,
        window_size=args.window_size,
        normalize_observation=False,
        action_type="discrete" if agent_type == "dqn" else "continuous",
    )


def _make_agent(system: RLTradingSystem, environment: TradingEnvironment, agent_type: str, params: dict[str, Any], args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = {"device": device, "batch_size": args.batch_size, **params.get("agent", params)}
    if agent_type == "dqn":
        return system.create_agent(
            "dqn", state_size=environment.observation_space.shape[0], action_size=environment.action_space.n,
            buffer_size=args.buffer_size, use_ucb=False, **base,
        )
    if agent_type == "ppo":
        base.pop("batch_size", None)
        return system.create_agent(
            "ppo", state_size=environment.observation_space.shape[0], action_size=1,
            update_epochs=args.ppo_update_epochs, mini_batch_size=args.batch_size, **base,
        )
    if agent_type == "sac":
        base.pop("hidden_size", None)
        return system.create_agent(
            "sac", state_size=environment.observation_space.shape[0], action_size=1,
            buffer_size=args.buffer_size, n_heads=2, num_layers=1,
            dim_feedforward=args.sac_feedforward, sequence_length=args.sac_sequence_length, **base,
        )
    raise ValueError(f"Agent inconnu: {agent_type}")


def _ledger_metrics(fills: list[dict]) -> dict:
    ledger = FifoTradeLedger()
    for fill in fills:
        if fill["side"] == "buy":
            ledger.buy(fill["timestamp"], fill["quantity"], fill["price"], fill["fee"])
        else:
            ledger.sell(fill["timestamp"], fill["quantity"], fill["price"], fill["fee"], fill["reason"])
    return ledger.metrics()


def _metrics_from_evaluation(evaluation, timeframe: str) -> dict[str, Any]:
    periods = {"1h": 8760, "2h": 4380, "4h": 2190, "1d": 365}[timeframe]
    return {
        "strategy": calculate_metrics(evaluation.equity, periods),
        "buy_and_hold": calculate_metrics(evaluation.benchmark_equity, periods),
        "risk_matched_buy_and_hold": calculate_metrics(
            evaluation.risk_matched_benchmark_equity
            if evaluation.risk_matched_benchmark_equity is not None
            else evaluation.benchmark_equity,
            periods,
        ),
        "causal_baselines": {
            name: calculate_metrics(values, periods)
            for name, values in (evaluation.baseline_equity or {}).items()
        },
        "closed_trade_metrics": _ledger_metrics(evaluation.fills),
        "action_diversity": action_diversity_metrics(
            evaluation.actions, action_type=evaluation.action_type
        ),
    }


def _evaluation_with_context(history: pd.DataFrame, target: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Ajoute le seul historique déjà observable au début d'une partition test."""
    if len(history) < window_size:
        raise ValueError("Historique insuffisant pour amorcer la partition d'évaluation")
    return pd.concat([history.tail(window_size), target])


def _train_and_evaluate(
    data: pd.DataFrame, evaluation_data: pd.DataFrame, agent_type: str,
    params: dict[str, Any], args: argparse.Namespace,
):
    system = RLTradingSystem()
    train_env = _environment(data, args, agent_type, params)
    system._env = train_env
    agent = _make_agent(system, train_env, agent_type, params, args)
    training = system.train(
        agent=agent,
        episodes=args.episodes,
        batch_size=args.batch_size,
        max_total_steps=args.max_training_steps,
        train_every=args.train_every,
        max_optimization_steps=args.max_optimization_steps,
    )
    evaluation = evaluate_out_of_sample(agent, _environment(evaluation_data, args, agent_type, params))
    return agent, training, evaluation


def _write_evaluation(
    directory: Path, asset: str, raw: pd.DataFrame, evaluation, training: dict[str, Any],
    args: argparse.Namespace, agent_type: str, params: dict[str, Any], stage: str,
) -> dict[str, Any]:
    profile = params.get("risk", {})
    paths = save_report(
        directory, asset, args.timeframe, raw, evaluation, args.transaction_fee,
        {
            **training, "stage": stage, "agent_type": agent_type, "parameters": params,
            "device": "cuda" if torch.cuda.is_available() else "cpu", "source_exchange": args.exchange,
            "slippage": args.slippage, "reward_function": args.reward_function, "test_evaluated_once": stage == "test",
            "max_position_fraction": args.max_position_fraction,
            "max_trade_fraction": profile.get("max_trade_fraction", args.max_trade_fraction),
            "min_trade_interval": profile.get("min_trade_interval", args.min_trade_interval),
            "min_signal_confidence": profile.get("min_signal_confidence", 0.0),
            "benchmark_comparison": "risk_matched_buy_and_hold",
            "risk_profile": params.get("risk", {}),
        },
        data_source=(
            "Yahoo Finance GC=F (proxy XAU/USD)"
            if asset == "XAU/USD"
            else "CCXT public OHLCV"
        ),
    )
    return json.loads(paths["metrics"].read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Path]:
    if not 0.0 <= args.max_agent_order_rate <= 1.0:
        raise ValueError("max_agent_order_rate doit être dans [0, 1]")
    _seed(args.seed)
    output_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    all_results: list[dict[str, Any]] = []
    assets_summary: dict[str, Any] = {}
    for asset in args.assets:
        raw = _fetch_ohlcv(asset, args)
        if args.external_features:
            external = pd.read_csv(args.external_features, parse_dates=["timestamp"])
            if "symbol" in external.columns:
                external = external.loc[external["symbol"] == asset].drop(columns="symbol")
            if external.empty:
                raise ValueError(f"Aucune feature externe disponible pour {asset}")
            external = external.set_index("timestamp")
            if getattr(external.index, "tz", None) is not None:
                external.index = external.index.tz_convert(None)
            raw = merge_causal_external_features(raw, external)
        windows = build_walk_forward_windows(
            raw, args.train_candles, args.validation_candles, args.test_candles, args.step_candles
        )
        if args.window_offset < 0 or args.window_offset >= len(windows):
            raise ValueError("window_offset doit désigner une fenêtre walk-forward existante")
        windows = windows[args.window_offset :]
        if args.max_windows is not None:
            windows = windows[: args.max_windows]
        asset_rows = []
        for window in windows:
            if min(len(window.train), len(window.validation), len(window.test)) <= args.window_size:
                raise ValueError("Une partition doit être plus longue que window_size")
            candidates = []
            for agent_type in args.agent_types:
                for candidate_index, params in enumerate(DEFAULT_CANDIDATES[agent_type]):
                    seed_validations = []
                    for seed in args.validation_seeds:
                        _seed(seed + window.index * 10_000 + candidate_index * 100)
                        _, training, validation_eval = _train_and_evaluate(
                            window.train,
                            _evaluation_with_context(window.train, window.validation, args.window_size),
                            agent_type, params, args,
                        )
                        validation_dir = (
                            output_dir / asset.replace("/", "_") / f"window_{window.index:02d}"
                            / "validation" / f"{agent_type}_{candidate_index}" / f"seed_{seed}"
                        )
                        seed_validations.append(_write_evaluation(
                            validation_dir, asset, raw, validation_eval, training, args, agent_type, params, "validation"
                        ))
                    candidates.append({
                        "agent_type": agent_type,
                        "parameters": params,
                        "validation": aggregate_validation_metrics(seed_validations),
                        "validation_by_seed": seed_validations,
                    })
            try:
                selected = select_candidate_on_validation(
                    candidates, min_closed_trades=args.min_closed_trades,
                    max_agent_order_rate=args.max_agent_order_rate,
                )
            except ValueError as error:
                # Ne jamais forcer un modèle non qualifié sur le test : cela
                # transformerait un rejet de validation en faux résultat OOS.
                row = {
                    "window": window.index,
                    "asset": asset,
                    "train_start": window.train.index[0].isoformat(), "train_end": window.train.index[-1].isoformat(),
                    "validation_start": window.validation.index[0].isoformat(), "validation_end": window.validation.index[-1].isoformat(),
                    "test_start": window.test.index[0].isoformat(), "test_end": window.test.index[-1].isoformat(),
                    "selected_agent_type": None,
                    "selected_parameters": None,
                    "validation_rejected": True,
                    "validation_rejection_reason": str(error),
                    "validation_candidates": [
                        {
                            "agent_type": candidate["agent_type"],
                            "parameters": candidate["parameters"],
                            "objective": validation_objective(candidate["validation"]),
                            "eligible": candidate["validation_eligible"],
                            "rejection_reason": candidate["validation_rejection_reason"],
                            "seed_count": candidate["validation"]["validation_seed_count"],
                        }
                        for candidate in candidates
                    ],
                }
                asset_rows.append(row)
                all_results.append(row)
                continue
            # Une fois le candidat choisi, il est ré-entraîné avec train+validation.
            # La partition test reste intacte et n'est évaluée qu'une seule fois.
            _seed(args.seed + window.index * 1000 + 99)
            train_validation = pd.concat([window.train, window.validation])
            _, training, test_eval = _train_and_evaluate(
                train_validation,
                _evaluation_with_context(train_validation, window.test, args.window_size),
                selected["agent_type"], selected["parameters"], args,
            )
            test_dir = output_dir / asset.replace("/", "_") / f"window_{window.index:02d}" / "test"
            test = _write_evaluation(
                test_dir, asset, raw, test_eval, training, args,
                selected["agent_type"], selected["parameters"], "test",
            )
            row = {
                "window": window.index,
                "asset": asset,
                "train_start": window.train.index[0].isoformat(), "train_end": window.train.index[-1].isoformat(),
                "validation_start": window.validation.index[0].isoformat(), "validation_end": window.validation.index[-1].isoformat(),
                "test_start": window.test.index[0].isoformat(), "test_end": window.test.index[-1].isoformat(),
                "selected_agent_type": selected["agent_type"], "selected_parameters": selected["parameters"],
                "validation_objective": validation_objective(selected["validation"]),
                "validation_candidates": [
                    {
                        "agent_type": candidate["agent_type"],
                        "parameters": candidate["parameters"],
                        "objective": validation_objective(candidate["validation"]),
                        "eligible": candidate["validation_eligible"],
                        "rejection_reason": candidate["validation_rejection_reason"],
                        "seed_count": candidate["validation"]["validation_seed_count"],
                    }
                    for candidate in candidates
                ],
                "test": test,
            }
            asset_rows.append(row)
            all_results.append(row)
        assets_summary[asset] = {"windows": asset_rows, "phase4_gate": phase4_gate(asset_rows, args.min_windows)}
    summary = {
        "protocol": {
            "selection": "validation only", "test": "one evaluation per selected model/window",
            "assets": args.assets, "agent_types": args.agent_types, "transaction_fee": args.transaction_fee,
            "slippage": args.slippage, "max_position_fraction": args.max_position_fraction,
            "window_offset": args.window_offset,
            "reward_function": args.reward_function,
            "min_closed_trades": args.min_closed_trades,
            "external_features": bool(args.external_features),
            "validation_seeds": args.validation_seeds,
            "synthetic_data": False,
        },
        "assets": assets_summary,
        "overall_phase4_gate": phase4_gate(all_results, args.min_windows * len(args.assets)),
    }
    summary_path = output_dir / "walk_forward_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=["BTC/USDT", "ETH/USDT", "XAU/USD"])
    parser.add_argument("--agent-types", nargs="+", choices=tuple(DEFAULT_CANDIDATES), default=["dqn", "ppo", "sac"])
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--timeframe", choices=("1h", "2h", "4h", "1d"), default="1d")
    parser.add_argument("--days", type=int, default=2200)
    parser.add_argument("--train-candles", type=int, default=600)
    parser.add_argument("--validation-candles", type=int, default=180)
    parser.add_argument("--test-candles", type=int, default=180)
    parser.add_argument("--step-candles", type=int, default=180)
    parser.add_argument("--min-windows", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=None, help="Borne le nombre de fenêtres sans les chevaucher.")
    parser.add_argument(
        "--window-offset",
        type=int,
        default=0,
        help="Ignore les premières fenêtres pour auditer une période donnée, sans chevauchement.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-training-steps", type=int, default=10_000)
    parser.add_argument("--max-optimization-steps", type=int, default=1_500)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--ppo-update-epochs", type=int, default=2)
    parser.add_argument("--sac-feedforward", type=int, default=128)
    parser.add_argument("--sac-sequence-length", type=int, default=4)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--transaction-fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--max-trade-fraction", type=float, default=0.30)
    parser.add_argument("--min-trade-interval", type=int, default=0, help="Nombre minimal de bougies entre deux ordres agent.")
    parser.add_argument(
        "--max-position-fraction",
        type=float,
        default=0.20,
        help="Exposition maximale du risque ; le benchmark comparatif utilise la même exposition.",
    )
    parser.add_argument("--invalid-action-penalty", type=float, default=0.0005)
    parser.add_argument("--reward-function", choices=("excess_return", "risk_adjusted_excess"), default="risk_adjusted_excess")
    parser.add_argument("--reward-drawdown-weight", type=float, default=0.05)
    parser.add_argument("--reward-downside-weight", type=float, default=0.01)
    parser.add_argument("--min-closed-trades", type=int, default=30)
    parser.add_argument("--max-agent-order-rate", type=float, default=0.35)
    parser.add_argument(
        "--validation-seeds", nargs="+", type=int, default=[42, 314, 2024],
        help="Seeds indépendantes, agrégées uniquement pour choisir sur validation.",
    )
    parser.add_argument("--external-features", help="CSV horodaté (timestamp, colonnes numériques, symbol facultatif) fusionné causalement.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="ai_trading/info_retour/walk_forward_reports")
    return parser.parse_args()


if __name__ == "__main__":
    for name, path in run(parse_args()).items():
        print(f"{name}: {path}")
