"""Exécute un entraînement RL borné et produit un rapport sur bougies réelles.

Exemple Docker :
python -m ai_trading.scripts.run_real_market_rl_report --symbol BTC/USDT --days 365 --timeframe 4h --episodes 20 --max-training-steps 20000
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai_trading.data_processor import DataProcessor
from ai_trading.rl.real_market_report import (
    EvaluationRun,
    buy_and_hold_equity,
    chronological_split,
    save_report,
    validate_ohlcv,
)
from ai_trading.rl.causal_baselines import causal_single_asset_baselines
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.trading_system import RLTradingSystem


def evaluate_out_of_sample(agent, environment: TradingEnvironment) -> EvaluationRun:
    """Évalue une politique figée sans exploration et conserve chaque bougie."""
    state, _ = environment.reset()
    equity = [environment.initial_balance]
    dates = [environment.df.index[environment.current_step]]
    rewards: list[float] = []
    actions: list[float | int] = []
    executions: list[bool] = []
    execution_reasons: list[str | None] = []
    fills: list[dict] = []
    done = False
    while not done:
        action = RLTradingSystem._select_agent_action(
            agent,
            np.asarray(state, dtype=np.float32),
            training=False,
            action_mask=environment.get_action_mask(),
        )
        action = environment.project_action(action)
        state, reward, terminated, truncated, info = environment.step(action)
        done = bool(terminated or truncated)
        equity.append(float(info["portfolio_value"]))
        dates.append(environment.df.index[environment.current_step])
        rewards.append(float(reward))
        actions.append(int(action) if environment.action_type == "discrete" else float(np.asarray(action).reshape(-1)[0]))
        executions.append(bool(info["trade_executed"]))
        execution_reasons.append(info["execution_reason"])
        fills = info["trade_events"]

    prices = environment.df["close"].iloc[
        environment.window_size : environment.window_size + len(equity)
    ]
    benchmark = buy_and_hold_equity(prices, environment.initial_balance, environment.transaction_fee)
    risk_exposure = (
        environment.risk_manager.max_position_size if environment.risk_management else 1.0
    )
    risk_matched_benchmark = buy_and_hold_equity(
        prices,
        environment.initial_balance,
        environment.transaction_fee,
        exposure_fraction=risk_exposure,
    )
    baselines = causal_single_asset_baselines(
        environment.df.iloc[
            environment.window_size : environment.window_size + len(equity)
        ],
        initial_balance=environment.initial_balance,
        fee=environment.transaction_fee,
        slippage=environment.slippage_value,
    )
    regimes = environment.df["market_regime"].iloc[
        environment.window_size : environment.window_size + len(equity)
    ].astype(str).tolist()
    return EvaluationRun(
        dates=pd.DatetimeIndex(dates),
        equity=np.asarray(equity, dtype=float),
        benchmark_equity=benchmark,
        rewards=rewards,
        actions=actions,
        executions=executions,
        execution_reasons=execution_reasons,
        fills=fills,
        action_type=environment.action_type,
        n_discrete_actions=environment.n_discrete_actions,
        risk_matched_benchmark_equity=risk_matched_benchmark,
        baseline_equity={name: result.equity for name, result in baselines.items()},
        market_regimes=regimes,
    )


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.days < 30:
        raise ValueError("Utiliser au moins 30 jours afin de conserver un jeu de test utile")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    processor = DataProcessor(use_float16=False, use_cache=False)
    raw = processor.download_historical_data(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start,
        end_date=end,
        save=False,
    )
    # La dernière bougie retournée peut être encore ouverte. Elle n'est jamais
    # utilisée, afin que le test hors-échantillon ne contienne que des prix clos.
    closed_until = pd.Timestamp.now(tz="UTC").floor(args.timeframe).tz_localize(None)
    raw = raw.loc[raw.index < closed_until]
    raw = validate_ohlcv(raw)
    train_data, test_data = chronological_split(raw, args.train_ratio)
    if len(test_data) < args.window_size + args.min_evaluation_candles:
        raise ValueError(
            "Jeu de test trop court après la fenêtre d'observation: "
            f"{len(test_data) - args.window_size} bougies évaluables, "
            f"minimum {args.min_evaluation_candles}"
        )
    train_environment = TradingEnvironment(
        train_data,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        reward_function=args.reward_function,
        reward_drawdown_weight=args.reward_drawdown_weight,
        reward_downside_weight=args.reward_downside_weight,
        window_size=args.window_size,
        slippage_value=args.slippage,
        max_trade_fraction=args.max_trade_fraction,
        min_trade_interval=args.min_trade_interval,
        risk_config={"max_position_size": args.max_position_fraction},
        benchmark_exposure=args.max_position_fraction,
        invalid_action_penalty=args.invalid_action_penalty,
        normalize_observation=args.adaptive_normalization,
    )
    system = RLTradingSystem()
    system._env = train_environment
    agent = system.create_agent(
        "dqn",
        state_size=train_environment.observation_space.shape[0],
        action_size=train_environment.action_space.n,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    training = system.train(
        agent=agent,
        episodes=args.episodes,
        batch_size=args.batch_size,
        max_total_steps=args.max_training_steps,
        train_every=args.train_every,
        max_optimization_steps=args.max_optimization_steps,
    )
    test_environment = TradingEnvironment(
        test_data,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        reward_function=args.reward_function,
        reward_drawdown_weight=args.reward_drawdown_weight,
        reward_downside_weight=args.reward_downside_weight,
        window_size=args.window_size,
        slippage_value=args.slippage,
        max_trade_fraction=args.max_trade_fraction,
        min_trade_interval=args.min_trade_interval,
        risk_config={"max_position_size": args.max_position_fraction},
        benchmark_exposure=args.max_position_fraction,
        invalid_action_penalty=args.invalid_action_penalty,
        normalize_observation=args.adaptive_normalization,
    )
    if test_environment.observation_space.shape != train_environment.observation_space.shape:
        raise RuntimeError("Les espaces d'observation train/test diffèrent")
    evaluation = evaluate_out_of_sample(agent, test_environment)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(args.output_dir) / args.symbol.replace("/", "_") / timestamp
    return save_report(
        output_dir=report_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
        raw_data=raw,
        run=evaluation,
        transaction_fee=args.transaction_fee,
        training_metadata={
            **training,
            "device": str(agent.device),
            "source_exchange": args.exchange,
            "train_candles": len(train_data),
            "test_candles_before_environment_warmup": len(test_data),
            "window_size": args.window_size,
            "slippage": args.slippage,
            "max_trade_fraction": args.max_trade_fraction,
            "min_trade_interval": args.min_trade_interval,
            "max_position_fraction": args.max_position_fraction,
            "train_every": args.train_every,
            "max_optimization_steps": args.max_optimization_steps,
            "reward_function": args.reward_function,
            "reward_drawdown_weight": args.reward_drawdown_weight,
            "reward_downside_weight": args.reward_downside_weight,
            "invalid_action_penalty": args.invalid_action_penalty,
            "adaptive_normalization": args.adaptive_normalization,
            "closed_candles_before": closed_until.isoformat(),
            "seed": args.seed,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--timeframe", default="4h", choices=("1h", "2h", "4h", "1d"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-training-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--max-optimization-steps", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--min-evaluation-candles", type=int, default=60)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--transaction-fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--max-trade-fraction", type=float, default=0.30)
    parser.add_argument("--min-trade-interval", type=int, default=0)
    parser.add_argument(
        "--max-position-fraction",
        type=float,
        default=0.20,
        help="Exposition maximale et benchmark de récompense comparable.",
    )
    parser.add_argument("--invalid-action-penalty", type=float, default=0.0005)
    parser.add_argument("--reward-function", choices=("excess_return", "risk_adjusted_excess"), default="risk_adjusted_excess")
    parser.add_argument("--reward-drawdown-weight", type=float, default=0.05)
    parser.add_argument("--reward-downside-weight", type=float, default=0.01)
    parser.add_argument(
        "--adaptive-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activer la normalisation adaptative coûteuse (désactivée pour les rapports GPU).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="ai_trading/info_retour/real_market_reports")
    return parser.parse_args()


if __name__ == "__main__":
    paths = run(parse_args())
    for name, path in paths.items():
        print(f"{name}: {path}")
