"""Entraînement SAC chronologique sur plusieurs unités de temps."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment


class MultiPeriodTrainer:
    """Entraîne le même contrat SAC du timeframe lent vers le plus rapide.

    Les données sont fournies explicitement par timeframe. Cela rend la validation
    reproductible et évite qu'un entraînement change silencieusement selon une API.
    """

    def __init__(
        self,
        symbol: str,
        days: int,
        periods: Iterable[int],
        agent_type: str = "sac",
        use_gru: bool = False,
        initial_balance: float = 10_000.0,
        use_curriculum: bool = True,
        epochs_per_period: int = 1,
        episodes_per_epoch: int = 1,
        validation_ratio: float = 0.2,
        include_sentiment: bool = True,
        sequence_length: int = 10,
        gru_units: int = 64,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        tau: float = 0.005,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        reward_scaling: float = 1.0,
        action_type: str = "continuous",
        save_dir: Union[str, Path, None] = None,
        max_steps: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if agent_type != "sac" or action_type != "continuous":
            raise ValueError("MultiPeriodTrainer ne supporte actuellement que SAC continu.")
        if not 0 < validation_ratio < 0.5:
            raise ValueError("validation_ratio doit être dans ]0, 0.5[.")
        self.symbol, self.days = symbol, days
        self.periods = sorted(set(periods), reverse=True)  # lent vers rapide
        self.agent_type, self.use_gru = agent_type, use_gru
        self.initial_balance, self.use_curriculum = initial_balance, use_curriculum
        self.epochs_per_period, self.episodes_per_epoch = epochs_per_period, episodes_per_epoch
        self.validation_ratio, self.include_sentiment = validation_ratio, include_sentiment
        self.sequence_length, self.gru_units = sequence_length, gru_units
        self.actor_lr, self.critic_lr, self.tau = actor_lr, critic_lr, tau
        self.batch_size, self.buffer_size = batch_size, buffer_size
        self.reward_scaling, self.max_steps = reward_scaling, max_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir) if save_dir else INFO_RETOUR_DIR / "models" / "multi_period"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.agents: Dict[int, SACAgent] = {}
        self.metrics: Dict[int, Dict[str, float]] = {}

    def prepare_datasets(
        self, market_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None,
        validation_ratio: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame]]:
        ratio = self.validation_ratio if validation_ratio is None else validation_ratio
        if len(market_data) < 12:
            raise ValueError("Au moins 12 lignes sont requises pour une séparation temporelle.")
        data = market_data.sort_index().copy()
        split = int(len(data) * (1 - ratio))
        train, validation = data.iloc[:split], data.iloc[split:]
        if sentiment_data is None:
            return train, None, validation, None
        sentiment = sentiment_data.sort_index()
        return train, sentiment.loc[sentiment.index < validation.index[0]], validation, sentiment.loc[sentiment.index >= validation.index[0]]

    def create_env(
        self, market_data: pd.DataFrame, is_training: bool = True,
        window_size: Optional[int] = None,
    ) -> TradingEnvironment:
        del is_training
        resolved_window_size = window_size or min(20, max(3, len(market_data) // 5))
        return TradingEnvironment(
            market_data.copy(), initial_balance=self.initial_balance,
            window_size=resolved_window_size,
            action_type="continuous", reward_function="sharpe", risk_management=True,
            # Les indicateurs sont calculés sur chaque split : les désactiver
            # ici préserve exactement le même espace entre train/validation.
            include_technical_indicators=False,
        )

    def create_agent(self, env: TradingEnvironment) -> SACAgent:
        return SACAgent(
            state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
            actor_learning_rate=self.actor_lr, critic_learning_rate=self.critic_lr,
            tau=self.tau, batch_size=self.batch_size, buffer_size=self.buffer_size,
            sequence_length=self.sequence_length, use_gru=self.use_gru,
            gru_units=self.gru_units, device=self.device,
        )

    def _run_episode(self, env: TradingEnvironment, agent: SACAgent, training: bool) -> float:
        state, _ = env.reset()
        reward_sum, steps, done = 0.0, 0, False
        while not done and (self.max_steps is None or steps < self.max_steps):
            action = agent.select_action(state, deterministic=not training)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if training:
                agent.remember(state, action, reward * self.reward_scaling, next_state, done)
                if len(agent.replay_buffer) >= agent.batch_size:
                    agent.train()
            state, reward_sum, steps = next_state, reward_sum + reward, steps + 1
        if training:
            agent.end_episode()
        return float(reward_sum)

    def train_multi_period(self, data_by_period: Dict[int, pd.DataFrame]) -> Dict[int, Dict[str, float]]:
        missing = set(self.periods).difference(data_by_period)
        if missing:
            raise ValueError(f"Données absentes pour les périodes: {sorted(missing)}")
        for period in self.periods:
            train_data, _, validation_data, _ = self.prepare_datasets(data_by_period[period])
            window_size = min(20, max(3, len(train_data) // 5))
            if len(validation_data) <= window_size:
                raise ValueError("Le split de validation est trop court pour la fenêtre d'observation.")
            train_env = self.create_env(train_data, window_size=window_size)
            validation_env = self.create_env(validation_data, window_size=window_size)
            agent = self.create_agent(train_env)
            rewards = [self._run_episode(train_env, agent, True) for _ in range(self.epochs_per_period * self.episodes_per_epoch)]
            validation_reward = self._run_episode(validation_env, agent, False)
            path = self.save_dir / f"{self.symbol}_{period}min.pt"
            agent.save(path)
            self.agents[period] = agent
            self.metrics[period] = {"train_reward": float(np.mean(rewards)), "validation_reward": validation_reward, "checkpoint": str(path)}
        return self.metrics
