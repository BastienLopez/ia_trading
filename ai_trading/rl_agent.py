"""Façade historique vers le système RL canonique.

Les endpoints API importaient auparavant une seconde implémentation incomplète
(portefeuille fictif, CPU forcé, actions non cohérentes). Cette façade conserve
l'API publique ``RLAgent`` tout en délégant à ``RLTradingSystem``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai_trading.rl.risk_manager import RiskManager
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.trading_system import RLTradingSystem


class RLAgent:
    """Compatibilité API pour l'entraînement, l'inférence et le backtest DQN."""

    def __init__(self, model_dir="info_retour/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.system = RLTradingSystem()
        self.agent = None
        self.env = None
        self.model_path = None

    def train(self, data, total_timesteps=10_000, save_path=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data doit être un DataFrame pandas")
        if total_timesteps <= 0:
            raise ValueError("total_timesteps doit être strictement positif")

        window_size = min(20, max(2, len(data) // 4))
        self.env = self.system.create_environment(
            data=data, window_size=window_size, action_type="discrete"
        )
        self.agent = self.system.create_agent(
            agent_type="dqn",
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            batch_size=min(64, max(4, min(total_timesteps, len(data) - window_size - 1))),
            buffer_size=max(1_000, total_timesteps),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.agent.environment_config = {
            "window_size": window_size,
            "action_type": "discrete",
        }

        steps_per_episode = max(1, len(data) - window_size - 1)
        episodes = math.ceil(total_timesteps / steps_per_episode)
        history = self.system.train(
            agent=self.agent,
            episodes=episodes,
            batch_size=self.agent.batch_size,
            max_steps=steps_per_episode,
            max_total_steps=total_timesteps,
            save_path=save_path,
        )
        if save_path:
            self.model_path = str(save_path)
        rewards = history["episode_rewards"]
        return {
            "total_episodes": history["episodes"],
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "optimization_steps": history["optimization_steps"],
            "total_steps": history["total_steps"],
        }

    def predict(self, state):
        if self.agent is None:
            raise ValueError("Aucun agent chargé ou entraîné")
        return int(self.system.predict_action(np.asarray(state, dtype=np.float32)))

    def backtest(self, data):
        if self.agent is None:
            raise ValueError("Aucun agent chargé ou entraîné")
        window_size = self.env.window_size if self.env is not None else min(20, max(2, len(data) // 4))
        env = self.system.create_environment(data=data, window_size=window_size)
        result = self.system.evaluate(agent=self.agent, env=env, num_episodes=1)
        initial_value = env.initial_balance
        final_value = result["final_value"]
        values = np.asarray(result["portfolio_history"], dtype=float)
        returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])
        executed = [
            value for value, action in zip(result["returns"], result["actions"])
            if np.asarray(action).any()
        ]
        first_price, last_price = float(data["close"].iloc[0]), float(data["close"].iloc[-1])
        return {
            "initial_balance": initial_value,
            "final_balance": final_value,
            "profit_pct": (final_value - initial_value) / initial_value * 100,
            "bh_profit_pct": (last_price - first_price) / first_price * 100,
            "sharpe_ratio": float(returns.mean() / (returns.std(ddof=0) + 1e-8) * np.sqrt(252)) if len(returns) else 0.0,
            "num_trades": len(executed),
            "trades": executed,
        }

    def load(self, model_path):
        self.agent = self.system.load(str(model_path))
        self.model_path = str(model_path)
        return self.agent

    def save(self, model_path):
        if self.agent is None:
            raise ValueError("Aucun agent à sauvegarder")
        self.system._agent = self.agent
        self.system.save(str(model_path))
        self.model_path = str(model_path)


__all__ = ["RLAgent", "RiskManager", "TradingEnvironment"]
