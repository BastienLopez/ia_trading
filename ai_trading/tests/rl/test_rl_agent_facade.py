"""Le point d'entrée API RL délègue au cœur RL, sans seconde implémentation."""

import numpy as np
import torch

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl_agent import RLAgent


def test_facade_uses_canonical_dqn_and_restores_its_environment_contract(tmp_path):
    data = RLDataIntegrator().generate_synthetic_data(
        n_samples=48, trend="bullish", volatility=0.01, with_sentiment=True
    )
    facade = RLAgent(model_dir=tmp_path)
    metrics = facade.train(data, total_timesteps=30)

    assert metrics["total_steps"] == 30
    assert metrics["optimization_steps"] >= 1
    assert facade.agent.device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert facade.agent.environment_config["window_size"] > 0

    checkpoint = tmp_path / "api_dqn.zip"
    facade.save(checkpoint)
    restored = RLAgent(model_dir=tmp_path)
    restored.load(checkpoint)
    assert restored.agent.environment_config == facade.agent.environment_config

    state, _ = facade.env.reset()
    assert restored.predict(state) in range(facade.env.action_space.n)
    result = facade.backtest(data)
    assert np.isfinite(result["profit_pct"])
