"""Contrats d'intégration courts pour l'orchestrateur RL réel."""

import numpy as np
import pytest
import torch

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.trading_system import RLTradingSystem


@pytest.fixture
def market_data():
    return RLDataIntegrator().generate_synthetic_data(
        n_samples=64, trend="bullish", volatility=0.01, with_sentiment=True
    )


def test_dqn_gpu_training_evaluation_and_checkpoint_roundtrip(tmp_path, market_data):
    """La même instance DQN est entraînée, évaluée et rechargée sans mock."""
    system = RLTradingSystem()
    env = system.create_environment(data=market_data, window_size=10, action_type="discrete")
    agent = system.create_agent(
        agent_type="dqn",
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=4,
        buffer_size=64,
        hidden_size=32,
        use_ucb=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    training = system.train(agent=agent, episodes=1, batch_size=4, max_steps=8)
    assert training["optimization_steps"] >= 1
    assert len(agent.memory) == 8

    evaluation = system.evaluate(agent=agent, env=env, num_episodes=1)
    assert np.isfinite(evaluation["final_value"])
    assert len(evaluation["portfolio_history"]) > 0
    assert all(0 <= action < env.action_space.n for action in evaluation["actions"])

    checkpoint = tmp_path / "dqn.pt"
    system.save(str(checkpoint))
    restored = RLTradingSystem().load(str(checkpoint))
    assert restored.state_size == env.observation_space.shape[0]
    assert restored.predict(env.reset()[0]) in range(env.action_space.n)


def test_sac_continuous_training_and_evaluation_uses_the_real_contract(market_data):
    """SAC reçoit des actions continues et effectue au moins une mise à jour réelle."""
    system = RLTradingSystem()
    env = system.create_environment(data=market_data, window_size=10, action_type="continuous")
    agent = system.create_agent(
        agent_type="sac",
        state_size=env.observation_space.shape[0],
        batch_size=2,
        buffer_size=32,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        sequence_length=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    training = system.train(agent=agent, episodes=1, batch_size=2, max_steps=3)
    assert training["optimization_steps"] >= 1


def test_ppo_continuous_training_uses_the_orchestrator_contract(market_data):
    system = RLTradingSystem()
    env = system.create_environment(
        data=market_data, window_size=10, action_type="continuous", risk_management=False
    )
    agent = system.create_agent(
        agent_type="ppo",
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        hidden_size=32,
        update_epochs=1,
        mini_batch_size=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    history = system.train(agent=agent, episodes=1, max_steps=4)
    assert history["optimization_steps"] == 1
    assert len(history["episode_rewards"]) == 1
    result = system.evaluate(agent=agent, env=env, num_episodes=1)
    assert np.isfinite(result["final_value"])
    assert all(np.asarray(action).shape == (1,) for action in result["actions"])


def test_dimension_mismatch_is_an_error_not_a_silent_agent_replacement(market_data):
    system = RLTradingSystem()
    env = system.create_environment(data=market_data, window_size=10)
    agent = system.create_agent(agent_type="dqn", state_size=3, action_size=3, device="cpu")

    with pytest.raises(ValueError, match="Dimensions incompatibles"):
        system.evaluate(agent=agent, env=env)


def test_orchestrator_rejects_implicit_synthetic_market_data():
    system = RLTradingSystem()

    with pytest.raises(ValueError, match="OHLCV réelles explicites"):
        system.create_environment()
    with pytest.raises(ValueError, match="OHLCV réelles explicites"):
        system.integrate_data()


def test_double_dueling_dqn_is_trainable_through_the_same_orchestrator(market_data):
    system = RLTradingSystem()
    env = system.create_environment(
        data=market_data, window_size=10, action_type="discrete", risk_management=False
    )
    agent = system.create_agent(
        agent_type="double_dueling_dqn",
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=4,
        buffer_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    history = system.train(agent=agent, episodes=1, batch_size=4, max_steps=5)
    assert history["optimization_steps"] >= 1
    assert agent.use_dueling
