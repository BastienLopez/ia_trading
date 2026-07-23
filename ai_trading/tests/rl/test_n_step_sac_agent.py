import numpy as np
import pytest

from ai_trading.rl.agents.n_step_sac_agent import NStepSACAgent


def _agent():
    return NStepSACAgent(
        state_size=4,
        action_size=1,
        n_steps=3,
        discount_factor=0.5,
        batch_size=2,
        d_model=8,
        n_heads=2,
        num_layers=1,
        dim_feedforward=16,
        device="cuda",
    )


def test_n_step_sac_stores_exact_discounted_return_and_horizon():
    agent = _agent()
    state = np.zeros(4, dtype=np.float32)
    action = np.array([0.1], dtype=np.float32)

    for reward in (1.0, 2.0, 3.0):
        agent.remember(state, action, reward, state + reward, False)

    assert len(agent.replay_buffer) == 1
    transition = agent.replay_buffer.buffer[0]
    assert transition[2] == pytest.approx(2.75)
    assert transition[5] == pytest.approx(0.125)


def test_n_step_sac_flushes_partial_episode_without_losing_transitions():
    agent = _agent()
    state = np.zeros(4, dtype=np.float32)
    action = np.array([0.1], dtype=np.float32)

    agent.remember(state, action, 1.0, state + 1, False)
    agent.remember(state + 1, action, 2.0, state + 2, False)
    agent.episode_end()

    assert len(agent.replay_buffer) == 2
    assert agent.replay_buffer.buffer[0][2] == pytest.approx(2.0)
    assert agent.replay_buffer.buffer[0][5] == pytest.approx(0.25)
    assert agent.replay_buffer.buffer[1][2] == pytest.approx(2.0)
    assert agent.replay_buffer.buffer[1][5] == pytest.approx(0.5)


def test_n_step_sac_runs_real_gpu_update_and_checkpoint_round_trip(tmp_path):
    agent = _agent()
    rng = np.random.default_rng(4)
    for _ in range(6):
        state = rng.normal(size=4).astype(np.float32)
        next_state = rng.normal(size=4).astype(np.float32)
        agent.remember(state, np.array([0.2], dtype=np.float32), 0.1, next_state, False)
    agent.episode_end()

    metrics = agent.train()
    path = tmp_path / "n_step_sac.pt"
    agent.save(path)
    agent.load(path)

    assert {"actor_loss", "critic1_loss", "critic2_loss"} <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())
    assert path.is_file()
