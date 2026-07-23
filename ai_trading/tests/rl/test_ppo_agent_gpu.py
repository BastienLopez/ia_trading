import numpy as np
import torch

from ai_trading.rl.agents.ppo_agent import PPOAgent


def test_ppo_continuous_gpu_update_with_partial_minibatch():
    torch.manual_seed(11)
    agent = PPOAgent(
        state_dim=4,
        action_dim=1,
        hidden_size=16,
        update_epochs=2,
        mini_batch_size=5,
        device="cuda",
    )
    states = np.random.default_rng(11).normal(size=(7, 4)).astype(np.float32)
    actions = np.vstack([agent.get_action(state)[0] for state in states]).astype(np.float32)
    metrics = agent.update(
        states,
        actions,
        np.linspace(-0.1, 0.1, 7, dtype=np.float32),
        np.roll(states, -1, axis=0),
        np.array([False, False, False, False, False, False, True]),
    )

    assert all(np.isfinite(value) for value in metrics.values())
    assert len(agent.actor_loss_history) == 1
