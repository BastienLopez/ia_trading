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


def test_ppo_gpu_repeated_large_continuous_updates_stay_finite():
    """Régression de la campagne multi-actifs : aucune moyenne PPO ne devient NaN."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    agent = PPOAgent(
        state_dim=452,
        action_dim=3,
        hidden_size=256,
        learning_rate=1e-4,
        update_epochs=10,
        mini_batch_size=64,
        device="cuda",
    )

    for _ in range(4):
        states = rng.uniform(-10, 10, size=(160, 452)).astype(np.float32)
        next_states = rng.uniform(-10, 10, size=(160, 452)).astype(np.float32)
        actions = np.vstack([agent.get_action(state)[0] for state in states]).astype(np.float32)
        metrics = agent.update(
            states,
            actions,
            rng.uniform(-1, 1, size=160).astype(np.float32),
            next_states,
            np.array([False] * 159 + [True]),
        )
        assert all(np.isfinite(value) for value in metrics.values())
        assert all(torch.isfinite(parameter).all() for parameter in agent.ac_network.parameters())
