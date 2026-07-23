import numpy as np
import torch

from ai_trading.rl.agents.multitask_agent import MultitaskTradingAgent


def test_multitask_agent_performs_real_gpu_gradient_update():
    agent = MultitaskTradingAgent(
        state_dim=5, action_dim=1, num_assets=1, d_model=8,
        n_heads=2, num_layers=1, max_seq_len=4, device="cuda",
    )
    before = next(agent.model.parameters()).detach().clone()
    states = np.random.default_rng(5).normal(size=(2, 5)).astype(np.float32)
    losses = agent.update(states, np.zeros((2, 1), dtype=np.float32), [0.01, -0.01], states + .1, [False, False])

    after = next(agent.model.parameters()).detach()
    assert {"total_loss", "multitask_loss", "policy_loss"} <= losses.keys()
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.allclose(before, after)
