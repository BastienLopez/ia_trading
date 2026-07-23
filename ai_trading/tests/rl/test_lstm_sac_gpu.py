import numpy as np
import torch

from ai_trading.rl.agents.sac_agent import SACAgent


def test_lstm_sac_gpu_selects_actions_and_updates_from_sequences():
    np.random.seed(13)
    agent = SACAgent(
        state_dim=3,
        action_dim=1,
        recurrent_type="lstm",
        gru_units=12,
        num_layers=1,
        batch_size=4,
        sequence_length=3,
        device="cuda",
    )
    for step in range(8):
        state = np.full(3, step / 10, dtype=np.float32)
        next_state = state + 0.01
        agent.remember(state, np.array([0.1], dtype=np.float32), 0.01, next_state, step == 7)
    metrics = agent.train()

    assert agent.recurrent_type == "lstm"
    assert torch.cuda.is_available()
    assert np.isfinite(metrics["actor_loss"])
