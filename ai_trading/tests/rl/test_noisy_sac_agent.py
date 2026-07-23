import pytest

from ai_trading.rl.agents.noisy_sac_agent import NoisySACAgent


def test_noisy_sac_compatibility_import_is_explicitly_deprecated():
    with pytest.warns(DeprecationWarning, match="OptimizedSACAgent"):
        agent = NoisySACAgent(
            state_dim=4,
            action_dim=1,
            d_model=8,
            n_heads=2,
            num_layers=1,
            dim_feedforward=16,
            batch_size=2,
        )

    assert agent.use_gru is False
