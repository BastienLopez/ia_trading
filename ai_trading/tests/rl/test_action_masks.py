import numpy as np
import pandas as pd

from ai_trading.rl.agents.dqn_agent import DQNAgent
from ai_trading.rl.agents.ppo_agent import PPOAgent
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment


def _ohlcv(rows=80):
    dates = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = np.linspace(100.0, 110.0, rows)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=dates,
    )


def test_discrete_mask_prevents_dqn_from_selecting_a_sell_without_position():
    env = TradingEnvironment(_ohlcv(), window_size=10, risk_management=False)
    state, _ = env.reset()
    mask = env.get_action_mask()
    assert mask[0] and mask[1 : env.n_discrete_actions + 1].all()
    assert not mask[env.n_discrete_actions + 1 :].any()
    agent = DQNAgent(state.size, env.action_space.n, use_ucb=False, epsilon_start=0.0, device="cpu")
    for parameter in agent.policy_net.parameters():
        parameter.data.zero_()
    agent.policy_net[-1].bias.data[env.n_discrete_actions + 1] = 99.0
    assert agent.select_action(state, training=False, action_mask=mask) == 0


def test_dqn_target_backup_masks_an_impossible_sell_action():
    agent = DQNAgent(2, 3, batch_size=1, use_ucb=False, device="cpu")
    for parameter in agent.policy_net.parameters():
        parameter.data.zero_()
    for parameter in agent.target_net.parameters():
        parameter.data.zero_()
    agent.target_net[-1].bias.data[2] = 100.0
    agent.remember(np.zeros(2), 0, 0.0, np.zeros(2), False, next_action_mask=[True, True, False])
    assert np.isfinite(agent.optimize_model())


def test_continuous_masks_project_ppo_and_sac_to_non_negative_actions_without_inventory():
    env = TradingEnvironment(_ohlcv(), window_size=10, action_type="continuous", risk_management=False)
    state, _ = env.reset()
    mask = env.get_action_mask()
    assert mask["low"][0] == 0.0
    ppo = PPOAgent(state.size, 1, hidden_size=16, device="cpu")
    sac = SACAgent(state.size, 1, d_model=16, n_heads=2, num_layers=1, dim_feedforward=32, device="cpu")
    ppo_action, _ = ppo.get_action(state, deterministic=True, action_mask=mask)
    sac_action = sac.select_action(state, deterministic=True, action_mask=mask)
    assert ppo_action.reshape(-1)[0] >= 0.0
    assert sac_action.reshape(-1)[0] >= 0.0


def test_low_fusion_confidence_blocks_buys_but_keeps_a_realizable_sell():
    env = TradingEnvironment(_ohlcv(), window_size=10, risk_management=False, min_signal_confidence=0.8)
    env.df.loc[:, "signal_confidence"] = 0.1
    env.reset()
    assert not env.get_action_mask()[1 : env.n_discrete_actions + 1].any()
    env.crypto_held = 1.0
    assert env.get_action_mask()[env.n_discrete_actions + 1 :].any()
