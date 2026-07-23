import numpy as np
import pandas as pd

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.trading_environment import TradingEnvironment


def test_market_sentiment_to_gpu_rl_environment_contract_has_no_lookahead():
    dates = pd.date_range("2024-01-01", periods=40, freq="h")
    close = 100 + np.linspace(0, 2, len(dates))
    market = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(len(dates), 1000.0),
        },
        index=dates,
    )
    sentiment = pd.DataFrame({"compound_score": [0.7]}, index=[dates[5]])
    integrated = RLDataIntegrator().integrate_sentiment_data(market, sentiment)
    assert integrated.loc[dates[0], "compound_score"] == 0.0

    env = TradingEnvironment(
        integrated,
        window_size=5,
        action_type="continuous",
        include_technical_indicators=False,
        reward_function="simple",
    )
    state, _ = env.reset(seed=17)
    agent = SACAgent(
        state_dim=state.shape[0],
        action_dim=1,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        sequence_length=4,
        max_seq_len=8,
        batch_size=4,
        device="cuda",
    )
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)

    assert next_state.shape == state.shape
    assert np.isfinite(reward)
    assert not (terminated and truncated)
