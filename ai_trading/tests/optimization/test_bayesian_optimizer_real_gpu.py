import numpy as np
import pandas as pd

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.bayesian_optimizer import BayesianOptimizer
from ai_trading.rl.trading_environment import TradingEnvironment


def test_bayesian_optimization_evaluates_real_sac_candidates_on_gpu(tmp_path):
    prices = 100 + np.cumsum(np.sin(np.linspace(0, 4, 36)) * 0.3 + 0.05)
    frame = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.full_like(prices, 1000.0),
        },
        index=pd.date_range("2024-01-01", periods=len(prices), freq="h"),
    )

    def make_env():
        return TradingEnvironment(
            frame,
            window_size=5,
            action_type="continuous",
            include_technical_indicators=False,
            reward_function="simple",
        )

    optimizer = BayesianOptimizer(
        env_creator=make_env,
        agent_class=SACAgent,
        param_space={
            "learning_rate": (1e-4, 2e-4),
            "batch_size": [4],
            "hidden_size": [16],
            "d_model": [16],
            "n_heads": [2],
            "num_layers": [1],
            "dim_feedforward": [32],
            "sequence_length": [4],
            "max_seq_len": [8],
            "device": ["cuda"],
        },
        n_episodes=1,
        max_steps=5,
        eval_episodes=1,
        save_dir=tmp_path,
        n_initial_points=2,
        n_iterations=1,
        verbose=0,
    )
    params, score = optimizer.bayesian_optimization()

    assert params["device"] == "cuda"
    assert np.isfinite(score)
    assert len(optimizer.results) == 3
    assert len(list(tmp_path.glob("bayesian_optimization_results_*.json"))) == 1
    assert len(list(tmp_path.glob("convergence_plot_*.png"))) == 1
