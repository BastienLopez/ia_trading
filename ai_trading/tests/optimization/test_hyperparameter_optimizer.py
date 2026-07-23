import numpy as np
import pytest

from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.hyperparameter_optimizer import HyperparameterOptimizer
from ai_trading.rl.trading_environment import TradingEnvironment


@pytest.fixture
def market_data():
    data = generate_synthetic_market_data(
        n_points=80, trend=0.001, volatility=0.01, start_price=100.0
    )
    return data.fillna(0.0)


def _environment_creator(market_data):
    return lambda: TradingEnvironment(
        df=market_data.copy(),
        initial_balance=10_000,
        transaction_fee=0.001,
        window_size=5,
        reward_function="simple",
        action_type="continuous",
    )


def test_parameter_combinations_and_score_use_real_contract(market_data, tmp_path):
    optimizer = HyperparameterOptimizer(
        env_creator=_environment_creator(market_data),
        agent_class=SACAgent,
        param_grid={"a": [1, 2], "b": [3, 4]},
        save_dir=tmp_path,
        verbose=0,
    )

    assert len(optimizer._get_param_combinations()) == 4
    score = optimizer._calculate_score(
        {
            "eval_avg_reward": 10.0,
            "total_reward": 100.0,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.1,
            "win_rate": 0.7,
        }
    )
    assert score == pytest.approx(33.46)


def test_grid_search_executes_a_real_sac_configuration_on_gpu(market_data, tmp_path):
    params = {
        "actor_learning_rate": [1e-3],
        "critic_learning_rate": [1e-3],
        "batch_size": [2],
        "hidden_size": [16],
        "d_model": [16],
        "n_heads": [2],
        "num_layers": [1],
        "dim_feedforward": [32],
        "sequence_length": [4],
        "max_seq_len": [4],
        "device": ["cuda"],
    }
    optimizer = HyperparameterOptimizer(
        env_creator=_environment_creator(market_data),
        agent_class=SACAgent,
        param_grid=params,
        n_episodes=1,
        max_steps=3,
        eval_episodes=1,
        save_dir=tmp_path,
        verbose=0,
    )

    best_params, best_score = optimizer.grid_search()

    assert best_params == {key: values[0] for key, values in params.items()}
    assert np.isfinite(best_score)
    assert len(optimizer.results) == 1
    assert {"sharpe_ratio", "max_drawdown", "win_rate"} <= optimizer.results[0]["metrics"].keys()


def test_gpu_grid_search_rejects_multiple_worker_processes(market_data, tmp_path):
    with pytest.raises(ValueError, match="GPU"):
        HyperparameterOptimizer(
            env_creator=_environment_creator(market_data),
            agent_class=SACAgent,
            param_grid={"device": ["cuda"]},
            n_jobs=2,
            save_dir=tmp_path,
            verbose=0,
        )
