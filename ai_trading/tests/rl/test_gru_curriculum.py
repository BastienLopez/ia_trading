import numpy as np
import pandas as pd
import pytest

from ai_trading.rl.curriculum_learning import (
    GRUCurriculumLearning,
    GRUCurriculumTrainer,
)


def _market_data(rows: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    close = 100 + np.cumsum(rng.normal(0.15, 0.45, rows))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": rng.integers(1_000, 5_000, rows),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="h"),
    )


def test_gru_curriculum_creates_real_gpu_capable_agent():
    curriculum = GRUCurriculumLearning(
        sequence_length=4, gru_units=8, hidden_size=16, batch_size=2
    )
    env = curriculum.create_environment(_market_data(), window_size=4)

    agent = curriculum.create_agent(env)

    assert agent.use_gru is True
    assert agent.sequence_length == 4
    assert agent.actor.gru.hidden_size == 8
    assert agent.device in {"cuda", "cpu"}


def test_gru_agent_learns_from_actual_sequences():
    curriculum = GRUCurriculumLearning(
        sequence_length=3, gru_units=8, hidden_size=16, batch_size=2
    )
    env = curriculum.create_environment(_market_data(), window_size=3)
    agent = curriculum.create_agent(env)
    state, _ = env.reset(seed=5)
    sequence = np.repeat(state[None, :], agent.sequence_length, axis=0).astype(np.float32)

    for _ in range(3):
        action = agent.select_action(sequence)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_sequence = np.vstack((sequence[1:], next_state))
        agent.remember(sequence, action, reward, next_sequence, terminated or truncated)
        sequence = next_sequence
        if terminated or truncated:
            break

    metrics = agent.train(batch_size=2)

    assert len(agent.replay_buffer) >= 2
    assert {"actor_loss", "critic1_loss", "critic2_loss"} <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())


def test_gru_curriculum_trainer_executes_a_real_episode(tmp_path):
    curriculum = GRUCurriculumLearning(
        initial_difficulty=1.0,
        sequence_length=3,
        gru_units=8,
        hidden_size=16,
        batch_size=2,
    )
    trainer = GRUCurriculumTrainer(
        curriculum=curriculum,
        data=_market_data(),
        max_episodes=1,
        eval_frequency=2,
        save_path=tmp_path,
    )

    agent, history = trainer.train(window_size=3)

    assert agent.use_gru is True
    assert len(history["rewards"]) == 1
    assert (tmp_path / "final_model").is_file()


def test_gru_curriculum_difficulty_progresses_from_real_scores():
    curriculum = GRUCurriculumLearning(
        initial_difficulty=0.2,
        difficulty_increment=0.1,
        success_threshold=0.5,
        evaluation_window=2,
    )

    assert curriculum.update_difficulty(0.7) is False
    assert curriculum.update_difficulty(0.8) is True
    assert curriculum.difficulty == pytest.approx(0.3)
