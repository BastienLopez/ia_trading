import numpy as np
import pandas as pd

from ai_trading.rl.multi_period_trainer import MultiPeriodTrainer


def _data(rows=48):
    close = 100 + np.cumsum(np.linspace(-0.2, 0.4, rows))
    return pd.DataFrame({"open": close - .2, "high": close + .5, "low": close - .5, "close": close, "volume": np.full(rows, 2_000.)}, index=pd.date_range("2024-01-01", periods=rows, freq="h"))


def test_multi_period_training_is_chronological_real_and_gpu(tmp_path):
    trainer = MultiPeriodTrainer("BTC", 2, [60, 240], use_gru=True, sequence_length=3, gru_units=8, batch_size=2, epochs_per_period=1, episodes_per_epoch=1, max_steps=3, save_dir=tmp_path, device="cuda")
    metrics = trainer.train_multi_period({60: _data(), 240: _data()})

    assert list(metrics) == [240, 60]
    assert all(np.isfinite(result["train_reward"]) and np.isfinite(result["validation_reward"]) for result in metrics.values())
    assert all((tmp_path / f"BTC_{period}min.pt").is_file() for period in metrics)


def test_multi_period_split_never_mixes_future_rows(tmp_path):
    trainer = MultiPeriodTrainer("BTC", 2, [60], save_dir=tmp_path)
    train, _, validation, _ = trainer.prepare_datasets(_data())
    assert train.index.max() < validation.index.min()
