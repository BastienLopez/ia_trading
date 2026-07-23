import numpy as np
import pandas as pd
import pytest

from ai_trading.dashboard.data_loader import DataLoader


def test_generated_portfolio_history_is_finite_and_realistic(tmp_path):
    history = DataLoader(str(tmp_path))._generate_portfolio_history(days=180)

    assert history["valeur"].gt(0).all()
    assert np.isfinite(history["valeur"]).all()
    assert history["rendement_quotidien"].abs().max() <= 0.15
    assert history["valeur"].iloc[-1] < history["valeur"].iloc[0] * 3


def test_corrupt_portfolio_history_is_rejected_before_dashboard_rendering(tmp_path):
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "valeur": [10_000.0, 1.0e16],
            "rendement_quotidien": [0.0, 1.0e12],
        }
    ).to_csv(tmp_path / "main_history.csv", index=False)

    with pytest.raises(ValueError, match="variation quotidienne supérieure à 50 %"):
        DataLoader(str(tmp_path)).load_portfolio_history()
