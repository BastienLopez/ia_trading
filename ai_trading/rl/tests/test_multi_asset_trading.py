import numpy as np
import pandas as pd
import pytest

from ai_trading.rl.multi_asset_trading import MultiAssetTradingSystem


@pytest.fixture
def trading_system():
    return MultiAssetTradingSystem(
        crypto_assets=["BTC", "ETH"],
        traditional_assets=["AAPL", "MSFT"],
        initial_balance=10000.0,
        risk_per_trade=0.02,
        max_position_size=0.2,
    )


@pytest.fixture
def sample_market_data():
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    data = {}
    for asset in ["BTC", "ETH", "AAPL", "MSFT"]:
        prices = np.random.normal(100, 10, len(dates))
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(1, 5, len(dates)),
                "low": prices - np.random.uniform(1, 5, len(dates)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, len(dates)),
            },
            index=dates,
        )
        data[asset] = df
    return data


def test_initialization(trading_system):
    assert len(trading_system.assets) == 4
    assert trading_system.initial_balance == 10000.0
    assert trading_system.risk_per_trade == 0.02
    assert trading_system.max_position_size == 0.2


def test_calculate_market_trend(trading_system, sample_market_data):
    trend = trading_system.calculate_market_trend(sample_market_data)
    assert isinstance(trend, float)
    assert -1 <= trend <= 1


def test_calculate_market_correlation(trading_system, sample_market_data):
    correlation = trading_system.calculate_market_correlation(sample_market_data)
    assert isinstance(correlation, float)
    assert -1 <= correlation <= 1


def test_predict_actions(trading_system, sample_market_data):
    actions = trading_system.predict_actions(sample_market_data)
    assert isinstance(actions, dict)
    assert len(actions) == len(trading_system.assets)
    for action in actions.values():
        assert isinstance(action, float)
        assert -1 <= action <= 1


def test_train(trading_system, sample_market_data):
    trading_system.train(sample_market_data, epochs=1)
    assert len(trading_system.trading_systems) == len(trading_system.assets)


def test_update_positions(trading_system):
    actions = {"BTC": 0.5, "ETH": -0.3, "AAPL": 0.1, "MSFT": 0.0}
    trading_system.update_positions(actions)
    assert len(trading_system.positions) == len(trading_system.assets)
    for position in trading_system.positions.values():
        assert isinstance(position, float)


def test_adjust_positions_for_correlation(trading_system, sample_market_data):
    # Créer des actions de test
    test_actions = {
        "BTC": 0.8,  # Position forte en BTC
        "ETH": 0.7,  # Position forte en ETH (corrélée avec BTC)
        "AAPL": -0.4,  # Position courte en AAPL
        "MSFT": -0.3,  # Position courte en MSFT (corrélée avec AAPL)
    }

    # Ajuster les positions
    adjusted_actions = trading_system.adjust_positions_for_correlation(
        test_actions, sample_market_data
    )

    # Vérifier que les positions ont été ajustées
    assert isinstance(adjusted_actions, dict)
    assert len(adjusted_actions) == len(test_actions)

    # Vérifier que les positions sont dans les limites
    for action in adjusted_actions.values():
        assert isinstance(action, float)
        assert -0.5 <= action <= 0.5  # Limites définies dans la méthode

    # Vérifier que les positions corrélées ont été réduites
    assert abs(adjusted_actions["BTC"]) < abs(test_actions["BTC"]) or abs(
        adjusted_actions["ETH"]
    ) < abs(test_actions["ETH"])
