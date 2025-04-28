from datetime import datetime
from unittest.mock import Mock

import pytest

from ai_trading.rl_agent import RiskManager, TradingEnvironment


class MockPortfolio:
    """Classe utilitaire pour simuler un portefeuille"""

    def __init__(self, exposure):
        self._exposure = exposure

    def current_exposure(self):
        return self._exposure


@pytest.fixture
def risk_manager():
    return RiskManager(max_exposure=0.2)


@pytest.fixture
def trading_env():
    mock_data = Mock()
    mock_data.historical_data = [{"close": 100}, {"close": 105}]
    return TradingEnvironment(
        initial_balance=10000, data_source=mock_data, risk_params={"max_exposure": 0.2}
    )


def test_risk_manager_initialization():
    rm = RiskManager(max_exposure=0.15)
    assert rm.max_exposure == 0.15
    assert rm.max_leverage == 2.0
    assert len(rm.exposure_history) == 0


def test_exposure_under_limit_buy(risk_manager):
    portfolio = MockPortfolio(0.1)
    action = ("buy", 0.05)
    adjusted = risk_manager.check_exposure(portfolio, action)
    assert adjusted == action
    assert len(risk_manager.exposure_history) == 1


def test_exposure_over_limit_buy(risk_manager):
    portfolio = MockPortfolio(0.18)
    action = ("buy", 0.05)
    adjusted = risk_manager.check_exposure(portfolio, action)
    assert adjusted == ("buy", 0.02)
    assert risk_manager.exposure_history[-1]["proposed"] == action


def test_max_exposure_block(risk_manager):
    portfolio = MockPortfolio(0.2)
    action = ("buy", 0.01)
    adjusted = risk_manager.check_exposure(portfolio, action)
    assert adjusted == ("hold", 0)


def test_short_exposure_adjustment(risk_manager):
    portfolio = MockPortfolio(-0.19)
    action = ("sell", 0.05)
    adjusted = risk_manager.check_exposure(portfolio, action)
    assert adjusted == ("sell", 0.01)


def test_penalty_application(trading_env):
    trading_env.portfolio = MockPortfolio(0.19)
    trading_env.reward = 0
    trading_env.step(("buy", 0.5))
    assert trading_env.reward == -0.1


def test_violation_logging(trading_env):
    trading_env.portfolio = MockPortfolio(0.19)
    trading_env.step(("buy", 0.5))
    assert len(trading_env.risk_manager.exposure_history) == 1
    log_entry = trading_env.risk_manager.exposure_history[0]
    assert isinstance(log_entry["timestamp"], datetime)
    assert log_entry["current"] == 0.19
    assert log_entry["proposed"] == ("buy", 0.5)


def test_risk_params_override():
    mock_data = Mock()
    mock_data.historical_data = [{"close": 100}]
    env = TradingEnvironment(
        initial_balance=10000, data_source=mock_data, risk_params={"max_exposure": 0.15}
    )
    assert env.risk_manager.max_exposure == 0.15
