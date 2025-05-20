"""
Tests unitaires pour le module de stress testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_trading.ml.backtesting.stress_testing import StressTester, ScenarioType, StressScenario, determine_data_frequency

@pytest.fixture
def sample_data():
    """Crée un jeu de données OHLCV pour les tests."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Assurer que high > open > close > low
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(2, 0.5, len(dates)))
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(2, 0.5, len(dates)))
    
    return data

def test_stress_tester_initialization():
    """Teste l'initialisation du StressTester."""
    tester = StressTester()
    assert tester.config is not None
    assert "scenarios" in tester.config
    assert "monte_carlo" in tester.config

def test_crash_scenario(sample_data):
    """Teste la génération d'un scénario de crash."""
    tester = StressTester()
    stressed_data = tester._generate_crash_scenario(sample_data)
    
    assert len(stressed_data) == len(sample_data)
    assert not stressed_data.equals(sample_data)
    assert all(col in stressed_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_volatility_spike_scenario(sample_data):
    """Teste la génération d'un scénario de pic de volatilité."""
    tester = StressTester()
    stressed_data = tester._generate_volatility_spike_scenario(sample_data)
    
    assert len(stressed_data) == len(sample_data)
    assert not stressed_data.equals(sample_data)
    
    # Vérifier l'augmentation de la volatilité
    original_vol = sample_data['close'].pct_change().std()
    stressed_vol = stressed_data['close'].pct_change().std()
    assert stressed_vol > original_vol

def test_liquidity_crisis_scenario(sample_data):
    """Teste la génération d'un scénario de crise de liquidité."""
    tester = StressTester()
    stressed_data = tester._generate_liquidity_crisis_scenario(sample_data)
    
    assert len(stressed_data) == len(sample_data)
    assert not stressed_data.equals(sample_data)
    
    # Vérifier la baisse du volume
    assert stressed_data['volume'].mean() < sample_data['volume'].mean()

def test_flash_crash_scenario(sample_data):
    """Teste la génération d'un scénario de crash éclair."""
    tester = StressTester()
    stressed_data = tester._generate_flash_crash_scenario(sample_data)
    
    assert len(stressed_data) == len(sample_data)
    assert not stressed_data.equals(sample_data)
    
    # Vérifier la présence d'une chute brutale suivie d'une récupération
    returns = stressed_data['close'].pct_change().dropna()  # Ignorer la première valeur qui est toujours NaN
    assert min(returns) < -0.1  # Au moins une chute de 10%

def test_custom_scenario(sample_data):
    """Teste la génération d'un scénario personnalisé."""
    tester = StressTester()
    custom_params = {
        'price_shock': -0.2,
        'volatility_multiplier': 2.0,
        'volume_multiplier': 0.5,
        'duration_days': 5
    }
    stressed_data = tester._generate_custom_scenario(sample_data, custom_params)
    
    assert len(stressed_data) == len(sample_data)
    assert not stressed_data.equals(sample_data)
    assert (stressed_data['close'] < sample_data['close']).any()

def test_stress_scenario_class():
    """Teste la classe StressScenario."""
    start_date = datetime.now()
    scenario = StressScenario(
        name="Test Scenario",
        price_shock=-0.2,
        volatility_multiplier=2.0,
        volume_multiplier=0.5,
        duration=timedelta(days=5),
        start_date=start_date
    )
    
    assert scenario.name == "Test Scenario"
    assert scenario.price_shock == -0.2
    assert scenario.volatility_multiplier == 2.0
    assert scenario.volume_multiplier == 0.5
    assert scenario.duration == timedelta(days=5)
    assert scenario.start_date == start_date

def test_stress_scenario_apply(sample_data):
    """Teste l'application d'un scénario de stress aux données."""
    start_date = sample_data.index[50]
    scenario = StressScenario(
        name="Test Scenario",
        price_shock=-0.2,
        volatility_multiplier=2.0,
        volume_multiplier=0.5,
        duration=timedelta(days=5),
        start_date=start_date
    )
    
    market_data = {'TEST': sample_data}
    stressed_data = scenario.apply_to_data(market_data)
    
    assert 'TEST' in stressed_data
    assert len(stressed_data['TEST']) == len(sample_data)
    assert not stressed_data['TEST'].equals(sample_data)

def test_determine_data_frequency(sample_data):
    """Teste la détermination de la fréquence des données."""
    freq = determine_data_frequency(sample_data)
    assert freq == "1d"  # Les données de test sont journalières

def test_empty_data():
    """Teste le comportement avec des données vides."""
    tester = StressTester()
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        tester.generate_stress_scenario(empty_data, ScenarioType.CRASH)

def test_invalid_scenario_type(sample_data):
    """Teste le comportement avec un type de scénario invalide."""
    tester = StressTester()
    with pytest.raises(ValueError):
        tester.generate_stress_scenario(sample_data, "INVALID_TYPE") 