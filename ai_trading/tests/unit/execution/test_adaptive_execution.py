"""
Tests unitaires pour le module d'exécution adaptative.

Ce module teste les différentes stratégies d'exécution et le fonctionnement
de l'AdaptiveExecutor qui sélectionne dynamiquement la meilleure stratégie.
"""

import pytest
import unittest.mock as mock
from ai_trading.execution.adaptive_execution import (
    ExecutionMode,
    ExecutionStrategy,
    PassiveExecutionStrategy,
    NormalExecutionStrategy,
    AggressiveExecutionStrategy,
    AdaptiveExecutionStrategy,
    AdaptiveExecutor
)


def test_execution_mode_enum():
    """Vérifie que les modes d'exécution sont correctement définis."""
    assert ExecutionMode.PASSIVE.value == "passive"
    assert ExecutionMode.NORMAL.value == "normal"
    assert ExecutionMode.AGGRESSIVE.value == "aggressive"
    assert ExecutionMode.ADAPTIVE.value == "adaptive"


def test_passive_execution_strategy():
    """Teste la stratégie d'exécution passive."""
    strategy = PassiveExecutionStrategy()
    
    # Test d'exécution d'ordre
    result = strategy.execute_order(symbol="BTC/USD", side="buy", quantity=1.0)
    
    assert result["symbol"] == "BTC/USD"
    assert result["side"] == "buy"
    assert result["quantity"] == 1.0
    assert result["strategy"] == "passive"
    assert result["status"] == "completed"
    
    # Pour un achat passif, le prix est généralement légèrement inférieur au prix du marché
    market_price = strategy._get_market_price("BTC/USD")
    assert result["executed_price"] < market_price
    
    # Test d'estimation d'impact
    impact = strategy.calculate_impact_estimate(symbol="BTC/USD", quantity=1.0, side="buy")
    assert impact > 0
    assert impact < 0.1  # L'impact devrait être faible pour une stratégie passive


def test_normal_execution_strategy():
    """Teste la stratégie d'exécution normale."""
    strategy = NormalExecutionStrategy()
    
    # Test d'exécution d'ordre
    result = strategy.execute_order(symbol="BTC/USD", side="sell", quantity=1.0)
    
    assert result["symbol"] == "BTC/USD"
    assert result["side"] == "sell"
    assert result["quantity"] == 1.0
    assert result["strategy"] == "normal"
    assert result["status"] == "completed"
    
    # Pour une vente normale, le prix est généralement légèrement supérieur au prix du marché
    market_price = strategy._get_market_price("BTC/USD")
    assert result["executed_price"] > market_price
    
    # Test d'estimation d'impact
    impact = strategy.calculate_impact_estimate(symbol="BTC/USD", quantity=1.0, side="sell")
    assert impact > 0


def test_aggressive_execution_strategy():
    """Teste la stratégie d'exécution agressive."""
    strategy = AggressiveExecutionStrategy()
    
    # Test d'exécution d'ordre
    result = strategy.execute_order(symbol="BTC/USD", side="buy", quantity=1.0)
    
    assert result["symbol"] == "BTC/USD"
    assert result["side"] == "buy"
    assert result["quantity"] == 1.0
    assert result["strategy"] == "aggressive"
    assert result["status"] == "completed"
    
    # Pour un achat agressif, le prix est généralement plus élevé que le prix du marché
    market_price = strategy._get_market_price("BTC/USD")
    assert result["executed_price"] > market_price
    
    # Test d'estimation d'impact
    impact = strategy.calculate_impact_estimate(symbol="BTC/USD", quantity=1.0, side="buy")
    assert impact > 0
    assert impact > strategy.calculate_impact_estimate(symbol="BTC/USD", quantity=0.5, side="buy")


def test_adaptive_execution_strategy():
    """Teste la stratégie d'exécution adaptative."""
    strategy = AdaptiveExecutionStrategy()
    
    # Mock de la méthode _analyze_market_conditions pour contrôler le comportement
    with mock.patch.object(strategy, '_analyze_market_conditions') as mock_analyze:
        # Test avec des conditions favorables à une exécution passive
        mock_analyze.return_value = {"volatility": 0.1, "liquidity": 0.8, "spread": 0.01}
        result_passive = strategy.execute_order(symbol="BTC/USD", side="buy", quantity=1.0)
        assert result_passive["strategy"] == "passive"
        
        # Test avec des conditions favorables à une exécution agressive
        mock_analyze.return_value = {"volatility": 0.8, "liquidity": 0.2, "spread": 0.1}
        result_aggressive = strategy.execute_order(symbol="BTC/USD", side="buy", quantity=1.0)
        assert result_aggressive["strategy"] == "aggressive"
        
        # Test avec des conditions normales
        mock_analyze.return_value = {"volatility": 0.5, "liquidity": 0.5, "spread": 0.05}
        result_normal = strategy.execute_order(symbol="BTC/USD", side="buy", quantity=1.0)
        assert result_normal["strategy"] == "normal"


def test_adaptive_executor():
    """Teste le gestionnaire d'exécution adaptatif."""
    executor = AdaptiveExecutor()
    
    # Test d'exécution avec différents modes
    result_passive = executor.execute(
        symbol="BTC/USD",
        side="buy",
        quantity=1.0,
        mode=ExecutionMode.PASSIVE
    )
    assert result_passive["strategy"] == "passive"
    
    result_normal = executor.execute(
        symbol="BTC/USD",
        side="buy",
        quantity=1.0,
        mode=ExecutionMode.NORMAL
    )
    assert result_normal["strategy"] == "normal"
    
    result_aggressive = executor.execute(
        symbol="BTC/USD",
        side="buy",
        quantity=1.0,
        mode=ExecutionMode.AGGRESSIVE
    )
    assert result_aggressive["strategy"] == "aggressive"
    
    # Test de la fonction d'estimation d'impact
    impact_passive = executor.estimate_impact(
        symbol="BTC/USD",
        quantity=1.0,
        side="buy",
        mode=ExecutionMode.PASSIVE
    )
    
    impact_aggressive = executor.estimate_impact(
        symbol="BTC/USD",
        quantity=1.0,
        side="buy",
        mode=ExecutionMode.AGGRESSIVE
    )
    
    # L'impact en mode agressif devrait être plus élevé qu'en mode passif
    assert impact_aggressive > impact_passive


def test_suggest_execution_mode():
    """Teste la suggestion de mode d'exécution."""
    executor = AdaptiveExecutor()
    
    # Mock des méthodes d'analyse pour contrôler le comportement
    with mock.patch.object(executor, 'analyze_market_volatility') as mock_volatility, \
         mock.patch.object(executor, 'analyze_market_liquidity') as mock_liquidity:
        
        # Cas favorable à une exécution passive
        mock_volatility.return_value = 0.1  # Faible volatilité
        mock_liquidity.return_value = 0.9   # Haute liquidité
        
        mode = executor.suggest_execution_mode(
            symbol="BTC/USD",
            quantity=1.0,
            side="buy",
            urgency=0.1  # Faible urgence
        )
        assert mode == ExecutionMode.PASSIVE
        
        # Cas favorable à une exécution agressive
        mock_volatility.return_value = 0.9  # Haute volatilité
        mock_liquidity.return_value = 0.1   # Faible liquidité
        
        mode = executor.suggest_execution_mode(
            symbol="BTC/USD",
            quantity=1.0,
            side="buy",
            urgency=0.9  # Haute urgence
        )
        assert mode == ExecutionMode.AGGRESSIVE
        
        # Cas intermédiaire
        mock_volatility.return_value = 0.5  # Volatilité moyenne
        mock_liquidity.return_value = 0.5   # Liquidité moyenne
        
        mode = executor.suggest_execution_mode(
            symbol="BTC/USD",
            quantity=1.0,
            side="buy",
            urgency=0.5  # Urgence moyenne
        )
        assert mode == ExecutionMode.NORMAL 