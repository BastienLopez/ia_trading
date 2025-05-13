"""
Tests unitaires pour le module de routage intelligent.

Ce module teste les fonctionnalités du routage intelligent des ordres
entre différentes venues d'exécution.
"""

import pytest
import unittest.mock as mock
from ai_trading.execution.smart_routing import (
    ExchangeVenue,
    ExchangePriority,
    ExchangeInfo,
    OrderRoutingStrategy,
    LowestFeeStrategy,
    BestLiquidityStrategy,
    FastestExecutionStrategy,
    SmartRoutingStrategy,
    SmartRouter,
    DummyLiquidityProvider
)


def test_exchange_venue_enum():
    """Vérifie que les venues d'exécution sont correctement définies."""
    assert ExchangeVenue.BINANCE.value == "binance"
    assert ExchangeVenue.COINBASE.value == "coinbase"
    assert ExchangeVenue.KRAKEN.value == "kraken"
    assert ExchangeVenue.FTX.value == "ftx"


def test_exchange_priority_enum():
    """Vérifie que les priorités d'exécution sont correctement définies."""
    assert ExchangePriority.FEES.value == "fees"
    assert ExchangePriority.LIQUIDITY.value == "liquidity"
    assert ExchangePriority.SPEED.value == "speed"
    assert ExchangePriority.SMART.value == "smart"


def test_exchange_info():
    """Teste la classe ExchangeInfo."""
    # Créer une instance d'ExchangeInfo
    info = ExchangeInfo(
        venue=ExchangeVenue.BINANCE,
        taker_fee=0.1,
        maker_fee=0.05,
        avg_latency_ms=50.0,
        min_order_size={"BTC/USD": 0.0001},
        max_order_size={"BTC/USD": 100.0},
        supports_market_orders=True,
        supports_limit_orders=True,
        supports_stop_orders=True
    )
    
    # Vérifier les attributs
    assert info.venue == ExchangeVenue.BINANCE
    assert info.taker_fee == 0.1
    assert info.maker_fee == 0.05
    assert info.avg_latency_ms == 50.0
    
    # Tester les méthodes
    assert info.get_fee_for_order_type(is_taker=True) == 0.1
    assert info.get_fee_for_order_type(is_taker=False) == 0.05
    
    # Vérifier la méthode can_execute_quantity
    assert info.can_execute_quantity("BTC/USD", 0.5) == True
    assert info.can_execute_quantity("BTC/USD", 0.00001) == False
    assert info.can_execute_quantity("BTC/USD", 200.0) == False


def test_lowest_fee_strategy():
    """Teste la stratégie de routage basée sur les frais les plus bas."""
    strategy = LowestFeeStrategy()
    
    # Créer des instances d'ExchangeInfo
    binance = ExchangeInfo(
        venue=ExchangeVenue.BINANCE,
        taker_fee=0.1,
        maker_fee=0.08,
        avg_latency_ms=50.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    coinbase = ExchangeInfo(
        venue=ExchangeVenue.COINBASE,
        taker_fee=0.3,
        maker_fee=0.15,
        avg_latency_ms=70.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    kraken = ExchangeInfo(
        venue=ExchangeVenue.KRAKEN,
        taker_fee=0.2,
        maker_fee=0.1,
        avg_latency_ms=100.0,
        min_order_size={"BTC/USD": 0.0002}
    )
    
    ftx = ExchangeInfo(
        venue=ExchangeVenue.FTX,
        taker_fee=0.07,
        maker_fee=0.02,
        avg_latency_ms=40.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    available_venues = [binance, coinbase, kraken, ftx]
    
    # Test pour un ordre market (taker)
    selected_venue = strategy.select_venue(
        symbol="BTC/USD",
        order_type="market",
        quantity=0.1,
        available_venues=available_venues
    )
    
    # FTX devrait être sélectionné car il a les frais taker les plus bas
    assert selected_venue == ExchangeVenue.FTX
    
    # Test pour un ordre limit (maker)
    selected_venue = strategy.select_venue(
        symbol="BTC/USD",
        order_type="limit",
        quantity=0.1,
        available_venues=available_venues
    )
    
    # FTX devrait être sélectionné car il a les frais maker les plus bas
    assert selected_venue == ExchangeVenue.FTX


def test_fastest_execution_strategy():
    """Teste la stratégie de routage basée sur la vitesse d'exécution."""
    strategy = FastestExecutionStrategy()
    
    # Créer des instances d'ExchangeInfo
    binance = ExchangeInfo(
        venue=ExchangeVenue.BINANCE,
        taker_fee=0.1,
        maker_fee=0.08,
        avg_latency_ms=50.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    coinbase = ExchangeInfo(
        venue=ExchangeVenue.COINBASE,
        taker_fee=0.3,
        maker_fee=0.15,
        avg_latency_ms=70.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    kraken = ExchangeInfo(
        venue=ExchangeVenue.KRAKEN,
        taker_fee=0.2,
        maker_fee=0.1,
        avg_latency_ms=100.0,
        min_order_size={"BTC/USD": 0.0002}
    )
    
    ftx = ExchangeInfo(
        venue=ExchangeVenue.FTX,
        taker_fee=0.07,
        maker_fee=0.02,
        avg_latency_ms=40.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    available_venues = [binance, coinbase, kraken, ftx]
    
    # Test pour un ordre quelconque
    selected_venue = strategy.select_venue(
        symbol="BTC/USD",
        order_type="market",
        quantity=0.1,
        available_venues=available_venues
    )
    
    # FTX devrait être sélectionné car il a la latence la plus basse
    assert selected_venue == ExchangeVenue.FTX


def test_dummy_liquidity_provider():
    """Teste le fournisseur de données de liquidité factice."""
    provider = DummyLiquidityProvider()
    
    # Obtenir des données pour différentes venues
    binance_data = provider.get_liquidity_data("BTC/USD", ExchangeVenue.BINANCE)
    coinbase_data = provider.get_liquidity_data("BTC/USD", ExchangeVenue.COINBASE)
    
    # Vérifier que les données sont au format attendu
    assert "bid_depth" in binance_data
    assert "ask_depth" in binance_data
    assert "spread" in binance_data
    assert "depth_factor" in binance_data
    
    # Les données de Binance devraient avoir une meilleure liquidité que Coinbase
    assert binance_data["depth_factor"] > coinbase_data["depth_factor"]


def test_best_liquidity_strategy():
    """Teste la stratégie de routage basée sur la meilleure liquidité."""
    # Créer un mock du fournisseur de liquidité
    mock_provider = mock.MagicMock()
    
    # Configurer le mock pour retourner des données de liquidité différentes par venue
    def get_liquidity_side_effect(symbol, venue):
        if venue == ExchangeVenue.BINANCE:
            return {"bid_depth": 100, "ask_depth": 120, "spread": 0.02, "depth_factor": 0.8}
        elif venue == ExchangeVenue.COINBASE:
            return {"bid_depth": 80, "ask_depth": 90, "spread": 0.03, "depth_factor": 0.7}
        elif venue == ExchangeVenue.KRAKEN:
            return {"bid_depth": 60, "ask_depth": 70, "spread": 0.04, "depth_factor": 0.6}
        elif venue == ExchangeVenue.FTX:
            return {"bid_depth": 110, "ask_depth": 130, "spread": 0.01, "depth_factor": 0.9}
        return {"bid_depth": 50, "ask_depth": 50, "spread": 0.05, "depth_factor": 0.5}
    
    mock_provider.get_liquidity_data.side_effect = get_liquidity_side_effect
    
    # Créer la stratégie avec le mock
    strategy = BestLiquidityStrategy(mock_provider)
    
    # Patch la méthode d'estimation du slippage pour retourner des valeurs prévisibles
    def estimate_slippage_side_effect(liquidity, quantity, is_buy):
        # Plus le depth_factor est élevé, plus le slippage sera faible
        return 0.2 - liquidity.get("depth_factor", 0) * 0.2
    
    strategy._estimate_slippage = estimate_slippage_side_effect
    
    # Créer des instances d'ExchangeInfo
    binance = ExchangeInfo(
        venue=ExchangeVenue.BINANCE,
        taker_fee=0.1,
        maker_fee=0.08,
        avg_latency_ms=50.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    coinbase = ExchangeInfo(
        venue=ExchangeVenue.COINBASE,
        taker_fee=0.3,
        maker_fee=0.15,
        avg_latency_ms=70.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    kraken = ExchangeInfo(
        venue=ExchangeVenue.KRAKEN,
        taker_fee=0.2,
        maker_fee=0.1,
        avg_latency_ms=100.0,
        min_order_size={"BTC/USD": 0.0002}
    )
    
    ftx = ExchangeInfo(
        venue=ExchangeVenue.FTX,
        taker_fee=0.07,
        maker_fee=0.02,
        avg_latency_ms=40.0,
        min_order_size={"BTC/USD": 0.0001}
    )
    
    available_venues = [binance, coinbase, kraken, ftx]
    
    # Test pour un ordre quelconque
    selected_venue = strategy.select_venue(
        symbol="BTC/USD",
        order_type="market",
        quantity=0.1,
        available_venues=available_venues
    )
    
    # FTX devrait être sélectionné car il a le meilleur depth_factor (0.9)
    assert selected_venue == ExchangeVenue.FTX


def test_smart_router():
    """Teste le routeur intelligent."""
    # Créer un mock du fournisseur de liquidité
    mock_provider = mock.MagicMock()
    
    # Configurer le mock pour retourner des données de liquidité différentes par venue
    def get_liquidity_side_effect(symbol, venue):
        if venue == ExchangeVenue.BINANCE:
            return {"bid_depth": 100, "ask_depth": 120, "spread": 0.02, "depth_factor": 0.8}
        elif venue == ExchangeVenue.COINBASE:
            return {"bid_depth": 80, "ask_depth": 90, "spread": 0.03, "depth_factor": 0.7}
        elif venue == ExchangeVenue.KRAKEN:
            return {"bid_depth": 60, "ask_depth": 70, "spread": 0.04, "depth_factor": 0.6}
        elif venue == ExchangeVenue.FTX:
            return {"bid_depth": 110, "ask_depth": 130, "spread": 0.01, "depth_factor": 0.9}
        return {"bid_depth": 50, "ask_depth": 50, "spread": 0.05, "depth_factor": 0.5}
    
    mock_provider.get_liquidity_data.side_effect = get_liquidity_side_effect
    
    # Créer le routeur avec le mock
    router = SmartRouter(mock_provider)
    
    # Test du routage avec priorité aux frais
    selected_venue = router.route_order(
        symbol="BTC/USD",
        side="buy",
        quantity=0.1,
        order_type="market",
        priority=ExchangePriority.FEES
    )
    
    # FTX devrait être sélectionné car il a les frais les plus bas
    assert selected_venue == ExchangeVenue.FTX
    
    # Test du routage avec priorité à la vitesse
    selected_venue = router.route_order(
        symbol="BTC/USD",
        side="buy",
        quantity=0.1,
        order_type="market",
        priority=ExchangePriority.SPEED
    )
    
    # FTX devrait être sélectionné car il a la latence la plus basse
    assert selected_venue == ExchangeVenue.FTX
    
    # Test du smart split
    allocation = router.smart_split_order(
        symbol="BTC/USD",
        side="buy",
        quantity=1.0,
        order_type="market",
        max_venues=3
    )
    
    # Vérifier que l'allocation est faite sur les venues les plus adaptées
    assert ExchangeVenue.FTX in allocation
    assert ExchangeVenue.BINANCE in allocation
    
    # Vérifier que la somme des quantités est égale à la quantité totale
    total_quantity = sum(allocation.values())
    assert abs(total_quantity - 1.0) < 0.00001
    
    # Test d'estimation des coûts d'exécution
    costs = router.estimate_execution_cost(
        symbol="BTC/USD",
        side="buy",
        quantity=1.0,
        venue=ExchangeVenue.BINANCE,
        order_type="market"
    )
    
    # Vérifier que l'estimation contient les champs attendus
    assert "fee_rate" in costs
    assert "slippage_pct" in costs
    assert "impact_pct" in costs
    assert "total_cost" in costs
    assert "total_cost_pct" in costs 