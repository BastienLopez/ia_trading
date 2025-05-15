"""
Tests d'intégration pour le module de routage intelligent.

Ce script teste le fonctionnement du routeur intelligent des ordres
qui sélectionne les meilleures venues d'exécution.
"""

from ai_trading.execution.smart_routing import (
    DummyLiquidityProvider,
    ExchangePriority,
    ExchangeVenue,
    SmartRouter,
)


def test_smart_routing_integration():
    """Test d'intégration du routage intelligent."""
    # Initialiser le routeur
    provider = DummyLiquidityProvider()
    router = SmartRouter(provider)

    print("\n=== Test du routage intelligent ===")

    # Tester différentes priorités de routage
    test_priorities = [
        ExchangePriority.FEES,
        ExchangePriority.LIQUIDITY,
        ExchangePriority.SPEED,
        ExchangePriority.SMART,
    ]

    for priority in test_priorities:
        # Router un ordre
        selected_venue = router.route_order(
            symbol="BTC/USD",
            side="buy",
            quantity=1.0,
            order_type="market",
            priority=priority,
        )

        print(
            f"Priorité: {priority.value} -> Venue sélectionnée: {selected_venue.value}"
        )

        # Vérifier que la venue est valide
        assert isinstance(selected_venue, ExchangeVenue)

    # Tester le smart split des ordres
    print("\n=== Test du smart split des ordres ===")
    allocation = router.smart_split_order(
        symbol="BTC/USD", side="buy", quantity=2.5, order_type="market", max_venues=3
    )

    print(f"Smart split pour 2.5 BTC/USD:")
    for venue, quantity in allocation.items():
        print(f"  - {venue.value}: {quantity:.4f} BTC")

    # Vérifier que l'allocation est correcte
    total_allocated = sum(allocation.values())
    assert abs(total_allocated - 2.5) < 0.0001

    # Tester l'estimation des coûts d'exécution
    print("\n=== Test d'estimation des coûts d'exécution ===")
    for venue in [ExchangeVenue.BINANCE, ExchangeVenue.COINBASE, ExchangeVenue.FTX]:
        costs = router.estimate_execution_cost(
            symbol="BTC/USD", side="buy", quantity=1.0, venue=venue, order_type="market"
        )

        print(f"Coûts d'exécution sur {venue.value}:")
        print(f"  - Prix estimé: ${costs['estimated_price']:,.2f}")
        print(f"  - Frais: {costs['fee_rate']}% (${costs['fee_amount']:,.2f})")
        print(
            f"  - Slippage: {costs['slippage_pct']}% (${costs['slippage_amount']:,.2f})"
        )
        print(
            f"  - Impact marché: {costs['impact_pct']}% (${costs['impact_amount']:,.2f})"
        )
        print(
            f"  - Coût total: ${costs['total_cost']:,.2f} ({costs['total_cost_pct']:.4f}%)"
        )

    print("\n=== Tests complétés avec succès ===")


def main():
    """Fonction principale pour les tests."""
    test_smart_routing_integration()


if __name__ == "__main__":
    main()
