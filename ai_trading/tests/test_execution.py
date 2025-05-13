"""
Script de test pour vérifier le fonctionnement du module d'exécution adaptatif.
"""

from ai_trading.execution.adaptive_execution import AdaptiveExecutor, ExecutionMode


def test_execution_modes():
    """Test des différents modes d'exécution."""
    # Création de l'exécuteur
    executor = AdaptiveExecutor()

    # Test avec différents modes d'exécution
    test_modes = [
        ExecutionMode.PASSIVE,
        ExecutionMode.NORMAL,
        ExecutionMode.AGGRESSIVE,
        ExecutionMode.ADAPTIVE,
    ]

    for mode in test_modes:
        # Exécution d'un ordre d'achat
        result = executor.execute(symbol="BTC/USD", side="buy", quantity=1.0, mode=mode)

        assert result["symbol"] == "BTC/USD"
        assert result["side"] == "buy"
        assert result["quantity"] == 1.0
        assert result["status"] == "completed"

        # Si c'est le mode adaptatif, la stratégie peut varier
        if mode != ExecutionMode.ADAPTIVE:
            assert result["strategy"] == mode.value

        # Estimation de l'impact
        impact = executor.estimate_impact(
            symbol="BTC/USD", quantity=1.0, side="buy", mode=mode
        )

        assert impact > 0


def test_suggest_execution_mode():
    """Test de la suggestion de mode d'exécution."""
    # Création de l'exécuteur
    executor = AdaptiveExecutor()

    # Test avec différentes urgences
    urgency_levels = [0.1, 0.5, 0.9]

    for urgency in urgency_levels:
        suggested_mode = executor.suggest_execution_mode(
            symbol="BTC/USD", quantity=1.0, side="buy", urgency=urgency
        )

        assert isinstance(suggested_mode, ExecutionMode)


def main():
    """Fonction principale pour tester le module d'exécution."""
    print("=== Test du module d'exécution adaptative ===")

    test_execution_modes()

    # Test de la suggestion de mode d'exécution
    print("\nTest de suggestion de mode d'exécution:")

    test_suggest_execution_mode()

    print("\n=== Fin des tests ===")


if __name__ == "__main__":
    main()
