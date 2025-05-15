"""
Tests d'intégration pour le module de gestion avancée du risque.

Ce script démontre le fonctionnement du gestionnaire de risque avancé
avec des exemples de scénarios réels.
"""

import numpy as np

from ai_trading.execution.risk_manager import (
    RiskConfig,
    RiskManager,
    StopLossConfig,
    StopType,
)


def simulate_trade_with_risk_management():
    """
    Simule un scénario de trading complet avec la gestion du risque.
    """
    print("\n=== Simulation de trading avec gestion du risque ===")

    # Configuration du gestionnaire de risque
    risk_config = RiskConfig(
        max_position_size=5.0,  # Maximum 5% du capital par position
        max_risk_per_trade=1.0,  # Maximum 1% de risque par trade
        max_drawdown=15.0,  # Drawdown maximal toléré de 15%
    )

    risk_manager = RiskManager(risk_config)

    # Simulation d'un capital de départ
    initial_capital = 100000.0
    current_capital = initial_capital
    print(f"Capital initial: ${initial_capital:,.2f}")

    # Mise à jour des métriques de marché initiales
    # Volatilité faible, rendements positifs
    returns = np.random.normal(0.002, 0.01, 30).tolist()  # 30 jours de rendements
    volatility = 0.02  # Volatilité quotidienne de 2%

    metrics = risk_manager.update_portfolio_risk(current_capital, returns, volatility)
    print(f"Exposition initiale au marché: {metrics['market_exposure']:.2f}")
    print(f"Niveau de risque: {metrics['risk_level']}")

    # Calcul de la taille de position pour un premier trade
    entry_price = 50000.0  # Prix d'entrée (ex: BTC/USD)
    stop_price = 48500.0  # Prix du stop loss
    stop_distance = entry_price - stop_price

    position_sizing = risk_manager.calculate_optimal_position_size(
        capital=current_capital, stop_distance=stop_distance, side="buy"
    )

    position_size = position_sizing["position_size"]
    risk_amount = position_sizing["risk_amount"]

    print(f"\n=== Trade #1 ===")
    print(f"Prix d'entrée: ${entry_price:,.2f}")
    print(f"Prix du stop: ${stop_price:,.2f}")
    print(f"Distance du stop: ${stop_distance:,.2f}")
    print(
        f"Taille de la position: {position_size:.6f} BTC (${position_size * entry_price:,.2f})"
    )
    print(f"Risque monétaire: ${risk_amount:,.2f}")

    # Initialisation d'une position avec un stop trailing
    position_id = "BTC-1"
    stop_config = StopLossConfig(
        type=StopType.TRAILING,
        value=3.0,  # 3% de trailing stop
        is_percent=True,
        trailing=True,
    )

    risk_manager.initialize_position(position_id, entry_price, "buy", stop_config)
    print(f"Position initialisée avec stop trailing de 3%")

    # Simulation d'évolution des prix
    price_moves = [50500, 51200, 51800, 52500, 53000, 52800, 52400, 51500]

    for i, current_price in enumerate(price_moves):
        # Mettre à jour le stop loss
        position_update = risk_manager.update_position(position_id, current_price)

        print(
            f"\nJour {i+1}: Prix = ${current_price:,.2f}, "
            f"Stop = ${position_update['stop_level']:,.2f}"
        )

        # Vérifier si le stop est déclenché
        if position_update["stop_triggered"]:
            print(
                f"Stop déclenché! Clôture de la position à ${position_update['stop_level']:,.2f}"
            )

            # Calculer le P&L
            exit_price = position_update["stop_level"]
            pnl = (exit_price - entry_price) * position_size
            current_capital += pnl

            print(f"P&L: ${pnl:,.2f}")
            print(f"Capital après le trade: ${current_capital:,.2f}")

            # Fermer la position
            risk_manager.close_position(position_id)
            break
    else:
        # Si on sort de la boucle sans déclencher le stop
        final_price = price_moves[-1]
        pnl = (final_price - entry_price) * position_size
        current_capital += pnl

        print(f"\nClôture manuelle de la position à ${final_price:,.2f}")
        print(f"P&L: ${pnl:,.2f}")
        print(f"Capital après le trade: ${current_capital:,.2f}")

        # Fermer la position
        risk_manager.close_position(position_id)

    # Simulation d'une période de drawdown
    print("\n=== Simulation de période de drawdown ===")
    drawdown_capital = current_capital * 0.9  # 10% de perte

    drawdown_metrics = risk_manager.update_portfolio_risk(
        equity=drawdown_capital,
        returns=np.random.normal(-0.005, 0.015, 30).tolist(),  # Rendements négatifs
        volatility=0.04,  # Volatilité plus élevée
    )

    print(f"Capital en drawdown: ${drawdown_capital:,.2f}")
    print(f"Drawdown actuel: {drawdown_metrics['drawdown']['current_drawdown']:.2%}")
    print(f"Exposition ajustée: {drawdown_metrics['market_exposure']:.2f}")
    print(
        f"Facteur de mise à l'échelle des positions: "
        f"{drawdown_metrics['drawdown']['position_scale_factor']:.2f}"
    )

    # Calcul de nouvelle taille de position pendant le drawdown
    new_position_sizing = risk_manager.calculate_optimal_position_size(
        capital=drawdown_capital, stop_distance=1500, side="buy"  # Distance du stop
    )

    print(
        f"Nouvelle taille de position (ajustée): {new_position_sizing['position_size']:.6f} BTC"
    )
    print(
        f"Nouvelle taille de position (montant): "
        f"${new_position_sizing['position_size'] * 50000:,.2f}"
    )

    # Récupération du drawdown
    recovery_capital = drawdown_capital * 1.12  # 12% de gain après le drawdown

    recovery_metrics = risk_manager.update_portfolio_risk(
        equity=recovery_capital,
        returns=np.random.normal(0.003, 0.01, 30).tolist(),  # Rendements positifs
        volatility=0.025,  # Volatilité normale
    )

    print("\n=== Simulation de récupération ===")
    print(f"Capital après récupération: ${recovery_capital:,.2f}")
    print(f"Drawdown actuel: {recovery_metrics['drawdown']['current_drawdown']:.2%}")
    print(f"Exposition ajustée: {recovery_metrics['market_exposure']:.2f}")

    # Résultat final
    final_capital = recovery_capital
    total_return = (final_capital / initial_capital - 1) * 100

    print(f"\n=== Résultat final ===")
    print(f"Capital initial: ${initial_capital:,.2f}")
    print(f"Capital final: ${final_capital:,.2f}")
    print(f"Performance: {total_return:.2f}%")
    print(
        f"Drawdown maximal observé: {recovery_metrics['drawdown']['max_drawdown_seen']:.2%}"
    )


def main():
    """Exécute les tests d'intégration."""
    simulate_trade_with_risk_management()


if __name__ == "__main__":
    main()
