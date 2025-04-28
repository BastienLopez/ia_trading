#!/usr/bin/env python
"""
Exemple d'utilisation du système de trading multi-actifs.
"""

import logging
from datetime import datetime, timedelta

from ai_trading.rl.multi_asset_trading import MultiAssetTradingSystem

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_multi_asset_trading_example():
    """
    Exemple d'utilisation du système de trading multi-actifs.
    """
    logger.info("Démarrage de l'exemple de trading multi-actifs...")

    # Initialiser le système de trading
    system = MultiAssetTradingSystem(
        crypto_assets=["BTC", "ETH"],
        traditional_assets=["XAU/USD", "AAPL", "NVDA"],
        initial_balance=10000.0,
        risk_per_trade=0.02,
        max_position_size=0.2,
    )

    # Définir la période
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )  # 1 an de données

    # Collecter les données de marché
    logger.info(f"Collecte des données de marché du {start_date} au {end_date}...")
    market_data = system.collect_market_data(start_date, end_date)

    # Entraîner les systèmes de trading
    logger.info("Entraînement des systèmes de trading...")
    system.train(market_data, epochs=10)

    # Simuler le trading
    logger.info("Simulation du trading...")
    for i in range(10):  # Simuler 10 jours de trading
        # Obtenir les données du jour
        daily_data = {
            asset: data.iloc[i : i + 1] for asset, data in market_data.items()
        }

        # Prédire les actions
        actions = system.predict_actions(daily_data)

        # Mettre à jour les positions
        system.update_positions(actions)

        # Calculer et afficher les métriques du portefeuille
        metrics = system.calculate_portfolio_metrics()
        logger.info(f"Jour {i+1}:")
        logger.info(f"Valeur totale: {metrics['total_value']:.2f}")
        logger.info(f"Rendement: {metrics['return']*100:.2f}%")
        logger.info("Positions:")
        for asset, position in metrics["positions"].items():
            logger.info(f"  {asset}: {position:.4f}")

    logger.info("Exemple terminé!")


if __name__ == "__main__":
    run_multi_asset_trading_example()
