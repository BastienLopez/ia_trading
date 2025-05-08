"""
Exemple d'utilisation des stratégies d'arbitrage.

Ce script montre comment utiliser les différentes stratégies d'arbitrage
implémentées dans le système de trading.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ai_trading.strategies.arbitrage import (
    FundingRateArbitrageStrategy,
    SpatialArbitrageStrategy,
    StatisticalArbitrageStrategy,
    TriangularArbitrageStrategy,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_mock_exchange_data():
    """
    Génère des données factices d'échange pour tester l'arbitrage spatial.

    Returns:
        Dictionnaire simulant des données de marché sur différentes plateformes
    """
    # Prix de base pour les actifs
    base_prices = {
        "BTC/USDT": 40000,
        "ETH/USDT": 3000,
        "XRP/USDT": 0.5,
        "ADA/USDT": 1.2,
        "SOL/USDT": 100,
        "BNB/USDT": 400,
        "AVAX/USDT": 30,
        "MATIC/USDT": 1.0,
        "DOT/USDT": 20,
        "LINK/USDT": 15,
    }

    # Plateformes d'échange
    exchanges = ["binance", "coinbase", "kraken", "kucoin", "huobi"]

    # Ajouter une variation aléatoire pour chaque plateforme
    np.random.seed(42)  # Pour la reproductibilité
    market_data = {}

    for symbol, base_price in base_prices.items():
        market_data[symbol] = {}

        for exchange in exchanges:
            # Ajouter une différence de prix (jusqu'à ±0.5%)
            price_diff = np.random.uniform(-0.005, 0.005)

            # Pour certaines paires, ajouter des écarts plus importants pour créer des opportunités
            if exchange == "binance" and symbol in ["BTC/USDT", "ETH/USDT"]:
                price_diff -= 0.01  # Prix 1% plus bas sur Binance
            elif exchange == "coinbase" and symbol in ["BTC/USDT", "SOL/USDT"]:
                price_diff += 0.015  # Prix 1.5% plus haut sur Coinbase

            # Calculer les prix ask et bid
            adjusted_price = base_price * (1 + price_diff)
            spread = base_price * 0.001  # 0.1% de spread

            market_data[symbol][exchange] = {
                "ask": adjusted_price + spread / 2,
                "bid": adjusted_price - spread / 2,
                "last": adjusted_price,
                "volume": np.random.uniform(10, 1000) * base_price,
                "timestamp": datetime.now().timestamp(),
            }

    return market_data


def generate_mock_triangle_data():
    """
    Génère des données factices pour tester l'arbitrage triangulaire.

    Returns:
        Dictionnaire simulant des données de marché pour des paires d'actifs
    """
    # Créer des données de base
    data = {
        "BTC/USDT": {"ask": 40000, "bid": 39900, "last": 39950, "volume": 100},
        "ETH/BTC": {"ask": 0.075, "bid": 0.074, "last": 0.0745, "volume": 50},
        "ETH/USDT": {"ask": 3010, "bid": 2990, "last": 3000, "volume": 200},
        "XRP/USDT": {"ask": 0.51, "bid": 0.50, "last": 0.505, "volume": 500000},
        "XRP/BTC": {
            "ask": 0.0000128,
            "bid": 0.0000126,
            "last": 0.0000127,
            "volume": 300000,
        },
        "BNB/USDT": {"ask": 405, "bid": 395, "last": 400, "volume": 5000},
        "BNB/BTC": {"ask": 0.0101, "bid": 0.0099, "last": 0.01, "volume": 3000},
        "SOL/USDT": {"ask": 101, "bid": 99, "last": 100, "volume": 20000},
        "SOL/BTC": {"ask": 0.0025, "bid": 0.0024, "last": 0.00245, "volume": 10000},
        "ADA/USDT": {"ask": 1.21, "bid": 1.19, "last": 1.2, "volume": 100000},
        "ADA/BTC": {
            "ask": 0.000030,
            "bid": 0.000029,
            "last": 0.0000295,
            "volume": 50000,
        },
        "MATIC/USDT": {"ask": 1.02, "bid": 0.99, "last": 1.01, "volume": 80000},
        "MATIC/BTC": {
            "ask": 0.0000252,
            "bid": 0.0000248,
            "last": 0.000025,
            "volume": 40000,
        },
        "LINK/USDT": {"ask": 15.2, "bid": 14.8, "last": 15, "volume": 30000},
        "LINK/BTC": {"ask": 0.00038, "bid": 0.00037, "last": 0.000375, "volume": 15000},
        "AVAX/USDT": {"ask": 30.5, "bid": 29.5, "last": 30, "volume": 25000},
        "AVAX/BTC": {"ask": 0.00076, "bid": 0.00074, "last": 0.00075, "volume": 12000},
        "DOT/USDT": {"ask": 20.2, "bid": 19.8, "last": 20, "volume": 35000},
        "DOT/BTC": {"ask": 0.0005, "bid": 0.00049, "last": 0.000495, "volume": 18000},
        "XRP/ETH": {"ask": 0.00017, "bid": 0.00016, "last": 0.000165, "volume": 200000},
        "BNB/ETH": {"ask": 0.135, "bid": 0.133, "last": 0.134, "volume": 2000},
        "SOL/ETH": {"ask": 0.034, "bid": 0.033, "last": 0.0335, "volume": 8000},
        "ADA/ETH": {"ask": 0.00041, "bid": 0.00039, "last": 0.0004, "volume": 30000},
        "MATIC/ETH": {
            "ask": 0.00034,
            "bid": 0.00033,
            "last": 0.000335,
            "volume": 25000,
        },
        "LINK/ETH": {"ask": 0.0051, "bid": 0.0049, "last": 0.005, "volume": 10000},
        "AVAX/ETH": {"ask": 0.01, "bid": 0.0098, "last": 0.0099, "volume": 7000},
        "DOT/ETH": {"ask": 0.0068, "bid": 0.0066, "last": 0.0067, "volume": 12000},
    }

    # Ajouter une petite inefficience pour créer une opportunité d'arbitrage triangulaire
    # BTC/USDT * ETH/BTC * (1/ETH/USDT) normalement = 1, mais nous allons créer un déséquilibre
    data["ETH/USDT"]["bid"] = 2950  # Réduire le prix de vente d'ETH contre USDT

    # Pour SOL, créer une opportunité en augmentant le prix de SOL/BTC
    data["SOL/BTC"]["bid"] = 0.00255  # Augmenter le prix de vente de SOL contre BTC

    return data


def generate_mock_statistical_data():
    """
    Génère des données factices pour tester l'arbitrage statistique.

    Returns:
        Dictionnaire simulant des données de marché pour des paires d'actifs
    """
    # Créer des prix fictifs pour plusieurs jours
    days = 60
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # Prix de base et coévolution
    btc_prices = [40000]
    eth_prices = [3000]
    sol_prices = [100]
    link_prices = [15]
    bnb_prices = [400]

    # Générer des prix avec une relation statistique
    for i in range(1, days):
        # BTC varie librement
        btc_change = np.random.normal(0.001, 0.03)
        btc_prices.append(btc_prices[-1] * (1 + btc_change))

        # ETH suit BTC avec un beta de 1.2 et un bruit
        eth_change = 1.2 * btc_change + np.random.normal(0, 0.01)
        eth_prices.append(eth_prices[-1] * (1 + eth_change))

        # SOL suit BTC avec un beta de 1.5 et un bruit
        sol_change = 1.5 * btc_change + np.random.normal(0, 0.015)
        sol_prices.append(sol_prices[-1] * (1 + sol_change))

        # LINK suit ETH avec un beta de 0.9 et un bruit
        link_change = 0.9 * eth_change + np.random.normal(0, 0.012)
        link_prices.append(link_prices[-1] * (1 + link_change))

        # BNB suit une combinaison de BTC et ETH
        bnb_change = 0.7 * btc_change + 0.3 * eth_change + np.random.normal(0, 0.01)
        bnb_prices.append(bnb_prices[-1] * (1 + bnb_change))

    # Créer un déséquilibre temporaire pour ETH (surévalué) et SOL (sous-évalué)
    eth_prices[-1] *= 1.04  # +4%
    sol_prices[-1] *= 0.96  # -4%

    # Créer le dictionnaire de données
    data = {
        "BTC": {
            "close": btc_prices[-1],
            "open": btc_prices[-2],
            "high": btc_prices[-1] * 1.01,
            "low": btc_prices[-1] * 0.99,
            "volume": 5000,
        },
        "ETH": {
            "close": eth_prices[-1],
            "open": eth_prices[-2],
            "high": eth_prices[-1] * 1.01,
            "low": eth_prices[-1] * 0.99,
            "volume": 10000,
        },
        "SOL": {
            "close": sol_prices[-1],
            "open": sol_prices[-2],
            "high": sol_prices[-1] * 1.01,
            "low": sol_prices[-1] * 0.99,
            "volume": 20000,
        },
        "LINK": {
            "close": link_prices[-1],
            "open": link_prices[-2],
            "high": link_prices[-1] * 1.01,
            "low": link_prices[-1] * 0.99,
            "volume": 15000,
        },
        "BNB": {
            "close": bnb_prices[-1],
            "open": bnb_prices[-2],
            "high": bnb_prices[-1] * 1.01,
            "low": bnb_prices[-1] * 0.99,
            "volume": 8000,
        },
    }

    # Historique des prix pour l'analyse
    price_history = {
        "BTC": pd.DataFrame({"timestamp": dates, "price": btc_prices}),
        "ETH": pd.DataFrame({"timestamp": dates, "price": eth_prices}),
        "SOL": pd.DataFrame({"timestamp": dates, "price": sol_prices}),
        "LINK": pd.DataFrame({"timestamp": dates, "price": link_prices}),
        "BNB": pd.DataFrame({"timestamp": dates, "price": bnb_prices}),
    }

    return data, price_history


def generate_mock_funding_rate_data():
    """
    Génère des données factices pour tester l'arbitrage de taux de financement.

    Returns:
        Dictionnaire simulant des données de marché pour les contrats à terme
    """
    future_time = datetime.now() + timedelta(hours=4)  # Prochain paiement dans 4 heures

    data = {
        "BTC/USDT": {
            "spot": {"price": 40000},
            "perp": {
                "price": 40100,  # 0.25% de base
                "funding_rate": 0.015,  # 1.5% taux positif
                "next_funding_time": future_time.timestamp(),
                "funding_interval": 8,
                "exchange": "binance",
            },
        },
        "ETH/USDT": {
            "spot": {"price": 3000},
            "perp": {
                "price": 2990,  # -0.33% de base
                "funding_rate": -0.01,  # -1% taux négatif
                "next_funding_time": future_time.timestamp(),
                "funding_interval": 8,
                "exchange": "binance",
            },
        },
        "SOL/USDT": {
            "spot": {"price": 100},
            "perp": {
                "price": 101,  # 1% de base
                "funding_rate": 0.025,  # 2.5% taux positif élevé
                "next_funding_time": future_time.timestamp(),
                "funding_interval": 8,
                "exchange": "bybit",
            },
        },
        "AVAX/USDT": {
            "spot": {"price": 30},
            "perp": {
                "price": 29.8,  # -0.67% de base
                "funding_rate": -0.02,  # -2% taux négatif important
                "next_funding_time": future_time.timestamp(),
                "funding_interval": 8,
                "exchange": "okex",
            },
        },
        "BNB/USDT": {
            "spot": {"price": 400},
            "perp": {
                "price": 402,  # 0.5% de base
                "funding_rate": 0.008,  # 0.8% taux faible mais positif
                "next_funding_time": future_time.timestamp(),
                "funding_interval": 8,
                "exchange": "binance",
            },
        },
    }

    return data


def run_spatial_arbitrage_example():
    """
    Exécute un exemple d'arbitrage spatial.
    """
    logger.info("=== Exemple d'arbitrage spatial ===")

    # Créer une stratégie d'arbitrage spatial
    exchanges = ["binance", "coinbase", "kraken", "kucoin", "huobi"]
    strategy = SpatialArbitrageStrategy(
        exchanges=exchanges,
        min_profit_threshold=0.005,  # 0.5%
        transaction_fee=0.001,  # 0.1%
        consider_withdrawal_fees=True,
    )

    # Générer des données de marché fictives
    market_data = generate_mock_exchange_data()

    # Trouver les opportunités d'arbitrage
    opportunities = strategy.find_opportunities(market_data)

    # Afficher les résultats
    logger.info(f"Trouvé {len(opportunities)} opportunités d'arbitrage spatial")

    for i, opportunity in enumerate(opportunities[:3]):  # Afficher les 3 premières
        logger.info(f"Opportunité {i+1}: {opportunity['symbol']}")
        logger.info(
            f"  Acheter sur {opportunity['buy_exchange']} à {opportunity['buy_price']:.2f}"
        )
        logger.info(
            f"  Vendre sur {opportunity['sell_exchange']} à {opportunity['sell_price']:.2f}"
        )
        logger.info(f"  Profit attendu: {opportunity['price_diff_pct']:.2f}%")
        logger.info(f"  Confiance: {opportunity['confidence']:.2f}")

        # Simuler l'exécution de l'arbitrage
        result = strategy.execute_arbitrage(opportunity)
        logger.info(
            f"  Résultat: {result['status']}, Profit après frais: {result['expected_profit']:.4f}%"
        )
        logger.info("")

    # Afficher un résumé
    logger.info(f"Résumé: {strategy.get_summary()}")


def run_triangular_arbitrage_example():
    """
    Exécute un exemple d'arbitrage triangulaire.
    """
    logger.info("\n=== Exemple d'arbitrage triangulaire ===")

    # Créer une stratégie d'arbitrage triangulaire
    strategy = TriangularArbitrageStrategy(
        exchange="binance",
        base_currencies=["BTC", "USDT", "ETH"],
        min_profit_threshold=0.001,  # 0.1%
        transaction_fee=0.001,  # 0.1%
    )

    # Générer des données de marché fictives
    market_data = generate_mock_triangle_data()

    # Trouver les opportunités d'arbitrage
    opportunities = strategy.find_opportunities(market_data)

    # Afficher les résultats
    logger.info(f"Trouvé {len(opportunities)} opportunités d'arbitrage triangulaire")

    for i, opportunity in enumerate(opportunities[:3]):  # Afficher les 3 premières
        logger.info(f"Opportunité {i+1}: {opportunity['triangle']}")
        logger.info(f"  Paires: {', '.join(opportunity['pairs'])}")
        logger.info(f"  Profit attendu: {opportunity['profit_pct']:.4f}%")
        logger.info(f"  Confiance: {opportunity['confidence']:.2f}")

        # Afficher le chemin
        path = opportunity["path"]
        for step in path:
            logger.info(
                f"  {step['direction'].upper()} {step['pair']}: {step['from']} → {step['to']}"
            )

        # Simuler l'exécution de l'arbitrage
        result = strategy.execute_arbitrage(opportunity)
        logger.info(
            f"  Résultat: {result['status']}, Profit après frais: {result['expected_profit']:.4f}%"
        )
        logger.info("")

    # Afficher un résumé
    logger.info(f"Résumé: {strategy.get_summary()}")


def run_statistical_arbitrage_example():
    """
    Exécute un exemple d'arbitrage statistique.
    """
    logger.info("\n=== Exemple d'arbitrage statistique ===")

    # Créer une stratégie d'arbitrage statistique
    pairs = [
        ("BTC", "ETH"),
        ("BTC", "SOL"),
        ("ETH", "LINK"),
        ("BTC", "BNB"),
        ("ETH", "BNB"),
    ]

    strategy = StatisticalArbitrageStrategy(
        pairs=pairs,
        lookback_period=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        min_profit_threshold=0.003,  # 0.3%
        transaction_fee=0.001,  # 0.1%
    )

    # Générer des données de marché fictives
    market_data, price_history = generate_mock_statistical_data()

    # Initialiser l'historique des prix pour la stratégie
    strategy.price_history = price_history

    # Mettre à jour les modèles
    strategy.update_models(force=True)

    # Trouver les opportunités d'arbitrage
    opportunities = strategy.find_opportunities(market_data)

    # Afficher les résultats
    logger.info(f"Trouvé {len(opportunities)} opportunités d'arbitrage statistique")

    for i, opportunity in enumerate(opportunities):
        logger.info(
            f"Opportunité {i+1}: {opportunity['asset1']}/{opportunity['asset2']}"
        )
        logger.info(f"  Z-score: {opportunity['z_score']:.2f}")
        logger.info(f"  Direction: {opportunity['direction']}")
        logger.info(
            f"  Prix actuel {opportunity['asset2']}: {opportunity['price2']:.2f}"
        )
        logger.info(
            f"  Prix théorique {opportunity['asset2']}: {opportunity['theo_price2']:.2f}"
        )
        logger.info(f"  Demi-vie: {opportunity['half_life']:.1f} jours")
        logger.info(f"  Profit attendu: {opportunity['profit_pct']:.2f}%")
        logger.info(f"  Confiance: {opportunity['confidence']:.2f}")

        # Simuler l'exécution de l'arbitrage
        result = strategy.execute_arbitrage(opportunity)
        logger.info(
            f"  Résultat: {result['status']}, Profit après frais: {result['expected_profit']:.4f}%"
        )
        logger.info("")

    # Afficher un résumé
    logger.info(f"Résumé: {strategy.get_summary()}")


def run_funding_rate_arbitrage_example():
    """
    Exécute un exemple d'arbitrage de taux de financement.
    """
    logger.info("\n=== Exemple d'arbitrage de taux de financement ===")

    # Créer une stratégie d'arbitrage de taux de financement
    strategy = FundingRateArbitrageStrategy(
        min_funding_rate=0.008,  # 0.8%
        min_profit_threshold=0.002,  # 0.2%
        transaction_fee=0.001,  # 0.1%
        max_holding_periods=3,
    )

    # Générer des données de marché fictives
    market_data = generate_mock_funding_rate_data()

    # Trouver les opportunités d'arbitrage
    opportunities = strategy.find_opportunities(market_data)

    # Afficher les résultats
    logger.info(
        f"Trouvé {len(opportunities)} opportunités d'arbitrage de taux de financement"
    )

    for i, opportunity in enumerate(opportunities):
        logger.info(f"Opportunité {i+1}: {opportunity['symbol']}")
        logger.info(f"  Taux de financement: {opportunity['funding_rate']*100:+.2f}%")
        logger.info(f"  Direction: {opportunity['direction']}")
        logger.info(f"  Base: {opportunity['basis']:.2f}%")
        logger.info(f"  Périodes attendues: {opportunity['expected_periods']}")
        logger.info(f"  Profit attendu: {opportunity['profit_pct']:.2f}%")
        logger.info(f"  Confiance: {opportunity['confidence']:.2f}")

        # Simuler l'exécution de l'arbitrage
        result = strategy.execute_arbitrage(opportunity)
        logger.info(
            f"  Résultat: {result['status']}, Profit après frais: {result['expected_profit']:.4f}%"
        )
        logger.info("")

    # Afficher un résumé
    logger.info(f"Résumé: {strategy.get_summary()}")


def main():
    """
    Fonction principale exécutant tous les exemples.
    """
    logger.info("Démarrage des exemples d'arbitrage...")

    # Exécuter les exemples pour chaque type d'arbitrage
    run_spatial_arbitrage_example()
    run_triangular_arbitrage_example()
    run_statistical_arbitrage_example()
    run_funding_rate_arbitrage_example()

    logger.info("Exemples d'arbitrage terminés.")


if __name__ == "__main__":
    main()
