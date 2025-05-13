"""
Exemple d'utilisation du système de gestion avancée des timeouts et retries.

Ce script démontre comment utiliser le module resilient_requester pour :
- Gérer les timeouts avec backoff exponentiel
- Implémenter le pattern Circuit Breaker
- Utiliser le fallback sur sources alternatives
- Monitorer les performances des requêtes
"""

import logging
import random
import time
from datetime import datetime
from typing import Any, Dict

import requests

from ai_trading.utils.resilient_requester import (
    AllSourcesFailedError,
    MultiSourceRequester,
    ResilientRequester,
    resilient,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Simulation d'APIs de données financières
def simulate_api_call(
    api_name: str,
    success_rate: float = 0.8,
    timeout_rate: float = 0.1,
    delay: float = 0.2,
) -> Dict[str, Any]:
    """
    Simule un appel API avec possibilité d'échec ou de timeout.

    Args:
        api_name: Nom de l'API
        success_rate: Taux de succès (0-1)
        timeout_rate: Taux de timeout (0-1)
        delay: Délai moyen de réponse

    Returns:
        Dict: Données de réponse

    Raises:
        TimeoutError: Si un timeout est simulé
        ConnectionError: Si une erreur de connexion est simulée
    """
    # Simuler un délai variable
    actual_delay = delay * (0.5 + random.random())
    time.sleep(actual_delay)

    # Simuler un timeout
    if random.random() < timeout_rate:
        logger.warning(f"API {api_name}: Timeout simulé")
        raise TimeoutError(f"Timeout lors de l'appel à {api_name}")

    # Simuler une erreur
    if random.random() > success_rate:
        logger.warning(f"API {api_name}: Erreur simulée")
        raise ConnectionError(f"Erreur de connexion à {api_name}")

    # Générer des données fictives
    return {
        "api": api_name,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "price": round(random.uniform(100, 1000), 2),
            "volume": random.randint(1000, 10000),
            "change": round(random.uniform(-5, 5), 2),
        },
    }


# Fonctions d'API simulées
def coinbase_api(symbol: str) -> Dict[str, Any]:
    """Simule l'API Coinbase."""
    return simulate_api_call("Coinbase", success_rate=0.9, delay=0.3)


def binance_api(symbol: str) -> Dict[str, Any]:
    """Simule l'API Binance."""
    return simulate_api_call("Binance", success_rate=0.8, delay=0.2)


def kraken_api(symbol: str) -> Dict[str, Any]:
    """Simule l'API Kraken."""
    return simulate_api_call("Kraken", success_rate=0.7, delay=0.4)


def demo_resilient_requester():
    """Démontre l'utilisation de ResilientRequester."""
    logger.info("=== Démonstration de ResilientRequester ===")

    # Créer un requester résilient
    requester = ResilientRequester(
        max_retries=3,
        base_delay=0.5,
        max_delay=2.0,
        timeout=1.0,
        jitter=0.2,
        circuit_breaker_config={
            "failure_threshold": 5,
            "recovery_timeout": 5,
            "half_open_max_calls": 2,
        },
        fallback_function=lambda symbol: {"api": "fallback", "data": {"price": 0}},
    )

    # Effectuer plusieurs requêtes
    for i in range(10):
        try:
            logger.info(f"Requête {i+1}/10 à Coinbase...")
            result = requester.request(coinbase_api, "BTC/USD")
            logger.info(f"Succès: {result['api']}, prix: {result['data']['price']}")
        except Exception as e:
            logger.error(f"Échec définitif: {e}")

        # Afficher les métriques périodiquement
        if (i + 1) % 5 == 0:
            metrics = requester.get_metrics()
            logger.info(f"Métriques après {i+1} requêtes: {metrics}")
            logger.info(f"État du circuit: {requester.get_circuit_state()}")


def demo_multi_source_requester():
    """Démontre l'utilisation de MultiSourceRequester."""
    logger.info("\n=== Démonstration de MultiSourceRequester ===")

    # Configurer les sources
    sources = {
        "coinbase": (coinbase_api, {"max_retries": 2, "timeout": 1.0}),
        "binance": (binance_api, {"max_retries": 2, "timeout": 0.8}),
        "kraken": (kraken_api, {"max_retries": 1, "timeout": 1.2}),
    }

    # Créer un requester multi-sources
    multi_requester = MultiSourceRequester(
        sources=sources,
        source_priority=["coinbase", "binance", "kraken"],
        global_timeout=3.0,
        requester_config={"base_delay": 0.3, "jitter": 0.1},
    )

    # Effectuer plusieurs requêtes
    for i in range(10):
        try:
            logger.info(f"Requête {i+1}/10 multi-sources...")
            result, source = multi_requester.request("BTC/USD")
            logger.info(f"Succès via {source}: prix = {result['data']['price']}")
        except AllSourcesFailedError as e:
            logger.error(f"Toutes les sources ont échoué: {e}")

        # Afficher les métriques périodiquement
        if (i + 1) % 5 == 0:
            metrics = multi_requester.get_metrics()
            logger.info(f"Métriques après {i+1} requêtes:")
            for source, source_metrics in metrics.items():
                success_rate = source_metrics.get("success_rate", 0) * 100
                logger.info(f"  - {source}: {success_rate:.1f}% de succès")

            states = multi_requester.get_circuit_states()
            logger.info(f"États des circuits: {states}")


def demo_resilient_decorator():
    """Démontre l'utilisation du décorateur resilient."""
    logger.info("\n=== Démonstration du décorateur resilient ===")

    # Définir une fonction de fallback
    def price_fallback(symbol):
        logger.warning(f"Utilisation du fallback pour {symbol}")
        return {"symbol": symbol, "price": 0.0, "source": "fallback"}

    # Décorer une fonction avec resilient
    @resilient(
        max_retries=2, base_delay=0.3, timeout=0.8, fallback_function=price_fallback
    )
    def get_price(symbol):
        # Simuler un comportement aléatoire
        if random.random() < 0.3:
            raise TimeoutError("Timeout simulé")
        if random.random() < 0.3:
            raise ConnectionError("Erreur de connexion simulée")

        return {
            "symbol": symbol,
            "price": round(random.uniform(100, 1000), 2),
            "source": "api",
        }

    # Utiliser la fonction décorée
    for i in range(10):
        symbol = "ETH/USD"
        result = get_price(symbol)
        logger.info(
            f"Prix pour {symbol}: {result['price']} (source: {result['source']})"
        )

        # Afficher les métriques à la fin
        if i == 9:
            metrics = get_price.get_metrics()
            logger.info(f"Métriques finales: {metrics}")


def demo_real_api_call():
    """Démontre l'utilisation avec une API réelle (CoinGecko)."""
    logger.info("\n=== Démonstration avec API réelle (CoinGecko) ===")

    # Fonction qui appelle l'API CoinGecko
    def coingecko_price(coin_id: str) -> Dict[str, Any]:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()

    # Fonction de fallback
    def price_fallback(coin_id: str) -> Dict[str, Any]:
        return {coin_id: {"usd": 0.0}}

    # Créer un requester résilient
    requester = ResilientRequester(
        max_retries=2, base_delay=1.0, timeout=3.0, fallback_function=price_fallback
    )

    # Liste de cryptomonnaies
    coins = ["bitcoin", "ethereum", "solana", "cardano", "polkadot"]

    # Récupérer les prix
    for coin in coins:
        try:
            result = requester.request(coingecko_price, coin)
            price = result.get(coin, {}).get("usd", "N/A")
            logger.info(f"Prix de {coin}: ${price}")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix de {coin}: {e}")

    # Afficher les métriques
    metrics = requester.get_metrics()
    logger.info(f"Métriques CoinGecko: {metrics}")


def main():
    """Fonction principale."""
    # Exécuter les démonstrations
    demo_resilient_requester()
    demo_multi_source_requester()
    demo_resilient_decorator()

    # Décommenter pour tester avec une API réelle
    # demo_real_api_call()


if __name__ == "__main__":
    main()
