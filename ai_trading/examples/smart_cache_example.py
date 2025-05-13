"""
Exemple d'utilisation du cache intelligent pour les données financières.

Ce script démontre comment utiliser SmartCache et DataCache pour :
- Mettre en cache des données financières
- Optimiser les requêtes répétitives
- Gérer la compression automatique
- Précharger des données fréquemment utilisées
"""

import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ai_trading.utils.smart_cache import DataCache, SmartCache

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Répertoire de cache
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ai_trading_cache")


def fetch_crypto_data(
    symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
) -> pd.DataFrame:
    """
    Récupère des données de cryptomonnaies depuis une API.

    Args:
        symbol: Symbole de la cryptomonnaie (ex: BTC/USDT)
        start_date: Date de début
        end_date: Date de fin
        interval: Intervalle de temps

    Returns:
        pd.DataFrame: Données OHLCV
    """
    try:
        # Simuler une requête API lente
        logger.info(
            f"Récupération des données pour {symbol} de {start_date} à {end_date}"
        )
        time.sleep(1)  # Simuler la latence réseau

        # Générer des données fictives pour la démo
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        seed = sum(
            ord(c) for c in symbol
        )  # Seed basée sur le symbole pour la cohérence
        np.random.seed(seed)

        data = pd.DataFrame(
            {
                "date": date_range,
                "open": np.random.randn(len(date_range)) * 10 + 100,
                "high": np.random.randn(len(date_range)) * 10 + 105,
                "low": np.random.randn(len(date_range)) * 10 + 95,
                "close": np.random.randn(len(date_range)) * 10 + 100,
                "volume": np.random.randint(1000, 10000, len(date_range)),
            }
        )

        logger.info(f"Données récupérées: {len(data)} points")
        return data

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données: {e}")
        return pd.DataFrame()


def fetch_stock_data(
    symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
) -> pd.DataFrame:
    """
    Récupère des données d'actions depuis une API.

    Args:
        symbol: Symbole de l'action
        start_date: Date de début
        end_date: Date de fin
        interval: Intervalle de temps

    Returns:
        pd.DataFrame: Données OHLCV
    """
    try:
        # Simuler une requête API lente
        logger.info(
            f"Récupération des données pour {symbol} de {start_date} à {end_date}"
        )
        time.sleep(1.5)  # Simuler la latence réseau

        # Générer des données fictives pour la démo
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        seed = sum(
            ord(c) for c in symbol
        )  # Seed basée sur le symbole pour la cohérence
        np.random.seed(seed)

        data = pd.DataFrame(
            {
                "date": date_range,
                "open": np.random.randn(len(date_range)) * 5 + 50,
                "high": np.random.randn(len(date_range)) * 5 + 52,
                "low": np.random.randn(len(date_range)) * 5 + 48,
                "close": np.random.randn(len(date_range)) * 5 + 50,
                "volume": np.random.randint(10000, 100000, len(date_range)),
            }
        )

        logger.info(f"Données récupérées: {len(data)} points")
        return data

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données: {e}")
        return pd.DataFrame()


def demo_basic_cache():
    """Démontre l'utilisation basique de SmartCache."""
    logger.info("=== Démonstration de SmartCache ===")

    # Créer un cache
    cache = SmartCache(
        max_size=100,
        ttl=3600,  # 1 heure
        compression_level=3,
        preload_threshold=3,
        cache_dir=os.path.join(CACHE_DIR, "basic_cache"),
        persist=True,
    )

    # Stocker des données simples
    cache.set("simple_key", "simple_value")
    logger.info(f"Valeur récupérée: {cache.get('simple_key')}")

    # Stocker des données complexes
    complex_data = {
        "numbers": list(range(1000)),
        "matrix": np.random.randn(100, 100).tolist(),
        "metadata": {"created_at": datetime.now().isoformat(), "version": "1.0.0"},
    }

    cache.set("complex_key", complex_data)
    retrieved_data = cache.get("complex_key")
    logger.info(
        f"Données complexes récupérées, taille: {len(str(retrieved_data))} caractères"
    )

    # Afficher les statistiques du cache
    stats = cache.get_stats()
    logger.info(f"Statistiques du cache: {stats}")


def demo_data_cache():
    """Démontre l'utilisation de DataCache pour les données financières."""
    logger.info("\n=== Démonstration de DataCache ===")

    # Créer un cache de données
    data_cache = DataCache(
        max_size=200,
        ttl=3600 * 24,  # 24 heures
        compression_level=3,
        preload_threshold=2,
        cache_dir=os.path.join(CACHE_DIR, "data_cache"),
        persist=True,
        data_sources={"crypto": fetch_crypto_data, "stock": fetch_stock_data},
    )

    # Définir la période
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Mesurer le temps pour la première requête (sans cache)
    start_time = time.time()
    btc_data = data_cache.get_timeseries(
        symbol="BTC/USDT",
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        source="crypto",
    )
    first_request_time = time.time() - start_time
    logger.info(
        f"Première requête: {len(btc_data)} points en {first_request_time:.2f} secondes"
    )

    # Mesurer le temps pour la deuxième requête (avec cache)
    start_time = time.time()
    btc_data_cached = data_cache.get_timeseries(
        symbol="BTC/USDT",
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        source="crypto",
    )
    second_request_time = time.time() - start_time
    logger.info(
        f"Deuxième requête (cachée): {len(btc_data_cached)} points en {second_request_time:.2f} secondes"
    )
    logger.info(f"Accélération: {first_request_time / second_request_time:.1f}x")

    # Récupérer des données d'une autre source
    start_time = time.time()
    aapl_data = data_cache.get_timeseries(
        symbol="AAPL",
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        source="stock",
    )
    logger.info(
        f"Données AAPL récupérées: {len(aapl_data)} points en {time.time() - start_time:.2f} secondes"
    )

    # Au lieu de visualiser les données avec matplotlib, afficher quelques statistiques
    logger.info(
        f"Statistiques BTC: min={btc_data['close'].min():.2f}, max={btc_data['close'].max():.2f}, moyenne={btc_data['close'].mean():.2f}"
    )
    logger.info(
        f"Statistiques AAPL: min={aapl_data['close'].min():.2f}, max={aapl_data['close'].max():.2f}, moyenne={aapl_data['close'].mean():.2f}"
    )

    # Afficher les statistiques du cache
    stats = data_cache.get_stats()
    logger.info(f"Statistiques du cache: {stats}")

    # Tester l'invalidation
    count = data_cache.invalidate("timeseries:BTC*")
    logger.info(f"{count} entrées invalidées")


def demo_preloading():
    """Démontre le préchargement des données."""
    logger.info("\n=== Démonstration du préchargement ===")

    # Créer un cache
    cache = SmartCache(
        max_size=50,
        ttl=3600,
        cache_dir=os.path.join(CACHE_DIR, "preload_cache"),
        persist=True,
    )

    # Définir une fonction de chargement lente
    def slow_data_loader():
        logger.info("Chargement lent des données...")
        time.sleep(2)
        return np.random.randn(1000, 10)

    # Précharger les données en arrière-plan
    logger.info("Démarrage du préchargement...")

    import threading

    preload_thread = threading.Thread(
        target=lambda: cache.preload("slow_data", slow_data_loader), daemon=True
    )
    preload_thread.start()

    # Simuler d'autres opérations
    logger.info("Exécution d'autres opérations pendant le préchargement...")
    for i in range(3):
        time.sleep(0.5)
        logger.info(f"Opération {i+1}")

    # Attendre que le préchargement soit terminé
    preload_thread.join()

    # Accéder aux données préchargées
    start_time = time.time()
    data = cache.get("slow_data")
    logger.info(
        f"Données préchargées récupérées en {time.time() - start_time:.4f} secondes"
    )
    logger.info(f"Forme des données: {data.shape}")


def demo_compression():
    """Démontre la compression automatique des données volumineuses."""
    logger.info("\n=== Démonstration de la compression ===")

    # Créer un cache avec différents niveaux de compression
    cache_low = SmartCache(
        max_size=10,
        compression_level=1,
        cache_dir=os.path.join(CACHE_DIR, "compression_low"),
        persist=False,
    )

    cache_high = SmartCache(
        max_size=10,
        compression_level=19,
        cache_dir=os.path.join(CACHE_DIR, "compression_high"),
        persist=False,
    )

    # Créer des données volumineuses
    large_data = pd.DataFrame(np.random.randn(100000, 10))

    # Mesurer la taille sans compression
    import pickle

    raw_size = len(pickle.dumps(large_data))
    logger.info(f"Taille des données brutes: {raw_size / 1024 / 1024:.2f} Mo")

    # Stocker dans les caches
    cache_low.set("large_data", large_data)
    cache_high.set("large_data", large_data)

    # Récupérer et vérifier
    data_low = cache_low.get("large_data")
    data_high = cache_high.get("large_data")

    # Vérifier que les données sont identiques
    pd.testing.assert_frame_equal(large_data, data_low)
    pd.testing.assert_frame_equal(large_data, data_high)

    # Afficher les statistiques
    stats_low = cache_low.get_stats()
    stats_high = cache_high.get_stats()

    logger.info(
        f"Cache avec compression faible: {stats_low['memory_usage_estimate'] / 1024 / 1024:.2f} Mo"
    )
    logger.info(
        f"Cache avec compression élevée: {stats_high['memory_usage_estimate'] / 1024 / 1024:.2f} Mo"
    )


def main():
    """Fonction principale."""
    # Créer le répertoire de cache s'il n'existe pas
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Exécuter les démonstrations
    demo_basic_cache()
    demo_data_cache()
    demo_preloading()
    demo_compression()


if __name__ == "__main__":
    main()
