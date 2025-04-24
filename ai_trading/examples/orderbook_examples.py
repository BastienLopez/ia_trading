import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..utils.orderbook_collector import OrderBookCollector

# Chargement des variables d'environnement
load_dotenv()


def init_collector(config=None):
    """Initialise le collector avec les clés API."""
    try:
        # Configuration par défaut si aucune n'est fournie
        if config is None:
            config = [{"exchange": "binance", "symbol": "BTC/USDT", "limit": 100}]

        # Initialisation avec les clés API
        collector = OrderBookCollector(symbols_config=config)
        return collector
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        return None


def exemple_simple():
    """Exemple simple d'utilisation avec un seul exchange."""
    try:
        collector = init_collector()
        if not collector:
            return

        print("Collecte de données pendant 1 minute...")
        data = collector.collect_orderbook_data(duration_minutes=1, interval_seconds=10)

        if "binance_BTC/USDT" in data:
            print("\nDonnées collectées pour BTC/USDT:")
            print(data["binance_BTC/USDT"].head())
        else:
            print("Aucune donnée collectée pour BTC/USDT")

    except Exception as e:
        print(f"Erreur dans l'exemple simple: {e}")


def exemple_multi_exchanges():
    """Exemple avec plusieurs exchanges et symboles."""
    try:
        config = [
            {"exchange": "binance", "symbol": "BTC/USDT", "limit": 100},
            {"exchange": "binance", "symbol": "ETH/USDT", "limit": 100},
        ]

        collector = init_collector(config)
        if not collector:
            return

        print("Collecte de données multi-exchanges...")
        data = collector.collect_orderbook_data(duration_minutes=1, interval_seconds=10)

        for symbol, df in data.items():
            if not df.empty:
                print(f"\nAnalyse pour {symbol}:")
                print(f"Profondeur moyenne: {df['depth_range_bids_5'].mean():.4f}")
                print(f"Spread moyen (%): {df['spread_pct'].mean():.4f}")
                print(
                    f"Volume total moyen: {(df['total_bid_volume'].mean() + df['total_ask_volume'].mean()):.4f}"
                )

    except Exception as e:
        print(f"Erreur dans l'exemple multi-exchanges: {e}")


def exemple_analyse_impact():
    """Exemple d'analyse d'impact marché et slippage."""
    try:
        collector = init_collector()
        if not collector:
            return

        orderbook = collector.fetch_orderbook("binance", "BTC/USDT")
        if not orderbook:
            print("Impossible de récupérer le carnet d'ordres")
            return

        volumes = [0.1, 0.5, 1.0, 5.0, 10.0]

        print("\nAnalyse d'impact marché pour différents volumes:")
        print("Volume | Slippage (%) | Impact immédiat (%) | Temps de récupération (s)")
        print("-" * 65)

        for volume in volumes:
            try:
                metrics = collector.get_execution_metrics(orderbook, "buy", volume)
                print(
                    f"{volume:6.1f} | {metrics['slippage_pct']:11.2f} | "
                    f"{metrics['immediate_impact_pct']:17.2f} | "
                    f"{metrics['estimated_recovery_time']:21.1f}"
                )
            except Exception as e:
                print(f"Erreur pour le volume {volume}: {e}")

    except Exception as e:
        print(f"Erreur dans l'analyse d'impact: {e}")


def exemple_calibration():
    """Exemple de calibration avec données historiques."""
    try:
        # Création de données historiques simulées
        n_samples = 1000
        historical_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2024-01-01", periods=n_samples, freq="1min"
                ),
                "volume": np.random.lognormal(0, 1, n_samples),
                "price_before": 50000 + np.random.normal(0, 100, n_samples),
                "execution_time": np.random.lognormal(0, 0.5, n_samples),
            }
        )

        historical_data["price_after"] = historical_data["price_before"] * (
            1 + 0.0001 * historical_data["volume"]
        )

        collector = init_collector()
        if not collector:
            return

        params = collector.calibrate_models(historical_data)

        print("\nParamètres calibrés:")
        for param, value in params.items():
            print(f"{param}: {value:.4f}")

        collector.apply_calibration(params)

        orderbook = collector.fetch_orderbook("binance", "BTC/USDT")
        if orderbook:
            metrics = collector.get_execution_metrics(orderbook, "buy", 1.0)
            print("\nMétriques avec modèle calibré:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"Erreur dans la calibration: {e}")


def exemple_complet():
    """Exemple complet d'utilisation."""
    try:
        config = [
            {"exchange": "binance", "symbol": "BTC/USDT", "limit": 100},
            {"exchange": "binance", "symbol": "ETH/USDT", "limit": 100},
        ]

        collector = init_collector(config)
        if not collector:
            return

        # 1. Collecte de données
        print("1. Collecte de données en cours...")
        data = collector.collect_orderbook_data(duration_minutes=1, interval_seconds=10)

        # 2. Analyse de la liquidité
        print("\n2. Analyse de la liquidité:")
        for symbol, df in data.items():
            if not df.empty:
                print(f"\n{symbol}:")
                print(f"Spread moyen (%): {df['spread_pct'].mean():.4f}")
                print(f"Profondeur moyenne: {df['depth_range_bids_5'].mean():.4f}")
                print(f"Déséquilibre volume moyen: {df['volume_imbalance'].mean():.4f}")

        # 3. Simulation d'ordres
        print("\n3. Simulation d'ordres:")
        orderbook = collector.fetch_orderbook("binance", "BTC/USDT")
        if orderbook:
            for volume in [0.1, 1.0, 5.0]:
                print(f"\nSimulation pour {volume} BTC:")
                metrics = collector.get_execution_metrics(orderbook, "buy", volume)
                print(f"Slippage estimé: {metrics['slippage_pct']:.4f}%")
                print(f"Délai d'exécution: {metrics['execution_delay_seconds']:.2f}s")
                print(f"Impact marché: {metrics['immediate_impact_pct']:.4f}%")

            # 4. Calcul VWAP
            vwap_bid, vwap_ask = collector.get_vwap_levels(
                orderbook, volume_threshold=1.0
            )
            if vwap_bid and vwap_ask:
                print(f"\n4. Niveaux VWAP pour 1 BTC:")
                print(f"VWAP bid: {vwap_bid:.2f}")
                print(f"VWAP ask: {vwap_ask:.2f}")

    except Exception as e:
        print(f"Erreur dans l'exemple complet: {e}")


if __name__ == "__main__":
    print("=== Exemple Simple ===")
    exemple_simple()

    print("\n=== Exemple Multi-Exchanges ===")
    exemple_multi_exchanges()

    print("\n=== Exemple Analyse d'Impact ===")
    exemple_analyse_impact()

    print("\n=== Exemple Calibration ===")
    exemple_calibration()

    print("\n=== Exemple Complet ===")
    exemple_complet()
