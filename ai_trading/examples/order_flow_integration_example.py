#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemple d'intégration des données de flux d'ordres et de profondeur du carnet
dans l'analyse de marché pour le trading de cryptomonnaies.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

# Configurer le logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ajouter le chemin d'accès pour importer depuis le répertoire parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ai_trading.data.enhanced_market_data import EnhancedMarketDataFetcher


def parse_args():
    """Analyse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Exemple d'intégration des données de flux d'ordres"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Symbole de la cryptomonnaie (ex: BTC, ETH)",
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Nombre de jours de données historiques"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1d", "1h", "1m"],
        help="Intervalle des données (1d, 1h, 1m)",
    )
    parser.add_argument(
        "--depth_levels",
        type=int,
        default=5,
        help="Nombre de niveaux de profondeur du carnet",
    )
    parser.add_argument("--plot", action="store_true", help="Afficher des graphiques")
    parser.add_argument(
        "--output", type=str, default=None, help="Nom du fichier de sortie (CSV)"
    )

    return parser.parse_args()


def fetch_and_analyze_data(symbol, days, interval, depth_levels):
    """
    Récupère et analyse les données de marché, de flux d'ordres et de profondeur
    du carnet pour une cryptomonnaie.

    Args:
        symbol (str): Symbole de la cryptomonnaie
        days (int): Nombre de jours de données
        interval (str): Intervalle des données
        depth_levels (int): Nombre de niveaux de profondeur

    Returns:
        pd.DataFrame: Données intégrées
    """
    logger.info(
        f"Récupération des données pour {symbol} sur {days} jours avec intervalle {interval}"
    )

    # Dates de début et de fin
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Créer un récupérateur de données amélioré
    fetcher = EnhancedMarketDataFetcher(use_synthetic=True)

    # Récupérer les données complètes
    data = fetcher.fetch_enhanced_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        include_order_flow=True,
        include_order_book=True,
        depth_levels=depth_levels,
    )

    logger.info(f"Données récupérées avec succès: {data.shape}")

    # Afficher les informations sur les données
    logger.info(f"Colonnes disponibles: {data.columns.tolist()}")

    return data


def analyze_order_flow(data):
    """
    Analyse les données de flux d'ordres et retourne des insights.

    Args:
        data (pd.DataFrame): Données intégrées

    Returns:
        dict: Insights sur le flux d'ordres
    """
    # Vérifier que les colonnes nécessaires sont présentes
    required_cols = [
        "close",
        "buy_volume",
        "sell_volume",
        "buy_sell_ratio",
        "order_book_imbalance",
    ]
    if not all(col in data.columns for col in required_cols):
        logger.warning("Les données ne contiennent pas toutes les colonnes requises")
        return {}

    # Calcul des corrélations
    price_correlation = data[["close", "buy_sell_ratio", "order_book_imbalance"]].corr()

    # Analyse de tendance
    if "normalized_pressure" in data.columns and len(data) > 5:
        # Détecter les zones de forte pression d'achat (>0.2) ou de vente (<-0.2)
        buy_pressure_zones = data[data["normalized_pressure"] > 0.2].index
        sell_pressure_zones = data[data["normalized_pressure"] < -0.2].index

        # Calculer le rendement moyen après une forte pression d'achat
        future_returns = []
        for idx in buy_pressure_zones:
            if idx < data.index[-5]:  # Vérifier qu'il reste au moins 5 périodes
                position = data.index.get_loc(idx)
                future_return = (
                    data.iloc[position + 5]["close"] / data.loc[idx]["close"]
                ) - 1
                future_returns.append(future_return)

        avg_return_after_buy_pressure = (
            np.mean(future_returns) if future_returns else None
        )
    else:
        buy_pressure_zones = []
        sell_pressure_zones = []
        avg_return_after_buy_pressure = None

    # Retourner les insights
    return {
        "price_correlation": price_correlation,
        "buy_pressure_zones": buy_pressure_zones,
        "sell_pressure_zones": sell_pressure_zones,
        "avg_return_after_buy_pressure": avg_return_after_buy_pressure,
        "buy_volume_proportion": data["buy_volume"].sum()
        / (data["buy_volume"].sum() + data["sell_volume"].sum()),
    }


def analyze_order_book(data):
    """
    Analyse les données de profondeur du carnet et retourne des insights.

    Args:
        data (pd.DataFrame): Données intégrées

    Returns:
        dict: Insights sur la profondeur du carnet
    """
    # Vérifier que les colonnes nécessaires sont présentes
    required_cols = ["spread", "ask_total_volume", "bid_total_volume"]
    if not all(col in data.columns for col in required_cols):
        logger.warning("Les données ne contiennent pas toutes les colonnes requises")
        return {}

    # Calculer le déséquilibre moyen
    if "book_imbalance" in data.columns:
        avg_imbalance = data["book_imbalance"].mean()
        imbalance_volatility = data["book_imbalance"].std()
    else:
        avg_imbalance = None
        imbalance_volatility = None

    # Calculer le spread moyen
    avg_spread = data["spread"].mean()
    spread_volatility = data["spread"].std()

    # Analyse de liquidité
    if "liquidity_measure" in data.columns:
        avg_liquidity = data["liquidity_measure"].mean()
        liquidity_trend = data["liquidity_measure"].diff().mean()
    else:
        avg_liquidity = None
        liquidity_trend = None

    # Retourner les insights
    return {
        "avg_imbalance": avg_imbalance,
        "imbalance_volatility": imbalance_volatility,
        "avg_spread": avg_spread,
        "spread_volatility": spread_volatility,
        "avg_liquidity": avg_liquidity,
        "liquidity_trend": liquidity_trend,
    }


def plot_data(data, symbol):
    """
    Crée des visualisations des données de marché avec les données alternatives.

    Args:
        data (pd.DataFrame): Données intégrées
        symbol (str): Symbole de la cryptomonnaie
    """
    # Configurer matplotlib
    plt.style.use("seaborn-v0_8-darkgrid")

    # Figure 1: Prix et volume avec données de flux d'ordres
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Sous-graphique 1: Prix et buy/sell ratio
    ax1 = axes[0]
    ax1.set_title(f"{symbol} - Prix et ratio achat/vente")
    ax1.plot(data.index, data["close"], "b-", label="Prix de clôture")
    ax1.set_ylabel("Prix", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(data.index, data["buy_sell_ratio"], "g-", label="Ratio achat/vente")
    ax1_twin.set_ylabel("Ratio achat/vente", color="g")
    ax1_twin.tick_params(axis="y", labelcolor="g")
    ax1_twin.set_ylim(0, 1)

    # Sous-graphique 2: Volume et déséquilibre du carnet
    ax2 = axes[1]
    ax2.set_title("Volume et déséquilibre du carnet")
    ax2.bar(data.index, data["volume"], color="blue", alpha=0.6, label="Volume")
    ax2.set_ylabel("Volume", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        data.index, data["order_book_imbalance"], "r-", label="Déséquilibre du carnet"
    )
    ax2_twin.set_ylabel("Déséquilibre", color="r")
    ax2_twin.tick_params(axis="y", labelcolor="r")
    ax2_twin.set_ylim(-1, 1)

    # Sous-graphique 3: Pression d'achat/vente normalisée
    if "normalized_pressure" in data.columns:
        ax3 = axes[2]
        ax3.set_title("Pression d'achat/vente normalisée")
        ax3.plot(
            data.index,
            data["normalized_pressure"],
            "purple",
            label="Pression normalisée",
        )
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax3.fill_between(
            data.index,
            0,
            data["normalized_pressure"],
            where=(data["normalized_pressure"] >= 0),
            color="green",
            alpha=0.3,
            interpolate=True,
        )
        ax3.fill_between(
            data.index,
            0,
            data["normalized_pressure"],
            where=(data["normalized_pressure"] <= 0),
            color="red",
            alpha=0.3,
            interpolate=True,
        )
        ax3.set_ylabel("Pression normalisée")
        ax3.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(f"{symbol}_order_flow_analysis.png", dpi=300)

    # Figure 2: Profondeur du carnet d'ordres
    if "ask_price_1" in data.columns and "bid_price_1" in data.columns:
        plt.figure(figsize=(14, 10))

        # Sélectionner la dernière entrée pour visualiser la profondeur du carnet
        last_entry = data.iloc[-1]

        # Trouver toutes les colonnes liées à la profondeur
        ask_price_cols = [col for col in data.columns if col.startswith("ask_price_")]
        bid_price_cols = [col for col in data.columns if col.startswith("bid_price_")]
        ask_volume_cols = [col for col in data.columns if col.startswith("ask_volume_")]
        bid_volume_cols = [col for col in data.columns if col.startswith("bid_volume_")]

        # Extraire les données
        ask_prices = [last_entry[col] for col in ask_price_cols]
        bid_prices = [last_entry[col] for col in bid_price_cols]
        ask_volumes = [last_entry[col] for col in ask_volume_cols]
        bid_volumes = [last_entry[col] for col in bid_volume_cols]

        # Créer un graphique de la profondeur du carnet
        plt.barh(
            ask_prices, ask_volumes, color="red", alpha=0.6, label="Offres de vente"
        )
        plt.barh(
            bid_prices,
            [-v for v in bid_volumes],
            color="green",
            alpha=0.6,
            label="Offres d'achat",
        )

        plt.axhline(
            y=last_entry["mid_price"],
            color="blue",
            linestyle="--",
            label="Prix du marché",
        )
        plt.xlabel("Volume")
        plt.ylabel("Prix")
        plt.title(f"{symbol} - Profondeur du carnet d'ordres (dernière entrée)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{symbol}_order_book_depth.png", dpi=300)

    plt.close("all")

    logger.info(
        f"Graphiques sauvegardés: {symbol}_order_flow_analysis.png et {symbol}_order_book_depth.png"
    )


def main():
    """Fonction principale."""
    args = parse_args()

    # Récupérer et analyser les données
    data = fetch_and_analyze_data(
        symbol=args.symbol,
        days=args.days,
        interval=args.interval,
        depth_levels=args.depth_levels,
    )

    # Analyser les données
    flow_insights = analyze_order_flow(data)
    book_insights = analyze_order_book(data)

    # Afficher les insights
    logger.info("=== Insights sur le flux d'ordres ===")
    for key, value in flow_insights.items():
        if key != "price_correlation" and key not in (
            "buy_pressure_zones",
            "sell_pressure_zones",
        ):
            logger.info(f"{key}: {value}")

    if "price_correlation" in flow_insights:
        logger.info("\nCorrelation prix vs flux d'ordres:")
        logger.info(flow_insights["price_correlation"])

    logger.info("\n=== Insights sur la profondeur du carnet ===")
    for key, value in book_insights.items():
        logger.info(f"{key}: {value}")

    # Créer des visualisations si demandé
    if args.plot:
        logger.info("\nCréation des visualisations...")
        plot_data(data, args.symbol)

    # Sauvegarder les données si demandé
    if args.output:
        output_file = args.output
        data.to_csv(output_file)
        logger.info(f"Données sauvegardées dans: {output_file}")


if __name__ == "__main__":
    main()
