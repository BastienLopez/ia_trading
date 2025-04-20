"""
Module pour la récupération de données de marché améliorées pour les cryptomonnaies.
Intègre les données de marché standards avec les données de flux d'ordres
et de profondeur du carnet d'ordres.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pandas as pd

# Configurer le logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Ajouter le chemin d'accès pour importer depuis le répertoire parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ai_trading.data.market_data import MarketDataFetcher
from ai_trading.data.order_flow import (
    OrderBookDepthCollector,
    OrderFlowCollector,
    integrate_order_flow_and_market_data,
)


class EnhancedMarketDataFetcher:
    """
    Classe étendue pour récupérer des données de marché complètes pour les cryptomonnaies.
    Intègre les données OHLCV standard avec les données de flux d'ordres et
    de profondeur du carnet d'ordres.
    """

    def __init__(self, use_synthetic: bool = True, api_key: Optional[str] = None):
        """
        Initialise le récupérateur de données de marché amélioré.

        Args:
            use_synthetic (bool): Si True, utilise des données synthétiques
                pour les flux d'ordres et la profondeur du carnet
            api_key (str, optional): Clé API pour les sources de données réelles
        """
        self.market_fetcher = MarketDataFetcher()
        self.order_flow_collector = OrderFlowCollector(
            use_synthetic=use_synthetic, api_key=api_key
        )
        self.order_book_collector = OrderBookDepthCollector(
            use_synthetic=use_synthetic, api_key=api_key
        )
        self.use_synthetic = use_synthetic
        logger.info(
            f"EnhancedMarketDataFetcher initialisé (mode {'synthétique' if use_synthetic else 'réel'} pour données alternatives)"
        )

    def fetch_enhanced_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        include_order_flow: bool = True,
        include_order_book: bool = True,
        depth_levels: int = 5,
    ) -> pd.DataFrame:
        """
        Récupère des données de marché améliorées, intégrant potentiellement
        les données de flux d'ordres et de profondeur du carnet.

        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: BTC, ETH, etc.)
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données ('1d', '1h', etc.)
            include_order_flow (bool): Si True, inclut les données de flux d'ordres
            include_order_book (bool): Si True, inclut les données de profondeur du carnet
            depth_levels (int): Nombre de niveaux de profondeur à inclure

        Returns:
            pd.DataFrame: DataFrame contenant les données de marché améliorées
        """
        # Récupérer les données de marché standard
        market_data = self.market_fetcher.fetch_crypto_data(
            symbol=symbol, start_date=start_date, end_date=end_date, interval=interval
        )

        # Vérifier si les données ont été récupérées avec succès
        if market_data is None or market_data.empty:
            logger.error(f"Impossible de récupérer les données de marché pour {symbol}")
            return pd.DataFrame()

        # Variables pour stocker les données supplémentaires
        order_flow_data = None
        order_book_data = None

        # Récupérer les données de flux d'ordres si demandé
        if include_order_flow:
            try:
                logger.info(f"Récupération des données de flux d'ordres pour {symbol}")
                order_flow_data = self.order_flow_collector.collect_order_flow(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                )

                # Calculer des métriques supplémentaires
                order_flow_data = self.order_flow_collector.calculate_flow_metrics(
                    order_flow_data
                )

                logger.info(
                    f"Données de flux d'ordres récupérées: {len(order_flow_data)} points"
                )
            except Exception as e:
                logger.error(
                    f"Erreur lors de la récupération des données de flux d'ordres: {e}"
                )
                order_flow_data = None

        # Récupérer les données de profondeur du carnet si demandé
        if include_order_book:
            try:
                logger.info(
                    f"Récupération des données de profondeur du carnet pour {symbol}"
                )
                order_book_data = self.order_book_collector.collect_order_book_depth(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    depth_levels=depth_levels,
                )

                # Calculer des métriques supplémentaires
                order_book_data = self.order_book_collector.calculate_depth_metrics(
                    order_book_data
                )

                logger.info(
                    f"Données de profondeur du carnet récupérées: {len(order_book_data)} points"
                )
            except Exception as e:
                logger.error(
                    f"Erreur lors de la récupération des données de profondeur du carnet: {e}"
                )
                order_book_data = None

        # Intégrer toutes les données
        try:
            enhanced_data = integrate_order_flow_and_market_data(
                market_data=market_data,
                order_flow_data=(
                    order_flow_data
                    if order_flow_data is not None
                    else pd.DataFrame(index=market_data.index)
                ),
                order_book_data=order_book_data,
            )

            logger.info(
                f"Données de marché améliorées intégrées avec succès: {enhanced_data.shape}"
            )
            return enhanced_data

        except Exception as e:
            logger.error(f"Erreur lors de l'intégration des données: {e}")
            logger.info("Retour des données de marché standard uniquement")
            return market_data

    def get_alternative_data_features(self) -> List[str]:
        """
        Retourne la liste des caractéristiques alternatives disponibles.

        Returns:
            List[str]: Liste des noms de colonnes pour les données alternatives
        """
        # Caractéristiques de flux d'ordres
        order_flow_features = [
            "buy_volume",
            "sell_volume",
            "total_volume",
            "buy_sell_ratio",
            "order_book_imbalance",
            "num_orders",
            "num_buy_orders",
            "num_sell_orders",
            "avg_order_size",
            "cancellation_rate",
            "cumulative_buy_pressure",
            "imbalance_ma5",
            "imbalance_ma20",
            "imbalance_signal",
            "buy_sell_momentum",
            "normalized_pressure",
        ]

        # Caractéristiques de profondeur du carnet
        order_book_features = [
            "mid_price",
            "spread",
            "ask_total_volume",
            "bid_total_volume",
            "ask_count",
            "bid_count",
            "book_imbalance",
            "bid_ask_ratio",
            "bid_ask_ratio_ma10",
            "imbalance_slope",
            "liquidity_measure",
            "liquidity_ma20",
            "depth_volatility",
        ]

        # Niveaux de profondeur (les 5 premiers sont généralement les plus importants)
        depth_levels = 5
        for level in range(1, depth_levels + 1):
            order_book_features.extend(
                [
                    f"ask_price_{level}",
                    f"bid_price_{level}",
                    f"ask_volume_{level}",
                    f"bid_volume_{level}",
                ]
            )

        return order_flow_features + order_book_features


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)

    # Créer une instance de EnhancedMarketDataFetcher
    enhanced_fetcher = EnhancedMarketDataFetcher(use_synthetic=True)

    # Récupérer des données améliorées
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    enhanced_data = enhanced_fetcher.fetch_enhanced_data(
        symbol="BTC",
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        include_order_flow=True,
        include_order_book=True,
        depth_levels=3,
    )

    # Afficher les informations sur les données
    print(f"\nDonnées améliorées récupérées:")
    print(f"Forme: {enhanced_data.shape}")
    print(f"Colonnes: {enhanced_data.columns.tolist()}")
    print("\nPremières lignes:")
    print(enhanced_data.head())
