"""
Module pour la collecte et l'analyse des données de flux d'ordres (order flow)
et de profondeur du carnet d'ordres (order book depth).
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np
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


class OrderFlowCollector:
    """
    Classe pour collecter et analyser les données de flux d'ordres.
    Peut collecter des données réelles depuis des APIs ou générer des données synthétiques.
    """

    def __init__(self, use_synthetic: bool = True, api_key: Optional[str] = None):
        """
        Initialise le collecteur de flux d'ordres.

        Args:
            use_synthetic (bool): Si True, génère des données synthétiques
            api_key (str, optional): Clé API pour les sources de données réelles
        """
        self.use_synthetic = use_synthetic
        self.api_key = api_key
        logger.info(
            f"OrderFlowCollector initialisé (mode {'synthétique' if use_synthetic else 'réel'})"
        )

    def collect_order_flow(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Collecte des données de flux d'ordres pour une cryptomonnaie.

        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: BTC, ETH)
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données ('1h', '1d', etc.)

        Returns:
            pd.DataFrame: DataFrame contenant les données de flux d'ordres
        """
        if self.use_synthetic:
            logger.info(
                f"Génération de données synthétiques de flux d'ordres pour {symbol}"
            )
            return self._generate_synthetic_order_flow(
                symbol, start_date, end_date, interval
            )
        else:
            try:
                logger.info(
                    f"Tentative de collecte de données réelles de flux d'ordres pour {symbol}"
                )
                return self._collect_real_order_flow(
                    symbol, start_date, end_date, interval
                )
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des données réelles: {e}")
                logger.info("Utilisation de données synthétiques comme fallback")
                return self._generate_synthetic_order_flow(
                    symbol, start_date, end_date, interval
                )

    def _collect_real_order_flow(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Collecte des données réelles de flux d'ordres depuis des APIs.

        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données

        Returns:
            pd.DataFrame: DataFrame contenant les données réelles
        """
        # Placeholder pour l'implémentation d'API réelle
        # Dans une implémentation réelle, on utiliserait des appels API à des
        # plateformes comme Binance, FTX, Coinbase, etc.

        logger.warning("La collecte de données réelles n'est pas encore implémentée")

        # Simuler un délai d'API
        time.sleep(0.5)

        # Générer des données synthétiques comme placeholder
        return self._generate_synthetic_order_flow(
            symbol, start_date, end_date, interval
        )

    def _generate_synthetic_order_flow(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Génère des données synthétiques de flux d'ordres.

        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données

        Returns:
            pd.DataFrame: DataFrame contenant des données synthétiques
        """
        # Convertir les dates en datetime
        start_dt = (
            pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        )
        end_dt = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        # Créer un index de dates
        if interval == "1d":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
        elif interval == "1h":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="H")
        elif interval == "1m":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="T")
        else:
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="H")

        n = len(date_range)

        # Générer des prix de base selon le symbole
        base_price = 100
        if symbol.upper() == "BTC":
            base_price = 30000
            liquidity_factor = 100
        elif symbol.upper() == "ETH":
            base_price = 2000
            liquidity_factor = 50
        elif symbol.upper() == "SOL":
            base_price = 100
            liquidity_factor = 20
        else:
            liquidity_factor = 10

        # Générer une tendance de prix pour la cohérence
        trend = np.cumsum(np.random.normal(0, 0.005, n))
        prices = base_price * (1 + trend)

        # Générer des données de flux d'ordres synthétiques

        # Buy/sell pressure (ratio entre 0.3 et 0.7, moyenne autour de 0.5)
        # Valeurs > 0.5 indiquent plus d'ordres d'achat, < 0.5 plus d'ordres de vente
        raw_buy_ratio = (
            0.5
            + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n))
            + np.random.normal(0, 0.05, n)
        )
        buy_sell_ratio = np.clip(raw_buy_ratio, 0.3, 0.7)

        # Volume total des ordres (en unités de la crypto)
        volume_base = liquidity_factor * (
            1 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, n))
        )
        volume_noise = np.random.lognormal(0, 0.2, n)
        order_volume = volume_base * volume_noise

        # Calculer buy et sell volume
        buy_volume = order_volume * buy_sell_ratio
        sell_volume = order_volume * (1 - buy_sell_ratio)

        # Taille moyenne des ordres
        avg_order_size = liquidity_factor * 0.01 * (1 + np.random.normal(0, 0.1, n))

        # Nombre d'ordres
        num_orders = np.round(order_volume / avg_order_size).astype(int)
        num_buy_orders = np.round(buy_volume / avg_order_size).astype(int)
        num_sell_orders = np.round(sell_volume / avg_order_size).astype(int)

        # Annulations d'ordres (en % du total)
        cancellation_rate = 0.2 + 0.1 * np.random.random(n)

        # Imbalance du carnet d'ordres (>0 indique plus d'acheteurs, <0 plus de vendeurs)
        # Valeur entre -1 et 1
        order_book_imbalance = 2 * (buy_sell_ratio - 0.5)

        # Calcul d'un indicateur de pression d'achat/vente cumulatif
        # Valeur entre 0 et 100
        cumul_pressure = (
            50 + np.cumsum(order_book_imbalance - np.mean(order_book_imbalance)) * 5
        )
        cumul_pressure = np.clip(cumul_pressure, 0, 100)

        # Créer le DataFrame
        df = pd.DataFrame(
            {
                "price": prices,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "total_volume": order_volume,
                "buy_sell_ratio": buy_sell_ratio,
                "order_book_imbalance": order_book_imbalance,
                "num_orders": num_orders,
                "num_buy_orders": num_buy_orders,
                "num_sell_orders": num_sell_orders,
                "avg_order_size": avg_order_size,
                "cancellation_rate": cancellation_rate,
                "cumulative_buy_pressure": cumul_pressure,
            },
            index=date_range,
        )

        logger.info(
            f"Données de flux d'ordres synthétiques générées: {len(df)} points pour {symbol}"
        )
        return df

    def calculate_flow_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des métriques supplémentaires basées sur le flux d'ordres.

        Args:
            df (pd.DataFrame): DataFrame contenant les données de flux d'ordres

        Returns:
            pd.DataFrame: DataFrame avec métriques supplémentaires
        """
        # Vérifier si le DataFrame contient les colonnes nécessaires
        required_cols = ["buy_volume", "sell_volume", "order_book_imbalance"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(
                "Le DataFrame ne contient pas toutes les colonnes requises pour les métriques"
            )
            return df

        # Copier le DataFrame pour ne pas modifier l'original
        result = df.copy()

        # Calculer des moyennes mobiles pour l'imbalance
        result["imbalance_ma5"] = df["order_book_imbalance"].rolling(window=5).mean()
        result["imbalance_ma20"] = df["order_book_imbalance"].rolling(window=20).mean()

        # Calculer la différence entre les moyennes mobiles (signal de tendance)
        result["imbalance_signal"] = result["imbalance_ma5"] - result["imbalance_ma20"]

        # Calculer le momentum du flux d'ordres
        result["buy_sell_momentum"] = (
            df["buy_sell_ratio"].diff(5).rolling(window=10).mean()
        )

        # Indicateur de pression d'achat/vente normalisé
        buy_volume_ma = df["buy_volume"].rolling(window=10).mean()
        sell_volume_ma = df["sell_volume"].rolling(window=10).mean()
        total_volume_ma = buy_volume_ma + sell_volume_ma

        # Éviter la division par zéro
        total_volume_ma = total_volume_ma.replace(0, np.nan)

        # Calculer la pression normalisée (-1 à 1)
        result["normalized_pressure"] = (
            buy_volume_ma - sell_volume_ma
        ) / total_volume_ma

        # Remplir les valeurs NaN
        result = result.bfill().ffill()

        return result


class OrderBookDepthCollector:
    """
    Classe pour collecter et analyser les données de profondeur du carnet d'ordres.
    """

    def __init__(self, use_synthetic: bool = True, api_key: Optional[str] = None):
        """
        Initialise le collecteur de profondeur du carnet d'ordres.

        Args:
            use_synthetic (bool): Si True, génère des données synthétiques
            api_key (str, optional): Clé API pour les sources de données réelles
        """
        self.use_synthetic = use_synthetic
        self.api_key = api_key
        logger.info(
            f"OrderBookDepthCollector initialisé (mode {'synthétique' if use_synthetic else 'réel'})"
        )

    def collect_order_book_depth(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1h",
        depth_levels: int = 10,
    ) -> pd.DataFrame:
        """
        Collecte des données de profondeur du carnet d'ordres.

        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données
            depth_levels (int): Nombre de niveaux de profondeur à collecter

        Returns:
            pd.DataFrame: DataFrame contenant les données de profondeur
        """
        if self.use_synthetic:
            logger.info(
                f"Génération de données synthétiques de profondeur du carnet pour {symbol}"
            )
            return self._generate_synthetic_depth(
                symbol, start_date, end_date, interval, depth_levels
            )
        else:
            try:
                logger.info(
                    f"Tentative de collecte de données réelles de profondeur pour {symbol}"
                )
                return self._collect_real_depth(
                    symbol, start_date, end_date, interval, depth_levels
                )
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des données réelles: {e}")
                logger.info("Utilisation de données synthétiques comme fallback")
                return self._generate_synthetic_depth(
                    symbol, start_date, end_date, interval, depth_levels
                )

    def _collect_real_depth(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1h",
        depth_levels: int = 10,
    ) -> pd.DataFrame:
        """
        Collecte des données réelles de profondeur depuis des APIs.

        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données
            depth_levels (int): Nombre de niveaux de profondeur

        Returns:
            pd.DataFrame: DataFrame contenant les données réelles
        """
        # Placeholder pour l'implémentation d'API réelle
        logger.warning(
            "La collecte de données réelles de profondeur n'est pas encore implémentée"
        )

        # Simuler un délai d'API
        time.sleep(0.5)

        # Générer des données synthétiques comme placeholder
        return self._generate_synthetic_depth(
            symbol, start_date, end_date, interval, depth_levels
        )

    def _generate_synthetic_depth(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1h",
        depth_levels: int = 10,
    ) -> pd.DataFrame:
        """
        Génère des données synthétiques de profondeur du carnet d'ordres.

        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données
            depth_levels (int): Nombre de niveaux de profondeur

        Returns:
            pd.DataFrame: DataFrame contenant des données synthétiques
        """
        # Convertir les dates en datetime
        start_dt = (
            pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        )
        end_dt = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        # Créer un index de dates
        if interval == "1d":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
        elif interval == "1h":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="H")
        elif interval == "1m":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="T")
        else:
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="H")

        n = len(date_range)

        # Générer des prix de base selon le symbole
        base_price = 100
        if symbol.upper() == "BTC":
            base_price = 30000
            liquidity_factor = 100
        elif symbol.upper() == "ETH":
            base_price = 2000
            liquidity_factor = 50
        elif symbol.upper() == "SOL":
            base_price = 100
            liquidity_factor = 20
        else:
            liquidity_factor = 10

        # Générer une tendance de prix pour la cohérence
        trend = np.cumsum(np.random.normal(0, 0.005, n))
        prices = base_price * (1 + trend)

        # Données agrégées de profondeur
        data = {
            "price": prices,
            "mid_price": prices,
            "spread": prices * np.random.uniform(0.0005, 0.002, n),
            "ask_total_volume": liquidity_factor * np.random.lognormal(3, 0.5, n),
            "bid_total_volume": liquidity_factor * np.random.lognormal(3, 0.5, n),
            "ask_count": np.random.randint(100, 500, n),
            "bid_count": np.random.randint(100, 500, n),
        }

        # Ajouter des métriques de déséquilibre du carnet
        data["book_imbalance"] = (
            data["bid_total_volume"] - data["ask_total_volume"]
        ) / (data["bid_total_volume"] + data["ask_total_volume"])

        # Ajouter des métriques pour chaque niveau de profondeur
        for level in range(
            1, min(depth_levels + 1, 11)
        ):  # Max 10 niveaux pour simplifier
            # Prix ask (vente) et bid (achat) à différents niveaux
            price_inc = level * data["spread"] / depth_levels
            data[f"ask_price_{level}"] = data["mid_price"] + price_inc
            data[f"bid_price_{level}"] = data["mid_price"] - price_inc

            # Volume à chaque niveau - décroît avec la distance du prix du marché
            level_factor = np.exp(-0.5 * level)
            data[f"ask_volume_{level}"] = (
                data["ask_total_volume"] * level_factor * np.random.uniform(0.1, 0.3, n)
            )
            data[f"bid_volume_{level}"] = (
                data["bid_total_volume"] * level_factor * np.random.uniform(0.1, 0.3, n)
            )

        # Créer le DataFrame
        df = pd.DataFrame(data, index=date_range)

        logger.info(
            f"Données de profondeur synthétiques générées: {len(df)} points pour {symbol}"
        )
        return df

    def calculate_depth_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des métriques supplémentaires basées sur la profondeur du carnet.

        Args:
            df (pd.DataFrame): DataFrame contenant les données de profondeur

        Returns:
            pd.DataFrame: DataFrame avec métriques supplémentaires
        """
        # Copier le DataFrame pour ne pas modifier l'original
        result = df.copy()

        # Vérifier si le DataFrame contient les colonnes nécessaires
        if "ask_total_volume" not in df.columns or "bid_total_volume" not in df.columns:
            logger.warning("Le DataFrame ne contient pas les colonnes nécessaires")
            return result

        # Calculer le ratio moyen bid/ask sur plusieurs périodes
        result["bid_ask_ratio"] = df["bid_total_volume"] / df["ask_total_volume"]
        result["bid_ask_ratio_ma10"] = result["bid_ask_ratio"].rolling(window=10).mean()

        # Calculer la pente du déséquilibre du carnet
        if "book_imbalance" in df.columns:
            result["imbalance_slope"] = (
                df["book_imbalance"].diff().rolling(window=5).mean()
            )

        # Calculer l'indicateur de liquidité
        result["liquidity_measure"] = (
            df["ask_total_volume"] + df["bid_total_volume"]
        ) / 2
        result["liquidity_ma20"] = result["liquidity_measure"].rolling(window=20).mean()

        # Calculer la volatilité de la profondeur
        if "ask_volume_1" in df.columns and "bid_volume_1" in df.columns:
            top_level_volume = (df["ask_volume_1"] + df["bid_volume_1"]) / 2
            result["depth_volatility"] = (
                top_level_volume.rolling(window=10).std()
                / top_level_volume.rolling(window=10).mean()
            )

        # Remplir les valeurs NaN
        result = result.bfill().ffill()

        return result


def integrate_order_flow_and_market_data(
    market_data: pd.DataFrame,
    order_flow_data: pd.DataFrame,
    order_book_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Intègre les données de marché avec les données de flux d'ordres et de profondeur du carnet.

    Args:
        market_data (pd.DataFrame): DataFrame avec données OHLCV
        order_flow_data (pd.DataFrame): DataFrame avec données de flux d'ordres
        order_book_data (pd.DataFrame, optional): DataFrame avec données de profondeur

    Returns:
        pd.DataFrame: DataFrame intégré
    """
    # S'assurer que tous les DataFrames ont un index datetime
    for df in [market_data, order_flow_data]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Tous les DataFrames doivent avoir un index DatetimeIndex")

    if order_book_data is not None and not isinstance(
        order_book_data.index, pd.DatetimeIndex
    ):
        raise ValueError("order_book_data doit avoir un index DatetimeIndex")

    # Réindexer les DataFrames sur l'index de market_data
    flow_reindexed = order_flow_data.reindex(market_data.index, method="nearest")

    # Fusionner les données de marché et de flux d'ordres
    result = pd.concat([market_data, flow_reindexed], axis=1)

    # Ajouter les données de profondeur du carnet si disponibles
    if order_book_data is not None:
        book_reindexed = order_book_data.reindex(market_data.index, method="nearest")

        # Éviter la duplication des colonnes
        book_columns = [
            col for col in book_reindexed.columns if col not in result.columns
        ]
        result = pd.concat([result, book_reindexed[book_columns]], axis=1)

    # Supprimer les colonnes dupliquées
    result = result.loc[:, ~result.columns.duplicated()]

    # Remplir les valeurs NaN
    result = result.bfill().ffill()

    return result


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)

    # Créer un collecteur de flux d'ordres
    order_flow_collector = OrderFlowCollector(use_synthetic=True)

    # Collecter des données de flux d'ordres
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    order_flow_data = order_flow_collector.collect_order_flow(
        symbol="BTC", start_date=start_date, end_date=end_date, interval="1d"
    )

    # Calculer des métriques supplémentaires
    enhanced_flow_data = order_flow_collector.calculate_flow_metrics(order_flow_data)

    # Afficher les premières lignes
    print("\nDonnées de flux d'ordres:")
    print(enhanced_flow_data.head())

    # Collecter des données de profondeur du carnet
    order_book_collector = OrderBookDepthCollector(use_synthetic=True)

    order_book_data = order_book_collector.collect_order_book_depth(
        symbol="BTC",
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        depth_levels=5,
    )

    # Calculer des métriques de profondeur
    enhanced_book_data = order_book_collector.calculate_depth_metrics(order_book_data)

    print("\nDonnées de profondeur du carnet:")
    print(enhanced_book_data.head())
