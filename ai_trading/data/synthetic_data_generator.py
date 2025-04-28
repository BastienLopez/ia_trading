"""
Module pour générer des données de marché synthétiques à des fins de test.
Ce module fournit des fonctions pour créer des datasets synthétiques
qui simulent le comportement des marchés financiers.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Configurer le logger
logger = logging.getLogger(__name__)


def generate_synthetic_market_data(
    n_points=1000,
    trend=0.0001,
    volatility=0.01,
    start_price=100.0,
    start_date=None,
    with_date=True,
    include_volume=True,
    cyclic_pattern=True,
):
    """
    Génère des données de marché synthétiques pour les tests.

    Args:
        n_points (int): Nombre de points de données à générer
        trend (float): Coefficient de tendance (positif pour haussier, négatif pour baissier)
        volatility (float): Degré de volatilité du prix
        start_price (float): Prix initial
        start_date (datetime): Date de début (si None, utilise la date du jour)
        with_date (bool): Si True, utilise un DatetimeIndex
        include_volume (bool): Si True, inclut des données de volume
        cyclic_pattern (bool): Si True, ajoute un motif cyclique aux prix

    Returns:
        pd.DataFrame: DataFrame contenant les données OHLCV générées
    """
    logger.info(f"Génération de {n_points} points de données synthétiques")

    # Initialiser le prix
    price = start_price

    # Liste pour stocker les prix
    prices = []

    # Générer une tendance de base
    time_steps = np.arange(n_points)
    trend_component = trend * time_steps

    # Ajouter un motif cyclique si demandé
    if cyclic_pattern:
        # Ajouter plusieurs cycles de différentes fréquences
        short_cycle = 0.04 * np.sin(time_steps / 20)  # Cycle court
        medium_cycle = 0.08 * np.sin(time_steps / 50)  # Cycle moyen
        long_cycle = 0.12 * np.sin(time_steps / 200)  # Cycle long

        cycles = short_cycle + medium_cycle + long_cycle
    else:
        cycles = np.zeros(n_points)

    # Générer les prix
    close_prices = []
    for i in range(n_points):
        # Ajouter du bruit aléatoire
        noise = np.random.normal(0, volatility)

        # Calculer le prochain prix avec tendance, cycles et bruit
        price_change = trend_component[i] + cycles[i] + noise

        # Assurer que le prix ne tombe pas en dessous d'un seuil minimal
        price = max(price * (1 + price_change), start_price * 0.1)

        close_prices.append(price)

    # Créer les données OHLCV
    open_prices = [close_prices[0]] + close_prices[:-1]
    high_prices = [p * (1 + np.random.uniform(0, volatility)) for p in close_prices]
    low_prices = [p * (1 - np.random.uniform(0, volatility)) for p in close_prices]

    # Créer un volume avec une corrélation positive au mouvement des prix
    if include_volume:
        # Volume de base
        base_volume = start_price * 10

        # Calculer les changements de prix
        price_changes = np.diff(np.array(close_prices), prepend=close_prices[0])

        # Générer le volume corrélé aux mouvements de prix
        # Plus le mouvement est important, plus le volume est élevé
        volume = base_volume * (1 + np.abs(price_changes) * 10)

        # Ajouter du bruit au volume
        volume = volume * np.random.uniform(0.8, 1.2, n_points)
    else:
        volume = np.zeros(n_points)

    # Créer l'index de dates si demandé
    if with_date:
        if start_date is None:
            start_date = datetime.now() - timedelta(days=n_points)

        dates = [start_date + timedelta(days=i) for i in range(n_points)]
        index = pd.DatetimeIndex(dates)
    else:
        index = np.arange(n_points)

    # Créer le DataFrame
    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=index,
    )

    logger.info(f"Données synthétiques générées avec succès: {len(df)} points")
    return df


def generate_market_with_indicators(n_points=1000, include_technical=True):
    """
    Génère des données de marché synthétiques avec des indicateurs techniques.

    Args:
        n_points (int): Nombre de points de données à générer
        include_technical (bool): Si True, ajoute des indicateurs techniques

    Returns:
        pd.DataFrame: DataFrame avec données OHLCV et indicateurs
    """
    # Générer les données de base
    df = generate_synthetic_market_data(n_points=n_points)

    if include_technical:
        # Ajouter des indicateurs techniques synthétiques

        # Moyennes mobiles
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

        # RSI simplifié (simulé)
        returns = df["close"].pct_change()
        up = returns.copy()
        up[up < 0] = 0
        down = -returns.copy()
        down[down < 0] = 0

        avg_gain = up.rolling(window=14).mean()
        avg_loss = down.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD simplifié
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Bandes de Bollinger simplifiées
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (std * 2)
        df["bb_lower"] = df["bb_middle"] - (std * 2)

        # ATR simplifié
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(14).mean()

        # Volume profile (simulé)
        df["volume_sma"] = df["volume"].rolling(window=20).mean()

        # Simuler un sentiment (entre -1 et 1)
        df["sentiment"] = np.random.normal(0, 0.3, len(df))
        df["sentiment"] = df["sentiment"].clip(-1, 1)

        # Corriger les NaN
        df = df.fillna(method="bfill")

    return df


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)

    # Générer et afficher un exemple de données
    data = generate_synthetic_market_data(n_points=100)
    print(data.head())

    # Générer des données avec indicateurs
    data_with_indicators = generate_market_with_indicators(n_points=100)
    print("\nDonnées avec indicateurs:")
    print(data_with_indicators.head())
