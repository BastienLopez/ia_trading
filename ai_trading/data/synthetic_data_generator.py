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
    depreciation_rate=0.0001,  # Taux de dépréciation quotidien
    depreciation_start=0.5,  # Point de départ de la dépréciation (0-1)
    depreciation_type="exponential",  # Type de dépréciation: "exponential", "linear", "step"
    depreciation_steps=None,  # Points de dépréciation pour le type "step"
    min_price_ratio=0.1,  # Prix minimum en ratio du prix initial
):
    """
    Génère des données de marché synthétiques pour les tests.
    Optimisé pour la performance et l'utilisation de la mémoire.
    """
    logger.info(f"Génération de {n_points} points de données synthétiques")

    # Utilisation de float16 pour éviter les problèmes de précision
    dtype = np.float16

    # Initialisation des tableaux numpy optimisés
    time_steps = np.arange(n_points, dtype=dtype)

    # Génération de la tendance de base avec contrôle des valeurs
    trend_component = np.clip(trend * time_steps, -0.1, 0.1)

    # Génération des cycles avec contrôle des valeurs
    if cyclic_pattern:
        cycles = np.clip(
            0.04 * np.sin(time_steps / 20)  # Cycle court
            + 0.08 * np.sin(time_steps / 50)  # Cycle moyen
            + 0.12 * np.sin(time_steps / 200),  # Cycle long
            -0.2,
            0.2,
        )
    else:
        cycles = np.zeros(n_points, dtype=dtype)

    # Génération du bruit aléatoire avec contrôle des valeurs
    noise = np.clip(np.random.normal(0, volatility, n_points), -0.1, 0.1).astype(dtype)

    # Calcul des changements de prix avec contrôle des valeurs
    price_changes = np.clip(trend_component + cycles + noise, -0.2, 0.2)

    # Calcul des prix de clôture avec contrôle des valeurs en utilisant float32 pour éviter l'overflow
    # Convertir en float32 pour le calcul des prix cumulés
    price_changes_f32 = price_changes.astype(np.float32)
    # Utiliser une approche plus stable pour calculer les prix cumulés
    log_returns = np.log1p(price_changes_f32)
    cumulative_returns = np.cumsum(log_returns)
    close_prices = start_price * np.exp(cumulative_returns)
    
    # Vérifier les valeurs avant la conversion
    close_prices = np.clip(close_prices, np.finfo(dtype).min, np.finfo(dtype).max)
    
    # Reconvertir en float16 si nécessaire après le calcul
    try:
        close_prices = close_prices.astype(dtype, casting='safe')
    except TypeError:
        # Si le casting sécurisé échoue, utiliser une approche plus conservative
        close_prices = np.clip(close_prices, -65000, 65000).astype(dtype)
        
    close_prices = np.maximum(close_prices, start_price * min_price_ratio)

    # Application de la dépréciation
    depreciation_point = int(n_points * depreciation_start)
    if depreciation_point < n_points:
        depreciation_factor = np.ones(n_points, dtype=dtype)

        if depreciation_type == "exponential":
            # Dépréciation exponentielle avec contrôle des valeurs
            decay = np.exp(
                -depreciation_rate * np.arange(n_points - depreciation_point)
            )
            depreciation_factor[depreciation_point:] = np.clip(decay, 0.1, 1.0)
        elif depreciation_type == "linear":
            # Dépréciation linéaire avec contrôle des valeurs
            linear_decay = 1 - (
                depreciation_rate * np.arange(n_points - depreciation_point)
            )
            depreciation_factor[depreciation_point:] = np.clip(linear_decay, 0.1, 1.0)
        elif depreciation_type == "step" and depreciation_steps:
            # Dépréciation par paliers avec contrôle des valeurs
            for step in sorted(depreciation_steps):
                step_point = int(n_points * step)
                if step_point > depreciation_point:
                    decay = 1 - depreciation_rate
                    depreciation_factor[step_point:] *= np.clip(decay, 0.1, 1.0)

        # Application du facteur de dépréciation avec contrôle des valeurs
        close_prices = close_prices * depreciation_factor
        close_prices = np.maximum(close_prices, start_price * min_price_ratio)

    # Génération des prix OHLC avec contrôle des valeurs
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price

    # Génération des prix high/low avec une volatilité contrôlée
    high_prices = close_prices * (
        1 + np.clip(np.random.uniform(0, volatility, n_points), 0, 0.1)
    )
    low_prices = close_prices * (
        1 - np.clip(np.random.uniform(0, volatility, n_points), 0, 0.1)
    )

    # Génération du volume avec prise en compte de la dépréciation
    if include_volume:
        base_volume = start_price * 10
        price_changes_abs = np.abs(price_changes)
        volume = base_volume * (1 + np.clip(price_changes_abs * 10, 0, 0.5))
        volume = volume * np.clip(np.random.uniform(0.8, 1.2, n_points), 0.5, 1.5)
        # Ajustement du volume en fonction de la dépréciation
        if depreciation_point < n_points:
            volume[depreciation_point:] = (
                volume[depreciation_point:] * depreciation_factor[depreciation_point:]
            )
            # Assurer un volume minimum
            min_volume = base_volume * 0.1
            volume = np.maximum(volume, min_volume)
    else:
        volume = np.zeros(n_points)

    # Création de l'index de dates
    if with_date:
        if start_date is None:
            start_date = datetime.now() - timedelta(days=n_points)
        dates = pd.date_range(start=start_date, periods=n_points, freq="D")
    else:
        dates = np.arange(n_points)

    # Création du DataFrame optimisé
    df = pd.DataFrame(
        {
            "open": np.clip(open_prices, -65000, 65000).astype(dtype),
            "high": np.clip(high_prices, -65000, 65000).astype(dtype),
            "low": np.clip(low_prices, -65000, 65000).astype(dtype),
            "close": np.clip(close_prices, -65000, 65000).astype(dtype),
            "volume": np.clip(volume, 0, 65000).astype(dtype),
        },
        index=dates,
    )

    # Ajout des colonnes pour suivre la dépréciation
    df["depreciation_factor"] = np.clip(depreciation_factor, 0, 1).astype(dtype)
    df["price_ratio"] = np.clip(df["close"] / start_price, 0, 65000).astype(dtype)
    df["volume_ratio"] = (
        np.clip(df["volume"] / base_volume, 0, 65000).astype(dtype) if include_volume else 0
    )

    logger.info(f"Données synthétiques générées avec succès: {len(df)} points")
    return df


def generate_market_with_indicators(n_points=1000, include_technical=True):
    """
    Génère des données de marché synthétiques avec des indicateurs techniques.
    Optimisé pour la performance.
    """
    # Génération des données de base
    df = generate_synthetic_market_data(n_points=n_points)

    if include_technical:
        # Utilisation de float16 pour les indicateurs
        dtype = np.float16

        # Calcul des rendements
        returns = df["close"].pct_change().astype(dtype)

        # Moyennes mobiles optimisées
        windows = [10, 20, 50]
        for window in windows:
            df[f"sma_{window}"] = (
                df["close"].rolling(window=window).mean().astype(dtype)
            )
            df[f"ema_{window}"] = (
                df["close"].ewm(span=window, adjust=False).mean().astype(dtype)
            )

        # RSI optimisé
        up = np.maximum(returns, 0)
        down = np.abs(np.minimum(returns, 0))
        avg_gain = up.rolling(window=14).mean()
        avg_loss = down.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = (100 - (100 / (1 + rs))).astype(dtype)

        # MACD optimisé
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = (ema_12 - ema_26).astype(dtype)
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean().astype(dtype)

        # Bandes de Bollinger optimisées
        df["bb_middle"] = df["close"].rolling(window=20).mean().astype(dtype)
        std = df["close"].rolling(window=20).std().astype(dtype)
        df["bb_upper"] = (df["bb_middle"] + (std * 2)).astype(dtype)
        df["bb_lower"] = (df["bb_middle"] - (std * 2)).astype(dtype)

        # ATR optimisé
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df["atr"] = true_range.rolling(14).mean().astype(dtype)

        # Volume profile optimisé
        df["volume_sma"] = df["volume"].rolling(window=20).mean().astype(dtype)

        # Sentiment simulé
        df["sentiment"] = np.random.normal(0, 0.3, len(df)).astype(dtype)
        df["sentiment"] = df["sentiment"].clip(-1, 1)

        # Remplissage des valeurs manquantes optimisé
        df = df.fillna(df.shift())
        df = df.fillna(df.shift(-1))

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
