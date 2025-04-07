import pandas as pd
import numpy as np
import ccxt
import ta
from datetime import datetime, timedelta
import os
from ai_trading.config import EMA_RIBBON_PERIODS


class DataProcessor:
    """
    Classe pour le traitement des données de trading
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_historical_data(
        self,
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=None,
        end_date=None,
        save=True,
    ):
        """
        Télécharge les données historiques depuis un exchange
        """
        print(f"Téléchargement des données {symbol} depuis {exchange_id}...")

        # Configuration de l'exchange
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(
            {
                "enableRateLimit": True,
            }
        )

        # Dates par défaut si non spécifiées
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # 30 jours par défaut

        # Convertir les dates en timestamps
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)

        # Télécharger les données
        all_candles = []
        while since < until:
            print(
                f"Téléchargement des données depuis {datetime.fromtimestamp(since/1000)}"
            )
            candles = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not candles:
                break

            all_candles.extend(candles)
            since = candles[-1][0] + 1

        # Convertir en DataFrame
        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Sauvegarder les données si demandé
        if save:
            filename = f"{self.data_dir}/{exchange_id}_{symbol.replace('/', '_')}_{timeframe}.csv"
            df.to_csv(filename)
            print(f"Données sauvegardées dans {filename}")

        return df

    def load_data(self, filepath):
        """
        Charge les données depuis un fichier CSV
        """
        df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques au DataFrame."""
        required_columns = {"open", "high", "low", "close", "volume"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes pour les indicateurs: {missing}")

        print("Ajout des indicateurs techniques...")

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["signal_line"] = macd.macd_signal()
        df["hist_line"] = macd.macd_diff()

        # EMAs
        df["ema9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df["high"], df["low"])
        df["tenkan"] = ichimoku.ichimoku_conversion_line()
        df["kijun"] = ichimoku.ichimoku_base_line()
        df["senkou_span_a"] = ichimoku.ichimoku_a()
        df["senkou_span_b"] = ichimoku.ichimoku_b()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bollinger.bollinger_hband()
        df["bb_middle"] = bollinger.bollinger_mavg()
        df["bb_lower"] = bollinger.bollinger_lband()

        # Volume ratio (normalized)
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).max()

        # Supprimer les lignes avec des NaN
        df.dropna(inplace=True)

        return df

    def preprocess_for_training(self, df, train_ratio=0.8):
        """
        Prépare les données pour l'entraînement et le test
        """
        # S'assurer que tous les indicateurs ont été ajoutés
        if "rsi" not in df.columns:
            df = self.add_indicators(df)

        # Normaliser certaines colonnes pour améliorer l'entraînement
        for col in ["open", "high", "low", "close"]:
            mean = df[col].mean()
            std = df[col].std()
            df[f"{col}_norm"] = (df[col] - mean) / std

        # Diviser les données en ensembles d'entraînement et de test
        split_idx = int(len(df) * train_ratio)
        train_data = df.iloc[:split_idx].copy()
        test_data = df.iloc[split_idx:].copy()

        print(
            f"Données divisées: {len(train_data)} pour l'entraînement, {len(test_data)} pour le test"
        )

        return train_data, test_data

    def prepare_backtesting_data(self, df):
        """
        Prépare les données pour le backtesting
        """
        # S'assurer que tous les indicateurs ont été ajoutés
        if "rsi" not in df.columns:
            df = self.add_indicators(df)

        return df

    def add_ema_features(
        self, df: pd.DataFrame, periods=EMA_RIBBON_PERIODS
    ) -> pd.DataFrame:
        """Ajoute les EMA configurées au DataFrame."""
        for period in periods:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # Calcul de la largeur du ruban EMA (différence entre EMA court et long)
        df["ema_ribbon_width"] = df["ema_5"] - df["ema_50"]

        # Calcul du gradient entre EMA5 et EMA30
        df["ema_gradient"] = (df["ema_5"] - df["ema_30"]) / df["ema_30"]
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processus complet de traitement des données."""
        df = self.add_indicators(df)
        df = self.add_ema_features(df)
        return df
