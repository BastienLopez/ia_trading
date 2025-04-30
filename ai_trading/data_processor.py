import gc
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import ccxt
import numpy as np
import pandas as pd
import ta

from ai_trading.config import EMA_RIBBON_PERIODS

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai_trading.data_processor")


def optimize_memory():
    """
    Optimise l'utilisation de la mémoire en forçant le garbage collection.
    """
    # Forcer le garbage collection Python
    collected = gc.collect()
    logger.debug(f"Objets collectés par GC: {collected}")

    # Libérer le cache CUDA si disponible
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cache CUDA vidé")
    except ImportError:
        pass  # PyTorch non disponible


class DataProcessor:
    """
    Classe pour le traitement des données de trading
    """

    def __init__(
        self,
        data_dir=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "ai_trading/info_retour/data"
        ),
        use_float16=True,  # Utiliser float16 par défaut pour économiser la mémoire
        use_cache=True,
    ):
        self.data_dir = Path(data_dir)
        self.use_float16 = use_float16
        self.use_cache = use_cache
        self.data = {}
        self.cache = {}

        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(self.data_dir, exist_ok=True)

        logger.info(f"DataProcessor initialisé. Utilisation de float16: {use_float16}")

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

    def preprocess_data(
        self, data: pd.DataFrame, add_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Prétraite les données en ajoutant des indicateurs techniques et en gérant les valeurs manquantes.

        Args:
            data: DataFrame contenant les données OHLCV
            add_indicators: Ajouter des indicateurs techniques

        Returns:
            DataFrame prétraité
        """
        # Copie pour éviter de modifier l'original
        df = data.copy()

        # Vérifier et gérer les valeurs manquantes
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Valeurs manquantes détectées: {df.isnull().sum()}")
            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)

        # Ajouter des indicateurs techniques si demandé
        if add_indicators:
            try:
                from ai_trading.indicators.technical_indicators import (
                    add_technical_indicators,
                )

                df = add_technical_indicators(df)

                # Filtrer les lignes avec des valeurs NaN (début de la série temporelle)
                df.dropna(inplace=True)

                # Convertir en float16 si demandé
                if self.use_float16:
                    for col in df.select_dtypes(include=["float64"]).columns:
                        df[col] = df[col].astype(np.float16)

            except Exception as e:
                logger.error(f"Erreur lors de l'ajout des indicateurs: {e}")

        # Nettoyer la mémoire après le prétraitement
        optimize_memory()

        return df

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        target_column: str = "close",
        feature_columns: List[str] = None,
        n_steps_ahead: int = 1,
        train_ratio: float = 0.8,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Prépare les données en séquences pour l'apprentissage.

        Args:
            data: DataFrame prétraité
            sequence_length: Longueur des séquences
            target_column: Colonne à prédire
            feature_columns: Colonnes à utiliser comme features
            n_steps_ahead: Nombre de pas de temps à prédire
            train_ratio: Ratio de données pour l'entraînement
            normalize: Normaliser les données

        Returns:
            Dict contenant les séquences d'entraînement et de test
        """
        # Utiliser toutes les colonnes sauf le target si feature_columns n'est pas spécifié
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]

        # Extraire les features et la cible
        features = data[feature_columns].values
        target = data[target_column].values.reshape(-1, 1)

        # Normaliser si demandé
        if normalize:
            # Calculer les statistiques sur l'ensemble d'entraînement uniquement
            train_size = int(len(features) * train_ratio)

            # Normaliser les features
            feature_mean = features[:train_size].mean(axis=0)
            feature_std = features[:train_size].std(axis=0)
            feature_std[feature_std == 0] = 1  # Éviter division par zéro
            normalized_features = (features - feature_mean) / feature_std

            # Normaliser la cible
            target_mean = target[:train_size].mean(axis=0)
            target_std = target[:train_size].std(axis=0)
            target_std[target_std == 0] = 1  # Éviter division par zéro
            normalized_target = (target - target_mean) / target_std

            # Utiliser les données normalisées
            features = normalized_features
            target = normalized_target

            # Sauvegarder les statistiques pour la dénormalisation ultérieure
            self.normalization_stats = {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "target_mean": target_mean,
                "target_std": target_std,
            }

        # Créer les séquences
        X, y = [], []
        for i in range(len(features) - sequence_length - n_steps_ahead + 1):
            X.append(features[i : (i + sequence_length)])
            y.append(target[i + sequence_length + n_steps_ahead - 1])

        # Convertir en tableaux numpy
        X = np.array(X)
        y = np.array(y)

        # Convertir en float16 si demandé
        if self.use_float16:
            X = X.astype(np.float16)
            y = y.astype(np.float16)

        # Diviser en ensembles d'entraînement et de test
        train_size = int(len(X) * train_ratio)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Nettoyer la mémoire après la préparation
        optimize_memory()

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_columns,
            "sequence_length": sequence_length,
            "normalization_stats": self.normalization_stats if normalize else None,
        }

    def save_processed_data(self, data_dict: Dict, filename: str):
        """
        Sauvegarde les données prétraitées pour une utilisation ultérieure.

        Args:
            data_dict: Dictionnaire contenant les données prétraitées
            filename: Nom du fichier de sauvegarde
        """
        save_path = self.data_dir / f"{filename}.npz"

        # Sauvegarder les tableaux numpy
        np.savez(
            save_path,
            X_train=data_dict["X_train"],
            y_train=data_dict["y_train"],
            X_test=data_dict["X_test"],
            y_test=data_dict["y_test"],
        )

        # Sauvegarder les métadonnées
        metadata = {
            "feature_columns": data_dict["feature_columns"],
            "sequence_length": data_dict["sequence_length"],
            "normalization_stats": data_dict["normalization_stats"],
        }

        with open(self.data_dir / f"{filename}_metadata.json", "w") as f:
            json.dump(
                metadata,
                f,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
            )

        logger.info(f"Données prétraitées sauvegardées dans {save_path}")

    def load_processed_data(self, filename: str) -> Dict:
        """
        Charge les données prétraitées précédemment sauvegardées.

        Args:
            filename: Nom du fichier à charger

        Returns:
            Dictionnaire contenant les données prétraitées
        """
        try:
            # Charger les tableaux numpy
            data_path = self.data_dir / f"{filename}.npz"
            data = np.load(data_path)

            # Charger les métadonnées
            with open(self.data_dir / f"{filename}_metadata.json", "r") as f:
                metadata = json.load(f)

            # Reconstituer le dictionnaire
            result = {
                "X_train": data["X_train"],
                "y_train": data["y_train"],
                "X_test": data["X_test"],
                "y_test": data["y_test"],
                "feature_columns": metadata["feature_columns"],
                "sequence_length": metadata["sequence_length"],
                "normalization_stats": metadata["normalization_stats"],
            }

            # Convertir en float16 si demandé
            if self.use_float16:
                result["X_train"] = result["X_train"].astype(np.float16)
                result["y_train"] = result["y_train"].astype(np.float16)
                result["X_test"] = result["X_test"].astype(np.float16)
                result["y_test"] = result["y_test"].astype(np.float16)

            logger.info(f"Données prétraitées chargées depuis {data_path}")

            # Nettoyer la mémoire après le chargement
            optimize_memory()

            return result

        except Exception as e:
            logger.error(f"Erreur lors du chargement des données prétraitées: {e}")
            raise
