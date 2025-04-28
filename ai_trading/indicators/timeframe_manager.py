from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd


class TimeframeManager:
    def __init__(self, timeframes: List[str]):
        """
        Initialise le gestionnaire de timeframes

        Args:
            timeframes: Liste des timeframes à gérer (ex: ['1m', '4h', '1d'])
        """
        self.timeframes = timeframes
        self.data: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}

    def _parse_timeframe(self, tf: str) -> str:
        """Convertit un string timeframe en format pandas"""
        unit = tf[-1]
        value = tf[:-1]

        if unit == "m":
            return f"{value}min"  # Minutes
        elif unit == "h":
            return f"{value}h"  # Heures
        elif unit == "d":
            return f"{value}d"  # Jours
        else:
            raise ValueError(f"Timeframe non supporté: {tf}")

    def update_data(self, timeframe: str, df: pd.DataFrame):
        """
        Met à jour les données pour un timeframe donné

        Args:
            timeframe: Le timeframe à mettre à jour
            df: DataFrame avec colonnes [open, high, low, close, volume]
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe non supporté: {timeframe}")

        # Vérifie le format des données
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame doit contenir les colonnes: {required_cols}")

        # S'assure que l'index est en datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index)

        # Convertit en période temporelle spécifiée
        period = self._parse_timeframe(timeframe)
        resampled = (
            df.resample(period)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        self.data[timeframe] = resampled
        self.last_update[timeframe] = datetime.now()

    def get_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données pour un timeframe donné

        Args:
            timeframe: Le timeframe demandé

        Returns:
            DataFrame avec les données ou None si pas de données
        """
        return self.data.get(timeframe)

    def is_data_fresh(self, timeframe: str, max_age: timedelta) -> bool:
        """
        Vérifie si les données d'un timeframe sont récentes

        Args:
            timeframe: Le timeframe à vérifier
            max_age: L'âge maximum accepté des données

        Returns:
            bool: True si les données sont récentes
        """
        if timeframe not in self.last_update:
            return False

        age = datetime.now() - self.last_update[timeframe]
        return age <= max_age

    def update_all_timeframes(self, base_df: pd.DataFrame):
        """Met à jour les données pour tous les timeframes"""
        for tf in self.timeframes:
            self.data[tf] = self.resample_data(base_df, tf)

    def get_current_candle(self, timeframe: str) -> pd.Series:
        """Récupère la dernière bougie pour un timeframe donné"""
        df = self.get_data(timeframe)
        return df.iloc[-1]

    def get_timeframe_minutes(self, timeframe: str) -> int:
        """Convertit un timeframe en minutes"""
        units = {"m": 1, "h": 60, "d": 1440, "w": 10080}

        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit not in units:
            raise ValueError(f"Unité de temps invalide: {unit}")

        return value * units[unit]

    def get_higher_timeframe(self, current_tf: str) -> str:
        """Trouve le timeframe supérieur le plus proche"""
        current_minutes = self.get_timeframe_minutes(current_tf)
        higher_tfs = [
            tf
            for tf in self.timeframes
            if self.get_timeframe_minutes(tf) > current_minutes
        ]

        if not higher_tfs:
            return None

        return min(higher_tfs, key=lambda x: self.get_timeframe_minutes(x))

    def get_lower_timeframe(self, current_tf: str) -> str:
        """Trouve le timeframe inférieur le plus proche"""
        current_minutes = self.get_timeframe_minutes(current_tf)
        lower_tfs = [
            tf
            for tf in self.timeframes
            if self.get_timeframe_minutes(tf) < current_minutes
        ]

        if not lower_tfs:
            return None

        return max(lower_tfs, key=lambda x: self.get_timeframe_minutes(x))

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Reéchantillonne les données pour un timeframe spécifique"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        resampled = df.resample(timeframe).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        return resampled.dropna()
