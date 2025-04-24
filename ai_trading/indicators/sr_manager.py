from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from .timeframe_manager import TimeframeManager


@dataclass
class SRLevel:
    price: float
    type: str  # 'support' ou 'resistance'
    strength: float
    timeframe: str
    creation_time: datetime
    breaks: List[datetime]
    retests: List[datetime]
    active: bool = True


class SRManager:
    def __init__(
        self,
        timeframes: List[str],
        max_levels: int = 10,
        strength_threshold: float = 0.7,
        merge_tolerance: float = 0.002,
    ):
        """
        Initialise le gestionnaire de niveaux S/R

        Args:
            timeframes: Liste des timeframes à analyser
            max_levels: Nombre maximum de niveaux à maintenir
            strength_threshold: Seuil minimum de force pour un niveau
            merge_tolerance: Tolérance pour fusionner des niveaux proches (en %)
        """
        self.tf_manager = TimeframeManager(timeframes)
        self.max_levels = max_levels
        self.strength_threshold = strength_threshold
        self.merge_tolerance = merge_tolerance
        self.levels: Dict[str, List[SRLevel]] = {tf: [] for tf in timeframes}

    def calculate_pivot_points(self, df: pd.DataFrame, window: int = 5) -> tuple:
        """
        Calcule les points pivots (hauts et bas)

        Args:
            df: DataFrame avec OHLCV
            window: Taille de la fenêtre pour les pivots

        Returns:
            tuple: (supports, resistances)
        """
        # Calcul des pivots hauts
        highs = (
            df["high"]
            .rolling(window=window, center=True)
            .apply(
                lambda x: (
                    x.iloc[len(x) // 2]
                    if len(x) > 0
                    and all(x.iloc[len(x) // 2] >= x.iloc[: len(x) // 2])
                    and all(x.iloc[len(x) // 2] >= x.iloc[len(x) // 2 + 1 :])
                    else np.nan
                )
            )
        )

        # Calcul des pivots bas
        lows = (
            df["low"]
            .rolling(window=window, center=True)
            .apply(
                lambda x: (
                    x.iloc[len(x) // 2]
                    if len(x) > 0
                    and all(x.iloc[len(x) // 2] <= x.iloc[: len(x) // 2])
                    and all(x.iloc[len(x) // 2] <= x.iloc[len(x) // 2 + 1 :])
                    else np.nan
                )
            )
        )

        # Extraction des valeurs non-NaN
        supports = df["low"][~pd.isna(lows)].values
        resistances = df["high"][~pd.isna(highs)].values

        return supports, resistances

    def calculate_level_strength(self, price: float, df: pd.DataFrame) -> float:
        """
        Calcule la force d'un niveau basée sur:
        - Nombre de touches
        - Volume aux touches
        - Distance temporelle

        Args:
            price: Prix du niveau
            df: DataFrame avec OHLCV

        Returns:
            float: Score de force entre 0 et 1
        """
        tolerance = price * self.merge_tolerance
        touches = (df["low"] >= price - tolerance) & (df["high"] <= price + tolerance)

        if not touches.any():
            return 0.0

        # Nombre de touches
        touch_count = touches.sum()
        touch_score = min(touch_count / 10, 1.0)  # Max 10 touches

        # Volume aux touches
        touch_volume = df[touches]["volume"].sum()
        total_volume = df["volume"].sum()
        volume_score = touch_volume / total_volume if total_volume > 0 else 0

        # Distance temporelle
        time_diffs = pd.Series(df.index[touches]).diff()
        time_score = 1.0 - (
            time_diffs.mean().total_seconds()
            / (df.index[-1] - df.index[0]).total_seconds()
        )

        # Score final
        return (touch_score + volume_score + time_score) / 3

    def merge_nearby_levels(self, levels: List[SRLevel]) -> List[SRLevel]:
        """
        Fusionne les niveaux proches

        Args:
            levels: Liste des niveaux à fusionner

        Returns:
            List[SRLevel]: Liste des niveaux fusionnés
        """
        if not levels:
            return []

        # Trie par prix
        sorted_levels = sorted(levels, key=lambda x: x.price)
        merged = []

        current = sorted_levels[0]
        for next_level in sorted_levels[1:]:
            # Si les niveaux sont proches
            if (
                next_level.price - current.price
            ) / current.price <= self.merge_tolerance:
                # Fusionne en gardant le plus fort
                if next_level.strength > current.strength:
                    current = next_level
            else:
                merged.append(current)
                current = next_level

        merged.append(current)
        return merged

    def update_levels(self, timeframe: str):
        """
        Met à jour les niveaux S/R pour un timeframe donné

        Args:
            timeframe: Le timeframe à mettre à jour
        """
        df = self.tf_manager.get_data(timeframe)
        if df is None:
            return

        # Calcule les nouveaux points pivots
        supports, resistances = self.calculate_pivot_points(df)

        # Crée les niveaux
        new_levels = []

        for price in supports:
            strength = self.calculate_level_strength(price, df)
            if strength >= self.strength_threshold:
                new_levels.append(
                    SRLevel(
                        price=price,
                        type="support",
                        strength=strength,
                        timeframe=timeframe,
                        creation_time=df.index[-1],
                        breaks=[],
                        retests=[],
                    )
                )

        for price in resistances:
            strength = self.calculate_level_strength(price, df)
            if strength >= self.strength_threshold:
                new_levels.append(
                    SRLevel(
                        price=price,
                        type="resistance",
                        strength=strength,
                        timeframe=timeframe,
                        creation_time=df.index[-1],
                        breaks=[],
                        retests=[],
                    )
                )

        # Fusionne avec les niveaux existants
        all_levels = self.levels[timeframe] + new_levels
        merged = self.merge_nearby_levels(all_levels)

        # Garde les plus forts
        self.levels[timeframe] = sorted(merged, key=lambda x: x.strength, reverse=True)[
            : self.max_levels
        ]

    def update_all_timeframes(self, df: pd.DataFrame):
        """
        Met à jour tous les timeframes avec de nouvelles données

        Args:
            df: DataFrame OHLCV de base
        """
        for tf in self.tf_manager.timeframes:
            self.tf_manager.update_data(tf, df)
            self.update_levels(tf)

    def get_active_levels(self, timeframe: str) -> List[SRLevel]:
        """
        Récupère les niveaux actifs pour un timeframe

        Args:
            timeframe: Le timeframe demandé

        Returns:
            List[SRLevel]: Liste des niveaux actifs
        """
        return [level for level in self.levels[timeframe] if level.active]

    def get_nearest_levels(self, price: float, timeframe: str, n: int = 2) -> tuple:
        """
        Trouve les n niveaux les plus proches au-dessus et en-dessous du prix

        Args:
            price: Prix actuel
            timeframe: Timeframe à utiliser
            n: Nombre de niveaux à retourner de chaque côté

        Returns:
            tuple: (supports en-dessous, résistances au-dessus)
        """
        active_levels = self.get_active_levels(timeframe)

        below = sorted(
            [l for l in active_levels if l.price < price], key=lambda x: price - x.price
        )[:n]
        above = sorted(
            [l for l in active_levels if l.price > price], key=lambda x: x.price - price
        )[:n]

        return below, above
