from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


@dataclass
class BreakEvent:
    timestamp: datetime
    price: float
    direction: str  # 'up' ou 'down'
    volume: float
    strength: float  # Force de la cassure basée sur le volume et la volatilité


@dataclass
class RetestEvent:
    timestamp: datetime
    price: float
    success: bool  # True si le retest a réussi (le niveau a tenu)
    strength: float  # Force du retest


class BreakRetestDetector:
    def __init__(
        self,
        volume_threshold: float = 1.5,  # Seuil de volume pour une cassure valide
        volatility_window: int = 20,  # Fenêtre pour calculer la volatilité
        retest_tolerance: float = 0.002,  # Tolérance pour les retests (0.2%)
    ):
        self.volume_threshold = volume_threshold
        self.volatility_window = volatility_window
        self.retest_tolerance = retest_tolerance

    def calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calcule la volatilité sur une fenêtre donnée"""
        return df["close"].rolling(window=self.volatility_window).std()

    def is_valid_break(
        self,
        candle: pd.Series,
        prev_candle: pd.Series,
        level: float,
        avg_volume: float,
        volatility: float,
    ) -> Tuple[bool, str]:
        """Vérifie si une cassure est valide"""
        # Calcul de la tolérance basée sur la volatilité
        tolerance = level * self.retest_tolerance

        # Volume suffisant
        if candle["volume"] < avg_volume * self.volume_threshold:
            print(
                f"Volume insuffisant: {candle['volume']} < {avg_volume * self.volume_threshold}"
            )
            return False, ""

        # Cassure haussière
        if prev_candle["close"] < level and candle["close"] > level + tolerance:
            print(
                f"Cassure haussière détectée: prev_close={prev_candle['close']}, close={candle['close']}, level={level}"
            )
            return True, "up"

        # Cassure baissière
        if prev_candle["close"] > level and candle["close"] < level - tolerance:
            print(
                f"Cassure baissière détectée: prev_close={prev_candle['close']}, close={candle['close']}, level={level}"
            )
            return True, "down"

        print(
            f"Pas de cassure: prev_close={prev_candle['close']}, close={candle['close']}, level={level}, tolerance={tolerance}"
        )
        return False, ""

    def calculate_break_strength(
        self, candle: pd.Series, level: float, avg_volume: float, volatility: float
    ) -> float:
        """Calcule la force d'une cassure"""
        # Force basée sur le volume
        volume_factor = min(
            candle["volume"] / (avg_volume * self.volume_threshold), 2.0
        )

        # Force basée sur le mouvement de prix
        price_move = abs(candle["close"] - level) / (level * self.retest_tolerance)
        price_factor = min(price_move, 2.0)

        # Force basée sur la volatilité
        volatility_factor = abs(candle["close"] - candle["open"]) / volatility

        return (volume_factor + price_factor + volatility_factor) / 3

    def detect_break(self, df: pd.DataFrame, level: float) -> Optional[BreakEvent]:
        """Détecte une cassure d'un niveau"""
        # Calcul des moyennes mobiles
        avg_volume = df["volume"].rolling(window=20).mean()
        volatility = self.calculate_volatility(df)

        # Parcours des bougies
        for i in range(20, len(df)):  # Commence après la période de calcul des moyennes
            candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]

            # Vérifie si le prix était d'un côté du niveau et a traversé de l'autre côté
            if (
                prev_candle["close"] <= level and candle["close"] > level
            ) or (  # Cassure haussière
                prev_candle["close"] >= level and candle["close"] < level
            ):  # Cassure baissière
                print(f"\nAnalyse de la bougie {i}:")
                print(f"Prix précédent: {prev_candle['close']}")
                print(f"Prix actuel: {candle['close']}")
                print(f"Niveau: {level}")

                is_break, direction = self.is_valid_break(
                    candle, prev_candle, level, avg_volume.iloc[i], volatility.iloc[i]
                )

                if is_break:
                    strength = self.calculate_break_strength(
                        candle, level, avg_volume.iloc[i], volatility.iloc[i]
                    )
                    print(f"Force de la cassure: {strength}")
                    return BreakEvent(
                        timestamp=df.index[i],
                        price=candle["close"],
                        direction=direction,
                        volume=candle["volume"],
                        strength=strength,
                    )

        print("\nAucune cassure détectée dans toute la série")
        return None

    def is_valid_retest(
        self, candle: pd.Series, level: float, break_direction: str
    ) -> bool:
        """Vérifie si un retest est valide"""
        tolerance = level * self.retest_tolerance

        if break_direction == "up":
            return candle["low"] >= level - tolerance and candle["close"] > level
        else:
            return candle["high"] <= level + tolerance and candle["close"] < level

    def calculate_retest_strength(
        self, candle: pd.Series, level: float, volatility: float
    ) -> float:
        """Calcule la force d'un retest"""
        price_precision = 1 - (abs(candle["close"] - level) / level)
        momentum = abs(candle["close"] - candle["open"]) / volatility
        return (price_precision + momentum) / 2

    def detect_retest(
        self, df: pd.DataFrame, level: float, break_event: BreakEvent
    ) -> Optional[RetestEvent]:
        """Détecte un retest après une cassure"""
        # On ne regarde que les données après la cassure
        post_break_df = df[df.index > break_event.timestamp]
        if len(post_break_df) == 0:
            print("Pas de données après la cassure")
            return None

        volatility = self.calculate_volatility(post_break_df)
        tolerance = level * self.retest_tolerance

        for i, (idx, candle) in enumerate(post_break_df.iterrows()):
            print(f"\nAnalyse du retest - bougie {idx}:")
            print(
                f"Prix: close={candle['close']}, high={candle['high']}, low={candle['low']}"
            )
            print(f"Niveau: {level}, Tolérance: {tolerance}")

            # Pour une cassure haussière, on cherche un retest par le haut
            if break_event.direction == "up":
                if candle["low"] <= level + tolerance and candle["close"] > level:
                    print("Retest haussier détecté")
                    strength = self.calculate_retest_strength(
                        candle, level, volatility.iloc[i]
                    )
                    print(f"Force du retest: {strength}")
                    return RetestEvent(
                        timestamp=idx,
                        price=candle["close"],
                        success=True,
                        strength=strength,
                    )

            # Pour une cassure baissière, on cherche un retest par le bas
            else:
                if candle["high"] >= level - tolerance and candle["close"] < level:
                    print("Retest baissier détecté")
                    strength = self.calculate_retest_strength(
                        candle, level, volatility.iloc[i]
                    )
                    print(f"Force du retest: {strength}")
                    return RetestEvent(
                        timestamp=idx,
                        price=candle["close"],
                        success=True,
                        strength=strength,
                    )

        print("Aucun retest détecté")
        return None
