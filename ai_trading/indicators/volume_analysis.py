"""
Module d'analyse volumétrique améliorée pour la détection de signaux de trading.

Ce module fournit des outils pour analyser les profils de volume, détecter les mouvements
de capitaux importants et calculer la corrélation entre volume et prix.
"""

from enum import Enum
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class VolumeProfileType(Enum):
    """Types de profils de volume supportés."""

    TIME = "time"  # Profil de volume basé sur le temps
    PRICE = "price"  # Profil de volume basé sur le prix (Volume Profile)
    VWAP = "vwap"  # Volume Weighted Average Price


class VolumeAnalyzer:
    """
    Analyseur de volume pour la détection de signaux de trading basés sur le volume.

    Cette classe permet:
    1. La création de profils de volume
    2. La détection de points de contrôle (niveaux de prix importants)
    3. L'identification de mouvements de capitaux significatifs
    4. La validation des signaux de prix par l'analyse du volume
    """

    def __init__(self, ohlcv_data: pd.DataFrame):
        """
        Initialise l'analyseur de volume.

        Args:
            ohlcv_data: DataFrame avec au moins les colonnes 'open', 'high', 'low', 'close' et 'volume'
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in ohlcv_data.columns:
                raise ValueError(f"La colonne {col} est requise dans les données")

        self.data = ohlcv_data.copy()

        # Ajouter une colonne de moyenne des prix pour le calcul de profil de volume
        self.data.loc[:, "avg_price"] = (self.data["high"] + self.data["low"]) / 2

    def create_volume_profile(
        self,
        num_bins: int = 100,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        profile_type: VolumeProfileType = VolumeProfileType.PRICE,
    ) -> Dict:
        """
        Crée un profil de volume basé sur le prix ou le temps.

        Args:
            num_bins: Nombre de niveaux de prix pour le profil (default: 100)
            start_idx: Indice de début de la période d'analyse (default: None = début des données)
            end_idx: Indice de fin de la période d'analyse (default: None = fin des données)
            profile_type: Type de profil à générer (default: VolumeProfileType.PRICE)

        Returns:
            Dictionnaire contenant le profil de volume, avec les niveaux de prix et les volumes
        """
        # Extraire la période d'intérêt
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data) - 1

        period_data = self.data.iloc[start_idx : end_idx + 1].copy()

        # Créer le profil de volume selon le type demandé
        if profile_type == VolumeProfileType.PRICE:
            # Définir les niveaux de prix
            price_min = period_data["low"].min()
            price_max = period_data["high"].max()
            price_bins = np.linspace(price_min, price_max, num_bins + 1)

            # Calculer le volume par niveau de prix
            volume_by_price = np.zeros(num_bins)

            for i, row in period_data.iterrows():
                # Déterminer la plage de prix couverte par cette bougie
                candle_low = row["low"]
                candle_high = row["high"]
                candle_volume = row["volume"]

                # Répartir le volume proportionnellement sur les niveaux de prix traversés
                for bin_idx in range(num_bins):
                    bin_low = price_bins[bin_idx]
                    bin_high = price_bins[bin_idx + 1]

                    # Vérifier si la bougie intersecte ce niveau de prix
                    if candle_high >= bin_low and candle_low <= bin_high:
                        # Calculer l'intersection
                        overlap_low = max(candle_low, bin_low)
                        overlap_high = min(candle_high, bin_high)

                        # Calculer la proportion du volume pour ce niveau
                        overlap_ratio = (
                            (overlap_high - overlap_low) / (candle_high - candle_low)
                            if candle_high > candle_low
                            else 1.0
                        )

                        # Ajouter le volume proportionnel à ce niveau
                        volume_by_price[bin_idx] += candle_volume * overlap_ratio

            # Préparer le résultat
            result = {
                "price_levels": [
                    (price_bins[i] + price_bins[i + 1]) / 2 for i in range(num_bins)
                ],
                "volumes": volume_by_price,
                "max_volume_price": (
                    price_bins[np.argmax(volume_by_price)]
                    + price_bins[np.argmax(volume_by_price) + 1]
                )
                / 2,
                "total_volume": np.sum(volume_by_price),
            }

        elif profile_type == VolumeProfileType.TIME:
            # Pour le profil temporel, on utilise simplement les volumes par période
            result = {
                "timestamps": period_data.index.tolist(),
                "volumes": period_data["volume"].values,
                "total_volume": period_data["volume"].sum(),
            }

        elif profile_type == VolumeProfileType.VWAP:
            # Calculer le VWAP et des bandes
            period_data.loc[:, "typical_price"] = (
                period_data["high"] + period_data["low"] + period_data["close"]
            ) / 3
            period_data.loc[:, "volume_price"] = (
                period_data["typical_price"] * period_data["volume"]
            )

            cumulative_volume = period_data["volume"].cumsum()
            cumulative_vol_price = period_data["volume_price"].cumsum()

            vwap = cumulative_vol_price / cumulative_volume

            # Calculer les écarts types
            period_data.loc[:, "deviation"] = period_data["typical_price"] - vwap
            period_data.loc[:, "squared_dev"] = (
                period_data["deviation"] ** 2 * period_data["volume"]
            )
            cumulative_squared_dev = period_data["squared_dev"].cumsum()

            # Calcul de l'écart type pondéré par le volume
            vwap_stddev = np.sqrt(cumulative_squared_dev / cumulative_volume)

            result = {
                "timestamps": period_data.index.tolist(),
                "vwap": vwap.values,
                "upper_band_1": (vwap + vwap_stddev).values,
                "lower_band_1": (vwap - vwap_stddev).values,
                "upper_band_2": (vwap + 2 * vwap_stddev).values,
                "lower_band_2": (vwap - 2 * vwap_stddev).values,
            }

        return result

    def find_control_points(
        self, lookback_periods: int = 20, min_volume_percentile: float = 0.85
    ) -> List[Dict]:
        """
        Identifie les points de contrôle (niveaux de prix importants) basés sur l'analyse du volume.

        Les points de contrôle sont des niveaux de prix où un volume important a été échangé.

        Args:
            lookback_periods: Nombre de périodes pour l'analyse glissante (default: 20)
            min_volume_percentile: Percentile minimum pour considérer un niveau comme important (default: 0.85)

        Returns:
            Liste des points de contrôle identifiés avec leur niveau de prix et importance
        """
        control_points = []

        # Analyser les données par fenêtres glissantes
        for i in range(lookback_periods, len(self.data)):
            window_start = i - lookback_periods
            window_end = i

            # Créer un profil de volume pour cette fenêtre
            profile = self.create_volume_profile(
                num_bins=50, start_idx=window_start, end_idx=window_end
            )

            # Identifier les niveaux avec un volume significatif
            volumes = profile["volumes"]
            volume_threshold = np.percentile(volumes, min_volume_percentile * 100)

            for j, volume in enumerate(volumes):
                if volume >= volume_threshold:
                    price_level = profile["price_levels"][j]

                    # Calculer un score d'importance
                    importance = volume / profile["total_volume"]

                    # Ajouter ce point de contrôle
                    control_point = {
                        "price": price_level,
                        "volume": volume,
                        "importance": importance,
                        "timestamp": self.data.index[window_end],
                    }

                    # Ajouter uniquement si ce niveau n'est pas déjà présent
                    if not any(
                        abs(cp["price"] - price_level) / price_level < 0.005
                        for cp in control_points
                    ):
                        control_points.append(control_point)

        # Trier par importance
        return sorted(control_points, key=lambda x: x["importance"], reverse=True)

    def detect_volume_anomalies(
        self, window_size: int = 20, threshold_sigma: float = 2.0
    ) -> pd.DataFrame:
        """
        Détecte les anomalies de volume (mouvements de capitaux importants).

        Args:
            window_size: Taille de la fenêtre pour calculer la moyenne mobile (default: 20)
            threshold_sigma: Seuil d'écart-type pour considérer un volume comme anormal (default: 2.0)

        Returns:
            DataFrame avec les anomalies détectées, leurs caractéristiques et importance
        """
        # Calculer les statistiques de base sur le volume
        self.data["volume_ma"] = self.data["volume"].rolling(window=window_size).mean()
        self.data["volume_std"] = self.data["volume"].rolling(window=window_size).std()

        # Calculer le z-score du volume
        self.data["volume_zscore"] = (
            self.data["volume"] - self.data["volume_ma"]
        ) / self.data["volume_std"]

        # Identifier les anomalies
        anomalies = self.data[abs(self.data["volume_zscore"]) > threshold_sigma].copy()

        # Distinguer les volumes anormalement élevés vs bas
        anomalies["anomaly_type"] = np.where(
            anomalies["volume_zscore"] > 0, "high_volume", "low_volume"
        )

        # Calculer le ratio volume / moyenne mobile
        anomalies["volume_ratio"] = anomalies["volume"] / anomalies["volume_ma"]

        # Calculer le pourcentage de mouvement de prix associé
        anomalies["price_change_pct"] = (
            (anomalies["close"] - anomalies["open"]) / anomalies["open"] * 100
        )

        # Calculer un score d'importance
        anomalies["importance"] = abs(anomalies["volume_zscore"]) * abs(
            anomalies["price_change_pct"]
        )

        # Déterminer si c'est un volume de pression acheteur ou vendeur
        anomalies["buying_pressure"] = np.where(
            anomalies["close"] > anomalies["open"], True, False
        )

        return anomalies.sort_values("importance", ascending=False)

    def calculate_volume_price_correlation(
        self, window_size: int = 20, method: str = "pearson"
    ) -> pd.Series:
        """
        Calcule la corrélation glissante entre volume et changement de prix.

        Args:
            window_size: Taille de la fenêtre pour la corrélation glissante (default: 20)
            method: Méthode de corrélation ('pearson', 'spearman', ou 'kendall') (default: 'pearson')

        Returns:
            Série temporelle de coefficients de corrélation
        """
        # Calculer les changements de prix
        self.data["price_change"] = self.data["close"].pct_change()

        # Approche alternative pour calculer la corrélation
        correlation = pd.Series(index=self.data.index)

        for i in range(window_size - 1, len(self.data)):
            window_data = self.data.iloc[i - window_size + 1 : i + 1]
            correlation.iloc[i] = window_data["volume"].corr(
                window_data["price_change"], method=method
            )

        return correlation

    def validate_signal(self, signal_idx: int, lookback: int = 10) -> Dict:
        """
        Valide un signal de trading en analysant le comportement du volume.

        Args:
            signal_idx: Indice du signal dans les données
            lookback: Nombre de périodes à analyser avant le signal (default: 10)

        Returns:
            Dictionnaire contenant les résultats de validation
        """
        # Vérifier que l'indice est valide
        if signal_idx < lookback or signal_idx >= len(self.data):
            raise ValueError(f"Indice de signal invalide: {signal_idx}")

        # Extraire les données pertinentes
        signal_data = self.data.iloc[signal_idx - lookback : signal_idx + 1]

        # 1. Analyser la tendance de volume récente
        recent_volume_trend = signal_data["volume"].pct_change().mean()

        # 2. Calculer le rapport entre le volume au signal et la moyenne récente
        avg_volume = signal_data["volume"].iloc[:-1].mean()
        signal_volume = signal_data["volume"].iloc[-1]
        volume_ratio = signal_volume / avg_volume if avg_volume > 0 else 1.0

        # 3. Calculer la corrélation volume/prix récente
        volume_price_corr = signal_data["volume"].corr(
            signal_data["close"] - signal_data["open"]
        )

        # 4. Déterminer si le volume confirme la direction du prix
        price_change = signal_data["close"].iloc[-1] - signal_data["open"].iloc[-1]
        price_direction = np.sign(price_change)

        # Un volume croissant avec un prix croissant est une confirmation haussière
        # Un volume croissant avec un prix décroissant est une confirmation baissière
        is_confirming = (
            volume_ratio > 1.2
        ) and (  # Volume significativement plus élevé
            (
                price_direction > 0 and volume_price_corr > 0.3
            )  # Hausse du prix, corr positive
            or (price_direction < 0 and volume_price_corr < -0.3)
        )  # Baisse du prix, corr négative

        # 5. Évaluer la qualité du signal
        if volume_ratio > 2.0 and abs(volume_price_corr) > 0.6:
            signal_strength = "strong"
        elif volume_ratio > 1.5 and abs(volume_price_corr) > 0.4:
            signal_strength = "moderate"
        elif volume_ratio > 1.2 and abs(volume_price_corr) > 0.2:
            signal_strength = "weak"
        else:
            signal_strength = "invalid"

        # Résultats de la validation
        return {
            "is_valid": is_confirming,
            "strength": signal_strength,
            "volume_ratio": volume_ratio,
            "volume_price_correlation": volume_price_corr,
            "recent_volume_trend": recent_volume_trend,
            "recommendation": (
                "buy"
                if is_confirming and price_direction > 0
                else "sell" if is_confirming and price_direction < 0 else "neutral"
            ),
        }

    def plot_volume_profile(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        num_bins: int = 50,
        profile_type: VolumeProfileType = VolumeProfileType.PRICE,
        ax=None,
        horizontal: bool = True,
    ):
        """
        Trace un profil de volume.

        Args:
            start_idx: Indice de début (default: None = début des données)
            end_idx: Indice de fin (default: None = fin des données)
            num_bins: Nombre de niveaux (default: 50)
            profile_type: Type de profil (default: VolumeProfileType.PRICE)
            ax: Axes matplotlib pour le tracé (default: None = création d'un nouvel axe)
            horizontal: Si True, trace un histogramme horizontal (default: True)
        """
        # Créer le profil
        profile = self.create_volume_profile(
            num_bins=num_bins,
            start_idx=start_idx,
            end_idx=end_idx,
            profile_type=profile_type,
        )

        # Créer l'axe si nécessaire
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if profile_type == VolumeProfileType.PRICE:
            if horizontal:
                ax.barh(profile["price_levels"], profile["volumes"], alpha=0.7)
                ax.set_ylabel("Prix")
                ax.set_xlabel("Volume")

                # Marquer le point de contrôle de volume maximal
                max_vol_idx = np.argmax(profile["volumes"])
                max_vol_price = profile["price_levels"][max_vol_idx]
                ax.axhline(y=max_vol_price, color="r", linestyle="--", alpha=0.5)
                ax.text(
                    profile["volumes"][max_vol_idx] * 0.5,
                    max_vol_price,
                    f"POC: {max_vol_price:.2f}",
                    verticalalignment="bottom",
                )
            else:
                ax.bar(profile["price_levels"], profile["volumes"], alpha=0.7)
                ax.set_xlabel("Prix")
                ax.set_ylabel("Volume")

                # Marquer le point de contrôle de volume maximal
                max_vol_idx = np.argmax(profile["volumes"])
                max_vol_price = profile["price_levels"][max_vol_idx]
                ax.axvline(x=max_vol_price, color="r", linestyle="--", alpha=0.5)
                ax.text(
                    max_vol_price,
                    profile["volumes"][max_vol_idx] * 0.5,
                    f"POC: {max_vol_price:.2f}",
                    horizontalalignment="left",
                )

        elif profile_type == VolumeProfileType.VWAP:
            # Période d'intérêt
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(self.data) - 1

            period_data = self.data.iloc[start_idx : end_idx + 1]

            # Tracer le VWAP et les bandes
            ax.plot(profile["timestamps"], profile["vwap"], label="VWAP", color="blue")
            ax.plot(
                profile["timestamps"],
                profile["upper_band_1"],
                "--",
                color="red",
                alpha=0.5,
                label="+1σ",
            )
            ax.plot(
                profile["timestamps"],
                profile["lower_band_1"],
                "--",
                color="green",
                alpha=0.5,
                label="-1σ",
            )
            ax.plot(
                profile["timestamps"],
                profile["upper_band_2"],
                ":",
                color="red",
                alpha=0.3,
                label="+2σ",
            )
            ax.plot(
                profile["timestamps"],
                profile["lower_band_2"],
                ":",
                color="green",
                alpha=0.3,
                label="-2σ",
            )

            # Tracer le prix de clôture
            ax.plot(
                period_data.index,
                period_data["close"],
                color="black",
                alpha=0.7,
                label="Close",
            )

            ax.set_xlabel("Temps")
            ax.set_ylabel("Prix")
            ax.legend()

        return ax


def volume_delta(ohlcv_data: pd.DataFrame, window_size: int = 14) -> pd.DataFrame:
    """
    Calcule le delta de volume (acheteur vs vendeur) pour chaque bougie.

    Cette fonction estime le volume acheteur vs vendeur en fonction du positionnement
    du prix de clôture dans la bougie.

    Args:
        ohlcv_data: DataFrame avec colonnes OHLCV
        window_size: Taille de la fenêtre pour les calculs cumulatifs (default: 14)

    Returns:
        DataFrame avec les colonnes de delta de volume ajoutées
    """
    # Vérification des colonnes requises
    required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in ohlcv_data.columns:
            raise ValueError(f"La colonne {col} est requise dans les données")

    # Créer une copie des données
    df = ohlcv_data.copy()

    # Calculer la plage de la bougie
    df["candle_range"] = df["high"] - df["low"]

    # Calculer la proportion du prix de clôture dans la plage
    df["close_position"] = (df["close"] - df["low"]) / df["candle_range"]

    # Estimer le volume acheteur vs vendeur
    df["buying_volume"] = df["volume"] * df["close_position"]
    df["selling_volume"] = df["volume"] * (1 - df["close_position"])

    # Calculer le delta de volume (achat - vente)
    df["volume_delta"] = df["buying_volume"] - df["selling_volume"]

    # Calculer le delta cumulatif sur la fenêtre
    df["cumulative_delta"] = df["volume_delta"].rolling(window=window_size).sum()

    # Normaliser le delta par rapport au volume total
    df["normalized_delta"] = df["volume_delta"] / df["volume"]

    return df


def on_balance_volume(ohlcv_data: pd.DataFrame) -> pd.Series:
    """
    Calcule l'indicateur On Balance Volume (OBV).

    L'OBV est un indicateur qui relie les changements de prix et de volume.

    Args:
        ohlcv_data: DataFrame avec colonnes OHLCV

    Returns:
        Série avec les valeurs OBV
    """
    # Vérification des colonnes requises
    required_columns = ["close", "volume"]
    for col in required_columns:
        if col not in ohlcv_data.columns:
            raise ValueError(f"La colonne {col} est requise dans les données")

    # Créer une copie des données
    df = ohlcv_data.copy()

    # Calculer les changements de prix
    df["price_change"] = df["close"].diff()

    # Initialiser la série OBV
    obv = pd.Series(0, index=df.index)

    # Calculer l'OBV de manière cumulative
    for i in range(1, len(df)):
        if df["price_change"].iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
        elif df["price_change"].iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    return obv


def accelerating_volume(ohlcv_data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Détecte l'accélération du volume qui peut indiquer un intérêt accru pour un actif.

    Args:
        ohlcv_data: DataFrame avec colonnes OHLCV
        window: Taille de la fenêtre pour les calculs (default: 5)

    Returns:
        DataFrame avec les métriques d'accélération du volume
    """
    # Vérification des colonnes requises
    if "volume" not in ohlcv_data.columns:
        raise ValueError("La colonne 'volume' est requise dans les données")

    # Créer une copie des données
    df = ohlcv_data.copy()

    # Calculer les changements de volume
    df["volume_change"] = df["volume"].pct_change()

    # Calculer la moyenne mobile du changement de volume
    df["volume_change_ma"] = df["volume_change"].rolling(window=window).mean()

    # Calculer l'accélération (dérivée seconde approximée par la différence des changements)
    df["volume_acceleration"] = df["volume_change"].diff()

    # Moyenne mobile de l'accélération
    df["volume_acceleration_ma"] = (
        df["volume_acceleration"].rolling(window=window).mean()
    )

    # Identifier les périodes d'accélération significative
    df["is_accelerating"] = df["volume_acceleration"] > 0.1

    return df
