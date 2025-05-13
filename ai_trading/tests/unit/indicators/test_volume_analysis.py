"""
Tests unitaires pour le module d'analyse volumétrique.
"""

import unittest

import numpy as np
import pandas as pd

from ai_trading.indicators.volume_analysis import (
    VolumeAnalyzer,
    VolumeProfileType,
    accelerating_volume,
    on_balance_volume,
    volume_delta,
)


def create_test_data(n_samples=100):
    """Crée des données de test réalistes."""
    np.random.seed(42)

    # Créer des prix simulés avec tendance et bruit
    trend = np.linspace(0, 10, n_samples)
    noise = np.random.normal(0, 1, n_samples)
    cycle = 5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))

    close = 100 + trend + cycle + noise

    # Volumes avec relation aux prix (plus de volume sur les mouvements importants)
    base_volume = 1000 + 500 * np.random.random(n_samples)
    price_change = np.diff(close, prepend=close[0])
    volume = base_volume + 200 * np.abs(price_change)

    # Créer quelques pics de volume (en tenant compte de la taille de n_samples)
    quarter = n_samples // 4
    mid = n_samples // 2
    three_quarters = 3 * n_samples // 4

    if quarter < n_samples:
        volume[quarter] = volume[quarter] * 3
    if mid < n_samples:
        volume[mid] = volume[mid] * 4
    if three_quarters < n_samples:
        volume[three_quarters] = volume[three_quarters] * 2.5

    # Générer open, high, low
    open_price = close - np.random.normal(0, 0.5, n_samples)
    high = np.maximum(close, open_price) + np.random.uniform(0, 1, n_samples)
    low = np.minimum(close, open_price) - np.random.uniform(0, 1, n_samples)

    # Créer le DataFrame
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestVolumeAnalyzer(unittest.TestCase):
    """Tests pour la classe VolumeAnalyzer."""

    def setUp(self):
        """Initialise les données et l'analyseur de volume pour les tests."""
        self.data = create_test_data(100)
        self.analyzer = VolumeAnalyzer(self.data)

    def test_initialization(self):
        """Teste l'initialisation de l'analyseur."""
        # Vérifier que l'analyseur a été correctement initialisé
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.data)

        # Vérifier que la colonne 'avg_price' a été ajoutée
        self.assertIn("avg_price", self.analyzer.data.columns)
        self.assertEqual(len(self.analyzer.data), len(self.data))

    def test_create_volume_profile_price(self):
        """Teste la création d'un profil de volume basé sur le prix."""
        profile = self.analyzer.create_volume_profile(
            num_bins=20, profile_type=VolumeProfileType.PRICE
        )

        # Vérifier les clés du résultat
        self.assertIn("price_levels", profile)
        self.assertIn("volumes", profile)
        self.assertIn("max_volume_price", profile)
        self.assertIn("total_volume", profile)

        # Vérifier les dimensions
        self.assertEqual(len(profile["price_levels"]), 20)
        self.assertEqual(len(profile["volumes"]), 20)

        # Vérifier que les volumes sont positifs
        self.assertTrue(np.all(profile["volumes"] >= 0))

        # Vérifier que le prix avec volume max est dans la plage des prix
        self.assertTrue(
            self.data["low"].min()
            <= profile["max_volume_price"]
            <= self.data["high"].max()
        )

    def test_create_volume_profile_vwap(self):
        """Teste la création d'un profil VWAP."""
        profile = self.analyzer.create_volume_profile(
            profile_type=VolumeProfileType.VWAP
        )

        # Vérifier les clés du résultat
        self.assertIn("vwap", profile)
        self.assertIn("upper_band_1", profile)
        self.assertIn("lower_band_1", profile)

        # Vérifier les dimensions
        self.assertEqual(len(profile["vwap"]), len(self.data))

    def test_find_control_points(self):
        """Teste la détection des points de contrôle."""
        control_points = self.analyzer.find_control_points(
            lookback_periods=20, min_volume_percentile=0.75
        )

        # Vérifier qu'on a trouvé des points de contrôle
        self.assertTrue(len(control_points) > 0)

        # Vérifier le format des points de contrôle
        for point in control_points:
            self.assertIn("price", point)
            self.assertIn("volume", point)
            self.assertIn("importance", point)

            # Vérifier que l'importance est entre 0 et 1
            self.assertTrue(0 <= point["importance"] <= 1)

    def test_detect_volume_anomalies(self):
        """Teste la détection des anomalies de volume."""
        anomalies = self.analyzer.detect_volume_anomalies(
            window_size=10, threshold_sigma=1.5
        )

        # Vérifier qu'on a détecté des anomalies
        self.assertTrue(len(anomalies) > 0)

        # Vérifier le format des anomalies
        self.assertIn("anomaly_type", anomalies.columns)
        self.assertIn("volume_ratio", anomalies.columns)
        self.assertIn("importance", anomalies.columns)

        # Vérifier que les pics de volume artificiels sont détectés
        self.assertTrue(
            25 in anomalies.index or 50 in anomalies.index or 75 in anomalies.index
        )

    def test_calculate_volume_price_correlation(self):
        """Teste le calcul de la corrélation volume/prix."""
        correlation = self.analyzer.calculate_volume_price_correlation(window_size=15)

        # Vérifier les dimensions
        self.assertEqual(len(correlation), len(self.data))

        # Vérifier que les corrélations sont dans la plage [-1, 1]
        valid_corr = correlation.dropna()
        self.assertTrue(np.all(valid_corr >= -1))
        self.assertTrue(np.all(valid_corr <= 1))

    def test_validate_signal(self):
        """Teste la validation d'un signal."""
        # Choisir un indice avec anomalie de volume pour le test
        signal_idx = 50  # Pic artificiel

        validation = self.analyzer.validate_signal(signal_idx=signal_idx, lookback=10)

        # Vérifier le format de la validation
        self.assertIn("is_valid", validation)
        self.assertIn("strength", validation)
        self.assertIn("volume_ratio", validation)
        self.assertIn("recommendation", validation)

        # Vérifier que le ratio de volume est élevé (pic artificiel)
        self.assertTrue(validation["volume_ratio"] > 1.5)


class TestVolumeDelta(unittest.TestCase):
    """Tests pour la fonction volume_delta."""

    def setUp(self):
        """Initialise les données pour les tests."""
        self.data = create_test_data(50)

    def test_volume_delta_calculation(self):
        """Teste le calcul du delta de volume."""
        result = volume_delta(self.data, window_size=10)

        # Vérifier les colonnes ajoutées
        self.assertIn("buying_volume", result.columns)
        self.assertIn("selling_volume", result.columns)
        self.assertIn("volume_delta", result.columns)
        self.assertIn("cumulative_delta", result.columns)

        # Vérifier les dimensions
        self.assertEqual(len(result), len(self.data))

        # Vérifier que le delta est cohérent
        for i in range(len(result)):
            row = result.iloc[i]
            self.assertAlmostEqual(
                row["buying_volume"] + row["selling_volume"], row["volume"], delta=0.01
            )
            self.assertAlmostEqual(
                row["volume_delta"],
                row["buying_volume"] - row["selling_volume"],
                delta=0.01,
            )


class TestOnBalanceVolume(unittest.TestCase):
    """Tests pour la fonction on_balance_volume."""

    def setUp(self):
        """Initialise les données pour les tests."""
        self.data = create_test_data(50)

    def test_obv_calculation(self):
        """Teste le calcul de l'OBV."""
        obv = on_balance_volume(self.data)

        # Vérifier les dimensions
        self.assertEqual(len(obv), len(self.data))

        # Vérifier que le premier élément est 0
        self.assertEqual(obv.iloc[0], 0)

        # Vérifier la cohérence des calculs
        for i in range(1, len(obv)):
            price_change = self.data["close"].iloc[i] - self.data["close"].iloc[i - 1]
            volume = self.data["volume"].iloc[i]

            if price_change > 0:
                expected_change = volume
            elif price_change < 0:
                expected_change = -volume
            else:
                expected_change = 0

            self.assertAlmostEqual(
                obv.iloc[i] - obv.iloc[i - 1], expected_change, delta=0.01
            )


class TestAcceleratingVolume(unittest.TestCase):
    """Tests pour la fonction accelerating_volume."""

    def setUp(self):
        """Initialise les données pour les tests."""
        self.data = create_test_data(50)

    def test_accelerating_volume_calculation(self):
        """Teste le calcul de l'accélération du volume."""
        result = accelerating_volume(self.data, window=5)

        # Vérifier les colonnes ajoutées
        self.assertIn("volume_change", result.columns)
        self.assertIn("volume_acceleration", result.columns)
        self.assertIn("is_accelerating", result.columns)

        # Vérifier les dimensions
        self.assertEqual(len(result), len(self.data))


if __name__ == "__main__":
    unittest.main()
