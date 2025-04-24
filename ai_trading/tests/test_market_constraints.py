import unittest

from ..rl.market_constraints import MarketConstraints


class TestMarketConstraints(unittest.TestCase):
    def setUp(self):
        """Prépare l'environnement de test."""
        self.constraints = MarketConstraints()

    def test_slippage_fixed(self):
        """Teste le calcul du slippage avec des valeurs fixes."""
        symbol = "BTC"
        action_value = 0.1
        volume = 1000.0
        volatility = 0.02
        avg_volume = 5000.0

        slippage = self.constraints.calculate_slippage(
            symbol, action_value, volume, volatility, avg_volume
        )

        self.assertGreater(slippage, 0)
        self.assertLess(slippage, 0.01)  # Max 1%

    def test_slippage_dynamic(self):
        """Teste le calcul du slippage dynamique."""
        # Test avec volume élevé
        high_volume = 10000.0
        high_volatility = 0.05
        slippage_high = self.constraints.calculate_slippage(
            "BTC", 0.1, high_volume, high_volatility, 5000.0
        )

        # Test avec volume faible
        low_volume = 100.0
        low_volatility = 0.01
        slippage_low = self.constraints.calculate_slippage(
            "BTC", 0.1, low_volume, low_volatility, 5000.0
        )

        # Le slippage devrait être plus élevé pour un volume plus important
        self.assertGreater(slippage_high, slippage_low)

    def test_market_impact(self):
        """Teste le calcul de l'impact marché."""
        symbol = "BTC"
        action_value = 0.2
        volume = 2000.0
        price = 50000.0
        avg_volume = 5000.0

        impact, recovery_time = self.constraints.calculate_market_impact(
            symbol, action_value, volume, price, avg_volume
        )

        self.assertGreater(impact, 0)
        self.assertLess(impact, self.constraints.max_impact)
        self.assertGreater(recovery_time, 0)
        self.assertLessEqual(recovery_time, self.constraints.max_recovery_time)

    def test_zero_volume(self):
        """Teste le comportement avec un volume nul."""
        slippage = self.constraints.calculate_slippage("BTC", 0.1, 0.0, 0.02, 1000.0)
        self.assertEqual(slippage, 0.0)

        impact, recovery_time = self.constraints.calculate_market_impact(
            "BTC", 0.1, 0.0, 50000.0, 1000.0
        )
        self.assertEqual(impact, 0.0)
        self.assertEqual(recovery_time, 1)

    def test_extreme_market_conditions(self):
        """Teste le comportement dans des conditions de marché extrêmes."""
        # Volume très élevé
        high_volume_slip = self.constraints.calculate_slippage(
            "BTC", 1.0, 100000.0, 0.1, 1000.0
        )
        self.assertLessEqual(high_volume_slip, 0.01)  # Max 1%

        # Volatilité très élevée
        high_vol_slip = self.constraints.calculate_slippage(
            "BTC", 0.1, 1000.0, 0.5, 1000.0
        )
        self.assertLessEqual(high_vol_slip, 0.01)

    def test_orderbook_depth(self):
        """Teste l'intégration avec la profondeur du carnet d'ordres."""
        # Mise à jour des données du carnet
        self.constraints.update_orderbook_depth(
            "BTC",
            {
                "spread_pct": 0.002,
                "total_volume": 1000.0,
                "volume_imbalance": 0.1,
            },
        )

        # Test du slippage avec les données du carnet
        slippage = self.constraints.calculate_slippage("BTC", 0.1, 1000.0, 0.02, 5000.0)
        self.assertGreater(slippage, 0)

    def test_invalid_inputs(self):
        """Teste le comportement avec des entrées invalides."""
        # Test avec des valeurs négatives
        slippage = self.constraints.calculate_slippage(
            "BTC", -0.1, -1000.0, -0.02, -5000.0
        )
        self.assertGreaterEqual(slippage, 0)

        impact, recovery_time = self.constraints.calculate_market_impact(
            "BTC", -0.1, -1000.0, -50000.0, -5000.0
        )
        self.assertGreaterEqual(impact, 0)
        self.assertGreaterEqual(recovery_time, 1)

    def test_gradual_market_degradation(self):
        """Teste la dégradation progressive des conditions de marché."""
        initial_slip = self.constraints.calculate_slippage(
            "BTC", 0.1, 1000.0, 0.02, 5000.0
        )

        # Simuler une dégradation du marché
        self.constraints.update_orderbook_depth(
            "BTC",
            {
                "spread_pct": 0.005,  # Spread plus large
                "total_volume": 500.0,  # Volume réduit
                "volume_imbalance": 0.3,  # Déséquilibre plus important
            },
        )

        degraded_slip = self.constraints.calculate_slippage(
            "BTC", 0.1, 1000.0, 0.02, 5000.0
        )
        self.assertGreater(degraded_slip, initial_slip)

    def test_execution_delay(self):
        """Teste le calcul du délai d'exécution."""
        # Test avec volume normal
        delay_normal = self.constraints.calculate_execution_delay(
            "BTC", 0.1, 1000.0, 5000.0
        )
        self.assertGreaterEqual(delay_normal, 0)
        self.assertLessEqual(delay_normal, 10)

        # Test avec gros volume
        delay_large = self.constraints.calculate_execution_delay(
            "BTC", 0.1, 10000.0, 5000.0
        )
        self.assertGreater(delay_large, delay_normal)


if __name__ == "__main__":
    unittest.main()
