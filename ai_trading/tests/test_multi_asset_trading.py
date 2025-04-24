import unittest

import numpy as np
import pandas as pd

from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment


class TestMultiAssetTrading(unittest.TestCase):
    def setUp(self):
        # Créer des données de test pour plusieurs actifs
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        self.data_dict = {
            "BTC": pd.DataFrame(
                {
                    "open": np.random.uniform(100, 200, 100),
                    "high": np.random.uniform(150, 250, 100),
                    "low": np.random.uniform(50, 150, 100),
                    "close": np.random.uniform(100, 200, 100),
                    "volume": np.random.uniform(1000, 5000, 100),
                },
                index=dates,
            ),
            "ETH": pd.DataFrame(
                {
                    "open": np.random.uniform(50, 100, 100),
                    "high": np.random.uniform(75, 125, 100),
                    "low": np.random.uniform(25, 75, 100),
                    "close": np.random.uniform(50, 100, 100),
                    "volume": np.random.uniform(2000, 6000, 100),
                },
                index=dates,
            ),
            "SOL": pd.DataFrame(
                {
                    "open": np.random.uniform(20, 40, 100),
                    "high": np.random.uniform(30, 50, 100),
                    "low": np.random.uniform(10, 30, 100),
                    "close": np.random.uniform(20, 40, 100),
                    "volume": np.random.uniform(3000, 7000, 100),
                },
                index=dates,
            ),
        }

    def test_initialization(self):
        """Teste l'initialisation de l'environnement multi-actifs."""
        env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            max_active_positions=2,
            correlation_threshold=0.7,
            volatility_threshold=0.05,
        )

        self.assertEqual(len(env.symbols), 3)
        self.assertEqual(env.max_active_positions, 2)
        self.assertIsNotNone(env.asset_correlations)
        self.assertIsNotNone(env.asset_volatilities)

    def test_asset_filtering(self):
        """Teste le filtrage des actifs."""
        env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            max_active_positions=2,
            correlation_threshold=0.7,
            volatility_threshold=0.05,
        )

        filtered_assets = env._filter_assets()
        self.assertLessEqual(len(filtered_assets), env.max_active_positions)

        # Vérifier que les actifs sélectionnés ont une faible corrélation
        if len(filtered_assets) > 1:
            for i in range(len(filtered_assets)):
                for j in range(i + 1, len(filtered_assets)):
                    corr = abs(
                        env.asset_correlations.loc[
                            filtered_assets[i], filtered_assets[j]
                        ]
                    )
                    self.assertLessEqual(corr, env.correlation_threshold)

    def test_portfolio_allocation(self):
        """Teste l'allocation du portefeuille."""
        env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            max_active_positions=2,
            correlation_threshold=0.7,
            volatility_threshold=0.05,
        )

        # Créer une action d'allocation
        action = np.array([0.5, -0.3, 0.2])  # BTC: 50%, ETH: -30%, SOL: 20%

        # Exécuter l'action
        env.step(action)

        # Vérifier que les allocations sont valides
        total_allocation = sum(
            abs(
                env.crypto_holdings[symbol]
                * env.data_dict[symbol].iloc[env.current_step]["close"]
            )
            for symbol in env.symbols
        )
        self.assertLessEqual(total_allocation, env.initial_balance)

    def test_rebalancing(self):
        """Teste le rééquilibrage périodique du portefeuille."""
        env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            max_active_positions=2,
            rebalance_frequency=5,
            correlation_threshold=0.7,
            volatility_threshold=0.05,
        )

        # Exécuter plusieurs pas
        for _ in range(10):
            action = np.random.uniform(-1, 1, size=3)
            env.step(action)

        # Vérifier que le rééquilibrage se produit au bon moment
        self.assertLessEqual(env.steps_since_rebalance, env.rebalance_frequency)

    def test_slippage(self):
        """Teste l'application du slippage."""
        env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict, slippage_model="constant", slippage_value=0.001
        )

        price = 100.0
        action_value = 0.5

        # Test achat
        price_with_slippage = env._apply_slippage(price, action_value)
        self.assertAlmostEqual(price_with_slippage, 100.1)  # 100 * (1 + 0.001)

        # Test vente
        price_with_slippage = env._apply_slippage(price, -action_value)
        self.assertAlmostEqual(price_with_slippage, 99.9)  # 100 * (1 - 0.001)


if __name__ == "__main__":
    unittest.main()
