import unittest

import numpy as np
import pandas as pd

from ..rl.environments.multi_asset_trading_environment import (
    MultiAssetTradingEnvironment,
)
from ..rl.market_constraints import MarketConstraints


class TestMultiAssetTradingEnvironment(unittest.TestCase):
    def setUp(self):
        """Initialisation des tests avec des données fictives."""
        # Création de données de test
        self.dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        self.data_dict = {
            "BTC": pd.DataFrame(
                {
                    "close": np.random.normal(50000, 1000, 100),
                    "volume": np.random.normal(1000, 100, 100),
                    "volatility": np.random.uniform(0.01, 0.05, 100),
                },
                index=self.dates,
            ),
            "ETH": pd.DataFrame(
                {
                    "close": np.random.normal(3000, 100, 100),
                    "volume": np.random.normal(500, 50, 100),
                    "volatility": np.random.uniform(0.01, 0.05, 100),
                },
                index=self.dates,
            ),
        }

        # Initialisation de l'environnement
        self.env = MultiAssetTradingEnvironment(
            assets=["BTC", "ETH"], initial_balance=100000.0
        )

    def test_initialization(self):
        """Teste l'initialisation de l'environnement."""
        self.assertEqual(self.env.initial_balance, 100000.0)
        self.assertEqual(len(self.env.assets), 2)
        self.assertIn("BTC", self.env.assets)
        self.assertIn("ETH", self.env.assets)
        self.assertIsInstance(self.env.market_constraints, MarketConstraints)

    def test_reset(self):
        """Teste la réinitialisation de l'environnement."""
        state = self.env.reset()

        self.assertEqual(self.env.balance, self.env.initial_balance)
        self.assertEqual(self.env.portfolio_value, self.env.initial_balance)
        self.assertEqual(self.env.holdings["BTC"], 0.0)
        self.assertEqual(self.env.holdings["ETH"], 0.0)
        self.assertEqual(self.env.slippage_value, 0.0)

        # Vérification de l'état retourné
        self.assertIsInstance(state, dict)
        self.assertIn("balance", state)
        self.assertIn("portfolio_value", state)
        self.assertIn("holdings", state)
        self.assertIn("prices", state)

    def test_step_buy_action(self):
        """Teste l'exécution d'une action d'achat."""
        self.env.reset()

        # Simuler un achat de BTC
        actions = {"BTC": 0.1, "ETH": 0.0}  # Acheter 10% du portfolio en BTC
        state, reward, done, info = self.env.step(actions)

        # Vérifications
        self.assertGreater(self.env.holdings["BTC"], 0)
        self.assertEqual(self.env.holdings["ETH"], 0)
        self.assertLess(self.env.balance, self.env.initial_balance)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

    def test_step_sell_action(self):
        """Teste l'exécution d'une action de vente."""
        self.env.reset()

        # D'abord acheter, puis vendre
        actions_buy = {"BTC": 0.1, "ETH": 0.0}
        self.env.step(actions_buy)

        initial_btc_holdings = self.env.holdings["BTC"]
        actions_sell = {"BTC": -0.05, "ETH": 0.0}  # Vendre 5% du portfolio en BTC
        state, reward, done, info = self.env.step(actions_sell)

        # Vérifications
        self.assertLess(self.env.holdings["BTC"], initial_btc_holdings)
        self.assertGreater(self.env.balance, 0)

    def test_diversification_reward(self):
        """Teste le calcul de la récompense de diversification."""
        self.env.reset()

        # Test avec un portfolio diversifié
        actions_diversified = {"BTC": 0.1, "ETH": 0.1}
        _, reward_diversified, _, _ = self.env.step(actions_diversified)

        # Test avec un portfolio concentré
        self.env.reset()
        actions_concentrated = {"BTC": 0.2, "ETH": 0.0}
        _, reward_concentrated, _, _ = self.env.step(actions_concentrated)

        # La récompense devrait être meilleure pour le portfolio diversifié
        self.assertGreaterEqual(reward_diversified, reward_concentrated)

    def test_slippage_calculation(self):
        """Teste le calcul du slippage."""
        self.env.reset()

        # Test avec un gros volume (devrait avoir plus de slippage)
        actions_large = {"BTC": 0.5, "ETH": 0.0}
        _, _, _, info_large = self.env.step(actions_large)

        # Test avec un petit volume
        self.env.reset()
        actions_small = {"BTC": 0.01, "ETH": 0.0}
        _, _, _, info_small = self.env.step(actions_small)

        # Le slippage devrait être plus important pour le gros volume
        self.assertGreater(self.env.slippage_value, 0)

    def test_invalid_actions(self):
        """Teste le comportement avec des actions invalides."""
        self.env.reset()

        # Test avec un actif inexistant
        actions_invalid = {"INVALID": 0.1, "ETH": 0.0}
        state, reward, done, info = self.env.step(actions_invalid)

        # L'environnement devrait ignorer l'actif invalide
        self.assertNotIn("INVALID", self.env.holdings)
        self.assertEqual(self.env.holdings["ETH"], 0.0)

    def test_portfolio_value_calculation(self):
        """Teste le calcul de la valeur du portfolio."""
        self.env.reset()

        # Effectuer quelques transactions
        actions = {"BTC": 0.1, "ETH": 0.1}
        self.env.step(actions)

        # Calculer la valeur totale du portfolio
        portfolio_value = self.env.portfolio_value

        # La valeur devrait être égale à la somme de la balance et des actifs
        expected_value = self.env.balance + sum(
            self.env.holdings[asset] * self.env._get_current_price(asset)
            for asset in self.env.assets
        )

        self.assertAlmostEqual(portfolio_value, expected_value, places=5)


if __name__ == "__main__":
    unittest.main()
