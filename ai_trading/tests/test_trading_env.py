import unittest

import numpy as np
import pandas as pd

from ai_trading.rl.trading_environment import TradingEnvironment


class TestTradingEnv(unittest.TestCase):
    """Tests de base pour l'environnement de trading."""

    def setUp(self):
        """Initialise les données de test."""
        # Créer des données de test
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(150, 250, 100),
                "low": np.random.uniform(50, 150, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

    def test_initialization(self):
        """Teste l'initialisation de l'environnement."""
        env = TradingEnvironment(df=self.test_data)
        self.assertIsNotNone(env)
        self.assertEqual(env.initial_balance, 10000)
