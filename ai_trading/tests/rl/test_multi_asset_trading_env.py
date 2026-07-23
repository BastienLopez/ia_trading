import unittest

import numpy as np
import pandas as pd

from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment
from ai_trading.rl.portfolio_allocator import PortfolioAllocator


class TestMultiAssetTradingEnv(unittest.TestCase):
    """Tests pour l'environnement de trading multi-actifs."""

    def setUp(self):
        """Initialise les données de test."""
        # Créer des données de test pour plusieurs actifs
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

        # Créer des données pour BTC avec une tendance haussière
        btc_data = pd.DataFrame(
            {
                "open": np.linspace(9000, 12000, 100) + np.random.normal(0, 200, 100),
                "high": np.linspace(9100, 12200, 100) + np.random.normal(0, 250, 100),
                "low": np.linspace(8900, 11800, 100) + np.random.normal(0, 250, 100),
                "close": np.linspace(9000, 12000, 100) + np.random.normal(0, 200, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        # Créer des données pour ETH avec une tendance baissière
        eth_data = pd.DataFrame(
            {
                "open": np.linspace(300, 200, 100) + np.random.normal(0, 10, 100),
                "high": np.linspace(310, 210, 100) + np.random.normal(0, 15, 100),
                "low": np.linspace(290, 190, 100) + np.random.normal(0, 15, 100),
                "close": np.linspace(300, 200, 100) + np.random.normal(0, 10, 100),
                "volume": np.random.uniform(500, 2500, 100),
            },
            index=dates,
        )

        # Créer des données pour LTC avec une tendance latérale
        ltc_data = pd.DataFrame(
            {
                "open": np.ones(100) * 50 + np.random.normal(0, 5, 100),
                "high": np.ones(100) * 55 + np.random.normal(0, 5, 100),
                "low": np.ones(100) * 45 + np.random.normal(0, 5, 100),
                "close": np.ones(100) * 50 + np.random.normal(0, 5, 100),
                "volume": np.random.uniform(200, 1000, 100),
            },
            index=dates,
        )

        # Créer le dictionnaire de données
        self.test_data = {"BTC": btc_data, "ETH": eth_data, "LTC": ltc_data}

    def test_initialization(self):
        """Teste l'initialisation de l'environnement multi-actifs."""
        env = MultiAssetTradingEnvironment(data_dict=self.test_data)

        # Vérifier les attributs de base
        self.assertIsNotNone(env)
        self.assertEqual(env.initial_balance, 10000.0)
        self.assertEqual(env.num_assets, 3)
        self.assertEqual(set(env.symbols), {"BTC", "ETH", "LTC"})

        # Vérifier l'espace d'action
        self.assertEqual(env.action_space.shape, (3,))
        self.assertEqual(env.action_space.low[0], -1.0)
        self.assertEqual(env.action_space.high[0], 1.0)

        # Vérifier l'état initial
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)

        # Vérifier les valeurs initiales
        self.assertEqual(env.balance, 10000.0)
        self.assertEqual(env.crypto_holdings["BTC"], 0.0)
        self.assertEqual(env.crypto_holdings["ETH"], 0.0)
        self.assertEqual(env.crypto_holdings["LTC"], 0.0)

    def test_action_execution(self):
        """Teste l'exécution d'actions dans l'environnement multi-actifs."""
        env = MultiAssetTradingEnvironment(
            data_dict=self.test_data,
            allocation_method="equal",
            rebalance_frequency=1,  # Rééquilibrer à chaque étape pour faciliter le test
        )

        # Réinitialiser l'environnement
        env.reset()

        # Exécuter une action d'achat pour tous les actifs
        action = np.array([0.5, 0.5, 0.5])  # Allocation égale entre les trois actifs
        obs, reward, done, _, info = env.step(action)

        # Vérifier que les achats ont été effectués
        self.assertGreater(env.crypto_holdings["BTC"], 0.0)
        self.assertGreater(env.crypto_holdings["ETH"], 0.0)
        self.assertGreater(env.crypto_holdings["LTC"], 0.0)
        self.assertLess(env.balance, 10000.0)

        # Vérifier que l'observation et les informations sont correctes
        self.assertIsInstance(obs, np.ndarray)
        self.assertGreater(
            reward, -1.0
        )  # La récompense ne devrait pas être trop négative
        self.assertFalse(done)
        self.assertIsInstance(info, dict)

    def test_multi_asset_masks_prevent_short_sales_and_keep_audit_fills(self):
        env = MultiAssetTradingEnvironment(data_dict=self.test_data, rebalance_frequency=1)
        env.reset()
        mask = env.get_action_mask()
        self.assertTrue((mask["low"] == 0.0).all())
        self.assertTrue((env.project_action(np.array([-1.0, -0.5, -0.2])) == 0.0).all())

        _, _, _, _, buy_info = env.step(np.array([0.5, 0.5, 0.5]))
        holdings_before_sell = env.crypto_holdings.copy()
        _, _, _, _, sell_info = env.step(np.array([-0.5, -0.5, -0.5]))

        self.assertTrue(all(env.crypto_holdings[symbol] >= 0.0 for symbol in env.symbols))
        self.assertTrue(all(env.crypto_holdings[symbol] < holdings_before_sell[symbol] for symbol in env.symbols))
        self.assertTrue(buy_info["trade_executed"] and sell_info["trade_executed"])
        self.assertTrue(any(fill["side"] == "buy" for fill in sell_info["trade_events"]))
        self.assertTrue(any(fill["side"] == "sell" for fill in sell_info["trade_events"]))

    def test_rebalance_cooldown_and_minimum_signal_block_churn(self):
        env = MultiAssetTradingEnvironment(
            data_dict=self.test_data,
            rebalance_frequency=3,
            min_trade_fraction=0.01,
            min_action_magnitude=0.1,
            risk_management=False,
        )
        env.reset()
        self.assertTrue((env.project_action(np.array([0.05, 0.04, 0.01])) == 0.0).all())

        env.step(np.array([0.5, 0.5, 0.5]))
        trade_count = env.transaction_count
        self.assertEqual(env.get_action_mask()["high"].sum(), 0.0)
        env.step(np.array([0.5, 0.5, 0.5]))
        self.assertEqual(env.transaction_count, trade_count)
        env.step(np.array([0.5, 0.5, 0.5]))
        self.assertEqual(env.transaction_count, trade_count)
        env.step(np.array([-0.5, -0.5, -0.5]))
        self.assertGreater(env.transaction_count, trade_count)

    def test_atr_stop_loss_closes_a_deteriorating_position(self):
        env = MultiAssetTradingEnvironment(
            data_dict=self.test_data,
            rebalance_frequency=1,
            atr_stop_multiplier=99.0,
            atr_trailing_multiplier=99.0,
            max_stop_loss_pct=0.1,
        )
        env.reset()
        env.step(np.array([1.0, 0.0, 0.0]))
        entry_price = env.position_entry_prices["BTC"]
        env.data_dict["BTC"].iloc[env.current_step, env.data_dict["BTC"].columns.get_loc("close")] = entry_price * 0.85
        env.step(np.zeros(3))
        self.assertEqual(env.crypto_holdings["BTC"], 0.0)
        self.assertIn("atr_stop_loss", [event["reason"] for event in env.trade_events])

    def test_max_asset_exposure_caps_a_single_all_in_action(self):
        env = MultiAssetTradingEnvironment(data_dict=self.test_data, max_asset_exposure=0.35)
        env.reset()
        projected = env.project_action(np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(projected[0], 0.35)

    def test_portfolio_value_calculation(self):
        """Teste le calcul de la valeur du portefeuille."""
        env = MultiAssetTradingEnvironment(data_dict=self.test_data)

        # Réinitialiser l'environnement
        env.reset()

        # Valeur initiale du portefeuille = solde initial
        initial_value = env.get_portfolio_value()
        self.assertEqual(initial_value, 10000.0)

        # Modifier manuellement les détentions de crypto pour tester
        btc_price = env.data_dict["BTC"].iloc[env.current_step]["close"]
        eth_price = env.data_dict["ETH"].iloc[env.current_step]["close"]

        env.crypto_holdings["BTC"] = 1.0  # Détenir 1 BTC
        env.crypto_holdings["ETH"] = 10.0  # Détenir 10 ETH
        env.balance = 5000.0  # Solde réduit

        # Calculer la valeur attendue du portefeuille
        expected_value = 5000.0 + (1.0 * btc_price) + (10.0 * eth_price)

        # Vérifier la valeur calculée
        calculated_value = env.get_portfolio_value()
        self.assertAlmostEqual(calculated_value, expected_value, places=5)

    def test_different_allocation_methods(self):
        """Teste différentes méthodes d'allocation du portefeuille."""
        methods = ["equal", "volatility", "momentum", "smart"]

        for method in methods:
            env = MultiAssetTradingEnvironment(
                data_dict=self.test_data,
                allocation_method=method,
                rebalance_frequency=1,
            )

            # Réinitialiser l'environnement
            env.reset()

            # Exécuter une action d'achat
            action = np.array([0.8, 0.8, 0.8])
            obs, reward, done, _, info = env.step(action)

            # Vérifier que l'allocation a été effectuée (on ne teste pas la valeur exacte)
            portfolio_value = env.get_portfolio_value()
            self.assertGreater(
                portfolio_value, 0.0
            )  # Simplement vérifier que la valeur est positive

            # Vérifier que l'allocation a été enregistrée
            self.assertEqual(len(env.allocation_history), 1)

            # Vérifier que l'allocation est valide (la somme des poids = 1)
            weights_sum = sum(env.allocation_history[0].values())
            self.assertAlmostEqual(weights_sum, 1.0, places=5)

    def test_portfolio_allocator(self):
        """Teste la classe PortfolioAllocator séparément."""
        allocator = PortfolioAllocator(method="equal", max_active_positions=2)

        # Tester l'allocation égale avec des poids clairement différents
        action_weights = np.array(
            [0.1, 0.8, 0.9]
        )  # LTC est clairement le meilleur, ETH le second
        symbols = ["BTC", "ETH", "LTC"]
        allocation = allocator.allocate(action_weights, symbols)

        # Vérifier que seuls les 2 meilleurs actifs sont inclus
        active_assets = [symbol for symbol, weight in allocation.items() if weight > 0]
        self.assertEqual(len(active_assets), 2)

        # Vérifier que les 2 actifs avec les poids les plus élevés sont retenus
        self.assertIn("ETH", active_assets)
        self.assertIn("LTC", active_assets)
        self.assertNotIn("BTC", active_assets)

        # Vérifier que chaque actif actif a le même poids
        active_weight = 0.5  # 1.0 / 2 actifs
        for symbol in active_assets:
            self.assertAlmostEqual(allocation[symbol], active_weight, places=5)

        # Tester avec volatilités
        volatilities = {"BTC": 0.5, "ETH": 0.3, "LTC": 0.2}
        allocator = PortfolioAllocator(method="volatility", max_active_positions=3)
        allocation = allocator.allocate(
            action_weights, symbols, volatilities=volatilities
        )

        # Vérifier que l'actif avec la volatilité la plus faible a plus de poids quand l'action_weight est similaire
        self.assertGreater(allocation["LTC"], allocation["ETH"])
        self.assertGreater(allocation["ETH"], allocation["BTC"])

    def test_execution_delay_and_advanced_rewards_use_real_history(self):
        env = MultiAssetTradingEnvironment(
            data_dict=self.test_data, execution_delay=1, reward_function="sortino"
        )
        env.reset(seed=3)
        env.step(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(env.crypto_holdings["BTC"], 0.0)
        self.assertEqual(len(env.pending_orders), 1)
        self.assertGreater(env.reserved_cash, 0.0)
        env.step(np.zeros(3))
        self.assertGreater(env.crypto_holdings["BTC"], 0.0)
        self.assertEqual(env.reserved_cash, 0.0)
        self.assertEqual(len(env.returns_history), 2)
        self.assertTrue(np.isfinite(env._drawdown_reward(0.0)))


if __name__ == "__main__":
    unittest.main()
