import unittest

import numpy as np
import pandas as pd

from ..rl.multi_asset_trading_environment import MultiAssetTradingEnvironment


class TestMultiAssetTradingEnvironment(unittest.TestCase):
    def setUp(self):
        """Prépare l'environnement de test."""
        # Création de données de test
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        # Création de données corrélées pour tester la diversification
        base_prices = np.random.uniform(45000, 55000, 100)
        eth_correlation = 0.8  # Forte corrélation avec BTC
        xrp_correlation = 0.2  # Faible corrélation avec BTC

        self.data_dict = {
            "BTC/USDT": pd.DataFrame(
                {
                    "open": base_prices,
                    "high": base_prices * 1.02,
                    "low": base_prices * 0.98,
                    "close": base_prices,
                    "volume": np.random.uniform(1000, 5000, 100),
                    "orderbook_depth": [
                        {
                            "spread_pct": 0.1,
                            "total_volume": 1000.0,
                            "volume_imbalance": 0.2,
                        }
                    ]
                    * 100,
                },
                index=dates,
            ),
            "ETH/USDT": pd.DataFrame(
                {
                    "open": base_prices * eth_correlation
                    + np.random.normal(0, 1000, 100),
                    "high": base_prices * eth_correlation * 1.02
                    + np.random.normal(0, 1000, 100),
                    "low": base_prices * eth_correlation * 0.98
                    + np.random.normal(0, 1000, 100),
                    "close": base_prices * eth_correlation
                    + np.random.normal(0, 1000, 100),
                    "volume": np.random.uniform(5000, 10000, 100),
                    "orderbook_depth": [
                        {
                            "spread_pct": 0.15,
                            "total_volume": 2000.0,
                            "volume_imbalance": -0.1,
                        }
                    ]
                    * 100,
                },
                index=dates,
            ),
            "XRP/USDT": pd.DataFrame(
                {
                    "open": base_prices * xrp_correlation
                    + np.random.normal(0, 100, 100),
                    "high": base_prices * xrp_correlation * 1.02
                    + np.random.normal(0, 100, 100),
                    "low": base_prices * xrp_correlation * 0.98
                    + np.random.normal(0, 100, 100),
                    "close": base_prices * xrp_correlation
                    + np.random.normal(0, 100, 100),
                    "volume": np.random.uniform(10000, 20000, 100),
                    "orderbook_depth": [
                        {
                            "spread_pct": 0.2,
                            "total_volume": 5000.0,
                            "volume_imbalance": 0.1,
                        }
                    ]
                    * 100,
                },
                index=dates,
            ),
        }

        self.env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            initial_balance=10000.0,
            window_size=10,
            slippage_model="dynamic",
            base_slippage=0.001,
            execution_delay=2,
            market_impact_factor=0.1,
            correlation_threshold=0.7,
            volatility_threshold=0.05,
        )

    def test_correlation_calculation(self):
        """Teste le calcul des corrélations entre actifs."""
        self.env._calculate_asset_correlations()
        correlations = self.env.asset_correlations

        # Vérification des corrélations
        self.assertGreater(
            correlations["BTC/USDT"]["ETH/USDT"],
            self.env.correlation_threshold,
            "BTC et ETH devraient être fortement corrélés",
        )
        self.assertLess(
            correlations["BTC/USDT"]["XRP/USDT"],
            self.env.correlation_threshold,
            "BTC et XRP devraient être faiblement corrélés",
        )

    def test_volatility_filtering(self):
        """Teste le filtrage des actifs basé sur la volatilité."""
        volatilities = self.env._calculate_asset_volatilities()
        filtered_assets = self.env._filter_assets()

        # Vérification que les actifs sont correctement filtrés
        for asset in self.data_dict.keys():
            if volatilities[asset] > self.env.volatility_threshold:
                self.assertIn(asset, filtered_assets)
            else:
                self.assertNotIn(asset, filtered_assets)

    def test_diversification_reward(self):
        """Teste la récompense de diversification."""
        # Test avec allocation équilibrée
        self.env.crypto_holdings = {
            "BTC/USDT": 0.1,
            "ETH/USDT": 1.67,
            "XRP/USDT": 5000,
        }

        base_reward = 1.0
        diversified_reward = self.env._diversification_reward(base_reward)

        # La récompense devrait être amplifiée pour une bonne diversification
        self.assertGreater(
            diversified_reward,
            base_reward,
            "La récompense devrait être amplifiée pour un portefeuille diversifié",
        )

        # Test avec allocation concentrée
        self.env.crypto_holdings = {
            "BTC/USDT": 0.3,
            "ETH/USDT": 0.0,
            "XRP/USDT": 0.0,
        }

        concentrated_reward = self.env._diversification_reward(base_reward)

        # La récompense devrait être réduite pour une mauvaise diversification
        self.assertLess(
            concentrated_reward,
            diversified_reward,
            "La récompense devrait être plus faible pour un portefeuille concentré",
        )

    def test_portfolio_rebalancing(self):
        """Teste le rééquilibrage du portefeuille."""
        # Configuration initiale
        self.env.reset()

        # Première allocation
        action1 = np.array([0.4, 0.3, 0.3])  # Allocation équilibrée
        _, _, _, _ = self.env.step(action1)

        initial_holdings = self.env.crypto_holdings.copy()

        # Deuxième allocation après changement de marché
        action2 = np.array([0.3, 0.4, 0.3])  # Modification de l'allocation
        _, _, _, _ = self.env.step(action2)

        # Vérifier que les holdings ont changé
        for asset in self.env.crypto_holdings:
            self.assertNotEqual(
                self.env.crypto_holdings[asset],
                initial_holdings[asset],
                f"Les holdings de {asset} devraient changer après le rééquilibrage",
            )

    def test_market_constraints_integration(self):
        """Teste l'intégration des contraintes de marché."""
        self.assertIsNotNone(self.env.market_constraints)
        self.assertEqual(self.env.market_constraints.slippage_model, "dynamic")
        self.assertEqual(self.env.market_constraints.base_slippage, 0.001)
        self.assertEqual(self.env.market_constraints.execution_delay, 2)
        self.assertEqual(self.env.market_constraints.market_impact_factor, 0.1)

    def test_delayed_execution(self):
        """Teste le délai d'exécution des ordres."""
        # Créer un ordre
        action = np.array([0.3, 0.0, 0.0])  # 30% en BTC
        initial_balance = self.env.balance

        # Premier pas
        _, _, _, _ = self.env.step(action)

        # Vérifier qu'un seul ordre est en attente
        self.assertEqual(len(self.env.pending_orders), 1)

        # Vérifier le délai de l'ordre
        order = self.env.pending_orders[0]
        self.assertEqual(order["delay"], self.env.market_constraints.execution_delay)

        # Attendre la moitié du délai
        for _ in range(self.env.market_constraints.execution_delay // 2):
            _, _, _, _ = self.env.step(np.zeros(3))
            self.assertEqual(len(self.env.pending_orders), 1)

        # Attendre le reste du délai
        for _ in range(self.env.market_constraints.execution_delay // 2 + 1):
            _, _, _, _ = self.env.step(np.zeros(3))

        # Vérifier que l'ordre a été exécuté
        self.assertEqual(len(self.env.pending_orders), 0)
        self.assertLess(self.env.balance, initial_balance)
        self.assertGreater(self.env.crypto_holdings["BTC/USDT"], 0)

    def test_slippage_impact(self):
        """Teste l'impact du slippage sur les prix d'exécution."""
        # Configuration initiale
        self.env.market_constraints.execution_delay = 0
        initial_balance = self.env.balance

        # Test avec différents niveaux d'achat
        test_sizes = [0.2, 0.5, 0.8]
        slippage_costs = []

        for size in test_sizes:
            # Réinitialiser l'environnement
            self.env.reset()
            self.env.balance = initial_balance

            # Exécuter l'action d'achat
            action = np.array([size, 0.0, 0.0])
            _, _, _, info = self.env.step(action)

            # Calculer le coût réel et théorique
            expected_cost = initial_balance * size
            actual_cost = initial_balance - self.env.balance
            slippage_cost = (actual_cost - expected_cost) / expected_cost
            slippage_costs.append(slippage_cost)

            # Vérifications
            self.assertGreater(actual_cost, expected_cost)
            self.assertLess(slippage_cost, 0.05)

        # Vérifier que le slippage augmente avec la taille
        self.assertLess(slippage_costs[0], slippage_costs[1])
        self.assertLess(slippage_costs[1], slippage_costs[2])

    def test_multi_order_processing(self):
        """Teste le traitement de plusieurs ordres simultanés."""
        # Créer plusieurs ordres
        action = np.array([0.4, 0.4, 0.0])  # 40% en BTC, 40% en ETH
        initial_balance = self.env.balance

        # Premier pas
        _, _, _, _ = self.env.step(action)

        # Vérifier que les deux ordres sont en attente
        self.assertEqual(len(self.env.pending_orders), 2)

        # Vérifier les délais
        for order in self.env.pending_orders:
            self.assertEqual(
                order["delay"], self.env.market_constraints.execution_delay
            )

        # Attendre l'exécution complète
        for _ in range(self.env.market_constraints.execution_delay + 1):
            _, _, _, _ = self.env.step(np.zeros(3))

        # Vérifier l'exécution
        self.assertEqual(len(self.env.pending_orders), 0)
        self.assertLess(self.env.balance, initial_balance)
        self.assertGreater(self.env.crypto_holdings["BTC/USDT"], 0)
        self.assertGreater(self.env.crypto_holdings["ETH/USDT"], 0)


if __name__ == "__main__":
    unittest.main()
