import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ai_trading.rl.multi_asset_trading import MultiAssetTradingSystem


class TestMultiAssetTradingSystem(unittest.TestCase):
    def setUp(self):
        """Initialise l'environnement de test."""
        self.crypto_assets = ["BTC", "ETH"]
        self.traditional_assets = ["XAU/USD", "AAPL", "NVDA"]
        self.system = MultiAssetTradingSystem(
            crypto_assets=self.crypto_assets,
            traditional_assets=self.traditional_assets,
            initial_balance=10000.0,
            risk_per_trade=0.02,
            max_position_size=0.2,
        )

        # Créer des données de marché synthétiques
        self.market_data = {}
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        for asset in self.crypto_assets + self.traditional_assets:
            self.market_data[asset] = pd.DataFrame(
                {
                    "open": np.random.uniform(100, 200, 100),
                    "high": np.random.uniform(200, 300, 100),
                    "low": np.random.uniform(50, 100, 100),
                    "close": np.random.uniform(100, 200, 100),
                    "volume": np.random.uniform(1000, 5000, 100),
                },
                index=dates,
            )

    def test_initialization(self):
        """Teste l'initialisation du système."""
        self.assertEqual(len(self.system.assets), 5)
        self.assertEqual(self.system.initial_balance, 10000.0)
        self.assertEqual(self.system.risk_per_trade, 0.02)
        self.assertEqual(self.system.max_position_size, 0.2)
        self.assertEqual(len(self.system.positions), 5)
        self.assertEqual(len(self.system.prices), 5)

    def test_collect_market_data(self):
        """Teste la collecte des données de marché."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        market_data = self.system.collect_market_data(start_date, end_date)

        self.assertIsInstance(market_data, dict)
        for asset in self.system.assets:
            self.assertIn(asset, market_data)
            self.assertIsInstance(market_data[asset], pd.DataFrame)

    def test_calculate_portfolio_metrics(self):
        """Teste le calcul des métriques du portefeuille."""
        # Mettre à jour les prix
        for asset in self.system.assets:
            self.system.prices[asset] = 100.0

        metrics = self.system.calculate_portfolio_metrics()

        self.assertIn("total_value", metrics)
        self.assertIn("return", metrics)
        self.assertIn("positions", metrics)
        self.assertIn("prices", metrics)
        self.assertEqual(metrics["total_value"], self.system.initial_balance)

    def test_update_positions(self):
        """Teste la mise à jour des positions."""
        # Initialiser les prix
        for asset in self.system.assets:
            self.system.prices[asset] = 100.0

        # Créer des actions
        actions = {asset: 0.5 for asset in self.system.assets}

        # Mettre à jour les positions
        self.system.update_positions(actions)

        # Vérifier que les positions ont été mises à jour
        for asset in self.system.assets:
            self.assertNotEqual(self.system.positions[asset], 0.0)

    def test_calculate_simple_allocation(self):
        """Test de l'allocation simple du portefeuille."""
        # Test avec un mix d'actifs
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
            risk_per_trade=0.02,
            max_position_size=0.2,
        )

        allocation = system.calculate_simple_allocation()

        # Vérifier que la somme des allocations est égale à 1
        assert sum(allocation.values()) == 1.0

        # Vérifier que chaque actif a la même allocation
        expected_share = 1.0 / 4  # 4 actifs au total
        for share in allocation.values():
            assert abs(share - expected_share) < 1e-10

        # Test avec uniquement des crypto-monnaies
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH", "SOL"],
            traditional_assets=[],
            initial_balance=10000.0,
        )

        allocation = system.calculate_simple_allocation()
        assert sum(allocation.values()) == 1.0
        expected_share = 1.0 / 3
        for share in allocation.values():
            assert abs(share - expected_share) < 1e-10

        # Test avec uniquement des actifs traditionnels
        system = MultiAssetTradingSystem(
            crypto_assets=[],
            traditional_assets=["AAPL", "NVDA", "MSFT"],
            initial_balance=10000.0,
        )

        allocation = system.calculate_simple_allocation()
        assert sum(allocation.values()) == 1.0
        expected_share = 1.0 / 3
        for share in allocation.values():
            assert abs(share - expected_share) < 1e-10

    def test_custom_allocation(self):
        """Test de l'allocation personnalisée."""
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
        )

        # Test d'une allocation personnalisée valide
        custom_weights = {"BTC": 0.4, "ETH": 0.3, "XAU/USD": 0.2, "AAPL": 0.1}
        system.set_custom_allocation(custom_weights)
        current_allocation = system.get_current_allocation()
        assert current_allocation == custom_weights

        # Test avec un actif non reconnu
        with pytest.raises(ValueError):
            system.set_custom_allocation({"UNKNOWN": 1.0})

        # Test avec une somme de poids incorrecte
        with pytest.raises(ValueError):
            system.set_custom_allocation({"BTC": 0.5, "ETH": 0.6})

    def test_volatility_based_allocation(self):
        """Test de l'allocation basée sur la volatilité."""
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
        )

        # Créer des données de marché avec des volatilités différentes
        market_data = {}
        for asset in system.assets:
            # Générer des rendements avec des volatilités différentes
            if asset == "BTC":
                returns = np.random.normal(0, 0.05, 100)  # Haute volatilité
            elif asset == "ETH":
                returns = np.random.normal(0, 0.04, 100)  # Volatilité moyenne
            elif asset == "XAU/USD":
                returns = np.random.normal(0, 0.02, 100)  # Faible volatilité
            else:  # AAPL
                returns = np.random.normal(0, 0.03, 100)  # Volatilité moyenne

            # Créer un DataFrame avec les prix
            prices = np.cumprod(1 + returns) * 100
            market_data[asset] = pd.DataFrame({"close": prices})

        # Calculer l'allocation
        allocation = system.calculate_volatility_based_allocation(market_data)

        # Vérifier que la somme est égale à 1
        assert abs(sum(allocation.values()) - 1.0) < 1e-10

        # Vérifier que XAU/USD a la plus grande allocation (volatilité la plus faible)
        assert allocation["XAU/USD"] > allocation["BTC"]

    def test_correlation_based_allocation(self):
        """Test de l'allocation basée sur la corrélation."""
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
        )

        # Créer des données de marché avec des corrélations différentes
        market_data = {}
        base_returns = np.random.normal(0, 0.02, 100)

        for asset in system.assets:
            if asset == "BTC":
                returns = base_returns  # Fortement corrélé avec ETH
            elif asset == "ETH":
                returns = base_returns + np.random.normal(
                    0, 0.01, 100
                )  # Fortement corrélé avec BTC
            elif asset == "XAU/USD":
                returns = np.random.normal(0, 0.02, 100)  # Peu corrélé
            else:  # AAPL
                returns = np.random.normal(0, 0.02, 100)  # Peu corrélé

            prices = np.cumprod(1 + returns) * 100
            market_data[asset] = pd.DataFrame({"close": prices})

        # Calculer l'allocation
        allocation = system.calculate_correlation_based_allocation(market_data)

        # Vérifier que la somme est égale à 1
        assert abs(sum(allocation.values()) - 1.0) < 1e-10

        # Vérifier que XAU/USD et AAPL ont des allocations plus grandes que BTC et ETH
        assert allocation["XAU/USD"] > allocation["BTC"]
        assert allocation["AAPL"] > allocation["ETH"]

    def test_risk_parity_allocation(self):
        """Test de l'allocation de parité de risque."""
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
        )

        # Créer des données de marché avec des caractéristiques de risque différentes
        market_data = {}
        for asset in system.assets:
            if asset == "BTC":
                returns = np.random.normal(0, 0.05, 100)  # Haut risque
            elif asset == "ETH":
                returns = np.random.normal(0, 0.04, 100)  # Risque moyen
            elif asset == "XAU/USD":
                returns = np.random.normal(0, 0.02, 100)  # Faible risque
            else:  # AAPL
                returns = np.random.normal(0, 0.03, 100)  # Risque moyen

            prices = np.cumprod(1 + returns) * 100
            market_data[asset] = pd.DataFrame({"close": prices})

        # Calculer l'allocation
        allocation = system.calculate_risk_parity_allocation(market_data)

        # Vérifier que la somme est égale à 1
        assert abs(sum(allocation.values()) - 1.0) < 1e-10

        # Vérifier que les allocations sont inversement proportionnelles au risque
        assert allocation["XAU/USD"] > allocation["BTC"]

    def test_adaptive_allocation(self):
        """Test de l'allocation adaptative."""
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
        )

        # Créer des données de marché avec des caractéristiques différentes
        market_data = {}
        base_returns = np.random.normal(0, 0.02, 100)

        for asset in system.assets:
            if asset == "BTC":
                returns = base_returns + np.random.normal(
                    0, 0.03, 100
                )  # Haut risque, fortement corrélé
            elif asset == "ETH":
                returns = base_returns + np.random.normal(
                    0, 0.02, 100
                )  # Risque moyen, fortement corrélé
            elif asset == "XAU/USD":
                returns = np.random.normal(0, 0.01, 100)  # Faible risque, peu corrélé
            else:  # AAPL
                returns = np.random.normal(0, 0.015, 100)  # Risque faible, peu corrélé

            prices = np.cumprod(1 + returns) * 100
            market_data[asset] = pd.DataFrame({"close": prices})

        # Test avec les poids par défaut
        allocation = system.calculate_adaptive_allocation(market_data)
        assert abs(sum(allocation.values()) - 1.0) < 1e-10

        # Test avec des poids personnalisés
        custom_weights = {"volatility": 0.5, "correlation": 0.3, "risk_parity": 0.2}
        custom_allocation = system.calculate_adaptive_allocation(
            market_data, custom_weights
        )
        assert abs(sum(custom_allocation.values()) - 1.0) < 1e-10

        # Vérifier que XAU/USD a une allocation plus grande que BTC
        # car il est moins risqué et moins corrélé
        assert custom_allocation["XAU/USD"] > custom_allocation["BTC"]

    def test_portfolio_rebalancing(self):
        """Test du rééquilibrage automatique du portefeuille."""
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
        )

        # Créer des données de marché
        market_data = {}
        for asset in system.assets:
            prices = np.random.uniform(100, 200, 100)
            market_data[asset] = pd.DataFrame({"close": prices})
            system.prices[asset] = prices[-1]  # Dernier prix

        # Initialiser une allocation déséquilibrée
        desequilibree = {
            "BTC": 0.4,  # 40%
            "ETH": 0.3,  # 30%
            "XAU/USD": 0.2,  # 20%
            "AAPL": 0.1,  # 10%
        }
        system.set_custom_allocation(desequilibree)

        # Vérifier que le rééquilibrage est nécessaire
        current_allocation = system.get_current_allocation()
        target_allocation = system.calculate_adaptive_allocation(market_data)
        assert system.needs_rebalancing(current_allocation, target_allocation)

        # Effectuer le rééquilibrage
        system.rebalance_portfolio(market_data)

        # Vérifier que les positions ont été ajustées
        new_allocation = system.get_current_allocation()
        for asset in system.assets:
            assert (
                abs(new_allocation[asset] - target_allocation[asset]) < 0.1
            )  # 10% de tolérance

        # Vérifier que la somme des positions est égale à 1
        assert abs(sum(new_allocation.values()) - 1.0) < 1e-10

    def test_dynamic_weights(self):
        """Test des poids dynamiques et des métriques de marché."""
        system = MultiAssetTradingSystem(
            crypto_assets=["BTC", "ETH"],
            traditional_assets=["XAU/USD", "AAPL"],
            initial_balance=10000.0,
        )

        # Créer des données de marché avec différentes conditions
        market_data = {}
        base_returns = np.random.normal(0, 0.02, 100)

        # Scénario 1: Marché volatil et baissier
        for asset in system.assets:
            if asset == "BTC":
                returns = base_returns + np.random.normal(
                    -0.01, 0.05, 100
                )  # Haut risque, tendance baissière
            elif asset == "ETH":
                returns = base_returns + np.random.normal(
                    -0.01, 0.04, 100
                )  # Risque moyen, tendance baissière
            elif asset == "XAU/USD":
                returns = np.random.normal(
                    -0.005, 0.02, 100
                )  # Faible risque, légère baisse
            else:  # AAPL
                returns = np.random.normal(
                    -0.005, 0.03, 100
                )  # Risque moyen, légère baisse

            prices = np.cumprod(1 + returns) * 100
            market_data[asset] = pd.DataFrame({"close": prices})

        # Calculer les poids dynamiques
        weights = system.calculate_dynamic_weights(market_data)

        # Vérifier que les poids sont normalisés
        assert abs(sum(weights.values()) - 1.0) < 1e-10

        # Vérifier que le poids de la volatilité est augmenté en période de haute volatilité
        assert weights["volatility"] > 0.4

        # Scénario 2: Marché stable et haussier
        market_data = {}
        base_returns = np.random.normal(
            0.01, 0.01, 100
        )  # Tendance haussière, faible volatilité

        for asset in system.assets:
            if asset == "BTC":
                returns = base_returns + np.random.normal(0.005, 0.02, 100)
            elif asset == "ETH":
                returns = base_returns + np.random.normal(0.005, 0.02, 100)
            elif asset == "XAU/USD":
                returns = base_returns + np.random.normal(0.005, 0.01, 100)
            else:  # AAPL
                returns = base_returns + np.random.normal(0.005, 0.015, 100)

            prices = np.cumprod(1 + returns) * 100
            market_data[asset] = pd.DataFrame({"close": prices})

        # Calculer les poids dynamiques
        weights = system.calculate_dynamic_weights(market_data)

        # Vérifier que les poids sont normalisés
        assert abs(sum(weights.values()) - 1.0) < 1e-10

        # Vérifier que le poids de la corrélation est augmenté en période stable
        assert weights["correlation"] > 0.3

    @pytest.mark.slow
    def test_train(self):
        """Teste l'entraînement des systèmes de trading."""
        self.system.train(self.market_data, epochs=2)

        # Vérifier que les systèmes de trading ont été créés
        for asset in self.system.assets:
            self.assertIn(asset, self.system.trading_systems)

    @pytest.mark.slow
    def test_predict_actions(self):
        """Teste la prédiction des actions."""
        # Entraîner d'abord
        self.system.train(self.market_data, epochs=2)

        # Prédire les actions
        actions = self.system.predict_actions(self.market_data)

        # Vérifier les actions
        self.assertIsInstance(actions, dict)
        for asset in self.system.assets:
            self.assertIn(asset, actions)
            self.assertIsInstance(actions[asset], float)
            self.assertTrue(-1 <= actions[asset] <= 1)


if __name__ == "__main__":
    unittest.main()
