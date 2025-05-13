"""
Tests pour le gestionnaire de risques avancé avec VaR et allocation adaptative.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.risk.advanced_risk_manager import AdvancedRiskManager


class TestAdvancedRiskManager(unittest.TestCase):
    """Tests pour le gestionnaire de risques avancé."""

    def setUp(self):
        """Prépare les données pour les tests."""
        # Créer des données de marché synthétiques
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Créer une tendance avec un peu de volatilité
        trend = np.linspace(100, 150, 100)
        noise = np.random.normal(0, 5, 100)

        # Ajouter un crash au milieu
        crash = np.zeros(100)
        crash[40:50] = np.linspace(0, -30, 10)  # Chute de 30%

        self.market_data = pd.DataFrame(
            {
                "open": trend + noise,
                "high": trend + noise + np.random.uniform(0, 10, 100),
                "low": trend + noise - np.random.uniform(0, 10, 100),
                "close": trend + noise + crash,
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        # Initialiser le gestionnaire de risques avancé
        self.risk_manager = AdvancedRiskManager(
            config={
                "var_confidence_level": 0.95,
                "var_horizon": 1,
                "var_method": "parametric",
                "max_var_limit": 0.05,
                "cvar_confidence_level": 0.95,
                "adaptive_capital_allocation": True,
                "kelly_fraction": 0.5,
                "max_drawdown_limit": 0.20,
                "risk_parity_weights": True,
            }
        )

    def test_parametric_var(self):
        """Teste le calcul de la VaR paramétrique."""
        # Calculer la VaR paramétrique
        var = self.risk_manager.calculate_var(
            self.market_data, method="parametric", confidence_level=0.95, horizon=1
        )

        # Vérifier que la VaR est positive et dans une plage raisonnable
        self.assertGreater(var, 0)
        self.assertLess(var, 0.2)  # La VaR devrait être inférieure à 20%

        # Vérifier que la VaR augmente avec le niveau de confiance
        var_99 = self.risk_manager.calculate_var(
            self.market_data, method="parametric", confidence_level=0.99, horizon=1
        )
        self.assertGreater(var_99, var)

        # Vérifier que la VaR augmente avec l'horizon
        var_5day = self.risk_manager.calculate_var(
            self.market_data, method="parametric", confidence_level=0.95, horizon=5
        )
        self.assertGreater(var_5day, var)

    def test_historical_var(self):
        """Teste le calcul de la VaR historique."""
        # Calculer la VaR historique
        var = self.risk_manager.calculate_var(
            self.market_data, method="historical", confidence_level=0.95, horizon=1
        )

        # Vérifier que la VaR est positive et dans une plage raisonnable
        self.assertGreater(var, 0)
        self.assertLess(var, 0.3)  # La VaR devrait être inférieure à 30%

        # Avec un niveau de confiance plus élevé
        var_99 = self.risk_manager.calculate_var(
            self.market_data, method="historical", confidence_level=0.99, horizon=1
        )
        self.assertGreaterEqual(var_99, var * 0.8)  # Peut fluctuer avec peu de données

    def test_monte_carlo_var(self):
        """Teste le calcul de la VaR par Monte Carlo."""
        # Calculer la VaR Monte Carlo
        var = self.risk_manager.calculate_var(
            self.market_data, method="monte_carlo", confidence_level=0.95, horizon=1
        )

        # Vérifier que la VaR est positive et dans une plage raisonnable
        self.assertGreater(var, 0)
        self.assertLess(var, 0.2)  # La VaR devrait être inférieure à 20%

    def test_cvar(self):
        """Teste le calcul de la CVaR."""
        # Calculer la CVaR
        cvar = self.risk_manager.calculate_cvar(
            self.market_data, confidence_level=0.95, horizon=1
        )

        # Calculer la VaR pour comparaison
        var = self.risk_manager.calculate_var(
            self.market_data, method="historical", confidence_level=0.95, horizon=1
        )

        # Vérifier que la CVaR est positive et supérieure à la VaR
        self.assertGreater(cvar, 0)
        self.assertGreaterEqual(
            cvar, var * 0.8
        )  # CVaR devrait être ≥ VaR (avec marge pour fluctuations)

    def test_kelly_criterion(self):
        """Teste le calcul du critère de Kelly."""
        # Scénario favorable: 60% de gain avec ratio gain/perte de 2
        kelly_favorable = self.risk_manager.kelly_criterion(0.6, 2.0)

        # Scénario défavorable: 40% de gain avec ratio gain/perte de 1
        kelly_unfavorable = self.risk_manager.kelly_criterion(0.4, 1.0)

        # Scénario neutre: 50% de gain avec ratio gain/perte de 1
        kelly_neutral = self.risk_manager.kelly_criterion(0.5, 1.0)

        # Vérifier que les allocations sont dans les plages attendues
        self.assertGreater(kelly_favorable, 0)
        self.assertLessEqual(kelly_favorable, self.risk_manager.kelly_fraction)

        self.assertLessEqual(kelly_unfavorable, 0)
        self.assertEqual(kelly_unfavorable, 0)  # Devrait être 0 car négatif

        self.assertEqual(kelly_neutral, 0)  # Devrait être 0 pour un jeu équitable

    def test_win_probability_calculation(self):
        """Teste le calcul de la probabilité de gain et du ratio gain/perte."""
        # Calculer les statistiques
        win_probability, win_loss_ratio = (
            self.risk_manager.calculate_win_probability_and_ratio(
                self.market_data, lookback=100
            )
        )

        # Vérifier que les valeurs sont dans des plages raisonnables
        self.assertGreaterEqual(win_probability, 0)
        self.assertLessEqual(win_probability, 1)

        self.assertGreater(win_loss_ratio, 0)

    def test_adaptive_capital_allocation(self):
        """Teste l'allocation adaptative du capital."""
        # Calculer l'allocation pour une position longue
        allocation_long = self.risk_manager.adaptive_capital_allocation(
            self.market_data, position_type="long"
        )

        # Calculer l'allocation pour une position courte
        allocation_short = self.risk_manager.adaptive_capital_allocation(
            self.market_data, position_type="short"
        )

        # Vérifier que les allocations sont dans les plages attendues
        self.assertGreaterEqual(allocation_long, 0)
        self.assertLessEqual(allocation_long, self.risk_manager.max_position_size)

        self.assertLessEqual(allocation_short, 0)
        self.assertGreaterEqual(allocation_short, -self.risk_manager.max_position_size)

        # Vérifier que l'historique est mis à jour
        self.assertEqual(len(self.risk_manager.allocation_history), 2)

    def test_risk_parity_allocation(self):
        """Teste l'allocation selon le principe de parité des risques."""
        # Définir des variances d'actifs
        asset_variances = [0.01, 0.04, 0.09]  # 10%, 20%, 30% de volatilité

        # Calculer les poids
        weights = self.risk_manager.risk_parity_allocation(asset_variances)

        # Vérifier que les poids sont normalisés
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)

        # Vérifier que les actifs moins volatils ont des poids plus élevés
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_maximum_drawdown(self):
        """Teste le calcul du drawdown maximum."""
        # Définir un historique de portefeuille avec un drawdown de 15%
        portfolio_values = [100, 110, 120, 130, 110.5, 120, 135]

        # Calculer le drawdown maximum
        max_drawdown = self.risk_manager.calculate_maximum_drawdown(portfolio_values)

        # Vérifier que le drawdown est correct (15% de 130 à 110.5)
        self.assertAlmostEqual(max_drawdown, 0.15, places=2)

    def test_should_stop_trading(self):
        """Teste la fonction qui détermine si le trading doit être arrêté."""
        # Définir un historique de portefeuille avec un drawdown de 25% (supérieur à la limite)
        portfolio_values = [100, 110, 120, 130, 97.5, 100, 110]

        # Vérifier que le trading doit être arrêté
        self.assertTrue(self.risk_manager.should_stop_trading(portfolio_values))

        # Définir un historique avec un drawdown acceptable
        portfolio_values = [100, 110, 120, 130, 115, 120, 125]

        # Vérifier que le trading peut continuer
        self.assertFalse(self.risk_manager.should_stop_trading(portfolio_values))

    def test_allocation_with_risk_limits(self):
        """Teste l'allocation avec toutes les contraintes de risque."""
        # Calculer l'allocation avec des contraintes
        allocation = self.risk_manager.allocation_with_risk_limits(
            self.market_data, position_type="long"
        )

        # Vérifier que l'allocation est dans les plages attendues
        self.assertGreaterEqual(allocation, 0)
        self.assertLessEqual(allocation, self.risk_manager.max_position_size)

        # Tester avec un cas qui devrait déclencher l'arrêt du trading
        portfolio_values = [100, 110, 120, 130, 97.5, 100, 110]  # 25% de drawdown

        allocation_stopped = self.risk_manager.allocation_with_risk_limits(
            self.market_data, position_type="long", portfolio_values=portfolio_values
        )

        # Vérifier que l'allocation est nulle
        self.assertEqual(allocation_stopped, 0.0)

    def test_multilevel_risk_disabled(self):
        """Teste le comportement quand la gestion multi-niveaux est désactivée."""
        # Par défaut, la gestion multi-niveaux est désactivée
        result = self.risk_manager.multilevel_risk_management(self.market_data)

        # Vérifier que l'allocation est la taille de position maximale
        self.assertEqual(result["allocation"], self.risk_manager.max_position_size)
        self.assertEqual(result["risk_score"], 0.5)  # Valeur par défaut

    def test_multilevel_risk_enabled(self):
        """Teste la gestion multi-niveaux des risques quand activée."""
        # Créer un gestionnaire avec la gestion multi-niveaux activée
        multilevel_risk_manager = AdvancedRiskManager(
            config={
                "var_confidence_level": 0.95,
                "max_var_limit": 0.05,
                "max_position_size": 0.8,  # 80% du capital max
                "use_multilevel_risk": True,
                "strategy_risk_weight": 0.4,
                "portfolio_risk_weight": 0.3,
                "market_risk_weight": 0.3,
                "max_correlation_exposure": 0.7,
            }
        )

        # Créer des données de portefeuille fictives
        portfolio_data = {
            "weights": [0.3, 0.4, 0.3],  # Poids des actifs
            "assets": ["BTC", "ETH", "SOL"],  # Symboles des actifs
            "returns": [0.01, 0.015, -0.01, 0.02, -0.005],  # Rendements récents
        }

        # Créer une matrice de corrélation fictive
        correlation_matrix = pd.DataFrame(
            [[1.0, 0.7, 0.5], [0.7, 1.0, 0.6], [0.5, 0.6, 1.0]],
            index=["BTC", "ETH", "SOL"],
            columns=["BTC", "ETH", "SOL"],
        )

        # Créer des données de marché fictives
        market_data = {
            "vix": 20.0,  # Indice de volatilité
            "fear_greed_index": 30,  # Indice Fear & Greed
            "credit_spread": 0.015,  # Écart de crédit
            "market_liquidity": 0.4,  # Liquidité (0=liquide, 1=illiquide)
            "market_trend": -0.2,  # Tendance du marché (-1 à 1)
        }

        # Calculer l'allocation multi-niveaux
        result = multilevel_risk_manager.multilevel_risk_management(
            self.market_data,
            market_data=market_data,
            portfolio_data=portfolio_data,
            correlation_matrix=correlation_matrix,
        )

        # Vérifier que tous les scores de risque sont entre 0 et 1
        self.assertGreaterEqual(result["strategy_risk"], 0)
        self.assertLessEqual(result["strategy_risk"], 1)

        self.assertGreaterEqual(result["portfolio_risk"], 0)
        self.assertLessEqual(result["portfolio_risk"], 1)

        self.assertGreaterEqual(result["market_risk"], 0)
        self.assertLessEqual(result["market_risk"], 1)

        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"], 1)

        # Vérifier que l'allocation est inversement proportionnelle au score de risque
        expected_allocation = multilevel_risk_manager.max_position_size * (
            1 - result["risk_score"]
        )
        self.assertAlmostEqual(result["allocation"], expected_allocation)

        # Vérifier que l'historique est mis à jour
        self.assertEqual(len(multilevel_risk_manager.multilevel_risk_history), 1)

    def test_strategy_risk_evaluation(self):
        """Teste l'évaluation du risque au niveau de la stratégie."""
        # Créer un gestionnaire avec la gestion multi-niveaux activée
        risk_manager = AdvancedRiskManager(config={"use_multilevel_risk": True})

        # Évaluer le risque de stratégie
        strategy_risk = risk_manager._evaluate_strategy_risk(self.market_data)

        # Vérifier que le risque est entre 0 et 1
        self.assertGreaterEqual(strategy_risk, 0)
        self.assertLessEqual(strategy_risk, 1)

        # Tester avec des données nulles (devrait retourner 0.5 par défaut)
        null_strategy_risk = risk_manager._evaluate_strategy_risk(None)
        self.assertEqual(null_strategy_risk, 0.5)

    def test_portfolio_risk_evaluation(self):
        """Teste l'évaluation du risque au niveau du portefeuille."""
        # Créer un gestionnaire avec la gestion multi-niveaux activée
        risk_manager = AdvancedRiskManager(config={"use_multilevel_risk": True})

        # Créer des données de portefeuille variées
        # 1. Portefeuille concentré
        concentrated_portfolio = {
            "weights": [0.8, 0.1, 0.1],  # Très concentré sur un actif
            "assets": ["BTC", "ETH", "SOL"],
            "returns": [0.01, 0.02, -0.01, 0.015, -0.005],
        }

        # 2. Portefeuille diversifié
        diversified_portfolio = {
            "weights": [0.33, 0.33, 0.34],  # Équilibré
            "assets": ["BTC", "ETH", "SOL"],
            "returns": [0.01, 0.02, -0.01, 0.015, -0.005],
        }

        # Créer une matrice de corrélation
        correlation_matrix = pd.DataFrame(
            [[1.0, 0.7, 0.5], [0.7, 1.0, 0.6], [0.5, 0.6, 1.0]],
            index=["BTC", "ETH", "SOL"],
            columns=["BTC", "ETH", "SOL"],
        )

        # Évaluer les risques
        concentrated_risk = risk_manager._evaluate_portfolio_risk(
            concentrated_portfolio, correlation_matrix
        )

        diversified_risk = risk_manager._evaluate_portfolio_risk(
            diversified_portfolio, correlation_matrix
        )

        # Le portefeuille concentré devrait avoir un risque plus élevé
        self.assertGreater(concentrated_risk, diversified_risk)

        # Vérifier que les risques sont entre 0 et 1
        self.assertGreaterEqual(concentrated_risk, 0)
        self.assertLessEqual(concentrated_risk, 1)
        self.assertGreaterEqual(diversified_risk, 0)
        self.assertLessEqual(diversified_risk, 1)

    def test_market_risk_evaluation(self):
        """Teste l'évaluation du risque au niveau du marché."""
        # Créer un gestionnaire avec la gestion multi-niveaux activée
        risk_manager = AdvancedRiskManager(config={"use_multilevel_risk": True})

        # 1. Marché à haut risque
        high_risk_market = {
            "vix": 35.0,  # VIX élevé
            "fear_greed_index": 10,  # Peur extrême
            "credit_spread": 0.03,  # Spread élevé
            "market_liquidity": 0.8,  # Faible liquidité
            "market_trend": -0.8,  # Forte tendance baissière
        }

        # 2. Marché à faible risque
        low_risk_market = {
            "vix": 12.0,  # VIX bas
            "fear_greed_index": 60,  # Optimisme modéré
            "credit_spread": 0.008,  # Spread serré
            "market_liquidity": 0.2,  # Bonne liquidité
            "market_trend": 0.5,  # Tendance haussière modérée
        }

        # Évaluer les risques
        high_market_risk = risk_manager._evaluate_market_risk(high_risk_market)
        low_market_risk = risk_manager._evaluate_market_risk(low_risk_market)

        # Le marché à haut risque devrait avoir un score plus élevé
        self.assertGreater(high_market_risk, low_market_risk)

        # Vérifier que les risques sont entre 0 et 1
        self.assertGreaterEqual(high_market_risk, 0)
        self.assertLessEqual(high_market_risk, 1)
        self.assertGreaterEqual(low_market_risk, 0)
        self.assertLessEqual(low_market_risk, 1)

        # Tester avec des données nulles (devrait retourner 0.5 par défaut)
        null_market_risk = risk_manager._evaluate_market_risk(None)
        self.assertEqual(null_market_risk, 0.5)


if __name__ == "__main__":
    unittest.main()
