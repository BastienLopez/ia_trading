import os
import sys
import unittest

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.trading_environment import TradingEnvironment


class TestTradingEnvironment(unittest.TestCase):
    """Tests pour l'environnement de trading."""

    def setUp(self):
        """Prépare les données et l'environnement pour les tests."""
        # Créer des données synthétiques pour les tests
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Créer une tendance haussière simple
        rng = np.random.default_rng(42)
        prices = np.linspace(100, 200, 100) + rng.normal(0, 5, 100)

        # Créer un DataFrame avec les données
        self.test_data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + rng.uniform(0, 10, 100),
                "low": prices - rng.uniform(0, 10, 100),
                "close": prices + rng.normal(0, 3, 100),
                "volume": rng.uniform(1000, 5000, 100),
                "compound_score": rng.uniform(-1, 1, 100),  # Sentiment
            },
            index=dates,
        )

        # Créer l'environnement standard
        self.env = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            risk_management=False,
        )

        # Créer l'environnement avec actions discrètes nuancées
        self.env_discrete = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            action_type="discrete",
            n_discrete_actions=5,
            risk_management=False,
        )

        # Créer l'environnement avec actions continues
        self.env_continuous = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            action_type="continuous",
            risk_management=False,
        )

    def test_reset(self):
        """Teste la réinitialisation de l'environnement."""
        # Réinitialiser l'environnement
        observation = self.env.reset()[0]

        # Vérifier que l'observation a la bonne forme
        self.assertIsInstance(observation, np.ndarray)

        # La taille de l'état peut varier en fonction des indicateurs techniques inclus
        # Ne pas vérifier une taille spécifique, mais s'assurer qu'elle est cohérente
        state_size = len(observation)
        self.assertGreater(state_size, 0, "La taille de l'état devrait être positive")

        # Vérifier que le solde est réinitialisé
        self.assertEqual(self.env.balance, self.env.initial_balance)

        # Vérifier que la crypto détenue est réinitialisée
        self.assertEqual(self.env.crypto_held, 0)

        # Vérifier que l'étape courante est réinitialisée
        self.assertEqual(self.env.current_step, self.env.window_size)

    def test_step_hold(self):
        """Teste l'action de conserver."""
        self.env.reset()
        initial_balance = self.env.balance
        initial_crypto = self.env.crypto_held

        # Action 0 = HOLD
        next_state, reward, terminated, truncated, info = self.env.step(0)

        # Vérifier que le solde et les actions n'ont pas changé
        self.assertEqual(self.env.balance, initial_balance)
        self.assertEqual(self.env.crypto_held, initial_crypto)

        # Vérifier que l'indice a avancé
        self.assertEqual(self.env.current_step, self.env.window_size + 1)

    def test_step_buy(self):
        """Teste l'action d'achat."""
        self.env.reset()
        initial_balance = self.env.balance

        # Action 1 = BUY
        next_state, reward, terminated, truncated, info = self.env.step(1)

        # Vérifier que des actions ont été achetées
        self.assertGreater(self.env.crypto_held, 0)

        # Vérifier que le solde a diminué
        self.assertLess(self.env.balance, initial_balance)

    def test_step_sell(self):
        """Teste l'action de vente."""
        self.env.reset()

        # D'abord acheter des actions
        self.env.step(1)
        crypto_held = self.env.crypto_held
        balance_after_buy = self.env.balance

        # Puis vendre
        # Dans l'environnement standard avec n_discrete_actions=5 par défaut,
        # les actions de vente commencent à partir de 6
        next_state, reward, terminated, truncated, info = self.env.step(
            6
        )  # Utiliser l'action 6 pour vendre

        # Vérifier que des crypto ont été vendues
        self.assertLess(self.env.crypto_held, crypto_held)

        # Vérifier que le solde a augmenté
        self.assertGreater(self.env.balance, balance_after_buy)

    def test_discrete_partial_buy(self):
        """Teste l'achat partiel avec actions discrètes."""
        self.env_discrete.reset()
        initial_balance = self.env_discrete.balance

        # Action 2 = Acheter 40% (2/5 du solde)
        next_state, reward, terminated, truncated, info = self.env_discrete.step(2)

        # Vérifier que des crypto ont été achetées
        self.assertGreater(self.env_discrete.crypto_held, 0)

        # Sans gestion des risques, le plafond par ordre reste 30 %.
        expected_balance = initial_balance * 0.7
        self.assertAlmostEqual(
            self.env_discrete.balance / initial_balance, expected_balance / initial_balance, delta=0.03
        )

    def test_discrete_partial_sell(self):
        """Teste la vente partielle avec actions discrètes."""
        self.env_discrete.reset()

        # D'abord acheter des crypto
        self.env_discrete.step(5)  # Acheter 100%
        crypto_held = self.env_discrete.crypto_held

        # Puis vendre partiellement (Action 8 = Vendre 60%)
        next_state, reward, terminated, truncated, info = self.env_discrete.step(8)

        # Vérifier qu'environ 40% des crypto sont encore détenues
        expected_crypto = crypto_held * 0.4  # 40% restant
        self.assertAlmostEqual(
            self.env_discrete.crypto_held / crypto_held, 0.4, delta=0.1
        )

    def test_continuous_buy(self):
        """Teste l'achat avec actions continues."""
        self.env_continuous.reset()
        initial_balance = self.env_continuous.balance

        # Action 0.5 = Acheter 50%
        next_state, reward, terminated, truncated, info = self.env_continuous.step(
            np.array([0.5])
        )

        # Vérifier que des crypto ont été achetées
        self.assertGreater(self.env_continuous.crypto_held, 0)

        # Vérifier que le solde a diminué d'environ 50% (limité à 30%)
        expected_balance = initial_balance * 0.7  # Au moins 70% restant
        self.assertGreaterEqual(
            self.env_continuous.balance, expected_balance * 0.95
        )  # Avec une marge de 5%

    def test_continuous_sell(self):
        """Teste la vente avec actions continues."""
        self.env_continuous.reset()

        # D'abord acheter des crypto
        self.env_continuous.step(np.array([1.0]))  # Acheter 100%
        crypto_held = self.env_continuous.crypto_held

        # Puis vendre partiellement (Action -0.7 = Vendre 70%)
        next_state, reward, terminated, truncated, info = self.env_continuous.step(
            np.array([-0.7])
        )

        # Vérifier qu'environ 30% des crypto sont encore détenues
        expected_crypto = crypto_held * 0.3  # 30% restant
        self.assertAlmostEqual(
            self.env_continuous.crypto_held / crypto_held, 0.3, delta=0.1
        )

    def test_continuous_neutral(self):
        """Teste l'action neutre avec actions continues."""
        self.env_continuous.reset()
        initial_balance = self.env_continuous.balance
        initial_crypto = self.env_continuous.crypto_held

        # Action 0.03 = Zone neutre, ne rien faire
        next_state, reward, terminated, truncated, info = self.env_continuous.step(
            np.array([0.03])
        )

        # Vérifier que le solde et les crypto n'ont pas changé
        self.assertEqual(self.env_continuous.balance, initial_balance)
        self.assertEqual(self.env_continuous.crypto_held, initial_crypto)

    def test_max_buy_limit(self):
        """Teste la limite d'achat maximum de 30% du portefeuille."""
        self.env.reset()
        initial_balance = self.env.balance
        initial_portfolio_value = self.env.get_portfolio_value()

        # Action 1 = BUY (avec la nouvelle limite de 30%)
        next_state, reward, terminated, truncated, info = self.env.step(1)

        # Calculer la valeur dépensée
        spent_value = initial_balance - self.env.balance

        # Vérifier que la dépense ne dépasse pas 30% du portefeuille initial
        self.assertLessEqual(
            spent_value / initial_portfolio_value, 0.3 + 1e-6
        )  # Ajouter une petite marge pour les erreurs d'arrondi

    def test_discrete_max_buy_limit(self):
        """Teste la limite d'achat maximum de 30% avec actions discrètes."""
        self.env_discrete.reset()
        initial_balance = self.env_discrete.balance
        initial_portfolio_value = self.env_discrete.get_portfolio_value()

        # Action 5 = Acheter 100% (mais devrait être limité à 30%)
        next_state, reward, terminated, truncated, info = self.env_discrete.step(5)

        # Calculer la valeur dépensée
        spent_value = initial_balance - self.env_discrete.balance

        # Vérifier que la dépense ne dépasse pas 30% du portefeuille initial
        self.assertLessEqual(spent_value / initial_portfolio_value, 0.3 + 1e-6)

    def test_continuous_max_buy_limit(self):
        """Teste la limite d'achat maximum de 30% avec actions continues."""
        self.env_continuous.reset()
        initial_balance = self.env_continuous.balance
        initial_portfolio_value = self.env_continuous.get_portfolio_value()

        # Action 1.0 = Acheter 100% (mais devrait être limité à 30%)
        next_state, reward, terminated, truncated, info = self.env_continuous.step(
            np.array([1.0])
        )

        # Calculer la valeur dépensée
        spent_value = initial_balance - self.env_continuous.balance

        # Vérifier que la dépense ne dépasse pas 30% du portefeuille initial
        self.assertLessEqual(spent_value / initial_portfolio_value, 0.3 + 1e-6)

    def test_sequential_buys(self):
        """Les limites d'exposition globales sont appliquées quand le risque est activé."""
        data = self.test_data.copy()
        data[["open", "high", "low", "close"]] = 1000.0
        env = TradingEnvironment(
            data, window_size=10, risk_config={"max_position_size": 0.2}
        )
        env.reset()
        initial_portfolio_value = env.get_portfolio_value()

        # Premier achat
        env.step(5)

        # Deuxième achat
        env.step(5)

        # Troisième achat
        env.step(5)

        # Calculer la valeur totale dépensée
        current_portfolio_value = env.get_portfolio_value()
        crypto_value = (
            env.crypto_held * data.iloc[env.current_step]["close"]
        )

        # Le plafond porte sur l'exposition totale, pas seulement sur chaque ordre.
        self.assertLessEqual(crypto_value / initial_portfolio_value, 0.2 + 0.03)

    def test_risk_manager(self):
        """Teste l'intégration du gestionnaire de risque."""
        # Créer des données avec prix élevé pour dépasser les limites
        test_data = self.test_data.copy()
        test_data["close"] = 1000  # Prix fixe à 1000$

        # Créer un environnement avec gestionnaire de risque
        env = TradingEnvironment(
            df=test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            use_risk_manager=True,
            risk_config={
                "max_position_size": 0.05,  # Limite encore plus basse à 5% pour garantir le déclenchement
                "volatility_threshold": 0.005,  # Seuil de volatilité très bas
                "risk_adjustment_factor": 0.5,  # Facteur d'ajustement pour que l'action soit modifiée
            },
        )

        env.reset()
        initial_value = env.get_portfolio_value()
        action = 1 if env.action_type == "discrete" else np.array([1.0])
        next_state, reward, terminated, truncated, info = env.step(action)
        position_value = env.crypto_held * info["current_price"]
        self.assertLessEqual(position_value, initial_value * 0.05 + 1e-6)
        self.assertTrue(info["action_adjusted"])
        self.assertTrue(info["trade_executed"])

    def test_observation_has_no_future_data_leakage(self):
        """Modifier le futur ne doit pas modifier l'état disponible au reset."""
        changed_future = self.test_data.copy()
        changed_future.loc[changed_future.index[30:], "close"] *= 100
        reference = TradingEnvironment(self.test_data, window_size=10)
        altered = TradingEnvironment(changed_future, window_size=10)

        reference_state, _ = reference.reset(seed=7)
        altered_state, _ = altered.reset(seed=7)

        np.testing.assert_allclose(reference_state, altered_state, rtol=0, atol=1e-6)

    def test_observation_contains_the_documented_technical_and_sentiment_features(self):
        env = TradingEnvironment(self.test_data, window_size=10, risk_management=False)
        required = {
            "macd", "stoch_k", "momentum", "obv", "rsi", "atr", "adx",
            "vp_poc", "pivot_P", "ichimoku_tenkan", "compound_score",
        }
        self.assertTrue(required.issubset(set(env.feature_columns)))
        observation, _ = env.reset()
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(np.isfinite(observation).all())

    def test_excess_return_penalizes_invalid_sell_without_position(self):
        env = TradingEnvironment(
            self.test_data,
            window_size=10,
            reward_function="excess_return",
            invalid_action_penalty=0.01,
            risk_management=False,
        )
        env.reset()
        _, reward, _, _, info = env.step(env.n_discrete_actions + 1)

        self.assertTrue(info["invalid_action"])
        self.assertFalse(info["trade_executed"])
        self.assertAlmostEqual(
            reward,
            info["portfolio_return"] - info["benchmark_return"] - 0.01,
            places=8,
        )

    def test_excess_return_uses_the_risk_matched_benchmark(self):
        env = TradingEnvironment(
            self.test_data,
            window_size=10,
            reward_function="excess_return",
            risk_management=False,
            benchmark_exposure=0.20,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        previous_price = float(self.test_data.iloc[10]["close"])
        current_price = float(self.test_data.iloc[11]["close"])
        full_market_return = current_price / previous_price - 1.0
        self.assertLess(abs(info["benchmark_return"]), abs(full_market_return))
        self.assertAlmostEqual(reward, -info["benchmark_return"], places=8)

    def test_risk_adjusted_excess_penalizes_drawdown_beyond_raw_excess_return(self):
        env = TradingEnvironment(
            self.test_data,
            window_size=10,
            reward_function="risk_adjusted_excess",
            risk_management=False,
            reward_drawdown_weight=0.5,
            reward_downside_weight=0.5,
        )
        env.reset()
        env.portfolio_value_history = [100.0, 90.0]
        env._last_execution = {"executed": False, "blocked_by_risk": False, "invalid_action": False}
        env._agent_turnover = 0.0
        assert env._calculate_reward(-0.02, 0.0) < -0.02

    def test_bear_regime_reduces_exposure_and_is_reported_in_risk_info(self):
        env = TradingEnvironment(self.test_data, window_size=10, risk_management=True)
        env.df["market_regime"] = "bear"
        env.df["regime_volatility"] = 1.0
        env.reset()
        self.assertEqual(env._regime_exposure_multiplier(), 0.5)
        _, _, _, _, info = env.step(0)
        assert info["risk_info"]["market_regime"] == "bear"
        assert info["risk_info"]["regime_exposure_multiplier"] == 0.5

    def test_turnover_penalty_uses_executed_agent_notional(self):
        env = TradingEnvironment(
            self.test_data,
            window_size=10,
            reward_function="simple",
            risk_management=False,
            max_turnover=0.0,
            turnover_penalty=0.01,
        )
        env.reset()
        _, reward, _, _, info = env.step(1)

        self.assertGreater(info["agent_turnover"], 0.0)
        self.assertLess(reward, info["portfolio_return"])

    def test_action_mask_rejects_residual_micro_buys(self):
        env = TradingEnvironment(
            self.test_data,
            window_size=10,
            risk_management=False,
            min_trade_fraction=0.01,
        )
        env.reset()
        price = float(env.df.iloc[env.current_step]["close"])
        env.balance = 50.0
        env.crypto_held = (env.initial_balance - env.balance) / price

        mask = env.get_action_mask()
        self.assertTrue(mask[0])
        self.assertFalse(mask[1 : env.n_discrete_actions + 1].any())
        self.assertTrue(mask[env.n_discrete_actions + 1 :].any())

    def test_action_mask_enforces_agent_trade_cooldown(self):
        env = TradingEnvironment(
            self.test_data, window_size=10, risk_management=False, min_trade_interval=3
        )
        env.reset()
        env.step(1)

        mask = env.get_action_mask()
        self.assertTrue(mask[0])
        self.assertEqual(int(mask.sum()), 1)

    def test_stop_loss_is_actually_enforced(self):
        data = self.test_data.copy()
        data.loc[data.index[10], ["open", "high", "low", "close"]] = 100.0
        data.loc[data.index[11], ["open", "high", "low", "close"]] = 80.0
        env = TradingEnvironment(data, window_size=10, max_trade_fraction=0.1)
        env.reset()

        _, _, _, _, info = env.step(1)

        self.assertTrue(info["risk_info"]["stop"]["stop_triggered"])
        self.assertEqual(info["risk_info"]["stop"]["stop_type"], "stop_loss")
        self.assertEqual(env.crypto_held, 0.0)
        self.assertTrue(info["trade_executed"])

    def test_delayed_orders_reserve_cash_before_execution(self):
        env = TradingEnvironment(
            self.test_data,
            window_size=10,
            risk_management=False,
            execution_delay=2,
            max_trade_fraction=0.3,
        )
        env.reset()
        initial_balance = env.balance
        for _ in range(4):
            env.step(5)

        self.assertLessEqual(env.reserved_cash, initial_balance + 1e-6)
        self.assertGreaterEqual(env.balance, 0.0)
        self.assertGreaterEqual(env.crypto_held, 0.0)


if __name__ == "__main__":
    unittest.main()
