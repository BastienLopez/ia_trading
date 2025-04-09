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
        prices = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)

        # Créer un DataFrame avec les données
        self.test_data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 10, 100),
                "low": prices - np.random.uniform(0, 10, 100),
                "close": prices + np.random.normal(0, 3, 100),
                "volume": np.random.uniform(1000, 5000, 100),
                "compound_score": np.random.uniform(-1, 1, 100),  # Sentiment
            },
            index=dates,
        )

        # Créer l'environnement standard
        self.env = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )
        
        # Créer l'environnement avec actions discrètes nuancées
        self.env_discrete = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            action_type="discrete",
            n_discrete_actions=5,
        )
        
        # Créer l'environnement avec actions continues
        self.env_continuous = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            action_type="continuous",
        )

    def test_reset(self):
        """Teste la méthode reset de l'environnement."""
        state, _ = self.env.reset()

        # Vérifier que l'état a la bonne forme
        self.assertEqual(len(state), 13)

        # Vérifier que le solde est réinitialisé
        self.assertEqual(self.env.balance, 10000)

        # Vérifier que la position est réinitialisée
        self.assertEqual(self.env.crypto_held, 0)

        # Vérifier que l'indice courant est correct
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
        next_state, reward, terminated, truncated, info = self.env.step(6)  # Utiliser l'action 6 pour vendre

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

        # Vérifier que le solde a diminué d'environ 40%
        expected_balance = initial_balance * 0.6  # 60% restant
        self.assertAlmostEqual(self.env_discrete.balance / initial_balance, 0.6, delta=0.1)

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
        self.assertAlmostEqual(self.env_discrete.crypto_held / crypto_held, 0.4, delta=0.1)

    def test_continuous_buy(self):
        """Teste l'achat avec actions continues."""
        self.env_continuous.reset()
        initial_balance = self.env_continuous.balance

        # Action 0.5 = Acheter 50%
        next_state, reward, terminated, truncated, info = self.env_continuous.step(np.array([0.5]))

        # Vérifier que des crypto ont été achetées
        self.assertGreater(self.env_continuous.crypto_held, 0)

        # Vérifier que le solde a diminué d'environ 50% (limité à 30%)
        expected_balance = initial_balance * 0.7  # Au moins 70% restant
        self.assertGreaterEqual(self.env_continuous.balance, expected_balance * 0.95)  # Avec une marge de 5%

    def test_continuous_sell(self):
        """Teste la vente avec actions continues."""
        self.env_continuous.reset()

        # D'abord acheter des crypto
        self.env_continuous.step(np.array([1.0]))  # Acheter 100%
        crypto_held = self.env_continuous.crypto_held
        
        # Puis vendre partiellement (Action -0.7 = Vendre 70%)
        next_state, reward, terminated, truncated, info = self.env_continuous.step(np.array([-0.7]))

        # Vérifier qu'environ 30% des crypto sont encore détenues
        expected_crypto = crypto_held * 0.3  # 30% restant
        self.assertAlmostEqual(self.env_continuous.crypto_held / crypto_held, 0.3, delta=0.1)

    def test_continuous_neutral(self):
        """Teste l'action neutre avec actions continues."""
        self.env_continuous.reset()
        initial_balance = self.env_continuous.balance
        initial_crypto = self.env_continuous.crypto_held

        # Action 0.03 = Zone neutre, ne rien faire
        next_state, reward, terminated, truncated, info = self.env_continuous.step(np.array([0.03]))

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
        self.assertLessEqual(spent_value / initial_portfolio_value, 0.3 + 1e-6)  # Ajouter une petite marge pour les erreurs d'arrondi
        
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
        next_state, reward, terminated, truncated, info = self.env_continuous.step(np.array([1.0]))
        
        # Calculer la valeur dépensée
        spent_value = initial_balance - self.env_continuous.balance
        
        # Vérifier que la dépense ne dépasse pas 30% du portefeuille initial
        self.assertLessEqual(spent_value / initial_portfolio_value, 0.3 + 1e-6)
        
    def test_sequential_buys(self):
        """Teste que plusieurs achats séquentiels respectent toujours la limite de 30%."""
        self.env.reset()
        initial_portfolio_value = self.env.get_portfolio_value()
        
        # Premier achat
        self.env.step(1)
        
        # Deuxième achat
        self.env.step(1)
        
        # Troisième achat
        self.env.step(1)
        
        # Calculer la valeur totale dépensée
        current_portfolio_value = self.env.get_portfolio_value()
        crypto_value = self.env.crypto_held * self.test_data.iloc[self.env.current_step]["close"]
        
        # Vérifier que la valeur en crypto ne dépasse pas 90% (3 x 30%) du portefeuille initial
        # Note: Ceci est une vérification approximative car la valeur du portefeuille peut changer avec le prix
        self.assertLessEqual(crypto_value / initial_portfolio_value, 0.9 + 1e-6)


if __name__ == "__main__":
    unittest.main()
