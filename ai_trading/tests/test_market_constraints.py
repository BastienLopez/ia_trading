import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ai_trading.rl.trading_environment import TradingEnvironment

class TestMarketConstraints(unittest.TestCase):
    def setUp(self):
        # Créer des données de test avec volatilité
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(150, 250, 100),
                "low": np.random.uniform(50, 150, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 5000, 100),
                "volatility": np.random.uniform(0.01, 0.05, 100),
            },
            index=dates,
        )
        
    def test_slippage_constant(self):
        """Teste le modèle de slippage constant."""
        env = TradingEnvironment(
            df=self.test_data,
            slippage_model="constant",
            slippage_value=0.001
        )
        
        price = 100.0
        action_value = 0.5
        
        # Test achat
        price_with_slippage = env._apply_slippage(price, action_value)
        self.assertAlmostEqual(price_with_slippage, 100.1)  # 100 * (1 + 0.001)
        
        # Test vente
        price_with_slippage = env._apply_slippage(price, -action_value)
        self.assertAlmostEqual(price_with_slippage, 99.9)  # 100 * (1 - 0.001)
        
    def test_slippage_proportional(self):
        """Teste le modèle de slippage proportionnel."""
        env = TradingEnvironment(
            df=self.test_data,
            slippage_model="proportional",
            slippage_value=0.001
        )
        
        price = 100.0
        action_value = 0.5
        
        # Test achat
        price_with_slippage = env._apply_slippage(price, action_value)
        self.assertAlmostEqual(price_with_slippage, 100.05)  # 100 * (1 + 0.001 * 0.5)
        
        # Test vente
        price_with_slippage = env._apply_slippage(price, -action_value)
        self.assertAlmostEqual(price_with_slippage, 99.95)  # 100 * (1 - 0.001 * 0.5)
        
    def test_slippage_dynamic(self):
        """Teste le modèle de slippage dynamique."""
        env = TradingEnvironment(
            df=self.test_data.copy(),  # Créer une copie pour éviter les avertissements
            slippage_model="dynamic",
            slippage_value=0.001
        )
        
        price = 100.0
        action_value = 0.5
        
        # Test avec des valeurs de volume et volatilité spécifiques
        env.df.at[env.df.index[env.current_step], "volatility"] = 0.02
        env.df.at[env.df.index[env.current_step], "volume"] = 2000
        
        # Définir le volume pour les 20 derniers pas
        start_idx = max(0, env.current_step - 20)
        env.df.loc[env.df.index[start_idx:env.current_step], "volume"] = 1000
        
        price_with_slippage = env._apply_slippage(price, action_value)
        expected_slippage = 0.001 * (1 + 0.02) * (2000 / 1000)
        self.assertAlmostEqual(price_with_slippage, price * (1 + expected_slippage), places=4)
        
    def test_execution_delay(self):
        """Teste le délai d'exécution des ordres."""
        env = TradingEnvironment(
            df=self.test_data.copy(),
            execution_delay=2,
            action_type="discrete",  # Spécifier le type d'action
            n_discrete_actions=5  # Spécifier le nombre d'actions discrètes
        )
        
        # Exécuter une action d'achat
        initial_balance = env.balance
        env.step(1)  # Action d'achat
        
        # Vérifier que l'ordre est en attente
        self.assertEqual(len(env.pending_orders), 1)
        self.assertEqual(env.pending_orders[0]["delay"], 2)
        self.assertEqual(env.balance, initial_balance)  # Le solde ne doit pas avoir changé
        
        # Avancer d'un pas
        env.step(0)  # Action neutre
        self.assertEqual(len(env.pending_orders), 1)
        self.assertEqual(env.pending_orders[0]["delay"], 1)
        self.assertEqual(env.balance, initial_balance)  # Le solde ne doit pas avoir changé
        
        # Avancer d'un autre pas
        env.step(0)  # Action neutre
        self.assertEqual(len(env.pending_orders), 0)  # L'ordre devrait être exécuté
        self.assertNotEqual(env.balance, initial_balance)  # Le solde doit avoir changé
        
if __name__ == "__main__":
    unittest.main() 