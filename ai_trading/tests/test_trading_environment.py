import unittest
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.trading_environment import TradingEnvironment

class TestTradingEnvironment(unittest.TestCase):
    """Tests pour l'environnement de trading."""
    
    def setUp(self):
        """Prépare les données et l'environnement pour les tests."""
        # Créer des données synthétiques pour les tests
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Créer une tendance haussière simple
        prices = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
        
        # Créer un DataFrame avec les données
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 10, 100),
            'low': prices - np.random.uniform(0, 10, 100),
            'close': prices + np.random.normal(0, 3, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'compound_score': np.random.uniform(-1, 1, 100)  # Sentiment
        }, index=dates)
        
        # Créer l'environnement
        self.env = TradingEnvironment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
    
    def test_reset(self):
        """Teste la méthode reset de l'environnement."""
        state = self.env.reset()
        
        # Vérifier que l'état a la bonne forme
        self.assertEqual(len(state), self.env.observation_space.shape[0])
        
        # Vérifier que le solde est réinitialisé
        self.assertEqual(self.env.balance, 10000)
        
        # Vérifier que la position est réinitialisée
        self.assertEqual(self.env.shares_held, 0)
        
        # Vérifier que l'indice courant est correct
        self.assertEqual(self.env.current_step, self.env.window_size)
    
    def test_step_hold(self):
        """Teste l'action de conserver."""
        self.env.reset()
        initial_balance = self.env.balance
        initial_shares = self.env.shares_held
        
        # Action 0 = HOLD
        next_state, reward, done, info = self.env.step(0)
        
        # Vérifier que le solde et les actions n'ont pas changé
        self.assertEqual(self.env.balance, initial_balance)
        self.assertEqual(self.env.shares_held, initial_shares)
        
        # Vérifier que l'indice a avancé
        self.assertEqual(self.env.current_step, self.env.window_size + 1)
    
    def test_step_buy(self):
        """Teste l'action d'achat."""
        self.env.reset()
        initial_balance = self.env.balance
        
        # Action 1 = BUY
        next_state, reward, done, info = self.env.step(1)
        
        # Vérifier que des actions ont été achetées
        self.assertGreater(self.env.shares_held, 0)
        
        # Vérifier que le solde a diminué
        self.assertLess(self.env.balance, initial_balance)
    
    def test_step_sell(self):
        """Teste l'action de vente."""
        self.env.reset()
        
        # D'abord acheter des actions
        self.env.step(1)
        shares_held = self.env.shares_held
        balance_after_buy = self.env.balance
        
        # Puis les vendre (Action 2 = SELL)
        next_state, reward, done, info = self.env.step(2)
        
        # Vérifier que les actions ont été vendues
        self.assertEqual(self.env.shares_held, 0)
        
        # Vérifier que le solde a augmenté
        self.assertGreater(self.env.balance, balance_after_buy)
    
    def test_done_condition(self):
        """Teste la condition de fin d'épisode."""
        self.env.reset()
        
        # Avancer jusqu'à la fin des données
        done = False
        steps = 0
        max_steps = len(self.test_data) - self.env.window_size - 1
        
        while not done and steps < max_steps + 10:  # +10 pour éviter une boucle infinie
            _, _, done, _ = self.env.step(0)  # Action HOLD
            steps += 1
        
        # Vérifier que l'épisode se termine au bon moment
        self.assertEqual(steps, max_steps)
        self.assertTrue(done)
    
    def test_reward_calculation(self):
        """Teste le calcul de la récompense."""
        self.env.reset()
        
        # Acheter des actions
        _, reward_buy, _, _ = self.env.step(1)
        
        # La récompense devrait être proche de zéro pour un achat
        self.assertAlmostEqual(reward_buy, 0, delta=0.1)
        
        # Vendre des actions avec profit
        # Pour ce test, on suppose que le prix augmente dans nos données synthétiques
        _, reward_sell, _, _ = self.env.step(2)
        
        # La récompense devrait être positive pour une vente profitable
        self.assertGreaterEqual(reward_sell, 0)
    
    def test_get_portfolio_value(self):
        """Teste le calcul de la valeur du portefeuille."""
        self.env.reset()
        
        # Acheter des actions
        self.env.step(1)
        
        # Calculer la valeur attendue du portefeuille
        current_price = self.test_data.iloc[self.env.current_step]['close']
        expected_value = self.env.balance + self.env.shares_held * current_price
        
        # Vérifier que la valeur du portefeuille est correcte
        self.assertAlmostEqual(self.env.get_portfolio_value(), expected_value, places=5)
    
    def test_get_portfolio_history(self):
        """Teste l'historique de la valeur du portefeuille."""
        self.env.reset()
        
        # Effectuer quelques actions
        self.env.step(1)  # BUY
        self.env.step(0)  # HOLD
        self.env.step(2)  # SELL
        
        # Récupérer l'historique
        history = self.env.get_portfolio_history()
        
        # Vérifier que l'historique a la bonne longueur
        self.assertEqual(len(history), self.env.current_step - self.env.window_size + 1)
        
        # Vérifier que les valeurs sont positives
        for value in history:
            self.assertGreater(value, 0)

if __name__ == '__main__':
    unittest.main() 