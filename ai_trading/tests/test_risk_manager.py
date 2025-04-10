import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Tests pour le gestionnaire de risques."""

    def setUp(self):
        """Prépare les données pour les tests."""
        # Créer des données de marché synthétiques
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        self.market_data = pd.DataFrame({
            "open": np.linspace(100, 150, 30) + np.random.normal(0, 5, 30),
            "high": np.linspace(105, 155, 30) + np.random.normal(0, 5, 30),
            "low": np.linspace(95, 145, 30) + np.random.normal(0, 5, 30),
            "close": np.linspace(102, 152, 30) + np.random.normal(0, 5, 30),
            "volume": np.random.uniform(1000, 5000, 30)
        }, index=dates)
        
        # Initialiser le gestionnaire de risques
        self.risk_manager = RiskManager(config={
            'stop_loss_atr_factor': 2.0,
            'take_profit_atr_factor': 3.0,
            'trailing_stop_activation': 0.02,
            'trailing_stop_distance': 0.01
        })
    
    def test_calculate_position_size(self):
        """Teste le calcul de la taille de position."""
        # Calculer la taille de position
        position_size = self.risk_manager.calculate_position_size(
            capital=10000, entry_price=100, stop_loss_price=95)
        
        # Vérifier que la taille de position est correcte
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 10000 * 0.2 / 100)  # max_position_size = 20%
    
    def test_calculate_atr_stop_loss(self):
        """Teste le calcul du stop-loss basé sur l'ATR."""
        # Calculer le stop-loss
        stop_loss = self.risk_manager.calculate_atr_stop_loss(
            data=self.market_data, period=14, direction='long', 
            current_price=100, position_id='test_position')
        
        # Vérifier que le stop-loss est inférieur au prix actuel pour une position longue
        self.assertLess(stop_loss, 100)
        
        # Vérifier que le stop-loss est enregistré
        self.assertIn('test_position', self.risk_manager.position_stops)
        self.assertEqual(self.risk_manager.position_stops['test_position']['stop_loss'], stop_loss)
    
    def test_calculate_atr_take_profit(self):
        """Teste le calcul du take-profit basé sur l'ATR."""
        # Calculer le take-profit
        take_profit = self.risk_manager.calculate_atr_take_profit(
            data=self.market_data, period=14, direction='long', 
            current_price=100, position_id='test_position')
        
        # Vérifier que le take-profit est supérieur au prix actuel pour une position longue
        self.assertGreater(take_profit, 100)
        
        # Vérifier que le take-profit est enregistré
        self.assertIn('test_position', self.risk_manager.position_stops)
        self.assertEqual(self.risk_manager.position_stops['test_position']['take_profit'], take_profit)
    
    def test_update_trailing_stop(self):
        """Teste la mise à jour du trailing stop."""
        # Configurer une position
        position_id = 'test_position'
        self.risk_manager.position_stops[position_id] = {
            'stop_loss': 95, 'take_profit': 110, 'trailing_stop': None
        }
        
        # Mettre à jour le trailing stop avec un profit insuffisant
        trailing_stop = self.risk_manager.update_trailing_stop(
            position_id=position_id, current_price=101, entry_price=100, direction='long')
        
        # Vérifier que le trailing stop n'est pas activé
        self.assertIsNone(trailing_stop)
        
        # Mettre à jour le trailing stop avec un profit suffisant
        trailing_stop = self.risk_manager.update_trailing_stop(
            position_id=position_id, current_price=103, entry_price=100, direction='long')
        
        # Vérifier que le trailing stop est activé
        self.assertIsNotNone(trailing_stop)
        self.assertAlmostEqual(trailing_stop, 103 * 0.99, delta=0.01)
    
    def test_check_stop_conditions(self):
        """Teste la vérification des conditions de stop."""
        # Configurer une position
        position_id = 'test_position'
        self.risk_manager.position_stops[position_id] = {
            'stop_loss': 95, 'take_profit': 110, 'trailing_stop': 98
        }
        
        # Vérifier que le stop-loss est déclenché
        result = self.risk_manager.check_stop_conditions(
            position_id=position_id, current_price=94, direction='long')
        self.assertTrue(result['stop_triggered'])
        self.assertEqual(result['stop_type'], 'stop_loss')
        
        # Vérifier que le take-profit est déclenché
        result = self.risk_manager.check_stop_conditions(
            position_id=position_id, current_price=111, direction='long')
        self.assertTrue(result['stop_triggered'])
        self.assertEqual(result['stop_type'], 'take_profit')
        
        # Vérifier que le trailing stop est déclenché
        result = self.risk_manager.check_stop_conditions(
            position_id=position_id, current_price=97, direction='long')
        self.assertTrue(result['stop_triggered'])
        self.assertEqual(result['stop_type'], 'trailing_stop')
        
        # Vérifier qu'aucun stop n'est déclenché
        result = self.risk_manager.check_stop_conditions(
            position_id=position_id, current_price=100, direction='long')
        self.assertFalse(result['stop_triggered'])
    
    def test_clear_position(self):
        """Teste la suppression des informations de position."""
        # Configurer une position
        position_id = 'test_position'
        self.risk_manager.position_stops[position_id] = {
            'stop_loss': 95, 'take_profit': 110, 'trailing_stop': 98
        }
        
        # Supprimer la position
        self.risk_manager.clear_position(position_id)
        
        # Vérifier que la position est supprimée
        self.assertNotIn(position_id, self.risk_manager.position_stops)

if __name__ == "__main__":
    unittest.main() 