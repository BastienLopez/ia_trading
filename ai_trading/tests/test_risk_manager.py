import unittest
import numpy as np
from ai_trading.rl.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Tests pour le gestionnaire de risque."""
    
    def setUp(self):
        """Initialise le gestionnaire de risque pour les tests."""
        self.risk_manager = RiskManager(
            max_drawdown=0.15,
            take_profit=0.3,
            stop_loss=0.1,
            trailing_stop=0.05
        )
    
    def test_should_limit_position(self):
        """Teste la détection des situations à risque."""
        # Test du drawdown maximum
        portfolio_history = [10000, 10500, 11000, 10000, 9500, 9000]  # -18% depuis le max
        self.assertTrue(
            self.risk_manager.should_limit_position(portfolio_history, 1.0),
            "Le drawdown de 18% devrait déclencher une limitation de position"
        )
        
        # Test du stop loss
        portfolio_history = [10000, 9500, 9000]  # -10% depuis le début
        self.assertTrue(
            self.risk_manager.should_limit_position(portfolio_history, 1.0),
            "La perte de 10% devrait déclencher le stop loss"
        )
        
        # Test sans déclenchement
        portfolio_history = [10000, 10200, 10100]
        self.assertFalse(
            self.risk_manager.should_limit_position(portfolio_history, 1.0),
            "Une légère fluctuation ne devrait pas déclencher de limitation"
        )
    
    def test_adjust_action(self):
        """Teste l'ajustement des actions."""
        # Test avec action discrète et position existante
        adjusted_action = self.risk_manager.adjust_action(1, 1.0)
        self.assertEqual(adjusted_action, 2, "L'action devrait être ajustée à 'vendre'")
        
        # Test avec action continue et position existante
        adjusted_action = self.risk_manager.adjust_action(0.5, 1.0)
        self.assertEqual(adjusted_action, -1.0, "L'action devrait être ajustée à 'vendre tout'")
        
        # Test sans position
        adjusted_action = self.risk_manager.adjust_action(0.5, 0.0)
        self.assertEqual(adjusted_action, 0.0, "Sans position, l'action devrait être 'ne rien faire'") 