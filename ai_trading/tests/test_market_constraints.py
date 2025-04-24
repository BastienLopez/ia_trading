import unittest
import numpy as np
from ..rl.market_constraints import MarketConstraints

class TestMarketConstraints(unittest.TestCase):
    def setUp(self):
        """Prépare l'environnement de test."""
        self.constraints = MarketConstraints(
            slippage_model="dynamic",
            base_slippage=0.001,
            execution_delay=2,
            market_impact_factor=0.1
        )
        
        # Données de profondeur du carnet d'ordres fictives
        self.mock_depth = {
            'spread_pct': 0.1,  # 0.1%
            'total_volume': 1000.0,
            'volume_imbalance': 0.2,  # Déséquilibre positif (plus d'ordres d'achat)
            'depth_range_bids_5': 100.0,
            'depth_range_asks_5': 120.0
        }
        
        self.constraints.update_orderbook_depth('BTC/USDT', self.mock_depth)
        
    def test_slippage_fixed(self):
        """Teste le modèle de slippage fixe."""
        constraints = MarketConstraints(slippage_model="fixed", base_slippage=0.002)
        
        slippage = constraints.calculate_slippage(
            symbol='BTC/USDT',
            action_value=0.5,
            volume=1.0,
            volatility=0.02,
            avg_volume=10.0
        )
        
        self.assertEqual(slippage, 0.002)
        
    def test_slippage_dynamic(self):
        """Teste le modèle de slippage dynamique."""
        slippage = self.constraints.calculate_slippage(
            symbol='BTC/USDT',
            action_value=0.5,
            volume=20.0,  # Volume supérieur à la moyenne
            volatility=0.02,
            avg_volume=10.0
        )
        
        # Le slippage devrait être plus élevé que le slippage de base
        self.assertGreater(slippage, self.constraints.base_slippage)
        # Mais ne devrait pas dépasser 1%
        self.assertLessEqual(slippage, 0.01)
        
    def test_slippage_orderbook(self):
        """Teste le modèle de slippage basé sur le carnet d'ordres."""
        constraints = MarketConstraints(slippage_model="orderbook", base_slippage=0.001)
        constraints.update_orderbook_depth('BTC/USDT', self.mock_depth)
        
        slippage = constraints.calculate_slippage(
            symbol='BTC/USDT',
            action_value=0.5,
            volume=500.0,  # 50% du volume total
            volatility=0.02,
            avg_volume=1000.0
        )
        
        # Le slippage devrait être proportionnel au spread
        expected_min = (self.mock_depth['spread_pct'] / 100) * 0.5  # 50% du spread
        self.assertGreater(slippage, expected_min)
        self.assertLessEqual(slippage, 0.01)  # Max 1%
        
    def test_market_impact(self):
        """Teste le calcul de l'impact sur le marché."""
        impact, recovery = self.constraints.calculate_market_impact(
            symbol='BTC/USDT',
            action_value=1.0,  # Action maximale
            volume=100.0,
            price=50000.0,
            avg_volume=1000.0
        )
        
        # L'impact devrait être positif mais limité
        self.assertGreater(impact, 0)
        self.assertLessEqual(impact, 0.05)  # Max 5%
        
        # Le temps de récupération devrait être proportionnel au carré du ratio de volume
        self.assertGreater(recovery, 0)
        self.assertLessEqual(recovery, 100)  # Max 100 pas
        
    def test_execution_delay(self):
        """Teste le calcul du délai d'exécution."""
        delay = self.constraints.calculate_execution_delay(
            symbol='BTC/USDT',
            action_value=0.5,
            volume=200.0,
            avg_volume=100.0
        )
        
        # Le délai devrait être supérieur au délai de base mais limité
        self.assertGreater(delay, self.constraints.execution_delay)
        self.assertLessEqual(delay, 10)  # Max 10 pas
        
    def test_zero_volume(self):
        """Teste le comportement avec un volume nul."""
        # Test slippage
        slippage = self.constraints.calculate_slippage(
            symbol='BTC/USDT',
            action_value=0.0,
            volume=0.0,
            volatility=0.02,
            avg_volume=10.0
        )
        self.assertEqual(slippage, 0.0)
        
        # Test impact
        impact, recovery = self.constraints.calculate_market_impact(
            symbol='BTC/USDT',
            action_value=0.0,
            volume=0.0,
            price=50000.0,
            avg_volume=1000.0
        )
        self.assertEqual(impact, 0.0)
        self.assertEqual(recovery, 0)
        
        # Test délai
        delay = self.constraints.calculate_execution_delay(
            symbol='BTC/USDT',
            action_value=0.0,
            volume=0.0,
            avg_volume=100.0
        )
        self.assertEqual(delay, 0)
        
    def test_orderbook_update(self):
        """Teste la mise à jour des données du carnet d'ordres."""
        new_depth = {
            'spread_pct': 0.2,
            'total_volume': 2000.0,
            'volume_imbalance': -0.1
        }
        
        self.constraints.update_orderbook_depth('ETH/USDT', new_depth)
        
        # Vérifier que les données ont été mises à jour
        self.assertEqual(self.constraints.orderbook_depth['ETH/USDT'], new_depth)
        
    def test_extreme_values(self):
        """Teste le comportement avec des valeurs extrêmes."""
        # Test avec un très grand volume
        slippage = self.constraints.calculate_slippage(
            symbol='BTC/USDT',
            action_value=1.0,
            volume=1e6,  # Volume très élevé
            volatility=0.5,  # Forte volatilité
            avg_volume=1000.0
        )
        self.assertLessEqual(slippage, 0.01)  # Ne devrait pas dépasser 1%
        
        # Test avec une forte volatilité
        impact, recovery = self.constraints.calculate_market_impact(
            symbol='BTC/USDT',
            action_value=1.0,
            volume=1e6,
            price=50000.0,
            avg_volume=1000.0
        )
        self.assertLessEqual(impact, 0.05)  # Ne devrait pas dépasser 5%
        self.assertLessEqual(recovery, 100)  # Ne devrait pas dépasser 100 pas
        
if __name__ == '__main__':
    unittest.main() 