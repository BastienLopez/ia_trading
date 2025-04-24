import unittest
import numpy as np
import pandas as pd
from ..rl.multi_asset_trading_environment import MultiAssetTradingEnvironment

class TestMultiAssetTradingEnvironment(unittest.TestCase):
    def setUp(self):
        """Prépare l'environnement de test."""
        # Création de données de test
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.data_dict = {
            'BTC/USDT': pd.DataFrame({
                'open': np.random.uniform(45000, 55000, 100),
                'high': np.random.uniform(46000, 56000, 100),
                'low': np.random.uniform(44000, 54000, 100),
                'close': np.random.uniform(45000, 55000, 100),
                'volume': np.random.uniform(1000, 5000, 100),
                'orderbook_depth': [
                    {
                        'spread_pct': 0.1,
                        'total_volume': 1000.0,
                        'volume_imbalance': 0.2
                    }
                ] * 100
            }, index=dates),
            'ETH/USDT': pd.DataFrame({
                'open': np.random.uniform(2800, 3200, 100),
                'high': np.random.uniform(2900, 3300, 100),
                'low': np.random.uniform(2700, 3100, 100),
                'close': np.random.uniform(2800, 3200, 100),
                'volume': np.random.uniform(5000, 10000, 100),
                'orderbook_depth': [
                    {
                        'spread_pct': 0.15,
                        'total_volume': 2000.0,
                        'volume_imbalance': -0.1
                    }
                ] * 100
            }, index=dates)
        }
        
        self.env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            initial_balance=10000.0,
            window_size=10,
            slippage_model="dynamic",
            base_slippage=0.001,
            execution_delay=2,
            market_impact_factor=0.1
        )
        
    def test_market_constraints_initialization(self):
        """Teste l'initialisation des contraintes de marché."""
        self.assertIsNotNone(self.env.market_constraints)
        self.assertEqual(self.env.market_constraints.slippage_model, "dynamic")
        self.assertEqual(self.env.market_constraints.base_slippage, 0.001)
        self.assertEqual(self.env.market_constraints.execution_delay, 2)
        self.assertEqual(self.env.market_constraints.market_impact_factor, 0.1)
        
    def test_delayed_execution(self):
        """Teste le délai d'exécution des ordres."""
        # Créer un ordre
        action = np.array([0.3, 0.0])  # 30% en BTC
        initial_balance = self.env.balance
        
        # Premier pas
        _, _, _, _ = self.env.step(action)
        
        # Vérifier qu'un seul ordre est en attente
        self.assertEqual(len(self.env.pending_orders), 1)
        
        # Vérifier le délai de l'ordre
        order = self.env.pending_orders[0]
        self.assertEqual(order['delay'], self.env.market_constraints.execution_delay)
        
        # Attendre la moitié du délai
        for _ in range(self.env.market_constraints.execution_delay // 2):
            _, _, _, _ = self.env.step(np.array([0.0, 0.0]))
            self.assertEqual(len(self.env.pending_orders), 1)  # L'ordre doit toujours être en attente
            
        # Attendre le reste du délai
        for _ in range(self.env.market_constraints.execution_delay // 2):
            _, _, _, _ = self.env.step(np.array([0.0, 0.0]))
            
        # Un pas supplémentaire pour l'exécution finale
        _, _, _, _ = self.env.step(np.array([0.0, 0.0]))
        
        # Vérifier que l'ordre a été exécuté
        self.assertEqual(len(self.env.pending_orders), 0)
        self.assertLess(self.env.balance, initial_balance)  # Le solde doit avoir diminué
        self.assertGreater(self.env.crypto_holdings['BTC/USDT'], 0)  # Doit avoir du BTC
        
    def test_slippage_impact(self):
        """Teste l'impact du slippage sur les prix d'exécution."""
        # Configuration initiale
        self.env.market_constraints.execution_delay = 0
        initial_balance = self.env.balance
        
        # Test avec différents niveaux d'achat
        test_sizes = [0.2, 0.5, 0.8]  # Petite, moyenne et grande transaction
        slippage_costs = []
        
        for size in test_sizes:
            # Réinitialiser l'environnement
            self.env.reset()
            self.env.balance = initial_balance
            
            # Exécuter l'action d'achat
            action = np.array([size, 0.0])
            _, _, _, info = self.env.step(action)
            
            # Calculer le coût réel et théorique
            expected_cost = initial_balance * size
            actual_cost = initial_balance - self.env.balance
            slippage_cost = (actual_cost - expected_cost) / expected_cost
            slippage_costs.append(slippage_cost)
            
            # Vérifications
            self.assertGreater(actual_cost, expected_cost, 
                             f"Le coût avec slippage devrait être plus élevé pour une taille de {size}")
            self.assertLess(slippage_cost, 0.05,  # Max 5% de slippage
                          f"Le slippage ne devrait pas dépasser 5%, obtenu: {slippage_cost:.2%}")
        
        # Vérifier que le slippage augmente avec la taille de l'ordre
        self.assertLess(slippage_costs[0], slippage_costs[1], 
                       "Le slippage devrait augmenter avec la taille de l'ordre")
        self.assertLess(slippage_costs[1], slippage_costs[2], 
                       "Le slippage devrait augmenter avec la taille de l'ordre")
        
    def test_market_impact(self):
        """Teste l'impact sur le marché des transactions."""
        # Action d'achat importante
        action = np.array([1.0, 0.0])  # 100% en BTC
        
        # Exécution
        _, _, _, _ = self.env.step(action)
        
        # Vérifier l'enregistrement de l'impact
        self.assertGreater(len(self.env.market_impacts['BTC/USDT']), 0)
        impact = self.env.market_impacts['BTC/USDT'][-1]
        
        # L'impact devrait être positif et le temps de récupération > 0
        self.assertGreater(impact['impact'], 0)
        self.assertGreater(impact['recovery_time'], 0)
        
    def test_orderbook_integration(self):
        """Teste l'intégration des données du carnet d'ordres."""
        # Vérifier que les données du carnet sont mises à jour
        self.env._update_orderbook_data()
        
        # Vérifier les données pour BTC
        btc_depth = self.env.market_constraints.orderbook_depth.get('BTC/USDT')
        self.assertIsNotNone(btc_depth)
        self.assertEqual(btc_depth['spread_pct'], 0.1)
        self.assertEqual(btc_depth['total_volume'], 1000.0)
        
        # Vérifier les données pour ETH
        eth_depth = self.env.market_constraints.orderbook_depth.get('ETH/USDT')
        self.assertIsNotNone(eth_depth)
        self.assertEqual(eth_depth['spread_pct'], 0.15)
        self.assertEqual(eth_depth['total_volume'], 2000.0)
        
    def test_extreme_market_conditions(self):
        """Teste le comportement dans des conditions de marché extrêmes."""
        # Simuler une forte volatilité et un faible volume
        self.data_dict['BTC/USDT'].loc[:, 'volume'] = 100.0  # Volume très faible
        self.data_dict['BTC/USDT'].loc[:, 'orderbook_depth'] = [
            {
                'spread_pct': 1.0,  # Spread très large
                'total_volume': 100.0,  # Faible liquidité
                'volume_imbalance': 0.8  # Fort déséquilibre
            }
        ] * 100
        
        # Action d'achat importante
        action = np.array([1.0, 0.0])
        initial_balance = self.env.balance
        
        # Exécution
        _, _, _, _ = self.env.step(action)
        
        # Vérifier que les protections ont fonctionné
        executed_cost = initial_balance - self.env.balance
        max_expected_cost = initial_balance * 1.05  # Max 5% de slippage
        self.assertLess(executed_cost, max_expected_cost)
        
    def test_multi_order_processing(self):
        """Teste le traitement de plusieurs ordres simultanés."""
        # Créer plusieurs ordres
        action = np.array([0.4, 0.4])  # 40% en BTC, 40% en ETH
        initial_balance = self.env.balance
        
        # Premier pas
        _, _, _, _ = self.env.step(action)
        
        # Vérifier que les deux ordres sont en attente
        self.assertEqual(len(self.env.pending_orders), 2)
        
        # Vérifier que les ordres ont des délais corrects
        for order in self.env.pending_orders:
            self.assertEqual(order['delay'], self.env.market_constraints.execution_delay)
            
        # Attendre l'exécution complète
        for _ in range(self.env.market_constraints.execution_delay):
            _, _, _, _ = self.env.step(np.array([0.0, 0.0]))
            
        # Un pas supplémentaire pour l'exécution finale
        _, _, _, _ = self.env.step(np.array([0.0, 0.0]))
        
        # Vérifier que tous les ordres ont été exécutés
        self.assertEqual(len(self.env.pending_orders), 0)
        self.assertLess(self.env.balance, initial_balance)  # Le solde doit avoir diminué
        self.assertGreater(self.env.crypto_holdings['BTC/USDT'], 0)  # Doit avoir du BTC
        self.assertGreater(self.env.crypto_holdings['ETH/USDT'], 0)  # Doit avoir du ETH
        
if __name__ == '__main__':
    unittest.main() 