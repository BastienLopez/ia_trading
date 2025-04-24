import unittest
import pandas as pd
import numpy as np
from ..utils.orderbook_collector import OrderBookCollector
from unittest.mock import Mock, patch

class TestOrderBookCollector(unittest.TestCase):
    def setUp(self):
        """Prépare l'environnement de test."""
        # Configuration multi-exchanges pour les tests
        self.symbols_config = [
            {'exchange': 'binance', 'symbol': 'BTC/USDT', 'limit': 100},
            {'exchange': 'kraken', 'symbol': 'ETH/USD', 'limit': 50}
        ]
        self.collector = OrderBookCollector(symbols_config=self.symbols_config)
        
        # Création d'un carnet d'ordres fictif pour les tests
        self.mock_orderbook = {
            'bids': [
                [50000.0, 1.0],  # prix, volume
                [49900.0, 2.0],
                [49800.0, 3.0],
                [49700.0, 4.0],
                [49600.0, 5.0],
            ],
            'asks': [
                [50100.0, 1.5],
                [50200.0, 2.5],
                [50300.0, 3.5],
                [50400.0, 4.5],
                [50500.0, 5.5],
            ]
        }
        
        # Création d'un carnet d'ordres avec forte pression d'achat
        self.mock_orderbook_buy_pressure = {
            'bids': [
                [50000.0, 5.0],
                [49900.0, 6.0],
                [49800.0, 7.0],
            ],
            'asks': [
                [50100.0, 1.0],
                [50200.0, 1.5],
                [50300.0, 2.0],
            ]
        }
        
        # Création d'un carnet d'ordres avec forte pression de vente
        self.mock_orderbook_sell_pressure = {
            'bids': [
                [50000.0, 1.0],
                [49900.0, 1.5],
                [49800.0, 2.0],
            ],
            'asks': [
                [50100.0, 5.0],
                [50200.0, 6.0],
                [50300.0, 7.0],
            ]
        }

    def test_multi_exchange_initialization(self):
        """Teste l'initialisation avec plusieurs exchanges."""
        self.assertEqual(len(self.collector.exchanges), 2)
        self.assertIn('binance', self.collector.exchanges)
        self.assertIn('kraken', self.collector.exchanges)

    def test_orderbook_features_with_pressure(self):
        """Teste le calcul des caractéristiques avec différentes pressions de marché."""
        # Test avec pression d'achat
        features_buy = self.collector.calculate_orderbook_features(self.mock_orderbook_buy_pressure)
        self.assertGreater(features_buy['volume_imbalance'], 0)
        
        # Test avec pression de vente
        features_sell = self.collector.calculate_orderbook_features(self.mock_orderbook_sell_pressure)
        self.assertLess(features_sell['volume_imbalance'], 0)

    def test_slippage_with_market_pressure(self):
        """Teste le calcul du slippage dans différentes conditions de marché."""
        # Test slippage avec pression d'achat
        slippage_buy_pressure = self.collector.calculate_slippage(
            self.mock_orderbook_buy_pressure, 'buy', 10.0
        )
        
        # Test slippage avec pression de vente
        slippage_sell_pressure = self.collector.calculate_slippage(
            self.mock_orderbook_sell_pressure, 'sell', 10.0
        )
        
        # Le slippage devrait être plus important dans le sens opposé à la pression
        self.assertGreater(slippage_buy_pressure, 
                          self.collector.calculate_slippage(self.mock_orderbook_buy_pressure, 'sell', 10.0))
        self.assertGreater(slippage_sell_pressure,
                          self.collector.calculate_slippage(self.mock_orderbook_sell_pressure, 'buy', 10.0))

    def test_market_impact_scenarios(self):
        """Teste l'impact sur le marché dans différents scénarios."""
        # Impact avec petit ordre
        small_impact = self.collector.estimate_market_impact(self.mock_orderbook, 'buy', 0.1)
        
        # Impact avec gros ordre
        large_impact = self.collector.estimate_market_impact(self.mock_orderbook, 'buy', 10.0)
        
        # L'impact devrait être plus important pour les gros ordres
        self.assertGreater(large_impact['immediate_impact_pct'], small_impact['immediate_impact_pct'])
        self.assertGreater(large_impact['estimated_recovery_time'], small_impact['estimated_recovery_time'])

    def test_execution_delay_edge_cases(self):
        """Teste les cas limites pour les délais d'exécution."""
        # Test avec charge maximale
        max_delay = self.collector.simulate_execution_delay(1.0, current_load=1.0)
        
        # Test avec charge minimale
        min_delay = self.collector.simulate_execution_delay(1.0, current_load=0.0)
        
        # Test avec très gros volume
        large_volume_delay = self.collector.simulate_execution_delay(1000.0, current_load=0.5)  # Volume augmenté à 1000
        
        # Vérifications
        self.assertGreater(max_delay, min_delay, "Le délai maximal devrait être plus grand que le délai minimal")
        self.assertGreater(large_volume_delay, max_delay, "Le délai pour un gros volume devrait être plus grand que le délai maximal")

    def test_vwap_with_market_pressure(self):
        """Teste le calcul VWAP dans différentes conditions de marché."""
        # VWAP avec pression d'achat
        vwap_bid_buy, vwap_ask_buy = self.collector.get_vwap_levels(
            self.mock_orderbook_buy_pressure, 5.0
        )
        
        # VWAP avec pression de vente
        vwap_bid_sell, vwap_ask_sell = self.collector.get_vwap_levels(
            self.mock_orderbook_sell_pressure, 5.0
        )
        
        # Vérification que le VWAP reflète la pression du marché
        self.assertGreater(vwap_bid_buy, vwap_bid_sell)
        self.assertGreater(vwap_ask_buy, vwap_ask_sell)

    @patch('ccxt.binance')
    def test_fetch_orderbook(self, mock_binance):
        """Teste la récupération du carnet d'ordres."""
        # Configuration du mock
        mock_exchange = Mock()
        mock_exchange.fetch_order_book.return_value = self.mock_orderbook
        mock_binance.return_value = mock_exchange
        
        # Test
        collector = OrderBookCollector()
        orderbook = collector.fetch_orderbook('binance', 'BTC/USDT')
        
        self.assertEqual(orderbook, self.mock_orderbook)
        mock_exchange.fetch_order_book.assert_called_once_with('BTC/USDT', 100)

    def test_calculate_orderbook_features(self):
        """Teste le calcul des caractéristiques du carnet d'ordres."""
        features = self.collector.calculate_orderbook_features(self.mock_orderbook)
        
        # Vérification des caractéristiques de base
        self.assertIn('mid_price', features)
        self.assertIn('spread', features)
        self.assertIn('spread_pct', features)
        self.assertIn('volume_imbalance', features)
        
        # Vérification des calculs
        expected_mid_price = (50000.0 + 50100.0) / 2
        self.assertEqual(features['mid_price'], expected_mid_price)
        
        expected_spread = 50100.0 - 50000.0
        self.assertEqual(features['spread'], expected_spread)
        
        # Vérification des caractéristiques de profondeur
        self.assertIn('depth_range_bids_5', features)
        self.assertIn('depth_range_asks_5', features)
        self.assertIn('depth_volume_bids_5', features)
        self.assertIn('depth_volume_asks_5', features)

    def test_get_vwap_levels(self):
        """Teste le calcul des niveaux VWAP."""
        volume_threshold = 3.0
        vwap_bid, vwap_ask = self.collector.get_vwap_levels(self.mock_orderbook, volume_threshold)
        
        # Vérification que les VWAP sont calculés
        self.assertIsNotNone(vwap_bid)
        self.assertIsNotNone(vwap_ask)
        
        # Vérification que les VWAP sont dans les limites attendues
        self.assertTrue(vwap_bid <= self.mock_orderbook['bids'][0][0])
        self.assertTrue(vwap_ask >= self.mock_orderbook['asks'][0][0])

    @patch('time.sleep')  # Pour éviter les délais dans les tests
    @patch('ccxt.binance')
    def test_collect_orderbook_data(self, mock_binance, mock_sleep):
        """Teste la collecte des données sur une période."""
        # Configuration des mocks
        mock_exchange = Mock()
        mock_exchange.fetch_order_book.return_value = self.mock_orderbook
        mock_binance.return_value = mock_exchange
        mock_sleep.return_value = None
        
        # Test avec une courte durée
        collector = OrderBookCollector([{'exchange': 'binance', 'symbol': 'BTC/USDT', 'limit': 100}])
        data_dict = collector.collect_orderbook_data(duration_minutes=0.1, interval_seconds=1)
        
        # Vérification du dictionnaire de DataFrames
        self.assertIsInstance(data_dict, dict)
        self.assertIn('binance_BTC/USDT', data_dict)
        
        # Vérification du DataFrame pour BTC/USDT
        df = data_dict['binance_BTC/USDT']
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertIn('mid_price', df.columns)
        self.assertIn('timestamp', df.columns)

    def test_calculate_slippage(self):
        """Teste le calcul du slippage."""
        # Test avec un ordre d'achat
        slippage_buy = self.collector.calculate_slippage(self.mock_orderbook, 'buy', 2.0)
        self.assertIsInstance(slippage_buy, float)
        self.assertGreaterEqual(slippage_buy, 0.0)
        
        # Test avec un ordre de vente
        slippage_sell = self.collector.calculate_slippage(self.mock_orderbook, 'sell', 2.0)
        self.assertIsInstance(slippage_sell, float)
        self.assertGreaterEqual(slippage_sell, 0.0)
        
        # Test avec un volume nul
        slippage_zero = self.collector.calculate_slippage(self.mock_orderbook, 'buy', 0.0)
        self.assertEqual(slippage_zero, 0.0)

    def test_estimate_market_impact(self):
        """Teste l'estimation de l'impact sur le marché."""
        impact = self.collector.estimate_market_impact(self.mock_orderbook, 'buy', 1.0)
        
        # Vérification des métriques retournées
        self.assertIn('immediate_impact_pct', impact)
        self.assertIn('volume_ratio', impact)
        self.assertIn('estimated_recovery_time', impact)
        
        # Vérification des valeurs
        self.assertGreaterEqual(impact['immediate_impact_pct'], 0.0)
        self.assertGreaterEqual(impact['volume_ratio'], 0.0)
        self.assertLess(impact['volume_ratio'], 1.0)
        self.assertGreaterEqual(impact['estimated_recovery_time'], 0.0)

    def test_simulate_execution_delay(self):
        """Teste la simulation des délais d'exécution."""
        # Test avec différentes charges de marché
        delay_low = self.collector.simulate_execution_delay(1.0, current_load=0.1)
        delay_high = self.collector.simulate_execution_delay(1.0, current_load=0.9)
        
        # Le délai devrait augmenter avec la charge
        self.assertGreater(delay_high, delay_low)
        
        # Test avec différents volumes
        delay_small = self.collector.simulate_execution_delay(0.1, current_load=0.5)
        delay_large = self.collector.simulate_execution_delay(10.0, current_load=0.5)
        
        # Le délai devrait augmenter avec le volume
        self.assertGreater(delay_large, delay_small)

    def test_get_execution_metrics(self):
        """Teste le calcul des métriques d'exécution complètes."""
        metrics = self.collector.get_execution_metrics(self.mock_orderbook, 'buy', 1.0)
        
        # Vérification de toutes les métriques
        self.assertIn('slippage_pct', metrics)
        self.assertIn('execution_delay_seconds', metrics)
        self.assertIn('immediate_impact_pct', metrics)
        self.assertIn('volume_ratio', metrics)
        self.assertIn('estimated_recovery_time', metrics)
        
        # Vérification des valeurs
        for value in metrics.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)

    def test_model_calibration(self):
        """Teste la calibration des modèles avec des données historiques."""
        # Création de données historiques fictives
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
            'volume': np.random.lognormal(0, 1, 100),
            'price_before': 50000 + np.random.normal(0, 100, 100),
            'price_after': 50000 + np.random.normal(0, 150, 100),
            'execution_time': np.random.lognormal(0, 0.5, 100)
        })
        
        # Test de la calibration
        params = self.collector.calibrate_models(historical_data)
        
        # Vérification des paramètres calibrés
        self.assertIn('impact_coefficient', params)
        self.assertIn('impact_intercept', params)
        self.assertIn('base_delay', params)
        self.assertIn('max_delay', params)
        self.assertIn('delay_coefficient', params)
        self.assertIn('slippage_coefficient', params)
        self.assertIn('slippage_intercept', params)
        
        # Test avec données vides
        empty_params = self.collector.calibrate_models(pd.DataFrame())
        self.assertEqual(empty_params, {})
        
    def test_calibration_impact(self):
        """Teste spécifiquement la calibration du modèle d'impact."""
        # Création de données avec une relation volume/impact connue
        volumes = np.array([1, 2, 5, 10, 20])
        prices_before = np.array([50000] * 5)
        prices_after = prices_before * (1 + 0.001 * volumes)  # Impact de 0.1% par unité de volume
        
        data = pd.DataFrame({
            'volume': volumes,
            'price_before': prices_before,
            'price_after': prices_after,
            'timestamp': pd.date_range(start='2024-01-01', periods=5),
            'execution_time': np.ones(5)
        })
        
        params = self.collector._calibrate_market_impact(data)
        
        # Vérification que le coefficient est positif (plus de volume = plus d'impact)
        self.assertGreater(params['impact_coefficient'], 0)
        
    def test_calibration_delay(self):
        """Teste spécifiquement la calibration du modèle de délai."""
        # Création de données avec une relation volume/délai connue
        volumes = np.array([1, 2, 5, 10, 20])
        delays = 0.1 * volumes + 0.05  # Délai linéaire avec le volume
        
        data = pd.DataFrame({
            'volume': volumes,
            'execution_time': delays,
            'price_before': np.ones(5),
            'price_after': np.ones(5),
            'timestamp': pd.date_range(start='2024-01-01', periods=5)
        })
        
        params = self.collector._calibrate_execution_delay(data)
        
        # Vérification des délais calibrés
        self.assertGreater(params['max_delay'], params['base_delay'])
        self.assertGreater(params['delay_coefficient'], 0)
        
    def test_calibration_application(self):
        """Teste l'application des paramètres calibrés."""
        params = {
            'impact_coefficient': 0.1,
            'impact_intercept': 0.01,
            'base_delay': 0.05,
            'max_delay': 1.0,
            'delay_coefficient': 0.2,
            'slippage_coefficient': 0.15,
            'slippage_intercept': 0.02
        }
        
        self.collector.apply_calibration(params)
        self.assertEqual(self.collector.model_params, params)

if __name__ == '__main__':
    unittest.main() 