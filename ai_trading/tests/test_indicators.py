import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading.indicators.timeframe_manager import TimeframeManager
from ai_trading.indicators.sr_manager import SRManager
from ai_trading.indicators.break_retest import BreakRetestDetector

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Création des données de test
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')
        n = len(dates)
        
        # Création d'un DataFrame avec des données OHLCV simulées
        np.random.seed(42)  # Pour la reproductibilité
        base_price = 100
        
        # Génération de prix avec une tendance
        trend = np.linspace(0, 10, n)
        noise = np.random.normal(0, 1, n)
        prices = base_price + trend + noise
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 0.5, n),
            'low': prices - np.random.uniform(0, 0.5, n),
            'close': prices + np.random.normal(0, 0.2, n),
            'volume': np.random.normal(1000, 100, n)
        }, index=dates)
        
        # S'assurer que high > low et cohérence OHLC
        self.test_data['high'] = self.test_data[['open', 'high', 'close']].max(axis=1)
        self.test_data['low'] = self.test_data[['open', 'low', 'close']].min(axis=1)
        
        # Initialisation des gestionnaires
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.tf_manager = TimeframeManager(self.timeframes)
        self.sr_manager = SRManager(self.timeframes)
        self.br_detector = BreakRetestDetector()

    def test_timeframe_manager(self):
        """Test du TimeframeManager"""
        # Test de l'initialisation
        self.assertEqual(len(self.tf_manager.timeframes), 5)
        
        # Test de la mise à jour des données
        self.tf_manager.update_data('1m', self.test_data)
        df_1m = self.tf_manager.get_data('1m')
        self.assertIsNotNone(df_1m)
        
        # Test du resampling
        self.tf_manager.update_data('5m', self.test_data)
        df_5m = self.tf_manager.get_data('5m')
        self.assertIsNotNone(df_5m)
        self.assertTrue(len(df_5m) < len(self.test_data))
        
        # Test de la fraîcheur des données
        self.assertTrue(self.tf_manager.is_data_fresh('1m', timedelta(hours=1)))

    def test_sr_manager(self):
        """Test du SRManager"""
        # Mise à jour des niveaux
        self.sr_manager.update_all_timeframes(self.test_data)
        
        # Vérification des niveaux créés
        for tf in self.timeframes:
            levels = self.sr_manager.get_active_levels(tf)
            self.assertIsInstance(levels, list)
            
        # Test de la recherche des niveaux proches
        current_price = 100.0
        below, above = self.sr_manager.get_nearest_levels(current_price, '1h')
        self.assertIsInstance(below, list)
        self.assertIsInstance(above, list)

    def test_break_retest_detector(self):
        """Test du BreakRetestDetector"""
        # Création d'une série de prix avec une cassure évidente
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        level_price = 100.0
        
        # Création des prix avec une cassure claire et un retest
        prices = []
        for i in range(100):
            if i < 50:
                prices.append(level_price - 0.5)  # Prix sous le niveau
            elif i == 50:
                prices.append(level_price + 2.0)  # Cassure haussière de 2%
            elif i > 50 and i < 70:
                prices.append(level_price + 2.5)  # Prix au-dessus du niveau
            elif i == 70:
                prices.append(level_price + 0.1)  # Retest du niveau
            else:
                prices.append(level_price + 2.0)  # Retour au-dessus
        
        # Augmentation du volume à la cassure et au retest
        volumes = np.ones(100) * 1000
        volumes[50] = 3000  # Volume triplé à la cassure
        volumes[70] = 2000  # Volume doublé au retest
        
        # Création du DataFrame avec une structure OHLC claire
        test_df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.2 for p in prices],
            'low': [p - 0.2 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Test de détection de cassure
        detector = BreakRetestDetector(volume_threshold=1.5)  # Seuil de volume plus bas pour le test
        break_event = detector.detect_break(test_df, level_price)
        
        self.assertIsNotNone(break_event, "La cassure n'a pas été détectée")
        if break_event:
            self.assertEqual(break_event.direction, 'up')
            self.assertTrue(break_event.strength > 0)
            
            # Test de détection de retest
            retest_event = detector.detect_retest(test_df, level_price, break_event)
            self.assertIsNotNone(retest_event, "Le retest n'a pas été détecté")

if __name__ == '__main__':
    unittest.main() 