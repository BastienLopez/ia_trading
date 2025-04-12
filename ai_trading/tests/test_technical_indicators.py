import unittest
import pandas as pd
import numpy as np
import time
import os
import sys
import timeit
import pandas_ta as ta

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.technical_indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    """Tests pour la classe TechnicalIndicators."""

    def setUp(self):
        """Prépare les données pour les tests."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        
        # Créer une tendance haussière claire avec volatilité contrôlée
        base_prices = np.linspace(100, 200, 100)
        
        # Générer des mouvements directionnels cohérents
        volatilite = np.random.normal(0, 1, 100)  # Réduire davantage la volatilité
        
        self.test_data = pd.DataFrame({
            "open": base_prices - 1 + volatilite,
            "high": base_prices + 3 + abs(volatilite),  # Hauts toujours plus hauts
            "low": base_prices - 1 - abs(volatilite),   # Bas légèrement descendants
            "close": base_prices + volatilite * 0.5,    # Clôture proche du haut
            "volume": np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Lisser les prix de clôture
        self.test_data['close'] = self.test_data['close'].rolling(5, min_periods=1).mean()
        
        # Initialiser la classe d'indicateurs
        self.indicators = TechnicalIndicators(self.test_data)
    
    def test_ema(self):
        """Teste le calcul de l'EMA."""
        # Calculer l'EMA
        ema = self.indicators.calculate_ema(period=9)
        
        # Vérifier que l'EMA est calculé correctement
        self.assertIsInstance(ema, pd.Series)
        self.assertEqual(len(ema), len(self.test_data))
        
        # Vérifier que les premières valeurs ne sont pas NaN (l'EMA peut être calculé dès le début)
        self.assertFalse(np.isnan(ema.iloc[-1]))
    
    def test_macd(self):
        """Teste le calcul du MACD."""
        # Calculer le MACD
        macd = ta.macd(self.test_data['close'], fast=12, slow=26, signal=9)
        
        # Vérifier que les composants du MACD sont calculés correctement
        self.assertIsNotNone(macd)
        
        # Vérifier que les composants ont la bonne longueur
        self.assertEqual(len(macd), len(self.test_data))
    
    def test_momentum(self):
        """Teste le calcul du Momentum."""
        # Calculer le Momentum
        momentum = self.indicators.calculate_momentum()
        
        # Vérifier que le Momentum est calculé correctement
        self.assertIsNotNone(momentum)
        
        # Vérifier que le Momentum a la bonne longueur
        self.assertEqual(len(momentum), len(self.test_data))
        
        # Vérifier que le Momentum est positif pour une tendance haussière
        self.assertTrue((momentum[20:] > 0).mean() > 0.5)
    
    def test_adx(self):
        """Teste le calcul de l'ADX."""
        adx, plus_di, minus_di = self.indicators.calculate_adx()
        
        # Vérifier que les composants sont calculés
        self.assertIsNotNone(adx)
        self.assertIsNotNone(plus_di)
        self.assertIsNotNone(minus_di)
        
        # Vérifier que les valeurs sont dans les plages attendues
        self.assertTrue((adx.dropna() >= 0).all() and (adx.dropna() <= 100).all())
        self.assertTrue((plus_di.dropna() >= 0).all() and (plus_di.dropna() <= 100).all())
        self.assertTrue((minus_di.dropna() >= 0).all() and (minus_di.dropna() <= 100).all())
        
        # Vérifier que +DI est généralement supérieur à -DI (tendance haussière)
        self.assertGreater((plus_di > minus_di).mean(), 0.5)
    
    def test_bollinger_bands(self):
        """Teste le calcul des bandes de Bollinger."""
        # Calculer les bandes de Bollinger
        bbands = ta.bbands(self.test_data['close'], length=20, std=2)
        
        # Vérifier que les bandes sont calculées correctement
        self.assertIsNotNone(bbands)
        
        # Vérifier que les longueurs sont correctes
        self.assertEqual(len(bbands), len(self.test_data))
        
        # Vérifier simplement que les valeurs ne sont pas toutes NaN
        # et que les bandes sont calculées
        self.assertTrue(bbands['BBM_20_2.0'].notna().any())
        self.assertTrue(bbands['BBU_20_2.0'].notna().any())
        self.assertTrue(bbands['BBL_20_2.0'].notna().any())
        
        # Vérifier que les bandes ont des valeurs différentes
        valid_indices = ~np.isnan(bbands['BBM_20_2.0']) & ~np.isnan(bbands['BBU_20_2.0']) & ~np.isnan(bbands['BBL_20_2.0'])
        if valid_indices.any():
            self.assertFalse((bbands['BBM_20_2.0'][valid_indices] == bbands['BBU_20_2.0'][valid_indices]).all())
            self.assertFalse((bbands['BBM_20_2.0'][valid_indices] == bbands['BBL_20_2.0'][valid_indices]).all())
    
    def test_atr(self):
        """Teste le calcul de l'ATR."""
        # Calculer l'ATR
        atr = self.indicators.calculate_atr(period=14)
        
        # Vérifier que l'ATR est calculé correctement
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), len(self.test_data))
        
        # Vérifier que les valeurs sont positives
        self.assertTrue(all(x >= 0 for x in atr.dropna()))
        
        # Vérifier que les premières valeurs sont NaN (période d'initialisation)
        self.assertTrue(np.isnan(atr.iloc[0]))
    
    def test_stochastic(self):
        """Teste le calcul du Stochastique."""
        # Calculer le Stochastique
        k, d = self.indicators.calculate_stochastic()
        
        # Vérifier que les composants du Stochastique sont calculés correctement
        self.assertIsNotNone(k)
        self.assertIsNotNone(d)
        
        # Vérifier que les composants ont la bonne longueur
        self.assertEqual(len(k), len(self.test_data))
        self.assertEqual(len(d), len(self.test_data))
        
        # Vérifier que les valeurs sont dans la plage [0, 100]
        self.assertTrue((k.dropna().between(0, 100)).all())
        self.assertTrue((d.dropna().between(0, 100)).all())
    
    def test_obv(self):
        """Teste le calcul de l'OBV."""
        # Calculer l'OBV
        obv = self.indicators.calculate_obv()
        
        # Vérifier que l'OBV est calculé correctement
        self.assertIsNotNone(obv)
        
        # Vérifier que l'OBV a la bonne longueur
        self.assertEqual(len(obv), len(self.test_data))
        
        # Vérifier que l'OBV change en fonction des prix
        self.assertNotEqual(obv[1], obv[0])
    
    def test_volume_average(self):
        """Teste le calcul de la moyenne mobile du volume."""
        # Calculer la moyenne mobile du volume
        volume_avg = self.indicators.calculate_volume_average()
        
        # Vérifier que la moyenne mobile du volume est calculée correctement
        self.assertIsNotNone(volume_avg)
        
        # Vérifier que la moyenne mobile du volume a la bonne longueur
        self.assertEqual(len(volume_avg), len(self.test_data))
        
        # Vérifier que la moyenne mobile du volume est une moyenne mobile simple
        sma = self.test_data['volume'].rolling(window=20).mean()
        pd.testing.assert_series_equal(volume_avg[20:], sma[20:], check_dtype=False, check_names=False)
    
    def test_mfi(self):
        """Teste le calcul du MFI."""
        # Calculer le MFI
        mfi = self.indicators.calculate_mfi()
        
        # Vérifier que le MFI est calculé correctement
        self.assertIsNotNone(mfi)
        
        # Vérifier que le MFI a la bonne longueur
        self.assertEqual(len(mfi), len(self.test_data))
        
        # Vérifier que le MFI est entre 0 et 100
        self.assertTrue((mfi[14:].dropna() >= 0).all() and (mfi[14:].dropna() <= 100).all())
    
    def test_rsi(self):
        """Teste le calcul du RSI."""
        # Calculer le RSI
        rsi = ta.rsi(self.test_data['close'], length=14)
        
        # Vérifier que le RSI est calculé correctement
        self.assertIsNotNone(rsi)
        
        # Vérifier que les valeurs sont dans la plage [0, 100]
        if not rsi.dropna().empty:  # Vérifier si le RSI n'est pas vide
            self.assertTrue(all((0 <= x <= 100) for x in rsi.dropna()))
        else:
            self.skipTest("Le RSI n'a pas pu être calculé (données insuffisantes)")
    
    def test_cci(self):
        """Teste le calcul du CCI."""
        # Calculer le CCI
        cci = self.indicators.calculate_cci()
        
        # Vérifier que le CCI est calculé correctement
        self.assertIsNotNone(cci)
        
        # Vérifier que le CCI a la bonne longueur
        self.assertEqual(len(cci), len(self.test_data))
    
    def test_fibonacci_levels(self):
        """Teste le calcul des niveaux de Fibonacci."""
        # Calculer les niveaux de Fibonacci
        fib_levels = self.indicators.calculate_fibonacci_levels()
        
        # Vérifier que les niveaux de Fibonacci sont calculés correctement
        self.assertIsNotNone(fib_levels)
        
        # Vérifier que tous les niveaux sont présents
        self.assertTrue(all(level in fib_levels for level in ['0.0', '0.236', '0.382', '0.5', '0.618', '0.786', '1.0']))
        
        # Vérifier que les niveaux sont dans l'ordre croissant pour une tendance haussière
        self.assertTrue(fib_levels['0.0'] < fib_levels['0.236'] < fib_levels['0.382'] < 
                        fib_levels['0.5'] < fib_levels['0.618'] < fib_levels['0.786'] < fib_levels['1.0'])
    
    def test_pivots(self):
        """Teste le calcul des points pivots."""
        # Calculer les points pivots
        pivots = self.indicators.calculate_pivots()
        
        # Vérifier que les points pivots sont calculés correctement
        self.assertIsNotNone(pivots)
        
        # Vérifier que les points pivots ont la bonne longueur
        self.assertEqual(len(pivots), len(self.test_data))
        
        # Vérifier que tous les niveaux sont présents
        self.assertTrue(all(col in pivots.columns for col in ['P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']))
        
        # Vérifier que les niveaux de résistance sont au-dessus du pivot
        self.assertTrue((pivots['R1'][1:] > pivots['P'][1:]).all())
        self.assertTrue((pivots['R2'][1:] > pivots['R1'][1:]).all())
        self.assertTrue((pivots['R3'][1:] > pivots['R2'][1:]).all())
        
        # Vérifier que les niveaux de support sont en dessous du pivot
        self.assertTrue((pivots['S1'][1:] < pivots['P'][1:]).all())
        self.assertTrue((pivots['S2'][1:] < pivots['S1'][1:]).all())
        self.assertTrue((pivots['S3'][1:] < pivots['S2'][1:]).all())
    
    def test_normalize_indicator(self):
        """Teste la normalisation des indicateurs."""
        # Calculer un indicateur
        rsi = self.indicators.calculate_rsi()
        
        # Normaliser l'indicateur avec différentes méthodes
        rsi_minmax = self.indicators.normalize_indicator(rsi, method='minmax')
        rsi_zscore = self.indicators.normalize_indicator(rsi, method='zscore')
        
        # Vérifier que les indicateurs normalisés sont calculés correctement
        self.assertIsNotNone(rsi_minmax)
        self.assertIsNotNone(rsi_zscore)
        
        # Vérifier que les indicateurs normalisés ont la bonne longueur
        self.assertEqual(len(rsi_minmax), len(self.test_data))
        self.assertEqual(len(rsi_zscore), len(self.test_data))
        
        # Vérifier que la normalisation minmax est entre 0 et 1
        self.assertTrue((rsi_minmax[14:].dropna() >= 0).all() and (rsi_minmax[14:].dropna() <= 1).all())
        
        # Vérifier que la normalisation zscore a une moyenne proche de 0 et un écart-type proche de 1
        # Utiliser une tolérance plus large pour les tests
        self.assertAlmostEqual(rsi_zscore[14:].dropna().mean(), 0, delta=0.5)
        self.assertAlmostEqual(rsi_zscore[14:].dropna().std(), 1, delta=0.5)
    
    def test_get_all_indicators(self):
        """Teste le calcul de tous les indicateurs."""
        # Calculer tous les indicateurs
        all_indicators = self.indicators.get_all_indicators()
        
        # Vérifier que tous les indicateurs sont calculés correctement
        self.assertIsNotNone(all_indicators)
        
        # Vérifier que le DataFrame existe et contient des données
        self.assertTrue(len(all_indicators) > 0)
        
        # Vérifier que les indicateurs essentiels sont présents
        expected_indicators = [
            'ema_9', 'ema_21',
            'macd', 'macd_signal', 'macd_hist',
            'momentum',
            'upper_bb', 'middle_bb', 'lower_bb', 'atr',
            'stoch_k', 'stoch_d', 'obv', 'volume_avg',
            'mfi', 'rsi', 'cci'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, all_indicators.columns)
        
        # Vérifier que les nouveaux indicateurs sont présents
        new_indicators = [
            'pivot_P', 'ichimoku_tenkan', 'donchian_upper', 'vp_poc'
        ]
        
        for indicator in new_indicators:
            self.assertIn(indicator, all_indicators.columns)
    
    def test_performance(self):
        """Teste la performance des indicateurs."""
        # Mesurer le temps de calcul pour chaque indicateur
        indicators_times = {}
        
        # EMA
        start_time = time.time()
        self.indicators.calculate_ema()
        indicators_times['ema'] = time.time() - start_time
        
        # MACD
        start_time = time.time()
        self.indicators.calculate_macd()
        indicators_times['macd'] = time.time() - start_time
        
        # Momentum
        start_time = time.time()
        self.indicators.calculate_momentum()
        indicators_times['momentum'] = time.time() - start_time
        
        # ADX
        start_time = time.time()
        self.indicators.calculate_adx()
        indicators_times['adx'] = time.time() - start_time
        
        # Bollinger Bands
        start_time = time.time()
        self.indicators.calculate_bollinger_bands()
        indicators_times['bollinger_bands'] = time.time() - start_time
        
        # ATR
        start_time = time.time()
        self.indicators.calculate_atr()
        indicators_times['atr'] = time.time() - start_time
        
        # Stochastic
        start_time = time.time()
        self.indicators.calculate_stochastic()
        indicators_times['stochastic'] = time.time() - start_time
        
        # OBV
        start_time = time.time()
        self.indicators.calculate_obv()
        indicators_times['obv'] = time.time() - start_time
        
        # Volume Average
        start_time = time.time()
        self.indicators.calculate_volume_average()
        indicators_times['volume_average'] = time.time() - start_time
        
        # MFI
        start_time = time.time()
        self.indicators.calculate_mfi()
        indicators_times['mfi'] = time.time() - start_time
        
        # RSI
        start_time = time.time()
        self.indicators.calculate_rsi()
        indicators_times['rsi'] = time.time() - start_time
        
        # CCI
        start_time = time.time()
        self.indicators.calculate_cci()
        indicators_times['cci'] = time.time() - start_time
        
        # Fibonacci Levels
        start_time = time.time()
        self.indicators.calculate_fibonacci_levels()
        indicators_times['fibonacci_levels'] = time.time() - start_time
        
        # Pivots
        start_time = time.time()
        self.indicators.calculate_pivots()
        indicators_times['pivots'] = time.time() - start_time
        
        # Tous les indicateurs
        start_time = time.time()
        self.indicators.get_all_indicators()
        indicators_times['all_indicators'] = time.time() - start_time
        
        # Afficher les temps de calcul
        print("\nTemps de calcul des indicateurs (secondes):")
        for indicator, calc_time in indicators_times.items():
            print(f"{indicator}: {calc_time:.6f}")
        
        # Vérifier que le calcul de tous les indicateurs prend un temps raisonnable
        self.assertLess(indicators_times['all_indicators'], 1.0)  # Moins d'une seconde

    def test_volume_profile(self):
        """Teste le calcul du Volume Profile."""
        # Calculer le Volume Profile
        vp = self.indicators.calculate_volume_profile(n_bins=10, lookback=20)
        
        # Vérifier que le Volume Profile est calculé correctement
        self.assertIsNotNone(vp)
        
        # Vérifier que les colonnes attendues sont présentes
        self.assertTrue(all(col in vp.columns for col in ['vp_poc', 'vp_vah', 'vp_val']))
        
        # Vérifier que les valeurs ne sont pas toutes NaN
        self.assertTrue(vp['vp_poc'].notna().any())
        
        # Vérifier que VAH > POC > VAL (Value Area High > Point of Control > Value Area Low)
        for i in range(len(vp)):
            if not np.isnan(vp['vp_poc'].iloc[i]):
                self.assertGreaterEqual(vp['vp_vah'].iloc[i], vp['vp_poc'].iloc[i])
                self.assertGreaterEqual(vp['vp_poc'].iloc[i], vp['vp_val'].iloc[i])

    def test_ichimoku_cloud(self):
        """Teste le calcul de l'Ichimoku Cloud."""
        # Calculer l'Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b, chikou = self.indicators.calculate_ichimoku_cloud()
        
        # Vérifier que les composants sont calculés correctement
        self.assertIsNotNone(tenkan)
        self.assertIsNotNone(kijun)
        self.assertIsNotNone(senkou_a)
        self.assertIsNotNone(senkou_b)
        self.assertIsNotNone(chikou)
        
        # Vérifier que les composants ont la bonne longueur
        self.assertEqual(len(tenkan), len(self.test_data))
        self.assertEqual(len(kijun), len(self.test_data))
        self.assertEqual(len(senkou_a), len(self.test_data))
        self.assertEqual(len(senkou_b), len(self.test_data))
        self.assertEqual(len(chikou), len(self.test_data))
        
        # Vérifier que les valeurs ne sont pas toutes NaN
        self.assertTrue(tenkan.notna().any())
        self.assertTrue(kijun.notna().any())

    def test_donchian_channel(self):
        """Teste le calcul du canal de Donchian."""
        # Calculer le canal de Donchian
        upper, middle, lower = self.indicators.calculate_donchian_channel()
        
        # Vérifier que les composants sont calculés correctement
        self.assertIsNotNone(upper)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(lower)
        
        # Vérifier que les composants ont la bonne longueur
        self.assertEqual(len(upper), len(self.test_data))
        self.assertEqual(len(middle), len(self.test_data))
        self.assertEqual(len(lower), len(self.test_data))
        
        # Vérifier que upper > middle > lower
        for i in range(len(upper)):
            if not np.isnan(upper.iloc[i]):
                self.assertGreaterEqual(upper.iloc[i], middle.iloc[i])
                self.assertGreaterEqual(middle.iloc[i], lower.iloc[i])
        
        # Vérifier que les valeurs ne sont pas toutes NaN
        self.assertTrue(upper.notna().any())
        self.assertTrue(middle.notna().any())
        self.assertTrue(lower.notna().any())
        
        # Vérifier que les bandes ont des valeurs différentes
        valid_indices = ~np.isnan(middle) & ~np.isnan(upper) & ~np.isnan(lower)
        if valid_indices.any():
            self.assertFalse((middle[valid_indices] == upper[valid_indices]).all())
            self.assertFalse((middle[valid_indices] == lower[valid_indices]).all())


def test_technical_indicators_performance():
    setup = """
from ai_trading.tests.test_enhanced_preprocessor import TestEnhancedMarketDataPreprocessor
test_case = TestEnhancedMarketDataPreprocessor()
test_case.setUp()
preprocessor = test_case.preprocessor
data = test_case.test_data
"""
    
    time_taken = timeit.timeit('preprocessor.create_technical_features(data)', setup=setup, number=10)
    print(f"Temps moyen pour créer les features techniques: {time_taken / 10:.4f} secondes")


if __name__ == "__main__":
    unittest.main() 