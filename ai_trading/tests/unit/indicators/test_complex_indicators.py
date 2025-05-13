"""
Tests unitaires pour le module d'indicateurs techniques complexes.
"""
import unittest
import numpy as np
import pandas as pd
from ai_trading.indicators.complex_indicators import (
    WaveletAnalyzer, 
    SpectralAnalyzer,
    TimeFrame,
    MultiTimeframeIndicator,
    DivergenceType,
    DivergenceDetector
)


class TestWaveletAnalyzer(unittest.TestCase):
    """Tests pour la classe WaveletAnalyzer."""
    
    def setUp(self):
        """Initialise les données de test."""
        # Créer une série temporelle simple pour les tests
        np.random.seed(42)
        self.trend = np.linspace(0, 10, 256)
        self.noise = np.random.normal(0, 0.5, 256)
        self.cycle = 2 * np.sin(np.linspace(0, 8 * np.pi, 256))
        
        # Série temporelle composite (tendance + cycle + bruit)
        self.data = self.trend + self.cycle + self.noise
        
        # Créer l'analyseur d'ondelettes
        self.analyzer = WaveletAnalyzer(wavelet='db4', level=3)
        
    def test_decompose(self):
        """Teste la décomposition en ondelettes."""
        # Décomposer les données
        coeffs = self.analyzer.decompose(self.data)
        
        # Vérifier les clés du dictionnaire de coefficients
        self.assertIn('approximation', coeffs)
        self.assertIn('detail_1', coeffs)
        self.assertIn('detail_2', coeffs)
        self.assertIn('detail_3', coeffs)
        
        # Vérifier que les coefficients d'approximation capturent la tendance
        self.assertTrue(len(coeffs['approximation']) > 0)
        
    def test_reconstruct(self):
        """Teste la reconstruction à partir des coefficients."""
        # Décomposer puis reconstruire
        coeffs = self.analyzer.decompose(self.data)
        reconstructed = self.analyzer.reconstruct(coeffs)
        
        # Vérifier que la reconstruction est proche de l'original
        self.assertEqual(len(reconstructed), len(self.data))
        # Tolérance de reconstruction
        np.testing.assert_allclose(reconstructed, self.data, rtol=1e-10, atol=1e-10)
        
    def test_denoise(self):
        """Teste le débruitage par ondelettes."""
        # Appliquer le débruitage
        denoised = self.analyzer.denoise(self.data, threshold=0.3)
        
        # Vérifier que le débruitage réduit la variance du bruit
        original_noise = np.std(self.data - (self.trend + self.cycle))
        denoised_noise = np.std(denoised - (self.trend + self.cycle))
        
        # Le bruit devrait être réduit après débruitage
        self.assertLess(denoised_noise, original_noise)


class TestSpectralAnalyzer(unittest.TestCase):
    """Tests pour la classe SpectralAnalyzer."""
    
    def setUp(self):
        """Initialise les données de test."""
        # Créer une série temporelle avec des cycles connus
        np.random.seed(42)
        t = np.linspace(0, 10, 1000)
        
        # Cycles de 50, 100 et 200 points
        cycle1 = 3 * np.sin(2 * np.pi * t * 20)  # Cycle de période 50 points
        cycle2 = 2 * np.sin(2 * np.pi * t * 10)  # Cycle de période 100 points
        cycle3 = 1 * np.sin(2 * np.pi * t * 5)   # Cycle de période 200 points
        
        # Ajouter du bruit et une tendance
        noise = np.random.normal(0, 0.5, 1000)
        trend = 0.2 * t
        
        # Série temporelle composite
        self.data = cycle1 + cycle2 + cycle3 + noise + trend
        
        # Créer l'analyseur spectral
        self.analyzer = SpectralAnalyzer(sampling_freq=100)  # 100 Hz
        
    def test_periodogram(self):
        """Teste le calcul du périodogramme."""
        # Calculer le périodogramme
        freqs, psd = self.analyzer.periodogram(self.data)
        
        # Vérifier les dimensions
        self.assertEqual(len(freqs), len(psd))
        self.assertTrue(len(freqs) > 0)
        
        # Vérifier que les fréquences sont positives
        self.assertTrue(np.all(freqs >= 0))
        
        # Vérifier que la PSD est positive
        self.assertTrue(np.all(psd >= 0))
        
    def test_find_dominant_cycles(self):
        """Teste la détection des cycles dominants."""
        # Trouver les cycles dominants
        cycles = self.analyzer.find_dominant_cycles(self.data, n_cycles=3)
        
        # Vérifier qu'on a bien trouvé 3 cycles
        self.assertEqual(len(cycles), 3)
        
        # Les périodes attendues (en échantillons)
        expected_periods = [5, 10, 20]  # Périodes en Hz
        expected_periods = [100 / p for p in expected_periods]  # Conversion en nombre de points
        
        # Vérifier que les cycles détectés sont proches des cycles attendus
        # avec une marge d'erreur de 10%
        for cycle, expected in zip(sorted(cycles), sorted(expected_periods)):
            self.assertAlmostEqual(cycle, expected, delta=expected * 0.1)
            
    def test_filter_by_frequency(self):
        """Teste le filtrage par fréquence."""
        # Filtrer pour ne garder que les basses fréquences (tendance)
        low_pass = self.analyzer.filter_by_frequency(self.data, high_cutoff=1)
        
        # Filtrer pour ne garder que les hautes fréquences (bruit)
        high_pass = self.analyzer.filter_by_frequency(self.data, low_cutoff=25)
        
        # Filtrer pour garder une bande de fréquences (cycle spécifique)
        band_pass = self.analyzer.filter_by_frequency(self.data, low_cutoff=9, high_cutoff=11)
        
        # Vérifier que les signaux filtrés ont la même longueur que l'original
        self.assertEqual(len(low_pass), len(self.data))
        self.assertEqual(len(high_pass), len(self.data))
        self.assertEqual(len(band_pass), len(self.data))
        
        # Vérifier que le filtre passe-bas réduit l'amplitude des hautes fréquences
        self.assertLess(np.std(low_pass), np.std(self.data))
        
        # Vérifier que le filtre passe-bande isole principalement le cycle de période 100
        # en comparant la puissance à cette fréquence
        _, psd_original = self.analyzer.periodogram(self.data)
        _, psd_band = self.analyzer.periodogram(band_pass)
        
        # Calculer la puissance relative dans la bande d'intérêt
        freq_idx = np.where((self.analyzer.periodogram(self.data)[0] >= 9) & 
                           (self.analyzer.periodogram(self.data)[0] <= 11))[0]
        
        power_ratio_original = np.sum(psd_original[freq_idx]) / np.sum(psd_original)
        power_ratio_band = np.sum(psd_band[freq_idx]) / np.sum(psd_band)
        
        # Le ratio devrait être plus élevé dans le signal filtré
        self.assertGreater(power_ratio_band, power_ratio_original)


class TestMultiTimeframeIndicator(unittest.TestCase):
    """Tests pour la classe MultiTimeframeIndicator."""
    
    def setUp(self):
        """Initialise les données de test."""
        # Créer des données OHLCV pour différents timeframes
        np.random.seed(42)
        
        # Timeframes à tester
        self.timeframes = [TimeFrame.HOUR_1, TimeFrame.HOUR_4, TimeFrame.DAY_1]
        
        # Créer des données pour chaque timeframe
        self.data = {}
        for tf in self.timeframes:
            # Nombre de points différents selon le timeframe
            n_points = 100 if tf == TimeFrame.HOUR_1 else (25 if tf == TimeFrame.HOUR_4 else 7)
            
            # Tendance selon le timeframe
            trend = np.linspace(0, 10, n_points)
            
            # Créer des données OHLCV réalistes
            close = 100 + trend + np.random.normal(0, 1, n_points)
            open_price = close - np.random.normal(0, 0.5, n_points)
            high = np.maximum(open_price, close) + np.random.uniform(0, 1, n_points)
            low = np.minimum(open_price, close) - np.random.uniform(0, 1, n_points)
            volume = 1000 + 500 * np.random.random(n_points)
            
            # Créer le DataFrame
            self.data[tf] = pd.DataFrame({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Créer l'indicateur multi-timeframes
        self.indicator = MultiTimeframeIndicator(self.timeframes)
        
        # Ajouter les données
        for tf, data in self.data.items():
            self.indicator.add_data(tf, data)
            
    def test_calculate_rsi(self):
        """Teste le calcul du RSI sur plusieurs timeframes."""
        # Calculer le RSI
        rsi_values = self.indicator.calculate_rsi(window=14)
        
        # Vérifier qu'on a des résultats pour chaque timeframe
        for tf in self.timeframes:
            self.assertIn(tf, rsi_values)
            
            # Vérifier que les valeurs sont dans la plage [0, 100]
            self.assertTrue(np.all(rsi_values[tf].dropna() >= 0))
            self.assertTrue(np.all(rsi_values[tf].dropna() <= 100))
            
            # Vérifier que la longueur est correcte
            self.assertEqual(len(rsi_values[tf]), len(self.data[tf]))
            
    def test_calculate_macd(self):
        """Teste le calcul du MACD sur plusieurs timeframes."""
        # Calculer le MACD
        macd_values = self.indicator.calculate_macd(fast=12, slow=26, signal=9)
        
        # Vérifier qu'on a des résultats pour chaque timeframe
        for tf in self.timeframes:
            self.assertIn(tf, macd_values)
            
            # Vérifier qu'on a bien les trois composantes
            self.assertIn('macd', macd_values[tf])
            self.assertIn('signal', macd_values[tf])
            self.assertIn('histogram', macd_values[tf])
            
            # Vérifier que les longueurs sont correctes
            self.assertEqual(len(macd_values[tf]['macd']), len(self.data[tf]))
            self.assertEqual(len(macd_values[tf]['signal']), len(self.data[tf]))
            self.assertEqual(len(macd_values[tf]['histogram']), len(self.data[tf]))
            
    def test_get_alignment_score(self):
        """Teste le calcul du score d'alignement."""
        # Calculer le score d'alignement pour le RSI
        alignment_score = self.indicator.get_alignment_score(self.indicator.calculate_rsi)
        
        # Vérifier que le score est dans la plage [-1, 1]
        self.assertTrue(-1 <= alignment_score <= 1)
        
        # Calculer le score d'alignement pour le MACD
        alignment_score = self.indicator.get_alignment_score(self.indicator.calculate_macd)
        
        # Vérifier que le score est dans la plage [-1, 1]
        self.assertTrue(-1 <= alignment_score <= 1)


class TestDivergenceDetector(unittest.TestCase):
    """Tests pour la classe DivergenceDetector."""
    
    def setUp(self):
        """Initialise les données de test."""
        # Créer une série de prix et d'indicateur avec divergences connues
        np.random.seed(42)
        
        # Longueur des données
        n = 100
        
        # Créer une série de prix avec des extremums bien définis
        t = np.linspace(0, 4 * np.pi, n)
        
        # Prix: deux creux de plus en plus bas
        # Indices approx: 25 (creux 1) et 75 (creux 2)
        price = 100 + 10 * np.sin(t) + 0.5 * t + np.random.normal(0, 0.5, n)
        
        # Indicateur avec deux creux, mais le second plus haut que le premier
        # pour créer une divergence haussière régulière
        indicator = np.zeros(n)
        
        # Premier creux profond de l'indicateur (autour de l'indice 25)
        indicator[0:50] = 50 + 15 * np.sin(np.linspace(0, np.pi, 50))
        
        # Second creux moins profond (autour de l'indice 75)
        indicator[50:] = 50 + 8 * np.sin(np.linspace(0, np.pi, 50)) + 5  # +5 pour que le second creux soit plus haut
        
        # Accentuer les divergences
        # Le second creux de prix est plus bas que le premier
        price[70:80] = price[70:80] - 3
        
        # Le second creux de l'indicateur est plus haut que le premier
        indicator[70:80] = indicator[70:80] + 5
        
        # Créer le DataFrame et la Series
        self.price_data = pd.DataFrame({'close': price})
        self.indicator_data = pd.Series(indicator)
        
        # Créer le détecteur de divergences
        self.detector = DivergenceDetector(self.price_data, self.indicator_data, window_size=5)
        
    def test_find_extrema(self):
        """Teste la détection des extremums locaux."""
        # Trouver les extremums
        maxima, minima = self.detector._find_extrema(self.price_data['close'])
        
        # Vérifier qu'on a trouvé des extremums
        self.assertTrue(len(maxima) > 0)
        self.assertTrue(len(minima) > 0)
        
        # Vérifier que les extremums sont correctement identifiés
        for idx in maxima:
            # Un maximum local devrait être plus grand que ses voisins immédiats
            self.assertTrue(self.price_data['close'].iloc[idx] > self.price_data['close'].iloc[idx-1])
            self.assertTrue(self.price_data['close'].iloc[idx] > self.price_data['close'].iloc[idx+1])
            
        for idx in minima:
            # Un minimum local devrait être plus petit que ses voisins immédiats
            self.assertTrue(self.price_data['close'].iloc[idx] < self.price_data['close'].iloc[idx-1])
            self.assertTrue(self.price_data['close'].iloc[idx] < self.price_data['close'].iloc[idx+1])
            
    def test_detect_divergences(self):
        """Teste la détection des divergences."""
        # Détecter les divergences
        divergences = self.detector.detect_divergences()
        
        # Vérifier que l'output a le format attendu avec les bons types de divergences
        self.assertIsInstance(divergences, dict)
        self.assertIn(DivergenceType.BULLISH_REGULAR, divergences)
        self.assertIn(DivergenceType.BEARISH_REGULAR, divergences)
        self.assertIn(DivergenceType.BULLISH_HIDDEN, divergences)
        self.assertIn(DivergenceType.BEARISH_HIDDEN, divergences)
        
        # Vérifier que le format des résultats est correct
        for div_type, div_list in divergences.items():
            self.assertIsInstance(div_list, list)
            for div in div_list:
                # Chaque divergence est un tuple (indice1, indice2)
                self.assertEqual(len(div), 2)
                self.assertTrue(isinstance(div[0], int))
                self.assertTrue(isinstance(div[1], int))
                
    def test_get_last_divergence(self):
        """Teste la récupération de la dernière divergence."""
        # Créer manuellement une divergence pour forcer un retour
        # Ceci est un test technique et ne reflète pas le fonctionnement normal
        detector = self.detector
        detector.detect_divergences = lambda: {
            DivergenceType.BULLISH_REGULAR: [(75, 75)],
            DivergenceType.BEARISH_REGULAR: [],
            DivergenceType.BULLISH_HIDDEN: [],
            DivergenceType.BEARISH_HIDDEN: []
        }
        
        # Récupérer la dernière divergence
        last_div = detector.get_last_divergence()
        
        # Vérifier qu'on a bien trouvé une divergence
        self.assertIsNotNone(last_div)
        
        # Vérifier le format du résultat
        self.assertEqual(len(last_div), 2)
        self.assertTrue(isinstance(last_div[0], DivergenceType))
        self.assertTrue(isinstance(last_div[1], tuple))
        self.assertEqual(len(last_div[1]), 2)


if __name__ == '__main__':
    unittest.main() 