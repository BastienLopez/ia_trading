"""
Module d'indicateurs techniques complexes pour l'analyse avancée des marchés crypto.
"""
import numpy as np
import pandas as pd
import pywt
from scipy import signal
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

class WaveletAnalyzer:
    """
    Classe d'analyse par ondelettes pour décomposer les séries temporelles en
    différentes fréquences et identifier les tendances et cycles.
    """
    
    def __init__(self, wavelet: str = 'db8', level: int = 5):
        """
        Initialise l'analyseur d'ondelettes.
        
        Args:
            wavelet: Type d'ondelette à utiliser (default: 'db8')
            level: Niveau de décomposition (default: 5)
        """
        self.wavelet = wavelet
        self.level = level
        
    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Décompose une série temporelle en composantes d'ondelettes.
        
        Args:
            data: Série temporelle à décomposer
            
        Returns:
            Dictionnaire contenant les coefficients d'approximation et de détail
        """
        # Assurez-vous que les données sont de longueur adéquate pour la décomposition
        data = self._pad_data(data)
        
        # Décomposition en ondelettes
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # Créer un dictionnaire des résultats
        result = {
            'approximation': coeffs[0]
        }
        
        for i, detail in enumerate(coeffs[1:], 1):
            result[f'detail_{i}'] = detail
            
        return result
    
    def reconstruct(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruit une série temporelle à partir des coefficients d'ondelettes.
        
        Args:
            coeffs: Dictionnaire des coefficients d'ondelettes
            
        Returns:
            Série temporelle reconstruite
        """
        # Convertir le dictionnaire en liste de coefficients
        coeff_list = [coeffs['approximation']]
        for i in range(1, self.level + 1):
            coeff_list.append(coeffs[f'detail_{i}'])
            
        # Reconstruction
        return pywt.waverec(coeff_list, self.wavelet)
    
    def denoise(self, data: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Applique un débruitage par seuillage des coefficients d'ondelettes.
        
        Args:
            data: Série temporelle à débruiter
            threshold: Seuil de coefficient (0-1) (default: 0.3)
            
        Returns:
            Série temporelle débruitée
        """
        # Décomposition
        coeffs = self.decompose(data)
        
        # Appliquer le seuillage aux détails (pas à l'approximation)
        for i in range(1, self.level + 1):
            key = f'detail_{i}'
            detail = coeffs[key]
            # Calculer le seuil adaptatif basé sur l'amplitude des coefficients
            adaptive_threshold = threshold * np.max(np.abs(detail))
            # Appliquer le seuillage doux
            coeffs[key] = pywt.threshold(detail, adaptive_threshold, mode='soft')
            
        # Reconstruction
        return self.reconstruct(coeffs)
    
    def _pad_data(self, data: np.ndarray) -> np.ndarray:
        """
        Rembourre les données pour qu'elles aient une longueur appropriée 
        pour la décomposition en ondelettes.
        
        Args:
            data: Données d'entrée
            
        Returns:
            Données rembourrées
        """
        # Calculer la prochaine puissance de 2
        next_pow_2 = 2 ** np.ceil(np.log2(len(data)))
        # Rembourrage (mirroring)
        padding = int(next_pow_2 - len(data))
        if padding > 0:
            return np.pad(data, (0, padding), mode='reflect')
        return data


class SpectralAnalyzer:
    """
    Classe d'analyse spectrale pour identifier les cycles et fréquences dominantes
    dans les séries temporelles de prix.
    """
    
    def __init__(self, sampling_freq: float = 1.0):
        """
        Initialise l'analyseur spectral.
        
        Args:
            sampling_freq: Fréquence d'échantillonnage des données (default: 1.0)
        """
        self.sampling_freq = sampling_freq
        
    def _blackmanharris(self, M):
        """
        Implémentation de la fenêtre de Blackman-Harris.
        
        Args:
            M: Longueur de la fenêtre
            
        Returns:
            Fenêtre de Blackman-Harris
        """
        n = np.arange(M)
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        return a0 - a1 * np.cos(2 * np.pi * n / (M - 1)) + a2 * np.cos(4 * np.pi * n / (M - 1)) - a3 * np.cos(6 * np.pi * n / (M - 1))
        
    def periodogram(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le périodogramme de la série temporelle.
        
        Args:
            data: Série temporelle à analyser
            
        Returns:
            Tuple contenant (fréquences, densité spectrale de puissance)
        """
        # Supprimer la tendance
        detrended = signal.detrend(data)
        
        # Appliquer une fenêtre de Blackman-Harris pour réduire les fuites spectrales
        window = self._blackmanharris(len(detrended))
        windowed = detrended * window
        
        # Calculer le périodogramme
        freqs, psd = signal.periodogram(windowed, fs=self.sampling_freq)
        
        return freqs, psd
    
    def find_dominant_cycles(self, data: np.ndarray, n_cycles: int = 3) -> List[float]:
        """
        Identifie les cycles dominants dans la série temporelle.
        
        Args:
            data: Série temporelle à analyser
            n_cycles: Nombre de cycles dominants à retourner (default: 3)
            
        Returns:
            Liste des périodes (en unités d'échantillonnage) des cycles dominants
        """
        freqs, psd = self.periodogram(data)
        
        # Éviter la fréquence zéro (composante DC)
        mask = freqs > 0
        freqs, psd = freqs[mask], psd[mask]
        
        # Trouver les pics dans le spectre
        peak_indices = signal.find_peaks(psd)[0]
        
        # Si aucun pic n'est trouvé, retourner une liste vide
        if len(peak_indices) == 0:
            return []
        
        # Trier les pics par puissance
        sorted_peaks = sorted(zip(peak_indices, psd[peak_indices]), key=lambda x: x[1], reverse=True)
        
        # Convertir les fréquences en périodes
        cycles = []
        for idx, _ in sorted_peaks[:n_cycles]:
            if freqs[idx] > 0:  # Éviter division par zéro
                # Période = échantillons par cycle = fréquence d'échantillonnage / fréquence du cycle
                cycles.append(self.sampling_freq / freqs[idx])
            
        return cycles
    
    def filter_by_frequency(self, data: np.ndarray, 
                           low_cutoff: Optional[float] = None, 
                           high_cutoff: Optional[float] = None) -> np.ndarray:
        """
        Applique un filtre passe-bande à la série temporelle.
        
        Args:
            data: Série temporelle à filtrer
            low_cutoff: Fréquence de coupure basse (None pour passe-bas)
            high_cutoff: Fréquence de coupure haute (None pour passe-haut)
            
        Returns:
            Série temporelle filtrée
        """
        # Déterminer le type de filtre
        if low_cutoff is None and high_cutoff is not None:
            # Filtre passe-bas
            b, a = signal.butter(4, high_cutoff / (self.sampling_freq/2), btype='low')
        elif low_cutoff is not None and high_cutoff is None:
            # Filtre passe-haut
            b, a = signal.butter(4, low_cutoff / (self.sampling_freq/2), btype='high')
        elif low_cutoff is not None and high_cutoff is not None:
            # Filtre passe-bande
            b, a = signal.butter(4, [low_cutoff / (self.sampling_freq/2), 
                                    high_cutoff / (self.sampling_freq/2)], btype='band')
        else:
            # Aucun filtre, retourner les données originales
            return data
        
        # Appliquer le filtre
        filtered = signal.filtfilt(b, a, data)
        
        return filtered


class TimeFrame(Enum):
    """Enum représentant les différentes échelles de temps d'analyse."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class MultiTimeframeIndicator:
    """
    Classe permettant l'analyse d'indicateurs sur plusieurs timeframes
    pour obtenir une vision consolidée du marché.
    """
    
    def __init__(self, timeframes: List[TimeFrame]):
        """
        Initialise l'indicateur multi-timeframes.
        
        Args:
            timeframes: Liste des timeframes à analyser
        """
        self.timeframes = timeframes
        self.data = {}
        
    def add_data(self, timeframe: TimeFrame, data: pd.DataFrame):
        """
        Ajoute des données pour un timeframe spécifique.
        
        Args:
            timeframe: Timeframe des données
            data: DataFrame contenant les données OHLCV
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Le timeframe {timeframe} n'est pas dans la liste des timeframes configurés")
        
        self.data[timeframe] = data
        
    def calculate_rsi(self, window: int = 14) -> Dict[TimeFrame, pd.Series]:
        """
        Calcule le RSI pour tous les timeframes configurés.
        
        Args:
            window: Période du RSI (default: 14)
            
        Returns:
            Dictionnaire de Series RSI indexé par timeframe
        """
        results = {}
        
        for tf in self.timeframes:
            if tf not in self.data:
                continue
                
            df = self.data[tf]
            if 'close' not in df.columns:
                continue
                
            # Calculer les variations
            delta = df['close'].diff()
            
            # Séparer les variations positives et négatives
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculer la moyenne des gains et des pertes
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            # Calculer le RS et le RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            results[tf] = rsi
            
        return results
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[TimeFrame, Dict[str, pd.Series]]:
        """
        Calcule le MACD pour tous les timeframes configurés.
        
        Args:
            fast: Période de la moyenne mobile rapide (default: 12)
            slow: Période de la moyenne mobile lente (default: 26)
            signal: Période de la ligne de signal (default: 9)
            
        Returns:
            Dictionnaire de dictionnaires contenant 'macd', 'signal' et 'histogram' indexé par timeframe
        """
        results = {}
        
        for tf in self.timeframes:
            if tf not in self.data:
                continue
                
            df = self.data[tf]
            if 'close' not in df.columns:
                continue
            
            # Calculer les moyennes mobiles exponentielles
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            
            # Calculer la ligne MACD
            macd_line = ema_fast - ema_slow
            
            # Calculer la ligne de signal
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculer l'histogramme
            histogram = macd_line - signal_line
            
            results[tf] = {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        return results
    
    def get_alignment_score(self, indicator_func, **kwargs) -> float:
        """
        Calcule un score d'alignement entre les différents timeframes pour un indicateur donné.
        
        Args:
            indicator_func: Fonction de calcul d'indicateur à utiliser
            **kwargs: Arguments à passer à la fonction d'indicateur
            
        Returns:
            Score d'alignement entre -1 (totalement opposé) et 1 (parfaitement aligné)
        """
        # Calculer l'indicateur pour tous les timeframes
        indicator_values = indicator_func(**kwargs)
        
        # Obtenir les dernières valeurs
        latest_values = {}
        signals = {}
        
        for tf, values in indicator_values.items():
            if isinstance(values, dict):  # Pour MACD ou indicateurs composites
                # Prendre l'histogramme pour le MACD
                latest_values[tf] = values['histogram'].iloc[-1]
                signals[tf] = 1 if latest_values[tf] > 0 else -1
            else:  # Pour RSI ou indicateurs simples
                latest_values[tf] = values.iloc[-1]
                # Pour le RSI, considérer >50 comme signal haussier, <50 comme baissier
                signals[tf] = 1 if latest_values[tf] > 50 else -1
        
        # Calculer le score d'alignement
        if not signals:
            return 0.0
            
        bullish_count = sum(1 for signal in signals.values() if signal > 0)
        bearish_count = sum(1 for signal in signals.values() if signal < 0)
        
        # Normaliser entre -1 et 1
        total_count = len(signals)
        return (bullish_count - bearish_count) / total_count


class DivergenceType(Enum):
    """Types de divergences possibles entre prix et indicateurs."""
    BULLISH_REGULAR = "divergence_haussière_régulière"
    BULLISH_HIDDEN = "divergence_haussière_cachée"
    BEARISH_REGULAR = "divergence_baissière_régulière"
    BEARISH_HIDDEN = "divergence_baissière_cachée"


class DivergenceDetector:
    """
    Classe pour détecter les divergences entre prix et indicateurs techniques,
    signaux importants pour les retournements de tendance.
    """
    
    def __init__(self, price_data: pd.DataFrame, indicator_data: pd.Series, 
                window_size: int = 10):
        """
        Initialise le détecteur de divergences.
        
        Args:
            price_data: DataFrame contenant au moins une colonne 'close'
            indicator_data: Series contenant les valeurs de l'indicateur
            window_size: Taille de la fenêtre pour chercher des extremums (default: 10)
        """
        if 'close' not in price_data.columns:
            raise ValueError("Le DataFrame price_data doit contenir une colonne 'close'")
            
        self.price = price_data['close']
        self.indicator = indicator_data
        self.window_size = window_size
        
    def _find_extrema(self, data: pd.Series) -> Tuple[List[int], List[int]]:
        """
        Trouve les indices des extremums locaux dans une série.
        
        Args:
            data: Série temporelle à analyser
            
        Returns:
            Tuple contenant (indices des maxima locaux, indices des minima locaux)
        """
        window = self.window_size
        data_len = len(data)
        
        maxima = []
        minima = []
        
        for i in range(window, data_len - window):
            # Créer une fenêtre centrée sur l'indice actuel
            left_window = data.iloc[i - window:i]
            right_window = data.iloc[i + 1:i + window + 1]
            current_value = data.iloc[i]
            
            # Vérifier si c'est un maximum local
            if current_value > left_window.max() and current_value > right_window.max():
                maxima.append(i)
                
            # Vérifier si c'est un minimum local
            if current_value < left_window.min() and current_value < right_window.min():
                minima.append(i)
                
        return maxima, minima
    
    def detect_divergences(self) -> Dict[DivergenceType, List[Tuple[int, int]]]:
        """
        Détecte toutes les divergences entre prix et indicateur.
        
        Returns:
            Dictionnaire des divergences détectées, avec des listes de tuples (indice1, indice2)
            représentant les paires de points formant chaque divergence
        """
        # Trouver les extremums
        price_maxima, price_minima = self._find_extrema(self.price)
        ind_maxima, ind_minima = self._find_extrema(self.indicator)
        
        results = {
            DivergenceType.BULLISH_REGULAR: [],
            DivergenceType.BULLISH_HIDDEN: [],
            DivergenceType.BEARISH_REGULAR: [],
            DivergenceType.BEARISH_HIDDEN: []
        }
        
        # Chercher les divergences haussières régulières
        # (prix fait des creux plus bas, indicateur fait des creux plus hauts)
        for i in range(len(price_minima) - 1):
            idx1 = price_minima[i]
            idx2 = price_minima[i + 1]
            
            if idx2 - idx1 < self.window_size:
                continue
            
            # Trouver les minima de l'indicateur proches des minima de prix
            ind_min1_candidates = [idx for idx in ind_minima if abs(idx - idx1) <= self.window_size]
            ind_min2_candidates = [idx for idx in ind_minima if abs(idx - idx2) <= self.window_size]
            
            if not ind_min1_candidates or not ind_min2_candidates:
                continue
                
            # Prendre les minima les plus proches
            ind_idx1 = min(ind_min1_candidates, key=lambda x: abs(x - idx1))
            ind_idx2 = min(ind_min2_candidates, key=lambda x: abs(x - idx2))
            
            # Vérifier la divergence haussière régulière
            if (self.price.iloc[idx2] < self.price.iloc[idx1] and 
                self.indicator.iloc[ind_idx2] > self.indicator.iloc[ind_idx1]):
                results[DivergenceType.BULLISH_REGULAR].append((idx2, ind_idx2))
                
            # Vérifier la divergence haussière cachée
            if (self.price.iloc[idx2] > self.price.iloc[idx1] and 
                self.indicator.iloc[ind_idx2] < self.indicator.iloc[ind_idx1]):
                results[DivergenceType.BULLISH_HIDDEN].append((idx2, ind_idx2))
        
        # Chercher les divergences baissières
        for i in range(len(price_maxima) - 1):
            idx1 = price_maxima[i]
            idx2 = price_maxima[i + 1]
            
            if idx2 - idx1 < self.window_size:
                continue
            
            # Trouver les maxima de l'indicateur proches des maxima de prix
            ind_max1_candidates = [idx for idx in ind_maxima if abs(idx - idx1) <= self.window_size]
            ind_max2_candidates = [idx for idx in ind_maxima if abs(idx - idx2) <= self.window_size]
            
            if not ind_max1_candidates or not ind_max2_candidates:
                continue
                
            # Prendre les maxima les plus proches
            ind_idx1 = min(ind_max1_candidates, key=lambda x: abs(x - idx1))
            ind_idx2 = min(ind_max2_candidates, key=lambda x: abs(x - idx2))
            
            # Vérifier la divergence baissière régulière
            if (self.price.iloc[idx2] > self.price.iloc[idx1] and 
                self.indicator.iloc[ind_idx2] < self.indicator.iloc[ind_idx1]):
                results[DivergenceType.BEARISH_REGULAR].append((idx2, ind_idx2))
                
            # Vérifier la divergence baissière cachée
            if (self.price.iloc[idx2] < self.price.iloc[idx1] and 
                self.indicator.iloc[ind_idx2] > self.indicator.iloc[ind_idx1]):
                results[DivergenceType.BEARISH_HIDDEN].append((idx2, ind_idx2))
        
        return results
    
    def get_last_divergence(self) -> Optional[Tuple[DivergenceType, Tuple[int, int]]]:
        """
        Retourne la dernière divergence détectée.
        
        Returns:
            Tuple contenant (type de divergence, (indice1, indice2)) ou None si aucune divergence
        """
        divergences = self.detect_divergences()
        
        # Trouver la divergence la plus récente
        latest_div = None
        latest_idx = -1
        
        for div_type, div_list in divergences.items():
            if not div_list:
                continue
                
            # Prendre la dernière divergence de chaque type
            idx, _ = div_list[-1]
            
            if idx > latest_idx:
                latest_idx = idx
                latest_div = (div_type, div_list[-1])
                
        return latest_div 