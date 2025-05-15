import logging
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from ai_trading.data.technical_indicators import TechnicalIndicators

# Configuration du logging
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types de signaux possibles."""
    BUY = "BUY"               # Signal d'achat fort
    WEAK_BUY = "WEAK_BUY"     # Signal d'achat faible
    SELL = "SELL"             # Signal de vente fort
    WEAK_SELL = "WEAK_SELL"   # Signal de vente faible
    NEUTRAL = "NEUTRAL"       # Pas de signal clair
    EXIT = "EXIT"             # Signal de sortie de position

@dataclass
class Signal:
    """Classe représentant un signal de trading."""
    type: SignalType
    symbol: str
    timestamp: pd.Timestamp
    price: float
    confidence: float  # Score de confiance entre 0 et 1
    source: str  # Indicateur ou stratégie source
    timeframe: str
    metadata: Dict = None  # Métadonnées supplémentaires
    
    def __post_init__(self):
        """Vérification des données après initialisation."""
        if self.metadata is None:
            self.metadata = {}
        
        # S'assurer que la confiance est entre 0 et 1
        self.confidence = max(0, min(1, self.confidence))
    
    def __str__(self):
        """Représentation en chaîne du signal."""
        return (f"{self.timestamp} | {self.symbol} | {self.type.value} | "
                f"Prix: {self.price:.2f} | Conf: {self.confidence:.2f} | "
                f"Source: {self.source} | TF: {self.timeframe}")

class SignalGenerator:
    """
    Générateur de signaux de trading basé sur divers indicateurs techniques.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le générateur de signaux.
        
        Args:
            config: Configuration des paramètres pour les signaux
        """
        self.config = config or {}
        self.indicators = TechnicalIndicators()
        
        # Paramètres par défaut
        self.default_params = {
            "rsi": {
                "period": 14,
                "overbought": 70,
                "oversold": 30
            },
            "macd": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            },
            "bollinger": {
                "period": 20,
                "std_dev": 2
            },
            "stochastic": {
                "k_period": 14,
                "d_period": 3,
                "overbought": 80,
                "oversold": 20
            },
            "adx": {
                "period": 14,
                "threshold": 25
            },
            "volume": {
                "period": 20
            }
        }
        
        # Fusionner avec la configuration fournie
        for key, default_value in self.default_params.items():
            if key in self.config:
                for param, value in default_value.items():
                    if param not in self.config[key]:
                        self.config[key][param] = value
            else:
                self.config[key] = default_value
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """
        Génère des signaux de trading basés sur les données fournies.
        
        Args:
            data: DataFrame contenant les données OHLCV
            symbol: Symbole de l'actif
            timeframe: Timeframe des données (ex: '1h', '1d')
            
        Returns:
            Liste des signaux générés
        """
        if data.empty:
            logger.warning(f"Aucune donnée fournie pour {symbol} en {timeframe}")
            return []
        
        # Vérification des colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Colonnes manquantes dans les données. Requises: {required_columns}")
            return []
        
        self.indicators.set_data(data)
        signals = []
        
        # Générer les signaux de chaque type d'indicateur
        rsi_signals = self._generate_rsi_signals(data, symbol, timeframe)
        macd_signals = self._generate_macd_signals(data, symbol, timeframe)
        bollinger_signals = self._generate_bollinger_signals(data, symbol, timeframe)
        stoch_signals = self._generate_stochastic_signals(data, symbol, timeframe)
        adx_signals = self._generate_adx_signals(data, symbol, timeframe)
        volume_signals = self._generate_volume_signals(data, symbol, timeframe)
        
        # Combiner tous les signaux
        signals.extend(rsi_signals)
        signals.extend(macd_signals)
        signals.extend(bollinger_signals)
        signals.extend(stoch_signals)
        signals.extend(adx_signals)
        signals.extend(volume_signals)
        
        # Filtrer les signaux pour ne garder que les plus récents
        latest_signals = self._filter_recent_signals(signals)
        
        # Trier par timestamp et confiance
        latest_signals.sort(key=lambda x: (x.timestamp, -x.confidence))
        
        return latest_signals
    
    def _filter_recent_signals(self, signals: List[Signal], max_signals: int = 5) -> List[Signal]:
        """
        Filtre les signaux pour ne garder que les plus récents et les plus pertinents.
        
        Args:
            signals: Liste de tous les signaux
            max_signals: Nombre maximum de signaux à retourner
            
        Returns:
            Liste filtrée des signaux
        """
        if not signals:
            return []
        
        # Grouper par type de signal et source
        signals_by_type = {}
        for signal in signals:
            key = (signal.type, signal.source)
            if key not in signals_by_type or signal.timestamp > signals_by_type[key].timestamp:
                signals_by_type[key] = signal
        
        # Convertir le dictionnaire en liste et trier par confiance
        filtered_signals = list(signals_by_type.values())
        filtered_signals.sort(key=lambda x: -x.confidence)
        
        return filtered_signals[:max_signals]
    
    def _generate_rsi_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Génère des signaux basés sur l'indicateur RSI."""
        params = self.config['rsi']
        rsi = self.indicators.calculate_rsi(period=params['period'])
        
        signals = []
        last_index = data.index[-1]
        last_price = data['close'].iloc[-1]
        last_rsi = rsi.iloc[-1]
        
        # Éviter les NaN
        if pd.isna(last_rsi):
            return signals
        
        # Signal de surachat
        if last_rsi > params['overbought']:
            confidence = min(1.0, (last_rsi - params['overbought']) / 30)
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=last_index,
                price=last_price,
                confidence=confidence,
                source="RSI",
                timeframe=timeframe,
                metadata={"rsi_value": last_rsi}
            ))
        
        # Signal de survente
        elif last_rsi < params['oversold']:
            confidence = min(1.0, (params['oversold'] - last_rsi) / 30)
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=last_index,
                price=last_price,
                confidence=confidence,
                source="RSI",
                timeframe=timeframe,
                metadata={"rsi_value": last_rsi}
            ))
        
        return signals
    
    def _generate_macd_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Génère des signaux basés sur l'indicateur MACD."""
        params = self.config['macd']
        macd_line, signal_line, histogram = self.indicators.calculate_macd(
            fast_period=params['fast_period'],
            slow_period=params['slow_period'],
            signal_period=params['signal_period']
        )
        
        signals = []
        if macd_line is None or signal_line is None or histogram is None:
            return signals
        
        last_index = data.index[-1]
        last_price = data['close'].iloc[-1]
        
        # Vérifier si MACD ligne croise au-dessus de la signal ligne
        if (histogram.iloc[-2] <= 0 and histogram.iloc[-1] > 0):
            strength = abs(histogram.iloc[-1]) / max(0.01, abs(macd_line.iloc[-1]))
            confidence = min(0.8, strength)
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=last_index,
                price=last_price,
                confidence=confidence,
                source="MACD",
                timeframe=timeframe,
                metadata={
                    "macd": macd_line.iloc[-1],
                    "signal": signal_line.iloc[-1],
                    "histogram": histogram.iloc[-1]
                }
            ))
        
        # Vérifier si MACD ligne croise en-dessous de la signal ligne
        elif (histogram.iloc[-2] >= 0 and histogram.iloc[-1] < 0):
            strength = abs(histogram.iloc[-1]) / max(0.01, abs(macd_line.iloc[-1]))
            confidence = min(0.8, strength)
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=last_index,
                price=last_price,
                confidence=confidence,
                source="MACD",
                timeframe=timeframe,
                metadata={
                    "macd": macd_line.iloc[-1],
                    "signal": signal_line.iloc[-1],
                    "histogram": histogram.iloc[-1]
                }
            ))
        
        return signals
    
    def _generate_bollinger_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Génère des signaux basés sur les bandes de Bollinger."""
        params = self.config['bollinger']
        upper, middle, lower = self.indicators.calculate_bollinger_bands(
            period=params['period'],
            std_dev=params['std_dev']
        )
        
        signals = []
        if upper is None or middle is None or lower is None:
            return signals
        
        last_index = data.index[-1]
        last_close = data['close'].iloc[-1]
        
        # Prix touche ou dépasse la bande inférieure (potentiel achat)
        if last_close <= lower.iloc[-1]:
            # Calculer la distance relative par rapport à la largeur des bandes
            band_width = upper.iloc[-1] - lower.iloc[-1]
            distance = (lower.iloc[-1] - last_close) / band_width if band_width > 0 else 0
            confidence = min(0.7, 0.3 + distance)
            
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=last_index,
                price=last_close,
                confidence=confidence,
                source="Bollinger",
                timeframe=timeframe,
                metadata={
                    "upper": upper.iloc[-1],
                    "middle": middle.iloc[-1],
                    "lower": lower.iloc[-1],
                    "band_width": band_width
                }
            ))
        
        # Prix touche ou dépasse la bande supérieure (potentiel vente)
        elif last_close >= upper.iloc[-1]:
            # Calculer la distance relative par rapport à la largeur des bandes
            band_width = upper.iloc[-1] - lower.iloc[-1]
            distance = (last_close - upper.iloc[-1]) / band_width if band_width > 0 else 0
            confidence = min(0.7, 0.3 + distance)
            
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=last_index,
                price=last_close,
                confidence=confidence,
                source="Bollinger",
                timeframe=timeframe,
                metadata={
                    "upper": upper.iloc[-1],
                    "middle": middle.iloc[-1],
                    "lower": lower.iloc[-1],
                    "band_width": band_width
                }
            ))
        
        return signals
    
    def _generate_stochastic_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Génère des signaux basés sur l'oscillateur stochastique."""
        params = self.config['stochastic']
        k, d = self.indicators.calculate_stochastic(
            k_period=params['k_period'],
            d_period=params['d_period']
        )
        
        signals = []
        if k is None or d is None:
            return signals
        
        last_index = data.index[-1]
        last_price = data['close'].iloc[-1]
        
        # Vérifier les conditions de surachat/survente et de croisement
        
        # Condition de survente et croisement à la hausse
        if (k.iloc[-1] < params['oversold'] and d.iloc[-1] < params['oversold'] and
            k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] > d.iloc[-1]):
            
            # Force du signal basée sur la distance par rapport au niveau de survente
            distance = abs(k.iloc[-1] - params['oversold']) / 20  # normaliser
            confidence = min(0.85, 0.5 + distance)
            
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=last_index,
                price=last_price,
                confidence=confidence,
                source="Stochastic",
                timeframe=timeframe,
                metadata={
                    "k": k.iloc[-1],
                    "d": d.iloc[-1]
                }
            ))
        
        # Condition de surachat et croisement à la baisse
        elif (k.iloc[-1] > params['overbought'] and d.iloc[-1] > params['overbought'] and
              k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] < d.iloc[-1]):
            
            # Force du signal basée sur la distance par rapport au niveau de surachat
            distance = abs(k.iloc[-1] - params['overbought']) / 20  # normaliser
            confidence = min(0.85, 0.5 + distance)
            
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=last_index,
                price=last_price,
                confidence=confidence,
                source="Stochastic",
                timeframe=timeframe,
                metadata={
                    "k": k.iloc[-1],
                    "d": d.iloc[-1]
                }
            ))
        
        return signals
    
    def _generate_adx_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Génère des signaux basés sur l'indicateur ADX (Directional Movement Index)."""
        params = self.config['adx']
        adx, plus_di, minus_di = self.indicators.calculate_adx(period=params['period'])
        
        signals = []
        if adx is None or plus_di is None or minus_di is None:
            return signals
        
        last_index = data.index[-1]
        last_price = data['close'].iloc[-1]
        
        # Vérifier si ADX est supérieur au seuil (tendance forte)
        if adx.iloc[-1] > params['threshold']:
            # Tendance haussière (+DI > -DI)
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                # Force du signal basée sur l'écart entre +DI et -DI et la valeur de l'ADX
                di_diff = (plus_di.iloc[-1] - minus_di.iloc[-1]) / 50  # normaliser
                adx_strength = (adx.iloc[-1] - params['threshold']) / 25  # normaliser
                confidence = min(0.9, 0.4 + 0.3 * di_diff + 0.3 * adx_strength)
                
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=last_index,
                    price=last_price,
                    confidence=confidence,
                    source="ADX",
                    timeframe=timeframe,
                    metadata={
                        "adx": adx.iloc[-1],
                        "plus_di": plus_di.iloc[-1],
                        "minus_di": minus_di.iloc[-1]
                    }
                ))
            
            # Tendance baissière (-DI > +DI)
            elif minus_di.iloc[-1] > plus_di.iloc[-1]:
                # Force du signal basée sur l'écart entre -DI et +DI et la valeur de l'ADX
                di_diff = (minus_di.iloc[-1] - plus_di.iloc[-1]) / 50  # normaliser
                adx_strength = (adx.iloc[-1] - params['threshold']) / 25  # normaliser
                confidence = min(0.9, 0.4 + 0.3 * di_diff + 0.3 * adx_strength)
                
                signals.append(Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=last_index,
                    price=last_price,
                    confidence=confidence,
                    source="ADX",
                    timeframe=timeframe,
                    metadata={
                        "adx": adx.iloc[-1],
                        "plus_di": plus_di.iloc[-1],
                        "minus_di": minus_di.iloc[-1]
                    }
                ))
        
        return signals
    
    def _generate_volume_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Génère des signaux basés sur l'analyse de volume."""
        params = self.config['volume']
        volume_ma = self.indicators.calculate_volume_average(period=params['period'])
        
        signals = []
        if volume_ma is None:
            return signals
        
        last_index = data.index[-1]
        last_price = data['close'].iloc[-1]
        last_volume = data['volume'].iloc[-1]
        
        # Calculer le ratio volume actuel / volume moyen
        volume_ratio = last_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
        
        # Vérifier un pic de volume significatif (2x la moyenne)
        if volume_ratio > 2.0:
            # Tendance du prix
            price_change = (data['close'].iloc[-1] - data['open'].iloc[-1]) / data['open'].iloc[-1]
            
            # Volume élevé + hausse de prix = signal d'achat
            if price_change > 0.01:  # hausse de prix de plus de 1%
                confidence = min(0.75, 0.5 + (volume_ratio - 2) / 4 + price_change)
                
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=last_index,
                    price=last_price,
                    confidence=confidence,
                    source="Volume",
                    timeframe=timeframe,
                    metadata={
                        "volume": last_volume,
                        "volume_ma": volume_ma.iloc[-1],
                        "volume_ratio": volume_ratio,
                        "price_change": price_change
                    }
                ))
            
            # Volume élevé + baisse de prix = signal de vente
            elif price_change < -0.01:  # baisse de prix de plus de 1%
                confidence = min(0.75, 0.5 + (volume_ratio - 2) / 4 + abs(price_change))
                
                signals.append(Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=last_index,
                    price=last_price,
                    confidence=confidence,
                    source="Volume",
                    timeframe=timeframe,
                    metadata={
                        "volume": last_volume,
                        "volume_ma": volume_ma.iloc[-1],
                        "volume_ratio": volume_ratio,
                        "price_change": price_change
                    }
                ))
        
        return signals
    
    def score_signals(self, signals: List[Signal]) -> float:
        """
        Calcule un score global basé sur tous les signaux.
        
        Args:
            signals: Liste des signaux générés
            
        Returns:
            Score entre -1 (très baissier) et 1 (très haussier)
        """
        if not signals:
            return 0.0
        
        buy_score = 0.0
        sell_score = 0.0
        
        for signal in signals:
            if signal.type in [SignalType.BUY, SignalType.WEAK_BUY]:
                buy_score += signal.confidence * (1.0 if signal.type == SignalType.BUY else 0.5)
            elif signal.type in [SignalType.SELL, SignalType.WEAK_SELL]:
                sell_score += signal.confidence * (1.0 if signal.type == SignalType.SELL else 0.5)
        
        # Normaliser les scores
        total_signals = len(signals)
        buy_ratio = buy_score / total_signals if total_signals > 0 else 0
        sell_ratio = sell_score / total_signals if total_signals > 0 else 0
        
        # Score final entre -1 et 1
        return buy_ratio - sell_ratio 