import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SRLevel:
    price: float
    timestamp: datetime
    type: str  # 'support' ou 'resistance'
    strength: float
    timeframe: str
    breaks: List[datetime] = None
    retests: List[datetime] = None
    is_valid: bool = True

class SupportResistanceDetector:
    def __init__(self, min_strength: float = 2, max_levels: int = 10):
        self.min_strength = min_strength
        self.max_levels = max_levels
        self.levels: List[SRLevel] = []
    
    def detect_pivots(self, df: pd.DataFrame, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Détecte les points pivots (hauts et bas)"""
        highs = df['high'].values
        lows = df['low'].values
        
        pivot_highs = np.zeros_like(highs)
        pivot_lows = np.zeros_like(lows)
        
        for i in range(window, len(df) - window):
            # Détection des sommets
            if all(highs[i] > highs[i-window:i]) and all(highs[i] > highs[i+1:i+window+1]):
                pivot_highs[i] = highs[i]
            
            # Détection des creux
            if all(lows[i] < lows[i-window:i]) and all(lows[i] < lows[i+1:i+window+1]):
                pivot_lows[i] = lows[i]
        
        return pivot_highs, pivot_lows
    
    def calculate_strength(self, price: float, df: pd.DataFrame, tolerance: float = 0.002) -> float:
        """Calcule la force d'un niveau en fonction du nombre de touches"""
        touches = 0
        price_range = price * tolerance
        
        for _, row in df.iterrows():
            if abs(row['high'] - price) <= price_range or abs(row['low'] - price) <= price_range:
                touches += 1
        
        return touches
    
    def detect_levels(self, df: pd.DataFrame, timeframe: str) -> List[SRLevel]:
        """Détecte les niveaux de support et résistance"""
        pivot_highs, pivot_lows = self.detect_pivots(df)
        
        # Création des niveaux
        new_levels = []
        
        # Traitement des résistances (pivots hauts)
        for i, price in enumerate(pivot_highs):
            if price == 0:  # Ignorer les non-pivots
                continue
                
            strength = self.calculate_strength(price, df)
            if strength >= self.min_strength:
                new_levels.append(SRLevel(
                    price=price,
                    timestamp=df.index[i],
                    type='resistance',
                    strength=strength,
                    timeframe=timeframe,
                    breaks=[],
                    retests=[]
                ))
        
        # Traitement des supports (pivots bas)
        for i, price in enumerate(pivot_lows):
            if price == 0:  # Ignorer les non-pivots
                continue
                
            strength = self.calculate_strength(price, df)
            if strength >= self.min_strength:
                new_levels.append(SRLevel(
                    price=price,
                    timestamp=df.index[i],
                    type='support',
                    strength=strength,
                    timeframe=timeframe,
                    breaks=[],
                    retests=[]
                ))
        
        # Trier par force et limiter le nombre de niveaux
        new_levels.sort(key=lambda x: x.strength, reverse=True)
        return new_levels[:self.max_levels]
    
    def detect_breaks_retests(self, df: pd.DataFrame, level: SRLevel, tolerance: float = 0.002):
        """Détecte les cassures et retests d'un niveau"""
        price_range = level.price * tolerance
        was_broken = False
        
        for timestamp, row in df.iterrows():
            if timestamp <= level.timestamp:
                continue
                
            if level.type == 'resistance':
                # Détection cassure résistance
                if row['close'] > level.price + price_range and not was_broken:
                    level.breaks.append(timestamp)
                    was_broken = True
                # Détection retest résistance
                elif was_broken and abs(row['high'] - level.price) <= price_range:
                    level.retests.append(timestamp)
            
            else:  # Support
                # Détection cassure support
                if row['close'] < level.price - price_range and not was_broken:
                    level.breaks.append(timestamp)
                    was_broken = True
                # Détection retest support
                elif was_broken and abs(row['low'] - level.price) <= price_range:
                    level.retests.append(timestamp)
        
        # Mise à jour de la validité
        level.is_valid = not was_broken 