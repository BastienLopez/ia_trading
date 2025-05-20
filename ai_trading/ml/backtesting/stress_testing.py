"""
Module de stress testing pour évaluer la robustesse des stratégies de trading
dans des conditions de marché extrêmes ou à fort stress.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timedelta

# Configuration du logging
logger = logging.getLogger(__name__)

# Fonctions utilitaires
def determine_data_frequency(data: pd.DataFrame) -> str:
    """
    Détermine la fréquence temporelle des données.
    
    Args:
        data: DataFrame avec un index temporel
        
    Returns:
        Chaîne indiquant la fréquence (1min, 5min, 1h, 1d, etc.)
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return "unknown"
        
    if len(data.index) < 2:
        return "unknown"
        
    diffs = []
    for i in range(1, len(data.index)):
        diff = data.index[i] - data.index[i-1]
        diffs.append(diff)
    
    diffs = pd.Series(diffs)
    
    if len(diffs.unique()) == 1:
        time_diff = diffs.iloc[0]
    else:
        time_diff = diffs.value_counts().index[0]
    
    minutes = time_diff.total_seconds() / 60
    
    if minutes < 1:
        return "1s"
    elif minutes == 1:
        return "1min"
    elif minutes == 5:
        return "5min"
    elif minutes == 15:
        return "15min"
    elif minutes == 30:
        return "30min"
    elif minutes == 60:
        return "1h"
    elif minutes == 240:
        return "4h"
    elif minutes >= 1440:
        return "1d"
    else:
        return f"{int(minutes)}min"

class ScenarioType(Enum):
    """Types de scénarios de stress testing."""
    CRASH = "CRASH"                     # Chute brutale des prix
    VOLATILITY_SPIKE = "VOL_SPIKE"      # Pic de volatilité
    LIQUIDITY_CRISIS = "LIQ_CRISIS"     # Crise de liquidité
    CORRELATION_BREAKDOWN = "CORR_BREAK" # Rupture des corrélations
    FLASH_CRASH = "FLASH_CRASH"         # Crash éclair avec rebond
    BLACK_SWAN = "BLACK_SWAN"           # Événement extrême imprévisible
    CUSTOM = "CUSTOM"                   # Scénario personnalisé

class StressScenario:
    """
    Scénario de stress pour tester la robustesse des stratégies.
    """
    
    def __init__(self, name, price_shock, volatility_multiplier, volume_multiplier, duration, start_date):
        """
        Initialise un scénario de stress.
        
        Args:
            name: Nom du scénario
            price_shock: Choc de prix (ex: -0.3 pour -30%)
            volatility_multiplier: Multiplicateur de volatilité
            volume_multiplier: Multiplicateur de volume
            duration: Durée du scénario (timedelta)
            start_date: Date de début du scénario
        """
        self.name = name
        self.price_shock = price_shock
        self.volatility_multiplier = volatility_multiplier
        self.volume_multiplier = volume_multiplier
        self.duration = duration
        self.start_date = start_date
        
    def apply_to_data(self, market_data):
        """
        Applique le scénario de stress aux données de marché.
        
        Args:
            market_data: Dictionnaire de DataFrames contenant les données par symbole
        
        Returns:
            Dictionnaire de DataFrames modifiés
        """
        stressed_data = {symbol: df.copy() for symbol, df in market_data.items()}
        end_date = self.start_date + self.duration
        
        for symbol, df in stressed_data.items():
            stress_mask = (df.index >= self.start_date) & (df.index <= end_date)
            
            if not any(stress_mask):
                continue
                
            if self.price_shock != 0:
                shock_factor = 1 + self.price_shock
                df.loc[stress_mask, 'open'] *= shock_factor
                df.loc[stress_mask, 'high'] *= shock_factor
                df.loc[stress_mask, 'low'] *= shock_factor
                df.loc[stress_mask, 'close'] *= shock_factor
            
            if self.volatility_multiplier != 1:
                avg_range = df['high'] - df['low']
                additional_range = avg_range * (self.volatility_multiplier - 1) / 2
                df.loc[stress_mask, 'high'] += additional_range[stress_mask]
                df.loc[stress_mask, 'low'] -= additional_range[stress_mask]
            
            if self.volume_multiplier != 1:
                df.loc[stress_mask, 'volume'] *= self.volume_multiplier
        
        return stressed_data

class StressTester:
    """
    Classe pour effectuer des tests de stress sur des stratégies de trading.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le testeur de stress.
        
        Args:
            config: Configuration pour les tests de stress
        """
        self.config = config or {}
        
        # Paramètres par défaut
        self.default_params = {
            "scenarios": {
                "crash": {
                    "price_drop": -0.4,       # 40% de chute
                    "duration_days": 7,       # Sur 7 jours
                    "recovery_days": 30,      # 30 jours pour récupérer
                    "recovery_percent": 0.6   # Récupération de 60% de la chute
                },
                "volatility_spike": {
                    "vol_multiplier": 5.0,    # Multiplication par 5 de la volatilité
                    "duration_days": 10,      # Sur 10 jours
                    "decay_days": 20          # 20 jours de retour à la normale
                },
                "liquidity_crisis": {
                    "volume_drop": -0.7,      # 70% de baisse du volume
                    "spread_increase": 10.0,  # Spread multiplié par 10
                    "duration_days": 5,       # Sur 5 jours
                    "recovery_days": 15       # 15 jours pour récupérer
                },
                "flash_crash": {
                    "price_drop": -0.2,       # 20% de chute
                    "duration_hours": 4,      # Sur 4 heures
                    "recovery_hours": 12,     # 12 heures pour récupérer
                    "recovery_percent": 0.9   # Récupération de 90%
                }
            },
            "monte_carlo": {
                "num_simulations": 100,       # Nombre de simulations
                "confidence_level": 0.95      # Niveau de confiance
            }
        }
        
        # Fusionner avec la configuration fournie
        for key, default_value in self.default_params.items():
            if key in self.config:
                if isinstance(default_value, dict) and isinstance(self.config[key], dict):
                    for sub_key, sub_value in default_value.items():
                        if sub_key not in self.config[key]:
                            self.config[key][sub_key] = sub_value
                else:
                    continue  # Conserver la valeur fournie
            else:
                self.config[key] = default_value
                
    def generate_stress_scenario(self, data: pd.DataFrame, scenario_type: ScenarioType, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario de stress sur les données fournies.
        
        Args:
            data: DataFrame avec les données OHLCV
            scenario_type: Type de scénario à générer
            custom_params: Paramètres personnalisés pour le scénario
            
        Returns:
            DataFrame avec les données stressées
        """
        if data.empty:
            raise ValueError("Les données d'entrée sont vides")
            
        if not isinstance(scenario_type, ScenarioType):
            raise ValueError(f"Type de scénario invalide: {scenario_type}")
            
        scenario_generators = {
            ScenarioType.CRASH: self._generate_crash_scenario,
            ScenarioType.VOLATILITY_SPIKE: self._generate_volatility_spike_scenario,
            ScenarioType.LIQUIDITY_CRISIS: self._generate_liquidity_crisis_scenario,
            ScenarioType.FLASH_CRASH: self._generate_flash_crash_scenario,
            ScenarioType.CORRELATION_BREAKDOWN: self._generate_correlation_breakdown_scenario,
            ScenarioType.BLACK_SWAN: self._generate_black_swan_scenario,
            ScenarioType.CUSTOM: self._generate_custom_scenario
        }
        
        generator = scenario_generators.get(scenario_type)
        if generator is None:
            raise ValueError(f"Générateur non trouvé pour le scénario: {scenario_type}")
            
        return generator(data, custom_params)
            
    def _generate_crash_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """Génère un scénario de crash."""
        params = self.config["scenarios"]["crash"].copy()
        if custom_params:
            params.update(custom_params)
            
        stressed_data = data.copy()
        start_idx = len(data) // 4  # Commence au premier quart des données
        
        # Appliquer la chute
        drop_period = int(params["duration_days"] * len(data) / 252)  # Convertir en périodes
        drop_factor = 1 + params["price_drop"]
        
        for col in ['open', 'high', 'low', 'close']:
            stressed_data.iloc[start_idx:start_idx + drop_period, stressed_data.columns.get_loc(col)] *= drop_factor
            
        # Appliquer la récupération
        recovery_period = int(params["recovery_days"] * len(data) / 252)
        recovery_factor = 1 + (params["price_drop"] * params["recovery_percent"])
        
        for col in ['open', 'high', 'low', 'close']:
            stressed_data.iloc[start_idx + drop_period:start_idx + drop_period + recovery_period, 
                             stressed_data.columns.get_loc(col)] *= recovery_factor
            
        return stressed_data
        
    def _generate_volatility_spike_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """Génère un scénario de pic de volatilité."""
        params = self.config["scenarios"]["volatility_spike"].copy()
        if custom_params:
            params.update(custom_params)
            
        stressed_data = data.copy()
        start_idx = len(data) // 4
        
        # Période de volatilité accrue
        vol_period = int(params["duration_days"] * len(data) / 252)
        
        # Calculer la volatilité normale
        normal_vol = data['close'].pct_change().std()
        
        # Augmenter la volatilité
        for i in range(start_idx, start_idx + vol_period):
            # Générer un facteur de volatilité plus important
            vol_factor = params["vol_multiplier"] * np.random.normal(1, normal_vol)
            
            # Calculer le prix médian
            mid_price = stressed_data.iloc[i]['close']
            
            # Ajuster les prix avec une volatilité accrue
            stressed_data.iloc[i, stressed_data.columns.get_loc('high')] = mid_price * (1 + abs(vol_factor))
            stressed_data.iloc[i, stressed_data.columns.get_loc('low')] = mid_price * (1 - abs(vol_factor))
            stressed_data.iloc[i, stressed_data.columns.get_loc('open')] = mid_price * (1 + np.random.normal(0, abs(vol_factor)))
            stressed_data.iloc[i, stressed_data.columns.get_loc('close')] = mid_price * (1 + np.random.normal(0, abs(vol_factor)))
            
            # Augmenter le volume pendant les périodes de forte volatilité
            stressed_data.iloc[i, stressed_data.columns.get_loc('volume')] *= (1 + abs(vol_factor))
            
        return stressed_data
        
    def _generate_liquidity_crisis_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """Génère un scénario de crise de liquidité."""
        params = self.config["scenarios"]["liquidity_crisis"].copy()
        if custom_params:
            params.update(custom_params)
            
        stressed_data = data.copy()
        start_idx = len(data) // 4
        
        # Période de crise de liquidité
        crisis_period = int(params["duration_days"] * len(data) / 252)
        
            # Réduire le volume
        volume_factor = 1 + params["volume_drop"]
        stressed_data.iloc[start_idx:start_idx + crisis_period, stressed_data.columns.get_loc('volume')] *= volume_factor
        
        # Augmenter les spreads (différence high-low)
        spread_factor = params["spread_increase"]
        for i in range(start_idx, start_idx + crisis_period):
            mid_price = (stressed_data.iloc[i]['high'] + stressed_data.iloc[i]['low']) / 2
            spread = stressed_data.iloc[i]['high'] - stressed_data.iloc[i]['low']
            new_spread = spread * spread_factor
            stressed_data.iloc[i, stressed_data.columns.get_loc('high')] = mid_price + new_spread/2
            stressed_data.iloc[i, stressed_data.columns.get_loc('low')] = mid_price - new_spread/2
            
        return stressed_data

    def _generate_flash_crash_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """Génère un scénario de crash éclair."""
        params = self.config["scenarios"]["flash_crash"].copy()
        if custom_params:
            params.update(custom_params)
            
        stressed_data = data.copy()
        start_idx = len(data) // 4
        
        # Convertir les heures en périodes
        crash_period = int(params["duration_hours"] * len(data) / (252 * 24))
        recovery_period = int(params["recovery_hours"] * len(data) / (252 * 24))
        
        # S'assurer que les périodes sont d'au moins 1
        crash_period = max(1, crash_period)
        recovery_period = max(1, recovery_period)
        
        # Appliquer le crash éclair
        drop_factor = 1 + params["price_drop"]  # price_drop est négatif, donc drop_factor < 1
        
        # S'assurer que la chute est d'au moins 10%
        if drop_factor > 0.9:  # Si la chute est inférieure à 10%
            drop_factor = 0.9  # Force une chute de 10%
            
        # Prix avant le crash
        pre_crash_price = stressed_data.iloc[start_idx - 1]['close']
        
        for i in range(start_idx, start_idx + crash_period):
            # Calculer un facteur de chute progressif
            progress = (i - start_idx) / crash_period
            current_drop = 1 - (1 - drop_factor) * (1 - progress)  # Commence à 1 et descend jusqu'à drop_factor
            
            # Appliquer la chute aux prix
            for col in ['open', 'high', 'low', 'close']:
                base_price = pre_crash_price if col == 'open' else stressed_data.iloc[i-1]['close']
                stressed_data.iloc[i, stressed_data.columns.get_loc(col)] = base_price * current_drop
            
            # Augmenter le volume pendant le crash
            stressed_data.iloc[i, stressed_data.columns.get_loc('volume')] *= (2 + abs(1 - current_drop))
            
        # Appliquer le rebond
        recovery_start = start_idx + crash_period
        recovery_end = min(recovery_start + recovery_period, len(stressed_data))
        
        # Prix après le crash
        post_crash_price = stressed_data.iloc[recovery_start - 1]['close']
        price_diff = pre_crash_price - post_crash_price
        recovery_amount = price_diff * params["recovery_percent"]
        
        for i in range(recovery_start, recovery_end):
            # Calculer un facteur de récupération progressif
            progress = (i - recovery_start) / recovery_period
            current_recovery = post_crash_price * (1 + (recovery_amount / post_crash_price) * progress)
            
            # Appliquer la récupération aux prix
            for col in ['open', 'high', 'low', 'close']:
                stressed_data.iloc[i, stressed_data.columns.get_loc(col)] = current_recovery * (1 + np.random.normal(0, 0.001))
                
                # Le volume reste élevé mais diminue progressivement
            stressed_data.iloc[i, stressed_data.columns.get_loc('volume')] *= (1.5 - 0.5 * progress)
                
        return stressed_data

    def _generate_correlation_breakdown_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """Génère un scénario de rupture des corrélations."""
        stressed_data = data.copy()
        
        # Cette méthode nécessite plusieurs actifs pour être pertinente
        # Pour un seul actif, on simule un comportement erratique
        start_idx = len(data) // 4
        period = len(data) // 8
        
        for i in range(start_idx, start_idx + period):
            random_factor = np.random.normal(1, 0.1)
            for col in ['open', 'high', 'low', 'close']:
                stressed_data.iloc[i, stressed_data.columns.get_loc(col)] *= random_factor
                
        return stressed_data

    def _generate_black_swan_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """Génère un scénario d'événement extrême imprévisible."""
        stressed_data = data.copy()
        
        # Simuler un événement extrême avec une probabilité très faible
        if np.random.random() < 0.01:  # 1% de chance
            start_idx = np.random.randint(0, len(data) - len(data)//4)
            duration = len(data) // 20  # 5% de la période
            
            # Choc extrême
            shock_factor = np.random.choice([-0.5, -0.4, -0.3, 0.3, 0.4, 0.5])
            for col in ['open', 'high', 'low', 'close']:
                stressed_data.iloc[start_idx:start_idx + duration, stressed_data.columns.get_loc(col)] *= (1 + shock_factor)
                
            # Impact sur le volume
            volume_factor = np.random.uniform(2, 5)
            stressed_data.iloc[start_idx:start_idx + duration, stressed_data.columns.get_loc('volume')] *= volume_factor
            
        return stressed_data

    def _generate_custom_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """Génère un scénario personnalisé."""
        if not custom_params:
            raise ValueError("Les paramètres personnalisés sont requis pour un scénario custom")
            
        stressed_data = data.copy()
        start_idx = len(data) // 4
        
        # Appliquer les modifications selon les paramètres personnalisés
        if 'price_shock' in custom_params:
            shock_factor = 1 + custom_params['price_shock']
            duration = int(custom_params.get('duration_days', 5) * len(data) / 252)
            
            for col in ['open', 'high', 'low', 'close']:
                stressed_data.iloc[start_idx:start_idx + duration, stressed_data.columns.get_loc(col)] *= shock_factor
                
        if 'volatility_multiplier' in custom_params:
            vol_factor = custom_params['volatility_multiplier']
            duration = int(custom_params.get('duration_days', 5) * len(data) / 252)
            
            for i in range(start_idx, start_idx + duration):
                mid_price = (stressed_data.iloc[i]['high'] + stressed_data.iloc[i]['low']) / 2
                spread = stressed_data.iloc[i]['high'] - stressed_data.iloc[i]['low']
                new_spread = spread * vol_factor
                stressed_data.iloc[i, stressed_data.columns.get_loc('high')] = mid_price + new_spread/2
                stressed_data.iloc[i, stressed_data.columns.get_loc('low')] = mid_price - new_spread/2
                
        if 'volume_multiplier' in custom_params:
            vol_mult = custom_params['volume_multiplier']
            duration = int(custom_params.get('duration_days', 5) * len(data) / 252)
            stressed_data.iloc[start_idx:start_idx + duration, stressed_data.columns.get_loc('volume')] *= vol_mult
            
        return stressed_data