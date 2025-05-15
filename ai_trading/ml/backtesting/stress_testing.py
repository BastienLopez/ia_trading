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

class ScenarioType(Enum):
    """Types de scénarios de stress testing."""
    CRASH = "CRASH"                     # Chute brutale des prix
    VOLATILITY_SPIKE = "VOL_SPIKE"      # Pic de volatilité
    LIQUIDITY_CRISIS = "LIQ_CRISIS"     # Crise de liquidité
    CORRELATION_BREAKDOWN = "CORR_BREAK" # Rupture des corrélations
    FLASH_CRASH = "FLASH_CRASH"         # Crash éclair avec rebond
    BLACK_SWAN = "BLACK_SWAN"           # Événement extrême imprévisible
    CUSTOM = "CUSTOM"                   # Scénario personnalisé

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
                
    def generate_stress_scenario(self, 
                              base_data: pd.DataFrame, 
                              scenario_type: ScenarioType,
                              custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario de stress en modifiant les données de base.
        
        Args:
            base_data: Données OHLCV de base
            scenario_type: Type de scénario à générer
            custom_params: Paramètres personnalisés optionnels
            
        Returns:
            DataFrame contenant les données modifiées
        """
        if base_data.empty:
            logger.error("Données de base vides, impossible de générer un scénario")
            return pd.DataFrame()
            
        # Copier les données pour éviter de modifier l'original
        data = base_data.copy()
        
        if scenario_type == ScenarioType.CRASH:
            return self._generate_crash_scenario(data, custom_params)
            
        elif scenario_type == ScenarioType.VOLATILITY_SPIKE:
            return self._generate_volatility_spike_scenario(data, custom_params)
            
        elif scenario_type == ScenarioType.LIQUIDITY_CRISIS:
            return self._generate_liquidity_crisis_scenario(data, custom_params)
            
        elif scenario_type == ScenarioType.FLASH_CRASH:
            return self._generate_flash_crash_scenario(data, custom_params)
            
        elif scenario_type == ScenarioType.CORRELATION_BREAKDOWN:
            return self._generate_correlation_breakdown_scenario(data, custom_params)
            
        elif scenario_type == ScenarioType.BLACK_SWAN:
            return self._generate_black_swan_scenario(data, custom_params)
            
        elif scenario_type == ScenarioType.CUSTOM:
            return self._generate_custom_scenario(data, custom_params)
            
        else:
            logger.error(f"Type de scénario non supporté: {scenario_type}")
            return data
            
    def _generate_crash_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario de crash du marché.
        
        Args:
            data: Données OHLCV
            custom_params: Paramètres personnalisés
            
        Returns:
            Données modifiées
        """
        params = custom_params or self.config["scenarios"]["crash"]
        
        # Paramètres du scénario
        price_drop = params.get("price_drop", -0.4)      # 40% de chute par défaut
        duration_days = params.get("duration_days", 7)   # 7 jours par défaut
        recovery_days = params.get("recovery_days", 30)  # 30 jours par défaut
        recovery_percent = params.get("recovery_percent", 0.6)  # 60% de récupération
        
        # Déterminer la fréquence des données
        freq = self._determine_data_frequency(data)
        
        # Convertir les jours en nombre de périodes
        duration_periods = self._days_to_periods(duration_days, freq)
        recovery_periods = self._days_to_periods(recovery_days, freq)
        
        # Vérifier que nous avons assez de données
        if len(data) < duration_periods + recovery_periods:
            logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
            # Ajuster les périodes proportionnellement
            total = duration_periods + recovery_periods
            ratio = len(data) / total
            duration_periods = int(duration_periods * ratio * 0.8)  # 80% du ratio pour être sûr
            recovery_periods = int(recovery_periods * ratio * 0.8)
            
        # Sélectionner un point de départ aléatoire pour le crash
        # (mais pas trop près du début ou de la fin)
        min_start = int(len(data) * 0.2)
        max_start = int(len(data) * 0.6)
        crash_start = np.random.randint(min_start, max_start)
        
        # Générer la phase de crash
        crash_end = min(crash_start + duration_periods, len(data))
        crash_range = np.arange(crash_start, crash_end)
        
        # Calculer les facteurs de modification pour la phase de crash
        crash_factors = np.linspace(0, 1, len(crash_range))
        crash_multipliers = 1 + price_drop * crash_factors
        
        # Appliquer les modifications aux données OHLCV
        for i, idx in enumerate(crash_range):
            multiplier = crash_multipliers[i]
            data.iloc[idx, data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= multiplier
            
            # Augmenter le volume pendant le crash (panique)
            panic_factor = 1 + (1 - multiplier) * 2  # Plus la chute est forte, plus le volume augmente
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= panic_factor
            
        # Générer la phase de récupération si demandée
        if recovery_percent > 0:
            recovery_start = crash_end
            recovery_end = min(recovery_start + recovery_periods, len(data))
            recovery_range = np.arange(recovery_start, recovery_end)
            
            # Calculer le niveau de prix après le crash
            post_crash_price = data.iloc[crash_end-1]['close']
            original_price = data.iloc[crash_start]['close'] / (1 + price_drop * 0)  # Prix avant le crash
            recovery_target = post_crash_price + (original_price - post_crash_price) * recovery_percent
            
            # Calculer les facteurs de récupération
            recovery_factors = np.linspace(0, 1, len(recovery_range))
            
            # Appliquer la récupération
            for i, idx in enumerate(recovery_range):
                factor = recovery_factors[i]
                price_now = post_crash_price * (1 + (recovery_target / post_crash_price - 1) * factor)
                multiplier = price_now / data.iloc[idx]['close']
                
                data.iloc[idx, data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= multiplier
                
                # Le volume reste élevé mais diminue progressivement
                vol_factor = 1.5 - 0.5 * factor  # De 1.5 à 1.0
                data.iloc[idx, data.columns.get_indexer(['volume'])] *= vol_factor
                
        return data
        
    def _generate_volatility_spike_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario de pic de volatilité.
        
        Args:
            data: Données OHLCV
            custom_params: Paramètres personnalisés
            
        Returns:
            Données modifiées
        """
        params = custom_params or self.config["scenarios"]["volatility_spike"]
        
        # Paramètres du scénario
        vol_multiplier = params.get("vol_multiplier", 5.0)     # Multiplication par 5 de la volatilité
        duration_days = params.get("duration_days", 10)        # 10 jours par défaut
        decay_days = params.get("decay_days", 20)              # 20 jours de retour à la normale
        
        # Déterminer la fréquence des données
        freq = self._determine_data_frequency(data)
        
        # Convertir les jours en nombre de périodes
        duration_periods = self._days_to_periods(duration_days, freq)
        decay_periods = self._days_to_periods(decay_days, freq)
        
        # Vérifier que nous avons assez de données
        if len(data) < duration_periods + decay_periods:
            logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
            # Ajuster les périodes proportionnellement
            total = duration_periods + decay_periods
            ratio = len(data) / total
            duration_periods = int(duration_periods * ratio * 0.8)
            decay_periods = int(decay_periods * ratio * 0.8)
            
        # Sélectionner un point de départ aléatoire pour le pic de volatilité
        min_start = int(len(data) * 0.2)
        max_start = int(len(data) * 0.6)
        spike_start = np.random.randint(min_start, max_start)
        
        # Calculer la volatilité de base sur les données précédant le pic
        look_back = min(30, spike_start)
        base_returns = data['close'].pct_change().iloc[spike_start-look_back:spike_start]
        base_volatility = base_returns.std()
        
        # Générer la phase de haute volatilité
        spike_end = min(spike_start + duration_periods, len(data))
        spike_range = np.arange(spike_start, spike_end)
        
        # Générer des rendements aléatoires avec volatilité accrue
        target_volatility = base_volatility * vol_multiplier
        increased_volatility = np.random.normal(0, target_volatility, len(spike_range))
        
        # Appliquer les rendements pour créer les nouveaux prix
        prices = data['close'].values.copy()
        for i, idx in enumerate(spike_range):
            prices[idx] = prices[idx-1] * (1 + increased_volatility[i])
            
            # Mettre à jour open, high, low
            daily_range = abs(increased_volatility[i]) * prices[idx]
            data.iloc[idx, data.columns.get_indexer(['open'])] = prices[idx] * (1 - 0.2 * np.random.random())
            data.iloc[idx, data.columns.get_indexer(['high'])] = prices[idx] * (1 + 0.3 * np.random.random())
            data.iloc[idx, data.columns.get_indexer(['low'])] = prices[idx] * (1 - 0.3 * np.random.random())
            data.iloc[idx, data.columns.get_indexer(['close'])] = prices[idx]
            
            # Augmenter le volume pendant la volatilité
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= 1.5 + np.random.random()
        
        # Générer la phase de retour à la normale
        decay_start = spike_end
        decay_end = min(decay_start + decay_periods, len(data))
        decay_range = np.arange(decay_start, decay_end)
        
        # Réduire progressivement la volatilité
        for i, idx in enumerate(decay_range):
            decay_factor = 1 - (i / len(decay_range))
            current_volatility = base_volatility * (1 + (vol_multiplier - 1) * decay_factor)
            
            # Générer le rendement pour cette période
            period_return = np.random.normal(0, current_volatility)
            
            # Appliquer le rendement
            prices[idx] = prices[idx-1] * (1 + period_return)
            
            # Mettre à jour open, high, low
            daily_range = abs(period_return) * prices[idx]
            data.iloc[idx, data.columns.get_indexer(['open'])] = prices[idx] * (1 - 0.1 * np.random.random() * decay_factor)
            data.iloc[idx, data.columns.get_indexer(['high'])] = prices[idx] * (1 + 0.2 * np.random.random() * decay_factor)
            data.iloc[idx, data.columns.get_indexer(['low'])] = prices[idx] * (1 - 0.2 * np.random.random() * decay_factor)
            data.iloc[idx, data.columns.get_indexer(['close'])] = prices[idx]
            
            # Réduire progressivement le volume
            vol_factor = 1 + 0.5 * decay_factor
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= vol_factor
            
        return data
        
    def _generate_liquidity_crisis_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario de crise de liquidité.
        
        Args:
            data: Données OHLCV
            custom_params: Paramètres personnalisés
            
        Returns:
            Données modifiées
        """
        params = custom_params or self.config["scenarios"]["liquidity_crisis"]
        
        # Paramètres du scénario
        volume_drop = params.get("volume_drop", -0.7)      # 70% de baisse du volume
        spread_increase = params.get("spread_increase", 10.0)  # Spread multiplié par 10
        duration_days = params.get("duration_days", 5)     # 5 jours par défaut
        recovery_days = params.get("recovery_days", 15)    # 15 jours de récupération
        
        # Déterminer la fréquence des données
        freq = self._determine_data_frequency(data)
        
        # Convertir les jours en nombre de périodes
        duration_periods = self._days_to_periods(duration_days, freq)
        recovery_periods = self._days_to_periods(recovery_days, freq)
        
        # Vérifier que nous avons assez de données
        if len(data) < duration_periods + recovery_periods:
            logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
            # Ajuster les périodes proportionnellement
            total = duration_periods + recovery_periods
            ratio = len(data) / total
            duration_periods = int(duration_periods * ratio * 0.8)
            recovery_periods = int(recovery_periods * ratio * 0.8)
            
        # Sélectionner un point de départ aléatoire
        min_start = int(len(data) * 0.2)
        max_start = int(len(data) * 0.6)
        crisis_start = np.random.randint(min_start, max_start)
        
        # Générer la phase de crise
        crisis_end = min(crisis_start + duration_periods, len(data))
        crisis_range = np.arange(crisis_start, crisis_end)
        
        # Calculer les facteurs de modification pour le volume
        crisis_factors = np.linspace(0, 1, len(crisis_range))
        volume_multipliers = 1 + volume_drop * crisis_factors
        
        # Appliquer les modifications
        for i, idx in enumerate(crisis_range):
            # Réduire le volume
            multiplier = volume_multipliers[i]
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= multiplier
            
            # Augmenter la volatilité et le spread en raison de la faible liquidité
            current_spread = spread_increase * crisis_factors[i]
            
            # Simuler un impact de spread augmenté sur les prix high/low
            current_price = data.iloc[idx]['close']
            high_low_range = current_price * 0.02 * (1 + current_spread)  # Augmenter l'écart high-low
            
            # Ajuster high et low pour refléter le spread plus large
            data.iloc[idx, data.columns.get_indexer(['high'])] = max(data.iloc[idx]['high'], current_price + high_low_range/2)
            data.iloc[idx, data.columns.get_indexer(['low'])] = min(data.iloc[idx]['low'], current_price - high_low_range/2)
            
        # Générer la phase de récupération
        recovery_start = crisis_end
        recovery_end = min(recovery_start + recovery_periods, len(data))
        recovery_range = np.arange(recovery_start, recovery_end)
        
        # Calculer les facteurs de récupération
        recovery_factors = np.linspace(0, 1, len(recovery_range))
        
        # Appliquer la récupération
        for i, idx in enumerate(recovery_range):
            factor = recovery_factors[i]
            
            # Augmenter progressivement le volume
            vol_recovery = 1 + volume_drop * (1 - factor)
            data.iloc[idx, data.columns.get_indexer(['volume'])] /= vol_recovery
            
            # Réduire progressivement le spread
            current_spread = spread_increase * (1 - factor)
            
            # Réduire l'écart high-low
            current_price = data.iloc[idx]['close']
            high_low_range = current_price * 0.02 * (1 + current_spread)
            
            # Ajuster high et low
            data.iloc[idx, data.columns.get_indexer(['high'])] = min(data.iloc[idx]['high'], current_price + high_low_range/2)
            data.iloc[idx, data.columns.get_indexer(['low'])] = max(data.iloc[idx]['low'], current_price - high_low_range/2)
            
        return data

    def _generate_flash_crash_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario de crash éclair.
        
        Args:
            data: Données OHLCV
            custom_params: Paramètres personnalisés
            
        Returns:
            Données modifiées
        """
        params = custom_params or self.config["scenarios"]["flash_crash"]
        
        # Paramètres du scénario
        price_drop = params.get("price_drop", -0.2)       # 20% de chute
        duration_hours = params.get("duration_hours", 4)     # Sur 4 heures
        recovery_hours = params.get("recovery_hours", 12)     # 12 heures pour récupérer
        recovery_percent = params.get("recovery_percent", 0.9) # Récupération de 90%
        
        # Déterminer la fréquence des données
        freq = self._determine_data_frequency(data)
        
        # Convertir les heures en nombre de périodes
        duration_periods = self._hours_to_periods(duration_hours, freq)
        recovery_periods = self._hours_to_periods(recovery_hours, freq)
        
        # Vérifier que nous avons assez de données
        if len(data) < duration_periods + recovery_periods:
            logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
            # Ajuster les périodes proportionnellement
            total = duration_periods + recovery_periods
            ratio = len(data) / total
            duration_periods = int(duration_periods * ratio * 0.8)
            recovery_periods = int(recovery_periods * ratio * 0.8)
            
        # Sélectionner un point de départ aléatoire pour le crash
        # (mais pas trop près du début ou de la fin)
        min_start = int(len(data) * 0.2)
        max_start = int(len(data) * 0.6)
        crash_start = np.random.randint(min_start, max_start)
        
        # Générer la phase de crash
        crash_end = min(crash_start + duration_periods, len(data))
        crash_range = np.arange(crash_start, crash_end)
        
        # Calculer les facteurs de modification pour la phase de crash
        crash_factors = np.linspace(0, 1, len(crash_range))
        crash_multipliers = 1 + price_drop * crash_factors
        
        # Appliquer les modifications aux données OHLCV
        for i, idx in enumerate(crash_range):
            multiplier = crash_multipliers[i]
            data.iloc[idx, data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= multiplier
            
            # Augmenter le volume pendant le crash (panique)
            panic_factor = 1 + (1 - multiplier) * 2  # Plus la chute est forte, plus le volume augmente
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= panic_factor
            
        # Générer la phase de récupération si demandée
        if recovery_percent > 0:
            recovery_start = crash_end
            recovery_end = min(recovery_start + recovery_periods, len(data))
            recovery_range = np.arange(recovery_start, recovery_end)
            
            # Calculer le niveau de prix après le crash
            post_crash_price = data.iloc[crash_end-1]['close']
            original_price = data.iloc[crash_start]['close'] / (1 + price_drop * 0)  # Prix avant le crash
            recovery_target = post_crash_price + (original_price - post_crash_price) * recovery_percent
            
            # Calculer les facteurs de récupération
            recovery_factors = np.linspace(0, 1, len(recovery_range))
            
            # Appliquer la récupération
            for i, idx in enumerate(recovery_range):
                factor = recovery_factors[i]
                price_now = post_crash_price * (1 + (recovery_target / post_crash_price - 1) * factor)
                multiplier = price_now / data.iloc[idx]['close']
                
                data.iloc[idx, data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= multiplier
                
                # Le volume reste élevé mais diminue progressivement
                vol_factor = 1.5 - 0.5 * factor  # De 1.5 à 1.0
                data.iloc[idx, data.columns.get_indexer(['volume'])] *= vol_factor
                
        return data

    def _generate_correlation_breakdown_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario de rupture des corrélations.
        
        Args:
            data: Données OHLCV
            custom_params: Paramètres personnalisés
            
        Returns:
            Données modifiées
        """
        params = custom_params or self.config["scenarios"]["correlation_breakdown"]
        
        # Paramètres du scénario
        correlation_drop = params.get("correlation_drop", -0.5)  # Réduction de la corrélation de 50%
        duration_days = params.get("duration_days", 7)            # Sur 7 jours
        
        # Déterminer la fréquence des données
        freq = self._determine_data_frequency(data)
        
        # Convertir les jours en nombre de périodes
        duration_periods = self._days_to_periods(duration_days, freq)
        
        # Vérifier que nous avons assez de données
        if len(data) < duration_periods:
            logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
            # Ajuster la durée proportionnellement
            ratio = len(data) / duration_periods
            duration_periods = int(duration_periods * ratio * 0.8)
            
        # Générer des rendements aléatoires avec corrélation réduite
        returns = np.random.normal(0, 0.001, len(data))
        
        # Appliquer les rendements pour créer les nouveaux prix
        prices = data['close'].values.copy()
        for i, idx in enumerate(range(len(data))):
            prices[idx] = prices[idx-1] * (1 + returns[i])
            
            # Mettre à jour open, high, low
            daily_range = abs(returns[i]) * prices[idx]
            data.iloc[idx, data.columns.get_indexer(['open'])] = prices[idx] * (1 - 0.2 * np.random.random())
            data.iloc[idx, data.columns.get_indexer(['high'])] = prices[idx] * (1 + 0.3 * np.random.random())
            data.iloc[idx, data.columns.get_indexer(['low'])] = prices[idx] * (1 - 0.3 * np.random.random())
            data.iloc[idx, data.columns.get_indexer(['close'])] = prices[idx]
            
            # Augmenter le volume pendant la volatilité
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= 1.5 + np.random.random()
        
        return data

    def _generate_black_swan_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario d'événement extrême imprévisible.
        
        Args:
            data: Données OHLCV
            custom_params: Paramètres personnalisés
            
        Returns:
            Données modifiées
        """
        params = custom_params or self.config["scenarios"]["black_swan"]
        
        # Paramètres du scénario
        probability = params.get("probability", 0.01)  # Probabilité d'occurrence
        impact = params.get("impact", 0.5)            # Impact sur le prix
        duration_days = params.get("duration_days", 7)  # Sur 7 jours
        
        # Déterminer la fréquence des données
        freq = self._determine_data_frequency(data)
        
        # Convertir les jours en nombre de périodes
        duration_periods = self._days_to_periods(duration_days, freq)
        
        # Vérifier que nous avons assez de données
        if len(data) < duration_periods:
            logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
            # Ajuster la durée proportionnellement
            ratio = len(data) / duration_periods
            duration_periods = int(duration_periods * ratio * 0.8)
            
        # Générer un événement aléatoire
        event = np.random.choice([True, False], p=[probability, 1-probability])
        
        if event:
            # Générer un impact aléatoire
            impact_value = np.random.uniform(0, impact)
            
            # Appliquer l'impact
            data['close'] *= (1 + impact_value)
            
            # Mettre à jour open, high, low
            daily_range = abs(impact_value) * data['close']
            data.iloc[duration_periods-1, data.columns.get_indexer(['open'])] = data.iloc[duration_periods-2]['close'] * (1 - 0.2 * np.random.random())
            data.iloc[duration_periods-1, data.columns.get_indexer(['high'])] = data.iloc[duration_periods-1]['close'] * (1 + 0.3 * np.random.random())
            data.iloc[duration_periods-1, data.columns.get_indexer(['low'])] = data.iloc[duration_periods-1]['close'] * (1 - 0.3 * np.random.random())
            data.iloc[duration_periods-1, data.columns.get_indexer(['close'])] = data.iloc[duration_periods-1]['close']
            
            # Augmenter le volume pendant la volatilité
            data.iloc[duration_periods-1, data.columns.get_indexer(['volume'])] *= 1.5 + np.random.random()
        
        return data

    def _generate_custom_scenario(self, data: pd.DataFrame, custom_params: Dict = None) -> pd.DataFrame:
        """
        Génère un scénario personnalisé.
        
        Args:
            data: Données OHLCV
            custom_params: Paramètres personnalisés
            
        Returns:
            Données modifiées
        """
        if custom_params is None:
            logger.error("Paramètres personnalisés manquants")
            return data
        
        # Copier les données pour éviter de modifier l'original
        data = data.copy()
        
        # Appliquer les modifications personnalisées
        for key, value in custom_params.items():
            if key in data.columns:
                data[key] = value
            else:
                logger.warning(f"Colonne '{key}' non trouvée dans les données")
        
        return data

    def _determine_data_frequency(self, data: pd.DataFrame) -> pd.Timedelta:
        """
        Détermine la fréquence des données.
        
        Args:
            data: DataFrame contenant les données
            
        Returns:
            Fréquence des données
        """
        if len(data) == 0:
            logger.error("Données vides, impossible de déterminer la fréquence")
            return pd.Timedelta(days=1)
        
        # Tenter de déterminer la fréquence en utilisant les index
        if len(data) > 1:
            # Calculer la différence médiane entre les timestamps
            diff_series = data.index.to_series().diff().dropna()
            if not diff_series.empty:
                freq = diff_series.median()
                return freq
        
        # Par défaut, supposer une fréquence quotidienne
        logger.warning("Fréquence des données non déterminable, utilisation par défaut (1 jour)")
        return pd.Timedelta(days=1)

    def _days_to_periods(self, days: int, freq: pd.Timedelta) -> int:
        """
        Convertit des jours en nombre de périodes.
        
        Args:
            days: Nombre de jours
            freq: Fréquence des données
            
        Returns:
            Nombre de périodes
        """
        return int(days * pd.Timedelta(days=1) / freq)

    def _hours_to_periods(self, hours: int, freq: pd.Timedelta) -> int:
        """
        Convertit des heures en nombre de périodes.
        
        Args:
            hours: Nombre d'heures
            freq: Fréquence des données
            
        Returns:
            Nombre de périodes
        """
        return int(hours * pd.Timedelta(hours=1) / freq) 