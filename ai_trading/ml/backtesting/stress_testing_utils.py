"""
Fonctions utilitaires pour le module de stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from datetime import timedelta

# Configuration du logging
logger = logging.getLogger(__name__)

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
        
    # Vérifier qu'il y a au moins deux points temporels pour déterminer une fréquence
    if len(data.index) < 2:
        return "unknown"
        
    # Calculer les différences entre les timestamps
    diffs = []
    for i in range(1, len(data.index)):
        diff = data.index[i] - data.index[i-1]
        diffs.append(diff)
    
    # Convertir en Series pour faciliter l'analyse
    diffs = pd.Series(diffs)
    
    # Si toutes les différences sont identiques, la fréquence est régulière
    if len(diffs.unique()) == 1:
        time_diff = diffs.iloc[0]
    else:
        # Sinon, prendre la différence la plus commune
        time_diff = diffs.value_counts().index[0]
    
    # Convertir en minutes pour simplifier la catégorisation
    minutes = time_diff.total_seconds() / 60
    
    # Déterminer la fréquence en fonction de l'intervalle en minutes
    if minutes < 1:
        return "1s"  # Secondes
    elif minutes == 1:
        return "1min"
    elif minutes < 5:
        return "1min"
    elif minutes < 15:
        return "5min"
    elif minutes < 30:
        return "15min"
    elif minutes < 60:
        return "30min"
    elif minutes == 60:
        return "1h"  # Exactement 1 heure
    elif minutes < 60 * 4:
        return "1h"
    elif minutes < 60 * 24:
        return "4h"
    elif abs(minutes - 60 * 24) < 1:  # À peu près 24 heures
        return "1d"
    else:
        return "1d"

def days_to_periods(days: int, frequency: str) -> int:
    """
    Convertit un nombre de jours en nombre de périodes selon la fréquence.
    
    Args:
        days: Nombre de jours
        frequency: Fréquence des données (1min, 5min, 1h, 1d, etc.)
        
    Returns:
        Nombre de périodes
    """
    if frequency == "1s":
        return days * 24 * 60 * 60
    elif frequency == "1min":
        return days * 24 * 60
    elif frequency == "5min":
        return days * 24 * 12
    elif frequency == "15min":
        return days * 24 * 4
    elif frequency == "30min":
        return days * 24 * 2
    elif frequency == "1h":
        return days * 24
    elif frequency == "4h":
        return days * 6
    elif frequency == "1d":
        return days
    else:
        logger.warning(f"Fréquence inconnue: {frequency}, retour par défaut")
        return days

def generate_flash_crash_scenario(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Génère un scénario de flash crash (chute rapide suivie d'un rebond).
    
    Args:
        data: Données OHLCV
        params: Paramètres du scénario
        
    Returns:
        Données modifiées
    """
    # Paramètres du scénario
    price_drop = params.get("price_drop", -0.2)      # 20% de chute par défaut
    duration_hours = params.get("duration_hours", 4)  # 4 heures par défaut
    recovery_hours = params.get("recovery_hours", 12) # 12 heures de récupération
    recovery_percent = params.get("recovery_percent", 0.9)  # 90% de récupération
    
    # Déterminer la fréquence des données
    freq = determine_data_frequency(data)
    
    # Convertir les heures en nombre de périodes
    if freq == "1d":
        duration_periods = max(1, int(duration_hours / 24))
        recovery_periods = max(1, int(recovery_hours / 24))
    elif freq == "4h":
        duration_periods = max(1, int(duration_hours / 4))
        recovery_periods = max(1, int(recovery_hours / 4))
    elif freq == "1h":
        duration_periods = duration_hours
        recovery_periods = recovery_hours
    elif freq == "30min":
        duration_periods = duration_hours * 2
        recovery_periods = recovery_hours * 2
    elif freq == "15min":
        duration_periods = duration_hours * 4
        recovery_periods = recovery_hours * 4
    elif freq == "5min":
        duration_periods = duration_hours * 12
        recovery_periods = recovery_hours * 12
    elif freq == "1min":
        duration_periods = duration_hours * 60
        recovery_periods = recovery_hours * 60
    else:
        duration_periods = duration_hours * 60
        recovery_periods = recovery_hours * 60
    
    # Vérifier que nous avons assez de données
    if len(data) < duration_periods + recovery_periods:
        logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
        # Ajuster les périodes proportionnellement
        total = duration_periods + recovery_periods
        ratio = len(data) / total
        duration_periods = max(1, int(duration_periods * ratio * 0.8))
        recovery_periods = max(1, int(recovery_periods * ratio * 0.8))
        
    # Sélectionner un point de départ aléatoire pour le flash crash
    min_start = int(len(data) * 0.3)
    max_start = int(len(data) * 0.7)
    crash_start = np.random.randint(min_start, max_start)
    
    # Données modifiées
    modified_data = data.copy()
    
    # Générer la phase de crash éclair
    crash_end = min(crash_start + duration_periods, len(data))
    crash_range = np.arange(crash_start, crash_end)
    
    # Calculer les facteurs de modification pour la phase de crash
    crash_factors = np.linspace(0, 1, len(crash_range))
    crash_multipliers = 1 + price_drop * crash_factors
    
    # Appliquer les modifications aux données OHLCV
    for i, idx in enumerate(crash_range):
        multiplier = crash_multipliers[i]
        modified_data.iloc[idx, modified_data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= multiplier
        
        # Augmenter fortement le volume pendant le flash crash
        panic_factor = 1 + (1 - multiplier) * 5  # Volume très élevé pendant la chute
        modified_data.iloc[idx, modified_data.columns.get_indexer(['volume'])] *= panic_factor
    
    # Générer la phase de rebond
    recovery_start = crash_end
    recovery_end = min(recovery_start + recovery_periods, len(data))
    recovery_range = np.arange(recovery_start, recovery_end)
    
    if len(recovery_range) > 0:
        # Calculer le niveau de prix après le crash
        post_crash_price = modified_data.iloc[crash_end-1]['close']
        original_price = data.iloc[crash_start]['close']
        recovery_target = post_crash_price + (original_price - post_crash_price) * recovery_percent
        
        # Calculer les facteurs de récupération
        recovery_factors = np.linspace(0, 1, len(recovery_range))
        
        # Appliquer la récupération
        for i, idx in enumerate(recovery_range):
            factor = recovery_factors[i]
            price_now = post_crash_price * (1 + (recovery_target / post_crash_price - 1) * factor)
            multiplier = price_now / modified_data.iloc[idx]['close']
            
            modified_data.iloc[idx, modified_data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= multiplier
            
            # Volume qui reste élevé pendant la récupération
            vol_factor = 2.0 - factor  # De 2.0 à 1.0
            modified_data.iloc[idx, modified_data.columns.get_indexer(['volume'])] *= vol_factor
    
    return modified_data

def generate_correlation_breakdown_scenario(data: pd.DataFrame, correlated_data: Dict[str, pd.DataFrame], params: Dict) -> Dict[str, pd.DataFrame]:
    """
    Génère un scénario de rupture de corrélation entre actifs.
    
    Args:
        data: Données OHLCV de l'actif principal
        correlated_data: Dictionnaire contenant les données d'autres actifs corrélés
        params: Paramètres du scénario
        
    Returns:
        Dictionnaire avec les données modifiées pour tous les actifs
    """
    # Paramètres du scénario
    duration_days = params.get("duration_days", 10)  # 10 jours par défaut
    breakdown_strength = params.get("breakdown_strength", 0.8)  # Force de la rupture (0-1)
    recovery_days = params.get("recovery_days", 15)  # 15 jours pour récupérer
    
    # Copier les données pour éviter de modifier l'original
    modified_data = {k: v.copy() for k, v in correlated_data.items()}
    modified_data["main"] = data.copy()
    
    # Déterminer la fréquence des données
    freq = determine_data_frequency(data)
    
    # Convertir les jours en nombre de périodes
    duration_periods = days_to_periods(duration_days, freq)
    recovery_periods = days_to_periods(recovery_days, freq)
    
    # Vérifier que nous avons assez de données
    if len(data) < duration_periods + recovery_periods:
        logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
        total = duration_periods + recovery_periods
        ratio = len(data) / total
        duration_periods = max(1, int(duration_periods * ratio * 0.8))
        recovery_periods = max(1, int(recovery_periods * ratio * 0.8))
    
    # Sélectionner un point de départ aléatoire
    min_start = int(len(data) * 0.2)
    max_start = int(len(data) * 0.6)
    breakdown_start = np.random.randint(min_start, max_start)
    
    # Calculer les corrélations initiales
    returns = {}
    for asset, asset_data in modified_data.items():
        if len(asset_data) > 5:  # Vérifier qu'il y a assez de données
            returns[asset] = asset_data['close'].pct_change().dropna()
    
    # Matrice de corrélation initiale
    correlation_matrix = pd.DataFrame({k: v for k, v in returns.items()}).corr()
    
    # Générer la phase de rupture
    breakdown_end = min(breakdown_start + duration_periods, len(data))
    breakdown_range = np.arange(breakdown_start, breakdown_end)
    
    # Pour chaque actif (sauf l'actif principal), modifier les rendements pour réduire la corrélation
    for asset in list(modified_data.keys()):
        if asset == "main" or len(modified_data[asset]) < len(breakdown_range):
            continue
            
        # Corrélation initiale avec l'actif principal
        initial_corr = correlation_matrix.loc["main", asset]
        
        # Créer des rendements qui s'éloignent de la corrélation initiale
        for i, idx in enumerate(breakdown_range):
            breakdown_factor = breakdown_strength * (i / len(breakdown_range))
            
            # Rendement actuel de l'actif principal
            if i > 0 and idx > 0 and idx < len(modified_data["main"]):
                main_return = (modified_data["main"].iloc[idx]['close'] / 
                            modified_data["main"].iloc[idx-1]['close']) - 1
                
                # Générer un rendement opposé
                if initial_corr > 0:
                    # Si corrélation positive, générer un rendement opposé
                    opposite_return = -main_return * breakdown_factor + main_return * (1 - breakdown_factor)
                else:
                    # Si corrélation négative, générer un rendement similaire
                    opposite_return = main_return * breakdown_factor + (-main_return) * (1 - breakdown_factor)
                
                # Ajouter du bruit aléatoire
                noise = np.random.normal(0, 0.01)
                modified_return = opposite_return + noise
                
                # Appliquer le rendement modifié
                if idx < len(modified_data[asset]):
                    old_price = modified_data[asset].iloc[idx-1]['close']
                    new_price = old_price * (1 + modified_return)
                    
                    # Mettre à jour les prix
                    price_ratio = new_price / modified_data[asset].iloc[idx]['close']
                    modified_data[asset].iloc[idx, modified_data[asset].columns.get_indexer(['open', 'high', 'low', 'close'])] *= price_ratio
    
    # Retourner les données modifiées
    return modified_data

def generate_black_swan_scenario(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Génère un scénario d'événement noir (black swan) - événement extrême et imprévu.
    
    Args:
        data: Données OHLCV
        params: Paramètres du scénario
        
    Returns:
        Données modifiées
    """
    # Paramètres du scénario
    price_change = params.get("price_change", -0.4)  # 40% de chute par défaut
    volatility_factor = params.get("volatility_factor", 5.0)  # Volatilité x5
    aftermath_days = params.get("aftermath_days", 30)  # 30 jours d'effets
    is_positive = params.get("is_positive", False)  # Par défaut, événement négatif
    
    # Inverser le mouvement de prix si c'est un événement positif
    if is_positive:
        price_change = abs(price_change)
    
    # Déterminer la fréquence des données
    freq = determine_data_frequency(data)
    
    # Convertir les jours en nombre de périodes
    aftermath_periods = days_to_periods(aftermath_days, freq)
    
    # Vérifier que nous avons assez de données
    if len(data) < aftermath_periods + 5:
        logger.warning("Pas assez de données pour le scénario. Ajustement de la durée.")
        aftermath_periods = max(5, int(len(data) * 0.7))
    
    # Sélectionner un point de départ aléatoire
    min_start = int(len(data) * 0.2)
    max_start = int(len(data) * 0.5)
    event_time = np.random.randint(min_start, max_start)
    
    # Données modifiées
    modified_data = data.copy()
    
    # Appliquer le changement de prix initial (le "swan")
    # Cet événement se produit brusquement sur 1-2 périodes
    shock_duration = min(2, len(data) - event_time)
    for i in range(shock_duration):
        idx = event_time + i
        shock_factor = price_change * (i + 1) / shock_duration
        modified_data.iloc[idx, modified_data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= (1 + shock_factor)
        
        # Volume extrêmement élevé pendant l'événement
        modified_data.iloc[idx, modified_data.columns.get_indexer(['volume'])] *= 3.0 + np.random.random() * 2.0
    
    # Période après l'événement (réactions du marché)
    aftermath_start = event_time + shock_duration
    aftermath_end = min(aftermath_start + aftermath_periods, len(data))
    aftermath_range = np.arange(aftermath_start, aftermath_end)
    
    # Calculer la volatilité de base avant l'événement
    pre_event_data = data.iloc[:event_time]
    base_volatility = pre_event_data['close'].pct_change().std()
    
    # Simuler une volatilité accrue suite à l'événement
    for i, idx in enumerate(aftermath_range):
        # La volatilité diminue progressivement
        time_factor = 1 - (i / len(aftermath_range))
        current_volatility = base_volatility * (1 + (volatility_factor - 1) * time_factor)
        
        # Générer un rendement avec cette volatilité
        random_return = np.random.normal(0, current_volatility)
        
        # Appliquer le rendement
        if idx > 0 and idx < len(modified_data):
            old_price = modified_data.iloc[idx-1]['close']
            new_price = old_price * (1 + random_return)
            
            # Mettre à jour les prix
            price_ratio = new_price / modified_data.iloc[idx]['close']
            modified_data.iloc[idx, modified_data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= price_ratio
            
            # Volume plus élevé mais diminuant
            volume_factor = 1.0 + 2.0 * time_factor
            modified_data.iloc[idx, modified_data.columns.get_indexer(['volume'])] *= volume_factor
    
    return modified_data 