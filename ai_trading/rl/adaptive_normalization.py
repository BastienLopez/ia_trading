import numpy as np
import pandas as pd
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AdaptiveNormalizer:
    """
    Normalise les données de manière adaptative en utilisant une fenêtre glissante.
    """
    
    def __init__(self, window_size=1000, method='minmax', clip_values=True, feature_names=None):
        """
        Initialise le normalisateur adaptatif.
        
        Args:
            window_size (int): Taille de la fenêtre glissante pour la normalisation
            method (str): Méthode de normalisation ('minmax' ou 'zscore')
            clip_values (bool): Si True, limite les valeurs normalisées entre 0 et 1 (minmax) ou -3 et 3 (zscore)
            feature_names (list): Liste des noms de features à normaliser
        """
        self.window_size = window_size
        self.method = method
        self.clip_values = clip_values
        self.feature_names = feature_names or []
        
        # Initialiser les fenêtres de données pour chaque feature
        self.feature_windows = {feature: deque(maxlen=window_size) for feature in self.feature_names}
        
        # Statistiques pour chaque feature
        self.stats = {feature: {'min': None, 'max': None, 'mean': None, 'std': None} 
                     for feature in self.feature_names}
        
        logger.info(f"Normalisateur adaptatif initialisé avec window_size={window_size}, "
                   f"method={method}, clip_values={clip_values}")
    
    def update(self, feature_dict):
        """
        Met à jour les statistiques avec de nouvelles valeurs.
        
        Args:
            feature_dict (dict): Dictionnaire de features et leurs valeurs
        """
        for feature, value in feature_dict.items():
            if feature in self.feature_names:
                # Ajouter la valeur à la fenêtre
                if value is not None and not np.isnan(value):
                    self.feature_windows[feature].append(value)
                
                # Mettre à jour les statistiques
                if self.feature_windows[feature]:
                    values = np.array(self.feature_windows[feature])
                    self.stats[feature]['min'] = np.min(values)
                    self.stats[feature]['max'] = np.max(values)
                    self.stats[feature]['mean'] = np.mean(values)
                    self.stats[feature]['std'] = np.std(values)
    
    def normalize(self, feature_dict):
        """
        Normalise un dictionnaire de features.
        
        Args:
            feature_dict (dict): Dictionnaire de features et leurs valeurs
            
        Returns:
            dict: Dictionnaire de features normalisées
        """
        normalized_dict = {}
        
        for feature, value in feature_dict.items():
            if feature in self.feature_names and value is not None and not np.isnan(value):
                if self.method == 'minmax':
                    min_val = self.stats[feature]['min']
                    max_val = self.stats[feature]['max']
                    
                    if min_val is None or max_val is None or max_val == min_val:
                        normalized_dict[feature] = 0.5
                    else:
                        normalized = (value - min_val) / (max_val - min_val)
                        if self.clip_values:
                            normalized = np.clip(normalized, 0, 1)
                        normalized_dict[feature] = normalized
                
                elif self.method == 'zscore':
                    mean = self.stats[feature]['mean']
                    std = self.stats[feature]['std']
                    
                    if mean is None or std is None or std == 0:
                        normalized_dict[feature] = 0
                    else:
                        normalized = (value - mean) / std
                        if self.clip_values:
                            normalized = np.clip(normalized, -3, 3)
                        normalized_dict[feature] = normalized
            else:
                normalized_dict[feature] = 0  # Valeur par défaut pour les features inconnues
        
        return normalized_dict
    
    def normalize_array(self, feature_array, feature_names=None):
        """
        Normalise un tableau de features.
        
        Args:
            feature_array (np.ndarray): Tableau de features à normaliser
            feature_names (list): Liste des noms de features correspondant au tableau
            
        Returns:
            np.ndarray: Tableau normalisé
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if len(feature_array) != len(feature_names):
            raise ValueError(f"Le tableau de features ({len(feature_array)}) et la liste de noms ({len(feature_names)}) doivent avoir la même longueur")
        
        # Créer un dictionnaire de features
        feature_dict = {name: value for name, value in zip(feature_names, feature_array)}
        
        # Normaliser le dictionnaire
        normalized_dict = self.normalize(feature_dict)
        
        # Convertir en tableau
        normalized_array = np.array([normalized_dict[name] for name in feature_names])
        
        return normalized_array 