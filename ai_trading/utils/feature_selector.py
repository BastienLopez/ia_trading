import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    """Classe pour la sélection automatique des caractéristiques importantes pour le trading."""
    
    def __init__(self, n_features: int = 20):
        """
        Initialise le sélecteur de caractéristiques.
        
        Args:
            n_features (int): Nombre de caractéristiques à sélectionner
        """
        self.n_features = n_features
        self.selected_features = None
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def _mutual_information_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Sélectionne les caractéristiques basées sur l'information mutuelle.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, float]: Scores d'importance des caractéristiques
        """
        selector = SelectKBest(score_func=mutual_info_regression, k='all')
        selector.fit(X, y)
        return dict(zip(X.columns, selector.scores_))

    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calcule l'importance des caractéristiques avec Random Forest.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, float]: Scores d'importance des caractéristiques
        """
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        return dict(zip(X.columns, rf.feature_importances_))

    def select_features(self, data: pd.DataFrame, target_col: str) -> List[str]:
        """
        Sélectionne les caractéristiques les plus importantes.
        
        Args:
            data (pd.DataFrame): DataFrame contenant toutes les caractéristiques
            target_col (str): Nom de la colonne cible
            
        Returns:
            List[str]: Liste des caractéristiques sélectionnées
        """
        logger.info("Début de la sélection des caractéristiques...")
        
        # Préparation des données
        y = data[target_col]
        X = data.drop(columns=[target_col])
        
        # Normalisation des données
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Calcul des scores d'importance avec différentes méthodes
        mi_scores = self._mutual_information_selection(X_scaled, y)
        rf_scores = self._random_forest_importance(X_scaled, y)
        
        # Combinaison des scores (moyenne normalisée)
        combined_scores = {}
        for feature in X.columns:
            combined_scores[feature] = (
                mi_scores[feature] / max(mi_scores.values()) +
                rf_scores[feature] / max(rf_scores.values())
            ) / 2
            
        # Sélection des meilleures caractéristiques
        self.feature_importance = combined_scores
        selected_features = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.n_features]
        
        self.selected_features = [feature for feature, _ in selected_features]
        
        logger.info(f"Sélection terminée. {len(self.selected_features)} caractéristiques sélectionnées.")
        return self.selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retourne les scores d'importance des caractéristiques.
        
        Returns:
            Dict[str, float]: Dictionnaire des scores d'importance
        """
        if not self.feature_importance:
            raise ValueError("Vous devez d'abord appeler select_features()")
        return self.feature_importance
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les données en ne gardant que les caractéristiques sélectionnées.
        
        Args:
            data (pd.DataFrame): Données à transformer
            
        Returns:
            pd.DataFrame: Données transformées
        """
        if not self.selected_features:
            raise ValueError("Vous devez d'abord appeler select_features()")
        return data[self.selected_features] 