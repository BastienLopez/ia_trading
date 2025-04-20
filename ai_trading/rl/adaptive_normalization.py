import logging
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveNormalizer:
    """
    Normalise les données de manière adaptative en utilisant une fenêtre glissante.
    """

    def __init__(
        self, window_size=1000, method="minmax", clip_values=True, feature_names=None
    ):
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
        self.feature_windows = {
            feature: deque(maxlen=window_size) for feature in self.feature_names
        }

        # Statistiques pour chaque feature
        self.stats = {
            feature: {"min": None, "max": None, "mean": None, "std": None}
            for feature in self.feature_names
        }

        logger.info(
            f"Normalisateur adaptatif initialisé avec window_size={window_size}, "
            f"method={method}, clip_values={clip_values}"
        )

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
                    self.stats[feature]["min"] = np.min(values)
                    self.stats[feature]["max"] = np.max(values)
                    self.stats[feature]["mean"] = np.mean(values)
                    self.stats[feature]["std"] = np.std(values)

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
            if (
                feature in self.feature_names
                and value is not None
                and not np.isnan(value)
            ):
                if self.method == "minmax":
                    min_val = self.stats[feature]["min"]
                    max_val = self.stats[feature]["max"]

                    if min_val is None or max_val is None or max_val == min_val:
                        normalized_dict[feature] = 0.5
                    else:
                        normalized = (value - min_val) / (max_val - min_val)
                        if self.clip_values:
                            normalized = np.clip(normalized, 0, 1)
                        normalized_dict[feature] = normalized

                elif self.method == "zscore":
                    mean = self.stats[feature]["mean"]
                    std = self.stats[feature]["std"]

                    if mean is None or std is None or std == 0:
                        normalized_dict[feature] = 0
                    else:
                        normalized = (value - mean) / std
                        if self.clip_values:
                            normalized = np.clip(normalized, -3, 3)
                        normalized_dict[feature] = normalized
            else:
                normalized_dict[feature] = (
                    0  # Valeur par défaut pour les features inconnues
                )

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
            raise ValueError(
                f"Le tableau de features ({len(feature_array)}) et la liste de noms ({len(feature_names)}) doivent avoir la même longueur"
            )

        # Créer un dictionnaire de features
        feature_dict = {
            name: value for name, value in zip(feature_names, feature_array)
        }

        # Normaliser le dictionnaire
        normalized_dict = self.normalize(feature_dict)

        # Convertir en tableau
        normalized_array = np.array([normalized_dict[name] for name in feature_names])

        return normalized_array

    def normalize_features(self, features_df, method="adaptive", window_size=None):
        """
        Normalise les features de manière adaptative.

        Args:
            features_df (pd.DataFrame): DataFrame contenant les features à normaliser
            method (str): Méthode de normalisation ('minmax', 'zscore', 'robust', 'adaptive')
            window_size (int, optional): Taille de la fenêtre pour la normalisation adaptative

        Returns:
            pd.DataFrame: DataFrame avec les features normalisées
        """
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        normalized_df = features_df.copy()

        # Déterminer la taille de la fenêtre si non spécifiée
        if window_size is None:
            window_size = min(
                len(features_df) // 10, 30
            )  # 10% des données ou 30 jours max

        # Normaliser chaque colonne
        for col in normalized_df.columns:
            # Ignorer les colonnes non numériques
            if not np.issubdtype(normalized_df[col].dtype, np.number):
                continue

            if method == "minmax":
                normalized_df[col] = self.normalize_minmax(normalized_df[col])
            elif method == "zscore":
                normalized_df[col] = self.normalize_zscore(normalized_df[col])
            elif method == "robust":
                normalized_df[col] = self.normalize_robust(normalized_df[col])
            elif method == "adaptive":
                # Normalisation adaptative avec fenêtre glissante
                values = normalized_df[col].values
                normalized_values = np.zeros_like(values, dtype=float)

                # Créer un masque pour les valeurs NaN
                nan_mask = np.isnan(values)

                for i in range(len(values)):
                    if nan_mask[i]:
                        # Conserver les NaN
                        normalized_values[i] = np.nan
                        continue

                    start_idx = max(0, i - window_size)
                    # Filtrer les valeurs NaN dans la fenêtre
                    window_values = values[start_idx : i + 1]
                    valid_window = window_values[~np.isnan(window_values)]

                    if len(valid_window) > 1:
                        # Utiliser les statistiques de la fenêtre pour normaliser
                        window_min = np.min(valid_window)
                        window_max = np.max(valid_window)
                        window_range = window_max - window_min

                        if window_range > 0:
                            normalized_values[i] = (
                                values[i] - window_min
                            ) / window_range
                        else:
                            normalized_values[i] = (
                                0.5  # Valeur par défaut si pas de variation
                            )
                    else:
                        normalized_values[i] = 0.5  # Valeur par défaut pour le début

                normalized_df[col] = normalized_values

        return normalized_df

    def normalize_minmax(self, series):
        """
        Normalise une série avec la méthode min-max.

        Args:
            series (pd.Series): Série à normaliser

        Returns:
            pd.Series: Série normalisée entre 0 et 1
        """
        if series.empty:
            return series

        # Gérer les valeurs NaN
        valid_values = series.dropna()
        if valid_values.empty:
            return series

        min_val = valid_values.min()
        max_val = valid_values.max()

        if max_val == min_val:
            # Si toutes les valeurs sont identiques, retourner 0.5
            normalized = pd.Series(0.5, index=series.index)
            # Conserver les NaN
            normalized[series.isna()] = np.nan
            return normalized

        # Normaliser entre 0 et 1
        normalized = (series - min_val) / (max_val - min_val)
        return normalized

    def normalize_zscore(self, series):
        """
        Normalise une série avec la méthode z-score.

        Args:
            series (pd.Series): Série à normaliser

        Returns:
            pd.Series: Série normalisée avec moyenne 0 et écart-type 1
        """
        if series.empty:
            return series

        # Gérer les valeurs NaN
        valid_values = series.dropna()
        if valid_values.empty:
            return series

        mean_val = valid_values.mean()
        std_val = valid_values.std()

        if std_val == 0:
            # Si l'écart-type est nul, retourner 0
            normalized = pd.Series(0, index=series.index)
            # Conserver les NaN
            normalized[series.isna()] = np.nan
            return normalized

        # Normaliser avec z-score
        normalized = (series - mean_val) / std_val
        return normalized

    def normalize_robust(self, series):
        """
        Normalise une série avec une méthode robuste (médiane et IQR).

        Args:
            series (pd.Series): Série à normaliser

        Returns:
            pd.Series: Série normalisée de manière robuste
        """
        if series.empty:
            return series

        # Gérer les valeurs NaN
        valid_values = series.dropna()
        if valid_values.empty:
            return series

        median_val = valid_values.median()
        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            # Si l'IQR est nul, retourner 0.5
            normalized = pd.Series(0.5, index=series.index)
            # Conserver les NaN
            normalized[series.isna()] = np.nan
            return normalized

        # Normaliser avec médiane et IQR
        normalized = (series - median_val) / iqr
        return normalized
