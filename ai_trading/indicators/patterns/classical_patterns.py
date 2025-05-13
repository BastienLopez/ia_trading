"""
Module d'implémentation de la détection de patterns chartistes classiques.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd


class PatternType(Enum):
    """Types de patterns chartistes classiques."""

    HEAD_AND_SHOULDERS = "tête_et_épaules"
    INVERSE_HEAD_AND_SHOULDERS = "tête_et_épaules_inversé"
    DOUBLE_TOP = "double_sommet"
    DOUBLE_BOTTOM = "double_plancher"
    TRIPLE_TOP = "triple_sommet"
    TRIPLE_BOTTOM = "triple_plancher"
    ASCENDING_TRIANGLE = "triangle_ascendant"
    DESCENDING_TRIANGLE = "triangle_descendant"
    SYMMETRICAL_TRIANGLE = "triangle_symétrique"
    RISING_WEDGE = "coin_montant"
    FALLING_WEDGE = "coin_descendant"
    RECTANGLE = "rectangle"
    CUP_AND_HANDLE = "tasse_et_anse"
    INVERSE_CUP_AND_HANDLE = "tasse_et_anse_inversée"
    FLAG = "drapeau"
    PENNANT = "fanion"


class PatternDetection:
    """Classe représentant un pattern détecté avec ses caractéristiques."""

    def __init__(
        self,
        pattern_type: PatternType,
        start_idx: int,
        end_idx: int,
        reliability: float,
        price_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ):
        self.pattern_type = pattern_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.reliability = reliability
        self.price_target = price_target
        self.stop_loss = stop_loss

    def __repr__(self):
        return f"PatternDetection({self.pattern_type.value}, {self.start_idx}-{self.end_idx}, reliability={self.reliability:.2f})"


def reliability_score(
    pattern: PatternDetection,
    volume_confirmation: bool = False,
    momentum_confirmation: bool = False,
    timeframe_alignment: bool = False,
) -> float:
    """
    Calcule un score de fiabilité amélioré pour un pattern détecté.

    Args:
        pattern: Le pattern détecté
        volume_confirmation: Si True, le volume confirme le pattern
        momentum_confirmation: Si True, le momentum confirme le pattern
        timeframe_alignment: Si True, le pattern est aligné sur plusieurs timeframes

    Returns:
        Score de fiabilité entre 0 et 1
    """
    # Score de base
    base_score = pattern.reliability

    # Ajustements
    if volume_confirmation:
        base_score += 0.15

    if momentum_confirmation:
        base_score += 0.1

    if timeframe_alignment:
        base_score += 0.2

    # Limiter à 1.0
    return min(base_score, 1.0)


class ClassicalPatternDetector:
    """
    Détecteur de patterns chartistes basé sur des techniques d'analyse technique classiques.
    """

    def __init__(self, min_pattern_points: int = 10, detection_threshold: float = 0.6):
        """
        Initialise le détecteur de patterns classiques.

        Args:
            min_pattern_points: Nombre minimum de points pour un pattern valide
            detection_threshold: Seuil minimal pour considérer un pattern détecté
        """
        self.min_pattern_points = min_pattern_points
        self.detection_threshold = detection_threshold

    def _find_peaks_and_troughs(
        self, data: pd.Series, window_size: int = 5
    ) -> Tuple[List[int], List[int]]:
        """
        Trouve les sommets et creux dans une série temporelle.

        Args:
            data: Série temporelle (généralement prix de clôture)
            window_size: Taille de la fenêtre pour la détection

        Returns:
            Tuple contenant (indices des sommets, indices des creux)
        """
        peaks = []
        troughs = []

        # Garantir que la fenêtre est impaire
        if window_size % 2 == 0:
            window_size += 1

        half_window = window_size // 2

        for i in range(half_window, len(data) - half_window):
            window = data.iloc[i - half_window : i + half_window + 1]
            center_value = data.iloc[i]

            if center_value == max(window):
                peaks.append(i)
            elif center_value == min(window):
                troughs.append(i)

        return peaks, troughs

    def detect_double_top(
        self, data: pd.DataFrame, tolerance: float = 0.02
    ) -> List[PatternDetection]:
        """
        Détecte les patterns de double sommet dans les données.

        Args:
            data: DataFrame avec au moins une colonne 'close'
            tolerance: Tolérance pour la correspondance des sommets

        Returns:
            Liste des patterns détectés
        """
        if "close" not in data.columns:
            return []

        # Trouver les sommets et creux
        peaks, troughs = self._find_peaks_and_troughs(data["close"])

        detected_patterns = []

        # Analyser les paires de sommets consécutifs
        for i in range(len(peaks) - 1):
            p1 = peaks[i]
            p2 = peaks[i + 1]

            # Vérifier que les sommets sont suffisamment éloignés
            if p2 - p1 < self.min_pattern_points:
                continue

            # Vérifier que les sommets sont approximativement à la même hauteur
            h1 = data["close"].iloc[p1]
            h2 = data["close"].iloc[p2]

            rel_diff = abs(h1 - h2) / max(h1, h2)

            if rel_diff > tolerance:
                continue

            # Vérifier qu'il y a au moins un creux entre les deux sommets
            valleys_between = [t for t in troughs if p1 < t < p2]
            if not valleys_between:
                continue

            # Trouver le creux le plus profond entre les sommets
            valley_idx = min(valleys_between, key=lambda t: data["close"].iloc[t])

            # Calculer la profondeur du creux (important pour le pattern)
            valley_depth = max(h1, h2) - data["close"].iloc[valley_idx]
            rel_depth = valley_depth / max(h1, h2)

            # Un bon double sommet a un creux significatif entre les sommets
            if rel_depth < 0.03:  # Moins de 3% de profondeur
                continue

            # Calculer la fiabilité
            reliability = min(0.5 + rel_depth * 2, 0.85)  # 0.56 à 0.85

            # Créer le pattern détecté
            pattern = PatternDetection(
                pattern_type=PatternType.DOUBLE_TOP,
                start_idx=p1 - self.min_pattern_points // 2,
                end_idx=p2 + self.min_pattern_points // 2,
                reliability=reliability,
                price_target=data["close"].iloc[valley_idx]
                - valley_depth,  # Projection vers le bas
                stop_loss=max(h1, h2) * 1.02,  # 2% au-dessus du sommet le plus haut
            )

            detected_patterns.append(pattern)

        return detected_patterns

    def detect_patterns(
        self, data: pd.DataFrame
    ) -> Dict[PatternType, List[PatternDetection]]:
        """
        Détecte tous les patterns supportés dans les données.

        Args:
            data: DataFrame avec colonnes OHLCV

        Returns:
            Dictionnaire des patterns détectés par type
        """
        # Pour l'instant, implementons seulement double top comme exemple
        patterns = {PatternType.DOUBLE_TOP: self.detect_double_top(data)}

        # Ajout d'implémentations pour les autres patterns
        # ...

        return patterns
