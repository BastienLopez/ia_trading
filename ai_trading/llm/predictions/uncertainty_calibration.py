"""
Module pour la calibration des incertitudes dans les prédictions de marché.

Ce module fournit des outils pour quantifier et calibrer l'incertitude des
prédictions générées par les modèles LLM et hybrides.
"""

import logging
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss

from ai_trading.utils import setup_logger
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.prediction_model import PredictionModel
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor

# Configuration du logger
logger = setup_logger("uncertainty_calibration")

class UncertaintyCalibrator:
    """
    Classe pour calibrer l'incertitude des prédictions de marché.
    
    Cette classe fournit des méthodes pour quantifier l'incertitude des prédictions,
    calculer des intervalles de confiance, et valider les prédictions par validation croisée.
    """
    
    def __init__(self, prediction_model=None, market_data: Optional[pd.DataFrame] = None):
        """
        Initialise le calibrateur d'incertitude.
        
        Args:
            prediction_model: Modèle de prédiction (PredictionModel ou MultiHorizonPredictor)
            market_data: Données de marché historiques pour la calibration (optionnel)
        """
        self.prediction_model = prediction_model
        self.market_data = market_data
        self.calibration_results = {}
        self.confidence_intervals = {}
        self.outliers = {}
        self.calibration_bins = []
        
        # Vérifier le type de modèle pour adapter la calibration
        if isinstance(prediction_model, PredictionModel):
            self.model_type = "hybrid"
        elif isinstance(prediction_model, MultiHorizonPredictor):
            self.model_type = "multi_horizon"
        elif isinstance(prediction_model, MarketPredictor):
            self.model_type = "llm"
        else:
            self.model_type = "unknown"
            
        logger.info(f"UncertaintyCalibrator initialisé avec un modèle de type: {self.model_type}")
    
    def calculate_confidence_intervals(self, 
                                     prediction: Dict[str, Any], 
                                     confidence_level: float = 0.95,
                                     method: str = "bootstrap") -> Dict[str, Any]:
        """
        Calcule les intervalles de confiance pour une prédiction.
        
        Args:
            prediction: Prédiction à calibrer
            confidence_level: Niveau de confiance (0.95 = 95%)
            method: Méthode de calcul ("bootstrap", "parametric", "bayesian")
            
        Returns:
            Dictionnaire contenant les intervalles de confiance
        """
        logger.info("Calcul d'intervalle empirique (%s, %.1f%%)", method, confidence_level * 100)
        if not self.calibration_bins:
            return {"error": "Calibration hors échantillon requise avant intervalle", "prediction_id": prediction.get("id", "unknown")}
        confidence_value = float(prediction.get("confidence", 0.0))
        matching_bin = next((item for item in self.calibration_bins if item["lower"] <= confidence_value <= item["upper"]), None)
        if matching_bin is None:
            matching_bin = min(self.calibration_bins, key=lambda item: abs(item["mean_confidence"] - confidence_value))
        alpha = 1 - confidence_level
        successes, count = matching_bin["successes"], matching_bin["count"]
        lower_probability = float(stats.beta.ppf(alpha / 2, successes + 1, count - successes + 1))
        upper_probability = float(stats.beta.ppf(1 - alpha / 2, successes + 1, count - successes + 1))
        result = {
            "prediction_id": prediction.get("id", "unknown"),
            "asset": prediction.get("asset", "unknown"),
            "direction": prediction.get("direction", "unknown"),
            "confidence_level": confidence_level,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "calibration_samples": count,
            "lower_confidence": lower_probability,
            "upper_confidence": upper_probability,
        }
        result.update({"point_estimate": confidence_value, "lower_bound": lower_probability, "upper_bound": upper_probability,
                       "interval_width": upper_probability - lower_probability, "method": "empirical_beta_bin"})
        
        # Stocker le résultat
        self.confidence_intervals[prediction.get("id", "unknown")] = result
        
        return result
    
    def estimate_probability_distribution(self, 
                                        prediction: Dict[str, Any],
                                        num_points: int = 100) -> Dict[str, Any]:
        """
        Estime la distribution de probabilité pour une prédiction.
        
        Args:
            prediction: Prédiction à analyser
            num_points: Nombre de points pour la distribution
            
        Returns:
            Dictionnaire contenant la distribution estimée
        """
        logger.info(f"Estimation de la distribution de probabilité pour {prediction.get('asset', 'unknown')}")
        
        del num_points
        raw_probabilities = prediction.get("probabilities")
        if not isinstance(raw_probabilities, dict):
            return {
                "prediction_id": prediction.get("id", "unknown"),
                "error": "Probabilités de prédiction requises; aucune distribution simulée n'est produite",
            }
        values = np.asarray([
            raw_probabilities.get("bearish", 0.0),
            raw_probabilities.get("neutral", 0.0),
            raw_probabilities.get("bullish", 0.0),
        ], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            return {"prediction_id": prediction.get("id", "unknown"), "error": "Probabilités invalides"}
        values /= values.sum()
        p_bearish, p_neutral, p_bullish = values.tolist()
        x = np.asarray([-1.0, 0.0, 1.0])
        y = values
        mean = float(p_bullish - p_bearish)
        std_dev = float(np.sqrt(np.dot(values, (x - mean) ** 2)))
        
        result = {
            "prediction_id": prediction.get("id", "unknown"),
            "asset": prediction.get("asset", "unknown"),
            "x_values": x.tolist(),
            "y_values": y.tolist(),
            "mean": mean,
            "std_dev": std_dev,
            "probabilities": {
                "bearish": p_bearish,
                "neutral": p_neutral,
                "bullish": p_bullish
            },
            "most_likely_direction": "bearish" if p_bearish > max(p_neutral, p_bullish) else 
                                  "neutral" if p_neutral > max(p_bearish, p_bullish) else 
                                  "bullish",
            "entropy": float(stats.entropy(values, base=3)),
            "distribution_source": "model_probabilities",
        }
        
        return result
    
    def detect_outliers(self, 
                      predictions: List[Dict[str, Any]], 
                      method: str = "z_score",
                      threshold: float = 3.0) -> Dict[str, Any]:
        """
        Détecte les prédictions outliers dans un ensemble de prédictions.
        
        Args:
            predictions: Liste des prédictions à analyser
            method: Méthode de détection ("z_score", "iqr")
            threshold: Seuil pour la détection (3.0 pour z_score, 1.5 pour IQR)
            
        Returns:
            Dictionnaire contenant les résultats de la détection
        """
        logger.info(f"Détection des outliers avec la méthode {method}")
        
        # Convertir les prédictions en valeurs numériques
        direction_map = {"bearish": -1, "neutral": 0, "bullish": 1}
        
        # Créer un DataFrame avec les directions et confiances
        pred_data = []
        for pred in predictions:
            direction_value = direction_map.get(pred.get("direction", "neutral"), 0)
            
            try:
                confidence_value = float(pred.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence_value = 0.0
            
            pred_data.append({
                "id": pred.get("id", "unknown"),
                "asset": pred.get("asset", "unknown"),
                "direction_value": direction_value,
                "confidence_value": confidence_value,
                "timeframe": pred.get("timeframe", "unknown")
            })
        
        df = pd.DataFrame(pred_data)
        
        # Détecter les outliers
        outlier_indices = []
        scores = []
        
        if df.empty:
            return {"total_predictions": 0, "outliers_detected": 0, "outlier_percentage": 0.0,
                    "method": method, "threshold": threshold, "outliers": []}
        if method == "z_score":
            # Méthode du Z-score
            z_scores = stats.zscore(df["direction_value"], nan_policy="omit")
            z_scores = np.nan_to_num(z_scores)
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
            scores = z_scores
            
        elif method == "iqr":
            # Méthode de l'écart interquartile
            Q1 = df["direction_value"].quantile(0.25)
            Q3 = df["direction_value"].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = df[
                (df["direction_value"] < lower_bound) | 
                (df["direction_value"] > upper_bound)
            ].index.tolist()
            
            # Calculer les scores comme distance normalisée à la médiane
            median = df["direction_value"].median()
            mad = stats.median_abs_deviation(df["direction_value"])
            scores = (df["direction_value"] - median) / (mad if mad > 0 else 1)
            
        else:
            logger.warning(f"Méthode de détection d'outliers non reconnue: {method}")
            return {"error": f"Méthode non reconnue: {method}"}
        
        # Préparer le résultat
        outlier_preds = []
        for idx in outlier_indices:
            outlier_data = df.iloc[idx].to_dict()
            outlier_data["outlier_score"] = scores[idx] if hasattr(scores, "__getitem__") else scores.iloc[idx]
            outlier_preds.append(outlier_data)
        
        result = {
            "total_predictions": len(predictions),
            "outliers_detected": len(outlier_indices),
            "outlier_percentage": len(outlier_indices) / len(predictions) if predictions else 0,
            "method": method,
            "threshold": threshold,
            "outliers": outlier_preds
        }
        
        # Stocker le résultat
        self.outliers[f"{method}_{datetime.now().strftime('%Y%m%d%H%M%S')}"] = result
        
        return result
    
    def perform_cross_validation(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               n_splits: int = 5,
                               random_state: int = 42,
                               min_valid_folds: int = 1) -> Dict[str, Any]:
        """
        Effectue une validation croisée pour évaluer la calibration du modèle.
        
        Args:
            X: Données d'entrée
            y: Cibles
            n_splits: Nombre de plis pour la validation croisée
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            Dictionnaire contenant les résultats de validation croisée
        """
        logger.info(f"Validation walk-forward avec {n_splits} plis")
        del random_state  # Une validation temporelle ne mélange pas les observations.
        if self.prediction_model is None or not getattr(self.prediction_model, "ml_model", None):
            logger.warning("Pas de modèle ML disponible pour la validation croisée")
            return {"error": "Pas de modèle ML disponible"}
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        if len(X) != len(y) or len(X) <= n_splits:
            return {"error": "Données insuffisantes pour walk-forward"}
        splitter = TimeSeriesSplit(n_splits=n_splits)
        oos_probabilities, oos_targets, valid_folds = [], [], 0
        templates = getattr(self.prediction_model, "calibration_model_templates", None) or self.prediction_model.ml_model
        for train_index, test_index in splitter.split(X):
            fold_probabilities = []
            if len(np.unique(y[train_index])) < 2:
                continue
            for model in templates:
                try:
                    if getattr(self.prediction_model, "config", {}).get("require_temporal_calibration", False):
                        fitted = self.prediction_model._temporal_prefit_calibration(
                            clone(model), X[train_index], y[train_index]
                        )
                    else:
                        fitted = clone(model).fit(X[train_index], y[train_index])
                except (ValueError, TypeError):
                    continue
                raw = fitted.predict_proba(X[test_index])
                expanded = np.zeros((len(test_index), 3), dtype=float)
                for column, class_id in enumerate(fitted.classes_):
                    if int(class_id) in (0, 1, 2):
                        expanded[:, int(class_id)] = raw[:, column]
                fold_probabilities.append(expanded)
            if not fold_probabilities:
                continue
            probabilities = np.mean(fold_probabilities, axis=0)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            oos_probabilities.append(probabilities)
            oos_targets.append(y[test_index])
            valid_folds += 1
        if valid_folds < min_valid_folds:
            return {"error": f"Plis walk-forward calibrés insuffisants: {valid_folds}/{min_valid_folds}"}
        probabilities = np.vstack(oos_probabilities)
        targets = np.concatenate(oos_targets)
        result = self.fit_calibration(probabilities, targets)
        result.update({"n_splits": n_splits, "valid_folds": valid_folds,
                       "validation_type": "walk_forward", "timestamp": datetime.now().isoformat()})
        
        # Stocker le résultat
        self.calibration_results["cross_validation"] = result
        
        return result

    @staticmethod
    def _normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("probabilités invalides")
        row_sums = values.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("distribution vide")
        return values / row_sums

    def _interval_width(self, confidence: float, confidence_level: float = 0.95) -> float:
        if not self.calibration_bins:
            return float("nan")
        selected = min(self.calibration_bins, key=lambda item: abs(item["mean_confidence"] - confidence))
        alpha = 1 - confidence_level
        successes, count = selected["successes"], selected["count"]
        return float(stats.beta.ppf(1 - alpha / 2, successes + 1, count - successes + 1)
                     - stats.beta.ppf(alpha / 2, successes + 1, count - successes + 1))

    def _metrics(self, probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 10,
                 abstention_threshold: float = 0.45) -> Dict[str, Any]:
        probabilities = self._normalise_probabilities(probabilities)
        y_true = np.asarray(y_true, dtype=int)
        if len(probabilities) != len(y_true) or (y_true < 0).any() or (y_true >= probabilities.shape[1]).any():
            raise ValueError("probabilités et labels invalides")
        predicted = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correct = (predicted == y_true).astype(int)
        bins = []
        for lower, upper in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
            mask = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
            if mask.any():
                bins.append({"lower": float(lower), "upper": float(upper), "count": int(mask.sum()),
                             "mean_confidence": float(confidence[mask].mean()), "empirical_accuracy": float(correct[mask].mean()),
                             "successes": int(correct[mask].sum())})
        one_hot = np.eye(probabilities.shape[1])[y_true]
        brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        ece = float(sum(abs(item["mean_confidence"] - item["empirical_accuracy"]) * item["count"] for item in bins) / len(y_true))
        return {"samples": int(len(y_true)), "brier_score": brier, "ece": ece, "coverage": float(correct.mean()),
                "abstention_threshold": float(abstention_threshold),
                "abstention_rate": float(np.mean(confidence < abstention_threshold)),
                "calibration_bins": bins, "probabilities": probabilities.tolist(), "targets": y_true.tolist()}

    def fit_calibration(self, probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 10,
                        abstention_threshold: float = 0.45) -> Dict[str, Any]:
        """Ajuste la confiance sur des prédictions réellement hors échantillon."""
        result = self._metrics(probabilities, y_true, n_bins, abstention_threshold)
        self.calibration_bins = result["calibration_bins"]
        widths = [self._interval_width(float(max(row))) for row in probabilities]
        result["mean_interval_width"] = float(np.nanmean(widths))
        self.calibration_results["cross_validation"] = result
        return result

    def evaluate_validation_and_test(self, validation_probabilities: np.ndarray, validation_targets: np.ndarray,
                                     test_probabilities: np.ndarray, test_targets: np.ndarray,
                                     abstention_threshold: float = 0.45) -> Dict[str, Any]:
        """Calibre sur validation OOS puis rapporte séparément le test final."""
        validation = self.fit_calibration(validation_probabilities, validation_targets,
                                          abstention_threshold=abstention_threshold)
        test = self._metrics(test_probabilities, test_targets, abstention_threshold=abstention_threshold)
        widths = [self._interval_width(float(max(row))) for row in self._normalise_probabilities(test_probabilities)]
        test["mean_interval_width"] = float(np.nanmean(widths))
        report = {"validation": validation, "test": test, "calibration_source": "validation_oos"}
        self.calibration_results["validation_test"] = report
        return report
    
    def plot_calibration_curve(self, save_path: Optional[str] = None):
        """
        Trace la courbe de calibration.
        
        Args:
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        if "cross_validation" not in self.calibration_results:
            logger.warning("Pas de résultats de validation croisée disponibles")
            return
        
        cv_results = self.calibration_results["cross_validation"]
        
        plt.figure(figsize=(10, 8))
        
        bins = cv_results.get("calibration_bins", [])
        if not bins:
            logger.warning("Aucun bin de calibration disponible")
            return
        prob_pred = [item["mean_confidence"] for item in bins]
        prob_true = [item["empirical_accuracy"] for item in bins]
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Courbe de calibration')
        
        # Ligne de référence (calibration parfaite)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibration parfaite')
        
        plt.xlabel('Probabilité prédite')
        plt.ylabel('Fréquence empirique')
        plt.title(f'Courbe de calibration (ECE: {cv_results.get("ece", 0.0):.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Courbe de calibration sauvegardée: {save_path}")
        else:
            plt.show()
    
    def plot_probability_distribution(self, 
                                    prediction: Dict[str, Any],
                                    distribution: Optional[Dict[str, Any]] = None,
                                    save_path: Optional[str] = None):
        """
        Trace la distribution de probabilité d'une prédiction.
        
        Args:
            prediction: Prédiction à visualiser
            distribution: Distribution précalculée (optionnel)
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        # Calculer la distribution si non fournie
        if distribution is None:
            distribution = self.estimate_probability_distribution(prediction)
        
        plt.figure(figsize=(12, 8))
        
        # Distribution continue
        plt.plot(distribution["x_values"], distribution["y_values"], 
                 linewidth=2, color='blue', label='Distribution de probabilité')
        
        # Marquer les zones
        plt.axvspan(-1.5, -0.5, alpha=0.2, color='red', label='Baissier')
        plt.axvspan(-0.5, 0.5, alpha=0.2, color='gray', label='Neutre')
        plt.axvspan(0.5, 1.5, alpha=0.2, color='green', label='Haussier')
        
        # Marquer la moyenne
        plt.axvline(distribution["mean"], color='navy', linestyle='--', 
                   label=f'Moyenne ({distribution["mean"]:.2f})')
        
        # Annotations des probabilités
        plt.annotate(f"P(Baissier) = {distribution['probabilities']['bearish']:.2f}", 
                    xy=(-1, np.max(distribution["y_values"]) * 0.9),
                    fontsize=12, color='darkred')
        
        plt.annotate(f"P(Neutre) = {distribution['probabilities']['neutral']:.2f}", 
                    xy=(0, np.max(distribution["y_values"]) * 0.8),
                    fontsize=12, color='darkgray')
        
        plt.annotate(f"P(Haussier) = {distribution['probabilities']['bullish']:.2f}", 
                    xy=(0.6, np.max(distribution["y_values"]) * 0.9),
                    fontsize=12, color='darkgreen')
        
        plt.xlabel('Direction')
        plt.ylabel('Densité de probabilité')
        plt.title(f'Distribution de probabilité pour {prediction.get("asset", "unknown")}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Distribution de probabilité sauvegardée: {save_path}")
        else:
            plt.show()
    
    def calibrate_prediction(self, 
                          prediction: Dict[str, Any],
                          calibration_method: str = "platt") -> Dict[str, Any]:
        """
        Calibre la confiance d'une prédiction.
        
        Args:
            prediction: Prédiction à calibrer
            calibration_method: Méthode de calibration ("platt", "isotonic")
            
        Returns:
            Prédiction avec confiance calibrée
        """
        logger.info(f"Calibration de prédiction avec méthode {calibration_method}")
        
        try:
            confidence_value = float(prediction.get("confidence", 0.0))
        except (TypeError, ValueError):
            return {**prediction, "calibration_error": "Confiance numérique requise"}
        
        if not self.calibration_bins:
            return {**prediction, "calibration_error": "Calibration hors échantillon requise"}
        requested_method = calibration_method
        selected_bin = min(self.calibration_bins, key=lambda item: abs(item["mean_confidence"] - confidence_value))
        calibrated_confidence = float(selected_bin["empirical_accuracy"])
        
        # Convertir la confiance calibrée en catégorie qualitative
        if calibrated_confidence < 0.4:
            calibrated_confidence_label = "low"
        elif calibrated_confidence < 0.7:
            calibrated_confidence_label = "medium"
        else:
            calibrated_confidence_label = "high"
        
        # Créer la prédiction calibrée
        calibrated_prediction = prediction.copy()
        calibrated_prediction.update({
            "original_confidence": confidence_value,
            "confidence": calibrated_confidence,
            "confidence_label": calibrated_confidence_label,
            "calibration_method": f"empirical_bin ({requested_method})"
        })
        
        return calibrated_prediction

# Exemple d'utilisation:
"""
calibrator = UncertaintyCalibrator(prediction_model)
confidence_intervals = calibrator.calculate_confidence_intervals(prediction)
distribution = calibrator.estimate_probability_distribution(prediction)
calibrated_prediction = calibrator.calibrate_prediction(prediction)
"""
