"""
Module pour la calibration des incertitudes dans les prédictions de marché.

Ce module fournit des outils pour quantifier et calibrer l'incertitude des
prédictions générées par les modèles LLM et hybrides.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold, cross_val_predict

from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor
from ai_trading.llm.predictions.prediction_model import PredictionModel
from ai_trading.utils import setup_logger

# Configuration du logger
logger = setup_logger("uncertainty_calibration")


class UncertaintyCalibrator:
    """
    Classe pour calibrer l'incertitude des prédictions de marché.

    Cette classe fournit des méthodes pour quantifier l'incertitude des prédictions,
    calculer des intervalles de confiance, et valider les prédictions par validation croisée.
    """

    def __init__(
        self, prediction_model=None, market_data: Optional[pd.DataFrame] = None
    ):
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

        # Vérifier le type de modèle pour adapter la calibration
        if isinstance(prediction_model, PredictionModel):
            self.model_type = "hybrid"
        elif isinstance(prediction_model, MultiHorizonPredictor):
            self.model_type = "multi_horizon"
        elif isinstance(prediction_model, MarketPredictor):
            self.model_type = "llm"
        else:
            self.model_type = "unknown"

        logger.info(
            f"UncertaintyCalibrator initialisé avec un modèle de type: {self.model_type}"
        )

    def calculate_confidence_intervals(
        self,
        prediction: Dict[str, Any],
        confidence_level: float = 0.95,
        method: str = "bootstrap",
    ) -> Dict[str, Any]:
        """
        Calcule les intervalles de confiance pour une prédiction.

        Args:
            prediction: Prédiction à calibrer
            confidence_level: Niveau de confiance (0.95 = 95%)
            method: Méthode de calcul ("bootstrap", "parametric", "bayesian")

        Returns:
            Dictionnaire contenant les intervalles de confiance
        """
        logger.info(
            f"Calcul des intervalles de confiance ({method}, {confidence_level*100}%)"
        )

        result = {
            "prediction_id": prediction.get("id", "unknown"),
            "asset": prediction.get("asset", "unknown"),
            "direction": prediction.get("direction", "unknown"),
            "confidence_level": confidence_level,
            "method": method,
            "timestamp": datetime.now().isoformat(),
        }

        # Direction qualitative à convertir en valeur numérique pour l'intervalle
        direction_map = {"bearish": -1, "neutral": 0, "bullish": 1}
        direction_value = direction_map.get(prediction.get("direction", "neutral"), 0)

        # Confiance qualitative à convertir en valeur numérique
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence_value = confidence_map.get(
            prediction.get("confidence", "medium"), 0.5
        )

        # Méthodes de calcul d'intervalle de confiance
        if method == "bootstrap":
            interval = self._bootstrap_confidence_interval(
                direction_value, confidence_value, confidence_level
            )
        elif method == "parametric":
            interval = self._parametric_confidence_interval(
                direction_value, confidence_value, confidence_level
            )
        elif method == "bayesian":
            interval = self._bayesian_confidence_interval(
                direction_value, confidence_value, confidence_level
            )
        else:
            # Méthode par défaut: approche paramétrique simple
            interval = self._parametric_confidence_interval(
                direction_value, confidence_value, confidence_level
            )

        # Conversion des valeurs numériques en directions qualitatives
        inverse_direction_map = {-1: "bearish", 0: "neutral", 1: "bullish"}

        # Arrondir aux directions discrètes les plus proches
        lower_dir = inverse_direction_map[max(-1, min(1, round(interval[0])))]
        upper_dir = inverse_direction_map[max(-1, min(1, round(interval[1])))]

        result.update(
            {
                "point_estimate": direction_value,
                "lower_bound": interval[0],
                "upper_bound": interval[1],
                "lower_direction": lower_dir,
                "upper_direction": upper_dir,
                "interval_width": interval[1] - interval[0],
            }
        )

        # Stocker le résultat
        self.confidence_intervals[prediction.get("id", "unknown")] = result

        return result

    def _bootstrap_confidence_interval(
        self, point_estimate: float, uncertainty: float, confidence_level: float
    ) -> Tuple[float, float]:
        """
        Calcule un intervalle de confiance par bootstrap.

        Args:
            point_estimate: Estimation ponctuelle
            uncertainty: Incertitude estimée
            confidence_level: Niveau de confiance

        Returns:
            Tuple (borne inférieure, borne supérieure)
        """
        # Simuler des prédictions par bootstrap
        n_bootstrap = 1000
        noise_scale = 0.5 * (1 - uncertainty)  # Plus d'incertitude = plus de bruit

        bootstrap_samples = np.random.normal(point_estimate, noise_scale, n_bootstrap)

        # Calculer les quantiles pour l'intervalle de confiance
        alpha = 1 - confidence_level
        lower_bound = np.quantile(bootstrap_samples, alpha / 2)
        upper_bound = np.quantile(bootstrap_samples, 1 - alpha / 2)

        return (lower_bound, upper_bound)

    def _parametric_confidence_interval(
        self, point_estimate: float, uncertainty: float, confidence_level: float
    ) -> Tuple[float, float]:
        """
        Calcule un intervalle de confiance paramétrique.

        Args:
            point_estimate: Estimation ponctuelle
            uncertainty: Incertitude estimée
            confidence_level: Niveau de confiance

        Returns:
            Tuple (borne inférieure, borne supérieure)
        """
        # Convertir l'incertitude en écart-type
        std_dev = 0.5 * (1 - uncertainty)

        # Calculer l'intervalle de confiance paramétrique
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin = z_score * std_dev

        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        return (lower_bound, upper_bound)

    def _bayesian_confidence_interval(
        self, point_estimate: float, uncertainty: float, confidence_level: float
    ) -> Tuple[float, float]:
        """
        Calcule un intervalle de confiance bayésien.

        Args:
            point_estimate: Estimation ponctuelle
            uncertainty: Incertitude estimée
            confidence_level: Niveau de confiance

        Returns:
            Tuple (borne inférieure, borne supérieure)
        """
        # Approche simplifiée d'un intervalle bayésien
        # Dans une implémentation réelle, on utiliserait PyMC ou un autre framework bayésien

        # Simuler une distribution a posteriori
        n_samples = 2000
        prior_mean = 0  # Neutre
        prior_std = 1.0

        # Force du prior vs. données (plus d'incertitude = plus de poids au prior)
        prior_weight = 0.5 * (1 - uncertainty)
        data_weight = 1 - prior_weight

        # Moyenne et écart-type de la distribution a posteriori
        posterior_mean = (prior_weight * prior_mean + data_weight * point_estimate) / (
            prior_weight + data_weight
        )
        posterior_std = np.sqrt(1 / (1 / prior_std**2 + 1 / (0.5**2 / data_weight)))

        # Générer des échantillons de la distribution a posteriori
        posterior_samples = np.random.normal(posterior_mean, posterior_std, n_samples)

        # Calculer les quantiles pour l'intervalle de confiance
        alpha = 1 - confidence_level
        lower_bound = np.quantile(posterior_samples, alpha / 2)
        upper_bound = np.quantile(posterior_samples, 1 - alpha / 2)

        return (lower_bound, upper_bound)

    def estimate_probability_distribution(
        self, prediction: Dict[str, Any], num_points: int = 100
    ) -> Dict[str, Any]:
        """
        Estime la distribution de probabilité pour une prédiction.

        Args:
            prediction: Prédiction à analyser
            num_points: Nombre de points pour la distribution

        Returns:
            Dictionnaire contenant la distribution estimée
        """
        logger.info(
            f"Estimation de la distribution de probabilité pour {prediction.get('asset', 'unknown')}"
        )

        # Direction qualitative à convertir en valeur numérique
        direction_map = {"bearish": -1, "neutral": 0, "bullish": 1}
        direction_value = direction_map.get(prediction.get("direction", "neutral"), 0)

        # Confiance qualitative à convertir en valeur numérique
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence_value = confidence_map.get(
            prediction.get("confidence", "medium"), 0.5
        )

        # Paramètres de la distribution
        mean = direction_value
        std_dev = 0.5 * (1 - confidence_value)  # Plus de confiance = moins de variance

        # Générer la distribution
        x = np.linspace(-1.5, 1.5, num_points)
        y = stats.norm.pdf(x, mean, std_dev)

        # Calculer les probabilités pour chaque direction
        p_bearish = stats.norm.cdf(-0.5, mean, std_dev)
        p_neutral = stats.norm.cdf(0.5, mean, std_dev) - stats.norm.cdf(
            -0.5, mean, std_dev
        )
        p_bullish = 1 - stats.norm.cdf(0.5, mean, std_dev)

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
                "bullish": p_bullish,
            },
            "most_likely_direction": (
                "bearish"
                if p_bearish > max(p_neutral, p_bullish)
                else "neutral" if p_neutral > max(p_bearish, p_bullish) else "bullish"
            ),
            "entropy": stats.entropy(
                [p_bearish, p_neutral, p_bullish], base=3
            ),  # Entropie normalisée (0-1)
        }

        return result

    def detect_outliers(
        self,
        predictions: List[Dict[str, Any]],
        method: str = "z_score",
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
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

            confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
            confidence_value = confidence_map.get(pred.get("confidence", "medium"), 0.5)

            pred_data.append(
                {
                    "id": pred.get("id", "unknown"),
                    "asset": pred.get("asset", "unknown"),
                    "direction_value": direction_value,
                    "confidence_value": confidence_value,
                    "timeframe": pred.get("timeframe", "unknown"),
                }
            )

        df = pd.DataFrame(pred_data)

        # Détecter les outliers
        outlier_indices = []
        scores = []

        if method == "z_score":
            # Méthode du Z-score
            z_scores = stats.zscore(df["direction_value"])
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
                (df["direction_value"] < lower_bound)
                | (df["direction_value"] > upper_bound)
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
            outlier_data["outlier_score"] = (
                scores[idx] if hasattr(scores, "__getitem__") else scores.iloc[idx]
            )
            outlier_preds.append(outlier_data)

        result = {
            "total_predictions": len(predictions),
            "outliers_detected": len(outlier_indices),
            "outlier_percentage": (
                len(outlier_indices) / len(predictions) if predictions else 0
            ),
            "method": method,
            "threshold": threshold,
            "outliers": outlier_preds,
        }

        # Stocker le résultat
        self.outliers[f"{method}_{datetime.now().strftime('%Y%m%d%H%M%S')}"] = result

        return result

    def perform_cross_validation(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42
    ) -> Dict[str, Any]:
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
        logger.info(f"Validation croisée avec {n_splits} plis")

        if (
            not hasattr(self.prediction_model, "ml_model")
            or self.prediction_model is None
        ):
            logger.warning("Pas de modèle ML disponible pour la validation croisée")
            return {"error": "Pas de modèle ML disponible"}

        model = self.prediction_model.ml_model

        # Configurer la validation croisée
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Prédictions par validation croisée
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

        # Calculer les métriques de calibration
        brier = brier_score_loss(y, y_pred_proba[:, 1])

        # Courbe de calibration
        prob_true, prob_pred = calibration_curve(y, y_pred_proba[:, 1], n_bins=10)

        # Calculer l'erreur de calibration moyenne
        calibration_error = np.mean(np.abs(prob_true - prob_pred))

        result = {
            "n_splits": n_splits,
            "brier_score": brier,
            "calibration_error": calibration_error,
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "timestamp": datetime.now().isoformat(),
        }

        # Stocker le résultat
        self.calibration_results["cross_validation"] = result

        return result

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

        # Courbe de calibration
        plt.plot(
            cv_results["prob_pred"],
            cv_results["prob_true"],
            marker="o",
            linewidth=2,
            label="Courbe de calibration",
        )

        # Ligne de référence (calibration parfaite)
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Calibration parfaite"
        )

        plt.xlabel("Probabilité prédite")
        plt.ylabel("Fréquence empirique")
        plt.title(
            f'Courbe de calibration (Erreur: {cv_results["calibration_error"]:.4f})'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Courbe de calibration sauvegardée: {save_path}")
        else:
            plt.show()

    def plot_probability_distribution(
        self,
        prediction: Dict[str, Any],
        distribution: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ):
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
        plt.plot(
            distribution["x_values"],
            distribution["y_values"],
            linewidth=2,
            color="blue",
            label="Distribution de probabilité",
        )

        # Marquer les zones
        plt.axvspan(-1.5, -0.5, alpha=0.2, color="red", label="Baissier")
        plt.axvspan(-0.5, 0.5, alpha=0.2, color="gray", label="Neutre")
        plt.axvspan(0.5, 1.5, alpha=0.2, color="green", label="Haussier")

        # Marquer la moyenne
        plt.axvline(
            distribution["mean"],
            color="navy",
            linestyle="--",
            label=f'Moyenne ({distribution["mean"]:.2f})',
        )

        # Annotations des probabilités
        plt.annotate(
            f"P(Baissier) = {distribution['probabilities']['bearish']:.2f}",
            xy=(-1, np.max(distribution["y_values"]) * 0.9),
            fontsize=12,
            color="darkred",
        )

        plt.annotate(
            f"P(Neutre) = {distribution['probabilities']['neutral']:.2f}",
            xy=(0, np.max(distribution["y_values"]) * 0.8),
            fontsize=12,
            color="darkgray",
        )

        plt.annotate(
            f"P(Haussier) = {distribution['probabilities']['bullish']:.2f}",
            xy=(0.6, np.max(distribution["y_values"]) * 0.9),
            fontsize=12,
            color="darkgreen",
        )

        plt.xlabel("Direction")
        plt.ylabel("Densité de probabilité")
        plt.title(
            f'Distribution de probabilité pour {prediction.get("asset", "unknown")}'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Distribution de probabilité sauvegardée: {save_path}")
        else:
            plt.show()

    def calibrate_prediction(
        self, prediction: Dict[str, Any], calibration_method: str = "platt"
    ) -> Dict[str, Any]:
        """
        Calibre la confiance d'une prédiction.

        Args:
            prediction: Prédiction à calibrer
            calibration_method: Méthode de calibration ("platt", "isotonic")

        Returns:
            Prédiction avec confiance calibrée
        """
        logger.info(f"Calibration de prédiction avec méthode {calibration_method}")

        # Confiance qualitative à convertir en valeur numérique
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence_value = confidence_map.get(
            prediction.get("confidence", "medium"), 0.5
        )

        # Direction qualitative à convertir en valeur numérique
        direction_map = {"bearish": -1, "neutral": 0, "bullish": 1}
        direction_value = direction_map.get(prediction.get("direction", "neutral"), 0)

        # Calibration simplifiée (dans une implémentation réelle, on utiliserait des données historiques)
        calibrated_confidence = confidence_value

        if calibration_method == "platt":
            # Version simplifiée de la calibration de Platt
            # Dans une implémentation réelle, on calibrerait avec LogisticRegression
            # Ici, on simule un ajustement de la confiance
            if confidence_value > 0.7:
                calibrated_confidence = 0.7 + 0.3 * (
                    confidence_value - 0.7
                )  # Réduire les confiances trop élevées
            elif confidence_value < 0.4:
                calibrated_confidence = (
                    0.2 + 0.5 * confidence_value
                )  # Augmenter les confiances très basses

        elif calibration_method == "isotonic":
            # Version simplifiée de la calibration isotonique
            # Dans une implémentation réelle, on utiliserait IsotonicRegression
            # Ici, on simule un mappage non-paramétrique
            breakpoints = [0, 0.3, 0.6, 0.9, 1.0]
            calibrated_values = [0, 0.25, 0.5, 0.8, 1.0]

            # Trouver l'intervalle approprié
            for i in range(len(breakpoints) - 1):
                if breakpoints[i] <= confidence_value < breakpoints[i + 1]:
                    # Interpolation linéaire
                    t = (confidence_value - breakpoints[i]) / (
                        breakpoints[i + 1] - breakpoints[i]
                    )
                    calibrated_confidence = calibrated_values[i] + t * (
                        calibrated_values[i + 1] - calibrated_values[i]
                    )
                    break

        # Convertir la confiance calibrée en catégorie qualitative
        if calibrated_confidence < 0.4:
            calibrated_confidence_label = "low"
        elif calibrated_confidence < 0.7:
            calibrated_confidence_label = "medium"
        else:
            calibrated_confidence_label = "high"

        # Créer la prédiction calibrée
        calibrated_prediction = prediction.copy()
        calibrated_prediction.update(
            {
                "original_confidence": prediction.get("confidence", "medium"),
                "confidence": calibrated_confidence_label,
                "confidence_value": calibrated_confidence,
                "calibration_method": calibration_method,
            }
        )

        return calibrated_prediction


# Exemple d'utilisation:
"""
calibrator = UncertaintyCalibrator(prediction_model)
confidence_intervals = calibrator.calculate_confidence_intervals(prediction)
distribution = calibrator.estimate_probability_distribution(prediction)
calibrated_prediction = calibrator.calibrate_prediction(prediction)
"""
