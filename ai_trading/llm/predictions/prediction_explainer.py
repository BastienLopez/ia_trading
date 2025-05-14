"""
Module pour l'explicabilité et l'interprétabilité des prédictions de marché.

Ce module fournit des outils pour expliquer les prédictions générées par les modèles LLM
et les modèles hybrides, en utilisant des techniques comme SHAP et LIME.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor
from ai_trading.llm.predictions.prediction_model import PredictionModel
from ai_trading.utils import setup_logger

# Configuration du logger
logger = setup_logger("prediction_explainer")


class PredictionExplainer:
    """
    Classe pour expliquer et interpréter les prédictions de marché.

    Cette classe fournit des méthodes pour analyser les facteurs qui influencent
    les prédictions et générer des visualisations et des rapports explicatifs.
    """

    def __init__(
        self, prediction_model=None, market_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialise l'analyseur d'explicabilité des prédictions.

        Args:
            prediction_model: Modèle de prédiction (PredictionModel ou MultiHorizonPredictor)
            market_data: Données de marché pour l'analyse (optionnel)
        """
        self.prediction_model = prediction_model
        self.market_data = market_data
        self.explainers = {}
        self.explanations = {}

        # Vérifiez le type de modèle pour adapter l'explication
        if isinstance(prediction_model, PredictionModel):
            self.model_type = "hybrid"
        elif isinstance(prediction_model, MultiHorizonPredictor):
            self.model_type = "multi_horizon"
        elif isinstance(prediction_model, MarketPredictor):
            self.model_type = "llm"
        else:
            self.model_type = "unknown"

        logger.info(
            f"PredictionExplainer initialisé avec un modèle de type: {self.model_type}"
        )

    def explain_with_shap(
        self, prediction: Dict[str, Any], feature_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Génère une explication SHAP pour une prédiction.

        Args:
            prediction: Prédiction à expliquer
            feature_data: Données des features utilisées pour la prédiction (optionnel)

        Returns:
            Dictionnaire contenant les valeurs SHAP et les métadonnées
        """
        logger.info("Génération d'explication SHAP")

        # Si aucune donnée fournie, utiliser les données de marché de classe
        if feature_data is None:
            feature_data = self.market_data

        if feature_data is None or self.model_type == "llm":
            # Pour les modèles purement LLM, expliquer autrement
            return self._explain_llm_prediction(prediction)

        # Pour les modèles hybrides avec ML
        if self.model_type == "hybrid" and hasattr(self.prediction_model, "ml_model"):
            ml_model = self.prediction_model.ml_model

            # Prétraiter les données pour le modèle
            X = feature_data.select_dtypes(include=[np.number])

            # Initialiser l'explainer SHAP
            try:
                # Créer un explainer approprié selon le type de modèle
                explainer = shap.Explainer(ml_model, X)
                shap_values = explainer(X)

                # Préparer le résultat
                shap_explanation = {
                    "shap_values": shap_values.values,
                    "base_value": shap_values.base_values,
                    "feature_names": X.columns.tolist(),
                    "prediction_id": prediction.get("id", "unknown"),
                    "asset": prediction.get("asset", "unknown"),
                    "direction": prediction.get("direction", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                }

                # Stocker l'explication
                self.explanations["shap"] = shap_explanation
                self.explainers["shap"] = explainer

                return shap_explanation

            except Exception as e:
                logger.error(f"Erreur lors de la génération de l'explication SHAP: {e}")
                return {"error": str(e), "message": "Échec de l'explication SHAP"}

        return {"error": "Modèle non compatible avec SHAP"}

    def explain_with_lime(
        self, prediction: Dict[str, Any], feature_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Génère une explication LIME pour une prédiction.

        Args:
            prediction: Prédiction à expliquer
            feature_data: Données des features utilisées pour la prédiction (optionnel)

        Returns:
            Dictionnaire contenant les explications LIME et les métadonnées
        """
        logger.info("Génération d'explication LIME")

        # Si aucune donnée fournie, utiliser les données de marché de classe
        if feature_data is None:
            feature_data = self.market_data

        if feature_data is None or self.model_type == "llm":
            # Pour les modèles purement LLM, expliquer autrement
            return self._explain_llm_prediction(prediction)

        # Pour les modèles hybrides avec ML
        if self.model_type == "hybrid" and hasattr(self.prediction_model, "ml_model"):
            ml_model = self.prediction_model.ml_model

            # Prétraiter les données pour le modèle
            X = feature_data.select_dtypes(include=[np.number])

            try:
                # Créer un explainer LIME
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=X.columns,
                    class_names=["bearish", "neutral", "bullish"],
                    mode="classification",
                )

                # Fonction de prédiction pour LIME
                def predict_fn(instances):
                    return ml_model.predict_proba(instances)

                # Générer l'explication pour la première instance
                exp = lime_explainer.explain_instance(
                    X.iloc[0].values, predict_fn, num_features=10
                )

                # Extraire les explications
                explanation_tuples = exp.as_list()
                features = [item[0] for item in explanation_tuples]
                scores = [item[1] for item in explanation_tuples]

                # Préparer le résultat
                lime_explanation = {
                    "features": features,
                    "scores": scores,
                    "prediction_id": prediction.get("id", "unknown"),
                    "asset": prediction.get("asset", "unknown"),
                    "direction": prediction.get("direction", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                }

                # Stocker l'explication
                self.explanations["lime"] = lime_explanation
                self.explainers["lime"] = lime_explainer

                return lime_explanation

            except Exception as e:
                logger.error(f"Erreur lors de la génération de l'explication LIME: {e}")
                return {"error": str(e), "message": "Échec de l'explication LIME"}

        return {"error": "Modèle non compatible avec LIME"}

    def _explain_llm_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explique une prédiction basée uniquement sur LLM.

        Args:
            prediction: Prédiction à expliquer

        Returns:
            Dictionnaire contenant l'explication et les métadonnées
        """
        # Extraire les facteurs cités par le LLM
        factors = prediction.get("factors", [])
        contradictions = prediction.get("contradictions", [])
        raw_response = prediction.get("raw_response", "")

        # Créer une explication textuelle
        explanation = {
            "factors": factors,
            "contradictions": contradictions,
            "factor_confidence": (
                {factor: 1.0 / len(factors) for factor in factors} if factors else {}
            ),
            "prediction_id": prediction.get("id", "unknown"),
            "asset": prediction.get("asset", "unknown"),
            "direction": prediction.get("direction", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "explanation_method": "llm_internal",
            "raw_response": raw_response,
        }

        return explanation

    def generate_report(
        self,
        prediction: Dict[str, Any],
        explanation_types: List[str] = ["shap", "lime", "llm"],
        output_format: str = "json",
    ) -> Dict[str, Any]:
        """
        Génère un rapport complet expliquant une prédiction.

        Args:
            prediction: Prédiction à expliquer
            explanation_types: Types d'explications à inclure
            output_format: Format de sortie ('json', 'html', 'pdf')

        Returns:
            Dictionnaire contenant le rapport complet
        """
        logger.info(
            f"Génération de rapport d'interprétabilité pour {prediction.get('asset', 'unknown')}"
        )

        report = {
            "prediction": prediction,
            "explanations": {},
            "timestamp": datetime.now().isoformat(),
            "report_format": output_format,
        }

        # Générer les explications demandées
        if "shap" in explanation_types:
            shap_explanation = self.explain_with_shap(prediction, self.market_data)
            report["explanations"]["shap"] = shap_explanation

        if "lime" in explanation_types:
            lime_explanation = self.explain_with_lime(prediction, self.market_data)
            report["explanations"]["lime"] = lime_explanation

        if "llm" in explanation_types or self.model_type == "llm":
            llm_explanation = self._explain_llm_prediction(prediction)
            report["explanations"]["llm"] = llm_explanation

        # Ajouter une analyse comparative
        report["comparative_analysis"] = self._compare_explanations(
            report["explanations"]
        )

        # Formatter la sortie
        if output_format == "json":
            return report
        elif output_format == "html":
            return self._generate_html_report(report)
        elif output_format == "pdf":
            return self._generate_pdf_report(report)
        else:
            return report

    def _compare_explanations(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare les différentes explications pour trouver des points communs.

        Args:
            explanations: Dictionnaire des explications

        Returns:
            Analyse comparative des explications
        """
        # Extraire les facteurs des différentes explications
        factors = set()

        # Facteurs du LLM
        if "llm" in explanations:
            llm_factors = explanations["llm"].get("factors", [])
            factors.update(llm_factors)

        # Facteurs importants de SHAP
        if "shap" in explanations and "feature_names" in explanations["shap"]:
            shap_values = explanations["shap"].get("shap_values", [])
            feature_names = explanations["shap"].get("feature_names", [])

            if len(shap_values) > 0 and len(feature_names) > 0:
                # Trouver les 5 caractéristiques les plus importantes
                important_indices = np.argsort(np.abs(shap_values).mean(0))[-5:]
                shap_factors = [feature_names[i] for i in important_indices]
                factors.update(shap_factors)

        # Facteurs de LIME
        if "lime" in explanations:
            lime_features = explanations["lime"].get("features", [])
            factors.update(lime_features)

        # Construire la réponse
        result = {
            "common_factors": list(factors),
            "agreement_score": self._calculate_agreement(explanations),
            "summary": "Analyse des facteurs influençant la prédiction",
        }

        return result

    def _calculate_agreement(self, explanations: Dict[str, Any]) -> float:
        """
        Calcule un score d'accord entre les différentes explications.

        Args:
            explanations: Dictionnaire des explications

        Returns:
            Score d'accord entre 0 et 1
        """
        # Méthode simplifiée pour calculer l'accord
        # Dans une implémentation réelle, on comparerait les importances des features

        # Compter le nombre d'explications
        num_explanations = len(explanations)

        if num_explanations <= 1:
            return 1.0  # Un seul modèle, aucun désaccord possible

        # Pour le moment, retourner un score arbitraire
        return 0.8  # Score d'accord élevé

    def _generate_html_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un rapport HTML à partir des explications.

        Args:
            report: Données du rapport

        Returns:
            Dictionnaire contenant le HTML et les métadonnées
        """
        # Ici, on générerait un vrai rapport HTML
        # Pour l'instant, on simule simplement
        html_content = f"""
        <html>
        <head>
            <title>Rapport d'explicabilité - {report["prediction"].get("asset", "Unknown")}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                .prediction {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .factors {{ display: flex; flex-wrap: wrap; }}
                .factor {{ background-color: #e9ecef; margin: 5px; padding: 8px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Rapport d'explicabilité pour {report["prediction"].get("asset", "Unknown")}</h1>
            <p>Date: {report["timestamp"]}</p>
            
            <h2>Prédiction</h2>
            <div class="prediction">
                <p>Direction: {report["prediction"].get("direction", "Unknown")}</p>
                <p>Confiance: {report["prediction"].get("confidence", "Unknown")}</p>
                <p>Horizon: {report["prediction"].get("timeframe", "Unknown")}</p>
            </div>
            
            <h2>Facteurs d'influence</h2>
            <div class="factors">
                {" ".join([f'<div class="factor">{factor}</div>' for factor in report["comparative_analysis"].get("common_factors", [])])}
            </div>
        </body>
        </html>
        """

        report_result = report.copy()
        report_result["html_content"] = html_content

        return report_result

    def _generate_pdf_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un rapport PDF à partir des explications.

        Args:
            report: Données du rapport

        Returns:
            Dictionnaire contenant le chemin du PDF et les métadonnées
        """
        # Dans une implémentation réelle, on générerait un PDF
        # Pour l'instant, simuler un chemin de fichier

        report_result = report.copy()
        report_result["pdf_path"] = (
            f"reports/prediction_{report['prediction'].get('id', 'unknown')}.pdf"
        )

        return report_result

    def plot_shap_summary(
        self, prediction_id: Optional[str] = None, save_path: Optional[str] = None
    ):
        """
        Génère un graphique résumant les valeurs SHAP.

        Args:
            prediction_id: ID de la prédiction à visualiser
            save_path: Chemin pour sauvegarder l'image (optionnel)

        Returns:
            None (affiche ou sauvegarde le graphique)
        """
        if "shap" not in self.explanations:
            logger.warning("Aucune explication SHAP disponible")
            return

        shap_explanation = self.explanations["shap"]

        # Créer la figure
        plt.figure(figsize=(10, 8))

        # Tracer le résumé SHAP
        shap_values = shap_explanation["shap_values"]
        feature_names = shap_explanation["feature_names"]

        # Créer un DataFrame pour le plot
        shap_df = pd.DataFrame(shap_values, columns=feature_names)

        # Calculer l'importance moyenne des features
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importance}
        ).sort_values("Importance", ascending=False)

        # Graphique à barres
        plt.barh(
            feature_importance_df["Feature"][:10],
            feature_importance_df["Importance"][:10],
        )
        plt.xlabel("SHAP Importance")
        plt.title(
            f"SHAP Feature Importance pour {shap_explanation.get('asset', 'Unknown')}"
        )
        plt.tight_layout()

        # Sauvegarder ou afficher
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Graphique SHAP sauvegardé: {save_path}")
        else:
            plt.show()

    def plot_lime_explanation(
        self, prediction_id: Optional[str] = None, save_path: Optional[str] = None
    ):
        """
        Génère un graphique expliquant les résultats LIME.

        Args:
            prediction_id: ID de la prédiction à visualiser
            save_path: Chemin pour sauvegarder l'image (optionnel)

        Returns:
            None (affiche ou sauvegarde le graphique)
        """
        if "lime" not in self.explanations:
            logger.warning("Aucune explication LIME disponible")
            return

        lime_explanation = self.explanations["lime"]

        # Créer la figure
        plt.figure(figsize=(10, 8))

        # Extraire les données
        features = lime_explanation.get("features", [])
        scores = lime_explanation.get("scores", [])

        # Créer un DataFrame
        df = pd.DataFrame({"Feature": features, "Score": scores})

        # Trier par score absolu
        df["AbsScore"] = df["Score"].abs()
        df = df.sort_values("AbsScore", ascending=True)

        # Créer un graphique à barres coloré selon l'impact positif ou négatif
        colors = ["red" if score < 0 else "green" for score in df["Score"]]
        plt.barh(df["Feature"], df["Score"], color=colors)

        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        plt.xlabel("LIME Score (effet sur la prédiction)")
        plt.title(f"Explication LIME pour {lime_explanation.get('asset', 'Unknown')}")
        plt.tight_layout()

        # Sauvegarder ou afficher
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Graphique LIME sauvegardé: {save_path}")
        else:
            plt.show()

    def plot_factor_importance(
        self, prediction: Dict[str, Any], save_path: Optional[str] = None
    ):
        """
        Génère un graphique montrant l'importance des facteurs cités par le LLM.

        Args:
            prediction: Prédiction à visualiser
            save_path: Chemin pour sauvegarder l'image (optionnel)

        Returns:
            None (affiche ou sauvegarde le graphique)
        """
        # Extraire les facteurs
        factors = prediction.get("factors", [])

        if not factors:
            logger.warning("Aucun facteur trouvé dans la prédiction")
            return

        # Créer des scores arbitraires décroissants (le premier facteur mentionné étant le plus important)
        scores = [1.0 - i * (0.9 / len(factors)) for i in range(len(factors))]

        # Créer la figure
        plt.figure(figsize=(10, 6))

        # Barres horizontales
        plt.barh(factors, scores, color="skyblue")
        plt.xlabel("Importance relative")
        plt.title(f"Facteurs d'influence pour {prediction.get('asset', 'Unknown')}")
        plt.xlim(0, 1)
        plt.tight_layout()

        # Sauvegarder ou afficher
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Graphique des facteurs sauvegardé: {save_path}")
        else:
            plt.show()


# Exemple d'utilisation:
"""
explainer = PredictionExplainer(prediction_model)
explanation = explainer.explain_with_shap(prediction, feature_data)
report = explainer.generate_report(prediction)
explainer.plot_shap_summary()
"""
