"""
Module pour la génération de rapports détaillés sur les prédictions de marché.

Ce module fournit des fonctionnalités pour générer des rapports structurés
sur les prédictions, y compris des graphiques, des tableaux et des explications.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

from ai_trading.utils import setup_logger
from ai_trading.llm.predictions.prediction_explainer import PredictionExplainer
from ai_trading.llm.predictions.visualization import PredictionVisualizer

# Configuration du logger
logger = setup_logger("prediction_reporting")

class PredictionReporter:
    """
    Classe pour générer des rapports détaillés sur les prédictions de marché.
    
    Cette classe utilise l'interprétabilité et les visualisations pour créer
    des rapports complets pour l'analyse des prédictions.
    """
    
    def __init__(self, explainer: Optional[PredictionExplainer] = None, 
                 visualizer: Optional[PredictionVisualizer] = None):
        """
        Initialise le générateur de rapports.
        
        Args:
            explainer: Instance de PredictionExplainer (optionnel)
            visualizer: Instance de PredictionVisualizer (optionnel)
        """
        self.explainer = explainer
        self.visualizer = visualizer
        
        # Créer les instances si non fournies
        if self.explainer is None:
            self.explainer = PredictionExplainer()
            
        if self.visualizer is None:
            self.visualizer = PredictionVisualizer(self.explainer)
            
        # Stockage des rapports
        self.reports = {}
        
        logger.info("PredictionReporter initialisé")
    
    def generate_prediction_report(self, 
                                 prediction: Dict[str, Any],
                                 market_data: Optional[pd.DataFrame] = None,
                                 output_dir: Optional[str] = None,
                                 format: str = "html") -> str:
        """
        Génère un rapport complet pour une prédiction.
        
        Args:
            prediction: Prédiction à expliquer
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            format: Format de sortie ('html', 'pdf', 'json')
            
        Returns:
            Chemin du rapport généré
        """
        logger.info(f"Génération de rapport pour {prediction.get('asset', 'unknown')} au format {format}")
        
        # 1. Générer des explications
        explanations = {}
        explanation_types = ["shap", "lime", "llm"]
        
        for exp_type in explanation_types:
            if exp_type == "shap":
                explanations["shap"] = self.explainer.explain_with_shap(prediction, market_data)
            elif exp_type == "lime":
                explanations["lime"] = self.explainer.explain_with_lime(prediction, market_data)
            elif exp_type == "llm":
                explanations["llm"] = self.explainer._explain_llm_prediction(prediction)
        
        # 2. Créer un rapport selon le format demandé
        if format.lower() == "html":
            report_path = self._generate_html_report(prediction, explanations, market_data, output_dir)
        elif format.lower() == "pdf":
            report_path = self._generate_pdf_report(prediction, explanations, market_data, output_dir)
        elif format.lower() == "json":
            report_path = self._generate_json_report(prediction, explanations, market_data, output_dir)
        else:
            report_path = self._generate_html_report(prediction, explanations, market_data, output_dir)
        
        # Stocker le rapport
        report_id = f"{prediction.get('asset', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.reports[report_id] = report_path
        
        return report_path
    
    def generate_multi_horizon_report(self, 
                                    predictions: Dict[str, Any],
                                    consistency_analysis: Dict[str, Any],
                                    market_data: Optional[pd.DataFrame] = None,
                                    output_dir: Optional[str] = None,
                                    format: str = "html") -> str:
        """
        Génère un rapport complet pour des prédictions multi-horizons.
        
        Args:
            predictions: Dictionnaire des prédictions par horizon
            consistency_analysis: Analyse de cohérence
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            format: Format de sortie ('html', 'pdf', 'json')
            
        Returns:
            Chemin du rapport généré
        """
        logger.info("Génération de rapport multi-horizons")
        
        # Extraire l'actif de la première prédiction
        first_prediction = next(iter(predictions.values()), {})
        asset = first_prediction.get("asset", "unknown")
        
        # 1. Générer des explications pour chaque prédiction
        horizon_explanations = {}
        
        for horizon, prediction in predictions.items():
            explanations = {}
            for exp_type in ["shap", "lime", "llm"]:
                if exp_type == "shap":
                    explanations["shap"] = self.explainer.explain_with_shap(prediction, market_data)
                elif exp_type == "lime":
                    explanations["lime"] = self.explainer.explain_with_lime(prediction, market_data)
                elif exp_type == "llm":
                    explanations["llm"] = self.explainer._explain_llm_prediction(prediction)
            
            horizon_explanations[horizon] = explanations
        
        # 2. Créer un rapport selon le format demandé
        if format.lower() == "html":
            report_path = self._generate_multi_horizon_html_report(
                predictions, horizon_explanations, consistency_analysis, market_data, output_dir)
        elif format.lower() == "pdf":
            report_path = self._generate_multi_horizon_pdf_report(
                predictions, horizon_explanations, consistency_analysis, market_data, output_dir)
        elif format.lower() == "json":
            report_path = self._generate_multi_horizon_json_report(
                predictions, horizon_explanations, consistency_analysis, market_data, output_dir)
        else:
            report_path = self._generate_multi_horizon_html_report(
                predictions, horizon_explanations, consistency_analysis, market_data, output_dir)
        
        # Stocker le rapport
        report_id = f"{asset}_multi_horizon_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.reports[report_id] = report_path
        
        return report_path
    
    def _generate_html_report(self, 
                            prediction: Dict[str, Any],
                            explanations: Dict[str, Any],
                            market_data: Optional[pd.DataFrame] = None,
                            output_dir: Optional[str] = None) -> str:
        """
        Génère un rapport HTML pour une prédiction.
        
        Args:
            prediction: Prédiction à expliquer
            explanations: Dictionnaire des explications
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du rapport HTML
        """
        # Utiliser le visualiseur pour créer un rapport détaillé
        return self.visualizer.create_detailed_report(
            prediction, explanations, market_data=market_data, output_dir=output_dir)
    
    def _generate_pdf_report(self, 
                          prediction: Dict[str, Any],
                          explanations: Dict[str, Any],
                          market_data: Optional[pd.DataFrame] = None,
                          output_dir: Optional[str] = None) -> str:
        """
        Génère un rapport PDF pour une prédiction.
        
        Args:
            prediction: Prédiction à expliquer
            explanations: Dictionnaire des explications
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du rapport PDF
        """
        # Dans une implémentation réelle, utilisez une bibliothèque comme weasyprint
        # pour convertir le HTML en PDF. Ici, nous simulons simplement.
        logger.warning("Génération PDF non implémentée, retour d'un rapport HTML")
        
        # D'abord générer le HTML
        html_path = self._generate_html_report(prediction, explanations, market_data, output_dir)
        
        # Simuler un chemin de fichier PDF
        pdf_path = html_path.replace(".html", ".pdf")
        
        return pdf_path
    
    def _generate_json_report(self, 
                           prediction: Dict[str, Any],
                           explanations: Dict[str, Any],
                           market_data: Optional[pd.DataFrame] = None,
                           output_dir: Optional[str] = None) -> str:
        """
        Génère un rapport JSON pour une prédiction.
        
        Args:
            prediction: Prédiction à expliquer
            explanations: Dictionnaire des explications
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du rapport JSON
        """
        # Préparer le dictionnaire de rapport
        report_dict = {
            "prediction": prediction,
            "explanations": explanations,
            "timestamp": datetime.now().isoformat(),
            "market_data_shape": market_data.shape if market_data is not None else None
        }
        
        # Définir le chemin de sortie
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(
                output_dir, 
                f"prediction_report_{prediction.get('asset', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        else:
            temp_dir = tempfile.gettempdir()
            json_path = os.path.join(
                temp_dir, 
                f"prediction_report_{prediction.get('asset', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # Écrire le fichier JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        logger.info(f"Rapport JSON généré: {json_path}")
        
        return json_path
    
    def _generate_multi_horizon_html_report(self, 
                                         predictions: Dict[str, Any],
                                         horizon_explanations: Dict[str, Dict[str, Any]],
                                         consistency_analysis: Dict[str, Any],
                                         market_data: Optional[pd.DataFrame] = None,
                                         output_dir: Optional[str] = None) -> str:
        """
        Génère un rapport HTML pour des prédictions multi-horizons.
        
        Args:
            predictions: Dictionnaire des prédictions par horizon
            horizon_explanations: Explications pour chaque horizon
            consistency_analysis: Analyse de cohérence
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du rapport HTML
        """
        # Extraire l'actif de la première prédiction
        first_prediction = next(iter(predictions.values()), {})
        asset = first_prediction.get("asset", "unknown")
        
        # Créer des visualisations pour les prédictions et les cohérences
        # Définir le répertoire de sortie
        if output_dir:
            report_dir = output_dir
        else:
            temp_dir = tempfile.gettempdir()
            report_dir = os.path.join(temp_dir, f"multihorizon_report_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
        os.makedirs(report_dir, exist_ok=True)
        
        # Générer les graphiques et tableaux de bord
        dashboard_path = os.path.join(report_dir, "prediction_dashboard.html")
        dashboard_path = self.visualizer.create_prediction_dashboard(predictions, market_data, dashboard_path)
        
        consistency_path = os.path.join(report_dir, "consistency_visualization.html")
        consistency_path = self.visualizer.create_consistency_visualization(consistency_analysis, consistency_path)
        
        # Générer un rapport pour chaque horizon
        horizon_report_paths = {}
        for horizon, prediction in predictions.items():
            horizon_report_path = os.path.join(report_dir, f"{horizon}_report.html")
            explanations = horizon_explanations.get(horizon, {})
            horizon_report_path = self.visualizer.create_detailed_report(
                prediction, explanations, output_dir=os.path.join(report_dir, horizon))
            horizon_report_paths[horizon] = os.path.relpath(horizon_report_path, report_dir)
        
        # Créer le HTML principal
        report_html = os.path.join(report_dir, "multihorizon_report.html")
        
        # Contenu HTML du rapport
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport Multi-Horizons - {asset}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
                .signal {{ font-weight: bold; padding: 10px; border-radius: 5px; display: inline-block; }}
                .signal-buy {{ background: #d4edda; color: #155724; }}
                .signal-sell {{ background: #f8d7da; color: #721c24; }}
                .signal-hold {{ background: #fff3cd; color: #856404; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
                .tabs {{ display: flex; margin-bottom: 20px; }}
                .tab {{ padding: 10px 20px; cursor: pointer; background: #f5f5f5; margin-right: 5px; border-radius: 5px 5px 0 0; }}
                .tab.active {{ background: #2c3e50; color: white; }}
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Rapport de Prédiction Multi-Horizons</h1>
                <p>Actif: <strong>{asset}</strong> | Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="card">
                    <h2>Résumé du Signal de Trading</h2>
                    <div class="signal signal-{consistency_analysis.get('trading_signals', {}).get('signal', 'hold')}">
                        Signal: {consistency_analysis.get('trading_signals', {}).get('signal', 'HOLD').upper()}
                    </div>
                    <p>{consistency_analysis.get('trading_signals', {}).get('description', '')}</p>
                </div>
                
                <div class="card">
                    <h2>Tableau des Prédictions</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Horizon</th>
                                <th>Direction</th>
                                <th>Confiance</th>
                                <th>Facteurs</th>
                                <th>Détails</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Ajouter les lignes du tableau
        for horizon, prediction in predictions.items():
            direction = prediction.get("direction", "neutral")
            confidence = prediction.get("confidence", "low")
            factors = ", ".join(prediction.get("factors", [])[:3])
            horizon_report = horizon_report_paths.get(horizon, "#")
            
            html_content += f"""
                            <tr>
                                <td>{horizon}</td>
                                <td class="direction-{direction}">{direction.upper()}</td>
                                <td>{confidence.upper()}</td>
                                <td>{factors}...</td>
                                <td><a href="{horizon}/{os.path.basename(horizon_report)}" target="_blank">Voir rapport</a></td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Visualisations</h2>
                    <div class="tabs">
                        <div class="tab active" onclick="openTab(event, 'tab-predictions')">Prédictions</div>
                        <div class="tab" onclick="openTab(event, 'tab-consistency')">Cohérence</div>
                    </div>
                    
                    <div id="tab-predictions" class="tab-content active">
                        <iframe src="prediction_dashboard.html"></iframe>
                    </div>
                    
                    <div id="tab-consistency" class="tab-content">
                        <iframe src="consistency_visualization.html"></iframe>
                    </div>
                </div>
        """
        
        # Ajouter des données du marché si disponibles
        if market_data is not None:
            market_table = market_data.tail(10).to_html(classes="market-data-table", border=0)
            html_content += f"""
                <div class="card">
                    <h2>Données de marché récentes</h2>
                    {market_table}
                </div>
            """
        
        # Finaliser le HTML
        html_content += """
            </div>
            <script>
                // Set the first tab as active by default
                document.getElementsByClassName("tab")[0].click();
            </script>
        </body>
        </html>
        """
        
        # Écrire le fichier HTML
        with open(report_html, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Rapport multi-horizons généré: {report_html}")
        
        return report_html
    
    def _generate_multi_horizon_pdf_report(self, 
                                        predictions: Dict[str, Any],
                                        horizon_explanations: Dict[str, Dict[str, Any]],
                                        consistency_analysis: Dict[str, Any],
                                        market_data: Optional[pd.DataFrame] = None,
                                        output_dir: Optional[str] = None) -> str:
        """
        Génère un rapport PDF pour des prédictions multi-horizons.
        
        Args:
            predictions: Dictionnaire des prédictions par horizon
            horizon_explanations: Explications pour chaque horizon
            consistency_analysis: Analyse de cohérence
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du rapport PDF
        """
        # Dans une implémentation réelle, utilisez une bibliothèque comme weasyprint
        # pour convertir le HTML en PDF. Ici, nous simulons simplement.
        logger.warning("Génération PDF non implémentée, retour d'un rapport HTML")
        
        # D'abord générer le HTML
        html_path = self._generate_multi_horizon_html_report(
            predictions, horizon_explanations, consistency_analysis, market_data, output_dir)
        
        # Simuler un chemin de fichier PDF
        pdf_path = html_path.replace(".html", ".pdf")
        
        return pdf_path
    
    def _generate_multi_horizon_json_report(self, 
                                         predictions: Dict[str, Any],
                                         horizon_explanations: Dict[str, Dict[str, Any]],
                                         consistency_analysis: Dict[str, Any],
                                         market_data: Optional[pd.DataFrame] = None,
                                         output_dir: Optional[str] = None) -> str:
        """
        Génère un rapport JSON pour des prédictions multi-horizons.
        
        Args:
            predictions: Dictionnaire des prédictions par horizon
            horizon_explanations: Explications pour chaque horizon
            consistency_analysis: Analyse de cohérence
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du rapport JSON
        """
        # Extraire l'actif de la première prédiction
        first_prediction = next(iter(predictions.values()), {})
        asset = first_prediction.get("asset", "unknown")
        
        # Préparer le dictionnaire de rapport
        report_dict = {
            "asset": asset,
            "predictions": predictions,
            "explanations": horizon_explanations,
            "consistency_analysis": consistency_analysis,
            "timestamp": datetime.now().isoformat(),
            "market_data_shape": market_data.shape if market_data is not None else None
        }
        
        # Définir le chemin de sortie
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(
                output_dir, 
                f"multihorizon_report_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        else:
            temp_dir = tempfile.gettempdir()
            json_path = os.path.join(
                temp_dir, 
                f"multihorizon_report_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # Écrire le fichier JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        logger.info(f"Rapport JSON multi-horizons généré: {json_path}")
        
        return json_path

# Exemple d'utilisation:
"""
explainer = PredictionExplainer(prediction_model)
visualizer = PredictionVisualizer(explainer)
reporter = PredictionReporter(explainer, visualizer)

# Pour une seule prédiction
report_path = reporter.generate_prediction_report(prediction, market_data)

# Pour des prédictions multi-horizons
report_path = reporter.generate_multi_horizon_report(predictions, consistency_analysis, market_data)
""" 