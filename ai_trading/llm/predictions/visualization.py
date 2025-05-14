"""
Module de visualisation pour les prédictions de marché et leurs explications.

Ce module fournit des outils avancés pour visualiser les prédictions, 
les explications SHAP/LIME et générer des graphiques interactifs.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ai_trading.utils import setup_logger
from ai_trading.llm.predictions.prediction_explainer import PredictionExplainer

# Configuration du logger
logger = setup_logger("visualization")

class PredictionVisualizer:
    """
    Classe pour la visualisation des prédictions et de leurs explications.
    
    Cette classe fournit des méthodes pour créer des visualisations interactives
    des prédictions de marché, des explications d'interprétabilité et des rapports.
    """
    
    def __init__(self, explainer: Optional[PredictionExplainer] = None):
        """
        Initialise le visualiseur de prédictions.
        
        Args:
            explainer: Instance de PredictionExplainer (optionnel)
        """
        self.explainer = explainer
        self.visualizations = {}
        
        logger.info("PredictionVisualizer initialisé")
    
    def create_prediction_dashboard(self, 
                                  predictions: Dict[str, Any], 
                                  market_data: Optional[pd.DataFrame] = None,
                                  output_path: Optional[str] = None) -> str:
        """
        Crée un tableau de bord interactif pour visualiser les prédictions.
        
        Args:
            predictions: Dictionnaire des prédictions par horizon
            market_data: Données de marché (optionnel)
            output_path: Chemin de sortie pour le HTML (optionnel)
            
        Returns:
            Chemin du fichier HTML généré
        """
        logger.info("Création du tableau de bord de prédiction")
        
        # Créer un dataframe de prédictions
        prediction_records = []
        
        for timeframe, prediction in predictions.items():
            prediction_records.append({
                "timeframe": timeframe,
                "direction": prediction.get("direction", "neutral"),
                "confidence": prediction.get("confidence", "low"),
                "asset": prediction.get("asset", "unknown")
            })
            
        pred_df = pd.DataFrame(prediction_records)
        
        # Convertir les directions et confiances en valeurs numériques pour le graphique
        direction_map = {"bearish": -1, "neutral": 0, "bullish": 1}
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        
        pred_df["direction_value"] = pred_df["direction"].map(direction_map)
        pred_df["confidence_value"] = pred_df["confidence"].map(confidence_map)
        pred_df["combined_signal"] = pred_df["direction_value"] * pred_df["confidence_value"]
        
        # Couleurs selon la direction
        direction_colors = {"bearish": "red", "neutral": "gray", "bullish": "green"}
        pred_df["color"] = pred_df["direction"].map(direction_colors)
        
        # Créer le graphique de base
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=["Prédictions par horizon temporel", 
                                           "Force du signal combiné"],
                           row_heights=[0.6, 0.4],
                           vertical_spacing=0.1)
        
        # Ajouter les barres de direction
        fig.add_trace(
            go.Bar(
                x=pred_df["timeframe"],
                y=pred_df["direction_value"],
                marker_color=pred_df["color"],
                name="Direction",
                text=pred_df["direction"],
                hovertemplate="<b>%{x}</b><br>Direction: %{text}<br>Valeur: %{y}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Ajouter les points de confiance
        fig.add_trace(
            go.Scatter(
                x=pred_df["timeframe"],
                y=pred_df["confidence_value"],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="circle",
                    line=dict(width=2, color="DarkSlateGrey")
                ),
                name="Confiance",
                text=pred_df["confidence"],
                hovertemplate="<b>%{x}</b><br>Confiance: %{text}<br>Valeur: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Ajouter le signal combiné
        fig.add_trace(
            go.Scatter(
                x=pred_df["timeframe"],
                y=pred_df["combined_signal"],
                mode="lines+markers",
                line=dict(width=3, dash="solid"),
                marker=dict(size=8),
                name="Signal combiné",
                hovertemplate="<b>%{x}</b><br>Signal: %{y:.2f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Ajouter une ligne zéro
        fig.add_shape(
            type="line",
            x0=pred_df["timeframe"].iloc[0],
            y0=0,
            x1=pred_df["timeframe"].iloc[-1],
            y1=0,
            line=dict(color="black", width=1, dash="dot"),
            row=2, col=1
        )
        
        # Mise en forme du graphique
        fig.update_layout(
            title=f"Prédictions pour {pred_df['asset'].iloc[0]}",
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        # Mise en forme des axes
        fig.update_yaxes(title_text="Direction", row=1, col=1)
        fig.update_yaxes(title_text="Signal combiné", row=2, col=1)
        fig.update_xaxes(title_text="Horizon temporel", row=2, col=1)
        
        # Générer le HTML
        if output_path:
            html_path = output_path
        else:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"prediction_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            
        fig.write_html(html_path)
        logger.info(f"Tableau de bord de prédiction sauvegardé: {html_path}")
        
        # Stocker la visualisation
        self.visualizations["prediction_dashboard"] = html_path
        
        return html_path
    
    def create_explanation_dashboard(self, 
                                   prediction: Dict[str, Any],
                                   explanations: Dict[str, Any],
                                   output_path: Optional[str] = None) -> str:
        """
        Crée un tableau de bord interactif pour les explications d'une prédiction.
        
        Args:
            prediction: Prédiction à expliquer
            explanations: Dictionnaire des explications (SHAP, LIME, etc.)
            output_path: Chemin de sortie pour le HTML (optionnel)
            
        Returns:
            Chemin du fichier HTML généré
        """
        logger.info("Création du tableau de bord d'explication")
        
        # Créer le graphique de base avec des sous-graphiques
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Facteurs LLM", 
                "Valeurs SHAP",
                "Explication LIME", 
                "Confiance globale"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Graphique des facteurs LLM
        if "llm" in explanations:
            llm_explanation = explanations["llm"]
            factors = llm_explanation.get("factors", [])
            
            if factors:
                # Créer des scores arbitraires décroissants
                scores = [1.0 - i * (0.9 / len(factors)) for i in range(len(factors))]
                
                fig.add_trace(
                    go.Bar(
                        y=factors,
                        x=scores,
                        orientation="h",
                        name="Facteurs LLM",
                        marker_color="lightskyblue",
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # 2. Valeurs SHAP
        if "shap" in explanations:
            shap_explanation = explanations["shap"]
            
            # Extraire les données SHAP si disponibles
            if "feature_names" in shap_explanation and "shap_values" in shap_explanation:
                feature_names = shap_explanation["feature_names"]
                shap_values = shap_explanation["shap_values"]
                
                if len(feature_names) > 0 and len(shap_values) > 0:
                    # Calculer l'importance moyenne des features
                    feature_importance = np.abs(shap_values).mean(0)
                    
                    # Créer un DataFrame et trier
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig.add_trace(
                        go.Bar(
                            y=shap_df['Feature'],
                            x=shap_df['Importance'],
                            orientation="h",
                            name="SHAP Importance",
                            marker_color="lightgreen",
                            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
                        ),
                        row=1, col=2
                    )
        
        # 3. Explication LIME
        if "lime" in explanations:
            lime_explanation = explanations["lime"]
            
            # Extraire les données LIME
            features = lime_explanation.get("features", [])
            scores = lime_explanation.get("scores", [])
            
            if features and scores:
                # Créer un DataFrame
                lime_df = pd.DataFrame({
                    'Feature': features,
                    'Score': scores
                })
                
                # Trier par score absolu
                lime_df['AbsScore'] = lime_df['Score'].abs()
                lime_df = lime_df.sort_values('AbsScore', ascending=False).head(10)
                
                # Déterminer les couleurs selon le signe du score
                colors = ['red' if score < 0 else 'green' for score in lime_df['Score']]
                
                fig.add_trace(
                    go.Bar(
                        y=lime_df['Feature'],
                        x=lime_df['Score'],
                        orientation="h",
                        name="LIME Scores",
                        marker_color=colors,
                        hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>"
                    ),
                    row=2, col=1
                )
        
        # 4. Indicateur de confiance
        confidence = prediction.get("confidence", "low")
        confidence_value = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(confidence, 0.5)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=confidence_value,
                title={"text": f"Confiance: {confidence}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.33], 'color': "lightgray"},
                        {'range': [0.33, 0.66], 'color': "gray"},
                        {'range': [0.66, 1], 'color': "darkgray"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        # Mise en forme du graphique
        fig.update_layout(
            title=f"Explication de la prédiction: {prediction.get('asset', 'inconnu')} ({prediction.get('direction', 'unknown')})",
            height=900,
            width=1200,
            showlegend=False,
            template="plotly_white"
        )
        
        # Mise en forme des axes
        fig.update_xaxes(title_text="Importance", row=1, col=1)
        fig.update_xaxes(title_text="Valeur SHAP", row=1, col=2)
        fig.update_xaxes(title_text="Score LIME", row=2, col=1)
        
        # Générer le HTML
        if output_path:
            html_path = output_path
        else:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"explanation_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            
        fig.write_html(html_path)
        logger.info(f"Tableau de bord d'explication sauvegardé: {html_path}")
        
        # Stocker la visualisation
        self.visualizations["explanation_dashboard"] = html_path
        
        return html_path
    
    def create_consistency_visualization(self, 
                                       consistency_analysis: Dict[str, Any],
                                       output_path: Optional[str] = None) -> str:
        """
        Crée une visualisation de la cohérence des prédictions multi-horizons.
        
        Args:
            consistency_analysis: Analyse de cohérence des prédictions
            output_path: Chemin de sortie pour le HTML (optionnel)
            
        Returns:
            Chemin du fichier HTML généré
        """
        logger.info("Création de la visualisation de cohérence")
        
        # Extraire les données d'analyse par horizon
        horizon_analysis = consistency_analysis.get("horizon_analysis", {})
        signal = consistency_analysis.get("trading_signals", {})
        
        # Créer un graphique à plusieurs parties
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Cohérence par horizon", "Répartition des directions"],
            vertical_spacing=0.2,
            row_heights=[0.4, 0.6]
        )
        
        # 1. Graphique de cohérence par horizon
        horizons = []
        consistencies = []
        
        for horizon, analysis in horizon_analysis.items():
            horizons.append(horizon)
            consistencies.append(analysis.get("consistency", 0))
        
        # Déterminer les couleurs selon la direction
        direction_colors = {
            "bullish": "green",
            "neutral": "gray",
            "bearish": "red"
        }
        
        colors = [direction_colors.get(horizon_analysis.get(h, {}).get("overall_direction", "neutral"), "gray") for h in horizons]
        
        fig.add_trace(
            go.Bar(
                x=horizons,
                y=consistencies,
                marker_color=colors,
                name="Cohérence",
                text=[f"{c:.2f}" for c in consistencies],
                hovertemplate="<b>%{x}</b><br>Cohérence: %{text}<br>Direction: %{customdata}<extra></extra>",
                customdata=[horizon_analysis.get(h, {}).get("overall_direction", "neutral") for h in horizons]
            ),
            row=1, col=1
        )
        
        # 2. Répartition des directions par horizon (graphique Sankey ou empilé)
        # Pour simplifier, utilisons un graphique à barres empilées
        
        # Préparer les données
        horizons_order = ["short_term", "medium_term", "long_term"]
        directions = ["bearish", "neutral", "bullish"]
        
        # Créer une matrice pour les comptes
        counts_matrix = []
        
        for horizon in horizons_order:
            if horizon in horizon_analysis:
                direction_counts = horizon_analysis[horizon].get("direction_counts", {})
                counts = [direction_counts.get(d, 0) for d in directions]
                counts_matrix.append(counts)
            else:
                counts_matrix.append([0, 0, 0])
        
        # Transposer pour créer une série par direction
        counts_matrix = np.array(counts_matrix).T
        
        # Ajouter les traces pour chaque direction
        for i, direction in enumerate(directions):
            fig.add_trace(
                go.Bar(
                    x=horizons_order,
                    y=counts_matrix[i],
                    name=direction,
                    marker_color=direction_colors[direction],
                    hovertemplate="<b>%{x}</b><br>%{y} %{fullData.name} prédictions<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Ajouter une annotation pour le signal de trading
        signal_text = f"Signal: {signal.get('signal', 'unknown')} - {signal.get('description', '')}"
        
        fig.add_annotation(
            text=signal_text,
            xref="paper", yref="paper",
            x=0.5, y=0,
            showarrow=False,
            font=dict(size=14),
            bgcolor="lightyellow",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
        
        # Mise en forme du graphique
        fig.update_layout(
            title="Analyse de cohérence des prédictions multi-horizons",
            height=800,
            width=1000,
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        # Mise en forme des axes
        fig.update_yaxes(title_text="Cohérence", row=1, col=1)
        fig.update_yaxes(title_text="Nombre de prédictions", row=2, col=1)
        
        # Générer le HTML
        if output_path:
            html_path = output_path
        else:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, f"consistency_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            
        fig.write_html(html_path)
        logger.info(f"Visualisation de cohérence sauvegardée: {html_path}")
        
        # Stocker la visualisation
        self.visualizations["consistency_visualization"] = html_path
        
        return html_path
    
    def create_detailed_report(self, 
                             prediction: Dict[str, Any], 
                             explanations: Dict[str, Any],
                             consistency_analysis: Optional[Dict[str, Any]] = None,
                             market_data: Optional[pd.DataFrame] = None,
                             output_dir: Optional[str] = None) -> str:
        """
        Génère un rapport détaillé complet avec toutes les visualisations.
        
        Args:
            prediction: Prédiction à expliquer
            explanations: Dictionnaire des explications
            consistency_analysis: Analyse de cohérence (optionnel)
            market_data: Données de marché (optionnel)
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du rapport HTML
        """
        logger.info("Génération du rapport détaillé")
        
        # Créer un répertoire de sortie
        if output_dir:
            report_dir = output_dir
        else:
            temp_dir = tempfile.gettempdir()
            report_dir = os.path.join(temp_dir, f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
        os.makedirs(report_dir, exist_ok=True)
        
        # Générer les visualisations individuelles
        explanation_path = os.path.join(report_dir, "explanation_dashboard.html")
        self.create_explanation_dashboard(prediction, explanations, explanation_path)
        
        # Générer le rapport HTML principal
        report_html = os.path.join(report_dir, "report.html")
        
        # Créer le contenu HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport de prédiction détaillé - {prediction.get('asset', 'unknown')}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
                .prediction {{ display: flex; align-items: center; }}
                .prediction-icon {{ font-size: 36px; margin-right: 20px; }}
                .prediction-details {{ flex: 1; }}
                .factors {{ display: flex; flex-wrap: wrap; }}
                .factor {{ background: #f5f5f5; margin: 5px; padding: 8px 12px; border-radius: 15px; font-size: 14px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                .direction-bullish {{ color: green; }}
                .direction-bearish {{ color: red; }}
                .direction-neutral {{ color: gray; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Rapport de prédiction détaillé</h1>
                <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="card">
                    <h2>Résumé de la prédiction</h2>
                    <div class="prediction">
                        <div class="prediction-icon">
                            {'📈' if prediction.get('direction') == 'bullish' else '📉' if prediction.get('direction') == 'bearish' else '📊'}
                        </div>
                        <div class="prediction-details">
                            <h3>Actif: {prediction.get('asset', 'unknown')}</h3>
                            <p>Direction: <span class="direction-{prediction.get('direction', 'neutral')}">{prediction.get('direction', 'neutral').upper()}</span></p>
                            <p>Confiance: {prediction.get('confidence', 'low').upper()}</p>
                            <p>Horizon: {prediction.get('timeframe', 'unknown')}</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Facteurs d'influence</h2>
                    <div class="factors">
                        {' '.join([f'<div class="factor">{factor}</div>' for factor in prediction.get('factors', [])])}
                    </div>
                    {'<p><strong>Contradictions:</strong> ' + ', '.join(prediction.get('contradictions', [])) + '</p>' if prediction.get('contradictions') else ''}
                </div>
                
                <div class="card">
                    <h2>Tableau de bord d'explication interactif</h2>
                    <iframe src="explanation_dashboard.html"></iframe>
                </div>
        """
        
        # Ajouter des sections conditionnelles
        if consistency_analysis:
            consistency_path = os.path.join(report_dir, "consistency_visualization.html")
            self.create_consistency_visualization(consistency_analysis, consistency_path)
            
            html_content += f"""
                <div class="card">
                    <h2>Analyse de cohérence multi-horizons</h2>
                    <iframe src="consistency_visualization.html"></iframe>
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
        </body>
        </html>
        """
        
        # Écrire le fichier HTML
        with open(report_html, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Rapport détaillé généré: {report_html}")
        
        # Stocker le chemin du rapport
        self.visualizations["detailed_report"] = report_html
        
        return report_html

# Exemple d'utilisation:
"""
visualizer = PredictionVisualizer(explainer)
dashboard_path = visualizer.create_prediction_dashboard(predictions)
report_path = visualizer.create_detailed_report(prediction, explanations)
""" 