#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du module d'interprétabilité des prédictions.

Ce script démontre comment utiliser les modules de PredictionExplainer,
PredictionVisualizer et PredictionReporter pour interpréter et visualiser
les prédictions générées par les modèles LLM.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importation des classes nécessaires
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor
from ai_trading.llm.predictions.prediction_explainer import PredictionExplainer
from ai_trading.llm.predictions.visualization import PredictionVisualizer
from ai_trading.llm.predictions.reporting import PredictionReporter

def generate_sample_market_data(size=30):
    """
    Génère des données de marché simulées pour l'exemple.
    
    Args:
        size: Nombre de points de données à générer
        
    Returns:
        DataFrame avec des données de marché simulées
    """
    # Générer une tendance de base
    base_trend = np.linspace(0, 1, size) * 100 + 10000
    
    # Ajouter du bruit et de la volatilité
    noise = np.random.normal(0, 300, size)
    trend = base_trend + noise
    
    # Calculer les données OHLCV
    dates = pd.date_range(end=datetime.now(), periods=size)
    
    df = pd.DataFrame({
        'date': dates,
        'open': trend - np.random.normal(0, 50, size),
        'high': trend + np.random.normal(100, 50, size),
        'low': trend - np.random.normal(100, 50, size),
        'close': trend,
        'volume': np.random.normal(1000, 300, size) * (1 + np.random.random(size)),
        'rsi': np.random.uniform(30, 70, size),
        'macd': np.random.normal(0, 10, size),
        'ema_12': trend + np.random.normal(0, 20, size),
        'sma_50': base_trend
    })
    
    # Ajouter des colonnes dérivées
    df['close_pct_change'] = df['close'].pct_change() * 100
    df['volume_change'] = df['volume'].pct_change() * 100
    df['volatility'] = df['high'] - df['low']
    
    # Définir date comme index
    df.set_index('date', inplace=True)
    
    return df

def create_sample_prediction():
    """
    Crée une prédiction simulée pour l'exemple.
    
    Returns:
        Dictionnaire contenant une prédiction simulée
    """
    return {
        "id": f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "asset": "BTC",
        "direction": "bullish",
        "confidence": "medium",
        "timeframe": "24h",
        "factors": [
            "Prix au-dessus de la moyenne mobile à 50 jours",
            "Volume en augmentation de 15%",
            "RSI en zone neutre à 55",
            "Sentiment positif sur les réseaux sociaux",
            "Résistance majeure à 12000 USD"
        ],
        "contradictions": [
            "MACD montre un signal de divergence baissière",
            "Niveau de surachat sur certains indicateurs"
        ],
        "volatility": "medium",
        "explanation": "Les facteurs haussiers semblent l'emporter sur les signaux baissiers. Le prix a montré une bonne résilience au-dessus des supports clés et le volume confirme la tendance haussière. Cependant, des signes de surachat pourraient limiter la hausse à court terme."
    }

def create_sample_multi_horizon_predictions():
    """
    Crée des prédictions multi-horizons simulées pour l'exemple.
    
    Returns:
        Dictionnaire contenant des prédictions par horizon
    """
    asset = "BTC"
    base_prediction = create_sample_prediction()
    
    # Prédictions pour différents horizons avec des variations
    predictions = {
        "short_term": {
            **base_prediction,
            "timeframe": "4h",
            "direction": "neutral",
            "confidence": "medium",
            "factors": [
                "Consolidation autour de la moyenne mobile à 20 périodes",
                "Volume en baisse",
                "RSI en zone neutre à 49"
            ],
            "contradictions": [
                "Support proche testé récemment"
            ],
            "volatility": "low",
            "explanation": "Le prix semble consolidé à court terme avec peu de direction claire."
        },
        "medium_term": {
            **base_prediction,
            "timeframe": "24h",
            "direction": "bullish",
            "confidence": "medium"
        },
        "long_term": {
            **base_prediction,
            "timeframe": "7d",
            "direction": "bullish",
            "confidence": "high",
            "factors": [
                "Tendance haussière intacte sur le graphique hebdomadaire",
                "Support majeur à 9500 USD",
                "Accumulation par les investisseurs institutionnels",
                "Analyse fondamentale positive"
            ],
            "contradictions": [
                "Résistance historique à 13000 USD"
            ],
            "volatility": "medium",
            "explanation": "La tendance à long terme reste haussière avec des fondamentaux solides soutenant le prix."
        }
    }
    
    return predictions

def create_sample_consistency_analysis():
    """
    Crée une analyse de cohérence simulée pour l'exemple.
    
    Returns:
        Dictionnaire contenant une analyse de cohérence
    """
    return {
        "overall_consistency": 0.75,
        "direction_agreement": "bullish",
        "horizon_analysis": {
            "short_term": {
                "overall_direction": "neutral",
                "consistency": 0.6,
                "direction_counts": {"bearish": 1, "neutral": 3, "bullish": 1}
            },
            "medium_term": {
                "overall_direction": "bullish",
                "consistency": 0.7,
                "direction_counts": {"bearish": 0, "neutral": 1, "bullish": 4}
            },
            "long_term": {
                "overall_direction": "bullish",
                "consistency": 0.9,
                "direction_counts": {"bearish": 0, "neutral": 0, "bullish": 5}
            }
        },
        "trading_signals": {
            "signal": "buy",
            "strength": "medium",
            "description": "Signal d'achat généré par la cohérence haussière entre les horizons moyen et long terme."
        }
    }

def main():
    """
    Fonction principale de démonstration.
    """
    print("=== Démonstration du module d'interprétabilité des prédictions ===")
    
    # 1. Générer des données simulées
    print("\n1. Génération de données de marché simulées...")
    market_data = generate_sample_market_data(size=60)
    print(f"  Données générées: {market_data.shape[0]} lignes, {market_data.shape[1]} colonnes")
    
    # 2. Créer une prédiction simulée
    print("\n2. Création d'une prédiction simulée...")
    prediction = create_sample_prediction()
    print(f"  Prédiction pour {prediction['asset']}: {prediction['direction']} ({prediction['confidence']})")
    
    # 3. Créer des prédictions multi-horizons
    print("\n3. Création de prédictions multi-horizons...")
    multi_horizon_predictions = create_sample_multi_horizon_predictions()
    print(f"  Prédictions générées pour {len(multi_horizon_predictions)} horizons temporels")
    
    # 4. Créer une analyse de cohérence
    print("\n4. Création d'une analyse de cohérence...")
    consistency_analysis = create_sample_consistency_analysis()
    print(f"  Cohérence globale: {consistency_analysis['overall_consistency']}")
    print(f"  Signal de trading: {consistency_analysis['trading_signals']['signal']} ({consistency_analysis['trading_signals']['strength']})")
    
    # 5. Initialiser l'explainer
    print("\n5. Initialisation de l'explainer...")
    explainer = PredictionExplainer(market_data=market_data)
    
    # 6. Générer des explications
    print("\n6. Génération d'explications...")
    shap_explanation = explainer.explain_with_shap(prediction, market_data)
    lime_explanation = explainer.explain_with_lime(prediction, market_data)
    
    # 7. Générer un rapport
    print("\n7. Génération d'un rapport complet...")
    report = explainer.generate_report(prediction, ["shap", "lime", "llm"], "json")
    print(f"  Rapport généré avec {len(report['explanations'])} types d'explications")
    
    # 8. Créer le visualiseur
    print("\n8. Initialisation du visualiseur...")
    visualizer = PredictionVisualizer(explainer)
    
    # 9. Créer des visualisations
    print("\n9. Création de visualisations...")
    
    # Créer un répertoire pour les sorties
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer les visualisations
    dashboard_path = visualizer.create_prediction_dashboard(
        multi_horizon_predictions, 
        market_data,
        os.path.join(output_dir, "prediction_dashboard.html")
    )
    
    explanation_path = visualizer.create_explanation_dashboard(
        prediction,
        {"shap": shap_explanation, "lime": lime_explanation, "llm": explainer._explain_llm_prediction(prediction)},
        os.path.join(output_dir, "explanation_dashboard.html")
    )
    
    consistency_path = visualizer.create_consistency_visualization(
        consistency_analysis,
        os.path.join(output_dir, "consistency_visualization.html")
    )
    
    print(f"  Tableau de bord des prédictions: {dashboard_path}")
    print(f"  Tableau de bord d'explication: {explanation_path}")
    print(f"  Visualisation de cohérence: {consistency_path}")
    
    # 10. Générer un rapport détaillé avec le reporter
    print("\n10. Génération d'un rapport détaillé...")
    reporter = PredictionReporter(explainer, visualizer)
    
    # Rapport pour une seule prédiction
    single_report_path = reporter.generate_prediction_report(
        prediction,
        market_data,
        os.path.join(output_dir, "single_prediction"),
        "html"
    )
    
    # Rapport pour les prédictions multi-horizons
    multi_report_path = reporter.generate_multi_horizon_report(
        multi_horizon_predictions,
        consistency_analysis,
        market_data,
        os.path.join(output_dir, "multi_horizon"),
        "html"
    )
    
    print(f"  Rapport de prédiction unique: {single_report_path}")
    print(f"  Rapport multi-horizons: {multi_report_path}")
    
    print("\n=== Démonstration terminée ===")
    print(f"Tous les fichiers de visualisation ont été sauvegardés dans: {output_dir}")
    
    return {
        "market_data": market_data,
        "prediction": prediction,
        "multi_horizon_predictions": multi_horizon_predictions,
        "consistency_analysis": consistency_analysis,
        "explainer": explainer,
        "visualizer": visualizer,
        "reporter": reporter,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    results = main() 