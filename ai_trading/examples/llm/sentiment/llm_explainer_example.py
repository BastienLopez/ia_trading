#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple simple d'utilisation du module d'interprétabilité des prédictions.

Ce script montre comment utiliser le PredictionExplainer pour comprendre
et visualiser les prédictions générées par les modèles LLM.
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
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor
from ai_trading.llm.predictions.prediction_explainer import PredictionExplainer
from ai_trading.llm.predictions.visualization import PredictionVisualizer
from ai_trading.llm.predictions.reporting import PredictionReporter

def main():
    """
    Fonction principale démontrant l'utilisation du module d'interprétabilité.
    """
    print("=== Exemple d'interprétabilité des prédictions ===")
    
    # Créer des données de marché simulées
    market_data = pd.DataFrame({
        'open': np.random.rand(20) * 100 + 10000,
        'high': np.random.rand(20) * 150 + 10050,
        'low': np.random.rand(20) * 90 + 9950,
        'close': np.random.rand(20) * 100 + 10000,
        'volume': np.random.rand(20) * 1000 + 5000,
        'rsi': np.random.rand(20) * 40 + 30,
        'macd': np.random.rand(20) * 10 - 5,
        'ema_12': np.random.rand(20) * 100 + 10000,
        'sma_50': np.random.rand(20) * 50 + 10000
    })
    
    # Créer une prédiction simulée
    prediction = {
        "id": "pred_123",
        "asset": "BTC",
        "direction": "bullish",
        "confidence": "medium",
        "timeframe": "24h",
        "factors": [
            "Prix au-dessus de la moyenne mobile à 50 jours",
            "Volume en augmentation de 15%",
            "RSI en zone neutre à 55",
            "Sentiment positif sur les réseaux sociaux",
            "Support majeur tenu à 9800 USD"
        ],
        "contradictions": [
            "MACD montre un signal de divergence baissière",
            "Résistance proche à 10500 USD"
        ],
        "volatility": "medium",
        "explanation": "Les facteurs haussiers semblent l'emporter sur les signaux baissiers. Le prix a montré une bonne résilience au-dessus des supports clés."
    }
    
    # Créer des prédictions multi-horizons
    predictions = {
        "short_term": {
            **prediction,
            "timeframe": "4h",
            "direction": "neutral"
        },
        "medium_term": prediction,
        "long_term": {
            **prediction,
            "timeframe": "7d",
            "direction": "bullish",
            "confidence": "high"
        }
    }
    
    # Analyse de cohérence simulée
    consistency_analysis = {
        "overall_consistency": 0.75,
        "horizon_analysis": {
            "short_term": {"overall_direction": "neutral", "consistency": 0.6},
            "medium_term": {"overall_direction": "bullish", "consistency": 0.7},
            "long_term": {"overall_direction": "bullish", "consistency": 0.9}
        },
        "trading_signals": {
            "signal": "buy",
            "description": "Signal d'achat basé sur la cohérence des prédictions à moyen et long terme."
        }
    }
    
    # 1. Initialiser l'explainer
    print("\n1. Création de l'explainer...")
    explainer = PredictionExplainer(market_data=market_data)
    
    # 2. Générer des explications LLM
    print("\n2. Génération d'explications LLM...")
    llm_explanation = explainer._explain_llm_prediction(prediction)
    print(f"  Facteurs identifiés: {len(llm_explanation['factors'])}")
    
    # 3. Visualiser l'importance des facteurs
    print("\n3. Visualisation de l'importance des facteurs...")
    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    explainer.plot_factor_importance(prediction, os.path.join(output_dir, "factor_importance.png"))
    
    # 4. Initialiser le visualiseur
    print("\n4. Création du visualiseur...")
    visualizer = PredictionVisualizer(explainer)
    
    # 5. Créer un tableau de bord
    print("\n5. Création d'un tableau de bord de prédictions...")
    dashboard_path = visualizer.create_prediction_dashboard(
        predictions,
        market_data,
        os.path.join(output_dir, "prediction_dashboard.html")
    )
    print(f"  Tableau de bord sauvegardé: {dashboard_path}")
    
    # 6. Visualiser la cohérence
    print("\n6. Visualisation de la cohérence des prédictions...")
    consistency_path = visualizer.create_consistency_visualization(
        consistency_analysis,
        os.path.join(output_dir, "consistency_visualization.html")
    )
    print(f"  Visualisation de cohérence sauvegardée: {consistency_path}")
    
    # 7. Générer un rapport détaillé
    print("\n7. Génération d'un rapport détaillé...")
    reporter = PredictionReporter(explainer, visualizer)
    report_path = reporter.generate_multi_horizon_report(
        predictions,
        consistency_analysis,
        market_data,
        os.path.join(output_dir, "report"),
        "html"
    )
    print(f"  Rapport détaillé sauvegardé: {report_path}")
    
    print("\n=== Fin de l'exemple ===")
    print(f"Tous les fichiers ont été sauvegardés dans: {os.path.abspath(output_dir)}")
    
    return {
        "market_data": market_data,
        "prediction": prediction,
        "predictions": predictions,
        "explainer": explainer,
        "visualizer": visualizer,
        "reporter": reporter,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    results = main() 