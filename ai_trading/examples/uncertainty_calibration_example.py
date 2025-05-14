#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du module de calibration d'incertitudes pour les prédictions.

Ce script démontre comment utiliser le module UncertaintyCalibrator pour
calibrer l'incertitude des prédictions, estimer des intervalles de confiance,
et gérer les outliers.
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
from ai_trading.llm.predictions.uncertainty_calibration import UncertaintyCalibrator

def generate_sample_market_data(size=100):
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

def create_sample_predictions(asset="BTC", num_predictions=20):
    """
    Crée un ensemble de prédictions simulées.
    
    Args:
        asset: Nom de l'actif
        num_predictions: Nombre de prédictions à générer
        
    Returns:
        Liste de dictionnaires contenant des prédictions simulées
    """
    predictions = []
    
    # Possibilités de direction et confiance
    directions = ["bearish", "neutral", "bullish"]
    confidences = ["low", "medium", "high"]
    
    # Ajouter une tendance (plus de bullish que bearish)
    direction_weights = [0.25, 0.25, 0.5]
    
    for i in range(num_predictions):
        # Générer une prédiction avec une légère tendance
        direction = np.random.choice(directions, p=direction_weights)
        confidence = np.random.choice(confidences)
        
        # Ajouter quelques outliers
        if i % 10 == 0:
            direction = "bearish" if directions.index(direction) != 0 else "bullish"
            confidence = "high"
        
        prediction = {
            "id": f"pred_{i}",
            "asset": asset,
            "direction": direction,
            "confidence": confidence,
            "timeframe": np.random.choice(["4h", "24h", "7d"]),
            "factors": [
                "Prix au-dessus de la moyenne mobile à 50 jours",
                "Volume en augmentation de 15%",
                "RSI en zone neutre à 55"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        predictions.append(prediction)
    
    return predictions

def main():
    """
    Fonction principale de démonstration.
    """
    print("=== Démonstration du module de calibration d'incertitudes ===")
    
    # 1. Générer des données simulées
    print("\n1. Génération de données de marché simulées...")
    market_data = generate_sample_market_data(size=120)
    print(f"  Données générées: {market_data.shape[0]} lignes, {market_data.shape[1]} colonnes")
    
    # 2. Créer un ensemble de prédictions
    print("\n2. Création de prédictions simulées...")
    predictions = create_sample_predictions(num_predictions=30)
    print(f"  Prédictions générées: {len(predictions)}")
    
    # 3. Initialiser le calibrateur
    print("\n3. Initialisation du calibrateur d'incertitude...")
    calibrator = UncertaintyCalibrator(market_data=market_data)
    
    # 4. Calculer les intervalles de confiance pour une prédiction
    print("\n4. Calcul des intervalles de confiance...")
    
    # Sélectionner une prédiction
    sample_prediction = predictions[0]
    print(f"  Prédiction: {sample_prediction['asset']} - {sample_prediction['direction']} ({sample_prediction['confidence']})")
    
    # Calculer avec différentes méthodes
    methods = ["bootstrap", "parametric", "bayesian"]
    for method in methods:
        confidence_interval = calibrator.calculate_confidence_intervals(
            sample_prediction, confidence_level=0.95, method=method)
        
        print(f"  Méthode {method}: [{confidence_interval['lower_direction']} à {confidence_interval['upper_direction']}]")
        print(f"    Intervalle numérique: [{confidence_interval['lower_bound']:.2f}, {confidence_interval['upper_bound']:.2f}]")
    
    # 5. Estimer la distribution de probabilité
    print("\n5. Estimation de la distribution de probabilité...")
    probability_dist = calibrator.estimate_probability_distribution(sample_prediction)
    
    print(f"  Direction la plus probable: {probability_dist['most_likely_direction']}")
    print(f"  Probabilités: Baissier={probability_dist['probabilities']['bearish']:.2f}, "
          f"Neutre={probability_dist['probabilities']['neutral']:.2f}, "
          f"Haussier={probability_dist['probabilities']['bullish']:.2f}")
    print(f"  Entropie (incertitude): {probability_dist['entropy']:.2f}")
    
    # 6. Détecter les outliers
    print("\n6. Détection des outliers...")
    outlier_results = calibrator.detect_outliers(predictions, method="z_score", threshold=2.5)
    
    print(f"  Total prédictions: {outlier_results['total_predictions']}")
    print(f"  Outliers détectés: {outlier_results['outliers_detected']} "
          f"({outlier_results['outlier_percentage']*100:.1f}%)")
    
    if outlier_results['outliers']:
        print("  Détails des outliers détectés:")
        for outlier in outlier_results['outliers']:
            print(f"    ID: {outlier['id']}, Direction: {outlier['direction_value']}, "
                  f"Score: {outlier['outlier_score']:.2f}")
    
    # 7. Calibrer les prédictions
    print("\n7. Calibration des prédictions...")
    
    for prediction in predictions[:3]:
        calibrated = calibrator.calibrate_prediction(prediction, calibration_method="platt")
        print(f"  Prédiction {prediction['id']}: {prediction['direction']} avec confiance {prediction['confidence']}")
        print(f"    Après calibration: confiance {calibrated['confidence']} (valeur: {calibrated['confidence_value']:.2f})")
    
    # 8. Créer des visualisations
    print("\n8. Création de visualisations...")
    
    # Créer un répertoire pour les sorties
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uncertainty_calibration_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualiser la distribution de probabilité
    dist_path = os.path.join(output_dir, "probability_distribution.png")
    calibrator.plot_probability_distribution(sample_prediction, distribution=probability_dist, save_path=dist_path)
    print(f"  Distribution de probabilité sauvegardée: {dist_path}")
    
    print("\n=== Démonstration terminée ===")
    print(f"Les visualisations ont été sauvegardées dans: {output_dir}")
    
    return {
        "market_data": market_data,
        "predictions": predictions,
        "calibrator": calibrator,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    results = main() 