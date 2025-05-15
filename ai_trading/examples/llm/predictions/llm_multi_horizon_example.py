#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple de prédictions multi-horizons basées sur les LLM.

Ce script démontre comment utiliser les modèles de langage pour générer
des prédictions de marché sur plusieurs horizons temporels.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importation des classes nécessaires
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor

def generate_market_data():
    """Génère des données de marché simulées pour l'exemple."""
    # Période de simulation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simuler une tendance de prix
    n = len(dates)
    base_trend = np.concatenate([
        np.linspace(45000, 50000, n//3),
        np.linspace(50000, 48000, n//3),
        np.linspace(48000, 53000, n - 2*(n//3))
    ])
    
    # Ajouter de la volatilité
    volatility = np.random.normal(0, 800, n)
    prices = base_trend + volatility
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'open': prices - np.random.uniform(0, 200, n),
        'high': prices + np.random.uniform(100, 500, n),
        'low': prices - np.random.uniform(100, 500, n),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n) * (1 + 0.2 * np.sin(np.linspace(0, 12, n)))
    }, index=dates)
    
    # Ajouter des indicateurs techniques
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_30'] = data['close'].rolling(window=30).mean()
    data['rsi'] = calculate_rsi(data['close'])
    data['volatility'] = data['close'].rolling(window=10).std()
    
    # Supprimer les NaN
    data = data.dropna()
    
    return data

def calculate_rsi(prices, period=14):
    """Calcule l'indicateur RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def gather_market_news(dates):
    """Simule la collecte d'actualités du marché."""
    # Simuler des événements d'actualité pour certaines dates
    news_events = [
        {"date": dates[10], "headline": "La FED annonce une nouvelle politique monétaire", "sentiment": -0.3},
        {"date": dates[20], "headline": "Adoption croissante des cryptomonnaies par les institutions", "sentiment": 0.7},
        {"date": dates[30], "headline": "Nouvelles régulations proposées pour le marché crypto", "sentiment": -0.5},
        {"date": dates[40], "headline": "Grande entreprise technologique annonce des investissements en Bitcoin", "sentiment": 0.8},
        {"date": dates[50], "headline": "Inquiétudes sur l'inflation affectent les marchés financiers", "sentiment": -0.4}
    ]
    
    # Créer un DataFrame
    news_df = pd.DataFrame(news_events)
    
    return news_df

def multi_horizon_prediction_example():
    """Exemple de prédictions multi-horizons avec LLM."""
    # Générer des données de marché
    market_data = generate_market_data()
    
    # Simuler des nouvelles du marché
    news_data = gather_market_news(market_data.index)
    
    # Initialiser le prédicteur multi-horizons
    predictor = MultiHorizonPredictor(
        base_model_type='combined',
        uncertainty_calibration=True,
        sentiment_integration=True,
        include_conflicting_scenarios=True
    )
    
    # Définir les horizons de prédiction
    horizons = {
        'short_term': 3,   # 3 jours
        'medium_term': 7,  # 7 jours
        'long_term': 14    # 14 jours
    }
    
    # Générer des prédictions pour chaque horizon
    print("Génération de prédictions multi-horizons...")
    predictions = predictor.predict_multi_horizon(
        market_data=market_data,
        news_data=news_data,
        horizons=horizons,
        confidence_level=0.9
    )
    
    # Analyser la cohérence entre les horizons
    consistency = predictor.analyze_consistency(predictions)
    
    # Afficher les résultats
    print("\n== Prédictions multi-horizons ==")
    for horizon, prediction in predictions.items():
        print(f"\n{horizon.upper()}:")
        print(f"  Direction: {prediction['direction']}")
        print(f"  Confiance: {prediction['confidence']:.2f}")
        print(f"  Facteurs clés: {', '.join(prediction['key_factors'][:3])}")
    
    # Afficher l'analyse de cohérence
    print("\n== Analyse de cohérence ==")
    print(f"Cohérence globale: {consistency['overall_consistency']:.2f}")
    print(f"Direction dominante: {consistency['dominant_direction']}")
    print(f"Fiabilité: {consistency['reliability']:.2f}")
    
    # Visualiser les prédictions
    plt.figure(figsize=(14, 10))
    
    # Tracer les prix historiques
    plt.subplot(2, 1, 1)
    plt.plot(market_data.index, market_data['close'], label='Prix historiques', color='blue')
    plt.plot(market_data.index, market_data['sma_10'], label='SMA 10', color='orange', alpha=0.7)
    plt.plot(market_data.index, market_data['sma_30'], label='SMA 30', color='green', alpha=0.7)
    
    # Tracer les prédictions pour chaque horizon
    last_date = market_data.index[-1]
    colors = {'short_term': 'red', 'medium_term': 'purple', 'long_term': 'brown'}
    
    for horizon, prediction in predictions.items():
        horizon_days = horizons[horizon]
        pred_dates = pd.date_range(start=last_date, periods=horizon_days+1)[1:]
        
        plt.plot(pred_dates, prediction['values'], 
                 label=f'Prédiction {horizon} ({horizon_days}j)', 
                 color=colors[horizon], 
                 linestyle='--')
        
        plt.fill_between(
            pred_dates, 
            prediction['lower_bound'], 
            prediction['upper_bound'], 
            color=colors[horizon], 
            alpha=0.2
        )
    
    # Ajouter les événements d'actualité
    ylim = plt.ylim()
    for _, news in news_data.iterrows():
        plt.axvline(x=news['date'], color='gray', linestyle=':', alpha=0.7)
        plt.text(news['date'], ylim[1]*0.98, news['headline'], 
                 rotation=90, fontsize=8, verticalalignment='top')
    
    plt.title('Prédictions de prix multi-horizons')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.grid(True)
    plt.legend(loc='upper left')
    
    # Tracer le graphique de cohérence
    plt.subplot(2, 1, 2)
    
    # Données pour le graphique
    horizon_names = list(predictions.keys())
    confidence_values = [predictions[h]['confidence'] for h in horizon_names]
    directions = [predictions[h]['direction'] for h in horizon_names]
    
    # Mapper les directions à des valeurs numériques pour la visualisation
    direction_map = {'bearish': -1, 'slightly_bearish': -0.5, 'neutral': 0, 
                     'slightly_bullish': 0.5, 'bullish': 1}
    direction_values = [direction_map[d] for d in directions]
    
    # Créer le graphique à barres
    bar_colors = ['red' if d < 0 else 'green' if d > 0 else 'gray' for d in direction_values]
    bars = plt.bar(horizon_names, direction_values, color=bar_colors, alpha=0.7)
    
    # Ajouter les valeurs de confiance
    for i, (bar, conf) in enumerate(zip(bars, confidence_values)):
        plt.text(i, bar.get_height() + 0.1 * np.sign(bar.get_height()), 
                 f'Conf: {conf:.2f}', ha='center', va='bottom' if bar.get_height() > 0 else 'top')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Directions de marché par horizon (avec confiance)')
    plt.ylabel('Direction (-1: baissier, +1: haussier)')
    plt.grid(True, axis='y')
    
    # Ajouter un texte de cohérence
    plt.figtext(0.5, 0.03, 
                f"Cohérence globale: {consistency['overall_consistency']:.2f} | "
                f"Direction dominante: {consistency['dominant_direction']} | "
                f"Fiabilité: {consistency['reliability']:.2f}",
                ha='center', bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    
    # Enregistrer le graphique
    plt.savefig("examples/llm/predictions/multi_horizon_predictions.png")
    plt.close()
    
    print("Graphique des prédictions multi-horizons enregistré sous 'examples/llm/predictions/multi_horizon_predictions.png'")
    
    return predictions, consistency

if __name__ == "__main__":
    predictions, consistency = multi_horizon_prediction_example() 