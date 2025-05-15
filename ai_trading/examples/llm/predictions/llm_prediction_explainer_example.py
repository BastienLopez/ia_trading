#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'explication des prédictions de marché basées sur les LLM.

Ce script démontre comment utiliser les modèles de langage pour expliquer
les prédictions de marché et fournir des insights contextuels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import des modules de prédiction LLM
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.prediction_model import PredictionModel
from ai_trading.llm.explainer.llm_explainer import LLMExplainer

def load_sample_data():
    """Charge des données d'exemple pour les démonstrations."""
    # Créer des données simulées
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simuler un prix avec tendance et bruit
    trend = np.concatenate([
        np.linspace(50000, 55000, 15),
        np.linspace(55000, 48000, 16)
    ])
    noise = np.random.normal(0, 500, 31)
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'close': trend + noise,
        'volume': np.random.uniform(1000, 5000, 31) * (1 + 0.1 * np.sin(np.linspace(0, 6, 31)))
    }, index=dates)
    
    # Ajouter des nouvelles et des données de sentiment
    news_headlines = [
        "La SEC approuve les ETF Bitcoin",
        "Les inquiétudes réglementaires pèsent sur le marché crypto",
        "La banque centrale annonce une hausse des taux d'intérêt",
        "Une grande entreprise technologique investit en Bitcoin",
        "Les traders inquiets face à la volatilité du marché"
    ]
    
    sentiment_scores = np.random.uniform(-0.8, 0.8, 5)
    
    news_data = pd.DataFrame({
        'headline': news_headlines,
        'sentiment': sentiment_scores,
        'date': [dates[5], dates[10], dates[15], dates[20], dates[25]]
    })
    
    return data, news_data

def prediction_explainer_example():
    """Exemple d'utilisation de l'explainer LLM pour les prédictions."""
    # Charger les données d'exemple
    market_data, news_data = load_sample_data()
    
    # Initialiser le prédicteur de marché et le modèle
    predictor = MarketPredictor(
        model_type='combined',
        sentiment_integration=True,
        uncertainty_calibration=True
    )
    
    # Initialiser l'explainer LLM
    explainer = LLMExplainer(
        model_name='advanced',
        context_inclusion=True,
        visualization_support=True
    )
    
    # Effectuer une prédiction
    print("Génération de prédictions de marché...")
    prediction = predictor.predict(
        market_data=market_data,
        news_data=news_data,
        horizon=7,  # Prédiction sur 7 jours
        confidence_level=0.9
    )
    
    # Obtenir une explication de la prédiction
    explanation = explainer.explain_prediction(
        prediction=prediction,
        market_data=market_data,
        news_data=news_data,
        level_of_detail='detailed',
        audience='investor'
    )
    
    # Afficher l'explication
    print("\n== Explication de la prédiction ==")
    print(explanation['summary'])
    
    # Afficher les facteurs les plus importants
    print("\n== Facteurs influençant la prédiction ==")
    factors = explanation['factors']
    for i, factor in enumerate(factors[:5], 1):
        print(f"{i}. {factor['description']} (impact: {factor['impact']:.2f})")
    
    # Visualiser la prédiction et son explication
    plt.figure(figsize=(12, 8))
    
    # Tracer les prix historiques
    plt.subplot(2, 1, 1)
    plt.plot(market_data.index, market_data['close'], label='Prix historiques', color='blue')
    
    # Tracer la prédiction
    pred_dates = pd.date_range(start=market_data.index[-1], periods=8)[1:]
    pred_values = prediction['values']
    pred_lower = prediction['lower_bound']
    pred_upper = prediction['upper_bound']
    
    plt.plot(pred_dates, pred_values, label='Prédiction', color='red', linestyle='--')
    plt.fill_between(pred_dates, pred_lower, pred_upper, color='red', alpha=0.2, label='Intervalle de confiance')
    
    # Ajouter les événements d'actualité
    for _, news in news_data.iterrows():
        plt.axvline(x=news['date'], color='green', linestyle=':')
        plt.text(news['date'], market_data['close'].max(), news['headline'], rotation=90, verticalalignment='top')
    
    plt.title('Prédiction de prix avec explicabilité')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.grid(True)
    plt.legend()
    
    # Tracer l'impact des facteurs
    plt.subplot(2, 1, 2)
    
    factor_names = [f['name'] for f in factors[:5]]
    factor_impacts = [f['impact'] for f in factors[:5]]
    
    colors = ['green' if impact > 0 else 'red' for impact in factor_impacts]
    plt.barh(factor_names, factor_impacts, color=colors)
    
    plt.title('Impact des facteurs sur la prédiction')
    plt.xlabel('Impact')
    plt.ylabel('Facteur')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Enregistrer le graphique
    plt.savefig("examples/llm/predictions/prediction_explanation.png")
    plt.close()
    
    print("Graphique d'explication de prédiction enregistré sous 'examples/llm/predictions/prediction_explanation.png'")
    
    return explanation

if __name__ == "__main__":
    explanation = prediction_explainer_example() 