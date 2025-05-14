#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du module d'adaptation en temps réel.

Ce script montre comment utiliser le module RealTimeAdapter 
pour mettre à jour les prédictions en temps réel en fonction
des nouvelles données de marché.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading

# Ajouter le répertoire racine au chemin d'importation Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importer les modules nécessaires
from ai_trading.llm.predictions.model_ensemble import ModelEnsemble
from ai_trading.llm.predictions.real_time_adapter import RealTimeAdapter, RealTimeMarketMonitor

# Modèle simple pour la démonstration
class SimpleMarketPredictor:
    """
    Modèle de prédiction simple pour la démonstration.
    
    Ce modèle prédit la direction du marché en fonction de la tendance
    récente et de la volatilité.
    """
    
    def __init__(self, bias=0.0, volatility_weight=0.5):
        """
        Initialise le prédicteur avec un biais et une pondération de volatilité.
        
        Args:
            bias: Biais de la prédiction (-1 à 1, où -1 = bearish, 1 = bullish)
            volatility_weight: Importance de la volatilité dans la prédiction
        """
        self.bias = bias
        self.volatility_weight = volatility_weight
        self.name = f"SimplePredictor(bias={bias:.2f})"
    
    def predict(self, market_data):
        """
        Prédit la direction du marché en fonction des données.
        
        Args:
            market_data: Dictionnaire contenant les données de marché
            
        Returns:
            Dict: Prédiction avec direction et confiance
        """
        # Extraire les données pertinentes
        price_change_1h = market_data.get('price_change_1h', 0.0)
        volatility = market_data.get('volatility', 0.01)
        trend = market_data.get('trend', 'neutral')
        
        # Calculer un score de direction (-1 à 1)
        trend_score = 0.0
        if trend == 'bullish':
            trend_score = 0.5
        elif trend == 'bearish':
            trend_score = -0.5
            
        # Ajouter la contribution du changement récent
        if price_change_1h is not None:
            trend_score += price_change_1h * 5  # Amplifier l'effet du changement horaire
        
        # Ajouter le biais
        score = trend_score + self.bias
        
        # Limiter le score à [-1, 1]
        score = max(-1.0, min(1.0, score))
        
        # Déterminer la direction en fonction du score
        if score > 0.2:
            direction = 'bullish'
        elif score < -0.2:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculer la confiance en fonction de la volatilité
        # Plus la volatilité est grande, moins la confiance est élevée
        base_confidence = 0.7 - self.volatility_weight * (volatility or 0.01)
        confidence = max(0.3, min(0.9, base_confidence))
        
        return {
            'direction': direction,
            'score': score,
            'confidence': confidence,
            'prediction_type': 'market_direction',
            'model_name': self.name
        }


def simulate_market_data(duration_seconds=60, interval_seconds=5):
    """
    Simule des données de marché avec une tendance et des fluctuations.
    
    Args:
        duration_seconds: Durée totale de la simulation en secondes
        interval_seconds: Intervalle entre les points de données en secondes
        
    Yields:
        Tuple: (prix, volume, horodatage)
    """
    # Paramètres de simulation
    base_price = 50000.0  # Prix de base
    trend = 0.001  # Tendance par seconde (positif = haussier)
    volatility = 0.002  # Volatilité (écart-type des rendements)
    
    # Changement de tendance à mi-parcours
    trend_change_time = duration_seconds / 2
    
    start_time = datetime.now()
    current_price = base_price
    
    elapsed_seconds = 0
    while elapsed_seconds < duration_seconds:
        # Mettre à jour le temps écoulé
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        
        # Changer la tendance à mi-parcours
        if elapsed_seconds > trend_change_time and trend > 0:
            trend = -0.002  # Tendance baissière plus prononcée
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Changement de tendance: maintenant baissier!")
        
        # Calculer le nouveau prix
        price_change = current_price * (trend * interval_seconds)
        random_change = current_price * np.random.normal(0, volatility) * np.sqrt(interval_seconds)
        current_price += price_change + random_change
        
        # Générer un volume simulé
        volume = np.random.gamma(2.0, 10.0) * (1 + 0.1 * np.random.randn())
        
        timestamp = datetime.now()
        
        yield (current_price, volume, timestamp)
        
        # Attendre jusqu'au prochain intervalle
        time.sleep(interval_seconds)


def prediction_update_callback(prediction):
    """
    Callback appelé lorsqu'une nouvelle prédiction est disponible.
    
    Args:
        prediction: Dictionnaire contenant la prédiction mise à jour
    """
    direction = prediction.get('direction', 'inconnu')
    confidence = prediction.get('confidence', 0.0)
    timestamp = prediction.get('timestamp', datetime.now())
    
    # Formater l'horodatage
    time_str = timestamp.strftime('%H:%M:%S')
    
    # Afficher la prédiction
    print(f"[{time_str}] Nouvelle prédiction: {direction} (confiance: {confidence:.2f})")


def significant_change_callback(prediction):
    """
    Callback appelé lorsqu'un changement significatif est détecté.
    
    Args:
        prediction: Dictionnaire contenant la prédiction mise à jour
    """
    direction = prediction.get('direction', 'inconnu')
    confidence = prediction.get('confidence', 0.0)
    timestamp = prediction.get('timestamp', datetime.now())
    
    # Formater l'horodatage
    time_str = timestamp.strftime('%H:%M:%S')
    
    # Afficher l'alerte
    print(f"\n[{time_str}] ALERTE: Changement significatif détecté!")
    print(f"  Nouvelle direction: {direction} (confiance: {confidence:.2f})")
    
    # Afficher des informations sur le marché si disponibles
    market_state = prediction.get('market_state', {})
    if market_state:
        price = market_state.get('price', 'N/A')
        trend = market_state.get('trend', 'N/A')
        volatility = market_state.get('volatility', 'N/A')
        
        print(f"  État du marché: Prix={price:.2f}, Tendance={trend}, Volatilité={volatility:.4f}")


def market_alert_callback(alert_data):
    """
    Callback appelé lorsqu'une alerte de marché est déclenchée.
    
    Args:
        alert_data: Données de l'alerte
    """
    alert_type = alert_data.get('type', 'unknown')
    timestamp = alert_data.get('timestamp', datetime.now())
    
    # Formater l'horodatage
    time_str = timestamp.strftime('%H:%M:%S')
    
    # Afficher l'alerte en fonction de son type
    if alert_type == 'volatility_spike':
        ratio = alert_data.get('ratio', 0)
        print(f"\n[{time_str}] ALERTE: Pic de volatilité détecté! (x{ratio:.2f})")
    
    elif alert_type == 'significant_price_move':
        price_change = alert_data.get('price_change', 0) * 100  # En pourcentage
        direction = alert_data.get('direction', '?')
        print(f"\n[{time_str}] ALERTE: Mouvement de prix significatif!")
        print(f"  {direction.upper()}: {price_change:.2f}%")


def main():
    """
    Fonction principale de démonstration.
    """
    print("=== Démonstration de l'adaptation en temps réel des prédictions ===\n")
    
    # 1. Créer le moniteur de marché
    print("1. Initialisation du moniteur de marché...")
    market_monitor = RealTimeMarketMonitor(
        observation_window=10,
        volatility_threshold=1.5,
        price_move_threshold=0.02
    )
    
    # Configurer les callbacks pour les alertes
    market_monitor.set_alert_callback('volatility_spike', market_alert_callback)
    market_monitor.set_alert_callback('significant_price_move', market_alert_callback)
    
    # 2. Créer le modèle de prédiction
    print("2. Initialisation du modèle de prédiction...")
    predictor = SimpleMarketPredictor(bias=0.1, volatility_weight=0.3)
    
    # 3. Créer l'adaptateur en temps réel
    print("3. Initialisation de l'adaptateur en temps réel...")
    adapter = RealTimeAdapter(
        prediction_model=predictor,
        update_frequency=5.0,  # Mise à jour toutes les 5 secondes
        change_detection_threshold=0.15,  # Seuil de détection des changements
        backtest_mode=False
    )
    
    # Configurer les callbacks
    adapter.set_callback('update', prediction_update_callback)
    adapter.set_callback('change', significant_change_callback)
    
    # 4. Démarrer l'adaptateur
    print("4. Démarrage de l'adaptateur en temps réel...")
    adapter.start()
    
    try:
        # 5. Simuler des données de marché
        print("\n5. Simulation de données de marché en temps réel...\n")
        print("Appuyez sur Ctrl+C pour arrêter.")
        
        for price, volume, timestamp in simulate_market_data(duration_seconds=120, interval_seconds=2):
            # Mettre à jour le moniteur de marché
            market_monitor.add_market_data(price, volume, timestamp)
            
            # Obtenir l'état actuel du marché
            market_state = market_monitor.get_market_state()
            
            # Envoyer l'état du marché à l'adaptateur
            adapter.add_data(market_state, timestamp)
            
            # Afficher le prix actuel
            print(f"Prix: {price:.2f}", end="\r")
    
    except KeyboardInterrupt:
        print("\n\nInterruption utilisateur.")
    
    finally:
        # 6. Arrêter l'adaptateur
        print("\n6. Arrêt de l'adaptateur...")
        adapter.stop()
        
        # Afficher un résumé des prédictions
        print("\nRésumé des dernières prédictions:")
        history = adapter.get_prediction_history(3)
        
        for i, pred in enumerate(history):
            timestamp = pred.get('timestamp', datetime.now()).strftime('%H:%M:%S')
            direction = pred.get('direction', 'inconnu')
            confidence = pred.get('confidence', 0.0)
            
            print(f"  {i+1}. [{timestamp}] Direction: {direction} (confiance: {confidence:.2f})")
        
        print("\n=== Démonstration terminée ===")


if __name__ == "__main__":
    main() 