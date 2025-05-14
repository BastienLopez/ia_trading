#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du MultiHorizonPredictor pour générer des prédictions
et des signaux de trading sur différents horizons temporels.
"""

import os
import sys

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Importation des classes nécessaires
from ai_trading.llm.predictions.multi_horizon_predictor import MultiHorizonPredictor


def main():
    """Fonction principale d'exemple."""
    print("=== Démarrage de l'exemple MultiHorizonPredictor ===")

    # Initialisation du prédicteur
    predictor = MultiHorizonPredictor(
        llm_model="gpt-4", use_hybrid=False  # Utiliser uniquement LLM pour cet exemple
    )
    print(f"Prédicteur initialisé avec modèle: {predictor.llm_model}")

    # Définir l'actif à analyser
    asset = "BTC"
    print(f"Analyse de l'actif: {asset}")

    # Obtenir des prédictions pour différents horizons temporels
    print("\n=== Génération des prédictions ===")

    # 1. Prédictions à court terme uniquement
    print("\n> Prédictions à court terme:")
    short_term_predictions = predictor.predict_all_horizons(
        asset=asset, short_term=True, medium_term=False, long_term=False
    )
    display_predictions(short_term_predictions)

    # 2. Prédictions à moyen terme uniquement
    print("\n> Prédictions à moyen terme:")
    medium_term_predictions = predictor.predict_all_horizons(
        asset=asset, short_term=False, medium_term=True, long_term=False
    )
    display_predictions(medium_term_predictions)

    # 3. Prédictions à long terme uniquement
    print("\n> Prédictions à long terme:")
    long_term_predictions = predictor.predict_all_horizons(
        asset=asset, short_term=False, medium_term=False, long_term=True
    )
    display_predictions(long_term_predictions)

    # Combiner toutes les prédictions
    all_predictions = {}
    all_predictions.update(short_term_predictions)
    all_predictions.update(medium_term_predictions)
    all_predictions.update(long_term_predictions)

    # Analyse de cohérence et génération de signaux
    print("\n=== Analyse de cohérence multi-horizons ===")
    consistency_analysis = predictor.analyze_consistency(all_predictions)

    # Affichage de l'analyse
    print("\n> Analyse par horizon:")
    for horizon, analysis in consistency_analysis["horizon_analysis"].items():
        print(f"  {horizon.upper()}:")
        print(f"    Direction: {analysis['overall_direction']}")
        print(f"    Confiance: {analysis['confidence']}")
        print(f"    Cohérence: {analysis['consistency']:.2f}")
        print(f"    Répartition: {analysis['direction_counts']}")

    # Affichage des signaux de trading
    print("\n> Signal de trading généré:")
    signal = consistency_analysis["trading_signals"]
    print(f"  Signal: {signal['signal']}")
    print(f"  Description: {signal['description']}")
    print(
        f"  Directions: court={signal['short_term_dir']}, moyen={signal['medium_term_dir']}, long={signal['long_term_dir']}"
    )
    print(f"  Confiance: {signal['confidence']}")

    print("\n=== Fin de l'exemple ===")


def display_predictions(predictions):
    """Affiche les prédictions de manière formatée."""
    for timeframe, prediction in predictions.items():
        direction = prediction.get("direction", "unknown")
        confidence = prediction.get("confidence", "unknown")

        # Formater l'affichage selon la direction
        if direction == "bullish":
            direction_display = "\033[92m↗ BULLISH\033[0m"  # Vert
        elif direction == "bearish":
            direction_display = "\033[91m↘ BEARISH\033[0m"  # Rouge
        else:
            direction_display = "\033[93m→ NEUTRAL\033[0m"  # Jaune

        # Formater l'affichage selon la confiance
        if confidence == "high":
            confidence_display = "\033[1mHIGH\033[0m"  # Gras
        elif confidence == "medium":
            confidence_display = "MEDIUM"
        else:
            confidence_display = "\033[2mLOW\033[0m"  # Faible

        print(f"  {timeframe}: {direction_display} (Confiance: {confidence_display})")

        # Afficher les facteurs principaux
        factors = prediction.get("factors", [])
        if factors:
            print(f"    Facteurs: {', '.join(factors[:3])}")


if __name__ == "__main__":
    main()
