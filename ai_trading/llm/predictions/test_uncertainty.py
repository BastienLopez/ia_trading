#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test simple du module UncertaintyCalibrator.
"""

import numpy as np
import pandas as pd
from uncertainty_calibration import UncertaintyCalibrator


def main():
    """Test de base du calibrateur d'incertitude."""
    print("Test du module UncertaintyCalibrator")

    # Créer des données de marché simulées
    market_data = pd.DataFrame(
        {
            "open": np.random.rand(10) * 100,
            "high": np.random.rand(10) * 100 + 10,
            "low": np.random.rand(10) * 100 - 10,
            "close": np.random.rand(10) * 100,
            "volume": np.random.rand(10) * 1000,
        }
    )

    # Créer une prédiction simulée
    prediction = {
        "id": "pred_123",
        "asset": "BTC",
        "direction": "bullish",
        "confidence": "medium",
        "timeframe": "24h",
    }

    # Initialiser le calibrateur
    calibrator = UncertaintyCalibrator(market_data=market_data)
    print("Calibrateur initialisé avec succès")

    # Calculer des intervalles de confiance
    ci = calibrator.calculate_confidence_intervals(prediction, method="parametric")
    print(
        f"Intervalle de confiance calculé: [{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]"
    )

    # Estimer la distribution de probabilité
    dist = calibrator.estimate_probability_distribution(prediction)
    print(
        f"Distribution estimée avec succès, direction la plus probable: {dist['most_likely_direction']}"
    )

    print("Test terminé avec succès")


if __name__ == "__main__":
    main()
