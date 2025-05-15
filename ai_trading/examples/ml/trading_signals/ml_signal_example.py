"""
Exemple d'utilisation des fonctionnalités avancées du modèle MLSignalModel
avec auto-évaluation et ajustement.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading.ml.trading_signals import MLSignalModel, Signal, SignalType

# Créer des données d'exemple
def generate_sample_data(n_samples=250, start_date=None, trend="bullish"):
    """Génère des données OHLCV pour tester le modèle."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_samples)
    
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Créer une tendance de base en fonction du paramètre
    if trend == "bullish":
        base_trend = np.linspace(100, 150, n_samples)
    elif trend == "bearish":
        base_trend = np.linspace(150, 100, n_samples)
    elif trend == "volatile":
        base_trend = 125 + 25 * np.sin(np.linspace(0, 3*np.pi, n_samples))
    else:  # sideways
        base_trend = np.full(n_samples, 125)
    
    # Ajouter du bruit
    noise = np.random.normal(0, 3, n_samples)
    close_prices = base_trend + noise
    
    # Créer des variations pour high, low, open basées sur close
    high_prices = close_prices + np.random.uniform(0, 3, n_samples)
    low_prices = close_prices - np.random.uniform(0, 3, n_samples)
    open_prices = close_prices - np.random.uniform(-2, 2, n_samples)
    
    # Volume avec quelques pics
    volume = np.random.uniform(1000, 5000, n_samples)
    volume[np.random.choice(range(n_samples), 10)] *= 3  # Quelques pics de volume
    
    # Assembler en DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return df

# Générer des données pour différentes phases de marché
training_data = generate_sample_data(250, trend="bullish")
validation_data = generate_sample_data(50, 
                                       start_date=training_data.index[-1] + timedelta(days=1), 
                                       trend="bullish")
recent_data = generate_sample_data(30, 
                                  start_date=validation_data.index[-1] + timedelta(days=1), 
                                  trend="volatile")
test_data = generate_sample_data(20, 
                                start_date=recent_data.index[-1] + timedelta(days=1), 
                                trend="volatile")

# Définir le symbole et le timeframe
symbol = "BTC/USD"
timeframe = "1d"

print(f"Démonstration du modèle MLSignalModel avec auto-évaluation et ajustement")
print(f"Génération des signaux de trading pour {symbol} en {timeframe}")
print(f"Données d'entraînement: {len(training_data)} points")
print(f"Données de validation: {len(validation_data)} points")
print(f"Données récentes: {len(recent_data)} points")
print(f"Données de test: {len(test_data)} points")

# 1. Initialiser et entraîner le modèle
model = MLSignalModel()
print("\n1. Entraînement du modèle...")
model.train(training_data, symbol, timeframe)
print(f"   Régime de marché détecté: {model.market_regime}")

# 2. Évaluer le modèle sur les données de validation
print("\n2. Évaluation initiale du modèle...")
try:
    initial_metrics = model.evaluate(validation_data, symbol, timeframe)
    print(f"   Accuracy globale: {initial_metrics.get('accuracy', 'N/A')}")
    print(f"   Précision des signaux d'achat: {initial_metrics.get('buy', {}).get('precision', 'N/A')}")
    print(f"   Précision des signaux de vente: {initial_metrics.get('sell', {}).get('precision', 'N/A')}")
except Exception as e:
    print(f"   Erreur d'évaluation: {str(e)}")

# 3. Évaluation détaillée par modèle
print("\n3. Évaluation détaillée par modèle...")
try:
    detailed_metrics = model.evaluate_with_model_breakdown(validation_data, symbol, timeframe)
    for model_name, metrics in detailed_metrics.get('model_metrics', {}).items():
        print(f"   {model_name}: Accuracy={metrics.get('accuracy', 'N/A')}, "
              f"Buy precision={metrics.get('buy_precision', 'N/A')}, "
              f"Sell precision={metrics.get('sell_precision', 'N/A')}")
except Exception as e:
    print(f"   Erreur d'évaluation détaillée: {str(e)}")

# 4. Auto-ajustement des poids basé sur les performances
print("\n4. Auto-ajustement des poids...")
try:
    # Simuler des métriques détaillées pour démonstration
    mock_metrics = {
        "model_metrics": {
            "random_forest": {"accuracy": 0.75},
            "gradient_boosting": {"accuracy": 0.65}
        }
    }
    historical_performance = [mock_metrics, mock_metrics]  # Deux cycles d'évaluation simulés
    result = model.auto_adjust_weights(historical_performance, symbol, timeframe)
    print(f"   Résultat de l'ajustement: {result}")
    print(f"   Nouveaux poids: {model.config['ensemble_weights']}")
except Exception as e:
    print(f"   Erreur d'auto-ajustement: {str(e)}")

# 5. Adaptation aux conditions de marché actuelles
print("\n5. Adaptation aux conditions de marché...")
print(f"   Régime de marché initial: {model.market_regime}")
print(f"   Seuil de confiance initial: {model.config['min_confidence_threshold']}")
print(f"   Horizon de prédiction initial: {model.config['prediction_horizon']}")
try:
    # Générer des données plus volatiles pour forcer un changement de régime
    volatile_data = generate_sample_data(30, trend="volatile")
    for i in range(len(volatile_data)):
        if i % 2 == 0:
            volatile_data.iloc[i, volatile_data.columns.get_loc('close')] *= 1.1
        else:
            volatile_data.iloc[i, volatile_data.columns.get_loc('close')] *= 0.9
    
    result = model.adapt_to_market_conditions(volatile_data, symbol, timeframe)
    print(f"   Résultat de l'adaptation: {result}")
    print(f"   Nouveau régime de marché: {model.market_regime}")
    print(f"   Nouveau seuil de confiance: {model.config['min_confidence_threshold']}")
    print(f"   Nouvel horizon de prédiction: {model.config['prediction_horizon']}")
except Exception as e:
    print(f"   Erreur d'adaptation: {str(e)}")

# 6. Générer des signaux sur les données de test
print("\n6. Génération des signaux de trading...")
try:
    signals = model.predict(test_data, symbol, timeframe)
    print(f"   {len(signals)} signaux générés:")
    for signal in signals[:3]:  # Afficher les 3 premiers signaux
        print(f"   - {signal}")
except Exception as e:
    print(f"   Erreur de prédiction: {str(e)}")

# 7. Simuler des résultats pour la calibration de confiance
print("\n7. Calibration des scores de confiance...")
try:
    # Créer manuellement des signaux passés
    past_signals = [
        Signal(SignalType.BUY, symbol, training_data.index[-20], training_data['close'].iloc[-20], 0.8, "RSI", timeframe),
        Signal(SignalType.SELL, symbol, training_data.index[-15], training_data['close'].iloc[-15], 0.7, "MACD", timeframe),
        Signal(SignalType.BUY, symbol, training_data.index[-10], training_data['close'].iloc[-10], 0.9, "Bollinger", timeframe)
    ]
    
    model.calibrate_confidence(past_signals, validation_data)
    print(f"   Facteurs de confiance calculés: {model.confidence_factors if hasattr(model, 'confidence_factors') else 'Aucun'}")
except Exception as e:
    print(f"   Erreur de calibration: {str(e)}")

# 8. Vérification de l'influence de l'adaptabilité sur les performances
print("\n8. Résumé des améliorations apportées par l'auto-ajustement...")
print(f"   - Détection du régime de marché: {model.market_regime}")
print(f"   - Ajustement des poids d'ensemble: {model.config['ensemble_weights']}")
print(f"   - Adaptation du seuil de confiance: {model.config['min_confidence_threshold']}")
print(f"   - Adaptation de l'horizon de prédiction: {model.config['prediction_horizon']}")
print(f"   - Calibration des scores de confiance: {hasattr(model, 'confidence_factors')}")

print("\nProcessus terminé. Le modèle MLSignalModel démontre des capacités d'auto-adaptation.") 