# Module de Signaux de Trading

Ce module génère des signaux de trading basés sur divers indicateurs techniques et utilise l'apprentissage automatique pour améliorer ces signaux.

## Composants Principaux

### 1. SignalGenerator

Une classe qui génère des signaux de trading à partir d'une série d'indicateurs techniques :
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bandes de Bollinger
- Oscillateur Stochastique
- ADX (Average Directional Index)
- Analyse de Volume

Chaque signal généré inclut :
- Le type de signal (achat/vente)
- Un score de confiance
- Le symbole et le timeframe
- L'indicateur source
- Des métadonnées supplémentaires

### 2. MLSignalModel

Un modèle d'apprentissage automatique qui améliore les signaux du générateur :
- Utilise un ensemble de modèles (Random Forest et Gradient Boosting)
- Détecte le régime de marché actuel (normal, haussier, baissier, volatile)
- Adapte les signaux au régime de marché
- Calcule des scores de confiance basés sur les probabilités des modèles

## Utilisation

```python
from ai_trading.ml.trading_signals import SignalGenerator, MLSignalModel

# Générer des signaux avec le générateur de base
generator = SignalGenerator()
signals = generator.generate_signals(data, "BTC/USD", "1h")

# Utiliser le modèle ML pour des signaux plus sophistiqués
ml_model = MLSignalModel()
ml_model.train(historical_data, "BTC/USD", "1h")
signals = ml_model.predict(current_data, "BTC/USD", "1h")

# Obtenir un score global (-1 à 1)
score = generator.score_signals(signals)
```

## Configuration

Les deux classes acceptent des dictionnaires de configuration pour personnaliser leur comportement :

```python
# Configuration personnalisée
config = {
    "rsi": {
        "period": 7,
        "overbought": 80,
        "oversold": 20
    },
    "prediction_horizon": 10,
    "min_confidence_threshold": 0.7
}

generator = SignalGenerator(config)
ml_model = MLSignalModel(config)
```

## Métriques d'Évaluation

Le modèle ML peut être évalué avec diverses métriques :

```python
# Évaluer le modèle sur des données de test
metrics = ml_model.evaluate(test_data, "BTC/USD", "1h")

# Métriques disponibles
accuracy = metrics["accuracy"]
buy_precision = metrics["buy"]["precision"]
sell_recall = metrics["sell"]["recall"]
``` 