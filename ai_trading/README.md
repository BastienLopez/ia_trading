# AI Trading System

## Architecture

Notre système d'IA pour le trading se compose de deux parties principales :

### 1. Modèles LLM
- **Analyse de sentiment**: Traitement des actualités et des réseaux sociaux
- **Prédictions**: Génération de prédictions basées sur l'analyse des données

### 2. Algorithmes ML
- **Signaux trading**: Génération de signaux d'achat/vente

## Structure du projet

```
ai_trading/
├── llm/
│   ├── sentiment_analysis/
│   │   ├── news_analyzer.py      # Analyse des actualités
│   │   ├── social_analyzer.py    # Analyse des réseaux sociaux
│   │   └── sentiment_model.py    # Modèle de sentiment
│   └── predictions/
│       ├── market_predictor.py   # Prédictions de marché
│       └── prediction_model.py   # Modèle de prédiction
├── ml/
│   └── trading_signals/
│       ├── signal_generator.py   # Générateur de signaux
│       ├── technical_analysis.py # Indicateurs techniques
│       └── ml_model.py          # Modèle ML pour les signaux
└── utils/
    ├── data_collector.py        # Collecte de données
    ├── preprocessor.py          # Prétraitement des données
    └── model_evaluator.py       # Évaluation des modèles
```

## État d'avancement

### Phase 1: Collecte et Prétraitement des Données ✅
- ✅ 1.1 Implémentation de `utils/data_collector.py` et `utils/minimal_data_collector.py`
  - Connexion aux APIs de cryptomonnaies
  - Collecte des données de prix et volumes
  - Scraping des actualités et réseaux sociaux
- ✅ 1.2 Implémentation de `utils/preprocessor.py`
  - Nettoyage des données
  - Normalisation
  - Création des features

### Phase 2: Analyse de Sentiment (LLM) 🔄
1. Implémenter `llm/sentiment_analysis/sentiment_model.py`
   - Intégration d'un modèle LLM (ex: BERT, GPT)
   - Fine-tuning pour l'analyse de sentiment crypto

2. Implémenter `llm/sentiment_analysis/news_analyzer.py`
   - Analyse des actualités crypto
   - Extraction des informations pertinentes

3. Implémenter `llm/sentiment_analysis/social_analyzer.py`
   - Analyse des réseaux sociaux
   - Détection des tendances et sentiments

### Phase 3: Prédictions (LLM)
1. Implémenter `llm/predictions/prediction_model.py`
   - Modèle de prédiction basé sur LLM
   - Intégration des données historiques

2. Implémenter `llm/predictions/market_predictor.py`
   - Génération de prédictions de marché
   - Combinaison des signaux de sentiment

### Phase 4: Signaux de Trading (ML)
1. Implémenter `ml/trading_signals/technical_analysis.py`
   - Calcul des indicateurs techniques
   - Génération des features pour ML

2. Implémenter `ml/trading_signals/ml_model.py`
   - Modèle ML pour la génération de signaux
   - Optimisation des hyperparamètres

3. Implémenter `ml/trading_signals/signal_generator.py`
   - Combinaison des signaux LLM et ML
   - Génération des signaux finaux

### Phase 5: Évaluation et Optimisation
1. Implémenter `utils/model_evaluator.py`
   - Métriques d'évaluation
   - Backtesting des stratégies
   - Optimisation des modèles

## Comment exécuter les tests

Pour vérifier que les modules de la Phase 1 fonctionnent correctement, exécutez les tests unitaires :

```bash
# Installation des dépendances
pip install pandas numpy scikit-learn nltk pycoingecko tweepy requests python-dotenv

# Exécution des tests pour le collecteur de données
python -m tests.test_data_collector

# Exécution des tests pour le préprocesseur
python -m tests.test_preprocessor
```

## Technologies utilisées

### LLM
- LangChain pour l'orchestration des LLM
- Transformers (Hugging Face) pour les modèles pré-entraînés
- BERT/GPT pour l'analyse de sentiment

### ML
- PyTorch pour les modèles personnalisés
- Scikit-learn pour les algorithmes classiques
- TensorFlow pour les réseaux de neurones

### Utils
- Pandas pour la manipulation des données
- NumPy pour les calculs numériques
- TA-Lib pour l'analyse technique

## Prochaines étapes

1. Commencer l'implémentation de la Phase 2 (Analyse de sentiment)
2. Collecter des données d'entraînement pour les modèles de sentiment
3. Intégrer les modèles LLM pour l'analyse des actualités et réseaux sociaux

## Notes importantes

- Toujours commencer par des tests sur des données historiques
- Implémenter une gestion robuste des erreurs
- Documenter chaque composant
- Maintenir des tests unitaires
- Suivre les bonnes pratiques de versionnage (git) 