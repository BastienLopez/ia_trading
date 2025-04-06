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
├── utils/
│   ├── enhanced_data_collector.py # Collecte de données multi-sources
│   └── enhanced_preprocessor.py   # Prétraitement avancé des données
├── examples/
│   └── enhanced_data_pipeline.py  # Exemple de pipeline de données complet
└── tests/
    ├── test_enhanced_collector.py # Tests pour le collecteur de données
    └── test_enhanced_preprocessor.py # Tests pour le préprocesseur
```

## État d'avancement

### Phase 1: Collecte et Prétraitement des Données ✅
- ✅ 1.1 Implémentation de `utils/enhanced_data_collector.py`
  - Connexion à plusieurs APIs de cryptomonnaies (CoinGecko, CoinCap, CryptoCompare)
  - Collecte des données de prix, volumes et capitalisation
  - Récupération de l'indice Fear & Greed
  - Scraping des actualités et réseaux sociaux
- ✅ 1.2 Implémentation de `utils/enhanced_preprocessor.py`
  - Nettoyage des données multi-sources
  - Normalisation avancée
  - Création de features techniques étendues
  - Prétraitement des données textuelles

### Phase 2: Analyse de Sentiment (LLM) 🔄
- 🔄 2.1 Implémentation de `llm/sentiment_analysis/news_analyzer.py`
  - Analyse des actualités crypto
  - Extraction des entités et sentiments
- 🔄 2.2 Implémentation de `llm/sentiment_analysis/social_analyzer.py`
  - Analyse des tweets et posts Reddit
  - Détection des tendances et sentiments

### Phase 3: Prédictions de Marché (LLM) 📅
- 📅 3.1 Implémentation de `llm/predictions/market_predictor.py`
  - Génération de prédictions basées sur les données de marché et le sentiment
- 📅 3.2 Implémentation de `llm/predictions/prediction_model.py`
  - Modèle de prédiction combinant données techniques et sentiment

### Phase 4: Signaux de Trading (ML) 📅
- 📅 4.1 Implémentation de `ml/trading_signals/signal_generator.py`
  - Génération de signaux d'achat/vente
- 📅 4.2 Implémentation de `ml/trading_signals/ml_model.py`
  - Modèle ML pour la prédiction des signaux

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/ai-trading.git
cd ai-trading

# Installer les dépendances
pip install -r requirements.txt
```

## Comment exécuter les tests

```bash
# Exécution des tests pour le collecteur de données
python -m ai_trading.tests.test_enhanced_collector

# Exécution des tests pour le préprocesseur
python -m ai_trading.tests.test_enhanced_preprocessor
```

## Comment exécuter les exemples

Le projet inclut des exemples pratiques pour vous aider à comprendre comment utiliser les différents modules :

### Pipeline de données complet

L'exemple `enhanced_data_pipeline.py` montre comment enchaîner les différentes étapes du traitement des données :
- Collecte des données de marché et d'actualités pour plusieurs cryptomonnaies
- Prétraitement des données collectées
- Visualisation des résultats

Pour exécuter cet exemple :

```bash
python -m ai_trading.examples.enhanced_data_pipeline
```

Cet exemple va :
1. Collecter des données pour Bitcoin, Ethereum et Solana
2. Prétraiter ces données (nettoyage, features techniques, normalisation)
3. Générer des visualisations des prix
4. Sauvegarder tous les résultats dans les dossiers `data/raw` et `data/processed`

Vous pouvez modifier le script pour :
- Ajouter d'autres cryptomonnaies à analyser
- Changer la période de collecte des données
- Utiliser différentes méthodes de prétraitement

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

## Modules de collecte et prétraitement des données

### Collecteurs de données
- **minimal_data_collector.py** : Version simple utilisant principalement l'API CoinGecko
- **enhanced_data_collector.py** : Version avancée utilisant plusieurs sources (CoinGecko, CoinCap, CryptoCompare, etc.)

### Préprocesseurs
- **preprocessor.py** : Préprocesseur de base pour les données de marché et textuelles
- **enhanced_preprocessor.py** : Préprocesseur avancé avec plus de features techniques et une meilleure gestion des données multi-sources

### Quand utiliser quelle version ?
- **Version minimale** : Pour les tests, l'apprentissage, ou quand la simplicité est prioritaire
- **Version améliorée** : Pour la production, quand la robustesse et les fonctionnalités avancées sont nécessaires 