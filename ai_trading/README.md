# AI Trading - Système de Trading Crypto basé sur l'IA

Ce module contient l'implémentation d'un système de trading de cryptomonnaies utilisant l'intelligence artificielle, l'apprentissage par renforcement et l'analyse de sentiment.

## État du Projet

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

### Phase 2: Analyse de Sentiment (LLM) ✅
- ✅ 2.1 Implémentation de `llm/sentiment_analysis/news_analyzer.py`
  - Analyse des actualités crypto
  - Extraction des entités et sentiments
  - Version améliorée avec visualisations et rapports
- ✅ 2.2 Implémentation de `llm/sentiment_analysis/social_analyzer.py`
  - Analyse des tweets et posts Reddit
  - Détection des tendances et sentiments

### Phase 3: Développement de l'Agent d'Apprentissage par Renforcement 🔄
- ⏳ 3.1 Implémentation de `rl_agent.py`
  - Définition de l'environnement de trading
  - Implémentation de l'agent DQN (Deep Q-Network)
  - Intégration des données de marché et de sentiment
- ⏳ 3.2 Entraînement et optimisation
  - Définition des hyperparamètres
  - Entraînement sur données historiques
  - Validation et ajustement

### Phase 4: Prédictions de Marché (LLM) ⏳
- ⏳ 4.1 Implémentation de `llm/predictions/market_predictor.py`
  - Génération de prédictions basées sur les données de marché et le sentiment
- ⏳ 4.2 Implémentation de `llm/predictions/prediction_model.py`
  - Modèle de prédiction combinant données techniques et sentiment

### Phase 5: Signaux de Trading (ML) ⏳
- ⏳ 5.1 Implémentation de `ml/trading_signals/signal_generator.py`
  - Génération de signaux d'achat/vente
- ⏳ 5.2 Implémentation de `ml/trading_signals/ml_model.py`
  - Modèle ML pour la prédiction des signaux

### Phase 6: Intégration et API ⏳
- ⏳ 6.1 Développement de l'API
  - Endpoints pour les prédictions
  - Intégration des différents composants
  - Documentation de l'API
- ⏳ 6.2 Tests d'intégration
  - Tests de bout en bout
  - Validation des performances

### Phase 7: Interface Web et Déploiement ⏳
- ⏳ 7.1 Développement de l'interface web
  - Dashboard de trading
  - Visualisations interactives
  - Gestion des utilisateurs
- ⏳ 7.2 Déploiement
  - Configuration pour production
  - Mise en place de la CI/CD
  - Monitoring et maintenance

## Feuille de Route

### Prochaines étapes immédiates (Sprint actuel)
1. **Finaliser l'agent RL**
   - Implémenter la structure de base de l'agent DQN dans `rl_agent.py`
   - Définir l'environnement de trading avec récompenses appropriées
   - Intégrer les données de marché prétraitées

2. **Résoudre les problèmes de tests**
   - Corriger les avertissements de dépréciation dans le préprocesseur
   - Améliorer la gestion des limites d'API (CoinGecko)
   - Résoudre les problèmes de dépendances TensorFlow/Keras

### Sprint suivant
1. **Entraînement et optimisation de l'agent RL**
   - Configurer l'infrastructure d'entraînement
   - Optimiser les hyperparamètres
   - Évaluer les performances sur différentes périodes de marché

2. **Commencer le développement des prédictions LLM**
   - Définir l'architecture de `market_predictor.py`
   - Intégrer les données de sentiment avec les données de marché
   - Développer un prototype de modèle de prédiction

### Objectifs à moyen terme
1. **Développement des signaux de trading ML**
   - Implémenter le générateur de signaux
   - Entraîner le modèle ML pour la prédiction des signaux
   - Évaluer la précision des signaux générés

2. **Développement initial de l'API**
   - Définir les endpoints principaux
   - Implémenter la logique de prédiction
   - Documenter l'utilisation

### Objectifs à long terme
1. **Interface utilisateur complète**
   - Dashboard interactif
   - Alertes et notifications
   - Personnalisation des stratégies

2. **Déploiement en production**
   - Infrastructure cloud
   - Scaling automatique
   - Monitoring continu

3. **Fonctionnalités avancées**
   - Trading multi-actifs
   - Optimisation de portefeuille
   - Intégration avec des exchanges

## Problèmes connus et limitations
- Limites de taux des APIs (particulièrement CoinGecko)
- Dépendances entre TensorFlow/Keras et les modèles Hugging Face
- Avertissements de dépréciation dans certaines fonctions pandas

## Comment contribuer
1. Forker le projet
2. Créer une branche pour votre fonctionnalité
3. Soumettre une pull request avec une description détaillée

## Licence
Ce projet est sous licence MIT.

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
- ✅ 2.1 Implémentation de `llm/sentiment_analysis/news_analyzer.py`
  - Analyse des actualités crypto
  - Extraction des entités et sentiments
  - Version améliorée avec visualisations et rapports
- ✅ 2.2 Implémentation de `llm/sentiment_analysis/social_analyzer.py`
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

# Exécution des tests pour l'analyseur de sentiment amélioré
python -m ai_trading.tests.test_enhanced_news_analyzer
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

### Analyse de sentiment des actualités

L'exemple `sentiment_analysis_example.py` montre comment analyser le sentiment des actualités crypto :
- Collecte des actualités pour plusieurs cryptomonnaies
- Analyse du sentiment et extraction des entités
- Affichage des résultats

Pour exécuter cet exemple :

```bash
python -m ai_trading.examples.sentiment_analysis_example
```

### Analyse de sentiment avancée

L'exemple `enhanced_sentiment_analysis_example.py` montre comment utiliser les fonctionnalités avancées d'analyse de sentiment :
- Collecte des actualités pour plusieurs cryptomonnaies
- Analyse avancée du sentiment avec visualisations
- Génération de rapports détaillés

Pour exécuter cet exemple :

```bash
python -m ai_trading.examples.enhanced_sentiment_analysis_example
```

### Démonstration simple d'analyse de sentiment

L'exemple `sentiment_analysis_demo.py` est une démonstration simple de l'analyse de sentiment :
- Analyse d'exemples d'actualités prédéfinis
- Affichage des résultats de sentiment

Pour exécuter cette démonstration :

```bash
python -m ai_trading.examples.sentiment_analysis_demo
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

## Modules d'analyse de sentiment

### Analyseurs de sentiment
- **news_analyzer.py** : Version de base pour l'analyse des actualités crypto
- **enhanced_news_analyzer.py** : Version avancée avec plus de fonctionnalités (visualisations, rapports, cache)

### Fonctionnalités principales
- Analyse du sentiment des actualités (positif, négatif, neutre)
- Extraction des entités (cryptomonnaies, montants, pourcentages)
- Génération de rapports détaillés
- Visualisation des tendances de sentiment
- Gestion du cache pour optimiser les performances

### Quand utiliser quelle version ?
- **Version de base** : Pour les tests simples ou quand les dépendances sont limitées
- **Version améliorée** : Pour une analyse complète avec visualisations et rapports détaillés

## Dépendances optionnelles

Pour profiter de toutes les fonctionnalités du système, vous pouvez installer ces dépendances supplémentaires :

```bash
# Pour la compatibilité avec Transformers et Keras 3
pip install tf-keras

# Pour les visualisations (si Tkinter n'est pas installé)
# Sur Ubuntu/Debian
sudo apt-get install python3-tk

# Sur Windows avec Anaconda
conda install -c anaconda tk
```

Les modules fonctionneront même sans ces dépendances, mais avec des fonctionnalités réduites. 

## Analyse des réseaux sociaux

### SocialAnalyzer
- `social_analyzer.py` - Analyse des tweets et posts Reddit
- Fonctionnalités :
  - Validation des données d'entrée
  - Détection des hashtags/mentions (Twitter)
  - Calcul d'engagement viral
  - Visualisations temporelles
  - Identification des posts viraux
  - Intégration avec les modèles de sentiment

### Bonnes pratiques
- **Vérifier les données d'entrée** : Les posts doivent contenir les champs requis
- **Gérer le timezone** : Convertir les dates en UTC pour l'analyse temporelle
- **Surveiller les limites d'API** : Respecter les quotas des plateformes sociales

### Utilisation
```python
from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer

# Analyse de tweets
twitter_analyzer = SocialAnalyzer(platform='twitter')
analyzed_tweets = twitter_analyzer.analyze_social_posts(tweets_df)

# Génération de rapport
report = twitter_analyzer.generate_social_report(analyzed_tweets)
```

### Tests
```bash
python -m ai_trading.tests.test_social_analyzer
```