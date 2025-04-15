# 🤖 AI Trading System 
**Système automatisé de trading crypto utilisant l'IA**  
*Apprentissage par renforcement • Analyse de sentiment en temps réel*

## Phases du Projet
Le projet est structuré en trois phases principales :

### 1️⃣ Collecte et Prétraitement des Données ✅
- Collecte multi-source (`utils/enhanced_data_collector.py`)
- Prétraitement avancé (`utils/enhanced_preprocessor.py`)
- Intégration de données de plusieurs APIs et sources

### 2️⃣ Analyse de Sentiment (LLM) ✅
- Analyse des actualités (`llm/sentiment_analysis/news_analyzer.py`)
- Analyse des réseaux sociaux (`llm/sentiment_analysis/social_analyzer.py`) 
- Modèles de sentiment avancés avec LLM

### 3️⃣ Apprentissage par Renforcement 🔄
- Environnement de trading (`rl/trading_environment.py`)
- Gestion des risques (`rl/risk_manager.py`)
- Fonctions de récompense sophistiquées (`rl/advanced_rewards.py`)

## Fonctionnalités Principales
- 🕸️ **Collecte de données multi-sources**  
  Prix, volumes, actualités, tweets et indicateurs techniques
- 🧠 **Analyse de sentiment avancée**  
  Modèles NLP customisés (BERT, GPT-3.5-turbo) pour l'analyse des marchés
- 🤖 **Agent de trading intelligent**  
  Algorithmes RL (Double Dueling DQN) avec gestion dynamique des risques
- 📈 **Indicateurs techniques avancés**  
  MACD, RSI, Bollinger Bands, EMA, ATR, Ichimoku Cloud et plus
- 🚨 **Gestion des risques sophistiquée**  
  Stop-loss adaptatifs, trailing stops, gestion de l'exposition
- 🔄 **Normalisation adaptative**  
  Traitement optimisé des données pour l'apprentissage

## Démarrage
```bash
# Installation
pip install -r requirements.txt

# Lancer avec Docker
docker-compose up --build
```

## Structure du Projet
```
ai_trading/
├── utils/                 # Collecte et prétraitement des données
├── llm/                   # Analyse de sentiment avec LLM
├── rl/                    # Apprentissage par renforcement
├── documentation/         # Documentation détaillée
├── examples/              # Exemples d'utilisation
├── tests/                 # Tests unitaires et d'intégration
├── visualizations/        # Visualisations générées
└── models/                # Modèles entraînés
```

*Configuration minimale : 4GB RAM, Python 3.11+*