# Organisation des Exemples du Projet AI Trading

Ce document décrit l'organisation des exemples du projet AI Trading, classés par domaine fonctionnel pour faciliter la navigation et la compréhension du système.

## Structure des Dossiers d'Exemples

```
ai_trading/examples/
├── llm/                  # Exemples liés aux modèles de langage (LLM)
│   ├── sentiment/        # Analyse de sentiment des actualités et réseaux sociaux
│   └── predictions/      # Prédictions de marché basées sur LLM
│
├── ml/                   # Exemples liés aux modèles de machine learning
│   ├── trading_signals/  # Génération et filtrage de signaux de trading
│   └── backtesting/      # Backtesting avancé et analyse de performance
│
├── rl/                   # Exemples liés à l'apprentissage par renforcement
│   ├── agents/           # Différents agents d'apprentissage par renforcement
│   └── environments/     # Environnements de trading pour RL
│
├── data/                 # Exemples de collecte et prétraitement des données
│   ├── raw/              # Données brutes
│   ├── processed/        # Données traitées
│   └── sentiment/        # Données d'analyse de sentiment
│
├── risk/                 # Exemples de gestion des risques
│   ├── position_sizing/  # Dimensionnement des positions
│   └── stop_loss/        # Stratégies de stop-loss et take-profit
│
├── optimization/         # Exemples d'optimisation
│   ├── hyperparameters/  # Optimisation d'hyperparamètres
│   └── portfolio/        # Optimisation de portefeuille
│
├── execution/            # Exemples d'exécution d'ordres
│   ├── market_orders/    # Exécution d'ordres au marché
│   └── limit_orders/     # Exécution d'ordres limites et stop
│
├── visualization/        # Exemples de visualisation
│   ├── dashboards/       # Tableaux de bord interactifs
│   └── performance/      # Visualisation des performances de trading
│
├── results/              # Résultats des exemples et tests
└── output/               # Fichiers de sortie générés par les exemples
```

## Classification des Exemples Organisés

### Exemples LLM
- **Analyse de sentiment** :
   - `ai_trading/examples/llm/sentiment/sentiment_analyzer_example.py` - Exemple d'analyse de sentiment combinée (actualités et médias sociaux)
   - `ai_trading/examples/llm/sentiment/enhanced_sentiment_analysis_example.py` - Analyse de sentiment améliorée avec visualisations
   - `ai_trading/examples/llm/sentiment/llm_explainer_example.py` - Explication des analyses de sentiment avec les LLM
- **Prédictions de marché** :
   - `ai_trading/examples/llm/predictions/llm_prediction_explainer_example.py` - Explicabilité des prédictions LLM
   - `ai_trading/examples/llm/predictions/llm_multi_horizon_example.py` - Prédictions multi-horizons avec les LLM

### Exemples ML
- **Signaux de trading** :
   - `ai_trading/examples/ml/trading_signals/trading_signal_generator_example.py` - Génération et filtrage de signaux de trading
   - `ai_trading/examples/ml/trading_signals/ml_signal_example.py` - Signaux basés sur des modèles ML
   - `ai_trading/examples/ml/trading_signals/multi_timeframe_signals_example.py` - Analyse multi-timeframes
- **Backtesting** :
   - `ai_trading/examples/ml/backtesting/survivorship_bias_example.py` - Correction du biais de survivance
   - `ai_trading/examples/ml/backtesting/advanced_backtesting_example.py` - Backtesting avancé avec exécution d'ordres réaliste

### Exemples RL
- **Agents RL** :
   - `ai_trading/examples/rl/agents/advanced_rl_agent_example.py` - Agent RL avancé avec architecture Dueling DQN et GRU
   - `ai_trading/examples/rl/agents/sac_training_example.py` - Entraînement SAC pour actions continues
   - `ai_trading/examples/rl/agents/transformer_sac_example.py` - Utilisation de transformers pour l'apprentissage RL
   - `ai_trading/examples/rl/agents/inverse_rl_example.py` - Apprentissage par renforcement inverse
- **Environnements** :
   - `ai_trading/examples/rl/environments/multi_asset_environment_example.py` - Environnement multi-actifs avec corrélation
   - `ai_trading/examples/rl/environments/test_trading_env.py` - Test d'environnement de trading

### Exemples d'optimisation
- **Hyperparamètres** :
   - `ai_trading/examples/optimization/hyperparameters/hyperparameter_optimization_example.py` - Optimisation des hyperparamètres avec grid search et bayesian
- **Portefeuille** :
   - `ai_trading/examples/optimization/portfolio/portfolio_optimization_example.py` - Optimisation de portefeuille avec Markowitz, Risk Parity et allocation hiérarchique

### Exemples d'exécution
- **Ordres limites et stop** :
   - `ai_trading/examples/execution/limit_orders/pro_order_example.py` - Système d'ordres professionnels (limite, stop-loss, trailing stop)

### Exemples de visualisation
- **Tableaux de bord** :
   - `ai_trading/examples/visualization/dashboards/trading_dashboard_example.py` - Tableau de bord interactif avec Dash pour le trading

## Avantages de cette Organisation

1. **Modulaire** : Facilite l'exploration des différentes fonctionnalités du système
2. **Pédagogique** : Permet de comprendre chaque composant individuellement
3. **Évolutive** : Structure extensible pour ajouter de nouveaux exemples
4. **Réutilisable** : Les exemples peuvent être adaptés et combinés pour des cas d'utilisation spécifiques
5. **Testable** : Les exemples servent également de démonstration du bon fonctionnement des modules

## Prochaines Étapes

1. Continuer à déplacer les exemples existants dans leurs catégories respectives
2. Créer des exemples supplémentaires pour les fonctionnalités encore non couvertes:
   - Exemples pour data/raw et data/processed
   - Exemples pour les différentes stratégies de gestion des risques
   - Exemples pour visualization/performance
3. Mettre à jour les références entre les exemples pour assurer une cohérence
4. Ajouter des exemples d'utilisation bout-en-bout intégrant plusieurs composants

Cette organisation reflète la structure modulaire du système AI Trading et facilite l'apprentissage et l'utilisation des différentes fonctionnalités. 