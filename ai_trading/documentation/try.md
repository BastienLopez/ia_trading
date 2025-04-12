# Procédure complète de test et visualisation

## 1. Installation des dépendances
```bash
pip install -r requirements.txt
pip install pytest-html matplotlib plotly
```

## 2. Tests automatisés avec couverture
```bash
# Nettoyer les anciens rapports
rm -rf reports/ && mkdir -p reports/{coverage,plots,test_results}

# Lancer tous les tests avec couverture
coverage run -m pytest ai_trading/tests/ -v --html=reports/test_results/report.html --self-contained-html
coverage html -d reports/coverage
```

## 3. Exécution des exemples avec visualisation

### 3.1 Analyse de sentiment (génère 4 graphiques)
```bash
python ai_trading/examples/enhanced_sentiment_analysis_example.py \
    --coins BTC ETH SOL \
    --days 30 \
    --plot \
    --output reports/plots/sentiment/
```

### 3.2 Entraînement RL avec monitoring (génère 3 graphiques)
```bash
python ai_trading/examples/rl_training_example.py \
    --train \
    --backtest \
    --epochs 100 \
    --render \
    --plot-dir reports/plots/rl_training/
```

### 3.3 Test de l'environnement de trading (génère 2 graphiques)
```bash
python ai_trading/examples/test_trading_env.py \
    --episodes 20 \
    --visualize \
    --output reports/plots/trading_env/
```

### 3.4 Backtesting stratégique (génère 5 graphiques)
```bash
python ai_trading/examples/strategic_backtest.py \
    --symbol BTC \
    --days 90 \
    --plot \
    --output reports/plots/backtesting/
```

## 4. Génération des rapports de performance
```bash
python ai_trading/examples/generate_performance_plots.py \
    --sentiment-dir reports/plots/sentiment/ \
    --rl-dir reports/plots/rl_training/ \
    --trading-dir reports/plots/trading_env/ \
    --output reports/plots/combined/
```

## 5. Tests avancés

### 5.1 Test de charge (500 requêtes)
```bash
python ai_trading/tests/load_test.py \
    --requests 500 \
    --output reports/plots/load_test/
```

### 5.2 Test de résistance aux pannes
```bash
python ai_trading/tests/failure_resistance_test.py \
    --plot \
    --output reports/plots/failure_test/
```

### 5.3 Hyperparameter tuning visuel
```bash
python ai_trading/examples/hyperparameter_tuning.py \
    --trials 50 \
    --plot \
    --output reports/plots/hyperparameters/
```

## Vérification des sorties

| Fichier | Description |
|---------|-------------|
| `reports/plots/sentiment/sentiment_distribution.png` | Distribution des sentiments |
| `reports/plots/sentiment/top_news_impact.png` | Impact des principales actualités |
| `reports/plots/rl_training/training_progress.png` | Progrès de l'entraînement RL |
| `reports/plots/rl_training/portfolio_performance.png` | Performance du portefeuille |
| `reports/plots/trading_env/episode_rewards.png` | Récompenses par épisode |
| `reports/plots/combined/risk_reward_analysis.png` | Analyse risque/rendement |
| `reports/test_results/report.html` | Rapport HTML des tests |
| `reports/coverage/index.html` | Couverture des tests |

## Nettoyage
```bash
# Supprimer tous les fichiers générés
rm -rf reports/ plots/ data/sentiment/
