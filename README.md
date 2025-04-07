# AI Trading Platform

Une plateforme d'analyse et de trading de cryptomonnaies utilisant l'intelligence artificielle, l'analyse de sentiment et l'apprentissage par renforcement.

## Structure du projet

```
ai_trading/
├── llm/                      # Modules d'analyse par Large Language Models
│   └── sentiment_analysis/   # Analyse de sentiment des news et réseaux sociaux
├── tests/                    # Tests unitaires du package AI Trading
├── utils/                    # Utilitaires de collecte et prétraitement
│   ├── enhanced_data_collector.py  # Collecte multi-sources
│   └── enhanced_preprocessor.py    # Prétraitement avancé
├── api.py                    # API FastAPI
├── config.py                 # Configuration globale
├── data_processor.py         # Traitement des données
├── rl_agent.py               # Agent d'apprentissage par renforcement
└── train.py                  # Script d'entraînement

tests/
└── web_app/                  # Tests de l'interface web
    ├── test_routes.py        # Tests des routes API
    ├── test_price_service.py # Tests du service de prix
    └── test_transaction_service.py # Tests du service de transactions

web_app/                      # Interface utilisateur web
```

## Installation

### Avec Docker (recommandé)

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/crypto-ai-trading.git
cd crypto-ai-trading

# Construire et démarrer les conteneurs
docker-compose build
docker-compose up -d
```

### Installation manuelle

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/crypto-ai-trading.git
cd crypto-ai-trading

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données NLTK
python -m nltk.downloader punkt wordnet
```

## Utilisation

### Collecte de données

```python
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector

collector = EnhancedDataCollector()
btc_data = collector.get_crypto_prices_coingecko(coin_id='bitcoin', days=30)
news = collector.get_crypto_news(limit=10)
```

### Analyse de sentiment

```python
from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer

analyzer = EnhancedNewsAnalyzer()
sentiment = analyzer.analyze_news(news_list)
```

### Exécution des tests

```bash
# Exécuter tous les tests
pytest

# Exécuter des tests spécifiques
pytest ai_trading/tests/test_enhanced_collector.py -v
pytest tests/web_app/test_routes.py -v
```

## Développement

### Structure des tests

- `ai_trading/tests/` - Tests unitaires des modules AI Trading
- `tests/web_app/` - Tests de l'interface web

### CI/CD

Le projet utilise GitHub Actions pour l'intégration continue. Voir `.github/workflows/tests.yml`.
