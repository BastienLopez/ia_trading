# AI Trading - Système de Trading Crypto basé sur l'IA

Ce projet implémente un système de trading de cryptomonnaies utilisant l'intelligence artificielle, l'apprentissage par renforcement et l'analyse de sentiment.

## Fonctionnalités principales

- Collecte et prétraitement de données multi-sources 
- Analyse de sentiment basée sur LLM pour les actualités et réseaux sociaux
- Agents de trading par apprentissage par renforcement (DQN, SAC, DDPG)
- Dashboard interactif pour visualiser performances et analyses
- Indicateurs techniques avancés et analyse post-trade
- Optimisations de performance (mémoire, GPU, calculs parallèles)

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/ai-trading.git
cd ai-trading

# Installer les dépendances
pip install -r requirements.txt
```

## Usage

### Dashboard

```bash
python -m ai_trading.dashboard.app
```

### Entraînement d'un modèle

```bash
python -m ai_trading.train --download --symbol BTC/USDT --timeframe 1h --days 60
```

### Tests

```bash
python -m pytest ai_trading/tests/ -v -rs
```

## Docker

Ce projet peut être exécuté avec Docker. Tous les fichiers Docker sont regroupés dans le dossier `docker/` pour une meilleure organisation.

Pour plus d'informations sur l'utilisation de Docker avec ce projet, consultez la documentation Docker complète dans [docker/README.md](docker/README.md).

Principales commandes :
```bash
# Démarrer tous les services
cd docker && docker compose up

# Démarrer uniquement le dashboard
cd docker && docker compose up dashboard

# Exécuter les tests
make docker-test
```

## Structure du projet

- `ai_trading/`
  - `dashboard/` - Interface utilisateur Dash/Plotly
  - `data/` - Collecte et preprocessing des données
  - `llm/` - Analyse de sentiment et prédictions
  - `ml/` - Modèles classiques et feature engineering
  - `rl/` - Agents et environnements d'apprentissage par renforcement
  - `tests/` - Tests unitaires et d'intégration
  - `utils/` - Utilitaires divers
- `docker/` - Configuration Docker
- `documentation/` - Documentation du projet
- `models/` - Modèles entraînés (gitignore)
- `data/` - Données brutes et prétraitées (gitignore)

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.