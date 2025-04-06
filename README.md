# Système de Trading Automatisé avec Apprentissage par Renforcement

Un système de trading automatisé utilisant l'apprentissage par renforcement pour trader les cryptomonnaies.

## Fonctionnalités

- **API de Trading** : Interface REST pour interagir avec le système
- **Analyse Technique** : Utilisation d'indicateurs techniques pour l'analyse
- **Apprentissage par Renforcement** : Modèles d'IA pour la prise de décision
- **Backtesting** : Simulation de stratégies sur des données historiques
- **Visualisation** : Graphiques et tableaux de bord pour le suivi

## Structure du Projet

```
.
├── ai_trading/           # Module principal de trading
├── tradingview/          # Intégration avec TradingView
├── tests/               # Tests unitaires et d'intégration
├── docs/                # Documentation
├── logs/                # Fichiers de logs
└── instance/            # Fichiers de configuration
```

## Configuration

1. Créez un fichier `.env` à la racine du projet avec les variables suivantes :

```env
API_KEY=votre_clé_api
API_SECRET=votre_secret_api
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
LOG_LEVEL=INFO
```

2. Installez les dépendances globalement :

```bash
pip install -r requirements.txt
```

## Utilisation

### API de Trading

L'API permet d'interagir avec le système via des endpoints REST :

```bash
# Démarrer l'API
python -m ai_trading.api
```

### Backtesting

Pour tester une stratégie sur des données historiques :

```bash
python -m ai_trading.train --backtest --symbol BTC/USDT --timeframe 1h --days 60
```

## Tests

### Tests Frontend
Les tests du frontend sont organisés dans le dossier `tests/web_app/` et couvrent :

- Routes et API endpoints (`test_routes.py`)
  - Test des routes principales (/, /dashboard)
  - Test des opérations CRUD sur les transactions

- Service de prix (`test_price_service.py`)
  - Test de récupération des prix en temps réel
  - Test de gestion des erreurs API
  - Test de récupération par symbole

- Service de transactions (`test_transaction_service.py`)
  - Test d'ajout de transactions (prix auto/manuel)
  - Test de récupération des transactions
  - Test de suppression de transactions

Pour lancer tous les tests :
```bash
# Tests frontend uniquement
pytest tests/web_app/ -v

# Tests backend uniquement
pytest tests/ai_trading/ -v

# Tous les tests avec couverture
pytest --cov=web_app --cov=ai_trading --cov-report=term-missing
```

## Documentation

- [Guide d'Installation](docs/INSTALLATION.md)
- [Guide de Développement](docs/DEVELOPMENT.md)
- [Guide de Test](docs/TESTING.md)
