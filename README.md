# Crypto Trading IA

Plateforme d'intelligence artificielle pour le trading de cryptomonnaies utilisant l'analyse de sentiment et l'apprentissage profond.

## Fonctionnalités

- Analyse de sentiment des actualités et réseaux sociaux
- Modèles de prédiction de prix
- Interface web de visualisation
- API REST pour l'intégration

## Installation

1. Cloner le dépôt
```bash
git clone https://github.com/votre-utilisateur/crypto-trading-ia.git
cd crypto-trading-ia
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
```

3. Configurer les variables d'environnement (.env)

4. Démarrer avec Docker
```bash
docker-compose up --build
```

## Architecture

```
.
├── ai_trading/          # Code principal
├── web_app/             # Application Flask
├── tests/               # Tests unitaires
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Contribution

1. Créer une branche
2. Ajouter les tests correspondants
3. Soumettre une Pull Request
