# Crypto Trading AI

Un système complet d'intelligence artificielle pour le trading de cryptomonnaies, utilisant l'apprentissage par renforcement et des outils avancés d'analyse de données.

## 🚀 Vue d'ensemble

Ce projet combine plusieurs technologies pour créer un système de trading automatisé:

- **Intelligence Artificielle**: Apprentissage par renforcement pour prendre des décisions de trading
- **API RESTful**: Interface permettant d'accéder aux prédictions et de gérer les modèles
- **Bot Discord**: Notifications et commandes pour interagir avec le système
- **Interface Web**: Tableau de bord pour visualiser les performances et gérer les stratégies

## 📋 Structure du projet

```
.
├── ai_trading/               # Module d'IA pour le trading
│   ├── data/                 # Données historiques
│   ├── models/               # Modèles entraînés
│   ├── rl_agent.py           # Agent d'apprentissage par renforcement
│   ├── data_processor.py     # Traitement des données
│   ├── api.py                # API pour servir les prédictions
│   └── utils.py              # Utilitaires divers
│
├── discord_bot/              # Bot Discord pour les notifications
│   └── bot.py                # Implémentation du bot
│
├── web_app/                  # Interface web
│   ├── templates/            # Templates HTML
│   ├── static/               # Ressources statiques (CSS, JS)
│   └── app.py                # Application Flask
│
├── tests/                    # Tests unitaires et d'intégration
│   ├── ai_trading/           # Tests du module d'IA
│   └── discord_bot/          # Tests du bot Discord
│
├── docs/                     # Documentation complémentaire
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation principale
```

## ⚙️ Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-username/crypto-trading-ai.git
   cd crypto-trading-ai
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**
   Créez un fichier `.env` à la racine du projet avec les variables suivantes:
   ```
   API_URL=http://localhost:8000
   DISCORD_TOKEN=votre_token_discord
   ```

## 💻 Composants principaux

### Module d'IA Trading

Le cœur du système est un agent d'apprentissage par renforcement qui analyse les données de marché pour prendre des décisions de trading optimales.

### API RESTful

Une API FastAPI qui expose les fonctionnalités du système:
- Prédictions en temps réel
- Gestion des modèles
- Backtesting des stratégies

### Bot Discord

Un bot permettant d'interagir avec le système via Discord:
- Notifications des signaux de trading
- Consultation des performances
- Commandes pour lancer des analyses

### Interface Web

Une application web Flask qui offre:
- Tableau de bord interactif
- Visualisation des performances
- Configuration des stratégies

## 🔍 Documentation détaillée

Pour plus d'informations sur les différents composants, consultez:

- [Guide de l'API](docs/API.md)
- [Guide du Bot Discord](docs/DISCORD.md)
- [Guide de l'Interface Web](web_app/README.md)
- [Guide des Tests](docs/TESTING.md)

## 🤝 Contribution

Les contributions sont les bienvenues! Consultez notre [guide de contribution](CONTRIBUTING.md) pour plus d'informations.

## ⚠️ Avertissement

Ce système est fourni à des fins éducatives et de recherche uniquement. Le trading de cryptomonnaies comporte des risques financiers importants.

## 📄 Licence

Ce projet est sous licence MIT.
