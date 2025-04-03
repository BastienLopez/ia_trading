# Interface Web - Crypto Trading AI

Interface web pour visualiser et gérer le système de trading par IA.

## 📋 Introduction

Cette interface web basée sur Flask permet de:
- Consulter le tableau de bord des performances
- Visualiser l'évolution du marché et les indicateurs techniques
- Exécuter et analyser des backtests
- Surveiller l'état du portefeuille

## ⚙️ Installation

1. **Assurez-vous d'avoir installé les dépendances**
   ```bash
   pip install flask jinja2 werkzeug itsdangerous flask-wtf flask-login
   ```
   
   Ces dépendances sont également incluses dans le fichier `requirements.txt` principal.

2. **Configuration**
   
   Pour communiquer avec l'API de trading, l'application utilise une variable d'environnement `API_URL`.
   Par défaut, elle pointe vers `http://localhost:8000`.

## 🚀 Lancement du serveur

Pour démarrer l'application web:

```bash
python -m web_app.app
```

L'interface sera accessible à l'adresse: http://localhost:5000

## 📊 Pages disponibles

### Page d'accueil
- Vue d'ensemble du projet
- Accès rapide aux principales fonctionnalités
- Résumé des performances récentes

### Tableau de bord
- Graphique d'évolution des prix
- Indicateurs techniques actuels
- Valeur du portefeuille
- Historique des transactions

### Backtests
- Formulaire pour configurer un nouveau backtest
- Visualisation des résultats de backtests
- Comparaison avec la stratégie "buy and hold"
- Métriques de performance (ratio de Sharpe, drawdown, etc.)

## 🔧 Structure des fichiers

```
web_app/
├── app.py                  # Application principale Flask
├── templates/              # Templates HTML
│   ├── base.html           # Template de base avec navigation
│   ├── index.html          # Page d'accueil
│   ├── dashboard.html      # Tableau de bord
│   └── backtest.html       # Page de backtest
├── static/                 # Ressources statiques
│   ├── css/                # Styles CSS
│   │   └── style.css       # Styles personnalisés
│   └── js/                 # Scripts JavaScript
│       └── main.js         # Fonctions JavaScript communes
└── README.md               # Documentation
```

## 🧪 Tests

Pour tester l'interface web:

```bash
# Lancer les tests de l'interface web
python -m pytest web_app/tests/

# Vérifier que le serveur fonctionne correctement
curl http://localhost:5000/
```

## 🔄 Flux de données

L'interface web fonctionne comme suit:
1. Récupère les données via l'API RESTful
2. Affiche les informations dans une interface conviviale
3. Envoie les demandes d'actions (backtests, prédictions) à l'API
4. Affiche les résultats retournés

## 📱 Responsive Design

L'interface est conçue pour fonctionner sur:
- Ordinateurs de bureau
- Tablettes
- Smartphones

## ⚠️ Dépannage

**Erreur 500 lors de l'accès aux pages**
- Vérifiez que l'API est en cours d'exécution
- Vérifiez les logs d'erreur dans la console
- Assurez-vous que toutes les dépendances sont installées

**Problèmes d'affichage**
- Videz le cache de votre navigateur
- Vérifiez que tous les fichiers statiques sont correctement chargés 