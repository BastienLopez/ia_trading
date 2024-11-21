# Discord Trading Bot - BTC/USDT Alerts

## Description

Ce bot Discord surveille la paire **BTC/USDT** et envoie des alertes dans un channel spécifique en fonction des indicateurs techniques :

- **EMA** (Exponential Moving Average)
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)

Les signaux incluent :

- **Achat (BUY)**
- **Vente (SELL)**
- **Croisements MACD (Bullish/Bearish)**

## Fonctionnalités

1. Alertes automatiques toutes les 10 minutes.
2. Commande manuelle `/force_signal` pour forcer une analyse.
3. Intégration avec Investing.com pour récupérer les données.

## Configuration

1. Créez un fichier `.env` :

   ```plaintext
   DISCORD_TOKEN=VOTRE_DISCORD_BOT_TOKEN
   DISCORD_CHANNEL_ID=ID_DU_CHANNEL
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Lancer le projet

1. Exécutez le script :

   ```bash
   python bot.py
   ```

2. Vérifiez le serveur local Flask pour l'état du bot :
   ```bash
   http://localhost:5000/health
   ```

## Commandes disponibles

- **`/force_signal`** : Force une analyse immédiate et envoie les résultats dans le channel.

---

**Note** : Ce bot utilise des données en temps réel via Investing.com. Assurez-vous d'avoir une connexion stable et d'ajuster les paramètres si nécessaire.
