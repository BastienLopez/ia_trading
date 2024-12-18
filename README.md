# Discord Trading Bot - BTC/USDT Alerts

## Description

Ce bot Discord surveille la paire **BTC/USDT** et envoie des alertes dans un channel spécifique en fonction des indicateurs techniques :

- **EMA** (Exponential Moving Average)
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Trendlines** avec breakouts.

Les signaux incluent :

- **Achat (BUY)**
- **Vente (SELL)**
- **Croisements MACD (Bullish/Bearish)**
- **Ruptures de résistance/support (Trendlines)**

---

## Configuration du bot

### 1. Créez un fichier `.env`
Ajoutez les informations suivantes dans un fichier `.env` :
```plaintext
DISCORD_TOKEN=VOTRE_DISCORD_BOT_TOKEN
DISCORD_CHANNEL_ID=ID_DU_CHANNEL
```

- **DISCORD_TOKEN** : Token de votre bot Discord.
  - Disponible dans le [Discord Developer Portal](https://discord.com/developers/applications) > Sélectionnez votre application > Bot > Reset Token > Copiez le token.
- **DISCORD_CHANNEL_ID** : ID du channel où le bot doit envoyer des messages.
  - Activez le **Mode développeur** dans Discord :
    - Paramètres utilisateur > Avancés > Mode développeur.
    - Faites un clic droit sur le channel > Copier l'identifiant.

---

### 2. Installez les dépendances
Exécutez la commande suivante pour installer toutes les dépendances nécessaires :
```bash
pip install -r requirements.txt
```

---

### 3. Ajouter le bot à votre serveur Discord
1. Accédez au [Discord Developer Portal](https://discord.com/developers/applications).
2. Sélectionnez votre application.
3. Allez dans l'onglet **OAuth2** > **URL Generator**.
4. Sous **Scopes**, cochez **bot**.
5. Sous **Bot Permissions**, cochez les permissions suivantes :
   - **Send Messages**
   - **Read Messages/View Channels**
   - **Use Slash Commands**
6. Copiez le lien généré et ouvrez-le dans un navigateur.
7. Sélectionnez votre serveur et cliquez sur **Autoriser**.

---

### 4. Lancer le bot
Exécutez le bot avec la commande :
```bash
python bot.py
```

Vous verrez des logs indiquant que le bot est connecté :
```
Bot connecté en tant que <NomDuBot>
```

---

### 5. Vérifiez le serveur Flask
Ouvrez un navigateur et accédez à :
```
http://localhost:5000/health
```
Si tout fonctionne, vous verrez le message suivant :
```json
{"status": "Bot is running"}
```

---

## Fonctionnalités

1. **`/force_signal`** : 
   - Force une analyse immédiate des données actuelles (BTC/USDT).
   - Affiche les signaux détectés (BUY, SELL, MACD, etc.) dans le channel Discord configuré.

2. **`/log`** : 
   - Permet de vérifier que le bot est bien connecté et fonctionne correctement.
   - Répond avec un message de confirmation.

---

**Utilisation** :
- Tapez `/force_signal` ou `/log` dans le channel où le bot est configuré.
- Assurez-vous que le bot a les permissions pour lire et écrire dans ce channel.

---

## Problèmes courants
1. **Le bot ne répond pas** :
   - Assurez-vous que le `DISCORD_TOKEN` et `DISCORD_CHANNEL_ID` dans `.env` sont corrects.
   - Vérifiez que le bot a les permissions nécessaires dans le channel.

2. **Erreur : Privileged Intent Missing** :
   - Activez **Message Content Intent** dans le portail développeur Discord :
     - [Discord Developer Portal](https://discord.com/developers/applications) > Bot > Privileged Gateway Intents > Activez **Message Content Intent**.

---

## Notes
- Ce bot utilise des données en temps réel via Investing.com. Assurez-vous d'avoir une connexion stable.
- Ne partagez pas votre `DISCORD_TOKEN` publiquement pour éviter tout usage malveillant.
```
