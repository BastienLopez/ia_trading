# Bot de Trading Crypto avec Notifications Discord

Ce projet est un bot de trading automatisé qui analyse les indicateurs techniques du Bitcoin et envoie des notifications sur Discord.

## Fonctionnalités

- Analyse des indicateurs techniques (RSI, MACD, EMA)
- Notifications automatiques sur Discord
- Script Pine pour TradingView
- Vérification périodique des signaux
- Commandes Discord personnalisées

## Installation

1. Clonez le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_REPO]
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurez le fichier `.env` :
```
DISCORD_TOKEN=votre_token_discord
DISCORD_CHANNEL_ID=votre_channel_id
```

## Utilisation

1. **Bot Discord** :
```bash
python src/bot.py
```

2. **TradingView** :
- Ouvrez TradingView
- Créez un nouvel indicateur
- Copiez-collez le contenu de `src/tradingview/btc_signals.pine`

## Commandes Discord

- `/force_check` : Force une vérification des signaux
- `/price` : Affiche le prix actuel du BTC

## Indicateurs Utilisés

- **RSI** (Relative Strength Index)
  - Période : 14
  - Survente : < 30
  - Surachat : > 70

- **MACD** (Moving Average Convergence Divergence)
  - Fast EMA : 12
  - Slow EMA : 26
  - Signal : 9

- **EMA** (Exponential Moving Average)
  - Court terme : 9
  - Long terme : 21

## Structure du Projet

```
src/
├── config/
│   └── config.py
├── indicators/
│   └── technical_indicators.py
├── utils/
│   └── data_fetcher.py
├── tradingview/
│   └── btc_signals.pine
└── bot.py
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.
