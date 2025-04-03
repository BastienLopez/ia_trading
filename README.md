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

# Guide d'utilisation de BTC Trading Signals

Ce document explique en détail tous les indicateurs, signaux et symboles présents dans le script TradingView Pine "BTC Trading Signals".

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Indicateurs techniques](#indicateurs-techniques)
3. [Signaux visuels](#signaux-visuels)
4. [Paramètres personnalisables](#paramètres-personnalisables)
5. [Guide d'interprétation](#guide-dinterprétation)

## Vue d'ensemble

"BTC Trading Signals" est un indicateur avancé pour TradingView qui combine plusieurs indicateurs techniques populaires pour générer des signaux d'achat et de vente sur le Bitcoin et d'autres crypto-monnaies. L'indicateur utilise une approche multi-facteurs, où la force d'un signal est déterminée par le nombre d'indicateurs qui confirment la même direction.

## Indicateurs techniques

### RSI (Relative Strength Index)
- **Fonction** : Mesure la vitesse et le changement des mouvements de prix
- **Paramètres par défaut** : Période 14, Survente 30, Surachat 70
- **Interprétation** : 
  - RSI < 30 : Condition de survente (signal haussier potentiel)
  - RSI > 70 : Condition de surachat (signal baissier potentiel)

### MACD (Moving Average Convergence Divergence)
- **Fonction** : Identifie les changements de tendance et de momentum
- **Paramètres par défaut** : Rapide 12, Lent 26, Signal 9
- **Interprétation** :
  - Croisement MACD au-dessus de la ligne de signal : Signal haussier
  - Croisement MACD en-dessous de la ligne de signal : Signal baissier
  - Points extrêmes (+ et -) : Points clés avant les croisements

### EMA (Exponential Moving Average)
- **Fonction** : Moyennes mobiles qui donnent plus de poids aux prix récents
- **Paramètres par défaut** : Court terme 9, Long terme 21
- **Interprétation** :
  - EMA Court > EMA Long : Tendance haussière
  - EMA Court < EMA Long : Tendance baissière

### Ichimoku Cloud
- **Composants** :
  - **Tenkan-Sen** (ligne de conversion) : Moyenne des plus hauts et plus bas sur 9 périodes
  - **Kijun-Sen** (ligne de base) : Moyenne des plus hauts et plus bas sur 26 périodes
  - **Senkou Span A** (première ligne du nuage) : Moyenne de Tenkan-Sen et Kijun-Sen
  - **Senkou Span B** (deuxième ligne du nuage) : Moyenne des plus hauts et plus bas sur 52 périodes
- **Interprétation** :
  - Prix au-dessus du nuage + Tenkan > Kijun : Tendance haussière forte
  - Prix en-dessous du nuage + Tenkan < Kijun : Tendance baissière forte

### Bollinger Bands
- **Composants** : Bande supérieure, bande médiane (SMA 20) et bande inférieure
- **Paramètres par défaut** : Période 20, Déviation standard 2.0
- **Interprétation** :
  - "Squeeze" (bandes resserrées) : Préparation à une expansion de volatilité

### Volume Profile
- **Fonction** : Analyse le volume récent pour détecter des déséquilibres acheteurs/vendeurs
- **Paramètres par défaut** : Lookback 20 périodes
- **Interprétation** :
  - Ratio > 0.7 + Volume élevé : Forte pression acheteuse
  - Ratio < 0.3 + Volume élevé : Forte pression vendeuse

## Signaux visuels

### Symboles des extrema MACD
- **"+"** (vert au-dessus des bougies) : Point le plus haut du MACD juste avant un croisement baissier
- **"-"** (rouge en-dessous des bougies) : Point le plus bas du MACD juste avant un croisement haussier
- **Importance** : Ces symboles marquent les points d'inflexion potentiels avant un changement de tendance

### Flèches MACD
- **Flèche vers le haut** (verte, grande) : Croisement haussier du MACD
- **Flèche vers le bas** (rouge, grande) : Croisement baissier du MACD
- **Importance** : Signaux principaux pour les entrées/sorties potentielles

### Triangles de force du signal
- **Triangles verts** (en-dessous des bougies) : Signaux d'achat
  - **Petit** : Signal d'achat faible (1 indicateur)
  - **Moyen** : Signal d'achat modéré (2 indicateurs)
  - **Grand** : Signal d'achat fort (3+ indicateurs)
- **Triangles rouges** (au-dessus des bougies) : Signaux de vente
  - **Petit** : Signal de vente faible (1 indicateur)
  - **Moyen** : Signal de vente modéré (2 indicateurs)
  - **Grand** : Signal de vente fort (3+ indicateurs)

### Signaux mineurs (optionnels)
- **Triangles** : Signaux RSI (survente/surachat)
- **Diamants** : Croisements MACD mineurs
- **Cercles** : Croisements EMA

## Paramètres personnalisables

### Principaux paramètres
- **Périodes des indicateurs** : RSI, MACD, EMA, Ichimoku, Bollinger Bands
- **Seuils** : RSI survente/surachat
- **Affichage** : Activer/désactiver l'Ichimoku Cloud, Bollinger Bands, signaux mineurs
- **MACD Extremes** : Activer/désactiver les symboles "+" et "-"

### Conseils pour l'optimisation
- **RSI** : Ajuster les seuils en fonction de la volatilité du marché
- **MACD** : Périodes plus courtes pour des signaux plus fréquents
- **Lookback pour extremes MACD** : Valeur plus élevée pour des extrema plus significatifs

## Guide d'interprétation

### Signaux forts (à privilégier)
1. **Signal d'achat fort** (triangle vert large) + "−" récent + croisement MACD haussier (flèche verte)
   - Action suggérée : Considérer une entrée longue
   - Confirmation supplémentaire : Prix au-dessus du nuage Ichimoku avec volume élevé

2. **Signal de vente fort** (triangle rouge large) + "+" récent + croisement MACD baissier (flèche rouge)
   - Action suggérée : Considérer une sortie ou position courte
   - Confirmation supplémentaire : Prix sous le nuage Ichimoku avec volume élevé

### Divergences importantes
Les divergences entre le prix et MACD peuvent être particulièrement puissantes :
- **Divergence haussière** : Prix fait des plus bas plus bas, mais MACD fait des plus bas plus hauts
- **Divergence baissière** : Prix fait des plus hauts plus hauts, mais MACD fait des plus hauts plus bas

### Stratégie d'utilisation recommandée
1. Utiliser les signaux forts comme alertes principales
2. Confirmer avec le contexte de marché plus large (tendance, support/résistance)
3. Porter attention aux symboles "+" et "-" pour anticiper les inversions potentielles
4. Utiliser les alertes configurées pour être notifié des opportunités

---

*Note : Cet indicateur est un outil d'aide à la décision et ne garantit pas les résultats futurs. Toujours combiner avec une analyse fondamentale et une gestion rigoureuse des risques.*
