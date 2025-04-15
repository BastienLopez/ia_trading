# Visualisations dans le projet AI Trading

Ce document décrit l'organisation des visualisations dans le projet AI Trading, structuré selon les trois phases principales du projet.

## Structure des visualisations

Toutes les visualisations générées par les scripts du projet sont enregistrées dans le dossier `ai_trading/visualizations/`. Ce dossier est organisé en sous-dossiers thématiques pour faciliter la navigation :

```
ai_trading/visualizations/
├── misc/            # Visualisations diverses
├── trading_env/     # Visualisations de l'environnement de trading
└── evaluation/      # Visualisations d'évaluation des performances
```

## Génération des visualisations par phase du projet

### Phase 1: Collecte et Prétraitement des Données
Les visualisations de cette phase concernent les données de marché et les indicateurs techniques :
- Graphiques de prix et volumes pour différentes cryptomonnaies
- Visualisations des indicateurs techniques (MACD, RSI, Bollinger Bands, etc.)
- Corrélations entre actifs et métriques
- Génération : `examples/enhanced_data_pipeline.py`, `visualizations/market_data/`

### Phase 2: Analyse de Sentiment (LLM)
Les visualisations liées à l'analyse de sentiment incluent :
- Nuages de mots et fréquences de termes clés 
- Évolution temporelle des scores de sentiment
- Corrélations entre sentiment et mouvements de prix
- Génération : `examples/enhanced_sentiment_analysis_example.py`, `visualizations/sentiment/`

### Phase 3: Apprentissage par Renforcement
Les visualisations pour les agents d'apprentissage par renforcement comprennent :
- Courbes d'apprentissage (récompenses par épisode)
- Distribution des actions (achat, vente, conservation)
- Évolution de la valeur du portefeuille
- Comparaison des différentes fonctions de récompense
- Génération : tests dans `rl/`, `visualizations/rl/`

## Types de visualisations spécifiques

### Fonctions de récompense 
Pour la comparaison des fonctions de récompense :
- `reward_comparison.png` : Comparaison des différentes fonctions de récompense
- `sharpe_vs_simple.png` : Comparaison entre la récompense Sharpe et la récompense simple
- `drawdown_penalty.png` : Impact des pénalités de drawdown sur la performance

## Utilisation dans les rapports

Les visualisations générées peuvent être utilisées dans des rapports ou des notebooks Jupyter pour illustrer les performances des différents algorithmes. Chaque phase du projet possède ses propres types de visualisations, adaptés aux spécificités de cette phase.

## Personnalisation

Pour personnaliser le style des visualisations, vous pouvez modifier les fonctions de génération de graphiques dans les fichiers correspondants. Tous les graphiques utilisent matplotlib et seaborn, ce qui permet une personnalisation flexible.

Pour générer toutes les visualisations en une seule fois, utilisez le script :
```bash
python -m ai_trading.examples.generate_all_visualizations
``` 