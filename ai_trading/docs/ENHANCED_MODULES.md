# Modules améliorés pour la collecte et le prétraitement des données

Ce document explique comment utiliser les modules améliorés pour la collecte et le prétraitement des données de cryptomonnaies.

## EnhancedDataCollector

Le collecteur de données amélioré permet de récupérer des données de plusieurs sources et de les fusionner pour obtenir des données plus complètes et robustes.

### Sources de données

- **CoinGecko API** - Données historiques de prix, volumes et capitalisation
- **CoinCap API** - Données en temps réel et historiques
- **CryptoCompare API** - Données OHLCV et informations sur les échanges
- **Alternative.me** - Indice Fear & Greed (sentiment du marché)
- **Crypto Panic** - Actualités sur les cryptomonnaies

### Exemples d'utilisation

```python
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector

# Initialisation du collecteur
collector = EnhancedDataCollector()

# Récupération des données de prix fusionnées
bitcoin_data = collector.get_merged_price_data(
    coin_id='bitcoin',
    days=30,
    vs_currency='usd'
)

# Récupération de l'indice Fear & Greed
fear_greed = collector.get_fear_greed_index(days=30)

# Récupération des actualités
news = collector.get_crypto_news(limit=10)

# Sauvegarde des données
collector.save_data(bitcoin_data, 'bitcoin_30d.csv')
```

## EnhancedPreprocessor

Le préprocesseur amélioré offre des fonctionnalités avancées pour le nettoyage, la normalisation et la création de features pour les données de marché et textuelles.

### Fonctionnalités principales

#### EnhancedMarketDataPreprocessor
- Nettoyage des données multi-sources
- Normalisation avancée
- Création de features techniques étendues (plus de 30 indicateurs)
- Création de features décalées (lags)
- Création de variables cibles personnalisées

#### EnhancedTextDataPreprocessor
- Nettoyage avancé des textes (suppression des URLs, mentions, hashtags, etc.)
- Tokenization robuste avec gestion des erreurs
- Extraction de mots-clés liés au sentiment
- Prétraitement des actualités et données sociales

### Exemples d'utilisation

```python
from ai_trading.utils.enhanced_preprocessor import EnhancedMarketDataPreprocessor, EnhancedTextDataPreprocessor

# Prétraitement des données de marché
market_preprocessor = EnhancedMarketDataPreprocessor(scaling_method='minmax')

# Chargement des données
data = pd.read_csv('bitcoin_30d.csv', index_col=0, parse_dates=True)

# Prétraitement complet
processed_data = market_preprocessor.preprocess_market_data(data)

# Prétraitement des données textuelles
text_preprocessor = EnhancedTextDataPreprocessor(language='english')

# Prétraitement des actualités
news_data = [
    {'title': 'Bitcoin Surges', 'body': 'Bitcoin price increased by 10%'},
    {'title': 'Ethereum Update', 'body': 'Ethereum 2.0 is coming soon'}
]
processed_news = text_preprocessor.preprocess_news_data(news_data)

# Extraction des mots-clés
keywords = text_preprocessor.extract_sentiment_keywords(
    processed_news['tokens_title'].tolist()
)
```

## Intégration avec les modèles

Les données prétraitées peuvent être utilisées directement avec les modèles d'analyse de sentiment et de prédiction :

```python
# Exemple d'intégration avec un modèle de sentiment
from ai_trading.llm.sentiment_analysis.sentiment_model import SentimentModel

sentiment_model = SentimentModel()
sentiment_scores = sentiment_model.analyze(processed_news['clean_title'].tolist())

# Exemple d'intégration avec un modèle de prédiction
from ai_trading.ml.trading_signals.ml_model import MLModel

ml_model = MLModel()
signals = ml_model.predict(processed_data)
```

## Bonnes pratiques

1. **Toujours nettoyer les données** avant de les normaliser ou de créer des features
2. **Gérer les valeurs manquantes** de manière appropriée selon le contexte
3. **Éviter le data leakage** en séparant correctement les données d'entraînement et de test
4. **Respecter les limites de rate** des APIs pour éviter d'être bloqué
5. **Sauvegarder les données brutes** avant de les prétraiter pour pouvoir les réutiliser 