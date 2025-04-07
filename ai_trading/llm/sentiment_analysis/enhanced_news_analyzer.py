"""
Module amélioré d'analyse des actualités crypto pour extraire le sentiment et les entités.
Utilise des modèles LLM avancés et des techniques de NLP pour une analyse plus complète.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import re
import json
import hashlib
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Importation conditionnelle des bibliothèques LLM
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "La bibliothèque 'transformers' n'est pas disponible. Certaines fonctionnalités seront limitées."
    )
    TRANSFORMERS_AVAILABLE = False

from .sentiment_tools import SentimentCache, text_hash
from .news_analyzer import NewsAnalyzer


class EnhancedNewsAnalyzer(NewsAnalyzer):
    """Version améliorée avec visualisations et gestion de cache."""

    def __init__(
        self,
        enable_cache: bool = True,
        cache_dir: str = "data/sentiment/cache",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cache = SentimentCache(cache_dir) if enable_cache else None

    def analyze_news(self, news_data: List[Dict]) -> pd.DataFrame:
        """Analyse un batch d'actualités avec gestion de cache."""
        df = super().analyze_news(news_data)

        # Création robuste du score de sentiment
        if "global_sentiment" not in df.columns:
            logger.error("Colonne 'global_sentiment' manquante - initialisation à zéro")
            df["sentiment_score"] = 0.0
            return df

        df["sentiment_score"] = df["global_sentiment"].apply(
            lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0
        )
        return df

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Version complète avec toutes les métriques requises"""
        report = {
            "total_articles": len(df),
            "average_sentiment": df.get("sentiment_score", pd.Series([0])).mean(),
            "period": {
                "start": df["published_at"].min() if not df.empty else None,
                "end": df["published_at"].max() if not df.empty else None,
            },
            "sentiment_distribution": (
                df["global_sentiment_label"].value_counts().to_dict()
                if "global_sentiment_label" in df.columns
                else {}
            ),
            "crypto_mentions": self._get_top_entities(df, "crypto_entities"),
            "sentiment_trend": self._calculate_daily_trend(df),
            "most_positive_article": self._find_extreme_article(df, "max"),
            "most_negative_article": self._find_extreme_article(df, "min"),
        }
        return report

    def _get_cached_result(self, text_hash: str) -> Optional[Dict]:
        """Récupère un résultat depuis le cache."""
        return self.cache.load(text_hash) if self.cache else None

    def _cache_result(self, text_hash: str, result: Dict) -> None:
        """Stocke un résultat dans le cache."""
        if self.cache:
            self.cache.save(text_hash, result)

    def _calculate_daily_trend(self, df: pd.DataFrame) -> Dict:
        """Version sécurisée avec gestion des colonnes manquantes"""
        if "sentiment_score" not in df.columns or "published_at" not in df.columns:
            logger.warning("Colonnes manquantes pour le calcul des tendances")
            return {}

        try:
            df["date"] = pd.to_datetime(df["published_at"]).dt.date
            return df.groupby("date")["sentiment_score"].mean().to_dict()
        except KeyError:
            logger.error("Erreur lors de l'accès aux colonnes nécessaires")
            return {}

    def _get_top_entities(self, df: pd.DataFrame, entity_type: str) -> Dict:
        """Version sécurisée avec gestion des colonnes manquantes"""
        if entity_type not in df.columns:
            return {}

        return dict(df[entity_type].explode().value_counts().head(10))

    def _find_extreme_article(self, df: pd.DataFrame, extremum: str) -> Dict:
        """Trouve l'article le plus positif/négatif de manière sécurisée"""
        if df.empty or "sentiment_score" not in df.columns:
            return {}

        try:
            if extremum == "max":
                return df.loc[df["sentiment_score"].idxmax()].to_dict()
            return df.loc[df["sentiment_score"].idxmin()].to_dict()
        except ValueError:
            return {}

    def plot_trends(self, df: pd.DataFrame, filename: str) -> None:
        """Génère une visualisation des tendances."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x="date",
            y="sentiment_score",
            data=df.assign(date=pd.to_datetime(df["published_at"]).dt.date),
            estimator="mean",
            ci=None,
        )
        plt.title("Évolution du sentiment moyen")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation de l'analyseur
    analyzer = EnhancedNewsAnalyzer()

    # Exemple d'actualités
    news_examples = [
        {
            "title": "Bitcoin Surges to $60,000 as Institutional Adoption Grows",
            "body": "Bitcoin reached a new all-time high of $60,000 today as more institutional investors are entering the cryptocurrency market. Major companies like Tesla and MicroStrategy have added BTC to their balance sheets.",
            "published_at": "2023-03-15T12:30:00Z",
        },
        {
            "title": "Ethereum Price Drops 10% Following Network Congestion",
            "body": "Ethereum (ETH) experienced a significant price drop of 10% in the last 24 hours due to network congestion and high gas fees. Developers are working on solutions to address these scaling issues.",
            "published_at": "2023-03-14T09:15:00Z",
        },
        {
            "title": "Solana Ecosystem Expands with New DeFi Projects",
            "body": "The Solana blockchain is seeing rapid growth in its DeFi ecosystem with several new projects launching this month. The total value locked (TVL) in Solana DeFi has increased by 25% in the past week.",
            "published_at": "2023-03-13T16:45:00Z",
        },
    ]

    # Analyse des actualités
    enriched_news = analyzer.analyze_news(news_examples)

    # Conversion en DataFrame
    news_df = pd.DataFrame(enriched_news)

    # Génération du rapport
    report = analyzer.generate_report(news_df)

    # Affichage des résultats
    print("\nRapport d'analyse de sentiment:")
    print(f"Total d'articles: {report['total_articles']}")
    print(f"Distribution des sentiments: {report['sentiment_distribution']}")
    print(f"Sentiment moyen: {report['average_sentiment']:.2f}")
    print("\nCryptomonnaies les plus mentionnées:")
    for crypto, count in report["crypto_mentions"].items():
        print(f"- {crypto}: {count} mentions")

    print("\nArticle le plus positif:")
    print(f"- {report['most_positive_article']['title']}")
    print(f"- Score: {report['most_positive_article']['score']:.2f}")

    print("\nArticle le plus négatif:")
    print(f"- {report['most_negative_article']['title']}")
    print(f"- Score: {report['most_negative_article']['score']:.2f}")

    print("\nSentiment par cryptomonnaie:")
    for crypto, data in report["sentiment_by_crypto"].items():
        print(
            f"- {crypto}: {data['sentiment_label']} ({data['average_sentiment']:.2f}) - {data['article_count']} articles"
        )
