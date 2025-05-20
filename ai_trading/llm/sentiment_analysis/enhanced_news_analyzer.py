"""
Module amélioré d'analyse des actualités crypto pour extraire le sentiment et les entités.
Utilise des modèles LLM avancés et des techniques de NLP pour une analyse plus complète.
"""

import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob

from .sentiment_utils import SentimentCache, get_llm_client

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNewsAnalyzer:
    """Version améliorée avec visualisations et gestion de cache."""

    def __init__(
        self,
        enable_cache: bool = True,
        cache_dir: str = "ai_trading/info_retour/sentiment_cache",
        **kwargs,
    ):
        """
        Initialise l'analyseur de nouvelles amélioré.
        
        Args:
            enable_cache: Active la mise en cache des résultats
            cache_dir: Répertoire pour le cache
            **kwargs: Arguments additionnels
        """
        self.llm_client = get_llm_client()
        self.cache = SentimentCache(cache_dir) if enable_cache else None
        self.visualization_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "visualizations",
            "sentiment",
        )
        os.makedirs(self.visualization_dir, exist_ok=True)

    def analyze_news(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Analyse un batch d'actualités avec gestion de cache.
        
        Args:
            news_data: Liste d'actualités à analyser
            
        Returns:
            DataFrame avec les résultats d'analyse
        """
        results = []
        for news in news_data:
            # Analyse du titre et du contenu
            title_sentiment = self._analyze_text(news.get("title", ""))
            body_sentiment = self._analyze_text(news.get("body", ""))
            
            # Extraction des entités
            entities = self._extract_entities(news.get("body", ""))
            
            # Calcul du sentiment global
            global_sentiment = self._calculate_global_sentiment(
                title_sentiment, body_sentiment
            )
            
            results.append({
                **news,
                "title_sentiment": title_sentiment,
                "body_sentiment": body_sentiment,
                "global_sentiment": global_sentiment,
                "entities": entities,
                "published_at": pd.to_datetime(news.get("published_at")),
            })
        
        return pd.DataFrame(results)

    def _analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyse le sentiment d'un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire avec le sentiment et le score
        """
        if not text:
            return {"label": "neutral", "score": 0.0}

        # Utilisation du cache si disponible
        if self.cache:
            cached_result = self.cache.load(text)
            if cached_result:
                return cached_result

        try:
            # Analyse avec le modèle LLM
            if self.llm_client:
                results = self.llm_client(text)
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                    if isinstance(result, dict):
                        sentiment = {
                            "label": result.get("label", "neutral"),
                            "score": result.get("score", 0.0),
                        }
                    else:
                        sentiment = {"label": "neutral", "score": 0.0}
                else:
                    sentiment = {"label": "neutral", "score": 0.0}
            else:
                # Repli sur TextBlob si le modèle n'est pas disponible
                blob = TextBlob(text)
                score = blob.sentiment.polarity
                sentiment = {
                    "label": self._get_sentiment_label(score),
                    "score": score,
                }

            # Mise en cache du résultat
            if self.cache:
                self.cache.save(text, sentiment)

            return sentiment

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du sentiment: {e}")
            return {"label": "neutral", "score": 0.0}

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait les entités du texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire avec les entités par catégorie
        """
        crypto_patterns = [
            r"bitcoin|btc",
            r"ethereum|eth",
            r"ripple|xrp",
            r"dogecoin|doge",
            r"cardano|ada",
        ]
        
        entities = {
            "crypto_entities": [],
            "money_entities": [],
            "percentage_entities": [],
        }
        
        # Extraction des cryptomonnaies
        for pattern in crypto_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                entities["crypto_entities"].extend(matches)
        
        # Extraction des montants
        money_matches = re.findall(r"\$\d+(?:,\d+)*(?:\.\d+)?", text)
        entities["money_entities"].extend(money_matches)
        
        # Extraction des pourcentages
        percentage_matches = re.findall(r"\d+(?:\.\d+)?%", text)
        entities["percentage_entities"].extend(percentage_matches)
        
        return entities

    def _calculate_global_sentiment(
        self, title_sentiment: Dict, body_sentiment: Dict
    ) -> Dict[str, Union[str, float]]:
        """
        Calcule le sentiment global en combinant titre et contenu.
        
        Args:
            title_sentiment: Sentiment du titre
            body_sentiment: Sentiment du contenu
            
        Returns:
            Sentiment global
        """
        # Pondération titre/contenu (60/40)
        title_weight = 0.6
        body_weight = 0.4
        
        global_score = (
            title_sentiment["score"] * title_weight
            + body_sentiment["score"] * body_weight
        )
        
        return {
            "label": self._get_sentiment_label(global_score),
            "score": global_score,
        }

    def _get_sentiment_label(self, score: float) -> str:
        """
        Convertit un score en label de sentiment.
        
        Args:
            score: Score de sentiment (-1 à 1)
            
        Returns:
            Label de sentiment
        """
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        return "neutral"

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """
        Génère un rapport d'analyse complet.
        
        Args:
            df: DataFrame avec les résultats d'analyse
            
        Returns:
            Rapport d'analyse
        """
        if df.empty:
            return self._empty_report()

        return {
            "total_articles": len(df),
            "average_sentiment": df["global_sentiment"].apply(
                lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0
            ).mean(),
            "period": {
                "start": df["published_at"].min(),
                "end": df["published_at"].max(),
            },
            "sentiment_distribution": df["global_sentiment"].apply(
                lambda x: x.get("label") if isinstance(x, dict) else "neutral"
            ).value_counts().to_dict(),
            "entities": self._aggregate_entities(df),
            "sentiment_trend": self._calculate_sentiment_trend(df),
        }

    def _empty_report(self) -> Dict:
        """Retourne un rapport vide."""
        return {
            "total_articles": 0,
            "average_sentiment": 0.0,
            "period": {"start": None, "end": None},
            "sentiment_distribution": {},
            "entities": {},
            "sentiment_trend": {},
        }

    def _aggregate_entities(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Agrège les entités trouvées."""
        all_entities = {
            "crypto": {},
            "money": {},
            "percentage": {},
        }
        
        for _, row in df.iterrows():
            entities = row.get("entities", {})
            
            for crypto in entities.get("crypto_entities", []):
                all_entities["crypto"][crypto] = all_entities["crypto"].get(crypto, 0) + 1
                
            for money in entities.get("money_entities", []):
                all_entities["money"][money] = all_entities["money"].get(money, 0) + 1
                
            for percentage in entities.get("percentage_entities", []):
                all_entities["percentage"][percentage] = all_entities["percentage"].get(percentage, 0) + 1
        
        return all_entities

    def _calculate_sentiment_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule la tendance du sentiment au fil du temps."""
        df["date"] = df["published_at"].dt.date
        df["sentiment_score"] = df["global_sentiment"].apply(
            lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0
        )
        
        return df.groupby("date")["sentiment_score"].mean().to_dict()

    def plot_trends(self, df: pd.DataFrame, filename: str = "sentiment_trends.png") -> None:
        """
        Génère une visualisation des tendances.
        
        Args:
            df: DataFrame avec les données
            filename: Nom du fichier de sortie
        """
        plt.figure(figsize=(12, 6))
        
        df["date"] = pd.to_datetime(df["published_at"]).dt.date
        df["sentiment_score"] = df["global_sentiment"].apply(
            lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0
        )
        
        sns.lineplot(
            data=df,
            x="date",
            y="sentiment_score",
            estimator="mean",
            errorbar=None,
        )
        
        plt.title("Évolution du sentiment")
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, filename))
        plt.close()

    def plot_distribution(self, df: pd.DataFrame, filename: str = "sentiment_distribution.png") -> None:
        """
        Génère une visualisation de la distribution des sentiments.
        
        Args:
            df: DataFrame avec les données
            filename: Nom du fichier de sortie
        """
        plt.figure(figsize=(10, 6))
        
        sentiment_counts = df["global_sentiment"].apply(
            lambda x: x.get("label") if isinstance(x, dict) else "neutral"
        ).value_counts()
        
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        
        plt.title("Distribution des sentiments")
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, filename))
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
    for crypto, count in report["entities"]["crypto"].items():
        print(f"- {crypto}: {count} mentions")

    print("\nArticle le plus positif:")
    positive_article = news_df[news_df["global_sentiment"] == "positive"].iloc[0]
    print(f"- {positive_article['title']}")
    print(f"- Score: {positive_article['global_sentiment']['score']:.2f}")

    print("\nArticle le plus négatif:")
    negative_article = news_df[news_df["global_sentiment"] == "negative"].iloc[0]
    print(f"- {negative_article['title']}")
    print(f"- Score: {negative_article['global_sentiment']['score']:.2f}")

    print("\nSentiment par cryptomonnaie:")
    for crypto, data in report["entities"]["crypto"].items():
        print(
            f"- {crypto}: {data} - {data} articles"
        )
