import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import praw
import tweepy
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlternativeDataCollector:
    """Collecte et analyse les données alternatives des réseaux sociaux."""

    def __init__(
        self,
        twitter_api_key: Optional[str] = None,
        twitter_api_secret: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
    ):
        """
        Initialise le collecteur de données alternatives.

        Args:
            twitter_api_key (str): Clé API Twitter
            twitter_api_secret (str): Secret API Twitter
            reddit_client_id (str): ID client Reddit
            reddit_client_secret (str): Secret client Reddit
        """
        # Initialisation Twitter
        if twitter_api_key and twitter_api_secret:
            try:
                auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
                self.twitter_api = tweepy.API(auth)
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de Twitter: {e}")
                self.twitter_api = None
        else:
            self.twitter_api = None

        # Initialisation Reddit
        if reddit_client_id and reddit_client_secret:
            try:
                self.reddit_api = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent="crypto_analysis_bot/1.0",
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de Reddit: {e}")
                self.reddit_api = None
        else:
            self.reddit_api = None

    def analyze_social_sentiment(
        self, keyword: str, limit: int = 100
    ) -> Dict[str, float]:
        """
        Analyse le sentiment sur les réseaux sociaux pour un mot-clé donné.

        Args:
            keyword (str): Mot-clé à analyser
            limit (int): Nombre de posts à analyser

        Returns:
            Dict[str, float]: Métriques de sentiment
        """
        sentiments = {
            "twitter": {"polarity": [], "subjectivity": [], "volume": 0},
            "reddit": {"polarity": [], "subjectivity": [], "volume": 0},
        }

        # Analyse Twitter
        if self.twitter_api:
            try:
                tweets = self.twitter_api.search_tweets(
                    q=keyword, lang="en", count=limit
                )
                for tweet in tweets:
                    analysis = TextBlob(tweet.text)
                    sentiments["twitter"]["polarity"].append(
                        analysis.sentiment.polarity
                    )
                    sentiments["twitter"]["subjectivity"].append(
                        analysis.sentiment.subjectivity
                    )
                    sentiments["twitter"]["volume"] += 1
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse Twitter: {e}")

        # Analyse Reddit
        if self.reddit_api:
            try:
                subreddit = self.reddit_api.subreddit("cryptocurrency")
                posts = subreddit.search(keyword, limit=limit)
                for post in posts:
                    analysis = TextBlob(post.title + " " + post.selftext)
                    sentiments["reddit"]["polarity"].append(analysis.sentiment.polarity)
                    sentiments["reddit"]["subjectivity"].append(
                        analysis.sentiment.subjectivity
                    )
                    sentiments["reddit"]["volume"] += 1
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse Reddit: {e}")

        # Calcul des moyennes
        results = {}
        for platform in ["twitter", "reddit"]:
            if sentiments[platform]["volume"] > 0:
                results[f"{platform}_avg_polarity"] = np.mean(
                    sentiments[platform]["polarity"]
                )
                results[f"{platform}_avg_subjectivity"] = np.mean(
                    sentiments[platform]["subjectivity"]
                )
                results[f"{platform}_volume"] = sentiments[platform]["volume"]
            else:
                # Ajout de valeurs par défaut si aucune donnée n'est disponible
                results[f"{platform}_avg_polarity"] = 0.0
                results[f"{platform}_avg_subjectivity"] = 0.0
                results[f"{platform}_volume"] = 0

        return results

    def collect_alternative_data(
        self,
        keywords: List[str],
        duration_minutes: int = 60,
        interval_seconds: int = 300,
    ) -> pd.DataFrame:
        """
        Collecte les données alternatives sur une période donnée.

        Args:
            keywords (List[str]): Liste des mots-clés à analyser
            duration_minutes (int): Durée totale de la collecte en minutes
            interval_seconds (int): Intervalle entre chaque collecte en secondes

        Returns:
            pd.DataFrame: DataFrame contenant les données collectées
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        data = []

        while datetime.now() < end_time:
            current_time = datetime.now()
            row = {"timestamp": current_time}

            # Analyse du sentiment pour chaque mot-clé
            for keyword in keywords:
                sentiment_data = self.analyze_social_sentiment(keyword)
                for key, value in sentiment_data.items():
                    row[f"{keyword}_{key}"] = value

            data.append(row)

            # Attendre jusqu'au prochain intervalle
            next_collection = datetime.now() + timedelta(seconds=interval_seconds)
            sleep_time = (next_collection - datetime.now()).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)

        return pd.DataFrame(data)
