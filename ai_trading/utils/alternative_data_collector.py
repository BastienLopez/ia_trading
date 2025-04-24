import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time
import requests
from web3 import Web3
import tweepy
import praw
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlternativeDataCollector:
    """Collecte et analyse les données alternatives (on-chain et réseaux sociaux)."""
    
    def __init__(self, 
                 eth_node_url: str = None,
                 twitter_api_key: Optional[str] = None,
                 twitter_api_secret: Optional[str] = None,
                 reddit_client_id: Optional[str] = None,
                 reddit_client_secret: Optional[str] = None):
        """
        Initialise le collecteur de données alternatives.
        
        Args:
            eth_node_url (str): URL du nœud Ethereum
            twitter_api_key (str): Clé API Twitter
            twitter_api_secret (str): Secret API Twitter
            reddit_client_id (str): ID client Reddit
            reddit_client_secret (str): Secret client Reddit
        """
        self.w3 = None
        if eth_node_url:
            try:
                provider = Web3.HTTPProvider(eth_node_url)
                self.w3 = Web3(provider)
                if not self.w3.is_connected():
                    logger.warning("Impossible de se connecter au nœud Ethereum")
                    self.w3 = None
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de Web3: {e}")
                self.w3 = None
            
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
                    user_agent="crypto_analysis_bot/1.0"
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de Reddit: {e}")
                self.reddit_api = None
        else:
            self.reddit_api = None
            
    def get_onchain_metrics(self, address: str) -> Dict[str, float]:
        """
        Récupère les métriques on-chain pour une adresse donnée.
        
        Args:
            address (str): Adresse Ethereum à analyser
            
        Returns:
            Dict[str, float]: Métriques on-chain
        """
        if not self.w3:
            logger.warning("Web3 n'est pas initialisé")
            return {}
            
        try:
            # Vérification de la validité de l'adresse
            if not self.w3.is_address(address):
                raise ValueError("Adresse Ethereum invalide")
                
            # Récupération des données de base
            balance = self.w3.eth.get_balance(address)
            tx_count = self.w3.eth.get_transaction_count(address)
            
            # Récupération des dernières transactions
            latest_block = self.w3.eth.block_number
            transactions = []
            
            # Analyse des 100 derniers blocs
            for i in range(100):
                block = self.w3.eth.get_block(latest_block - i, True)
                for tx in block.transactions:
                    if tx['from'].lower() == address.lower() or tx['to'] and tx['to'].lower() == address.lower():
                        transactions.append(tx)
                        
            # Calcul des métriques
            if transactions:
                volumes = [float(tx['value']) / 1e18 for tx in transactions]  # Conversion en ETH
                avg_volume = np.mean(volumes)
                max_volume = np.max(volumes)
                tx_frequency = len(transactions) / 100  # Transactions par bloc
            else:
                avg_volume = max_volume = tx_frequency = 0
                
            return {
                'balance': float(balance) / 1e18,  # Conversion en ETH
                'transaction_count': tx_count,
                'average_volume': avg_volume,
                'max_volume': max_volume,
                'transaction_frequency': tx_frequency
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques on-chain: {e}")
            return {}
            
    def analyze_social_sentiment(self, keyword: str, limit: int = 100) -> Dict[str, float]:
        """
        Analyse le sentiment sur les réseaux sociaux pour un mot-clé donné.
        
        Args:
            keyword (str): Mot-clé à analyser
            limit (int): Nombre de posts à analyser
            
        Returns:
            Dict[str, float]: Métriques de sentiment
        """
        sentiments = {
            'twitter': {'polarity': [], 'subjectivity': [], 'volume': 0},
            'reddit': {'polarity': [], 'subjectivity': [], 'volume': 0}
        }
        
        # Analyse Twitter
        if self.twitter_api:
            try:
                tweets = self.twitter_api.search_tweets(q=keyword, lang="en", count=limit)
                for tweet in tweets:
                    analysis = TextBlob(tweet.text)
                    sentiments['twitter']['polarity'].append(analysis.sentiment.polarity)
                    sentiments['twitter']['subjectivity'].append(analysis.sentiment.subjectivity)
                    sentiments['twitter']['volume'] += 1
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse Twitter: {e}")
                
        # Analyse Reddit
        if self.reddit_api:
            try:
                subreddit = self.reddit_api.subreddit('cryptocurrency')
                posts = subreddit.search(keyword, limit=limit)
                for post in posts:
                    analysis = TextBlob(post.title + " " + post.selftext)
                    sentiments['reddit']['polarity'].append(analysis.sentiment.polarity)
                    sentiments['reddit']['subjectivity'].append(analysis.sentiment.subjectivity)
                    sentiments['reddit']['volume'] += 1
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse Reddit: {e}")
                
        # Calcul des moyennes
        results = {}
        for platform in ['twitter', 'reddit']:
            if sentiments[platform]['volume'] > 0:
                results[f'{platform}_avg_polarity'] = np.mean(sentiments[platform]['polarity'])
                results[f'{platform}_avg_subjectivity'] = np.mean(sentiments[platform]['subjectivity'])
                results[f'{platform}_volume'] = sentiments[platform]['volume']
                
        return results
        
    def collect_alternative_data(self, 
                               addresses: List[str],
                               keywords: List[str],
                               duration_minutes: int = 60,
                               interval_seconds: int = 300) -> pd.DataFrame:
        """
        Collecte les données alternatives sur une période donnée.
        
        Args:
            addresses (List[str]): Liste d'adresses Ethereum à surveiller
            keywords (List[str]): Liste de mots-clés pour l'analyse de sentiment
            duration_minutes (int): Durée de collecte en minutes
            interval_seconds (int): Intervalle entre chaque collecte en secondes
            
        Returns:
            pd.DataFrame: Données collectées
        """
        logger.info(f"Début de la collecte des données alternatives pour {duration_minutes} minutes...")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        data = []
        
        while datetime.now() < end_time:
            current_data = {'timestamp': datetime.now()}
            
            # Collecte des métriques on-chain
            for address in addresses:
                metrics = self.get_onchain_metrics(address)
                current_data.update({f'{address}_{k}': v for k, v in metrics.items()})
                
            # Collecte des sentiments sociaux
            for keyword in keywords:
                sentiments = self.analyze_social_sentiment(keyword)
                current_data.update({f'{keyword}_{k}': v for k, v in sentiments.items()})
                
            data.append(current_data)
            time.sleep(interval_seconds)
            
        df = pd.DataFrame(data)
        logger.info(f"Collecte terminée. {len(df)} échantillons collectés.")
        return df 