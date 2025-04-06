"""
Module de collecte de données pour le trading de cryptomonnaies.
Gère la collecte des données de prix, volumes, actualités et réseaux sociaux.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import os

import pandas as pd
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException
import tweepy
from newsapi import NewsApiClient
from pycoingecko import CoinGeckoAPI
from requests import Session
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class CryptoDataCollector:
    """Collecte les données de prix et volumes des cryptomonnaies."""
    
    def __init__(
        self,
        binance_api_key: str = None,
        binance_api_secret: str = None,
        cmc_api_key: str = None
    ):
        """
        Initialise le collecteur de données crypto.
        
        Args:
            binance_api_key: Clé API Binance
            binance_api_secret: Secret API Binance
            cmc_api_key: Clé API CoinMarketCap
        """
        # Configuration Binance
        self.binance_api_key = binance_api_key or os.getenv('BINANCE_API_KEY')
        self.binance_api_secret = binance_api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not (self.binance_api_key and self.binance_api_secret):
            logger.warning("Clés API Binance non fournies. Utilisation de l'API publique avec limites.")
            self.binance_client = Client()
        else:
            self.binance_client = Client(self.binance_api_key, self.binance_api_secret)
        
        # Configuration CoinGecko
        self.coingecko_client = CoinGeckoAPI()
        
        # Configuration CoinMarketCap
        self.cmc_api_key = cmc_api_key or os.getenv('CMC_API_KEY')
        if self.cmc_api_key:
            self.cmc_headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': self.cmc_api_key,
            }
            self.cmc_session = Session()
            self.cmc_session.headers.update(self.cmc_headers)
        else:
            logger.warning("Clé API CoinMarketCap non fournie. Fonctionnalités CMC désactivées.")
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Union[str, datetime],
        end_time: Optional[Union[str, datetime]] = None,
        source: str = 'binance'
    ) -> pd.DataFrame:
        """
        Récupère les données historiques pour un symbole.
        
        Args:
            symbol: Paire de trading (ex: 'BTCUSDT' pour Binance, 'bitcoin' pour CoinGecko)
            interval: Intervalle de temps ('1m', '5m', '1h', '1d', etc.)
            start_time: Date de début
            end_time: Date de fin (optionnel, utilise la date actuelle par défaut)
            source: Source des données ('binance', 'coingecko')
            
        Returns:
            DataFrame avec les données OHLCV
        """
        if source.lower() == 'binance':
            return self._get_binance_historical_klines(symbol, interval, start_time, end_time)
        elif source.lower() == 'coingecko':
            return self._get_coingecko_historical_data(symbol, start_time, end_time)
        else:
            raise ValueError(f"Source non supportée: {source}")
    
    def _get_binance_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Union[str, datetime],
        end_time: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """Récupère les données historiques depuis Binance."""
        try:
            # Conversion des dates si nécessaire
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
            elif end_time is None:
                end_time = datetime.now()
            
            # Récupération des données
            klines = self.binance_client.get_historical_klines(
                symbol,
                interval,
                start_time.strftime('%Y-%m-%d'),
                end_time.strftime('%Y-%m-%d')
            )
            
            # Conversion en DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # Nettoyage et formatage
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Conversion des types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération des données Binance: {e}")
            raise
    
    def _get_coingecko_historical_data(
        self,
        coin_id: str,
        start_time: Union[str, datetime],
        end_time: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """Récupère les données historiques depuis CoinGecko."""
        try:
            # Conversion des dates en timestamps
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
            elif end_time is None:
                end_time = datetime.now()
            
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            # Récupération des données
            data = self.coingecko_client.get_coin_market_chart_range_by_id(
                id=coin_id,
                vs_currency='usd',
                from_timestamp=start_ts,
                to_timestamp=end_ts
            )
            
            # Création des séries temporelles
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            
            # Fusion des données
            df = prices.merge(volumes, on='timestamp').merge(market_caps, on='timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ajout des colonnes OHLC (approximation basée sur le prix de clôture)
            df['close'] = df['price']
            df['open'] = df['price'].shift(1)
            df['high'] = df['price']
            df['low'] = df['price']
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données CoinGecko: {e}")
            raise
    
    def get_current_price(
        self,
        symbol: str,
        source: str = 'binance'
    ) -> Dict[str, float]:
        """
        Récupère le prix actuel d'un symbole depuis différentes sources.
        
        Args:
            symbol: Identifiant de la crypto ('BTCUSDT' pour Binance, 'bitcoin' pour CoinGecko)
            source: Source des données ('binance', 'coingecko', 'cmc', 'all')
            
        Returns:
            Dict avec les prix par source
        """
        prices = {}
        
        if source.lower() in ['binance', 'all']:
            try:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                prices['binance'] = float(ticker['price'])
            except BinanceAPIException as e:
                logger.error(f"Erreur Binance: {e}")
        
        if source.lower() in ['coingecko', 'all']:
            try:
                data = self.coingecko_client.get_price(
                    ids=symbol,
                    vs_currencies='usd'
                )
                prices['coingecko'] = data[symbol]['usd']
            except Exception as e:
                logger.error(f"Erreur CoinGecko: {e}")
        
        if source.lower() in ['cmc', 'all'] and self.cmc_api_key:
            try:
                url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
                params = {'symbol': symbol.replace('USDT', '')}
                response = self.cmc_session.get(url, params=params)
                data = response.json()
                prices['cmc'] = data['data'][symbol]['quote']['USD']['price']
            except Exception as e:
                logger.error(f"Erreur CoinMarketCap: {e}")
        
        if not prices:
            raise ValueError(f"Aucun prix récupéré pour {symbol} depuis {source}")
        
        return prices
    
    def get_market_info(
        self,
        symbol: str,
        source: str = 'coingecko'
    ) -> Dict:
        """
        Récupère les informations détaillées sur une cryptomonnaie.
        
        Args:
            symbol: Identifiant de la crypto
            source: Source des données ('coingecko', 'cmc')
            
        Returns:
            Dict avec les informations de marché
        """
        if source.lower() == 'coingecko':
            try:
                return self.coingecko_client.get_coin_by_id(
                    id=symbol,
                    localization=False,
                    tickers=True,
                    market_data=True,
                    community_data=True,
                    developer_data=True
                )
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des infos CoinGecko: {e}")
                raise
                
        elif source.lower() == 'cmc' and self.cmc_api_key:
            try:
                url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/info'
                params = {'symbol': symbol.replace('USDT', '')}
                response = self.cmc_session.get(url, params=params)
                return response.json()['data'][symbol]
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des infos CMC: {e}")
                raise
        else:
            raise ValueError(f"Source non supportée: {source}")
    
    def get_trending_coins(self, source: str = 'coingecko') -> List[Dict]:
        """
        Récupère les cryptomonnaies tendance.
        
        Args:
            source: Source des données ('coingecko', 'cmc')
            
        Returns:
            Liste des cryptos tendance
        """
        if source.lower() == 'coingecko':
            try:
                return self.coingecko_client.get_search_trending()['coins']
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des tendances CoinGecko: {e}")
                raise
                
        elif source.lower() == 'cmc' and self.cmc_api_key:
            try:
                url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/latest'
                response = self.cmc_session.get(url)
                return response.json()['data']
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des tendances CMC: {e}")
                raise
        else:
            raise ValueError(f"Source non supportée: {source}")

class NewsCollector:
    """Collecte les actualités crypto depuis diverses sources."""
    
    def __init__(self, api_key: str = None):
        """
        Initialise le collecteur d'actualités.
        
        Args:
            api_key: Clé API NewsAPI (optionnel, utilise les variables d'environnement par défaut)
        """
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        if not self.api_key:
            raise ValueError("Clé API NewsAPI requise")
        
        self.client = NewsApiClient(api_key=self.api_key)
    
    def get_crypto_news(
        self,
        query: str = "bitcoin OR cryptocurrency OR blockchain",
        from_date: Optional[Union[str, datetime]] = None,
        to_date: Optional[Union[str, datetime]] = None,
        language: str = "en",
        sort_by: str = "publishedAt"
    ) -> List[Dict]:
        """
        Récupère les actualités crypto.
        
        Args:
            query: Termes de recherche
            from_date: Date de début
            to_date: Date de fin
            language: Code langue ('en', 'fr', etc.)
            sort_by: Tri ('relevancy', 'popularity', 'publishedAt')
            
        Returns:
            Liste d'articles
        """
        try:
            # Gestion des dates
            if from_date is None:
                from_date = datetime.now() - timedelta(days=7)
            if isinstance(from_date, datetime):
                from_date = from_date.strftime('%Y-%m-%d')
                
            if to_date is None:
                to_date = datetime.now()
            if isinstance(to_date, datetime):
                to_date = to_date.strftime('%Y-%m-%d')
            
            # Récupération des articles
            response = self.client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by=sort_by
            )
            
            return response['articles']
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des actualités: {e}")
            raise

class SocialMediaCollector:
    """Collecte les données des réseaux sociaux (Twitter)."""
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        access_token: str = None,
        access_token_secret: str = None
    ):
        """
        Initialise le collecteur de données sociales.
        
        Args:
            api_key: Clé API Twitter
            api_secret: Secret API Twitter
            access_token: Token d'accès
            access_token_secret: Secret du token d'accès
        """
        self.api_key = api_key or os.getenv('TWITTER_API_KEY')
        self.api_secret = api_secret or os.getenv('TWITTER_API_SECRET')
        self.access_token = access_token or os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = access_token_secret or os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            raise ValueError("Toutes les clés API Twitter sont requises")
        
        # Authentification
        auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth)
    
    def get_crypto_tweets(
        self,
        query: str,
        count: int = 100,
        lang: str = "en",
        result_type: str = "mixed"
    ) -> List[Dict]:
        """
        Récupère les tweets sur les cryptomonnaies.
        
        Args:
            query: Termes de recherche
            count: Nombre de tweets à récupérer
            lang: Code langue
            result_type: Type de résultats ('mixed', 'recent', 'popular')
            
        Returns:
            Liste de tweets
        """
        try:
            tweets = []
            for tweet in tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                lang=lang,
                result_type=result_type,
                tweet_mode="extended"
            ).items(count):
                tweets.append({
                    'id': tweet.id,
                    'created_at': tweet.created_at,
                    'text': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'retweets': tweet.retweet_count,
                    'likes': tweet.favorite_count
                })
            return tweets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des tweets: {e}")
            raise 