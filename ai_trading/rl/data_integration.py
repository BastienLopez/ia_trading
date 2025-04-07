import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime, timedelta

from ai_trading.rl.data_processor import prepare_data_for_rl
from ai_trading.utils.enhanced_data_collector import DataCollector
from ai_trading.utils.enhanced_preprocessor import DataPreprocessor
from ai_trading.llm.sentiment_analysis.news_analyzer import NewsAnalyzer
from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer

# Configuration du logger
logger = logging.getLogger('DataIntegration')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class RLDataIntegrator:
    """
    Classe pour intégrer les données de marché et de sentiment pour l'apprentissage par renforcement.
    """
    
    def __init__(self, config=None):
        """
        Initialise l'intégrateur de données.
        
        Args:
            config (dict, optional): Configuration pour la collecte et le prétraitement des données
        """
        self.config = config or {}
        self.data_collector = DataCollector()
        self.data_preprocessor = DataPreprocessor()
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialAnalyzer()
        
        logger.info("Intégrateur de données RL initialisé")
    
    def collect_market_data(self, symbol, start_date, end_date, interval='1d'):
        """
        Collecte les données de marché pour une cryptomonnaie.
        
        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: 'BTC')
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')
            interval (str): Intervalle de temps ('1d', '1h', etc.)
            
        Returns:
            DataFrame: Données de marché
        """
        logger.info(f"Collecte des données de marché pour {symbol} du {start_date} au {end_date}")
        
        try:
            # Utiliser le collecteur de données existant
            market_data = self.data_collector.fetch_crypto_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            logger.info(f"Données de marché collectées: {len(market_data)} points")
            return market_data
        
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des données de marché: {str(e)}")
            # Créer des données synthétiques en cas d'erreur
            return self._generate_synthetic_market_data(start_date, end_date, interval)
    
    def collect_sentiment_data(self, symbol, start_date, end_date):
        """
        Collecte et analyse les données de sentiment pour une cryptomonnaie.
        
        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: 'BTC')
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame: Données de sentiment
        """
        logger.info(f"Collecte des données de sentiment pour {symbol} du {start_date} au {end_date}")
        
        try:
            # Collecter les actualités
            news_data = self.data_collector.fetch_crypto_news(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Analyser le sentiment des actualités
            if news_data is not None and not news_data.empty:
                news_sentiment = self.news_analyzer.analyze_news(news_data)
                logger.info(f"Sentiment des actualités analysé: {len(news_sentiment)} points")
            else:
                news_sentiment = pd.DataFrame()
                logger.warning("Aucune actualité collectée")
            
            # Collecter les données sociales (tweets, posts Reddit)
            social_data = self.data_collector.fetch_social_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Analyser le sentiment des données sociales
            if social_data is not None and not social_data.empty:
                social_sentiment = self.social_analyzer.analyze_social_posts(social_data)
                logger.info(f"Sentiment social analysé: {len(social_sentiment)} points")
            else:
                social_sentiment = pd.DataFrame()
                logger.warning("Aucune donnée sociale collectée")
            
            # Fusionner les sentiments des actualités et des données sociales
            sentiment_data = self._merge_sentiment_data(news_sentiment, social_sentiment)
            
            if sentiment_data.empty:
                logger.warning("Aucune donnée de sentiment disponible, génération de données synthétiques")
                sentiment_data = self._generate_synthetic_sentiment_data(start_date, end_date)
            
            return sentiment_data
        
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des données de sentiment: {str(e)}")
            # Créer des données synthétiques en cas d'erreur
            return self._generate_synthetic_sentiment_data(start_date, end_date)
    
    def preprocess_market_data(self, market_data):
        """
        Prétraite les données de marché.
        
        Args:
            market_data (DataFrame): Données de marché brutes
            
        Returns:
            DataFrame: Données de marché prétraitées
        """
        logger.info("Prétraitement des données de marché")
        
        try:
            # Utiliser le préprocesseur existant
            preprocessed_data = self.data_preprocessor.preprocess(market_data)
            
            # Ajouter des indicateurs techniques supplémentaires si nécessaire
            if 'rsi' not in preprocessed_data.columns:
                preprocessed_data = self.data_preprocessor.add_technical_indicators(preprocessed_data)
            
            logger.info(f"Données de marché prétraitées: {len(preprocessed_data)} points avec {len(preprocessed_data.columns)} features")
            return preprocessed_data
        
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données de marché: {str(e)}")
            # Retourner les données originales en cas d'erreur
            return market_data
    
    def integrate_data(self, market_data, sentiment_data=None, window_size=5, test_split=0.2):
        """
        Intègre les données de marché et de sentiment pour l'apprentissage par renforcement.
        
        Args:
            market_data (DataFrame): Données de marché prétraitées
            sentiment_data (DataFrame, optional): Données de sentiment
            window_size (int): Taille de la fenêtre d'observation
            test_split (float): Proportion des données à utiliser pour le test
            
        Returns:
            tuple: (train_data, test_data) DataFrames prêts pour l'RL
        """
        logger.info("Intégration des données pour l'apprentissage par renforcement")
        
        # Utiliser la fonction de préparation des données pour RL
        train_data, test_data = prepare_data_for_rl(
            market_data=market_data,
            sentiment_data=sentiment_data,
            window_size=window_size,
            test_split=test_split
        )
        
        return train_data, test_data
    
    def _merge_sentiment_data(self, news_sentiment, social_sentiment):
        """
        Fusionne les données de sentiment des actualités et des réseaux sociaux.
        
        Args:
            news_sentiment (DataFrame): Sentiment des actualités
            social_sentiment (DataFrame): Sentiment des réseaux sociaux
            
        Returns:
            DataFrame: Données de sentiment fusionnées
        """
        # Si l'une des sources est vide, retourner l'autre
        if news_sentiment.empty and not social_sentiment.empty:
            return social_sentiment
        elif not news_sentiment.empty and social_sentiment.empty:
            return news_sentiment
        elif news_sentiment.empty and social_sentiment.empty:
            return pd.DataFrame()
        
        # Agréger les sentiments par jour
        news_daily = news_sentiment.resample('D').mean()
        social_daily = social_sentiment.resample('D').mean()
        
        # Fusionner les deux sources
        merged = pd.merge(
            news_daily,
            social_daily,
            left_index=True,
            right_index=True,
            suffixes=('_news', '_social'),
            how='outer'
        )
        
        # Remplir les valeurs manquantes
        merged = merged.fillna(method='ffill').fillna(method='bfill')
        
        # Calculer un score de sentiment combiné
        if 'compound_score_news' in merged.columns and 'compound_score_social' in merged.columns:
            merged['compound_score'] = (merged['compound_score_news'] + merged['compound_score_social']) / 2
        
        return merged
    
    def _generate_synthetic_market_data(self, start_date, end_date, interval='1d'):
        """
        Génère des données de marché synthétiques pour les tests.
        
        Args:
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')
            interval (str): Intervalle de temps
            
        Returns:
            DataFrame: Données de marché synthétiques
        """
        logger.warning("Génération de données de marché synthétiques")
        
        # Convertir les dates en objets datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Déterminer le nombre de jours
        if interval == '1d':
            delta = timedelta(days=1)
        elif interval == '1h':
            delta = timedelta(hours=1)
        else:
            delta = timedelta(days=1)
        
        # Générer les dates
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += delta
        
        # Générer une marche aléatoire pour les prix
        np.random.seed(42)  # Pour la reproductibilité
        price = 10000  # Prix initial
        prices = [price]
        
        for _ in range(1, len(dates)):
            change = np.random.normal(0, 200)  # Changement aléatoire
            price += change
            prices.append(max(price, 100))  # Éviter les prix négatifs
        
        # Générer les volumes
        volumes = np.random.uniform(1000, 10000, len(dates))
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'close': prices,
            'open': [p * np.random.uniform(0.98, 1.0) for p in prices],
            'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'volume': volumes
        }, index=dates)
        
        return df
    
    def _generate_synthetic_sentiment_data(self, start_date, end_date):
        """
        Génère des données de sentiment synthétiques pour les tests.
        
        Args:
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame: Données de sentiment synthétiques
        """
        logger.warning("Génération de données de sentiment synthétiques")
        
        # Convertir les dates en objets datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Générer les dates (quotidien)
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        
        # Générer des scores de sentiment aléatoires
        np.random.seed(42)  # Pour la reproductibilité
        
        # Sentiment avec une tendance légèrement positive
        compound_scores = np.random.normal(0.1, 0.3, len(dates))
        compound_scores = np.clip(compound_scores, -1, 1)  # Limiter entre -1 et 1
        
        # Volume de sentiment
        sentiment_volumes = np.random.uniform(10, 100, len(dates))
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'compound_score': compound_scores,
            'positive_score': [max(0, (1 + cs) / 2) for cs in compound_scores],
            'negative_score': [max(0, (1 - cs) / 2) for cs in compound_scores],
            'neutral_score': np.random.uniform(0.2, 0.5, len(dates)),
            'sentiment_volume': sentiment_volumes
        }, index=dates)
        
        return df
    
    def visualize_integrated_data(self, market_data, sentiment_data=None, save_path=None):
        """
        Visualise les données intégrées.
        
        Args:
            market_data (DataFrame): Données de marché
            sentiment_data (DataFrame, optional): Données de sentiment
            save_path (str, optional): Chemin pour sauvegarder les visualisations
        """
        # Créer un dossier pour les visualisations
        if save_path:
            viz_dir = os.path.join(os.path.dirname(save_path), 'data_visualizations')
            os.makedirs(viz_dir, exist_ok=True)
        else:
            viz_dir = 'data_visualizations'
            os.makedirs(viz_dir, exist_ok=True)
        
        # Visualiser les prix
        plt.figure(figsize=(12, 6))
        plt.plot(market_data.index, market_data['close'])
        plt.title('Prix de clôture')
        plt.xlabel('Date')
        plt.ylabel('Prix ($)')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'prices.png'))
        plt.close()
        
        # Visualiser les rendements
        if 'returns' in market_data.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(market_data.index, market_data['returns'])
            plt.title('Rendements quotidiens')
            plt.xlabel('Date')
            plt.ylabel('Rendement (%)')
            plt.grid(True)
            plt.savefig(os.path.join(viz_dir, 'returns.png'))
            plt.close()
        
        # Visualiser le sentiment si disponible
        if sentiment_data is not None and not sentiment_data.empty:
            if 'compound_score' in sentiment_data.columns:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Prix sur l'axe principal
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Prix ($)', color='tab:blue')
                ax1.plot(market_data.index, market_data['close'], color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                
                # Sentiment sur l'axe secondaire
                ax2 = ax1.twinx()
                ax2.set_ylabel('Score de sentiment', color='tab:red')
                ax2.plot(sentiment_data.index, sentiment_data['compound_score'], color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                ax2.set_ylim(-1, 1)
                
                plt.title('Prix vs Sentiment')
                fig.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'price_vs_sentiment.png'))
                plt.close()
        
        logger.info(f"Visualisations sauvegardées dans {viz_dir}") 