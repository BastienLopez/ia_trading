"""
Module d'analyse des actualités crypto.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .sentiment_model import SentimentAnalyzer

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """Classe pour l'analyse des actualités crypto."""
    
    def __init__(self):
        """Initialisation de l'analyseur d'actualités."""
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def analyze_news(
        self,
        news_data: pd.DataFrame,
        detailed: bool = False
    ) -> List[Dict]:
        """
        Analyse les actualités et leur sentiment.
        
        Args:
            news_data: DataFrame contenant les actualités
            detailed: Si True, effectue une analyse détaillée
            
        Returns:
            Liste des analyses d'actualités
        """
        analyzed_news = []
        
        for _, news in news_data.iterrows():
            try:
                # Combinaison du titre et de la description pour l'analyse
                text = f"{news['title']} {news['description']}"
                
                # Analyse du sentiment
                if detailed:
                    sentiment = self.sentiment_analyzer.analyze_detailed(text)
                else:
                    sentiment = self.sentiment_analyzer.analyze_quick(text)[0]
                
                # Création de l'entrée analysée
                analyzed_entry = {
                    'title': news['title'],
                    'description': news['description'],
                    'source': news['source'],
                    'published_at': news['published_at'],
                    'sentiment': sentiment
                }
                
                analyzed_news.append(analyzed_entry)
                
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de l'actualité: {e}")
                continue
        
        return analyzed_news
    
    def get_sentiment_timeline(
        self,
        analyzed_news: List[Dict],
        interval: str = '1H'
    ) -> pd.DataFrame:
        """
        Crée une timeline des sentiments.
        
        Args:
            analyzed_news: Liste des actualités analysées
            interval: Intervalle de temps pour l'agrégation
            
        Returns:
            DataFrame avec les sentiments agrégés par intervalle
        """
        # Création d'un DataFrame avec les données analysées
        df = pd.DataFrame(analyzed_news)
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # Extraction des scores de sentiment
        df['sentiment_score'] = df['sentiment'].apply(
            lambda x: x.get('normalized_score', 0.0)
            if isinstance(x, dict) else 0.0
        )
        
        # Agrégation par intervalle
        timeline = df.set_index('published_at').resample(interval).agg({
            'sentiment_score': 'mean',
            'title': 'count'
        }).fillna(0)
        
        timeline.columns = ['average_sentiment', 'news_count']
        
        return timeline
    
    def get_top_news(
        self,
        analyzed_news: List[Dict],
        n: int = 5,
        by: str = 'impact'
    ) -> List[Dict]:
        """
        Récupère les actualités les plus importantes.
        
        Args:
            analyzed_news: Liste des actualités analysées
            n: Nombre d'actualités à retourner
            by: Critère de tri ('impact' ou 'sentiment')
            
        Returns:
            Liste des actualités les plus importantes
        """
        # Conversion en DataFrame pour faciliter le tri
        df = pd.DataFrame(analyzed_news)
        
        if by == 'impact':
            # Tri par impact (combinaison de sentiment et confiance)
            df['impact_score'] = df['sentiment'].apply(
                lambda x: abs(x.get('normalized_score', 0.0)) * x.get('confidence', 0.0)
                if isinstance(x, dict) else 0.0
            )
            sorted_df = df.nlargest(n, 'impact_score')
        else:
            # Tri par sentiment absolu
            df['abs_sentiment'] = df['sentiment'].apply(
                lambda x: abs(x.get('normalized_score', 0.0))
                if isinstance(x, dict) else 0.0
            )
            sorted_df = df.nlargest(n, 'abs_sentiment')
        
        return sorted_df.to_dict('records')
    
    def analyze_market_impact(
        self,
        analyzed_news: List[Dict],
        price_data: pd.DataFrame,
        window: str = '1H'
    ) -> List[Tuple[Dict, float]]:
        """
        Analyse l'impact des actualités sur le prix.
        
        Args:
            analyzed_news: Liste des actualités analysées
            price_data: DataFrame avec les données de prix
            window: Fenêtre de temps pour analyser l'impact
            
        Returns:
            Liste de tuples (actualité, impact sur le prix)
        """
        impact_analysis = []
        
        for news in analyzed_news:
            try:
                # Récupération de la date de publication
                pub_date = pd.to_datetime(news['published_at'])
                
                # Calcul du prix avant et après la publication
                pre_window = price_data.loc[
                    (price_data.index >= pub_date - pd.Timedelta(window)) &
                    (price_data.index < pub_date)
                ]['close']
                
                post_window = price_data.loc[
                    (price_data.index > pub_date) &
                    (price_data.index <= pub_date + pd.Timedelta(window))
                ]['close']
                
                if not pre_window.empty and not post_window.empty:
                    # Calcul du changement de prix en pourcentage
                    price_impact = (
                        (post_window.iloc[-1] - pre_window.iloc[0]) /
                        pre_window.iloc[0] * 100
                    )
                    
                    impact_analysis.append((news, price_impact))
                
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de l'impact: {e}")
                continue
        
        return impact_analysis
    
    def get_market_summary(
        self,
        analyzed_news: List[Dict],
        timeframe: str = '24H'
    ) -> Dict:
        """
        Génère un résumé du marché basé sur les actualités.
        
        Args:
            analyzed_news: Liste des actualités analysées
            timeframe: Période de temps à analyser
            
        Returns:
            Dictionnaire contenant le résumé du marché
        """
        # Filtrage des actualités par timeframe
        cutoff = datetime.now() - pd.Timedelta(timeframe)
        recent_news = [
            news for news in analyzed_news
            if pd.to_datetime(news['published_at']) > cutoff
        ]
        
        # Agrégation des sentiments
        sentiments = [
            news['sentiment']
            for news in recent_news
            if isinstance(news.get('sentiment'), dict)
        ]
        
        aggregated_sentiment = self.sentiment_analyzer.aggregate_sentiment(sentiments)
        
        # Comptage des actualités par sentiment
        sentiment_counts = {
            'positive': sum(1 for s in sentiments if s.get('sentiment') == 'positive'),
            'negative': sum(1 for s in sentiments if s.get('sentiment') == 'negative'),
            'neutral': sum(1 for s in sentiments if s.get('sentiment') == 'neutral')
        }
        
        return {
            'timeframe': timeframe,
            'total_news': len(recent_news),
            'sentiment_distribution': sentiment_counts,
            'overall_sentiment': aggregated_sentiment,
            'top_news': self.get_top_news(recent_news, n=3)
        } 