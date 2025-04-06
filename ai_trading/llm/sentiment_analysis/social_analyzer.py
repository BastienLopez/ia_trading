"""
Module d'analyse des réseaux sociaux pour le trading crypto.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .sentiment_model import SentimentAnalyzer

logger = logging.getLogger(__name__)

class SocialAnalyzer:
    """Classe pour l'analyse des données des réseaux sociaux."""
    
    def __init__(self):
        """Initialisation de l'analyseur de réseaux sociaux."""
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def analyze_posts(
        self,
        social_data: pd.DataFrame,
        detailed: bool = False
    ) -> List[Dict]:
        """
        Analyse les posts des réseaux sociaux.
        
        Args:
            social_data: DataFrame contenant les posts
            detailed: Si True, effectue une analyse détaillée
            
        Returns:
            Liste des analyses de posts
        """
        analyzed_posts = []
        
        for _, post in social_data.iterrows():
            try:
                # Analyse du sentiment
                if detailed:
                    sentiment = self.sentiment_analyzer.analyze_detailed(post['text'])
                else:
                    sentiment = self.sentiment_analyzer.analyze_quick(post['text'])[0]
                
                # Calcul du score d'engagement
                engagement_score = self._calculate_engagement_score(
                    post['retweets'],
                    post['likes']
                )
                
                # Création de l'entrée analysée
                analyzed_entry = {
                    'text': post['text'],
                    'user': post['user'],
                    'created_at': post['created_at'],
                    'engagement_score': engagement_score,
                    'sentiment': sentiment,
                    'metrics': {
                        'retweets': post['retweets'],
                        'likes': post['likes']
                    }
                }
                
                analyzed_posts.append(analyzed_entry)
                
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse du post: {e}")
                continue
        
        return analyzed_posts
    
    def _calculate_engagement_score(
        self,
        retweets: int,
        likes: int,
        retweet_weight: float = 1.5,
        like_weight: float = 1.0
    ) -> float:
        """
        Calcule le score d'engagement pour un post.
        
        Args:
            retweets: Nombre de retweets
            likes: Nombre de likes
            retweet_weight: Poids des retweets
            like_weight: Poids des likes
            
        Returns:
            Score d'engagement normalisé
        """
        return (retweet_weight * retweets + like_weight * likes) / (retweet_weight + like_weight)
    
    def get_sentiment_timeline(
        self,
        analyzed_posts: List[Dict],
        interval: str = '1H',
        weighted_by_engagement: bool = True
    ) -> pd.DataFrame:
        """
        Crée une timeline des sentiments.
        
        Args:
            analyzed_posts: Liste des posts analysés
            interval: Intervalle de temps pour l'agrégation
            weighted_by_engagement: Si True, pondère les sentiments par l'engagement
            
        Returns:
            DataFrame avec les sentiments agrégés par intervalle
        """
        # Création d'un DataFrame avec les données analysées
        df = pd.DataFrame(analyzed_posts)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Extraction des scores de sentiment
        df['sentiment_score'] = df['sentiment'].apply(
            lambda x: x.get('normalized_score', 0.0)
            if isinstance(x, dict) else 0.0
        )
        
        if weighted_by_engagement:
            # Pondération des sentiments par l'engagement
            df['weighted_sentiment'] = df['sentiment_score'] * df['engagement_score']
            df['total_engagement'] = df['engagement_score']
            
            # Agrégation par intervalle
            timeline = df.set_index('created_at').resample(interval).agg({
                'weighted_sentiment': 'sum',
                'total_engagement': 'sum',
                'text': 'count'
            }).fillna(0)
            
            # Calcul du sentiment moyen pondéré
            timeline['average_sentiment'] = (
                timeline['weighted_sentiment'] /
                timeline['total_engagement'].replace(0, 1)
            )
            timeline['post_count'] = timeline['text']
            timeline = timeline[['average_sentiment', 'post_count', 'total_engagement']]
            
        else:
            # Agrégation simple par intervalle
            timeline = df.set_index('created_at').resample(interval).agg({
                'sentiment_score': 'mean',
                'text': 'count',
                'engagement_score': 'sum'
            }).fillna(0)
            
            timeline.columns = ['average_sentiment', 'post_count', 'total_engagement']
        
        return timeline
    
    def get_trending_topics(
        self,
        analyzed_posts: List[Dict],
        n_topics: int = 5,
        min_occurrences: int = 3
    ) -> List[Dict]:
        """
        Identifie les sujets tendance dans les posts.
        
        Args:
            analyzed_posts: Liste des posts analysés
            n_topics: Nombre de sujets à retourner
            min_occurrences: Nombre minimum d'occurrences pour un sujet
            
        Returns:
            Liste des sujets tendance avec leurs métriques
        """
        # Extraction des hashtags et mentions
        topics = []
        for post in analyzed_posts:
            text = post['text'].lower()
            # Extraction des hashtags
            hashtags = [
                word for word in text.split()
                if word.startswith('#')
            ]
            # Extraction des mentions
            mentions = [
                word for word in text.split()
                if word.startswith('@')
            ]
            
            topics.extend(hashtags + mentions)
        
        # Comptage des occurrences
        topic_counts = pd.Series(topics).value_counts()
        
        # Filtrage des sujets avec suffisamment d'occurrences
        trending_topics = []
        for topic, count in topic_counts.items():
            if count >= min_occurrences:
                # Calcul du sentiment moyen pour ce sujet
                topic_posts = [
                    post for post in analyzed_posts
                    if topic.lower() in post['text'].lower()
                ]
                
                topic_sentiments = [
                    post['sentiment'].get('normalized_score', 0.0)
                    for post in topic_posts
                    if isinstance(post.get('sentiment'), dict)
                ]
                
                if topic_sentiments:
                    avg_sentiment = sum(topic_sentiments) / len(topic_sentiments)
                else:
                    avg_sentiment = 0.0
                
                trending_topics.append({
                    'topic': topic,
                    'occurrences': count,
                    'average_sentiment': avg_sentiment,
                    'example_posts': topic_posts[:3]  # Quelques exemples de posts
                })
                
                if len(trending_topics) >= n_topics:
                    break
        
        return trending_topics
    
    def get_influencer_analysis(
        self,
        analyzed_posts: List[Dict],
        min_posts: int = 2
    ) -> List[Dict]:
        """
        Analyse les utilisateurs les plus influents.
        
        Args:
            analyzed_posts: Liste des posts analysés
            min_posts: Nombre minimum de posts pour être considéré
            
        Returns:
            Liste des utilisateurs influents avec leurs métriques
        """
        # Groupement par utilisateur
        user_posts = {}
        for post in analyzed_posts:
            user = post['user']
            if user not in user_posts:
                user_posts[user] = []
            user_posts[user].append(post)
        
        # Analyse des métriques par utilisateur
        influencers = []
        for user, posts in user_posts.items():
            if len(posts) >= min_posts:
                # Calcul des métriques
                total_engagement = sum(post['engagement_score'] for post in posts)
                avg_engagement = total_engagement / len(posts)
                
                sentiments = [
                    post['sentiment'].get('normalized_score', 0.0)
                    for post in posts
                    if isinstance(post.get('sentiment'), dict)
                ]
                
                avg_sentiment = (
                    sum(sentiments) / len(sentiments)
                    if sentiments else 0.0
                )
                
                influencers.append({
                    'user': user,
                    'post_count': len(posts),
                    'total_engagement': total_engagement,
                    'average_engagement': avg_engagement,
                    'average_sentiment': avg_sentiment,
                    'recent_posts': sorted(
                        posts,
                        key=lambda x: x['created_at'],
                        reverse=True
                    )[:3]  # 3 posts les plus récents
                })
        
        # Tri par engagement total
        influencers.sort(key=lambda x: x['total_engagement'], reverse=True)
        
        return influencers
    
    def get_social_summary(
        self,
        analyzed_posts: List[Dict],
        timeframe: str = '24H'
    ) -> Dict:
        """
        Génère un résumé de l'activité sociale.
        
        Args:
            analyzed_posts: Liste des posts analysés
            timeframe: Période de temps à analyser
            
        Returns:
            Dictionnaire contenant le résumé de l'activité
        """
        # Filtrage des posts par timeframe
        cutoff = datetime.now() - pd.Timedelta(timeframe)
        recent_posts = [
            post for post in analyzed_posts
            if pd.to_datetime(post['created_at']) > cutoff
        ]
        
        # Agrégation des sentiments
        sentiments = [
            post['sentiment']
            for post in recent_posts
            if isinstance(post.get('sentiment'), dict)
        ]
        
        aggregated_sentiment = self.sentiment_analyzer.aggregate_sentiment(sentiments)
        
        # Calcul des métriques globales
        total_engagement = sum(post['engagement_score'] for post in recent_posts)
        avg_engagement = total_engagement / len(recent_posts) if recent_posts else 0
        
        # Identification des sujets tendance
        trending = self.get_trending_topics(recent_posts, n_topics=3)
        
        # Identification des influenceurs principaux
        top_influencers = self.get_influencer_analysis(recent_posts)[:3]
        
        return {
            'timeframe': timeframe,
            'total_posts': len(recent_posts),
            'total_engagement': total_engagement,
            'average_engagement': avg_engagement,
            'sentiment_summary': aggregated_sentiment,
            'trending_topics': trending,
            'top_influencers': top_influencers
        } 