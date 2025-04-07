"""
Analyse de sentiment pour les réseaux sociaux (Twitter, Reddit).
Combine l'analyse LLM avec des métriques sociales (engagements, viralité).
"""

import os
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import re  # Ajout de l'import manquant pour les regex
import numpy as np

from ai_trading.utils.enhanced_preprocessor import EnhancedTextDataPreprocessor
from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer
from ai_trading.llm.sentiment_analysis.sentiment_tools import SentimentCache

logger = logging.getLogger(__name__)

class SocialAnalyzer(EnhancedNewsAnalyzer):
    """Analyseur de sentiment spécialisé pour les réseaux sociaux."""
    
    def __init__(self, platform: str = 'twitter', **kwargs):
        """
        Args:
            platform: 'twitter' ou 'reddit'
            **kwargs: Arguments hérités d'EnhancedNewsAnalyzer
        """
        super().__init__(**kwargs)
        self.platform = platform
        self.text_preprocessor = EnhancedTextDataPreprocessor()
        self.engagement_metrics = self._get_platform_metrics(platform)
        
        # Configurations spécifiques aux plateformes
        self.platform_config = {
            'twitter': {
                'metrics': [
                    'retweet_count', 
                    'favorite_count', 
                    'reply_count',
                    'user.followers_count'  # Champ déplié
                ],
                'text_field': 'full_text'
            },
            'reddit': {
                'metrics': [
                    'score', 
                    'num_comments', 
                    'upvote_ratio',
                    'author.comment_karma'  # Champ déplié
                ],
                'text_field': 'selftext'
            }
        }
        
        if platform not in self.platform_config:
            raise ValueError(f"Plateforme non supportée: {platform}. Choisissez entre {list(self.platform_config.keys())}")
    
    def analyze_social_posts(self, posts: List[Dict]) -> pd.DataFrame:
        """Analyse un batch de posts sociaux."""
        logger.info(f"Analyse de {len(posts)} posts {self.platform}")
        
        # Nettoyage et prétraitement
        df = self._preprocess_social_data(posts)
        
        # Analyse de sentiment conditionnelle
        if not df.empty and 'clean_text' in df.columns:
            df = self.analyze_news_dataframe(df)
            # Renommage des colonnes de sentiment pour Reddit
            if self.platform == 'reddit':
                df = df.rename(columns={
                    'global_sentiment_label': 'sentiment_label',
                    'global_sentiment_score': 'sentiment_score'
                })
        else:
            logger.warning("Données insuffisantes pour l'analyse de sentiment")
            df['sentiment_label'] = 'neutral'
            df['sentiment_score'] = 0.0
        
        # Calcul des métriques d'engagement
        df = self._calculate_engagement_metrics(df)
        
        return df
    
    def _preprocess_social_data(self, posts: List[Dict]) -> pd.DataFrame:
        """Prétraitement spécifique aux données sociales."""
        # Ajout du dépliage des champs imbriqués
        if self.platform == 'twitter':
            posts = [{
                **post,
                'user.followers_count': post.get('user', {}).get('followers_count', 0)
            } for post in posts]
        elif self.platform == 'reddit':
            posts = [{
                **post,
                'author.comment_karma': post.get('author', {}).get('comment_karma', 0)
            } for post in posts]
        
        df = pd.json_normalize(posts)  # Déplie les structures imbriquées
        
        # Vérification des colonnes requises
        required_columns = {
            'twitter': ['full_text', 'retweet_count', 'favorite_count', 'reply_count'],
            'reddit': ['title', 'selftext', 'score', 'upvote_ratio', 'num_comments']
        }
        
        missing = [col for col in required_columns[self.platform] if col not in df.columns]
        if missing:
            logger.error(f"Colonnes manquantes pour {self.platform}: {missing}")
            return pd.DataFrame()  # Retourne un DataFrame vide au lieu de lever une exception
        
        # Prétraitement spécifique à la plateforme
        if self.platform == 'twitter':
            df['text'] = df['full_text']
        elif self.platform == 'reddit':
            df['text'] = df['title'] + ' ' + df['selftext']
        
        # Nettoyage du texte
        df['clean_text'] = df['text'].apply(
            lambda x: self.text_preprocessor.clean_text(x)
        )
        
        # Extraction des hashtags/mentions (Twitter)
        if self.platform == 'twitter':
            df['hashtags'] = df['text'].apply(
                lambda x: self._extract_hashtags(x)
            )
            df['mentions'] = df['text'].apply(
                lambda x: self._extract_mentions(x)
            )
        
        # Conversion des dates
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
    
    def _calculate_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        existing_metrics = [m for m in self.platform_config[self.platform]['metrics'] if m in df.columns]
        
        for metric in existing_metrics:
            # Remplissage des valeurs manquantes et conversion en numérique
            df[metric] = pd.to_numeric(df[metric].fillna(0), errors='coerce').fillna(0)
            
            # Calcul de la normalisation seulement si la plage est valide
            if (df[metric].max() - df[metric].min()) > 0:
                df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            else:
                df[f'{metric}_norm'] = 0.5  # Valeur neutre quand pas de variation
        
        # Calcul du score d'engagement avec vérification des colonnes
        norm_columns = [f'{m}_norm' for m in existing_metrics if f'{m}_norm' in df.columns]
        if norm_columns:
            df['engagement_score'] = df[norm_columns].mean(axis=1)
                else:
            df['engagement_score'] = 0.0
        
        return df
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extrait les hashtags d'un tweet."""
        return re.findall(r'#(\w+)', text.lower())
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extrait les mentions @ d'un tweet."""
        return re.findall(r'@(\w+)', text.lower())
    
    def generate_social_report(self, df: pd.DataFrame) -> Dict:
        """Génère un rapport complet pour les données sociales."""
        report = super().generate_sentiment_report(df)
        
        # Métriques spécifiques aux réseaux sociaux
        report.update({
            'top_hashtags': dict(Counter([h for sublist in df['hashtags'] for h in sublist]).most_common(10)),
            'engagement_stats': {
                'mean': df['engagement_score'].mean(),
                'max': df['engagement_score'].max(),
                'min': df['engagement_score'].min()
            },
            'viral_posts': self._identify_viral_posts(df)
        })
        
        return report
    
    def _identify_viral_posts(self, df: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Identifie les posts viraux basés sur le score d'engagement."""
        viral = df[df['engagement_score'] >= threshold]
        return viral.to_dict('records')
    
    def generate_engagement_plot(self, df: pd.DataFrame, filename: str) -> None:
        """Génère un graphique d'engagement temporel."""
        plt.figure(figsize=(12, 6))
        
        if 'created_at' in df.columns and 'engagement_score' in df.columns:
            df.set_index('created_at').sort_index()['engagement_score'].plot(
                title='Évolution de l\'engagement',
                ylabel='Score d\'engagement',
                xlabel='Date'
            )
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
        else:
            logger.warning("Données insuffisantes pour générer le graphique d'engagement")
    
    def _get_platform_metrics(self, platform: str) -> List[str]:
        """Retourne les métriques clés selon la plateforme."""
        return {
            'twitter': ['retweet_count', 'favorite_count', 'reply_count'],
            'reddit': ['score', 'num_comments', 'upvote_ratio']
        }[platform]
    
    def calculate_virality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule des métriques de viralité avancées."""
        return df.assign(
            engagement_score=lambda x: self._normalize_metrics(x),
            viral_risk=lambda x: self._calculate_viral_risk(x)
        )
    
    def _enhance_with_engagement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit les données avec des métriques d'engagement."""
        df['engagement'] = df.apply(
            lambda x: self._compute_composite_engagement(x), 
            axis=1
        )
        df['time_decay'] = df['published_at'].apply(
            lambda x: self._compute_time_decay(x)
        )
        return df
    
    def _compute_composite_engagement(self, row: pd.Series) -> float:
        """Calcule un score d'engagement composite."""
        weights = {
            'twitter': {'retweets': 0.4, 'likes': 0.3, 'replies': 0.3},
            'reddit': {'upvotes': 0.6, 'comments': 0.4}
        }
        return sum(
            row[metric] * weight 
            for metric, weight in weights[self.platform].items()
        )
    
    def _compute_time_decay(self, post_date: str) -> float:
        """Calcule un facteur de dépréciation temporelle."""
        delta = datetime.now() - pd.to_datetime(post_date)
        return np.exp(-delta.days / 7)  # Décroissance exponentielle sur 1 semaine
    
    def _normalize_metrics(self, df: pd.DataFrame) -> pd.Series:
        """Normalise les métriques entre 0 et 1."""
        return (
            df[self.engagement_metrics]
            .apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            .mean(axis=1)
        )
    
    def _calculate_viral_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calcule un indice de risque viral."""
        return (
            df['engagement_score'] * 
            (1 - df['time_decay']) * 
            df['sentiment_score'].abs()
        )

# Exemple d'utilisation
if __name__ == "__main__":
    # Analyse de tweets
    twitter_analyzer = SocialAnalyzer(platform='twitter')
    sample_tweets = [...]  # Données de test
    analyzed_tweets = twitter_analyzer.analyze_social_posts(sample_tweets)
    twitter_report = twitter_analyzer.generate_social_report(analyzed_tweets)
    
    # Analyse de posts Reddit
    reddit_analyzer = SocialAnalyzer(platform='reddit')
    sample_reddit = [...]  # Données de test
    analyzed_reddit = reddit_analyzer.analyze_social_posts(sample_reddit)
    reddit_report = reddit_analyzer.generate_social_report(analyzed_reddit) 