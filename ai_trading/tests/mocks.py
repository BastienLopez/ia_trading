import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Mock pour NewsAnalyzer
class MockNewsAnalyzer:
    def __init__(self):
        """Un mock pour la classe NewsAnalyzer qui évite de charger les modèles HuggingFace."""
        pass
    
    def analyze_text(self, text):
        """Retourne un résultat d'analyse de sentiment factice."""
        return {
            "sentiment": "positive",
            "score": 0.8,
            "compound_score": 0.6,
            "positive_score": 0.7,
            "negative_score": 0.1,
            "neutral_score": 0.2,
            "entities": ["crypto", "bitcoin"],
            "keywords": ["bullish", "market"]
        }
    
    def analyze_news_dataframe(self, news_df):
        """Simule l'analyse d'un DataFrame de news et retourne des scores de sentiment factices."""
        if news_df.empty:
            return pd.DataFrame()
        
        # Créer un DataFrame avec des résultats factices
        dates = pd.date_range(start='2023-01-01', periods=len(news_df), freq='D')
        result = pd.DataFrame({
            'compound_score': np.random.uniform(0, 1, len(news_df)),
            'positive_score': np.random.uniform(0, 1, len(news_df)),
            'negative_score': np.random.uniform(0, 0.3, len(news_df)),
            'neutral_score': np.random.uniform(0, 0.5, len(news_df)),
            'polarity': np.random.uniform(-1, 1, len(news_df)),
            'subjectivity': np.random.uniform(0, 1, len(news_df))
        }, index=dates)
        
        return result

# Mock pour SocialAnalyzer
class MockSocialAnalyzer:
    def __init__(self):
        """Un mock pour la classe SocialAnalyzer qui évite de charger les modèles HuggingFace."""
        pass
    
    def analyze_text(self, text):
        """Retourne un résultat d'analyse de sentiment factice."""
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "compound_score": 0.2,
            "positive_score": 0.4,
            "negative_score": 0.2,
            "neutral_score": 0.4,
            "entities": ["crypto"],
            "keywords": ["market", "trading"]
        }
    
    def analyze_tweet(self, tweet):
        """Simule l'analyse d'un tweet."""
        return self.analyze_text(tweet)
    
    def analyze_social_dataframe(self, social_df):
        """Simule l'analyse d'un DataFrame de données sociales."""
        if social_df.empty:
            return pd.DataFrame()
        
        # Créer un DataFrame avec des résultats factices
        dates = pd.date_range(start='2023-01-01', periods=len(social_df), freq='D')
        result = pd.DataFrame({
            'compound_score': np.random.uniform(-0.5, 0.5, len(social_df)),
            'positive_score': np.random.uniform(0, 0.6, len(social_df)),
            'negative_score': np.random.uniform(0, 0.4, len(social_df)),
            'neutral_score': np.random.uniform(0.2, 0.8, len(social_df)),
            'polarity': np.random.uniform(-0.8, 0.8, len(social_df)),
            'subjectivity': np.random.uniform(0.1, 0.9, len(social_df))
        }, index=dates)
        
        return result 