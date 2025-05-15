"""
Exemple d'analyse de sentiment pour les actualités et médias sociaux crypto.

Ce script montre comment utiliser le module d'analyse de sentiment 
pour extraire des insights à partir de différentes sources d'information.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import des modules d'analyse de sentiment
from ai_trading.llm.sentiment_analysis.news_analyzer import NewsAnalyzer
from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer

def sentiment_analysis_example():
    """Exemple d'utilisation des analyseurs de sentiment."""
    
    # Configuration des analyseurs
    news_analyzer = NewsAnalyzer(
        api_keys={'newsapi': 'YOUR_API_KEY'},
        llm_model='advanced',
        entity_extraction=True,
        credibility_scoring=True
    )
    
    social_analyzer = SocialAnalyzer(
        platforms=['twitter', 'reddit'],
        credentials={'twitter_api_key': 'YOUR_API_KEY'},
        bot_detection=True
    )
    
    # Définition de la période d'analyse
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Récupération et analyse des actualités
    news_sentiment = news_analyzer.analyze_period(
        keywords=['bitcoin', 'ethereum', 'crypto'],
        start_date=start_date,
        end_date=end_date
    )
    
    # Récupération et analyse des médias sociaux
    social_sentiment = social_analyzer.analyze_period(
        keywords=['BTC', 'ETH', 'crypto'],
        start_date=start_date,
        end_date=end_date
    )
    
    # Combinaison des résultats
    combined_results = news_analyzer.combine_sentiment(
        news_sentiment=news_sentiment,
        social_sentiment=social_sentiment,
        weights={'news': 0.6, 'social': 0.4}
    )
    
    # Affichage des résultats
    print("== Analyse de sentiment combinée ==")
    print(f"Période: {start_date.date()} à {end_date.date()}")
    print(f"Score de sentiment global: {combined_results['global_score']:.2f}")
    print(f"Confiance: {combined_results['confidence']:.2f}")
    
    # Analyse des entités
    print("\n== Entités les plus mentionnées ==")
    entities = combined_results['entities']
    for entity, data in sorted(entities.items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
        print(f"{entity}: {data['sentiment']:.2f} (mentions: {data['count']})")
    
    # Évolution temporelle
    print("\n== Évolution du sentiment ==")
    timeline = combined_results['timeline']
    for date, score in timeline.items():
        print(f"{date}: {score:.2f}")
    
    return combined_results

if __name__ == "__main__":
    results = sentiment_analysis_example()
    
    # Pour aller plus loin: visualisation des résultats
    # import matplotlib.pyplot as plt
    # timeline = results['timeline']
    # plt.figure(figsize=(10, 6))
    # plt.plot(list(timeline.keys()), list(timeline.values()))
    # plt.title("Évolution du sentiment crypto sur 7 jours")
    # plt.xlabel("Date")
    # plt.ylabel("Score de sentiment")
    # plt.grid(True)
    # plt.savefig("sentiment_timeline.png")
    # plt.show() 