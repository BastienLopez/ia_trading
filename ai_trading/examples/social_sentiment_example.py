"""
Exemple d'analyse de sentiment pour les réseaux sociaux.
Montre comment analyser des tweets et des posts Reddit.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer


def main():
    # 1. Analyse de tweets
    print("Analyse de tweets...")
    twitter_analyzer = SocialAnalyzer(platform="twitter")

    # Exemple de tweets (remplacer par des données réelles)
    tweets = [
        {
            "full_text": "Bitcoin to the moon! 🚀 #BTC #crypto",
            "retweet_count": 1500,
            "favorite_count": 4500,
            "created_at": datetime.now().isoformat(),
        },
        {
            "full_text": "Ethereum merge incoming! Bullish on $ETH @VitalikButerin",
            "retweet_count": 890,
            "favorite_count": 2300,
            "created_at": datetime.now().isoformat(),
        },
    ]

    analyzed_tweets = twitter_analyzer.analyze_social_posts(tweets)
    analyzed_tweets.to_csv("data/social/twitter_analyzed.csv", index=False)

    # 2. Analyse de posts Reddit
    print("\nAnalyse de posts Reddit...")
    reddit_analyzer = SocialAnalyzer(platform="reddit")

    # Exemple de posts Reddit
    reddit_posts = [
        {
            "body": "Solana network outage again... $SOL losing credibility 😞",
            "score": 850,
            "num_comments": 300,
            "created_utc": datetime.now().timestamp(),
        }
    ]

    analyzed_reddit = reddit_analyzer.analyze_social_posts(reddit_posts)
    analyzed_reddit.to_csv("data/social/reddit_analyzed.csv", index=False)

    # 3. Génération de visualisations
    print("\nGénération des visualisations...")
    os.makedirs("data/social/visualizations", exist_ok=True)

    # Graphique d'engagement Twitter
    twitter_analyzer.generate_engagement_plot(
        analyzed_tweets, "data/social/visualizations/twitter_engagement.png"
    )

    # Rapport complet
    twitter_report = twitter_analyzer.generate_social_report(analyzed_tweets)
    print("\nTop hashtags Twitter:", twitter_report["top_hashtags"])


if __name__ == "__main__":
    main()
