"""
Exemple d'utilisation du module enhanced_news_analyzer.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Exemple principal d'utilisation."""
    # Initialisation de l'analyseur
    analyzer = EnhancedNewsAnalyzer(enable_cache=True)

    # Création de données d'exemple
    news_data = [
        {
            "title": "Bitcoin Surges Past $100,000 in Historic Rally",
            "body": "Bitcoin (BTC) has reached a new all-time high of $100,000, representing a 15% increase in the last 24 hours. The surge comes as institutional investors continue to pour money into the cryptocurrency market.",
            "published_at": datetime.now().isoformat(),
        },
        {
            "title": "Ethereum Network Upgrade Successfully Implemented",
            "body": "Ethereum (ETH) has completed its latest network upgrade, resulting in a 20% reduction in gas fees. The price of ETH has increased by 10% to $5,000 following the announcement.",
            "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
        {
            "title": "Market Correction: Major Cryptocurrencies Face Pressure",
            "body": "The cryptocurrency market experienced a sharp correction today, with Bitcoin (BTC) dropping 8% to $95,000 and Ethereum (ETH) falling 12% to $4,500.",
            "published_at": (datetime.now() - timedelta(hours=4)).isoformat(),
        },
    ]

    # Analyse des actualités
    logger.info("Analyse des actualités...")
    results_df = analyzer.analyze_news(news_data)

    # Génération du rapport
    logger.info("Génération du rapport...")
    report = analyzer.generate_report(results_df)

    # Affichage des résultats
    print("\nRésultats de l'analyse :")
    print(f"Nombre total d'articles : {report['total_articles']}")
    print(f"Sentiment moyen : {report['average_sentiment']:.2f}")
    
    print("\nDistribution des sentiments :")
    for sentiment, count in report["sentiment_distribution"].items():
        print(f"- {sentiment}: {count} articles")
    
    print("\nEntités détectées :")
    for category, entities in report["entities"].items():
        print(f"\n{category.upper()} :")
        for entity, count in entities.items():
            print(f"- {entity}: {count} mentions")
    
    print("\nTendance du sentiment :")
    for date, score in report["sentiment_trend"].items():
        print(f"- {date}: {score:.2f}")

    # Génération des visualisations
    logger.info("Génération des visualisations...")
    analyzer.plot_trends(results_df)
    analyzer.plot_distribution(results_df)

    logger.info("Visualisations sauvegardées dans le dossier 'visualizations/sentiment'")


if __name__ == "__main__":
    main() 