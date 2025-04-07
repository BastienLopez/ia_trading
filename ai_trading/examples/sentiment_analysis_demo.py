"""
Démonstration simple de l'utilisation du module d'analyse de sentiment amélioré.
"""

import os

from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import (
    EnhancedNewsAnalyzer,
)

# Exemple d'actualités
news_examples = [
    {
        "title": "Bitcoin Surges to $60,000 as Institutional Adoption Grows",
        "body": "Bitcoin reached a new all-time high of $60,000 today as more institutional investors are entering the cryptocurrency market. Major companies like Tesla and MicroStrategy have added BTC to their balance sheets.",
        "published_at": "2023-03-15T12:30:00Z",
    },
    {
        "title": "Ethereum Price Drops 10% Following Network Congestion",
        "body": "Ethereum (ETH) experienced a significant price drop of 10% in the last 24 hours due to network congestion and high gas fees. Developers are working on solutions to address these scaling issues.",
        "published_at": "2023-03-14T09:15:00Z",
    },
    {
        "title": "Solana Ecosystem Expands with New DeFi Projects",
        "body": "The Solana blockchain is seeing rapid growth in its DeFi ecosystem with several new projects launching this month. The total value locked (TVL) in Solana DeFi has increased by 25% in the past week.",
        "published_at": "2023-03-13T16:45:00Z",
    },
]


def main():
    # Création des dossiers nécessaires
    os.makedirs("data/sentiment/visualizations", exist_ok=True)
    os.makedirs("data/sentiment/reports", exist_ok=True)

    # Initialisation de l'analyseur
    analyzer = EnhancedNewsAnalyzer()

    # Analyse des actualités
    print("Analyse des actualités...")
    enriched_news = analyzer.analyze_news(news_examples)

    # Affichage des résultats
    for i, news in enumerate(enriched_news):
        print(f"\nActualité {i+1}: {news['title']}")
        print(
            f"Sentiment du titre: {news['title_sentiment']['label']} ({news['title_sentiment']['score']:.2f})"
        )
        print(
            f"Sentiment global: {news['global_sentiment']['label']} ({news['global_sentiment']['score']:.2f})"
        )

        if news["mentioned_cryptos"]:
            print(f"Cryptomonnaies mentionnées: {', '.join(news['mentioned_cryptos'])}")

    print(
        "\nLes visualisations ont été sauvegardées dans le dossier data/sentiment/visualizations/"
    )


if __name__ == "__main__":
    main()
