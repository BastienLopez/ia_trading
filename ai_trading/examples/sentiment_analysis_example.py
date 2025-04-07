"""
Exemple d'utilisation du module d'analyse de sentiment pour les actualités crypto.
"""

# Configuration du logging
import logging
import os

import pandas as pd

from ai_trading.llm.sentiment_analysis.news_analyzer import NewsAnalyzer
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Fonction principale démontrant l'analyse de sentiment des actualités."""

    # 1. Collecte des actualités
    logger.info("Étape 1: Collecte des actualités")
    collector = EnhancedDataCollector()

    # Liste des cryptomonnaies à analyser
    coins = ["bitcoin", "ethereum"]

    # Création du dossier de données s'il n'existe pas
    os.makedirs("data/sentiment", exist_ok=True)

    all_news = []
    for coin in coins:
        logger.info(f"Collecte des actualités pour {coin}")

        # Récupération des actualités
        news = collector.get_crypto_news(coin=coin, limit=5)
        if news:
            for item in news:
                item["coin"] = coin
            all_news.extend(news)

    if not all_news:
        logger.error("Aucune actualité récupérée")
        return

    # Conversion en DataFrame
    news_df = pd.DataFrame(all_news)
    news_df.to_csv("data/sentiment/raw_news.csv", index=False)
    logger.info(f"Actualités sauvegardées: {len(news_df)} articles")

    # 2. Analyse de sentiment
    logger.info("Étape 2: Analyse de sentiment")
    analyzer = NewsAnalyzer()

    # Analyse du DataFrame d'actualités
    enriched_df = analyzer.analyze_news_dataframe(news_df)

    # Sauvegarde des résultats
    enriched_df.to_csv("data/sentiment/analyzed_news.csv", index=False)
    logger.info(f"Analyse de sentiment terminée et sauvegardée")

    # 3. Affichage des résultats
    logger.info("Étape 3: Résultats de l'analyse")

    # Comptage des sentiments
    sentiment_counts = enriched_df["global_sentiment_label"].value_counts()
    logger.info(f"Répartition des sentiments: {sentiment_counts.to_dict()}")

    # Affichage des actualités les plus positives et négatives
    if "global_sentiment_score" in enriched_df.columns:
        most_positive = enriched_df.loc[enriched_df["global_sentiment_score"].idxmax()]
        most_negative = enriched_df.loc[enriched_df["global_sentiment_score"].idxmin()]

        logger.info("\nActualité la plus positive:")
        logger.info(f"Titre: {most_positive['title']}")
        logger.info(f"Score: {most_positive['global_sentiment_score']}")

        logger.info("\nActualité la plus négative:")
        logger.info(f"Titre: {most_negative['title']}")
        logger.info(f"Score: {most_negative['global_sentiment_score']}")

    logger.info("Analyse de sentiment terminée avec succès!")


if __name__ == "__main__":
    main()
