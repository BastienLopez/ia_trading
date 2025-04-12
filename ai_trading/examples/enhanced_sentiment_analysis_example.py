"""
Exemple avancé d'utilisation du module d'analyse de sentiment pour les actualités crypto.
Démontre les fonctionnalités avancées de l'EnhancedNewsAnalyzer.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration du logging
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import (
    EnhancedNewsAnalyzer,
)
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
from ai_trading.llm.sentiment_analysis import NewsAnalyzer, SocialAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def collect_news(coins: list, days: int = 7, limit_per_coin: int = 10) -> pd.DataFrame:
    """
    Collecte des actualités pour plusieurs cryptomonnaies.

    Args:
        coins: Liste des cryptomonnaies à analyser
        days: Nombre de jours d'historique
        limit_per_coin: Nombre maximum d'actualités par cryptomonnaie

    Returns:
        DataFrame contenant les actualités collectées
    """
    logger.info(
        f"Collecte des actualités pour {len(coins)} cryptomonnaies sur {days} jours"
    )
    collector = EnhancedDataCollector()

    all_news = []
    for coin in coins:
        logger.info(f"Collecte des actualités pour {coin}")

        # Récupération des actualités
        try:
            news = collector.get_crypto_news(coin)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des actualités pour {coin}: {e}")
            news = []
        if news:
            for item in news:
                item["coin"] = coin
            all_news.extend(news)
            logger.info(f"  - {len(news)} actualités récupérées")
        else:
            logger.warning(f"  - Aucune actualité récupérée pour {coin}")

    if not all_news:
        logger.error("Aucune actualité récupérée")
        return pd.DataFrame()

    # Conversion en DataFrame
    news_df = pd.DataFrame(all_news)

    # Filtrage par date si la colonne existe
    if "published_at" in news_df.columns:
        try:
            news_df["published_at"] = pd.to_datetime(news_df["published_at"])
            min_date = datetime.now() - timedelta(days=days)
            news_df = news_df[news_df["published_at"] >= min_date]
        except Exception as e:
            logger.error(f"Erreur lors du filtrage par date: {e}")

    logger.info(f"Total d'actualités collectées: {len(news_df)}")
    return news_df


def analyze_news(
    news_df: pd.DataFrame, use_gpu: bool = False, enable_cache: bool = True
) -> pd.DataFrame:
    """
    Analyse le sentiment des actualités.

    Args:
        news_df: DataFrame contenant les actualités
        use_gpu: Si True, utilise le GPU pour l'inférence
        enable_cache: Si True, active le cache pour éviter de réanalyser les mêmes textes

    Returns:
        DataFrame contenant les actualités enrichies avec l'analyse de sentiment
    """
    if news_df.empty:
        logger.error("Aucune actualité à analyser")
        return news_df

    logger.info(f"Analyse de sentiment pour {len(news_df)} actualités")
    analyzer = EnhancedNewsAnalyzer(use_gpu=use_gpu, enable_cache=enable_cache)

    # Analyse du DataFrame d'actualités
    enriched_df = analyzer.analyze_news_dataframe(news_df)

    # Génération du rapport
    report = analyzer.generate_sentiment_report(enriched_df)

    # Affichage du résumé
    logger.info("\nRésumé de l'analyse de sentiment:")
    logger.info(f"Total d'articles: {report['total_articles']}")
    logger.info(f"Distribution des sentiments: {report['sentiment_distribution']}")
    logger.info(f"Sentiment moyen: {report['average_sentiment']:.2f}")

    # Affichage des cryptomonnaies les plus mentionnées
    if report["crypto_mentions"]:
        logger.info("\nCryptomonnaies les plus mentionnées:")
        for crypto, count in report["crypto_mentions"].items():
            logger.info(f"- {crypto}: {count} mentions")

    # Affichage des articles les plus positifs et négatifs
    if "most_positive_article" in report and "title" in report["most_positive_article"]:
        logger.info("\nArticle le plus positif:")
        logger.info(f"- {report['most_positive_article']['title']}")
        logger.info(f"- Score: {report['most_positive_article']['score']:.2f}")

    if "most_negative_article" in report and "title" in report["most_negative_article"]:
        logger.info("\nArticle le plus négatif:")
        logger.info(f"- {report['most_negative_article']['title']}")
        logger.info(f"- Score: {report['most_negative_article']['score']:.2f}")

    # Affichage du sentiment par cryptomonnaie
    if report["sentiment_by_crypto"]:
        logger.info("\nSentiment par cryptomonnaie:")
        for crypto, data in report["sentiment_by_crypto"].items():
            logger.info(
                f"- {crypto}: {data['sentiment_label']} ({data['average_sentiment']:.2f}) - {data['article_count']} articles"
            )

    return enriched_df


def analyze_news_demo():
    analyzer = NewsAnalyzer()
    # Exemple d'analyse d'actualités
    news_data = [...]  # Données d'actualités
    results = analyzer.analyze(news_data)
    print(results)


def analyze_social_media():
    analyzer = SocialAnalyzer()
    # Exemple d'analyse de réseaux sociaux
    social_data = [...]  # Données de réseaux sociaux
    results = analyzer.analyze(social_data)
    print(results)


def main():
    """Fonction principale démontrant l'analyse de sentiment avancée des actualités."""

    # Parsing des arguments
    parser = argparse.ArgumentParser(
        description="Analyse de sentiment des actualités crypto"
    )
    parser.add_argument(
        "--coins",
        type=str,
        nargs="+",
        default=["bitcoin", "ethereum", "solana", "cardano", "ripple"],
        help="Liste des cryptomonnaies à analyser",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Nombre de jours d'historique"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Nombre maximum d'actualités par cryptomonnaie",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Utiliser le GPU pour l'inférence"
    )
    parser.add_argument("--no-cache", action="store_true", help="Désactiver le cache")
    parser.add_argument('--plot', action='store_true', help='Générer des graphiques')
    parser.add_argument('--output', type=str, default='plots/', help='Répertoire de sortie')
    args = parser.parse_args()

    # Création des dossiers de données
    os.makedirs("data/sentiment", exist_ok=True)
    os.makedirs("data/sentiment/raw", exist_ok=True)
    os.makedirs("data/sentiment/analyzed", exist_ok=True)

    # 1. Collecte des actualités
    logger.info("Étape 1: Collecte des actualités")
    news_df = collect_news(args.coins, args.days, args.limit)

    if news_df.empty:
        logger.error("Aucune actualité collectée. Arrêt du programme.")
        return

    # Sauvegarde des actualités brutes
    raw_file = f"data/sentiment/raw/news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    news_df.to_csv(raw_file, index=False)
    logger.info(f"Actualités brutes sauvegardées dans {raw_file}")

    # 2. Analyse de sentiment
    logger.info("\nÉtape 2: Analyse de sentiment")
    enriched_df = analyze_news(
        news_df, use_gpu=args.gpu, enable_cache=not args.no_cache
    )

    if enriched_df.empty:
        logger.error("Échec de l'analyse de sentiment. Arrêt du programme.")
        return

    # Sauvegarde des actualités analysées
    analyzed_file = f"data/sentiment/analyzed/news_analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    enriched_df.to_csv(analyzed_file, index=False)
    logger.info(f"Actualités analysées sauvegardées dans {analyzed_file}")

    logger.info("\nAnalyse de sentiment terminée avec succès!")
    logger.info(
        "Les visualisations ont été sauvegardées dans data/sentiment/visualizations/"
    )
    logger.info("Le rapport complet a été sauvegardé dans data/sentiment/reports/")

    analyze_news_demo()
    analyze_social_media()


if __name__ == "__main__":
    main()
