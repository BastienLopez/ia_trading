"""
Exemple avancé d'utilisation du module d'analyse de sentiment pour les actualités crypto.
Démontre les fonctionnalités avancées de l'EnhancedNewsAnalyzer.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configuration du logging
import logging
from datetime import datetime, timedelta

import pandas as pd

from ai_trading.llm.sentiment_analysis import NewsAnalyzer, SocialAnalyzer
from ai_trading.llm.sentiment_analysis.enhanced_news_analyzer import (
    EnhancedNewsAnalyzer,
)
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector

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
            news = collector.get_crypto_news(limit=limit_per_coin)

            # Ajout manuel du coin à chaque actualité
            if news:
                for item in news:
                    item["coin"] = coin
                all_news.extend(news)
                logger.info(f"  - {len(news)} actualités récupérées")
            else:
                logger.warning(f"  - Aucune actualité récupérée pour {coin}")
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des actualités pour {coin}: {e}"
            )
            news = []

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
    # Vérifier les noms des colonnes dans notre DataFrame
    if "content" in news_df.columns and "title" not in news_df.columns:
        # Renommer content en body si nécessaire
        news_df = news_df.rename(columns={"content": "body"})

    if "body" not in news_df.columns and "content" not in news_df.columns:
        # Si ni body ni content ne sont présents, utiliser une colonne vide
        news_df["body"] = ""

    # Utiliser la méthode analyze_news_dataframe qui est conçue pour les DataFrames
    enriched_df = analyzer.analyze_news_dataframe(news_df)

    # Génération du rapport
    report = analyzer.generate_report(enriched_df)

    # Affichage du résumé
    logger.info("\nRésumé de l'analyse de sentiment:")
    logger.info(f"Total d'articles: {report['total_articles']}")
    logger.info(f"Distribution des sentiments: {report['sentiment_distribution']}")
    logger.info(f"Sentiment moyen: {report['average_sentiment']:.2f}")

    # Affichage des cryptomonnaies les plus mentionnées
    if "crypto_mentions" in report and report["crypto_mentions"]:
        logger.info("\nCryptomonnaies les plus mentionnées:")
        for crypto, count in report["crypto_mentions"].items():
            logger.info(f"- {crypto}: {count} mentions")

    # Affichage des articles les plus positifs et négatifs
    if (
        "most_positive_article" in report
        and report["most_positive_article"]
        and "title" in report["most_positive_article"]
    ):
        logger.info("\nArticle le plus positif:")
        logger.info(f"- {report['most_positive_article']['title']}")
        score_key = (
            "score" if "score" in report["most_positive_article"] else "sentiment_score"
        )
        if score_key in report["most_positive_article"]:
            logger.info(f"- Score: {report['most_positive_article'][score_key]:.2f}")

    if (
        "most_negative_article" in report
        and report["most_negative_article"]
        and "title" in report["most_negative_article"]
    ):
        logger.info("\nArticle le plus négatif:")
        logger.info(f"- {report['most_negative_article']['title']}")
        score_key = (
            "score" if "score" in report["most_negative_article"] else "sentiment_score"
        )
        if score_key in report["most_negative_article"]:
            logger.info(f"- Score: {report['most_negative_article'][score_key]:.2f}")

    # Affichage du sentiment par cryptomonnaie
    if "sentiment_by_crypto" in report and report["sentiment_by_crypto"]:
        logger.info("\nSentiment par cryptomonnaie:")
        for crypto, data in report["sentiment_by_crypto"].items():
            logger.info(
                f"- {crypto}: {data['sentiment_label']} ({data['average_sentiment']:.2f}) - {data['article_count']} articles"
            )

    return enriched_df


def analyze_news_demo():
    analyzer = NewsAnalyzer()
    # Exemple d'analyse d'actualités
    news_data = [
        {
            "title": "Bitcoin surpasses $65,000 as institutional adoption increases",
            "body": "Bitcoin has reached a new all-time high, exceeding $65,000 per coin. This surge is attributed to increasing institutional adoption and growing interest from traditional finance.",
        },
        {
            "title": "Ethereum upgrade improves network scalability",
            "body": "The recent Ethereum network upgrade has significantly improved transaction throughput and reduced gas fees, addressing long-standing scalability concerns.",
        },
    ]
    results = analyzer.analyze_news(news_data)
    print(results)


def analyze_social_media():
    analyzer = SocialAnalyzer(platform="twitter")
    # Exemple d'analyse de réseaux sociaux pour Twitter
    social_data = [
        {
            "full_text": "Le Bitcoin est en train de révolutionner le système financier mondial! #BTC #ToTheMoon",
            "retweet_count": 150,
            "favorite_count": 300,
            "reply_count": 45,
            "created_at": "2023-09-15T14:30:00Z",
            "user": {
                "screen_name": "crypto_enthusiast",
                "followers_count": 15000,
                "verified": True,
            },
        },
        {
            "full_text": "Je suis inquiet de la volatilité récente des marchés crypto. Que pensez-vous de la correction actuelle?",
            "retweet_count": 50,
            "favorite_count": 100,
            "reply_count": 25,
            "created_at": "2023-09-16T09:15:00Z",
            "user": {
                "screen_name": "investor_123",
                "followers_count": 5000,
                "verified": False,
            },
        },
    ]
    results = analyzer.analyze_social_posts(social_data)
    print(results)


def enhanced_analyze_news_demo():
    analyzer = EnhancedNewsAnalyzer()
    # Exemple d'analyse d'actualités avec l'analyseur amélioré
    news_data = [
        {
            "title": "Bitcoin atteint un nouveau sommet historique",
            "body": "Le prix du Bitcoin a atteint un nouveau sommet historique aujourd'hui, dépassant les 70 000 dollars.",
            "published_at": "2023-09-15T14:30:00Z",
        },
        {
            "title": "Ethereum 2.0 améliore considérablement l'évolutivité",
            "body": "La mise à jour d'Ethereum 2.0 a permis d'améliorer l'évolutivité et de réduire les frais de transaction.",
            "published_at": "2023-09-16T09:15:00Z",
        },
    ]
    results = analyzer.analyze_news(news_data)
    print("Résultats de l'analyse améliorée:")
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Générer un graphique de visualisation des sentiments",
    )
    args = parser.parse_args()

    # Création des dossiers de données
    os.makedirs(os.path.join(os.path.dirname(__file__), "info_retour/data/sentiment"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "info_retour/data/sentiment/raw"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "info_retour/data/sentiment/analyzed"), exist_ok=True)

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

    # Génération d'un graphique si demandé
    if args.plot:
        logger.info("\nGénération du graphique de visualisation des sentiments")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Visualisation des sentiments par crypto-monnaie
            try:
                if enriched_df is not None:
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x="coin", y="sentiment_score", data=enriched_df)
                    plt.title("Distribution des scores de sentiment par cryptomonnaie")
                    plt.xlabel("Cryptomonnaie")
                    plt.ylabel("Score de sentiment")
                    plt.xticks(rotation=45)

                    # Créer le répertoire visualizations/sentiment s'il n'existe pas
                    visualization_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "visualizations",
                        "sentiment",
                    )
                    if not os.path.exists(visualization_dir):
                        os.makedirs(visualization_dir)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    viz_path = os.path.join(
                        visualization_dir, f"sentiment_viz_{timestamp}.png"
                    )
                    plt.savefig(viz_path)
                    plt.close()

                    logger.info(f"Visualisation de sentiment enregistrée: {viz_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la génération du graphique : {e}")

        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique : {e}")

    logger.info("\nAnalyse de sentiment terminée avec succès!")
    logger.info("Le rapport complet a été sauvegardé dans data/sentiment/reports/")

    analyze_news_demo()
    analyze_social_media()
    enhanced_analyze_news_demo()


if __name__ == "__main__":
    main()
