"""
Exemple d'utilisation du pipeline de données amélioré.
Ce script montre comment collecter, prétraiter et analyser les données de cryptomonnaies.
"""

# Configuration du logging
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
from ai_trading.utils.enhanced_preprocessor import (
    EnhancedMarketDataPreprocessor,
    EnhancedTextDataPreprocessor,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Fonction principale démontrant le pipeline de données complet."""

    # 1. Collecte des données
    logger.info("Étape 1: Collecte des données")
    collector = EnhancedDataCollector()

    # Liste des cryptomonnaies à analyser
    coins = ["bitcoin", "ethereum", "solana"]
    days = 30

    # Création du dossier de données s'il n'existe pas
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "info_retour/data/raw"), exist_ok=True
    )
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "info_retour/data/processed"),
        exist_ok=True,
    )

    for coin in coins:
        logger.info(f"Collecte des données pour {coin}")

        # Récupération des données de prix
        price_data = collector.get_merged_price_data(
            coin_id=coin,
            days=days,
            include_fear_greed=True,
            mock_data=False,  # Mettre à True si les APIs sont indisponibles
        )

        if not price_data.empty:
            # Sauvegarde des données brutes
            collector.save_data(
                price_data, f"ai_trading/info_retour/data/raw/{coin}_prices_{days}d.csv"
            )
            logger.info(f"Données de prix sauvegardées pour {coin}")
        else:
            logger.warning(f"Aucune donnée de prix disponible pour {coin}")

        # Récupération des actualités
        news = collector.get_crypto_news(limit=10)
        if news:
            news_df = pd.DataFrame(news)
            news_df.to_csv(
                f"ai_trading/info_retour/data/raw/{coin}_news.csv", index=False
            )
            logger.info(f"Actualités sauvegardées pour {coin}")

    # 2. Prétraitement des données
    logger.info("Étape 2: Prétraitement des données")
    market_preprocessor = EnhancedMarketDataPreprocessor(scaling_method="minmax")
    text_preprocessor = EnhancedTextDataPreprocessor()

    for coin in coins:
        price_file = f"ai_trading/info_retour/data/raw/{coin}_prices_{days}d.csv"
        if os.path.exists(price_file):
            logger.info(f"Prétraitement des données de prix pour {coin}")

            # Prétraitement complet des données de marché
            processed_data = market_preprocessor.preprocess_market_data(
                price_file,
                create_target=True,
                target_method="direction",
                target_horizon=1,
            )

            if processed_data is not None:
                # Sauvegarde des données prétraitées
                processed_data.to_csv(
                    f"ai_trading/info_retour/data/processed/{coin}_processed.csv"
                )
                logger.info(f"Données prétraitées sauvegardées pour {coin}")

                # Division des données pour l'entraînement
                train_df, val_df, test_df = market_preprocessor.split_data(
                    processed_data
                )
                train_df.to_csv(
                    f"ai_trading/info_retour/data/processed/{coin}_train.csv"
                )
                val_df.to_csv(f"ai_trading/info_retour/data/processed/{coin}_val.csv")
                test_df.to_csv(f"ai_trading/info_retour/data/processed/{coin}_test.csv")
                logger.info(f"Données divisées pour {coin}")

        # Prétraitement des actualités
        news_file = f"ai_trading/info_retour/data/raw/{coin}_news.csv"
        if os.path.exists(news_file):
            logger.info(f"Prétraitement des actualités pour {coin}")
            news_df = pd.read_csv(news_file)

            # Nettoyage et tokenization des titres
            if "title" in news_df.columns:
                news_df["cleaned_title"] = news_df["title"].apply(
                    text_preprocessor.clean_text
                )
                news_df["tokens"] = news_df["cleaned_title"].apply(
                    text_preprocessor.tokenize_text
                )

                # Sauvegarde des actualités prétraitées
                news_df.to_csv(
                    f"ai_trading/info_retour/data/processed/{coin}_news_processed.csv",
                    index=False,
                )
                logger.info(f"Actualités prétraitées sauvegardées pour {coin}")

    # 3. Visualisation des données (exemple simple)
    logger.info("Étape 3: Visualisation des données")
    for coin in coins:
        processed_file = f"ai_trading/info_retour/data/processed/{coin}_processed.csv"
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

            # Créer un graphique simple et le sauvegarder
            plt.figure(figsize=(12, 6))
            plt.plot(df["close"], label="Prix de clôture")
            plt.title(f"Évolution du prix de {coin} sur les {days} derniers jours")
            plt.xlabel("Date")
            plt.ylabel("Prix ($)")
            plt.legend()
            plt.grid(True)

            # Créer le dossier de visualisation s'il n'existe pas
            visualization_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "info_retour",
                "visualisations",
                "misc",
            )
            os.makedirs(visualization_dir, exist_ok=True)

            plt.savefig(os.path.join(visualization_dir, f"{coin}_price_chart.png"))
            plt.close()
            logger.info(
                f"Graphique créé pour {coin} et enregistré dans {visualization_dir}"
            )

    logger.info("Pipeline de données terminé avec succès!")


if __name__ == "__main__":
    main()
