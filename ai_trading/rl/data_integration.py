import logging
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ai_trading.llm.sentiment_analysis.news_analyzer import NewsAnalyzer
from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer
from ai_trading.rl.data_processor import prepare_data_for_rl
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
from ai_trading.utils.enhanced_preprocessor import EnhancedMarketDataPreprocessor

# Configuration du logger
logger = logging.getLogger("DataIntegration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RLDataIntegrator:
    """
    Classe pour intégrer les données de marché et de sentiment pour l'apprentissage par renforcement.
    """

    def __init__(self, config=None):
        """
        Initialise l'intégrateur de données.

        Args:
            config (dict, optional): Configuration pour la collecte et le prétraitement des données
        """
        self.config = config or {}
        self.data_collector = EnhancedDataCollector()
        self.data_preprocessor = EnhancedMarketDataPreprocessor()
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialAnalyzer()

        logger.info("Intégrateur de données RL initialisé")

    def collect_market_data(self, symbol, start_date, end_date, interval="1d"):
        """
        Collecte les données de marché pour une cryptomonnaie.

        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: 'BTC')
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')
            interval (str): Intervalle de temps ('1d', '1h', etc.)

        Returns:
            DataFrame: Données de marché
        """
        logger.info(
            f"Collecte des données de marché pour {symbol} du {start_date} au {end_date}"
        )

        try:
            # Utiliser le collecteur de données existant
            market_data = self.data_collector.get_merged_price_data(
                coin_id=symbol.lower(),
                days=self._calculate_days(start_date, end_date),
                vs_currency="usd",
            )

            logger.info(f"Données de marché collectées: {len(market_data)} points")
            return market_data

        except Exception as e:
            logger.error(f"Erreur lors de la collecte des données de marché: {str(e)}")
            # Créer des données synthétiques en cas d'erreur
            return self._generate_synthetic_market_data(start_date, end_date, interval)

    def collect_sentiment_data(self, symbol, start_date, end_date):
        """
        Collecte et analyse les données de sentiment pour une cryptomonnaie.

        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: 'BTC')
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')

        Returns:
            DataFrame: Données de sentiment
        """
        logger.info(
            f"Collecte des données de sentiment pour {symbol} du {start_date} au {end_date}"
        )

        try:
            # Collecter les actualités
            news_data = self.data_collector.get_crypto_news(limit=20)

            # Analyser le sentiment des actualités
            if news_data:
                news_df = pd.DataFrame(news_data)
                news_sentiment = self.news_analyzer.analyze_news_dataframe(news_df)
                logger.info(
                    f"Sentiment des actualités analysé: {len(news_sentiment)} points"
                )
            else:
                news_sentiment = pd.DataFrame()
                logger.warning("Aucune actualité collectée")

            # Pour le moment, ne pas collecter les données sociales car la méthode n'existe pas
            social_sentiment = pd.DataFrame()

            # Fusionner les sentiments des actualités et des données sociales
            sentiment_data = self._merge_sentiment_data(
                news_sentiment, social_sentiment
            )

            if sentiment_data.empty:
                logger.warning(
                    "Aucune donnée de sentiment disponible, génération de données synthétiques"
                )
                sentiment_data = self._generate_synthetic_sentiment_data(
                    start_date, end_date
                )

            return sentiment_data

        except Exception as e:
            logger.error(
                f"Erreur lors de la collecte des données de sentiment: {str(e)}"
            )
            # Créer des données synthétiques en cas d'erreur
            return self._generate_synthetic_sentiment_data(start_date, end_date)

    def preprocess_market_data(self, market_data):
        """
        Prétraite les données de marché.

        Args:
            market_data (DataFrame): Données de marché brutes

        Returns:
            DataFrame: Données de marché prétraitées
        """
        logger.info("Prétraitement des données de marché")

        try:
            # Utiliser le préprocesseur existant
            preprocessed_data = self.data_preprocessor.preprocess_market_data(market_data)

            # Ajouter des indicateurs techniques supplémentaires si nécessaire
            if "rsi" not in preprocessed_data.columns:
                preprocessed_data = self.data_preprocessor.add_technical_indicators(
                    preprocessed_data
                )

            logger.info(
                f"Données de marché prétraitées: {len(preprocessed_data)} points avec {len(preprocessed_data.columns)} features"
            )
            return preprocessed_data

        except Exception as e:
            logger.error(
                f"Erreur lors du prétraitement des données de marché: {str(e)}"
            )
            # Retourner les données originales en cas d'erreur
            return market_data

    def integrate_data(
        self, market_data, sentiment_data=None, window_size=5, test_split=0.2
    ):
        """
        Intègre les données de marché et de sentiment pour l'apprentissage par renforcement.

        Args:
            market_data (DataFrame): Données de marché prétraitées
            sentiment_data (DataFrame, optional): Données de sentiment
            window_size (int): Taille de la fenêtre d'observation
            test_split (float): Proportion des données à utiliser pour le test

        Returns:
            tuple: (train_data, test_data) DataFrames prêts pour l'RL
        """
        logger.info("Intégration des données pour l'apprentissage par renforcement")

        # Utiliser la fonction de préparation des données pour RL
        train_data, test_data = prepare_data_for_rl(
            market_data=market_data,
            sentiment_data=sentiment_data,
            window_size=window_size,
            test_split=test_split,
        )

        return train_data, test_data

    def _merge_sentiment_data(self, news_sentiment, social_sentiment):
        """
        Fusionne les données de sentiment des actualités et des réseaux sociaux.

        Args:
            news_sentiment (DataFrame): Sentiment des actualités
            social_sentiment (DataFrame): Sentiment des réseaux sociaux

        Returns:
            DataFrame: Données de sentiment fusionnées
        """
        # Si l'une des sources est vide, retourner l'autre
        if news_sentiment.empty and not social_sentiment.empty:
            return social_sentiment
        elif not news_sentiment.empty and social_sentiment.empty:
            return news_sentiment
        elif news_sentiment.empty and social_sentiment.empty:
            return pd.DataFrame()

        # Agréger les sentiments par jour
        news_daily = news_sentiment.resample("D").mean()
        social_daily = social_sentiment.resample("D").mean()

        # Fusionner les deux sources
        merged = pd.merge(
            news_daily,
            social_daily,
            left_index=True,
            right_index=True,
            suffixes=("_news", "_social"),
            how="outer",
        )

        # Remplir les valeurs manquantes
        merged = merged.fillna(method="ffill").fillna(method="bfill")

        # Calculer un score de sentiment combiné
        if (
            "compound_score_news" in merged.columns
            and "compound_score_social" in merged.columns
        ):
            merged["compound_score"] = (
                merged["compound_score_news"] + merged["compound_score_social"]
            ) / 2

        return merged

    def _generate_synthetic_market_data(self, start_date, end_date, interval="1d"):
        """
        Génère des données de marché synthétiques pour les tests.

        Args:
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')
            interval (str): Intervalle de temps

        Returns:
            DataFrame: Données de marché synthétiques
        """
        logger.warning("Génération de données de marché synthétiques")

        # Convertir les dates en objets datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Déterminer le nombre de jours
        if interval == "1d":
            delta = timedelta(days=1)
        elif interval == "1h":
            delta = timedelta(hours=1)
        else:
            delta = timedelta(days=1)

        # Générer les dates
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += delta

        # Générer une marche aléatoire pour les prix
        np.random.seed(42)  # Pour la reproductibilité
        price = 10000  # Prix initial
        prices = [price]

        for _ in range(1, len(dates)):
            change = np.random.normal(0, 200)  # Changement aléatoire
            price += change
            prices.append(max(price, 100))  # Éviter les prix négatifs

        # Générer les volumes
        volumes = np.random.uniform(1000, 10000, len(dates))

        # Créer le DataFrame
        df = pd.DataFrame(
            {
                "close": prices,
                "open": [p * np.random.uniform(0.98, 1.0) for p in prices],
                "high": [p * np.random.uniform(1.0, 1.05) for p in prices],
                "low": [p * np.random.uniform(0.95, 1.0) for p in prices],
                "volume": volumes,
            },
            index=dates,
        )

        return df

    def _generate_synthetic_sentiment_data(self, start_date, end_date):
        """
        Génère des données de sentiment synthétiques pour les tests.

        Args:
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')

        Returns:
            DataFrame: Données de sentiment synthétiques
        """
        logger.warning("Génération de données de sentiment synthétiques")

        # Convertir les dates en objets datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Générer les dates (quotidien)
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        # Générer des scores de sentiment aléatoires
        np.random.seed(42)  # Pour la reproductibilité

        # Sentiment avec une tendance légèrement positive
        compound_scores = np.random.normal(0.1, 0.3, len(dates))
        compound_scores = np.clip(compound_scores, -1, 1)  # Limiter entre -1 et 1

        # Volume de sentiment
        sentiment_volumes = np.random.uniform(10, 100, len(dates))

        # Créer le DataFrame
        df = pd.DataFrame(
            {
                "compound_score": compound_scores,
                "positive_score": [max(0, (1 + cs) / 2) for cs in compound_scores],
                "negative_score": [max(0, (1 - cs) / 2) for cs in compound_scores],
                "neutral_score": np.random.uniform(0.2, 0.5, len(dates)),
                "sentiment_volume": sentiment_volumes,
            },
            index=dates,
        )

        return df

    def _calculate_days(self, start_date, end_date):
        """
        Calcule le nombre de jours entre deux dates.
        
        Args:
            start_date (str): Date de début (format: 'YYYY-MM-DD')
            end_date (str): Date de fin (format: 'YYYY-MM-DD')
            
        Returns:
            int: Nombre de jours entre les deux dates
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end - start
        return delta.days

    def visualize_integrated_data(self, data, save_dir=None):
        """
        Visualise les données intégrées.

        Args:
            data (pd.DataFrame): Données intégrées
            save_dir (str): Répertoire de sauvegarde

        Returns:
            list: Chemins des fichiers de visualisation
        """
        import matplotlib

        matplotlib.use("Agg")  # Utiliser le backend non-interactif
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = tempfile.mkdtemp(prefix="data_visualizations")

        os.makedirs(save_dir, exist_ok=True)

        # Liste des fichiers générés
        files = []

        # 1. Prix de clôture
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["close"])
        plt.title("Prix de clôture")
        plt.xlabel("Date")
        plt.ylabel("Prix")
        plt.grid(True)
        close_path = os.path.join(save_dir, "close_price.png")
        plt.savefig(close_path)
        plt.close()
        files.append("close_price.png")

        # 2. Volume
        if "volume" in data.columns:
            plt.figure(figsize=(12, 6))
            plt.bar(data.index, data["volume"])
            plt.title("Volume")
            plt.xlabel("Date")
            plt.ylabel("Volume")
            plt.grid(True)
            volume_path = os.path.join(save_dir, "volume.png")
            plt.savefig(volume_path)
            plt.close()
            files.append("volume.png")

        # 3. Sentiment si disponible
        if "sentiment" in data.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data["sentiment"])
            plt.title("Sentiment")
            plt.xlabel("Date")
            plt.ylabel("Score de sentiment")
            plt.grid(True)
            sentiment_path = os.path.join(save_dir, "sentiment.png")
            plt.savefig(sentiment_path)
            plt.close()
            files.append("sentiment.png")

        logger.info(f"Visualisations sauvegardées dans {save_dir}")

        return files

    def generate_synthetic_data(
        self, n_samples=100, trend="bullish", volatility=0.02, with_sentiment=True
    ):
        """Génère des données synthétiques pour les tests."""
        # Créer des dates
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

        # Générer des prix selon la tendance
        if trend == "bullish":
            prices = np.linspace(100, 200, n_samples)
        elif trend == "bearish":
            prices = np.linspace(200, 100, n_samples)
        else:  # sideways
            prices = np.ones(n_samples) * 150

        # Ajouter de la volatilité
        prices += np.random.normal(0, volatility * 100, n_samples)

        # Créer le DataFrame avec des valeurs cohérentes
        base_prices = prices.copy()
        highs = base_prices + np.random.uniform(5, 15, n_samples)
        lows = base_prices - np.random.uniform(5, 15, n_samples)
        closes = np.random.uniform(lows, highs, n_samples)

        df = pd.DataFrame(
            {
                "open": base_prices,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.random.uniform(1000, 5000, n_samples),
            },
            index=dates,
        )

        # Ajouter des données de sentiment si demandé
        if with_sentiment:
            df["compound_score"] = np.random.uniform(-1, 1, n_samples)

        logger.info(f"Données synthétiques générées: {n_samples} points")
        return df

    def integrate_sentiment_data(self, market_data, sentiment_data):
        """
        Intègre les données de sentiment aux données de marché.
        
        Args:
            market_data (pd.DataFrame): DataFrame contenant les données de marché
            sentiment_data (pd.DataFrame): DataFrame contenant les données de sentiment
                avec au moins les colonnes 'date', 'polarity', 'subjectivity'
                
        Returns:
            pd.DataFrame: DataFrame combiné avec les données de marché et de sentiment
        """
        logger.info("Intégration des données de sentiment")
        
        if sentiment_data is None or sentiment_data.empty:
            logger.warning("Aucune donnée de sentiment fournie. Retour des données de marché uniquement.")
            return market_data
        
        # S'assurer que les index sont des dates
        market_data = market_data.copy()
        sentiment_data = sentiment_data.copy()
        
        if not isinstance(market_data.index, pd.DatetimeIndex):
            if 'date' in market_data.columns:
                market_data.set_index('date', inplace=True)
            else:
                logger.error("Les données de marché n'ont pas de colonne 'date'.")
                return market_data
        
        if not isinstance(sentiment_data.index, pd.DatetimeIndex):
            if 'date' in sentiment_data.columns:
                sentiment_data.set_index('date', inplace=True)
            else:
                logger.error("Les données de sentiment n'ont pas de colonne 'date'.")
                return market_data
        
        # Resampler les données de sentiment à la même fréquence que les données de marché
        market_freq = pd.infer_freq(market_data.index)
        if market_freq:
            # Calculer la moyenne des scores de sentiment pour chaque période
            sentiment_resampled = sentiment_data.resample(market_freq).mean()
        else:
            # Si la fréquence ne peut pas être déterminée, utiliser une méthode d'alignement d'index
            sentiment_resampled = sentiment_data
        
        # Fusionner les données
        combined_data = market_data.join(sentiment_resampled, how='left')
        
        # Remplir les valeurs manquantes
        sentiment_columns = ['polarity', 'subjectivity', 'compound_score']
        for col in sentiment_columns:
            if col in combined_data.columns:
                # Utiliser une méthode de remplissage avant/arrière pour les valeurs manquantes
                combined_data[col] = combined_data[col].fillna(method='ffill').fillna(method='bfill')
                
                # Si toujours des NaN, remplacer par 0 (neutre)
                combined_data[col] = combined_data[col].fillna(0)
        
        logger.info(f"Données intégrées avec succès. Colonnes: {combined_data.columns.tolist()}")
        return combined_data
