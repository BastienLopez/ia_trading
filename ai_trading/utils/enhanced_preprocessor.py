"""
Préprocesseur amélioré pour les données de cryptomonnaies provenant de différentes sources.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import ta  # Bibliothèque pour les indicateurs techniques
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedPreprocessor")


class EnhancedMarketDataPreprocessor:
    """Classe améliorée pour prétraiter les données de marché des cryptomonnaies."""

    def __init__(self, scaling="minmax"):
        self.scaling = scaling
        self.scaler = None
        self.logger = logging.getLogger("EnhancedPreprocessor")
        self.logger.setLevel(logging.INFO)
        
        # Configuration du logger
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"Préprocesseur amélioré de données de marché initialisé avec scaling: {scaling}")

    def clean_market_data(self, data):
        """Nettoie les données de marché."""
        try:
            # Conversion en float32 pour éviter les problèmes de précision
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data = data.copy()
            data[numeric_cols] = data[numeric_cols].astype(np.float32)
            
            # Traitement des valeurs manquantes
            data = data.ffill().bfill()
            
            # Suppression des doublons
            data = data.drop_duplicates()
            
            # Traitement des valeurs aberrantes avec IQR
            for col in numeric_cols:
                Q1 = data[col].astype(np.float32).quantile(0.25)
                Q3 = data[col].astype(np.float32).quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower_bound, upper_bound).astype(np.float32)
            
            # S'assurer que toutes les colonnes numériques sont en float32
            data[numeric_cols] = data[numeric_cols].astype(np.float32)
            
            return data
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
            raise

    def normalize_market_data(self, data):
        """Normalise les données de marché."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if self.scaling == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            
            # Normalisation avec clipping pour éviter les valeurs infinies
            normalized_data = data.copy()
            normalized_data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
            normalized_data[numeric_cols] = normalized_data[numeric_cols].clip(-1e6, 1e6)
            
            return normalized_data
        except Exception as e:
            self.logger.error(f"Erreur lors de la normalisation des données: {str(e)}")
            raise

    def create_target_variable(self, data, horizon=1, method="return"):
        """Crée la variable cible pour l'apprentissage."""
        try:
            self.logger.info(f"Création de la variable cible avec horizon={horizon}, méthode={method}")
            
            # Conversion en float32 pour éviter les problèmes de précision
            data = data.astype(np.float32)
            
            if method == "return":
                target = data["close"].pct_change(horizon).shift(-horizon)
            elif method == "direction":
                target = np.sign(data["close"].pct_change(horizon).shift(-horizon))
            elif method == "threshold":
                returns = data["close"].pct_change(horizon).shift(-horizon)
                target = pd.cut(returns, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[-1, 0, 1])
            else:
                raise ValueError(f"Méthode {method} non supportée")
            
            # Suppression des valeurs manquantes
            target = target.dropna()
            
            self.logger.info(f"Création de la cible terminée. Dimensions finales: {target.shape}")
            return target
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de la variable cible: {str(e)}")
            raise

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features techniques à partir des données de marché.

        Args:
            df: DataFrame contenant les données de marché

        Returns:
            DataFrame avec les features techniques ajoutées
        """
        logger.info("Création des features techniques")

        # Vérification des colonnes nécessaires
        required_columns = ["open", "high", "low", "close", "volume"]

        # Si 'close' n'existe pas mais 'price' existe, utiliser 'price' comme 'close'
        if "close" not in df.columns and "price" in df.columns:
            df["close"] = df["price"]

        # Vérifier si les colonnes nécessaires existent
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(
                f"Colonnes manquantes pour les features techniques: {missing_columns}"
            )
            # Si certaines colonnes sont manquantes, créer des colonnes de substitution
            if "open" not in df.columns and "close" in df.columns:
                df["open"] = df["close"]
            if "high" not in df.columns and "close" in df.columns:
                df["high"] = df["close"]
            if "low" not in df.columns and "close" in df.columns:
                df["low"] = df["close"]
            if "volume" not in df.columns:
                df["volume"] = 0

        # Copie du DataFrame pour éviter les modifications en place
        df_tech = df.copy()

        try:
            # Calcul des rendements avec gestion explicite des NaN
            df_tech = df_tech.copy()  # Pour éviter les avertissements de SettingWithCopyWarning
            close_prices = df_tech["close"]
            df_tech["returns"] = (close_prices - close_prices.shift(1)) / close_prices.shift(1)
            df_tech["returns"] = df_tech["returns"].fillna(0)  # Remplacer les NaN par 0

            # Moyennes mobiles
            df_tech["sma_7"] = df_tech["close"].rolling(window=7).mean()
            df_tech["sma_21"] = df_tech["close"].rolling(window=21).mean()
            df_tech["sma_50"] = df_tech["close"].rolling(window=50).mean()

            # Écart-type (volatilité)
            df_tech["volatility_7"] = df_tech["close"].rolling(window=7).std()

            # RSI
            delta = df_tech["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df_tech["rsi_14"] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df_tech["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df_tech["close"].ewm(span=26, adjust=False).mean()
            df_tech["macd"] = ema_12 - ema_26
            df_tech["macd_signal"] = df_tech["macd"].ewm(span=9, adjust=False).mean()
            df_tech["macd_hist"] = df_tech["macd"] - df_tech["macd_signal"]

            # Bollinger Bands
            df_tech["bb_middle"] = df_tech["close"].rolling(window=20).mean()
            df_tech["bb_std"] = df_tech["close"].rolling(window=20).std()
            df_tech["bb_upper"] = df_tech["bb_middle"] + 2 * df_tech["bb_std"]
            df_tech["bb_lower"] = df_tech["bb_middle"] - 2 * df_tech["bb_std"]

            # Suppression des lignes avec des NaN (dues aux fenêtres de calcul)
            df_tech = df_tech.dropna()

            logger.info(
                f"Création des features techniques terminée. Nouvelles dimensions: {df_tech.shape}"
            )
            return df_tech

        except Exception as e:
            logger.error(f"Erreur lors de la création des features techniques: {e}")
            # En cas d'erreur, retourner le DataFrame original
            return df

    def create_lagged_features(
        self, df: pd.DataFrame, columns: List[str], lags: List[int]
    ) -> pd.DataFrame:
        """
        Crée des features décalées (lagged) pour capturer les dépendances temporelles.

        Args:
            df: DataFrame contenant les données
            columns: Liste des colonnes pour lesquelles créer des lags
            lags: Liste des décalages à créer

        Returns:
            DataFrame avec features décalées
        """
        logger.info(
            f"Création des features décalées pour {len(columns)} colonnes avec {len(lags)} lags"
        )

        # Copie pour éviter de modifier l'original
        df_lagged = df.copy()

        # Création des features décalées
        for col in columns:
            if col in df_lagged.columns:
                for lag in lags:
                    df_lagged[f"{col}_lag_{lag}"] = df_lagged[col].shift(lag)

        # Suppression des lignes avec NaN (début des séries temporelles)
        df_lagged = df_lagged.dropna()

        logger.info(
            f"Création des lags terminée. Nouvelles dimensions: {df_lagged.shape}"
        )
        return df_lagged

    def split_data(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divise les données en ensembles d'entraînement, validation et test.

        Args:
            df: DataFrame contenant les données
            train_ratio: Proportion des données pour l'entraînement
            val_ratio: Proportion des données pour la validation

        Returns:
            Tuple de DataFrames (train, validation, test)
        """
        test_ratio = 1.0 - train_ratio - val_ratio
        logger.info(
            f"Division des données avec ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

        # Tri par date pour s'assurer que la division est chronologique
        df_sorted = df.sort_index()

        # Calcul des indices de division
        n = len(df_sorted)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        # Division des données
        train_df = df_sorted.iloc[:train_size]
        val_df = df_sorted.iloc[train_size : train_size + val_size]
        test_df = df_sorted.iloc[train_size + val_size :]

        logger.info(
            f"Division terminée. Dimensions: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}"
        )
        return train_df, val_df, test_df

    def preprocess_market_data(self, market_data, **kwargs):
        """
        Prétraite les données de marché complètes.

        Args:
            market_data: DataFrame ou chemin vers un fichier CSV contenant les données de marché
            **kwargs: Arguments supplémentaires pour les différentes étapes de prétraitement

        Returns:
            DataFrame prétraité
        """
        logger.info(f"Prétraitement complet des données de marché")

        # Chargement des données si un chemin de fichier est fourni
        if isinstance(market_data, str):
            try:
                if os.path.exists(market_data):
                    df = pd.read_csv(market_data, index_col=0, parse_dates=True)
                    logger.info(f"Données chargées depuis {market_data}")
                else:
                    logger.error(f"Fichier non trouvé: {market_data}")
                    return None
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données: {e}")
                return None
        else:
            # Si c'est déjà un DataFrame
            try:
                df = market_data.copy()
                logger.info(f"Utilisation du DataFrame fourni, dimensions: {df.shape}")
            except Exception as e:
                logger.error(f"Erreur lors de la copie du DataFrame: {e}")
                return None

        try:
            # Nettoyage des données
            df_clean = self.clean_market_data(df)

            # Création des features techniques
            df_features = self.create_technical_features(df_clean)

            # Création des lags
            lag_columns = (
                ["close", "volume"]
                if "close" in df_features.columns
                else ["price", "volume"]
            )
            lag_columns = [col for col in lag_columns if col in df_features.columns]

            if lag_columns:
                df_lagged = self.create_lagged_features(
                    df_features, lag_columns, lags=[1, 2, 3]
                )
            else:
                logger.warning(
                    "Aucune colonne pour les lags trouvée, utilisation des données sans lags"
                )
                df_lagged = df_features

            # Normalisation
            df_normalized = self.normalize_market_data(df_lagged)

            # Création de la variable cible (si demandé)
            target_method = kwargs.get("target_method", "direction")
            target_horizon = kwargs.get("target_horizon", 1)

            if kwargs.get("create_target", False):
                df_target = self.create_target_variable(
                    df_normalized, horizon=target_horizon, method=target_method
                )
            else:
                df_target = df_normalized

            logger.info(f"Prétraitement terminé, dimensions finales: {df_target.shape}")
            return df_target

        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données: {e}")
            # En cas d'erreur, retourner les données nettoyées si disponibles
            if "df_clean" in locals():
                logger.warning("Retour des données nettoyées uniquement")
                return df_clean
            return None


class EnhancedTextDataPreprocessor:
    """Classe améliorée pour prétraiter les données textuelles (actualités, réseaux sociaux)."""

    def __init__(self, language: str = "english"):
        """
        Initialise le préprocesseur de données textuelles.

        Args:
            language: Langue pour les stopwords ('english', 'french', etc.)
        """
        self.language = language

        # Téléchargement des ressources NLTK si nécessaire
        try:
            nltk.data.find(f"corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        try:
            nltk.data.find(f"tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find(f"corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")

        logger.info(
            f"Préprocesseur amélioré de données textuelles initialisé avec langue: {language}"
        )

    def clean_text(self, text: str) -> str:
        """
        Nettoie un texte (suppression des caractères spéciaux, URLs, etc.).

        Args:
            text: Texte à nettoyer

        Returns:
            Texte nettoyé
        """
        if not isinstance(text, str):
            return ""

        # Conversion en minuscules
        text = text.lower()

        # Suppression des URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Suppression des mentions (@user)
        text = re.sub(r"@\w+", "", text)

        # Suppression des hashtags
        text = re.sub(r"#\w+", "", text)

        # Suppression des symboles de cryptomonnaies ($BTC)
        text = re.sub(r"\$\w+", "", text)

        # Suppression des caractères spéciaux et chiffres
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)

        # Suppression des espaces multiples
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize le texte et supprime les stopwords.

        Args:
            text: Texte à tokenizer

        Returns:
            Liste de tokens
        """
        if not text:
            return []

        try:
            # Utilisation d'une méthode de tokenization plus simple en cas d'erreur
            try:
                tokens = word_tokenize(text)
            except LookupError:
                # Fallback simple si word_tokenize échoue
                tokens = text.split()

            # Suppression des stopwords
            stop_words = set(stopwords.words(self.language))
            tokens = [
                token.lower() for token in tokens if token.lower() not in stop_words
            ]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

            return tokens
        except Exception as e:
            logger.error(f"Erreur lors de la tokenization: {e}")
            return []

    def preprocess_news_data(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Prétraite les données d'actualités.

        Args:
            news_data: Liste de dictionnaires contenant les actualités

        Returns:
            DataFrame avec les données prétraitées
        """
        logger.info(f"Prétraitement de {len(news_data)} actualités")

        # Conversion en DataFrame
        df = pd.DataFrame(news_data)

        # Nettoyage des textes
        if "title" in df.columns:
            df["clean_title"] = df["title"].apply(self.clean_text)
            df["tokens_title"] = df["clean_title"].apply(self.tokenize_text)

        if "body" in df.columns:
            df["clean_body"] = df["body"].apply(self.clean_text)
            df["tokens_body"] = df["clean_body"].apply(self.tokenize_text)

        # Conversion des dates
        if "published_on" in df.columns:
            df["published_on"] = pd.to_datetime(df["published_on"])

        logger.info("Prétraitement des actualités terminé")
        return df

    def preprocess_social_data(self, social_data: List[Dict]) -> pd.DataFrame:
        """
        Prétraite les données des réseaux sociaux.

        Args:
            social_data: Liste de dictionnaires contenant les posts sociaux

        Returns:
            DataFrame avec les données prétraitées
        """
        logger.info(f"Prétraitement de {len(social_data)} posts sociaux")

        # Conversion en DataFrame
        df = pd.DataFrame(social_data)

        # Nettoyage des textes
        if "text" in df.columns:
            df["clean_text"] = df["text"].apply(self.clean_text)
            df["tokens"] = df["clean_text"].apply(self.tokenize_text)

        # Conversion des dates
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])

        logger.info("Prétraitement des données sociales terminé")
        return df

    def extract_sentiment_keywords(
        self, tokens_list: List[List[str]], top_n: int = 20
    ) -> List[str]:
        """
        Extrait les mots-clés les plus fréquents liés au sentiment.

        Args:
            tokens_list: Liste de listes de tokens
            top_n: Nombre de mots-clés à extraire

        Returns:
            Liste des mots-clés les plus fréquents
        """
        # Aplatir la liste de listes
        all_tokens = [token for tokens in tokens_list for token in tokens]

        # Comptage des fréquences
        from collections import Counter

        token_counts = Counter(all_tokens)

        # Extraction des mots-clés les plus fréquents
        top_keywords = [word for word, count in token_counts.most_common(top_n)]

        return top_keywords


class DataPreprocessor:
    """Classe pour prétraiter les données de marché"""

    def __init__(self):
        self.logger = logging.getLogger("DataPreprocessor")
        self.logger.setLevel(logging.INFO)

    def preprocess_market_data(self, data):
        """Prétraite les données de marché"""
        # Implémentation de base
        return data

    def preprocess(self, data):
        """Alias pour preprocess_market_data"""
        return self.preprocess_market_data(data)

    def add_technical_indicators(self, data):
        """Ajoute des indicateurs techniques aux données de marché"""
        # Implémentation simple
        df = data.copy()

        # Calculer quelques indicateurs techniques de base
        if "close" in df.columns:
            # Moyennes mobiles
            df["sma_5"] = df["close"].rolling(window=5).mean()
            df["sma_20"] = df["close"].rolling(window=20).mean()

            # RSI (version simplifiée)
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)  # Éviter division par zéro
            df["rsi"] = 100 - (100 / (1 + rs))

        # Remplir les valeurs NaN
        df = df.ffill().bfill()

        return df


# Exemple d'utilisation
if __name__ == "__main__":
    # Prétraitement des données de marché
    market_preprocessor = EnhancedMarketDataPreprocessor()

    # Exemple avec un fichier de données fusionnées
    market_data_path = "data/bitcoin_merged_30d.csv"
    if os.path.exists(market_data_path):
        processed_data = market_preprocessor.preprocess_market_data(market_data_path)
        print(f"Données prétraitées: {processed_data.shape}")
    else:
        print(f"Fichier non trouvé: {market_data_path}")

    # Prétraitement des données textuelles
    text_preprocessor = EnhancedTextDataPreprocessor()

    # Exemple avec des actualités fictives
    sample_news = [
        {
            "title": "Bitcoin Surges to New Highs",
            "body": "Bitcoin reached $60,000 today, setting a new record.",
            "published_on": "2023-01-01",
        },
        {
            "title": "Ethereum Update Delayed",
            "body": "The Ethereum 2.0 update has been delayed until next quarter.",
            "published_on": "2023-01-02",
        },
    ]

    processed_news = text_preprocessor.preprocess_news_data(sample_news)
    print(f"Actualités prétraitées: {processed_news.shape}")

    # Extraction des mots-clés
    if "tokens_title" in processed_news.columns:
        keywords = text_preprocessor.extract_sentiment_keywords(
            processed_news["tokens_title"].tolist()
        )
        print(f"Mots-clés extraits: {keywords}")
