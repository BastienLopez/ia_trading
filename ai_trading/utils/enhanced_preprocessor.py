"""
Préprocesseur amélioré pour les données de cryptomonnaies provenant de différentes sources.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ta  # Bibliothèque pour les indicateurs techniques
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedPreprocessor")

class EnhancedMarketDataPreprocessor:
    """Classe améliorée pour prétraiter les données de marché des cryptomonnaies."""
    
    def __init__(self, scaling_method: str = 'minmax'):
        """
        Initialise le préprocesseur de données de marché.
        
        Args:
            scaling_method: Méthode de normalisation ('minmax' ou 'standard')
        """
        self.scaling_method = scaling_method
        self.scalers = {}  # Stockage des scalers pour chaque colonne
        
        logger.info(f"Préprocesseur amélioré de données de marché initialisé avec scaling: {scaling_method}")
    
    def clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données de marché (valeurs manquantes, outliers, etc.).
        
        Args:
            df: DataFrame contenant les données de marché
            
        Returns:
            DataFrame nettoyé
        """
        logger.info("Nettoyage des données de marché")
        
        # Copie du DataFrame pour éviter les modifications en place
        df_clean = df.copy()
        
        # Vérification des valeurs manquantes
        missing_values = df_clean.isna().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Valeurs manquantes détectées: {missing_values}")
            
            # Conversion des types pour éviter les warnings
            for col in df_clean.select_dtypes(include=['object']).columns:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass
            
            # Inférence des types d'objets avant interpolation (pour éviter le FutureWarning)
            df_clean = df_clean.infer_objects(copy=False)
            
            # Interpolation des valeurs manquantes
            df_clean = df_clean.interpolate(method='time')
            
            # Remplissage des valeurs manquantes restantes (pour éviter le FutureWarning)
            df_clean = df_clean.ffill().bfill()
        
        # Détection et traitement des outliers
        for col in ['open', 'high', 'low', 'close', 'price', 'volume']:
            if col in df_clean.columns:
                # Calcul des limites pour les outliers (méthode IQR)
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Remplacement des outliers par les limites
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        logger.info(f"Nettoyage terminé. Dimensions: {df_clean.shape}")
        return df_clean
    
    def normalize_market_data(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalise les données de marché.
        
        Args:
            df: DataFrame contenant les données de marché
            columns: Liste des colonnes à normaliser (toutes les colonnes numériques par défaut)
            
        Returns:
            DataFrame avec données normalisées
        """
        logger.info("Normalisation des données de marché")
        
        # Copie pour éviter de modifier l'original
        df_norm = df.copy()
        
        # Sélection des colonnes à normaliser
        if columns is None:
            numeric_cols = df_norm.select_dtypes(include=['number']).columns.tolist()
            # Exclure les colonnes catégorielles ou binaires
            columns = [col for col in numeric_cols if col not in ['target', 'source']]
        
        # Normalisation des colonnes sélectionnées
        for col in columns:
            if col in df_norm.columns:
                # Création du scaler approprié
                if self.scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.scaling_method == 'standard':
                    scaler = StandardScaler()
                else:
                    raise ValueError(f"Méthode de scaling non supportée: {self.scaling_method}")
                
                # Reshape pour le format attendu par le scaler
                values = df_norm[col].values.reshape(-1, 1)
                
                # Fit et transform
                normalized_values = scaler.fit_transform(values).flatten()
                
                # Stockage du scaler pour une utilisation ultérieure
                self.scalers[col] = scaler
                
                # Remplacement des valeurs
                df_norm[col] = normalized_values
        
        logger.info(f"Normalisation terminée pour les colonnes: {columns}")
        return df_norm
    
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
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Si 'close' n'existe pas mais 'price' existe, utiliser 'price' comme 'close'
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
        
        # Vérifier si les colonnes nécessaires existent
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Colonnes manquantes pour les features techniques: {missing_columns}")
            # Si certaines colonnes sont manquantes, créer des colonnes de substitution
            if 'open' not in df.columns and 'close' in df.columns:
                df['open'] = df['close']
            if 'high' not in df.columns and 'close' in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns and 'close' in df.columns:
                df['low'] = df['close']
            if 'volume' not in df.columns:
                df['volume'] = 0
        
        # Copie du DataFrame pour éviter les modifications en place
        df_tech = df.copy()
        
        try:
            # Calcul des rendements
            df_tech['returns'] = df_tech['close'].pct_change(fill_method='pad')
            
            # Moyennes mobiles
            df_tech['sma_7'] = df_tech['close'].rolling(window=7).mean()
            df_tech['sma_21'] = df_tech['close'].rolling(window=21).mean()
            df_tech['sma_50'] = df_tech['close'].rolling(window=50).mean()
            
            # Écart-type (volatilité)
            df_tech['volatility_7'] = df_tech['close'].rolling(window=7).std()
            
            # RSI
            delta = df_tech['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df_tech['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df_tech['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_tech['close'].ewm(span=26, adjust=False).mean()
            df_tech['macd'] = ema_12 - ema_26
            df_tech['macd_signal'] = df_tech['macd'].ewm(span=9, adjust=False).mean()
            df_tech['macd_hist'] = df_tech['macd'] - df_tech['macd_signal']
            
            # Bollinger Bands
            df_tech['bb_middle'] = df_tech['close'].rolling(window=20).mean()
            df_tech['bb_std'] = df_tech['close'].rolling(window=20).std()
            df_tech['bb_upper'] = df_tech['bb_middle'] + 2 * df_tech['bb_std']
            df_tech['bb_lower'] = df_tech['bb_middle'] - 2 * df_tech['bb_std']
            
            # Suppression des lignes avec des NaN (dues aux fenêtres de calcul)
            df_tech = df_tech.dropna()
            
            logger.info(f"Création des features techniques terminée. Nouvelles dimensions: {df_tech.shape}")
            return df_tech
        
        except Exception as e:
            logger.error(f"Erreur lors de la création des features techniques: {e}")
            # En cas d'erreur, retourner le DataFrame original
            return df
    
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Crée des features décalées (lagged) pour capturer les dépendances temporelles.
        
        Args:
            df: DataFrame contenant les données
            columns: Liste des colonnes pour lesquelles créer des lags
            lags: Liste des décalages à créer
            
        Returns:
            DataFrame avec features décalées
        """
        logger.info(f"Création des features décalées pour {len(columns)} colonnes avec {len(lags)} lags")
        
        # Copie pour éviter de modifier l'original
        df_lagged = df.copy()
        
        # Création des features décalées
        for col in columns:
            if col in df_lagged.columns:
                for lag in lags:
                    df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
        
        # Suppression des lignes avec NaN (début des séries temporelles)
        df_lagged = df_lagged.dropna()
        
        logger.info(f"Création des lags terminée. Nouvelles dimensions: {df_lagged.shape}")
        return df_lagged
    
    def create_target_variable(self, df: pd.DataFrame, horizon: int = 1, method: str = 'return') -> pd.DataFrame:
        """
        Crée la variable cible pour l'apprentissage supervisé.
        
        Args:
            df: DataFrame contenant les données
            horizon: Horizon de prédiction (nombre de périodes)
            method: Méthode de calcul ('return', 'direction', 'threshold')
            
        Returns:
            DataFrame avec variable cible
        """
        logger.info(f"Création de la variable cible avec horizon={horizon}, méthode={method}")
        
        # Copie pour éviter de modifier l'original
        df_target = df.copy()
        
        # Vérification des colonnes requises
        if 'close' not in df_target.columns:
            if 'price' in df_target.columns:
                df_target['close'] = df_target['price']
            else:
                raise ValueError("Colonne 'close' ou 'price' manquante pour la création de la cible")
        
        # Création de la variable cible selon la méthode
        if method == 'return':
            # Rendement futur
            df_target['target'] = df_target['close'].shift(-horizon) / df_target['close'] - 1
            
        elif method == 'direction':
            # Direction du prix (hausse/baisse)
            future_price = df_target['close'].shift(-horizon)
            df_target['target'] = (future_price > df_target['close']).astype(int)
            
        elif method == 'threshold':
            # Mouvement significatif (seuil de 1%)
            future_return = df_target['close'].shift(-horizon) / df_target['close'] - 1
            df_target['target'] = pd.cut(
                future_return,
                bins=[-np.inf, -0.01, 0.01, np.inf],
                labels=[-1, 0, 1]
            ).fillna(0).astype(int)
            
        else:
            raise ValueError(f"Méthode non supportée: {method}")
        
        # Suppression des lignes avec NaN (fin des séries temporelles)
        df_target = df_target.dropna()
        
        logger.info(f"Création de la cible terminée. Dimensions finales: {df_target.shape}")
        return df_target
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        logger.info(f"Division des données avec ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # Tri par date pour s'assurer que la division est chronologique
        df_sorted = df.sort_index()
        
        # Calcul des indices de division
        n = len(df_sorted)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Division des données
        train_df = df_sorted.iloc[:train_size]
        val_df = df_sorted.iloc[train_size:train_size+val_size]
        test_df = df_sorted.iloc[train_size+val_size:]
        
        logger.info(f"Division terminée. Dimensions: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
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
            lag_columns = ['close', 'volume'] if 'close' in df_features.columns else ['price', 'volume']
            lag_columns = [col for col in lag_columns if col in df_features.columns]
            
            if lag_columns:
                df_lagged = self.create_lagged_features(df_features, lag_columns, lags=[1, 2, 3])
            else:
                logger.warning("Aucune colonne pour les lags trouvée, utilisation des données sans lags")
                df_lagged = df_features
            
            # Normalisation
            df_normalized = self.normalize_market_data(df_lagged)
            
            # Création de la variable cible (si demandé)
            target_method = kwargs.get('target_method', 'direction')
            target_horizon = kwargs.get('target_horizon', 1)
            
            if kwargs.get('create_target', False):
                df_target = self.create_target_variable(df_normalized, horizon=target_horizon, method=target_method)
            else:
                df_target = df_normalized
            
            logger.info(f"Prétraitement terminé, dimensions finales: {df_target.shape}")
            return df_target
        
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données: {e}")
            # En cas d'erreur, retourner les données nettoyées si disponibles
            if 'df_clean' in locals():
                logger.warning("Retour des données nettoyées uniquement")
                return df_clean
            return None


class EnhancedTextDataPreprocessor:
    """Classe améliorée pour prétraiter les données textuelles (actualités, réseaux sociaux)."""
    
    def __init__(self, language: str = 'english'):
        """
        Initialise le préprocesseur de données textuelles.
        
        Args:
            language: Langue pour les stopwords ('english', 'french', etc.)
        """
        self.language = language
        
        # Téléchargement des ressources NLTK si nécessaire
        try:
            nltk.data.find(f'corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find(f'tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find(f'corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        logger.info(f"Préprocesseur amélioré de données textuelles initialisé avec langue: {language}")
    
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
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Suppression des mentions (@user)
        text = re.sub(r'@\w+', '', text)
        
        # Suppression des hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Suppression des symboles de cryptomonnaies ($BTC)
        text = re.sub(r'\$\w+', '', text)
        
        # Suppression des caractères spéciaux et chiffres
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
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
            tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
            
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
        if 'title' in df.columns:
            df['clean_title'] = df['title'].apply(self.clean_text)
            df['tokens_title'] = df['clean_title'].apply(self.tokenize_text)
        
        if 'body' in df.columns:
            df['clean_body'] = df['body'].apply(self.clean_text)
            df['tokens_body'] = df['clean_body'].apply(self.tokenize_text)
        
        # Conversion des dates
        if 'published_on' in df.columns:
            df['published_on'] = pd.to_datetime(df['published_on'])
        
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
        if 'text' in df.columns:
            df['clean_text'] = df['text'].apply(self.clean_text)
            df['tokens'] = df['clean_text'].apply(self.tokenize_text)
        
        # Conversion des dates
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        logger.info("Prétraitement des données sociales terminé")
        return df
    
    def extract_sentiment_keywords(self, tokens_list: List[List[str]], top_n: int = 20) -> List[str]:
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
            'title': 'Bitcoin Surges to New Highs',
            'body': 'Bitcoin reached $60,000 today, setting a new record.',
            'published_on': '2023-01-01'
        },
        {
            'title': 'Ethereum Update Delayed',
            'body': 'The Ethereum 2.0 update has been delayed until next quarter.',
            'published_on': '2023-01-02'
        }
    ]
    
    processed_news = text_preprocessor.preprocess_news_data(sample_news)
    print(f"Actualités prétraitées: {processed_news.shape}")
    
    # Extraction des mots-clés
    if 'tokens_title' in processed_news.columns:
        keywords = text_preprocessor.extract_sentiment_keywords(processed_news['tokens_title'].tolist())
        print(f"Mots-clés extraits: {keywords}") 