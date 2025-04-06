"""
Module de prétraitement des données pour le trading de cryptomonnaies.
Gère le nettoyage, la normalisation et la création de features pour l'analyse.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Téléchargement des ressources NLTK nécessaires (à exécuter une seule fois)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataPreprocessor")

class MarketDataPreprocessor:
    """Classe pour prétraiter les données de marché (OHLCV)."""
    
    def __init__(self, scaling_method: str = 'minmax'):
        """
        Initialise le préprocesseur de données de marché.
        
        Args:
            scaling_method: Méthode de mise à l'échelle ('minmax' ou 'standard')
        """
        self.scaling_method = scaling_method
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Méthode de scaling non supportée: {scaling_method}")
        
        logger.info(f"Préprocesseur de données de marché initialisé avec scaling: {scaling_method}")
    
    def clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données de marché.
        
        Args:
            df: DataFrame contenant les données OHLCV
            
        Returns:
            DataFrame nettoyé
        """
        logger.info("Nettoyage des données de marché")
        
        # Copie pour éviter de modifier l'original
        df_clean = df.copy()
        
        # Vérification des colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans les données: {missing_columns}")
        
        # Conversion des types
        for col in required_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Gestion des valeurs manquantes
        if df_clean.isnull().sum().sum() > 0:
            logger.warning(f"Valeurs manquantes détectées: {df_clean.isnull().sum()}")
            # Interpolation pour les valeurs de prix
            for col in ['open', 'high', 'low', 'close']:
                df_clean[col] = df_clean[col].interpolate(method='time')
            # Remplacement des volumes manquants par 0
            df_clean['volume'] = df_clean['volume'].fillna(0)
        
        # Suppression des doublons
        duplicates = df_clean.index.duplicated()
        if duplicates.any():
            logger.warning(f"Doublons détectés: {duplicates.sum()}")
            df_clean = df_clean[~duplicates]
        
        # Tri par index temporel
        df_clean = df_clean.sort_index()
        
        logger.info(f"Nettoyage terminé. Dimensions: {df_clean.shape}")
        return df_clean
    
    def normalize_market_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Normalise les données de marché.
        
        Args:
            df: DataFrame contenant les données OHLCV
            columns: Liste des colonnes à normaliser (toutes par défaut)
            
        Returns:
            DataFrame avec données normalisées
        """
        logger.info("Normalisation des données de marché")
        
        # Copie pour éviter de modifier l'original
        df_norm = df.copy()
        
        # Colonnes à normaliser
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Vérification des colonnes
        missing_columns = [col for col in columns if col not in df_norm.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes pour la normalisation: {missing_columns}")
        
        # Normalisation
        df_norm[columns] = self.scaler.fit_transform(df_norm[columns])
        
        logger.info(f"Normalisation terminée pour les colonnes: {columns}")
        return df_norm
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features techniques à partir des données OHLCV.
        
        Args:
            df: DataFrame contenant les données OHLCV
            
        Returns:
            DataFrame avec features techniques ajoutées
        """
        logger.info("Création des features techniques")
        
        # Copie pour éviter de modifier l'original
        df_features = df.copy()
        
        # Vérification des colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_features.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes pour les features: {missing_columns}")
        
        # 1. Rendements
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # 2. Moyennes mobiles
        for window in [5, 10, 20, 50, 200]:
            df_features[f'ma_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'ma_vol_{window}'] = df_features['volume'].rolling(window=window).mean()
        
        # 3. Volatilité
        for window in [5, 10, 20]:
            df_features[f'volatility_{window}'] = df_features['returns'].rolling(window=window).std()
        
        # 4. RSI (Relative Strength Index)
        delta = df_features['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df_features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 5. MACD (Moving Average Convergence Divergence)
        ema_12 = df_features['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_features['close'].ewm(span=26, adjust=False).mean()
        df_features['macd'] = ema_12 - ema_26
        df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
        df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']
        
        # 6. Bollinger Bands
        for window in [20]:
            df_features[f'bb_middle_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'bb_std_{window}'] = df_features['close'].rolling(window=window).std()
            df_features[f'bb_upper_{window}'] = df_features[f'bb_middle_{window}'] + 2 * df_features[f'bb_std_{window}']
            df_features[f'bb_lower_{window}'] = df_features[f'bb_middle_{window}'] - 2 * df_features[f'bb_std_{window}']
            df_features[f'bb_width_{window}'] = (df_features[f'bb_upper_{window}'] - df_features[f'bb_lower_{window}']) / df_features[f'bb_middle_{window}']
        
        # 7. Momentum
        for window in [5, 10, 20]:
            df_features[f'momentum_{window}'] = df_features['close'] / df_features['close'].shift(window) - 1
        
        # 8. Stochastic Oscillator
        for window in [14]:
            df_features[f'stoch_k_{window}'] = 100 * ((df_features['close'] - df_features['low'].rolling(window=window).min()) / 
                                                    (df_features['high'].rolling(window=window).max() - df_features['low'].rolling(window=window).min()))
            df_features[f'stoch_d_{window}'] = df_features[f'stoch_k_{window}'].rolling(window=3).mean()
        
        # 9. Price to Moving Average Ratios
        for window in [5, 10, 20, 50, 200]:
            df_features[f'price_to_ma_{window}'] = df_features['close'] / df_features[f'ma_{window}']
        
        # 10. Volume Features
        df_features['volume_change'] = df_features['volume'].pct_change()
        df_features['volume_ma_ratio_5'] = df_features['volume'] / df_features['volume'].rolling(window=5).mean()
        
        # Suppression des lignes avec NaN (début des séries temporelles)
        df_features = df_features.dropna()
        
        logger.info(f"Création des features terminée. Nouvelles dimensions: {df_features.shape}")
        return df_features
    
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Crée des features décalées (lagged) pour l'apprentissage supervisé.
        
        Args:
            df: DataFrame contenant les données
            columns: Liste des colonnes pour lesquelles créer des lags
            lags: Liste des périodes de décalage
            
        Returns:
            DataFrame avec features décalées
        """
        logger.info(f"Création des features décalées pour {len(columns)} colonnes avec {len(lags)} lags")
        
        # Copie pour éviter de modifier l'original
        df_lagged = df.copy()
        
        # Vérification des colonnes
        missing_columns = [col for col in columns if col not in df_lagged.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes pour les lags: {missing_columns}")
        
        # Création des features décalées
        for col in columns:
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
            raise ValueError("Colonne 'close' manquante pour la création de la cible")
        
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
        logger.info(f"Division des données avec ratios: train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio}")
        
        # Vérification des ratios
        if train_ratio + val_ratio >= 1:
            raise ValueError("La somme des ratios train et validation doit être inférieure à 1")
        
        # Tri par index temporel
        df = df.sort_index()
        
        # Calcul des indices de division
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        # Division des données
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        logger.info(f"Division terminée. Dimensions: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
        return train_df, val_df, test_df


class TextDataPreprocessor:
    """Classe pour prétraiter les données textuelles (actualités, tweets)."""
    
    def __init__(self, language: str = 'english'):
        """
        Initialise le préprocesseur de données textuelles.
        
        Args:
            language: Langue pour les stopwords ('english', 'french', etc.)
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
        logger.info(f"Préprocesseur de données textuelles initialisé avec langue: {language}")
    
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
    
    def preprocess_text_data(self, texts: List[str]) -> List[List[str]]:
        """
        Prétraite une liste de textes.
        
        Args:
            texts: Liste de textes à prétraiter
            
        Returns:
            Liste de listes de tokens
        """
        logger.info(f"Prétraitement de {len(texts)} textes")
        
        processed_texts = []
        for text in texts:
            # Nettoyage
            clean = self.clean_text(text)
            # Tokenization
            tokens = self.tokenize_text(clean)
            processed_texts.append(tokens)
        
        logger.info("Prétraitement des textes terminé")
        return processed_texts
    
    def preprocess_news_data(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Prétraite des données d'actualités.
        
        Args:
            news_data: Liste de dictionnaires contenant les actualités
            
        Returns:
            DataFrame avec les actualités prétraitées
        """
        logger.info(f"Prétraitement de {len(news_data)} actualités")
        
        # Conversion en DataFrame
        df = pd.DataFrame(news_data)
        
        # Vérification des colonnes requises
        required_columns = ['title', 'body', 'published_on']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans les actualités: {missing_columns}")
        
        # Prétraitement des titres et contenus
        df['clean_title'] = df['title'].apply(self.clean_text)
        df['clean_body'] = df['body'].apply(self.clean_text)
        
        # Tokenization
        df['tokens_title'] = df['clean_title'].apply(self.tokenize_text)
        df['tokens_body'] = df['clean_body'].apply(self.tokenize_text)
        
        # Conversion des dates si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df['published_on']):
            df['published_on'] = pd.to_datetime(df['published_on'])
        
        # Tri par date
        df = df.sort_values('published_on')
        
        logger.info("Prétraitement des actualités terminé")
        return df
    
    def preprocess_social_data(self, social_data: List[Dict]) -> pd.DataFrame:
        """
        Prétraite des données de réseaux sociaux.
        
        Args:
            social_data: Liste de dictionnaires contenant les posts sociaux
            
        Returns:
            DataFrame avec les données sociales prétraitées
        """
        logger.info(f"Prétraitement de {len(social_data)} posts sociaux")
        
        # Conversion en DataFrame
        df = pd.DataFrame(social_data)
        
        # Vérification des colonnes requises
        required_columns = ['text', 'created_at']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans les données sociales: {missing_columns}")
        
        # Prétraitement du texte
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        # Tokenization
        df['tokens'] = df['clean_text'].apply(self.tokenize_text)
        
        # Conversion des dates si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Tri par date
        df = df.sort_values('created_at')
        
        logger.info("Prétraitement des données sociales terminé")
        return df


# Exemple d'utilisation
if __name__ == "__main__":
    import os
    
    # Chargement des données de marché (exemple)
    market_data_path = "data/btc_prices_30d.csv"
    if os.path.exists(market_data_path):
        # Prétraitement des données de marché
        market_data = pd.read_csv(market_data_path, index_col='timestamp', parse_dates=True)
        
        # Initialisation du préprocesseur
        market_preprocessor = MarketDataPreprocessor(scaling_method='minmax')
        
        # Nettoyage
        clean_data = market_preprocessor.clean_market_data(market_data)
        
        # Création des features techniques
        feature_data = market_preprocessor.create_technical_features(clean_data)
        
        # Création des features décalées
        lagged_columns = ['close', 'volume', 'returns', 'rsi_14', 'macd']
        lagged_data = market_preprocessor.create_lagged_features(feature_data, lagged_columns, [1, 2, 3, 5, 10])
        
        # Création de la variable cible
        target_data = market_preprocessor.create_target_variable(lagged_data, horizon=1, method='direction')
        
        # Division des données
        train_df, val_df, test_df = market_preprocessor.split_data(target_data)
        
        # Sauvegarde des données prétraitées
        os.makedirs("data/processed", exist_ok=True)
        train_df.to_csv("data/processed/train_data.csv")
        val_df.to_csv("data/processed/val_data.csv")
        test_df.to_csv("data/processed/test_data.csv")
        
        print(f"Données prétraitées sauvegardées. Dimensions finales: {target_data.shape}")
        print(f"Nombre de features: {target_data.shape[1] - 1}")  # -1 pour exclure la cible
    else:
        print(f"Fichier de données non trouvé: {market_data_path}")
        print("Exécutez d'abord le collecteur de données pour générer les données de marché.") 