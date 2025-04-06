"""
Module de prétraitement des données pour le trading de cryptomonnaies.
"""

import logging
from typing import Dict, List, Optional, Union
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ta

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Classe pour le prétraitement des données de trading."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialise le préprocesseur de données.
        
        Args:
            cache_dir: Répertoire pour le cache des données
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialisation des scalers
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = StandardScaler()
    
    def clean_market_data(
        self,
        df: pd.DataFrame,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Nettoie les données de marché.
        
        Args:
            df: DataFrame avec les données OHLCV
            fill_method: Méthode pour remplir les valeurs manquantes
            
        Returns:
            DataFrame nettoyé
        """
        # Copie pour éviter la modification en place
        df = df.copy()
        
        # Suppression des doublons
        df = df[~df.index.duplicated(keep='first')]
        
        # Tri par index temporel
        df = df.sort_index()
        
        # Gestion des valeurs manquantes
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        if fill_method == 'ffill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif fill_method == 'interpolate':
            df = df.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        price_cols: List[str] = ['open', 'high', 'low', 'close'],
        volume_cols: List[str] = ['volume'],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalise les données de prix et volume.
        
        Args:
            df: DataFrame avec les données
            price_cols: Colonnes de prix à normaliser
            volume_cols: Colonnes de volume à normaliser
            fit: Si True, ajuste les scalers aux données
            
        Returns:
            DataFrame avec données normalisées
        """
        df = df.copy()
        
        if price_cols:
            if fit:
                df[price_cols] = self.price_scaler.fit_transform(df[price_cols])
            else:
                df[price_cols] = self.price_scaler.transform(df[price_cols])
        
        if volume_cols:
            if fit:
                df[volume_cols] = self.volume_scaler.fit_transform(df[volume_cols])
            else:
                df[volume_cols] = self.volume_scaler.transform(df[volume_cols])
        
        return df
    
    def create_features(
        self,
        df: pd.DataFrame,
        feature_types: List[str] = ['momentum', 'trend', 'volatility', 'volume']
    ) -> pd.DataFrame:
        """
        Crée des features techniques.
        
        Args:
            df: DataFrame avec données OHLCV
            feature_types: Types de features à créer
            
        Returns:
            DataFrame avec features ajoutées
        """
        df = df.copy()
        
        if 'momentum' in feature_types:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        
        if 'trend' in feature_types:
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
        
        if 'volatility' in feature_types:
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close']
            ).average_true_range()
        
        if 'volume' in feature_types:
            # Volume features
            df['volume_ema'] = ta.trend.EMAIndicator(
                df['volume'], window=20
            ).ema_indicator()
            
            # On-Balance Volume
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                df['close'], df['volume']
            ).on_balance_volume()
        
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        clean: bool = True,
        normalize: bool = True,
        add_features: bool = True,
        feature_types: List[str] = ['momentum', 'trend', 'volatility', 'volume']
    ) -> pd.DataFrame:
        """
        Prépare les données en appliquant nettoyage, normalisation et création de features.
        
        Args:
            df: DataFrame avec données brutes
            clean: Si True, nettoie les données
            normalize: Si True, normalise les données
            add_features: Si True, ajoute des features techniques
            feature_types: Types de features à créer
            
        Returns:
            DataFrame préparé
        """
        if clean:
            df = self.clean_market_data(df)
        
        if add_features:
            df = self.create_features(df, feature_types)
        
        if normalize:
            # Séparation des colonnes de prix et volume
            price_cols = ['open', 'high', 'low', 'close']
            volume_cols = ['volume']
            
            # Normalisation
            df = self.normalize_data(df, price_cols, volume_cols)
        
        return df
    
    def prepare_sequence_data(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> tuple:
        """
        Prépare les données en séquences pour l'apprentissage.
        
        Args:
            df: DataFrame préparé
            sequence_length: Longueur des séquences
            target_column: Colonne cible pour la prédiction
            feature_columns: Colonnes à utiliser comme features
            
        Returns:
            Tuple (X, y) avec séquences et cibles
        """
        if feature_columns is None:
            feature_columns = df.columns.tolist()
        
        # Création des séquences
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            sequence = df[feature_columns].iloc[i:i + sequence_length].values
            target = df[target_column].iloc[i + sequence_length]
            
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets) 