# -*- coding: utf-8 -*-
"""
Module pour la récupération de données de marché pour les cryptomonnaies.
Ce module fournit des interfaces pour récupérer des données historiques
et actuelles des marchés de cryptomonnaies.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Configurer le logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Ajouter le chemin d'accès pour importer depuis le répertoire parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector

class MarketDataFetcher:
    """
    Classe pour récupérer des données de marché pour les cryptomonnaies.
    Cette classe encapsule les fonctionnalités de EnhancedDataCollector.
    """
    
    def __init__(self):
        self.data_collector = EnhancedDataCollector()
        logger.info("MarketDataFetcher initialisé")
    
    def fetch_crypto_data(self, symbol, start_date, end_date, interval='1d'):
        """
        Récupère les données historiques pour une cryptomonnaie.
        
        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: BTC, ETH, etc.)
            start_date (str): Date de début au format 'YYYY-MM-DD'
            end_date (str): Date de fin au format 'YYYY-MM-DD'
            interval (str): Intervalle des données ('1d', '1h', etc.)
            
        Returns:
            pd.DataFrame: DataFrame contenant les données historiques
        """
        logger.info(f"Récupération des données pour {symbol} du {start_date} au {end_date}")
        
        # Convertir les dates en nombre de jours si elles sont au format string
        if isinstance(start_date, str) and isinstance(end_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days + 1
        else:
            days = 30  # Valeur par défaut
        
        try:
            # Utiliser EnhancedDataCollector pour récupérer les données
            df = self.data_collector.get_merged_price_data(
                coin_id=symbol.lower(),
                days=days,
                vs_currency="usd",
                include_fear_greed=True
            )
            
            # Filtrer les données selon les dates spécifiées
            if not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[(df.index >= pd.to_datetime(start_date)) & 
                            (df.index <= pd.to_datetime(end_date))]
                
                logger.info(f"Données récupérées avec succès: {len(df)} points")
                return df
            else:
                logger.warning(f"Aucune donnée récupérée pour {symbol}")
                
                # Générer des données synthétiques si aucune donnée n'est récupérée
                return self._generate_synthetic_data(symbol, start_date, end_date, interval)
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            logger.info("Génération de données synthétiques comme fallback")
            return self._generate_synthetic_data(symbol, start_date, end_date, interval)
    
    def _generate_synthetic_data(self, symbol, start_date, end_date, interval):
        """
        Génère des données synthétiques en cas d'échec de récupération des données réelles.
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date (str): Date de début au format 'YYYY-MM-DD'
            end_date (str): Date de fin au format 'YYYY-MM-DD'
            interval (str): Intervalle des données
            
        Returns:
            pd.DataFrame: DataFrame contenant des données synthétiques
        """
        import numpy as np
        
        # Convertir les dates en datetime
        start_dt = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end_dt = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        
        # Créer un index de dates
        if interval == '1d':
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        elif interval == '1h':
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
        else:
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Générer des prix de base selon le symbole
        base_price = 100
        if symbol.upper() == 'BTC':
            base_price = 30000
        elif symbol.upper() == 'ETH':
            base_price = 2000
        elif symbol.upper() == 'SOL':
            base_price = 100
        
        # Générer des données aléatoires avec tendance
        n = len(date_range)
        trend = np.linspace(-0.2, 0.2, n)  # Tendance légère
        noise = np.random.normal(0, 0.02, n)  # Bruit aléatoire
        change = trend + noise
        
        prices = [base_price]
        for i in range(1, n):
            next_price = prices[-1] * (1 + change[i])
            prices.append(next_price)
        
        prices = np.array(prices)
        
        # Créer le DataFrame
        df = pd.DataFrame(
            {
                'open': prices * np.random.uniform(0.99, 1.0, n),
                'high': prices * np.random.uniform(1.0, 1.05, n),
                'low': prices * np.random.uniform(0.95, 1.0, n),
                'close': prices,
                'volume': np.random.uniform(1000, 10000, n) * base_price / 100,
                'source': ['synthetic'] * n
            },
            index=date_range
        )
        
        logger.info(f"Données synthétiques générées: {len(df)} points pour {symbol}")
        return df
    
    def get_crypto_price_data(self, symbol, start_date, end_date, interval='1d'):
        """
        Alias pour fetch_crypto_data pour la compatibilité avec l'exemple.
        """
        return self.fetch_crypto_data(symbol, start_date, end_date, interval) 