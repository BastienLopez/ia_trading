"""
Module pour la récupération des données de marché pour les cryptomonnaies.
Fournit les fonctionnalités pour obtenir des données OHLCV (Open, High, Low, Close, Volume).
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any

import pandas as pd
import numpy as np

# Configurer le logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class MarketDataFetcher:
    """
    Classe pour récupérer les données de marché des cryptomonnaies.
    """

    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        """
        Initialise le récupérateur de données de marché.

        Args:
            api_key (str, optional): Clé API pour les sources de données réelles
            use_cache (bool): Si True, utilise des données en cache lorsque disponibles
        """
        self.api_key = api_key
        self.use_cache = use_cache
        self.cache_dir = os.path.join(os.path.dirname(__file__), "../info_retour/cache/market_data")
        
        # Créer le répertoire cache s'il n'existe pas
        if use_cache and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Répertoire de cache créé: {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Impossible de créer le répertoire de cache: {e}")
                self.use_cache = False
                
        logger.info(f"MarketDataFetcher initialisé (cache {'activé' if use_cache else 'désactivé'})")

    def fetch_crypto_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        source: str = "synthetic",
    ) -> pd.DataFrame:
        """
        Récupère les données historiques pour une cryptomonnaie.

        Args:
            symbol (str): Symbole de la cryptomonnaie (ex: BTC, ETH, etc.)
            start_date: Date de début
            end_date: Date de fin
            interval (str): Intervalle des données ('1d', '1h', etc.)
            source (str): Source des données ('binance', 'coinbase', 'synthetic', etc.)

        Returns:
            pd.DataFrame: DataFrame contenant les données OHLCV
        """
        # Convertir les dates en objets datetime si nécessaire
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        # Vérifier le cache si activé
        if self.use_cache:
            cache_file = self._get_cache_filename(symbol, start_date, end_date, interval, source)
            if os.path.exists(cache_file):
                try:
                    data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    logger.info(f"Données chargées depuis le cache: {cache_file}")
                    return data
                except Exception as e:
                    logger.warning(f"Échec du chargement depuis le cache: {e}")
        
        # Générer des données synthétiques par défaut
        if source == "synthetic" or not self.api_key:
            return self._generate_synthetic_data(symbol, start_date, end_date, interval)
        
        # À ce stade, nous pourrions implémenter différentes sources de données réelles
        # Mais pour la version de base, nous utiliserons simplement des données synthétiques
        logger.warning(f"Source '{source}' non implémentée, utilisation de données synthétiques")
        return self._generate_synthetic_data(symbol, start_date, end_date, interval)

    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Génère des données synthétiques pour les tests et le développement.

        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date (datetime): Date de début
            end_date (datetime): Date de fin
            interval (str): Intervalle des données

        Returns:
            pd.DataFrame: DataFrame contenant les données synthétiques
        """
        # Déterminer le nombre de périodes en fonction de l'intervalle
        if interval == "1d":
            periods = (end_date - start_date).days + 1
            freq = "D"
        elif interval == "1h":
            periods = int((end_date - start_date).total_seconds() / 3600) + 1
            freq = "H"
        else:
            periods = 100  # Valeur par défaut
            freq = "D"
            
        # Créer une série d'index de dates
        date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Initialiser un DataFrame vide
        df = pd.DataFrame(index=date_range)
        
        # Générer des prix de base en fonction du symbole
        base_price = {
            "BTC": 45000.0,
            "ETH": 3000.0,
            "SOL": 100.0,
            "XRP": 0.5,
        }.get(symbol.upper(), 100.0)  # Prix par défaut si symbole non reconnu
        
        # Générer des prix avec un processus brownien géométrique
        np.random.seed(42)  # Pour reproductibilité
        returns = np.random.normal(0.0005, 0.02, size=len(date_range))
        price_series = base_price * np.cumprod(1 + returns)
        
        # Créer les colonnes OHLCV
        df["close"] = price_series
        df["open"] = df["close"].shift(1)
        df.loc[df.index[0], "open"] = price_series[0] * 0.99  # Premier prix d'ouverture
        
        # Ajouter une variation pour high et low
        daily_volatility = 0.02
        df["high"] = df["close"] * (1 + np.random.uniform(0, daily_volatility, size=len(df)))
        df["low"] = df["close"] * (1 - np.random.uniform(0, daily_volatility, size=len(df)))
        
        # S'assurer que high est toujours >= open/close et low est toujours <= open/close
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        
        # Générer le volume
        avg_volume = base_price * 1000  # Volume proportionnel au prix de base
        df["volume"] = np.random.gamma(shape=2, scale=avg_volume/2, size=len(df))
        
        # Enregistrer dans le cache si activé
        if self.use_cache:
            cache_file = self._get_cache_filename(symbol, start_date, end_date, interval, "synthetic")
            try:
                df.to_csv(cache_file)
                logger.info(f"Données synthétiques enregistrées dans le cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Échec de l'enregistrement dans le cache: {e}")
        
        logger.info(f"Données synthétiques générées pour {symbol}: {len(df)} points")
        return df

    def _get_cache_filename(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        interval: str,
        source: str
    ) -> str:
        """
        Génère un nom de fichier pour le cache.
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
            start_date (datetime): Date de début
            end_date (datetime): Date de fin
            interval (str): Intervalle des données
            source (str): Source des données
            
        Returns:
            str: Chemin du fichier de cache
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return os.path.join(
            self.cache_dir, 
            f"{symbol.upper()}_{start_str}_{end_str}_{interval}_{source}.csv"
        )


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)
    
    # Créer une instance de MarketDataFetcher
    fetcher = MarketDataFetcher()
    
    # Récupérer des données pour Bitcoin sur les 30 derniers jours
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    data = fetcher.fetch_crypto_data(
        symbol="BTC",
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    
    print(data.head()) 