import os
import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MinimalDataCollector")

class MinimalDataCollector:
    """Version simplifiée du collecteur de données qui fonctionne sans clés API."""
    
    def __init__(self):
        """Initialise le collecteur de données minimal."""
        # CoinGecko ne nécessite pas de clé API
        self.coingecko = CoinGeckoAPI()
        logger.info("Collecteur de données minimal initialisé")
    
    def get_crypto_prices(self, coin_id='bitcoin', vs_currency='usd', days=30):
        """
        Récupère les prix historiques d'une cryptomonnaie.
        
        Args:
            coin_id: ID de la crypto sur CoinGecko (ex: 'bitcoin', 'ethereum')
            vs_currency: Devise de référence (ex: 'usd', 'eur')
            days: Nombre de jours d'historique
            
        Returns:
            DataFrame avec les données de prix
        """
        try:
            logger.info(f"Récupération des prix pour {coin_id} sur {days} jours")
            data = self.coingecko.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # Création du DataFrame
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            
            # Fusion des données
            df = prices_df.merge(volumes_df, on='timestamp')
            
            # Conversion des timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Données récupérées: {len(df)} entrées")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prix: {e}")
            return None
    
    def get_trending_coins(self):
        """
        Récupère les cryptomonnaies tendance sur CoinGecko.
        
        Returns:
            Liste des cryptos tendance
        """
        try:
            logger.info("Récupération des cryptos tendance")
            trending = self.coingecko.get_search_trending()
            return trending['coins']
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des tendances: {e}")
            return []
    
    def save_data(self, data, filename):
        """
        Sauvegarde les données dans un fichier CSV.
        
        Args:
            data: DataFrame à sauvegarder
            filename: Nom du fichier
        """
        try:
            # Créer le dossier data s'il n'existe pas
            os.makedirs('data', exist_ok=True)
            
            filepath = f"data/{filename}"
            data.to_csv(filepath)
            
            logger.info(f"Données sauvegardées dans {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {e}")


# Exemple d'utilisation
if __name__ == "__main__":
    collector = MinimalDataCollector()
    
    # Récupérer les prix du Bitcoin
    btc_data = collector.get_crypto_prices(coin_id='bitcoin', days=30)
    if btc_data is not None:
        collector.save_data(btc_data, 'btc_prices_30d.csv')
    
    # Récupérer les cryptos tendance
    trending = collector.get_trending_coins()
    if trending:
        print("Cryptos tendance:")
        for coin in trending:
            print(f"- {coin['item']['name']} ({coin['item']['symbol']})") 