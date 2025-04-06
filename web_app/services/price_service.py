import requests
import logging
import json
from web_app.config import Config

logger = logging.getLogger(__name__)

class PriceService:
    @staticmethod
    def get_real_crypto_prices():
        """Récupère les prix réels des cryptomonnaies depuis CoinGecko"""
        try:
            logger.info("Récupération des prix depuis CoinGecko...")
            
            # Construire la liste des IDs pour la requête
            crypto_ids = ",".join(Config.COINGECKO_IDS.values())
            
            # Faire la requête à CoinGecko
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_ids}&vs_currencies=usd"
            logger.info(f"URL CoinGecko: {url}")
            
            response = requests.get(url)
            logger.info(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                coingecko_data = response.json()
                logger.info(f"Données brutes de CoinGecko: {json.dumps(coingecko_data, indent=2)}")
                
                prices = {}
                for symbol, coin_id in Config.COINGECKO_IDS.items():
                    if coin_id in coingecko_data:
                        current_price = coingecko_data[coin_id]["usd"]
                        prices[symbol] = current_price
                        logger.info(f"Prix {symbol}: ${current_price:,.2f}")
                
                return prices
            else:
                logger.error(f"Erreur CoinGecko API: {response.status_code}")
                logger.error(f"Réponse: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prix: {str(e)}")
            return None

    @staticmethod
    def get_price_for_symbol(symbol, transactions=None):
        """Récupère le prix pour un symbole donné"""
        # Essayer d'abord de récupérer le prix depuis CoinGecko
        current_prices = PriceService.get_real_crypto_prices()
        
        if current_prices and symbol in current_prices:
            return float(current_prices[symbol])
        
        # Si CoinGecko n'est pas disponible, chercher le dernier prix connu
        if transactions:
            for t in reversed(transactions):
                if t['symbol'] == symbol:
                    return t['price']
        
        # Si aucun prix n'est trouvé, utiliser la valeur par défaut
        return Config.DEFAULT_PRICES.get(symbol, 1) 