import requests
import logging
import json
from web_app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceService:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.crypto_ids = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'AVAX': 'avalanche-2',
            'DOT': 'polkadot',
            'MATIC': 'matic-network',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'AAVE': 'aave',
            'ATOM': 'cosmos',
            'DOGE': 'dogecoin',
            'SHIB': 'shiba-inu',
            'USDT': 'tether',
            'USDC': 'usd-coin',
            'DAI': 'dai'
        }

    def get_real_crypto_prices(self, symbols=None):
        """Récupère les prix réels des cryptomonnaies depuis CoinGecko"""
        logger.info("Récupération des prix depuis CoinGecko...")
        
        if symbols is None:
            ids = list(self.crypto_ids.values())
        else:
            ids = [self.crypto_ids[symbol] for symbol in symbols if symbol in self.crypto_ids]
        
        if not ids:
            return None

        url = f"{self.base_url}/simple/price"
        params = {
            'ids': ','.join(ids),
            'vs_currencies': 'usd'
        }
        
        logger.info(f"URL CoinGecko: {url}?ids={params['ids']}&vs_currencies={params['vs_currencies']}")
        
        try:
            response = requests.get(url, params=params)
            logger.info(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Données brutes de CoinGecko: {data}")
                
                # Convertir les données en format {symbol: price}
                prices = {}
                for symbol, coin_id in self.crypto_ids.items():
                    if coin_id in data:
                        price = data[coin_id]['usd']
                        prices[symbol] = price
                        logger.info(f"Prix {symbol}: ${price:,.2f}")
                
                return prices
            else:
                logger.error(f"Erreur lors de la récupération des prix: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prix: {str(e)}")
            return None

    def get_price_for_symbol(self, symbol):
        """Récupère le prix pour un symbole spécifique"""
        prices = self.get_real_crypto_prices([symbol])
        if prices and symbol in prices:
            return prices[symbol]
        return None 