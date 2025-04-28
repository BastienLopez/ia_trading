import os
from datetime import datetime

# Configuration de base
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_very_secret')
    API_URL = os.environ.get('API_URL', 'http://localhost:8000')
    DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'transactions.json')
    
    # Liste des cryptos supportées
    SUPPORTED_CRYPTOS = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", 
        "MATIC", "LINK", "UNI", "AAVE", "ATOM", "DOGE", "SHIB",
        "USDT", "USDC", "DAI"
    ]
    
    # Prix par défaut pour les cryptos
    DEFAULT_PRICES = {
        'BTC': 50000,
        'ETH': 3000,
        'BNB': 400,
        'SOL': 100,
        'XRP': 0.5,
        'ADA': 0.5,
        'AVAX': 30,
        'DOT': 7,
        'MATIC': 1,
        'LINK': 15,
        'UNI': 7,
        'AAVE': 100,
        'ATOM': 10,
        'DOGE': 0.1,
        'SHIB': 0.00001,
        'USDT': 1,
        'USDC': 1,
        'DAI': 1
    }
    
    # Mapping des symboles vers les IDs CoinGecko
    COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "DOT": "polkadot",
        "MATIC": "matic-network",
        "LINK": "chainlink",
        "UNI": "uniswap",
        "AAVE": "aave",
        "ATOM": "cosmos",
        "DOGE": "dogecoin",
        "SHIB": "shiba-inu",
        "USDT": "tether",
        "USDC": "usd-coin",
        "DAI": "dai"
    } 