import os
from datetime import datetime
from pathlib import Path

# Configuration de base
class Config:
    # Configuration générale
    SECRET_KEY = 'your-secret-key'
    DEBUG = True
    
    # Chemins de fichiers
    BASE_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'ai_trading', 'info_retour', 'data')
    
    # S'assurer que le dossier existe
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Chemin du fichier de transactions
    DATA_FILE = os.path.join(DATA_DIR, 'transactions.json')
    
    # Configuration API
    API_URL = 'https://api.coingecko.com/api/v3'
    API_KEY = ''  # Votre clé API si nécessaire
    
    # Liste des cryptos supportées
    SUPPORTED_CRYPTOS = [
        'BTC', 'ETH', 'USDT', 'BNB', 'USDC', 
        'XRP', 'ADA', 'DOGE', 'SOL', 'DOT'
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

BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
LOG_DIR = PROJECT_ROOT / "ai_trading" / "info_retour" / "logs"

# Création des dossiers nécessaires
os.makedirs(LOG_DIR, exist_ok=True) 