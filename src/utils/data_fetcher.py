import pandas as pd
import requests
from datetime import datetime, timedelta
from src.config.config import *

def fetch_binance_data(symbol=TRADING_PAIR.replace('/', ''), interval=INTERVAL, limit=100):
    """Récupère les données historiques depuis Binance"""
    base_url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convertir les données en DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convertir les types de données
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"Erreur lors de la récupération des données: {str(e)}")
        return None

def get_current_price(symbol=TRADING_PAIR.replace('/', '')):
    """Récupère le prix actuel depuis Binance"""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Erreur lors de la récupération du prix: {str(e)}")
        return None 