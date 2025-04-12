import requests
import pandas as pd

class DataCollector:
    def fetch_crypto_data(self, symbol, start_date, end_date, interval="1d"):
        """Récupère les données de marché depuis CoinGecko"""
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': symbol.lower(),
            'from': pd.Timestamp(start_date).timestamp(),
            'to': pd.Timestamp(end_date).timestamp(),
            'interval': interval
        }
        response = requests.get(url, params=params)
        return pd.DataFrame(response.json())

    def fetch_crypto_news(self, symbol):
        """Récupère les actualités depuis CryptoPanic"""
        params = {
            'auth_token': 'VOTRE_TOKEN',  # À remplacer
            'currencies': symbol.upper(),
            'public': 'true'
        }
        response = requests.get("https://cryptopanic.com/api/v1/posts/", params=params)
        return pd.DataFrame(response.json()['results']) 