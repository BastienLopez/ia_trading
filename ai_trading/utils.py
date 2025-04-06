import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional

class TradeLogger:
    """
    Classe pour enregistrer et analyser les trades
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.trades_file = f"{log_dir}/trades.csv"
        
        # Créer le fichier s'il n'existe pas
        if not os.path.exists(self.trades_file):
            pd.DataFrame(columns=[
                'timestamp', 'symbol', 'action', 'price', 'quantity', 
                'value', 'fee', 'profit', 'profit_pct', 'model'
            ]).to_csv(self.trades_file, index=False)
    
    def log_trade(self, trade_info: Dict[str, Any]):
        """
        Enregistre un trade dans le fichier de log
        """
        # Ajouter le timestamp s'il n'existe pas
        if 'timestamp' not in trade_info:
            trade_info['timestamp'] = datetime.now().isoformat()
        
        # Lire le fichier existant
        df = pd.read_csv(self.trades_file)
        
        # Ajouter le nouveau trade
        df = pd.concat([df, pd.DataFrame([trade_info])], ignore_index=True)
        
        # Enregistrer le fichier mis à jour
        df.to_csv(self.trades_file, index=False)
        
        return True
    
    def get_trades(self, symbol=None, start_date=None, end_date=None):
        """
        Récupère les trades selon les critères spécifiés
        """
        df = pd.read_csv(self.trades_file)
        
        # Filtrer par symbole
        if symbol:
            df = df[df['symbol'] == symbol]
        
        # Convertir en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filtrer par date
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        return df
    
    def get_performance_metrics(self, symbol=None, period='all'):
        """
        Calcule les métriques de performance des trades
        """
        df = self.get_trades(symbol=symbol)
        
        if df.empty:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_sum": 0,
                "avg_profit_pct": 0,
                "max_profit_pct": 0,
                "max_loss_pct": 0,
                "sharpe_ratio": 0
            }
        
        # Filtrer par période
        if period != 'all':
            if period == 'week':
                start_date = pd.Timestamp.now() - pd.Timedelta(days=7)
            elif period == 'month':
                start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
            elif period == 'year':
                start_date = pd.Timestamp.now() - pd.Timedelta(days=365)
            
            df = df[df['timestamp'] >= start_date]
        
        # Calculer les métriques
        total_trades = len(df)
        profitable_trades = len(df[df['profit'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        profit_sum = df['profit'].sum()
        avg_profit_pct = df['profit_pct'].mean()
        max_profit_pct = df['profit_pct'].max()
        max_loss_pct = df['profit_pct'].min()
        
        # Sharpe ratio (si on a assez de données)
        daily_returns = df.groupby(df['timestamp'].dt.date)['profit_pct'].sum()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 and daily_returns.std() > 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_sum": float(profit_sum),
            "avg_profit_pct": float(avg_profit_pct),
            "max_profit_pct": float(max_profit_pct),
            "max_loss_pct": float(max_loss_pct),
            "sharpe_ratio": float(sharpe_ratio)
        }
    
    def plot_performance(self, symbol=None, save_path=None):
        """
        Génère un graphique de performance des trades
        """
        df = self.get_trades(symbol=symbol)
        
        if df.empty:
            plt.figure(figsize=(10, 6))
            plt.title("Aucun trade à afficher")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.close()
            return
        
        # Convertir en datetime si nécessaire
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Trier par date
        df = df.sort_values('timestamp')
        
        # Calculer le profit cumulatif
        df['cumulative_profit'] = df['profit'].cumsum()
        
        # Créer le graphique
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Graphique du profit cumulatif
        ax1.plot(df['timestamp'], df['cumulative_profit'], label='Profit cumulatif')
        ax1.set_title(f"Performance de trading {symbol if symbol else 'tous symboles'}")
        ax1.set_ylabel('Profit ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Graphique des trades individuels
        colors = ['green' if p > 0 else 'red' for p in df['profit']]
        ax2.bar(df['timestamp'], df['profit'], color=colors)
        ax2.set_title('Profits/Pertes par trade')
        ax2.set_ylabel('Profit/Perte ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder si demandé
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class MarketDataAPI:
    """
    Classe pour interagir avec des APIs externes de données de marché
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('CRYPTO_API_KEY')
    
    def get_current_price(self, symbol="BTC/USDT"):
        """
        Récupère le prix actuel d'une cryptomonnaie
        """
        try:
            # Utiliser CoinGecko API (gratuit, sans clé API)
            coin_id = symbol.split('/')[0].lower()
            vs_currency = symbol.split('/')[1].lower()
            
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={vs_currency}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return data[coin_id][vs_currency]
            else:
                return None
        except Exception as e:
            print(f"Erreur lors de la récupération du prix: {str(e)}")
            return None
    
    def get_market_sentiment(self, symbol="BTC"):
        """
        Récupère le sentiment du marché pour une cryptomonnaie
        """
        try:
            # Utiliser Fear & Greed Index API (alternative de démo)
            url = "https://api.alternative.me/fng/"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "value": int(data['data'][0]['value']),
                    "classification": data['data'][0]['value_classification'],
                    "timestamp": data['data'][0]['timestamp']
                }
            else:
                return None
        except Exception as e:
            print(f"Erreur lors de la récupération du sentiment: {str(e)}")
            return None
    
def config_loader(config_file="config.json"):
    """
    Charge un fichier de configuration
    """
    if not os.path.exists(config_file):
        default_config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "INFO"
            },
            "rl_agent": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "default_timesteps": 50000
            },
            "data": {
                "default_exchange": "binance",
                "default_symbol": "BTC/USDT",
                "default_timeframe": "1h",
                "default_days": 60
            },
            "trading": {
                "initial_balance": 10000,
                "transaction_fee": 0.001,
                "max_position_size": 0.1
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return default_config
    
    with open(config_file, 'r') as f:
        return json.load(f) 