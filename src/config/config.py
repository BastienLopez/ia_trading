import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration Discord
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

# Configuration Trading
TRADING_PAIR = "BTC/USDT"
INTERVAL = "1d"  # Daily timeframe

# Configuration des indicateurs
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

EMA_SHORT = 9
EMA_LONG = 21

# Seuils de volume
VOLUME_THRESHOLD = 1000000  # Volume minimum pour considérer un signal

# Configuration des signaux
SIGNAL_STRENGTH = {
    'STRONG': 3,  # Tous les indicateurs sont alignés
    'MEDIUM': 2,  # Deux indicateurs sont alignés
    'WEAK': 1     # Un seul indicateur
} 