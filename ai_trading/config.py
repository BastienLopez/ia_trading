"""
Configuration principale du système de trading IA.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Chemins des dossiers
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Création des dossiers nécessaires
for directory in [MODELS_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)

# Configuration des APIs
BINANCE_CONFIG = {
    "api_key": os.getenv("BINANCE_API_KEY"),
    "api_secret": os.getenv("BINANCE_API_SECRET"),
    "testnet": True  # Mettre à False pour le trading réel
}

# Configuration des modèles LLM
LLM_CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
    "sentiment_model_path": os.getenv("SENTIMENT_MODEL_PATH", "models/sentiment"),
    "prediction_model_path": os.getenv("PREDICTION_MODEL_PATH", "models/prediction")
}

# Configuration des réseaux sociaux
TWITTER_CONFIG = {
    "api_key": os.getenv("TWITTER_API_KEY"),
    "api_secret": os.getenv("TWITTER_API_SECRET"),
    "access_token": os.getenv("TWITTER_ACCESS_TOKEN"),
    "access_token_secret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
}

# Configuration de la base de données
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL")
}

# Paramètres de trading
TRADING_CONFIG = {
    "risk_percentage": float(os.getenv("RISK_PERCENTAGE", 1)),
    "max_position_size": float(os.getenv("MAX_POSITION_SIZE", 5000)),
    "stop_loss_percentage": float(os.getenv("STOP_LOSS_PERCENTAGE", 2)),
    "take_profit_percentage": float(os.getenv("TAKE_PROFIT_PERCENTAGE", 6))
}

# Configuration des paires de trading
TRADING_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT"
]

# Intervalles de temps pour l'analyse
TIME_FRAMES = {
    "sentiment": "1h",
    "technical": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "prediction": "1h"
}

# Configuration des modèles ML
ML_CONFIG = {
    "model_path": os.getenv("ML_MODEL_PATH", "models/trading"),
    "features": [
        "close", "volume", "rsi", "macd", "bollinger_bands",
        "sentiment_score", "market_trend", "volatility"
    ],
    "target": "price_direction",
    "train_split": 0.8,
    "test_split": 0.2,
    "random_state": 42
}

# Configuration des indicateurs techniques
TECHNICAL_INDICATORS = {
    "RSI": {"timeperiod": 14},
    "MACD": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "BB": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
    "EMA": {"timeperiod": 20},
    "ATR": {"timeperiod": 14}
}

# Configuration du logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True
        }
    }
} 