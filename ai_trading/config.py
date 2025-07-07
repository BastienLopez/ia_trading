"""
Configuration globale pour le projet ai_trading.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Obtenir le chemin absolu du répertoire du projet
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir les chemins des répertoires
INFO_RETOUR_DIR = PROJECT_ROOT / "ai_trading" / "info_retour"
VISUALIZATION_DIR = INFO_RETOUR_DIR / "visualization"
DATA_DIR = INFO_RETOUR_DIR / "data"
MODELS_DIR = INFO_RETOUR_DIR / "models"
LOGS_DIR = INFO_RETOUR_DIR / "logs"
CHECKPOINTS_DIR = INFO_RETOUR_DIR / "checkpoints"
EVALUATION_DIR = INFO_RETOUR_DIR / "evaluation"
TEST_DIR = INFO_RETOUR_DIR / "test"
SENTIMENT_CACHE_DIR = INFO_RETOUR_DIR / "sentiment_cache"

# Créer les répertoires s'ils n'existent pas
for directory in [
    INFO_RETOUR_DIR,
    VISUALIZATION_DIR,
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    EVALUATION_DIR,
    TEST_DIR,
    SENTIMENT_CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration du modèle
MODEL_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "buffer_size": 100000,
    "target_update_tau": 0.005,
    "reward_scaling": 1.0,
    "use_automatic_entropy_tuning": True,
    "hidden_sizes": [256, 256],
}

# Configuration de l'environnement
ENV_CONFIG = {
    "initial_balance": 10000.0,
    "transaction_fee": 0.001,
    "window_size": 20,
    "include_position": True,
    "include_balance": True,
    "include_technical_indicators": True,
    "risk_management": True,
    "normalize_observation": True,
    "reward_function": "sharpe",
    "risk_aversion": 0.1,
    "transaction_penalty": 0.001,
    "lookback_window": 20,
    "action_type": "continuous",
    "n_discrete_actions": 5,
    "slippage_model": "dynamic",
    "slippage_value": 0.001,
    "execution_delay": 1,
}

# Configuration du trading
TRADING_CONFIG = {
    "max_episodes": 1000,
    "max_steps": 10000,
    "evaluation_interval": 10,
    "save_interval": 100,
    "log_interval": 1,
    "random_seed": 42,
}

# Configuration des APIs
BINANCE_CONFIG = {
    "api_key": os.getenv("BINANCE_API_KEY"),
    "api_secret": os.getenv("BINANCE_API_SECRET"),
    "testnet": True,  # Mettre à False pour le trading réel
}

# Configuration des modèles LLM
LLM_CONFIG = {
    "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
    "sentiment_model_path": os.getenv(
        "SENTIMENT_MODEL_PATH", "ai_trading/info_retour/models/sentiment"
    ),
    "prediction_model_path": os.getenv(
        "PREDICTION_MODEL_PATH", "ai_trading/info_retour/models/prediction"
    ),
}

# Configuration des réseaux sociaux
TWITTER_CONFIG = {
    "api_key": os.getenv("TWITTER_API_KEY"),
    "api_secret": os.getenv("TWITTER_API_SECRET"),
    "access_token": os.getenv("TWITTER_ACCESS_TOKEN"),
    "access_token_secret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
}

# Configuration de la base de données
DATABASE_CONFIG = {"url": os.getenv("DATABASE_URL")}

# Configuration des paires de trading
TRADING_PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]

# Intervalles de temps pour l'analyse
TIME_FRAMES = {
    "sentiment": "1h",
    "technical": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "prediction": "1h",
}

# Configuration des modèles ML
ML_CONFIG = {
    "model_path": os.getenv("ML_MODEL_PATH", "ai_trading/info_retour/models/trading"),
    "features": [
        "close",
        "volume",
        "rsi",
        "macd",
        "bollinger_bands",
        "sentiment_score",
        "market_trend",
        "volatility",
    ],
    "target": "price_direction",
    "train_split": 0.8,
    "test_split": 0.2,
    "random_state": 42,
}

# Configuration des indicateurs techniques
TECHNICAL_INDICATORS = {
    "RSI": {"timeperiod": 14},
    "MACD": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "BB": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
    "EMA": {"timeperiod": 20},
    "ATR": {"timeperiod": 14},
}

# Configuration du logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {"": {"handlers": ["default"], "level": "INFO", "propagate": True}},
}

EMA_RIBBON_PERIODS = [5, 10, 15, 20, 25, 30, 50]
EMA_GRADIENT_THRESHOLDS = {"bullish": 0.015, "bearish": -0.01}
