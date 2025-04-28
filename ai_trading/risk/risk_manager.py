from ai_trading.config import VISUALIZATION_DIR

# Utiliser directement VISUALIZATION_DIR de config.py
VISUALIZATION_DIR = VISUALIZATION_DIR / "risk"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
