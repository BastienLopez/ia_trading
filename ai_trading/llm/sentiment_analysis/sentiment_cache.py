from ai_trading.config import SENTIMENT_CACHE_DIR
import os

# ... existing code ...

# Supprimer l'ancienne définition de SENTIMENT_CACHE_DIR
# SENTIMENT_CACHE_DIR = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#     "sentiment_cache",
# )
# os.makedirs(SENTIMENT_CACHE_DIR, exist_ok=True)

# ... existing code ...

# Mettre à jour tous les chemins du cache
cache_path = os.path.join(SENTIMENT_CACHE_DIR, f"{symbol}_sentiment_cache.json") 