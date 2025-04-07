"""Utilitaires communs pour l'analyse de sentiment."""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SentimentCache:
    """Gestion centralisée du cache pour les analyses."""

    def __init__(self, cache_dir: str = "data/sentiment/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def load(self, key: str) -> Any:
        """Charge un élément du cache."""
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None

        try:
            with cache_file.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Erreur de chargement du cache {key}: {e}")
            return None

    def save(self, key: str, data: Any) -> None:
        """Sauvegarde un élément dans le cache."""
        cache_file = self._get_cache_path(key)
        try:
            with cache_file.open("wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Erreur de sauvegarde du cache {key}: {e}")


def text_hash(text: str) -> str:
    """Génère un hash unique pour un texte."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
