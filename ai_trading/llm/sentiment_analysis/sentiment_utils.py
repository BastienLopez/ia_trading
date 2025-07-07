"""
Module utilitaire pour l'analyse de sentiment combinant la visualisation,
le cache et les outils communs.
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import pandas as pd

logger = logging.getLogger(__name__)

class SentimentCache:
    """Gestion centralisée du cache pour les analyses."""

    def __init__(self, cache_dir: str = "ai_trading/info_retour/sentiment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Retourne le chemin du fichier de cache pour une clé donnée."""
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

class SentimentVisualizer:
    """Classe pour la visualisation des analyses de sentiment."""

    def __init__(self, output_dir: str = "ai_trading/visualizations/sentiment"):
        """
        Initialise le visualiseur.

        Args:
            output_dir: Répertoire de sortie pour les graphiques
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sentiment_trends(self, data: pd.DataFrame, filename: str = "sentiment_trends.png"):
        """
        Génère un graphique des tendances de sentiment.

        Args:
            data: DataFrame avec les colonnes 'date' et 'score'
            filename: Nom du fichier de sortie
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x='date', y='score', errorbar=None)
        plt.title("Évolution du sentiment au fil du temps")
        plt.xlabel("Date")
        plt.ylabel("Score de sentiment")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Graphique sauvegardé dans {output_path}")

    def plot_sentiment_distribution(self, data: pd.DataFrame, filename: str = "sentiment_distribution.png"):
        """
        Génère un graphique de distribution des sentiments.

        Args:
            data: DataFrame avec la colonne 'sentiment'
            filename: Nom du fichier de sortie
        """
        plt.figure(figsize=(10, 6))
        sentiment_counts = data['sentiment'].value_counts()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        plt.title("Distribution des sentiments")
        plt.xlabel("Sentiment")
        plt.ylabel("Nombre d'occurrences")
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Graphique sauvegardé dans {output_path}")

def text_hash(text: str) -> str:
    """
    Génère un hash unique pour un texte donné.
    
    Args:
        text: Texte à hasher
        
    Returns:
        Hash du texte en hexadécimal
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_llm_client(model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment") -> Any:
    """
    Retourne un client LLM configuré pour l'analyse de sentiment.

    Args:
        model_name: Nom du modèle à utiliser

    Returns:
        Pipeline de classification de texte
    """
    try:
        return pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
        )
    except Exception as e:
        logger.error(f"Erreur lors de la création du client LLM: {e}")
        return None

def calculate_sentiment_metrics(sentiments: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcule des métriques agrégées à partir d'une liste de sentiments.

    Args:
        sentiments: Liste de dictionnaires contenant les scores de sentiment

    Returns:
        Dictionnaire des métriques calculées
    """
    if not sentiments:
        return {
            "average_score": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 0.0
        }

    total = len(sentiments)
    scores = [s.get("score", 0.0) for s in sentiments]
    labels = [s.get("label", "neutral").lower() for s in sentiments]

    return {
        "average_score": sum(scores) / total,
        "positive_ratio": labels.count("positive") / total,
        "negative_ratio": labels.count("negative") / total,
        "neutral_ratio": labels.count("neutral") / total
    } 