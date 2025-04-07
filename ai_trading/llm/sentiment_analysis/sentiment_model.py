"""
Module d'analyse de sentiment pour le trading de cryptomonnaies.
Utilise un modèle de langage pour analyser les sentiments des textes.
"""

import logging
from typing import Dict, List, Optional, Union
import os

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Classe pour l'analyse de sentiment des textes crypto."""

    def __init__(
        self,
        model_name: str = "finiteautomata/bertweet-base-sentiment-analysis",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: str = "models/sentiment",
    ):
        """
        Initialise l'analyseur de sentiment.

        Args:
            model_name: Nom du modèle à utiliser
            device: Dispositif de calcul ('cuda' ou 'cpu')
            cache_dir: Répertoire de cache pour les modèles
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Chargement du modèle et du tokenizer
        logger.info(f"Chargement du modèle {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device)

        # Pipeline de sentiment
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
        )

        logger.info("Modèle chargé avec succès.")

    def analyze_quick(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        """
        Analyse rapide du sentiment d'un ou plusieurs textes.

        Args:
            texts: Texte ou liste de textes à analyser

        Returns:
            Liste de scores de sentiment normalisés entre -1 et 1
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Analyse des sentiments
            results = self.sentiment_pipeline(texts)

            # Normalisation des scores
            normalized_results = []
            for result in results:
                # Conversion du label en score (-1 pour négatif, 0 pour neutre, 1 pour positif)
                if result["label"] == "POSITIVE":
                    base_score = 1
                elif result["label"] == "NEGATIVE":
                    base_score = -1
                else:
                    base_score = 0

                # Pondération par la confiance
                normalized_score = base_score * result["score"]

                normalized_results.append(
                    {
                        "normalized_score": normalized_score,
                        "label": result["label"],
                        "confidence": result["score"],
                    }
                )

            return normalized_results

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse rapide du sentiment: {e}")
            return [
                {"normalized_score": 0.0, "label": "NEUTRAL", "confidence": 0.0}
            ] * len(texts)

    def analyze_detailed(
        self,
        text: str,
        aspects: List[str] = ["price", "technology", "adoption", "regulation"],
    ) -> Dict:
        """
        Analyse détaillée du sentiment d'un texte avec aspects spécifiques.

        Args:
            text: Texte à analyser
            aspects: Aspects à analyser

        Returns:
            Dictionnaire avec scores de sentiment par aspect
        """
        try:
            # Analyse générale
            general_sentiment = self.analyze_quick(text)[0]

            # Analyse par aspect
            aspect_sentiments = {}
            for aspect in aspects:
                # Création d'une requête spécifique à l'aspect
                aspect_query = f"What is the sentiment regarding {aspect} in: {text}"
                aspect_result = self.analyze_quick(aspect_query)[0]
                aspect_sentiments[aspect] = aspect_result

            return {"general": general_sentiment, "aspects": aspect_sentiments}

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse détaillée du sentiment: {e}")
            return {
                "general": {
                    "normalized_score": 0.0,
                    "label": "NEUTRAL",
                    "confidence": 0.0,
                },
                "aspects": {
                    aspect: {
                        "normalized_score": 0.0,
                        "label": "NEUTRAL",
                        "confidence": 0.0,
                    }
                    for aspect in aspects
                },
            }

    def analyze_batch(
        self, texts: List[str], batch_size: int = 32, detailed: bool = False
    ) -> List[Dict]:
        """
        Analyse un lot de textes de manière efficace.

        Args:
            texts: Liste de textes à analyser
            batch_size: Taille des lots
            detailed: Si True, effectue une analyse détaillée

        Returns:
            Liste des analyses de sentiment
        """
        results = []

        try:
            # Traitement par lots
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                if detailed:
                    batch_results = [self.analyze_detailed(text) for text in batch]
                else:
                    batch_results = self.analyze_quick(batch)

                results.extend(batch_results)

            return results

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse par lots: {e}")
            return [
                {"normalized_score": 0.0, "label": "NEUTRAL", "confidence": 0.0}
            ] * len(texts)

    def aggregate_sentiment(
        self, sentiments: List[Dict], weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Agrège plusieurs scores de sentiment en un seul score.

        Args:
            sentiments: Liste de sentiments à agréger
            weights: Poids optionnels pour la pondération

        Returns:
            Score de sentiment agrégé
        """
        if not sentiments:
            return {"normalized_score": 0.0, "label": "NEUTRAL", "confidence": 0.0}

        try:
            # Normalisation des poids
            if weights is None:
                weights = [1.0] * len(sentiments)
            weights = np.array(weights) / sum(weights)

            # Extraction des scores et confiances
            scores = []
            confidences = []

            for sentiment in sentiments:
                if isinstance(sentiment, dict) and "normalized_score" in sentiment:
                    scores.append(sentiment["normalized_score"])
                    confidences.append(sentiment.get("confidence", 1.0))
                else:
                    scores.append(0.0)
                    confidences.append(0.0)

            # Calcul du score pondéré
            weighted_score = np.average(scores, weights=weights)
            avg_confidence = np.average(confidences, weights=weights)

            # Détermination du label
            if weighted_score > 0.1:
                label = "POSITIVE"
            elif weighted_score < -0.1:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            return {
                "normalized_score": float(weighted_score),
                "label": label,
                "confidence": float(avg_confidence),
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation des sentiments: {e}")
            return {"normalized_score": 0.0, "label": "NEUTRAL", "confidence": 0.0}
