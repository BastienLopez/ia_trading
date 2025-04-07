"""
Module d'analyse des actualités crypto pour extraire le sentiment et les entités.
Utilise des modèles LLM pour analyser le contenu des actualités.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import re
import json
import tensorflow as tf
from keras import layers, models
from collections import Counter

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importation conditionnelle des bibliothèques LLM
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    from transformers import BertTokenizer, BertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("La bibliothèque 'transformers' n'est pas disponible. Certaines fonctionnalités seront limitées.")
    TRANSFORMERS_AVAILABLE = False

class NewsAnalyzer:
    """Classe pour analyser les actualités crypto et en extraire le sentiment et les entités."""
    
    def __init__(self, sentiment_model: str = "finiteautomata/bertweet-base-sentiment-analysis",
                 entity_model: str = "dslim/bert-base-NER",
                 use_gpu: bool = False, **kwargs):
        """
        Initialise l'analyseur d'actualités.
        
        Args:
            sentiment_model: Modèle à utiliser pour l'analyse de sentiment
            entity_model: Modèle à utiliser pour la reconnaissance d'entités
            use_gpu: Si True, utilise le GPU pour l'inférence
        """
        try:
            self.sentiment_model_name = sentiment_model
            self.entity_model_name = entity_model
            self.use_gpu = use_gpu and TRANSFORMERS_AVAILABLE
            self.device = 0 if self.use_gpu else -1
            
            # Initialisation des modèles
            if TRANSFORMERS_AVAILABLE:
                try:
                    logger.info(f"Chargement du modèle de sentiment: {sentiment_model}")
                    self.sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=sentiment_model,
                        device=self.device
                    )
                    
                    logger.info(f"Chargement du modèle de reconnaissance d'entités: {entity_model}")
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=entity_model,
                        aggregation_strategy="simple",
                        device=self.device
                    )
                    
                    logger.info("Modèles chargés avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors du chargement des modèles: {e}")
                    self.sentiment_pipeline = None
                    self.ner_pipeline = None
            else:
                logger.warning("Fonctionnement en mode dégradé sans modèles LLM")
                self.sentiment_pipeline = None
                self.ner_pipeline = None
            
            # Liste des entités crypto à rechercher
            self.crypto_entities = {
                "BTC": ["bitcoin", "btc", "xbt"],
                "ETH": ["ethereum", "eth"],
                "BNB": ["binance coin", "bnb"],
                "SOL": ["solana", "sol"],
                "ADA": ["cardano", "ada"],
                "XRP": ["ripple", "xrp"],
                "DOGE": ["dogecoin", "doge"],
                "DOT": ["polkadot", "dot"],
                "AVAX": ["avalanche", "avax"],
                "MATIC": ["polygon", "matic"]
            }
        except ImportError as e:
            logger.error(f"Erreur critique de dépendance : {e}")
            raise RuntimeError("Veuillez installer les dépendances avec 'pip install tf-keras'") from e
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyse le sentiment d'un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire contenant le sentiment et le score
        """
        if not text or not isinstance(text, str):
            return {"label": "neutral", "score": 0.5}
        
        # Nettoyage du texte
        text = self._clean_text(text)
        
        # Si le texte est trop court après nettoyage
        if len(text.split()) < 3:
            return {"label": "neutral", "score": 0.5}
        
        # Analyse avec le modèle LLM si disponible
        if self.sentiment_pipeline:
            try:
                # Tronquer le texte si nécessaire pour éviter les dépassements de mémoire
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]
                
                result = self.sentiment_pipeline(text)[0]
                
                # Normalisation des étiquettes
                label = result["label"].lower()
                if label in ["positive", "pos"]:
                    label = "positive"
                elif label in ["negative", "neg"]:
                    label = "negative"
                else:
                    label = "neutral"
                
                return {"label": label, "score": result["score"]}
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de sentiment: {e}")
                return self._fallback_sentiment_analysis(text)
        else:
            # Analyse de repli si le modèle n'est pas disponible
            return self._fallback_sentiment_analysis(text)
    
    def extract_entities(self, text: str) -> Dict[str, List]:
        """
        Extrait les entités d'un texte et retourne un dictionnaire.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire contenant les entités extraites
        """
        if not text or not isinstance(text, str):
            return {"crypto_entities": [], "money_entities": [], "percentage_entities": []}
        
        # Nettoyage du texte
        text = self._clean_text(text)
        
        # Si le texte est trop court après nettoyage
        if len(text.split()) < 3:
            return {"crypto_entities": [], "money_entities": [], "percentage_entities": []}
        
        # Extraction avec le modèle LLM si disponible
        if self.ner_pipeline:
            try:
                # Tronquer le texte si nécessaire
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]
                
                entities = self.ner_pipeline(text)
                
                # Filtrage et formatage des entités
                formatted_entities = []
                for entity in entities:
                    formatted_entity = {
                        "text": entity["word"],
                        "type": entity["entity_group"],
                        "score": entity["score"],
                        "start": entity["start"],
                        "end": entity["end"]
                    }
                    formatted_entities.append(formatted_entity)
                
                # Ajout des entités crypto spécifiques
                crypto_entities = self._extract_crypto_entities(text)
                formatted_entities.extend(crypto_entities)
                
                # Extraction des montants d'argent
                money_pattern = r'\$\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*(?:dollars|usd|€|euros)'
                money_matches = re.finditer(money_pattern, text.lower())
                
                money_entities = []
                for match in money_matches:
                    entity = {
                        "text": match.group(),
                        "type": "MONEY",
                        "score": 1.0,
                        "start": match.start(),
                        "end": match.end()
                    }
                    money_entities.append(entity)
                
                # Extraction des pourcentages
                percentage_pattern = r'\d+(?:[.,]\d+)?\s*%'
                percentage_matches = re.finditer(percentage_pattern, text.lower())
                
                percentage_entities = []
                for match in percentage_matches:
                    entity = {
                        "text": match.group(),
                        "type": "PERCENTAGE",
                        "score": 1.0,
                        "start": match.start(),
                        "end": match.end()
                    }
                    percentage_entities.append(entity)
                
                return {
                    'crypto_entities': crypto_entities,
                    'money_entities': money_entities,
                    'percentage_entities': percentage_entities
                }
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'entités: {e}")
                return self._fallback_entity_extraction(text)
        else:
            # Extraction de repli si le modèle n'est pas disponible
            return self._fallback_entity_extraction(text)
    
    def analyze_news(self, news_data: List[Dict]) -> pd.DataFrame:
        results = []
        for news in news_data:
            title_sentiment = self.analyze_sentiment(news.get("title", ""))
            body_sentiment = self.analyze_sentiment(news.get("body", ""))
            
            # Extraction des entités du titre
            title_entities = self.extract_entities(news.get("title", ""))
            
            # Calcul du sentiment global
            global_score = (title_sentiment['score'] + body_sentiment['score']) / 2
            global_label = "positive" if global_score > 0.5 else "negative" if global_score < 0.5 else "neutral"
            
            results.append({
                **news,
                "title_sentiment": title_sentiment,
                "body_sentiment": body_sentiment,
                "title_entities": title_entities,
                "global_sentiment": {"label": global_label, "score": global_score}
            })
        
        return pd.DataFrame(results)
    
    def analyze_news_dataframe(self, df: pd.DataFrame, 
                              title_col: str = "title", 
                              body_col: Optional[str] = "body") -> pd.DataFrame:
        """
        Analyse un DataFrame d'actualités.
        
        Args:
            df: DataFrame contenant les actualités
            title_col: Nom de la colonne contenant les titres
            body_col: Nom de la colonne contenant le corps des actualités (optionnel)
            
        Returns:
            DataFrame enrichi avec le sentiment et les entités
        """
        if df.empty:
            return df
        
        # Vérification des colonnes
        if title_col not in df.columns:
            logger.error(f"La colonne '{title_col}' n'existe pas dans le DataFrame")
            return df
        
        # Création d'une copie pour éviter de modifier l'original
        enriched_df = df.copy()
        
        # Analyse du sentiment des titres
        enriched_df["title_sentiment"] = enriched_df[title_col].apply(
            lambda x: self.analyze_sentiment(x)
        )
        
        # Extraction des labels et scores de sentiment des titres
        enriched_df["title_sentiment_label"] = enriched_df["title_sentiment"].apply(
            lambda x: x["label"] if isinstance(x, dict) and "label" in x else "neutral"
        )
        enriched_df["title_sentiment_score"] = enriched_df["title_sentiment"].apply(
            lambda x: x["score"] if isinstance(x, dict) and "score" in x else 0.5
        )
        
        # Extraction des entités des titres
        enriched_df["title_entities"] = enriched_df[title_col].apply(
            lambda x: self.extract_entities(x)
        )
        
        # Si le corps des actualités est disponible, l'analyser également
        if body_col and body_col in df.columns:
            # Analyse du sentiment du corps
            enriched_df["body_sentiment"] = enriched_df[body_col].apply(
                lambda x: self.analyze_sentiment(x)
            )
            
            # Extraction des labels et scores de sentiment du corps
            enriched_df["body_sentiment_label"] = enriched_df["body_sentiment"].apply(
                lambda x: x["label"] if isinstance(x, dict) and "label" in x else "neutral"
            )
            enriched_df["body_sentiment_score"] = enriched_df["body_sentiment"].apply(
                lambda x: x["score"] if isinstance(x, dict) and "score" in x else 0.5
            )
            
            # Extraction des entités du corps
            enriched_df["body_entities"] = enriched_df[body_col].apply(
                lambda x: self.extract_entities(x)
            )
            
            # Calcul du sentiment global (moyenne pondérée titre/corps)
            enriched_df["global_sentiment_score"] = 0.7 * enriched_df["title_sentiment_score"] + 0.3 * enriched_df["body_sentiment_score"]
            
            # Détermination du sentiment global
            def get_global_label(row):
                if row["global_sentiment_score"] > 0.6:
                    return "positive"
                elif row["global_sentiment_score"] < 0.4:
                    return "negative"
                else:
                    return "neutral"
            
            enriched_df["global_sentiment_label"] = enriched_df.apply(get_global_label, axis=1)
        else:
            # Si pas de corps, le sentiment du titre est le sentiment global
            enriched_df["global_sentiment_score"] = enriched_df["title_sentiment_score"]
            enriched_df["global_sentiment_label"] = enriched_df["title_sentiment_label"]
        
        return enriched_df
    
    def _analyze_single_news(self, news: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse une seule actualité.
        
        Args:
            news: Actualité à analyser
            
        Returns:
            Actualité enrichie avec le sentiment et les entités
        """
        if not isinstance(news, dict):
            logger.error(f"Format d'actualité invalide: {type(news)}")
            return news
        
        # Création d'une copie pour éviter de modifier l'original
        enriched_news = news.copy()
        
        # Analyse du titre si disponible
        if "title" in news and news["title"]:
            enriched_news["title_sentiment"] = self.analyze_sentiment(news["title"])
            enriched_news["title_entities"] = self.extract_entities(news["title"])
        
        # Analyse du corps si disponible
        if "body" in news and news["body"]:
            enriched_news["body_sentiment"] = self.analyze_sentiment(news["body"])
            enriched_news["body_entities"] = self.extract_entities(news["body"])
            
            # Calcul du sentiment global (moyenne pondérée titre/corps)
            if "title_sentiment" in enriched_news:
                title_score = enriched_news["title_sentiment"]["score"]
                body_score = enriched_news["body_sentiment"]["score"]
                global_score = 0.7 * title_score + 0.3 * body_score
                
                # Détermination du sentiment global
                if global_score > 0.6:
                    global_label = "positive"
                elif global_score < 0.4:
                    global_label = "negative"
                else:
                    global_label = "neutral"
                
                enriched_news["global_sentiment"] = {
                    "label": global_label,
                    "score": global_score
                }
            else:
                enriched_news["global_sentiment"] = enriched_news["body_sentiment"]
        else:
            # Si pas de corps, le sentiment du titre est le sentiment global
            if "title_sentiment" in enriched_news:
                enriched_news["global_sentiment"] = enriched_news["title_sentiment"]
        
        return enriched_news
    
    def _clean_text(self, text: str) -> str:
        """
        Nettoie un texte pour l'analyse.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte nettoyé
        """
        if not isinstance(text, str):
            return ""
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Suppression des caractères spéciaux mais conservation des points et virgules
        text = re.sub(r'[^\w\s.,;]', '', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_crypto_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait les entités crypto spécifiques d'un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste des entités crypto extraites
        """
        if not isinstance(text, str):
            return []
        
        text = text.lower()
        entities = []
        
        for symbol, keywords in self.crypto_entities.items():
            for keyword in keywords:
                # Recherche du mot entier
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    entity = {
                        "text": match.group(),
                        "type": "CRYPTO",
                        "symbol": symbol,
                        "score": 1.0,
                        "start": match.start(),
                        "end": match.end()
                    }
                    entities.append(entity)
        
        return entities
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Analyse de sentiment de repli basée sur des règles simples.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire contenant le sentiment et le score
        """
        # Liste de mots positifs et négatifs
        positive_words = [
            "bullish", "surge", "soar", "gain", "rally", "rise", "up", "high", "growth",
            "positive", "good", "great", "excellent", "amazing", "success", "profit",
            "win", "boom", "breakthrough", "opportunity", "potential", "promising"
        ]
        
        negative_words = [
            "bearish", "crash", "plunge", "drop", "fall", "down", "low", "decline",
            "negative", "bad", "poor", "terrible", "awful", "failure", "loss",
            "lose", "bust", "risk", "danger", "threat", "problem", "concern", "worry"
        ]
        
        # Comptage des mots positifs et négatifs
        text_words = text.lower().split()
        positive_count = sum(1 for word in text_words if word in positive_words)
        negative_count = sum(1 for word in text_words if word in negative_words)
        
        # Calcul du score de sentiment
        total_count = positive_count + negative_count
        if total_count == 0:
            return {"label": "neutral", "score": 0.5}
        
        positive_ratio = positive_count / total_count
        
        # Détermination du sentiment
        if positive_ratio > 0.6:
            label = "positive"
            score = 0.5 + (positive_ratio - 0.5)
        elif positive_ratio < 0.4:
            label = "negative"
            score = 0.5 - (0.5 - positive_ratio)
        else:
            label = "neutral"
            score = 0.5
        
        return {"label": label, "score": score}
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, List]:
        """
        Extraction d'entités de repli basée sur des règles simples.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire contenant les entités extraites
        """
        # Extraction des entités crypto
        crypto_entities = self._extract_crypto_entities(text)
        
        # Extraction des montants d'argent
        money_pattern = r'\$\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*(?:dollars|usd|€|euros)'
        money_matches = re.finditer(money_pattern, text.lower())
        
        money_entities = []
        for match in money_matches:
            entity = {
                "text": match.group(),
                "type": "MONEY",
                "score": 1.0,
                "start": match.start(),
                "end": match.end()
            }
            money_entities.append(entity)
        
        # Extraction des pourcentages
        percentage_pattern = r'\d+(?:[.,]\d+)?\s*%'
        percentage_matches = re.finditer(percentage_pattern, text.lower())
        
        percentage_entities = []
        for match in percentage_matches:
            entity = {
                "text": match.group(),
                "type": "PERCENTAGE",
                "score": 1.0,
                "start": match.start(),
                "end": match.end()
            }
            percentage_entities.append(entity)
        
        return {
            'crypto_entities': crypto_entities,
            'money_entities': money_entities,
            'percentage_entities': percentage_entities
        }

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Génère un rapport d'analyse."""
        return {
            'top_cryptos': self._get_top_cryptos(df),
            'sentiment_distribution': self._get_sentiment_distribution(df)
        }

    def _get_top_cryptos(self, df: pd.DataFrame) -> List[Dict]:
        """Retourne les cryptos les plus mentionnées."""
        crypto_counts = Counter()
        for entities in df['title_entities']:
            crypto_counts.update([e['text'] for e in entities['crypto_entities']])
        return crypto_counts.most_common(5)

    def _get_sentiment_distribution(self, df: pd.DataFrame) -> Dict:
        """Calcule la distribution des sentiments."""
        return df['global_sentiment'].apply(lambda x: x['label']).value_counts().to_dict()

    def plot_trends(self, df: pd.DataFrame, filename: str) -> None:
        """Génère un graphique des tendances de sentiment."""
        try:
            # Utiliser un backend non-interactif
            import matplotlib
            matplotlib.use('Agg')  # Backend qui ne nécessite pas Tkinter
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            
            if 'published_at' in df.columns and 'global_sentiment' in df.columns:
                df['date'] = pd.to_datetime(df['published_at'])
                df.set_index('date')['global_sentiment'].apply(
                    lambda x: x['score'] if isinstance(x, dict) else 0.5
                ).plot(title='Évolution du sentiment global')
                plt.tight_layout()
                plt.savefig(filename)
                plt.close()
            else:
                logger.warning("Données insuffisantes pour générer le graphique")
        except ImportError:
            logger.error("Matplotlib non installé. Impossible de générer le graphique")

class EnhancedNewsAnalyzer(NewsAnalyzer):
    """Version améliorée de l'analyseur avec fonctionnalités supplémentaires."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = kwargs.get('cache_dir', 'sentiment_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def analyze_news_batch(self, news_list: List[Dict], batch_size: int = 32) -> pd.DataFrame:
        """Analyse par lots avec mise en cache."""
        cached_results = []
        for news in news_list:
            cache_key = self._generate_cache_key(news)
            cached = self._load_from_cache(cache_key)
            if cached:
                cached_results.append(cached)
        
        uncached_news = [news for news in news_list if not self._is_cached(news)]
        for i in range(0, len(uncached_news), batch_size):
            batch = uncached_news[i:i+batch_size]
            analyzed_batch = super().analyze_news(batch)
            for _, row in analyzed_batch.iterrows():
                self._save_to_cache(row.to_dict())
            cached_results.extend(analyzed_batch.to_dict('records'))
        
        return pd.DataFrame(cached_results)
    
    def _generate_cache_key(self, news: Dict) -> str:
        """Génère une clé de cache unique pour une actualité."""
        return f"{hash(news.get('title', '') + news.get('body', ''))}.json"
    
    def _is_cached(self, news: Dict) -> bool:
        """Vérifie si une actualité est déjà en cache."""
        return os.path.exists(os.path.join(self.cache_dir, self._generate_cache_key(news)))
    
    def _save_to_cache(self, data: Dict):
        """Sauvegarde les résultats dans le cache."""
        cache_path = os.path.join(self.cache_dir, self._generate_cache_key(data))
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Charge les résultats depuis le cache."""
        cache_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

# Exporter les classes et fonctions
__all__ = ['EnhancedNewsAnalyzer', 'NewsAnalyzer']

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation de l'analyseur
    analyzer = NewsAnalyzer()
    
    # Exemple d'actualités
    news_examples = [
        {
            "title": "Bitcoin Surges to $60,000 as Institutional Adoption Grows",
            "body": "Bitcoin reached a new all-time high of $60,000 today as more institutional investors are entering the cryptocurrency market. Major companies like Tesla and MicroStrategy have added BTC to their balance sheets."
        },
        {
            "title": "Ethereum Price Drops 10% Following Network Congestion",
            "body": "Ethereum (ETH) experienced a significant price drop of 10% in the last 24 hours due to network congestion and high gas fees. Developers are working on solutions to address these scaling issues."
        }
    ]
    
    # Analyse des actualités
    enriched_news = analyzer.analyze_news(news_examples)
    
    # Affichage des résultats
    for i, news in enumerate(enriched_news):
        print(f"\nActualité {i+1}:")
        print(f"Titre: {news['title']}")
        print(f"Sentiment du titre: {news['title_sentiment']}")
        print(f"Entités du titre: {news['title_entities']}")
        print(f"Sentiment global: {news['global_sentiment']}") 