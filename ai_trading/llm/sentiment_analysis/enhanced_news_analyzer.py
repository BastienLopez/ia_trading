"""
Module amélioré d'analyse des actualités crypto pour extraire le sentiment et les entités.
Utilise des modèles LLM avancés et des techniques de NLP pour une analyse plus complète.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import re
import json
import hashlib
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importation conditionnelle des bibliothèques LLM
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("La bibliothèque 'transformers' n'est pas disponible. Certaines fonctionnalités seront limitées.")
    TRANSFORMERS_AVAILABLE = False

class EnhancedNewsAnalyzer:
    """Classe améliorée pour analyser les actualités crypto et en extraire le sentiment et les entités."""
    
    def __init__(self, 
                 sentiment_model: str = "finiteautomata/bertweet-base-sentiment-analysis",
                 entity_model: str = "dslim/bert-base-NER",
                 use_gpu: bool = False,
                 cache_dir: str = "data/cache",
                 enable_cache: bool = True):
        """
        Initialise l'analyseur d'actualités amélioré.
        
        Args:
            sentiment_model: Modèle à utiliser pour l'analyse de sentiment
            entity_model: Modèle à utiliser pour la reconnaissance d'entités
            use_gpu: Si True, utilise le GPU pour l'inférence
            cache_dir: Répertoire pour stocker le cache des analyses
            enable_cache: Si True, active le cache pour éviter de réanalyser les mêmes textes
        """
        self.sentiment_model_name = sentiment_model
        self.entity_model_name = entity_model
        self.use_gpu = use_gpu and TRANSFORMERS_AVAILABLE
        self.device = 0 if self.use_gpu else -1
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        
        # Création du répertoire de cache si nécessaire
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.sentiment_cache_file = os.path.join(self.cache_dir, "sentiment_cache.pkl")
            self.entity_cache_file = os.path.join(self.cache_dir, "entity_cache.pkl")
            
            # Chargement du cache existant
            self.sentiment_cache = self._load_cache(self.sentiment_cache_file)
            self.entity_cache = self._load_cache(self.entity_cache_file)
        else:
            self.sentiment_cache = {}
            self.entity_cache = {}
        
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
            "MATIC": ["polygon", "matic"],
            "LINK": ["chainlink", "link"],
            "UNI": ["uniswap", "uni"],
            "ATOM": ["cosmos", "atom"],
            "LTC": ["litecoin", "ltc"],
            "ALGO": ["algorand", "algo"]
        }
        
        # Dictionnaires de mots positifs et négatifs pour l'analyse de sentiment de repli
        self.positive_words = [
            "bullish", "surge", "soar", "rally", "gain", "rise", "increase", "growth", "profit", 
            "success", "positive", "optimistic", "promising", "potential", "opportunity", "adoption", 
            "breakthrough", "innovation", "support", "strong", "confidence", "secure", "stability"
        ]
        
        self.negative_words = [
            "bearish", "crash", "plunge", "drop", "fall", "decline", "decrease", "loss", "risk", 
            "danger", "negative", "pessimistic", "concerning", "threat", "problem", "issue", "fear", 
            "uncertainty", "volatility", "weak", "failure", "scam", "hack", "vulnerability", "attack"
        ]
    
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
        
        # Vérification du cache
        if self.enable_cache:
            text_hash = self._hash_text(text)
            if text_hash in self.sentiment_cache:
                return self.sentiment_cache[text_hash]
        
        # Analyse avec le modèle si disponible
        if TRANSFORMERS_AVAILABLE and self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text)[0]
                
                # Normalisation du résultat
                if result["label"] == "POSITIVE":
                    label = "positive"
                    score = result["score"]
                elif result["label"] == "NEGATIVE":
                    label = "negative"
                    score = 1 - result["score"]
                else:
                    label = "neutral"
                    score = 0.5
                
                sentiment = {"label": label, "score": score}
                
                # Mise à jour du cache
                if self.enable_cache:
                    self.sentiment_cache[text_hash] = sentiment
                    self._save_cache(self.sentiment_cache, self.sentiment_cache_file)
                
                return sentiment
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de sentiment: {e}")
                # Utilisation de la méthode de repli en cas d'erreur
                return self._fallback_sentiment_analysis(text)
        else:
            # Utilisation de la méthode de repli si le modèle n'est pas disponible
            return self._fallback_sentiment_analysis(text)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait les entités d'un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste des entités extraites
        """
        if not text or not isinstance(text, str):
            return []
        
        # Nettoyage du texte
        text = self._clean_text(text)
        
        # Si le texte est trop court après nettoyage
        if len(text.split()) < 3:
            return []
        
        # Vérification du cache
        if self.enable_cache:
            text_hash = self._hash_text(text)
            if text_hash in self.entity_cache:
                return self.entity_cache[text_hash]
        
        # Extraction avec le modèle si disponible
        if TRANSFORMERS_AVAILABLE and self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                
                # Traitement des résultats
                entities = []
                for entity in ner_results:
                    entity_info = {
                        "text": entity["word"],
                        "type": entity["entity_group"],
                        "score": entity["score"],
                        "start": entity["start"],
                        "end": entity["end"]
                    }
                    entities.append(entity_info)
                
                # Ajout des entités crypto
                crypto_entities = self._extract_crypto_entities(text)
                entities.extend(crypto_entities)
                
                # Mise à jour du cache
                if self.enable_cache:
                    self.entity_cache[text_hash] = entities
                    self._save_cache(self.entity_cache, self.entity_cache_file)
                
                return entities
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'entités: {e}")
                # Utilisation de la méthode de repli en cas d'erreur
                return self._fallback_entity_extraction(text)
        else:
            # Utilisation de la méthode de repli si le modèle n'est pas disponible
            return self._fallback_entity_extraction(text)
    
    def analyze_news(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyse une liste d'actualités.
        
        Args:
            news_list: Liste d'actualités à analyser
            
        Returns:
            Liste d'actualités enrichies avec l'analyse de sentiment et d'entités
        """
        if not news_list:
            return []
        
        enriched_news = []
        
        for news in news_list:
            enriched_item = news.copy()
            
            # Analyse du titre
            if "title" in news and news["title"]:
                title_sentiment = self.analyze_sentiment(news["title"])
                title_entities = self.extract_entities(news["title"])
                
                enriched_item["title_sentiment"] = title_sentiment
                enriched_item["title_entities"] = title_entities
            
            # Analyse du corps
            if "body" in news and news["body"]:
                body_sentiment = self.analyze_sentiment(news["body"])
                body_entities = self.extract_entities(news["body"])
                
                enriched_item["body_sentiment"] = body_sentiment
                enriched_item["body_entities"] = body_entities
            
            # Calcul du sentiment global
            if "title_sentiment" in enriched_item and "body_sentiment" in enriched_item:
                title_weight = 0.4
                body_weight = 0.6
                
                title_score = enriched_item["title_sentiment"]["score"]
                body_score = enriched_item["body_sentiment"]["score"]
                
                global_score = title_score * title_weight + body_score * body_weight
                
                if global_score > 0.6:
                    global_label = "positive"
                elif global_score < 0.4:
                    global_label = "negative"
                else:
                    global_label = "neutral"
                
                enriched_item["global_sentiment"] = {
                    "label": global_label,
                    "score": global_score
                }
            
            # Extraction des cryptomonnaies mentionnées
            mentioned_cryptos = set()
            
            if "title_entities" in enriched_item:
                for entity in enriched_item["title_entities"]:
                    if entity.get("type") == "CRYPTO" and "symbol" in entity:
                        mentioned_cryptos.add(entity["symbol"])
            
            if "body_entities" in enriched_item:
                for entity in enriched_item["body_entities"]:
                    if entity.get("type") == "CRYPTO" and "symbol" in entity:
                        mentioned_cryptos.add(entity["symbol"])
            
            enriched_item["mentioned_cryptos"] = list(mentioned_cryptos)
            
            enriched_news.append(enriched_item)
        
        return enriched_news
    
    def analyze_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse un DataFrame d'actualités.
        
        Args:
            df: DataFrame contenant les actualités
            
        Returns:
            DataFrame enrichi avec l'analyse de sentiment et d'entités
        """
        if df.empty:
            return df
        
        enriched_df = df.copy()
        
        # Analyse des titres
        if "title" in df.columns:
            logger.info("Analyse des titres...")
            title_sentiments = []
            title_entities = []
            
            for title in df["title"]:
                sentiment = self.analyze_sentiment(title)
                entities = self.extract_entities(title)
                
                title_sentiments.append(sentiment)
                title_entities.append(entities)
            
            enriched_df["title_sentiment_label"] = [s["label"] for s in title_sentiments]
            enriched_df["title_sentiment_score"] = [s["score"] for s in title_sentiments]
            enriched_df["title_entities"] = title_entities

        # Analyse des corps
        if "body" in df.columns:
            logger.info("Analyse des corps...")
            body_sentiments = []
            body_entities = []
            
            for body in df["body"]:
                sentiment = self.analyze_sentiment(body)
                entities = self.extract_entities(body)
                
                body_sentiments.append(sentiment)
                body_entities.append(entities)
            
            enriched_df["body_sentiment_label"] = [s["label"] for s in body_sentiments]
            enriched_df["body_sentiment_score"] = [s["score"] for s in body_sentiments]
            enriched_df["body_entities"] = body_entities
        
        # Calcul du sentiment global
        if "title_sentiment_score" in enriched_df.columns and "body_sentiment_score" in enriched_df.columns:
            logger.info("Calcul du sentiment global...")
            title_weight = 0.4
            body_weight = 0.6
            
            enriched_df["global_sentiment_score"] = (
                enriched_df["title_sentiment_score"] * title_weight +
                enriched_df["body_sentiment_score"] * body_weight
            )
            
            # Détermination du label global
            def get_global_label(score):
                if score > 0.6:
                    return "positive"
                elif score < 0.4:
                    return "negative"
                else:
                    return "neutral"
            
            enriched_df["global_sentiment_label"] = enriched_df["global_sentiment_score"].apply(get_global_label)
        
        # Extraction des cryptomonnaies mentionnées
        logger.info("Extraction des cryptomonnaies mentionnées...")
        mentioned_cryptos = []
        
        if "title_entities" in enriched_df.columns and "body_entities" in enriched_df.columns:
            for i, row in enriched_df.iterrows():
                cryptos = []
                
                # Extraction depuis le titre
                for entity in row["title_entities"]:
                    if entity.get("type") == "CRYPTO" and "symbol" in entity:
                        cryptos.append(entity["symbol"])
                
                # Extraction depuis le corps
                for entity in row["body_entities"]:
                    if entity.get("type") == "CRYPTO" and "symbol" in entity:
                        cryptos.append(entity["symbol"])
                
                # Suppression des doublons
                mentioned_cryptos.append(list(set(cryptos)))
            
            enriched_df["mentioned_cryptos"] = mentioned_cryptos
        
        return enriched_df
    
    def generate_sentiment_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Génère un rapport complet sur le sentiment des actualités.
        
        Args:
            df: DataFrame contenant les actualités analysées
            
        Returns:
            Dictionnaire contenant le rapport de sentiment
        """
        report = {
            "total_articles": len(df),
            "sentiment_distribution": {},
            "crypto_mentions": {},
            "most_positive_article": {},
            "most_negative_article": {},
            "average_sentiment": 0.0,
            "sentiment_by_crypto": {},
            "generated_at": datetime.now().isoformat()
        }
        
        if df.empty:
            return report
        
        # Distribution des sentiments
        if "global_sentiment_label" in df.columns:
            sentiment_counts = df["global_sentiment_label"].value_counts().to_dict()
            report["sentiment_distribution"] = sentiment_counts
        
        # Mentions de cryptomonnaies
        if "mentioned_cryptos" in df.columns:
            all_mentions = []
            for mentions in df["mentioned_cryptos"]:
                if isinstance(mentions, list):
                    all_mentions.extend(mentions)
            
            mention_counts = Counter(all_mentions)
            report["crypto_mentions"] = dict(mention_counts.most_common(10))
        
        # Articles les plus positifs et négatifs
        if "global_sentiment_score" in df.columns and "title" in df.columns:
            if not df.empty:
                most_positive_idx = df["global_sentiment_score"].idxmax()
                most_negative_idx = df["global_sentiment_score"].idxmin()
                
                report["most_positive_article"] = {
                    "title": df.loc[most_positive_idx, "title"],
                    "score": float(df.loc[most_positive_idx, "global_sentiment_score"]),
                    "url": df.loc[most_positive_idx, "url"] if "url" in df.columns else None
                }
                
                report["most_negative_article"] = {
                    "title": df.loc[most_negative_idx, "title"],
                    "score": float(df.loc[most_negative_idx, "global_sentiment_score"]),
                    "url": df.loc[most_negative_idx, "url"] if "url" in df.columns else None
                }
        
        # Sentiment moyen
        if "global_sentiment_score" in df.columns:
            report["average_sentiment"] = float(df["global_sentiment_score"].mean())
        
        # Sentiment par cryptomonnaie
        if "mentioned_cryptos" in df.columns and "global_sentiment_score" in df.columns:
            crypto_sentiments = {}
            
            for crypto in set(all_mentions) if 'all_mentions' in locals() else []:
                # Filtrer les articles mentionnant cette crypto
                crypto_articles = df[df["mentioned_cryptos"].apply(lambda x: crypto in x if isinstance(x, list) else False)]
                
                if not crypto_articles.empty:
                    avg_sentiment = float(crypto_articles["global_sentiment_score"].mean())
                    count = len(crypto_articles)
                    
                    crypto_sentiments[crypto] = {
                        "average_sentiment": avg_sentiment,
                        "article_count": count,
                        "sentiment_label": "positive" if avg_sentiment > 0.6 else "negative" if avg_sentiment < 0.4 else "neutral"
                    }
            
            report["sentiment_by_crypto"] = crypto_sentiments
        
        # Génération des visualisations
        try:
            self.visualize_sentiment_distribution(df)
            self.visualize_sentiment_trend(df)
            self.visualize_crypto_mentions(df)
            report["visualizations_generated"] = True
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations: {e}")
            report["visualizations_generated"] = False
        
        # Sauvegarde du rapport au format JSON
        try:
            os.makedirs("data/sentiment/reports", exist_ok=True)
            report_file = f"data/sentiment/reports/sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, "w") as f:
                json.dump(report, f, indent=4)
            
            report["report_file"] = report_file
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du rapport: {e}")
        
        return report
    
    def visualize_sentiment_distribution(self, df: pd.DataFrame, 
                                        sentiment_col: str = "global_sentiment_label",
                                        title: str = "Distribution des sentiments") -> None:
        """
        Visualise la distribution des sentiments.
        
        Args:
            df: DataFrame contenant les sentiments
            sentiment_col: Nom de la colonne contenant les labels de sentiment
            title: Titre du graphique
        """
        if sentiment_col not in df.columns:
            logger.error(f"La colonne '{sentiment_col}' n'existe pas dans le DataFrame")
            return
        
        # Comptage des sentiments
        sentiment_counts = df[sentiment_col].value_counts()
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Définition des couleurs
            colors = {
                "positive": "green",
                "neutral": "gray",
                "negative": "red"
            }
            
            # Création du graphique
            ax = sentiment_counts.plot(kind='bar', color=[colors.get(x, 'blue') for x in sentiment_counts.index])
            
            # Ajout des valeurs sur les barres
            for i, v in enumerate(sentiment_counts):
                ax.text(i, v + 0.1, str(v), ha='center')
            
            plt.title(title)
            plt.xlabel("Sentiment")
            plt.ylabel("Nombre d'articles")
            plt.xticks(rotation=0)
            plt.tight_layout()
            
            # Sauvegarde du graphique
            os.makedirs("data/sentiment/visualizations", exist_ok=True)
            plt.savefig("data/sentiment/visualizations/sentiment_distribution.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Impossible de générer la visualisation: {e}")
            logger.info("Distribution des sentiments (format texte):")
            for sentiment, count in sentiment_counts.items():
                logger.info(f"- {sentiment}: {count}")
    
    def visualize_sentiment_trend(self, df: pd.DataFrame, 
                                 date_col: str = "published_at",
                                 sentiment_col: str = "global_sentiment_score",
                                 title: str = "Évolution du sentiment au fil du temps") -> None:
        """
        Visualise l'évolution du sentiment au fil du temps.
        
        Args:
            df: DataFrame contenant les sentiments et les dates
            date_col: Nom de la colonne contenant les dates
            sentiment_col: Nom de la colonne contenant les scores de sentiment
            title: Titre du graphique
        """
        if date_col not in df.columns or sentiment_col not in df.columns:
            logger.error(f"Les colonnes '{date_col}' ou '{sentiment_col}' n'existent pas dans le DataFrame")
            return
        
        # Conversion des dates si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                logger.error(f"Erreur lors de la conversion des dates: {e}")
                return
        
        # Tri par date
        df_sorted = df.sort_values(by=date_col)
        
        # Calcul de la moyenne quotidienne
        daily_sentiment = df_sorted.groupby(df_sorted[date_col].dt.date)[sentiment_col].mean()
        
        plt.figure(figsize=(12, 6))
        
        # Création du graphique
        plt.plot(daily_sentiment.index, daily_sentiment.values, marker='o', linestyle='-', color='blue')
        
        # Ajout de zones colorées pour les sentiments positifs/négatifs
        plt.fill_between(daily_sentiment.index, daily_sentiment.values, 0.5, 
                         where=(daily_sentiment.values >= 0.5),
                         color='green', alpha=0.3)
        
        plt.fill_between(daily_sentiment.index, daily_sentiment.values, 0.5, 
                         where=(daily_sentiment.values < 0.5),
                         color='red', alpha=0.3)
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Score de sentiment moyen")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarde du graphique
        os.makedirs("data/sentiment/visualizations", exist_ok=True)
        plt.savefig("data/sentiment/visualizations/sentiment_trend.png")
        plt.close()
    
    def visualize_crypto_mentions(self, df: pd.DataFrame, 
                                 crypto_col: str = "mentioned_cryptos",
                                 title: str = "Cryptomonnaies les plus mentionnées") -> None:
        """
        Visualise les cryptomonnaies les plus mentionnées.
        
        Args:
            df: DataFrame contenant les mentions de cryptomonnaies
            crypto_col: Nom de la colonne contenant les listes de cryptomonnaies
            title: Titre du graphique
        """
        if crypto_col not in df.columns:
            logger.error(f"La colonne '{crypto_col}' n'existe pas dans le DataFrame")
            return
        
        # Comptage des mentions
        all_mentions = []
        for mentions in df[crypto_col]:
            if isinstance(mentions, list):
                all_mentions.extend(mentions)
        
        if not all_mentions:
            logger.warning("Aucune mention de cryptomonnaie trouvée")
            return
        
        mention_counts = Counter(all_mentions)
        
        plt.figure(figsize=(12, 6))
        
        # Création du graphique
        cryptos = [crypto for crypto, _ in mention_counts.most_common(10)]
        counts = [count for _, count in mention_counts.most_common(10)]
        
        ax = sns.barplot(x=cryptos, y=counts)
        
        # Ajout des valeurs sur les barres
        for i, v in enumerate(counts):
            ax.text(i, v + 0.1, str(v), ha='center')
        
        plt.title(title)
        plt.xlabel("Cryptomonnaie")
        plt.ylabel("Nombre de mentions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarde du graphique
        os.makedirs("data/sentiment/visualizations", exist_ok=True)
        plt.savefig("data/sentiment/visualizations/crypto_mentions.png")
        plt.close()
    
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
        
        # Suppression des URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Suppression des balises HTML
        text = re.sub(r'<.*?>', '', text)
        
        # Suppression des caractères spéciaux et normalisation des espaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _hash_text(self, text: str) -> str:
        """
        Génère un hash pour un texte.
        
        Args:
            text: Texte à hasher
            
        Returns:
            Hash du texte
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self, cache_file: str) -> Dict[str, Any]:
        """
        Charge un cache depuis un fichier.
        
        Args:
            cache_file: Chemin vers le fichier de cache
            
        Returns:
            Dictionnaire contenant le cache
        """
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du cache: {e}")
        
        return {}
    
    def _save_cache(self, cache: Dict[str, Any], cache_file: str) -> None:
        """
        Sauvegarde un cache dans un fichier.
        
        Args:
            cache: Dictionnaire contenant le cache
            cache_file: Chemin vers le fichier de cache
        """
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde du cache: {e}")
    
    def _extract_crypto_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait les entités crypto d'un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste des entités crypto extraites
        """
        if not text or not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        entities = []
        
        # Recherche des cryptomonnaies dans le texte
        for symbol, names in self.crypto_entities.items():
            for name in names:
                # Recherche du nom complet ou du symbole
                pattern = r'\b' + re.escape(name) + r'\b'
                matches = list(re.finditer(pattern, text_lower))
                
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
        Méthode de repli pour l'analyse de sentiment lorsque le modèle n'est pas disponible.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire contenant le sentiment et le score
        """
        if not text or not isinstance(text, str):
            return {"label": "neutral", "score": 0.5}
        
        text_lower = text.lower()
        
        # Comptage des mots positifs et négatifs
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return {"label": "neutral", "score": 0.5}
        
        # Calcul du score
        score = positive_count / total_count
        
        # Détermination du label
        if score > 0.6:
            label = "positive"
        elif score < 0.4:
            label = "negative"
        else:
            label = "neutral"
        
        return {"label": label, "score": score}
    
    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Méthode de repli pour l'extraction d'entités lorsque le modèle n'est pas disponible.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste des entités extraites
        """
        # Extraction des entités crypto
        crypto_entities = self._extract_crypto_entities(text)
        
        # Extraction des montants d'argent
        money_pattern = r'\$\s*\d+(?:[.,]\d+)?(?:\s*[kmbt])?|\d+(?:[.,]\d+)?\s*(?:dollars|usd|€|euros)'
        money_matches = re.finditer(money_pattern, text.lower())
        
        for match in money_matches:
            entity = {
                "text": match.group(),
                "type": "MONEY",
                "score": 1.0,
                "start": match.start(),
                "end": match.end()
            }
            crypto_entities.append(entity)
        
        # Extraction des pourcentages
        percentage_pattern = r'\d+(?:[.,]\d+)?\s*%'
        percentage_matches = re.finditer(percentage_pattern, text.lower())
        
        for match in percentage_matches:
            entity = {
                "text": match.group(),
                "type": "PERCENTAGE",
                "score": 1.0,
                "start": match.start(),
                "end": match.end()
            }
            crypto_entities.append(entity)
        
        return crypto_entities

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation de l'analyseur
    analyzer = EnhancedNewsAnalyzer()
    
    # Exemple d'actualités
    news_examples = [
        {
            "title": "Bitcoin Surges to $60,000 as Institutional Adoption Grows",
            "body": "Bitcoin reached a new all-time high of $60,000 today as more institutional investors are entering the cryptocurrency market. Major companies like Tesla and MicroStrategy have added BTC to their balance sheets.",
            "published_at": "2023-03-15T12:30:00Z"
        },
        {
            "title": "Ethereum Price Drops 10% Following Network Congestion",
            "body": "Ethereum (ETH) experienced a significant price drop of 10% in the last 24 hours due to network congestion and high gas fees. Developers are working on solutions to address these scaling issues.",
            "published_at": "2023-03-14T09:15:00Z"
        },
        {
            "title": "Solana Ecosystem Expands with New DeFi Projects",
            "body": "The Solana blockchain is seeing rapid growth in its DeFi ecosystem with several new projects launching this month. The total value locked (TVL) in Solana DeFi has increased by 25% in the past week.",
            "published_at": "2023-03-13T16:45:00Z"
        }
    ]
    
    # Analyse des actualités
    enriched_news = analyzer.analyze_news(news_examples)
    
    # Conversion en DataFrame
    news_df = pd.DataFrame(enriched_news)
    
    # Génération du rapport
    report = analyzer.generate_sentiment_report(news_df)
    
    # Affichage des résultats
    print("\nRapport d'analyse de sentiment:")
    print(f"Total d'articles: {report['total_articles']}")
    print(f"Distribution des sentiments: {report['sentiment_distribution']}")
    print(f"Sentiment moyen: {report['average_sentiment']:.2f}")
    print("\nCryptomonnaies les plus mentionnées:")
    for crypto, count in report['crypto_mentions'].items():
        print(f"- {crypto}: {count} mentions")
    
    print("\nArticle le plus positif:")
    print(f"- {report['most_positive_article']['title']}")
    print(f"- Score: {report['most_positive_article']['score']:.2f}")
    
    print("\nArticle le plus négatif:")
    print(f"- {report['most_negative_article']['title']}")
    print(f"- Score: {report['most_negative_article']['score']:.2f}")
    
    print("\nSentiment par cryptomonnaie:")
    for crypto, data in report['sentiment_by_crypto'].items():
        print(f"- {crypto}: {data['sentiment_label']} ({data['average_sentiment']:.2f}) - {data['article_count']} articles") 