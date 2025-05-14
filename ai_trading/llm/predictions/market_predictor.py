"""
Module de prédiction de marché basé sur les LLM.

Ce module fournit des fonctionnalités pour générer des prédictions
de direction du marché en utilisant les grands modèles de langage.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import torch
from openai import OpenAI

import ai_trading.config as config
from ai_trading.llm.predictions.cache_manager import CacheManager, cached
from ai_trading.llm.predictions.performance_profiler import profile
from ai_trading.llm.predictions.rtx_optimizer import (
    RTXOptimizer,
    detect_rtx_gpu,
    setup_rtx_environment,
)

# Importations internes
from ai_trading.llm.sentiment_analysis.news_analyzer import NewsAnalyzer
from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer
from ai_trading.utils import setup_logger

# Configuration du logger
logger = setup_logger("market_predictor")

# Configuration initiale de l'environnement GPU
rtx_gpu_info = None
if torch.cuda.is_available():
    # Vérifier d'abord si un GPU RTX est disponible
    rtx_gpu_info = detect_rtx_gpu()
    if rtx_gpu_info:
        # Optimisation spécifique pour RTX
        setup_rtx_environment()
        logger.info(
            f"Environnement GPU RTX {rtx_gpu_info['series']} configuré pour les prédictions de marché"
        )
    else:
        logger.info(
            "Aucun GPU RTX détecté, exécution standard pour les prédictions de marché"
        )
else:
    logger.info("Aucun GPU détecté, exécution en mode CPU")


class MarketPredictor:
    """
    Classe pour générer des prédictions de marché basées sur les LLM.

    Cette classe utilise des grands modèles de langage pour analyser les données
    de marché et de sentiment afin de produire des prédictions sur les tendances futures.
    """

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialise le prédicteur de marché.

        Args:
            custom_config: Configuration personnalisée optionnelle
        """
        # Utilisation de la configuration du projet
        self.config = custom_config or {}

        # Configuration LLM avec valeurs par défaut
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 1000)

        # Initialisation des analyseurs de sentiment
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialAnalyzer()

        # Client LLM avec la clé API de config
        self.client = OpenAI(api_key=config.LLM_CONFIG.get("openai_api_key"))

        # Historique des prédictions
        self.predictions_history = {}

        # Initialisation du cache
        cache_dir = self.config.get(
            "cache_dir", str(config.DATA_DIR / "cache" / "predictions")
        )
        self.cache = CacheManager(
            capacity=self.config.get("cache_capacity", 100),
            ttl=self.config.get("cache_ttl", 3600),  # 1 heure par défaut
            persist_path=cache_dir,
            enable_disk_cache=self.config.get("enable_disk_cache", True),
        )

        # Initialisation de l'optimiseur RTX si disponible
        use_gpu = self.config.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            # Vérification si RTX est disponible
            if rtx_gpu_info:
                # Utiliser l'optimiseur RTX spécifique
                self.rtx_optimizer = RTXOptimizer(
                    device_id=self.config.get("gpu_device_id", None),
                    enable_tensor_cores=self.config.get("enable_tensor_cores", True),
                    enable_half_precision=self.config.get(
                        "enable_half_precision", True
                    ),
                    optimize_memory=self.config.get("optimize_memory", True),
                    enable_tensorrt=self.config.get("enable_tensorrt", False),
                )
                logger.info(
                    f"Accélération GPU RTX activée: {self.rtx_optimizer.get_optimization_info()}"
                )
            else:
                # Pas d'optimiseur RTX
                self.rtx_optimizer = None
                logger.info("Support GPU standard activé (non RTX)")
        else:
            self.rtx_optimizer = None
            if use_gpu:
                logger.info("Accélération GPU demandée mais aucun GPU disponible")
            else:
                logger.info("Accélération GPU désactivée par configuration")

        logger.info("MarketPredictor initialisé avec modèle: %s", self.model_name)

    @profile(output_dir=str(config.DATA_DIR / "profiling" / "market_predictor"))
    def predict_market_direction(
        self,
        asset: str,
        timeframe: str = "24h",
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Génère une prédiction directionnelle pour un actif spécifique.

        Args:
            asset: Symbole de l'actif (ex: "BTC", "ETH")
            timeframe: Horizon temporel de la prédiction (ex: "1h", "24h", "7d")
            market_data: Données de marché optionnelles, si non fournies, elles seront récupérées

        Returns:
            Dictionnaire contenant la prédiction et les métadonnées associées
        """
        logger.info(f"Génération de prédiction pour {asset} sur {timeframe}")

        # Génération de la clé de cache
        cache_key = f"predict_market_direction:{asset}:{timeframe}"

        # Vérification du cache
        cached_prediction = self.cache.get(cache_key)
        if cached_prediction:
            logger.info(f"Prédiction récupérée du cache pour {asset} sur {timeframe}")
            return cached_prediction

        # Récupération des données si non fournies
        if market_data is None:
            market_data = self._fetch_market_data(asset, timeframe)

        # Récupération des analyses de sentiment (avec cache)
        text_query = f"{asset} cryptocurrency market analysis for {timeframe} timeframe"
        news_sentiment = self._get_cached_news_sentiment(text_query)
        social_sentiment = self._get_cached_social_sentiment(text_query)

        # Création du prompt
        prompt = self._format_prompt(
            data=market_data,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            asset=asset,
            timeframe=timeframe,
        )

        # Utilisation du contexte GPU RTX si disponible
        if self.rtx_optimizer:
            # Priorité à l'optimiseur RTX pour les opérations GPU intensives
            with self.rtx_optimizer.autocast_context():
                # Appel au LLM (avec gestion des retries)
                response = self._call_llm_with_retry(prompt)
        else:
            # Appel standard au LLM
            response = self._call_llm_with_retry(prompt)

        # Analyse de la réponse
        prediction = self._parse_prediction(response, asset, timeframe)

        # Enregistrement de la prédiction
        prediction_id = str(uuid.uuid4())
        prediction["id"] = prediction_id
        prediction["timestamp"] = datetime.now().isoformat()
        self.predictions_history[prediction_id] = prediction

        # Mise en cache de la prédiction
        # TTL adapté selon le timeframe pour éviter de réutiliser des prédictions obsolètes
        ttl = self._get_ttl_for_timeframe(timeframe)
        self.cache.set(cache_key, prediction, ttl)

        # Ajout d'informations sur l'utilisation du GPU RTX
        if self.rtx_optimizer:
            prediction["gpu_info"] = self.rtx_optimizer.get_optimization_info()

        return prediction

    def _get_ttl_for_timeframe(self, timeframe: str) -> int:
        """
        Détermine le TTL approprié selon le timeframe.

        Args:
            timeframe: Horizon temporel (ex: "1h", "24h", "7d")

        Returns:
            TTL en secondes
        """
        # Conversion du timeframe en secondes pour le TTL
        if "m" in timeframe:
            minutes = int(timeframe.replace("m", ""))
            return min(minutes * 12, 300)  # Max 5 minutes pour les timeframes courts
        elif "h" in timeframe:
            hours = int(timeframe.replace("h", ""))
            return min(hours * 60, 1800)  # Max 30 minutes pour les timeframes en heures
        elif "d" in timeframe:
            days = int(timeframe.replace("d", ""))
            return min(days * 600, 3600)  # Max 1 heure pour les timeframes en jours
        else:
            return 1800  # 30 minutes par défaut

    def _get_cached_news_sentiment(self, text_query: str) -> Dict[str, Any]:
        """
        Récupère l'analyse de sentiment des actualités avec mise en cache.

        Args:
            text_query: Requête pour l'analyse

        Returns:
            Résultat de l'analyse de sentiment
        """
        cache_key = f"news_sentiment:{text_query}"
        cached_result = self.cache.get(cache_key)

        if cached_result:
            return cached_result

        result = self.news_analyzer.analyze_sentiment(text_query)
        self.cache.set(cache_key, result, 3600)  # TTL: 1 heure

        return result

    def _get_cached_social_sentiment(self, text_query: str) -> Dict[str, Any]:
        """
        Récupère l'analyse de sentiment social avec mise en cache.

        Args:
            text_query: Requête pour l'analyse

        Returns:
            Résultat de l'analyse de sentiment
        """
        cache_key = f"social_sentiment:{text_query}"
        cached_result = self.cache.get(cache_key)

        if cached_result:
            return cached_result

        result = self.social_analyzer.analyze_sentiment(text_query)
        self.cache.set(
            cache_key, result, 1800
        )  # TTL: 30 minutes (données sociales plus volatiles)

        return result

    @profile(output_dir=str(config.DATA_DIR / "profiling" / "market_predictor"))
    def generate_market_insights(self, asset: str) -> Dict[str, Any]:
        """
        Génère des insights qualitatifs sur un actif.

        Args:
            asset: Symbole de l'actif

        Returns:
            Dictionnaire contenant les insights et métadonnées
        """
        logger.info(f"Génération d'insights pour {asset}")

        # Génération de la clé de cache
        cache_key = f"market_insights:{asset}"

        # Vérification du cache
        cached_insights = self.cache.get(cache_key)
        if cached_insights:
            logger.info(f"Insights récupérés du cache pour {asset}")
            return cached_insights

        # Récupération des données
        market_data = self._fetch_market_data(asset, "7d")

        # Récupération des analyses de sentiment avec cache
        text_query = f"{asset} cryptocurrency market analysis for 7d timeframe"
        news_sentiment = self._get_cached_news_sentiment(text_query)
        social_sentiment = self._get_cached_social_sentiment(text_query)

        # Création du prompt pour insights
        prompt = self._format_insights_prompt(
            data=market_data,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            asset=asset,
        )

        # Appel au LLM avec gestion des retries
        response = self._call_llm_with_retry(prompt)

        # Structuration de la réponse
        insights = {
            "asset": asset,
            "timestamp": datetime.now().isoformat(),
            "insights": response,
            "id": str(uuid.uuid4()),
        }

        # Mise en cache des insights (TTL: 4 heures)
        self.cache.set(cache_key, insights, 14400)

        return insights

    def get_confidence_score(self, prediction: Dict[str, Any]) -> float:
        """
        Calcule un score de confiance pour une prédiction.

        Args:
            prediction: Dictionnaire de prédiction

        Returns:
            Score de confiance entre 0 et 1
        """
        # Implémentation simple basée sur la cohérence entre sentiment et prédiction
        direction = prediction.get("direction", "")
        sentiment_score = prediction.get("sentiment_score", 0)

        if (direction == "bullish" and sentiment_score > 0) or (
            direction == "bearish" and sentiment_score < 0
        ):
            # Les deux sont alignés
            return min(0.5 + abs(sentiment_score) / 2, 1.0)
        else:
            # Ils sont en désaccord
            return max(0.5 - abs(sentiment_score) / 4, 0.1)

    def explain_prediction(self, prediction_id: str) -> Dict[str, Any]:
        """
        Fournit une explication détaillée d'une prédiction.

        Args:
            prediction_id: ID de la prédiction à expliquer

        Returns:
            Dictionnaire contenant l'explication
        """
        if prediction_id not in self.predictions_history:
            logger.warning(f"Prédiction {prediction_id} non trouvée")
            return {"error": "Prédiction non trouvée"}

        # Génération de la clé de cache
        cache_key = f"explain_prediction:{prediction_id}"

        # Vérification du cache
        cached_explanation = self.cache.get(cache_key)
        if cached_explanation:
            logger.info(f"Explication récupérée du cache pour {prediction_id}")
            return cached_explanation

        prediction = self.predictions_history[prediction_id]

        # Création du prompt pour l'explication
        prompt = f"""
        En vous basant sur les facteurs suivants, expliquez en détail pourquoi vous avez prédit une tendance {prediction['direction']} 
        pour {prediction['asset']} sur un horizon de {prediction['timeframe']}.
        
        Facteurs à considérer:
        - Données techniques: {prediction.get('technical_factors', [])}
        - Sentiment des nouvelles: {prediction.get('news_sentiment', {})}
        - Sentiment social: {prediction.get('social_sentiment', {})}
        - Événements du marché: {prediction.get('market_events', [])}
        
        Fournissez une explication complète et structurée qui pourrait aider un trader à comprendre votre raisonnement.
        """

        # Appel au LLM avec gestion des retries
        explanation_content = self._call_llm_with_retry(prompt)

        explanation = {
            "prediction_id": prediction_id,
            "asset": prediction["asset"],
            "direction": prediction["direction"],
            "explanation": explanation_content,
            "timestamp": datetime.now().isoformat(),
        }

        # Mise en cache de l'explication (longue durée car ne change pas)
        self.cache.set(cache_key, explanation, 86400)  # TTL: 24 heures

        return explanation

    @cached(ttl=1800)  # Cache de 30 minutes pour les données de marché
    def _fetch_market_data(self, asset: str, timeframe: str) -> pd.DataFrame:
        """
        Récupère les données de marché pour un actif avec mise en cache.

        Args:
            asset: Symbole de l'actif
            timeframe: Horizon temporel

        Returns:
            DataFrame contenant les données de marché
        """
        # Cette fonction devrait être implémentée pour récupérer les données réelles
        # Pour l'instant, nous retournons un DataFrame fictif
        logger.info(f"Récupération des données pour {asset} sur {timeframe}")

        # Mock data - à remplacer par une vraie implémentation
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
        data = {
            "date": dates,
            "open": [100 + i for i in range(30)],
            "high": [105 + i for i in range(30)],
            "low": [95 + i for i in range(30)],
            "close": [102 + i for i in range(30)],
            "volume": [1000000 - i * 10000 for i in range(30)],
        }

        return pd.DataFrame(data)

    def _format_prompt(
        self,
        data: pd.DataFrame,
        news_sentiment: Dict[str, Any],
        social_sentiment: Dict[str, Any],
        asset: str,
        timeframe: str,
    ) -> str:
        """
        Formate le prompt pour la prédiction de marché.

        Args:
            data: Données de marché (OHLCV, indicateurs)
            news_sentiment: Analyse de sentiment des actualités
            social_sentiment: Analyse de sentiment des médias sociaux
            asset: Symbole de l'actif
            timeframe: Horizon temporel de la prédiction

        Returns:
            Prompt formaté pour le LLM
        """
        # Création d'un résumé des données de marché
        data_summary = data.tail(10).describe().round(2)

        # Extraction des indicateurs techniques clés
        key_indicators = {}
        indicator_cols = [
            col
            for col in data.columns
            if any(
                ind in col.lower()
                for ind in ["rsi", "ma", "ema", "macd", "bb", "volume", "volatility"]
            )
        ]

        if indicator_cols:
            key_indicators = data[indicator_cols].tail(5).mean().to_dict()

        # Extraction des scores de sentiment
        news_score = news_sentiment.get("sentiment_score", 0)
        social_score = social_sentiment.get("sentiment_score", 0)

        # Construction du prompt optimisé pour JSON
        prompt = f"""
Agissez en tant qu'analyste financier expert spécialisé dans les cryptomonnaies. 

Votre tâche: Générer une prédiction directionnelle (bullish, bearish ou neutral) pour {asset} sur un horizon {timeframe}.

Veuillez analyser les données suivantes et fournir une réponse UNIQUEMENT au format JSON:

## 1. Données récentes de prix (10 derniers points)
{data.tail(10)[['close', 'volume']].to_string()}

## 2. Statistiques des données
{data_summary}

## 3. Indicateurs techniques clés
{json.dumps(key_indicators, indent=2)}

## 4. Analyses de sentiment
- News: score={news_score:.2f}, source_count={news_sentiment.get("source_count", 0)}
- Social: score={social_score:.2f}, source_count={social_sentiment.get("source_count", 0)}

## 5. Top mots-clés des news
{news_sentiment.get("keywords", [])}

## Structure exacte de la réponse JSON attendue:
{{
  "direction": "bullish|bearish|neutral",
  "confidence": x.xx,
  "analysis": "Explication concise de la prédiction",
  "key_factors": ["facteur 1", "facteur 2", "facteur 3"]
}}

Votre réponse doit être structurée UNIQUEMENT sous ce format JSON.
"""

        return prompt

    def _format_insights_prompt(
        self,
        data: pd.DataFrame,
        news_sentiment: Dict[str, Any],
        social_sentiment: Dict[str, Any],
        asset: str,
    ) -> str:
        """
        Formate le prompt pour la génération d'insights de marché.

        Args:
            data: Données de marché (OHLCV, indicateurs)
            news_sentiment: Analyse de sentiment des actualités
            social_sentiment: Analyse de sentiment des médias sociaux
            asset: Symbole de l'actif

        Returns:
            Prompt formaté pour le LLM
        """
        # Extraction de statistiques clés
        latest_price = data["close"].iloc[-1] if not data.empty else "N/A"
        price_change_24h = (
            (data["close"].iloc[-1] - data["close"].iloc[-24])
            / data["close"].iloc[-24]
            * 100
            if len(data) >= 24
            else "N/A"
        )
        volume_24h = data["volume"].iloc[-24:].sum() if len(data) >= 24 else "N/A"

        # Préparation des tendances des indicateurs
        indicator_trends = {}
        for indicator in ["rsi", "macd", "ma"]:
            cols = [c for c in data.columns if indicator in c.lower()]
            if cols:
                for col in cols:
                    if len(data) >= 5:
                        current = data[col].iloc[-1]
                        prev = data[col].iloc[-5]
                        change = (
                            ((current - prev) / abs(prev)) * 100 if prev != 0 else 0
                        )
                        indicator_trends[col] = {
                            "current": round(current, 2),
                            "change": round(change, 2),
                        }

        # Construction du prompt optimisé pour JSON
        prompt = f"""
Agissez en tant qu'analyste de marché crypto professionnel.

Votre tâche: Générer des insights concis et exploitables sur {asset}.

## Données de marché
- Prix actuel: {latest_price}
- Variation 24h: {price_change_24h}%
- Volume 24h: {volume_24h}

## Indicateurs techniques
{json.dumps(indicator_trends, indent=2)}

## Sentiment
- News: score={news_sentiment.get("sentiment_score", 0)}, sources={news_sentiment.get("source_count", 0)}
- Social: score={social_sentiment.get("sentiment_score", 0)}, mentions={social_sentiment.get("source_count", 0)}

## Sujets tendance
{news_sentiment.get("topics", [])}

## Générez une réponse JSON avec EXACTEMENT cette structure:
{{
  "market_state": "bull|bear|sideways|uncertain",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "risk_assessment": {{
    "level": "low|medium|high",
    "factors": ["facteur 1", "facteur 2"]
  }},
  "trading_opportunities": [
    {{
      "type": "entry|exit|hold",
      "rationale": "justification concise",
      "confidence": x.xx
    }}
  ]
}}

Votre réponse doit être UNIQUEMENT au format JSON spécifié ci-dessus.
"""

        return prompt

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """
        Appelle le LLM avec gestion des erreurs et retries.

        Args:
            prompt: Le prompt à envoyer au modèle
            max_retries: Nombre maximal de tentatives

        Returns:
            Réponse du modèle
        """
        retries = 0
        while retries < max_retries:
            try:
                return self._call_llm(prompt)
            except Exception as e:
                retries += 1
                wait_time = 2**retries  # Backoff exponentiel
                logger.warning(
                    f"Erreur lors de l'appel au LLM (tentative {retries}/{max_retries}): {e}"
                )
                if retries < max_retries:
                    logger.info(f"Nouvelle tentative dans {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Échec de l'appel au LLM après {max_retries} tentatives"
                    )
                    raise

    def _call_llm(self, prompt: str) -> str:
        """
        Appelle le LLM avec le prompt spécifié.

        Args:
            prompt: Le prompt à envoyer au modèle

        Returns:
            Réponse du modèle
        """
        start_time = time.time()

        # Configuration du modèle
        messages = [
            {
                "role": "system",
                "content": "Vous êtes un expert en analyse de marché crypto. Fournissez des prédictions fondées sur les données de marché et de sentiment. Répondez toujours dans le format JSON exact demandé sans explications supplémentaires.",
            },
            {"role": "user", "content": prompt},
        ]

        # Utilisation du client
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={
                    "type": "json_object"
                },  # Forcer le format JSON pour faciliter le parsing
            )

            # Calcul du temps d'exécution
            execution_time = time.time() - start_time
            logger.info(
                f"Appel au LLM {self.model_name} terminé en {execution_time:.2f}s"
            )

            # Extraction de la réponse
            response_text = response.choices[0].message.content
            return response_text

        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API: {e}")
            raise

    # Nouvelle méthode pour gérer les appels par lots
    def batch_predict_directions(
        self, assets: list, timeframe: str = "24h"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Génère des prédictions en lot pour plusieurs actifs.

        Args:
            assets: Liste des symboles d'actifs
            timeframe: Horizon temporel

        Returns:
            Dictionnaire des prédictions par actif
        """
        logger.info(
            f"Génération de prédictions par lots pour {len(assets)} actifs sur {timeframe}"
        )
        results = {}

        # Vérifier le cache pour chaque actif
        assets_to_predict = []
        cached_results = {}

        for asset in assets:
            cache_key = f"predict_market_direction:{asset}:{timeframe}"
            cached_prediction = self.cache.get(cache_key)

            if cached_prediction:
                cached_results[asset] = cached_prediction
                logger.debug(f"Prédiction récupérée du cache pour {asset}")
            else:
                assets_to_predict.append(asset)

        # Si tous les résultats sont en cache, retourner directement
        if not assets_to_predict:
            logger.info("Toutes les prédictions récupérées du cache")
            return cached_results

        # Définir la taille maximale du lot pour éviter les requêtes trop grandes
        batch_size = min(5, len(assets_to_predict))

        # Traiter par lots
        for i in range(0, len(assets_to_predict), batch_size):
            batch = assets_to_predict[i : i + batch_size]
            logger.info(f"Traitement du lot {i//batch_size+1}: {batch}")

            # Préparer les données pour le lot
            batch_data = {}
            for asset in batch:
                market_data = self._fetch_market_data(asset, timeframe)
                text_query = (
                    f"{asset} cryptocurrency market analysis for {timeframe} timeframe"
                )
                news_sentiment = self._get_cached_news_sentiment(text_query)
                social_sentiment = self._get_cached_social_sentiment(text_query)

                batch_data[asset] = {
                    "data": market_data,
                    "news_sentiment": news_sentiment,
                    "social_sentiment": social_sentiment,
                }

            # Traiter le lot avec une seule requête API si possible
            if len(batch) == 1:
                # Un seul actif, utiliser la méthode standard
                asset = batch[0]
                results[asset] = self.predict_market_direction(
                    asset, timeframe, batch_data[asset]["data"]
                )
            else:
                # Plusieurs actifs, utiliser une requête groupée
                grouped_results = self._batch_call_llm(batch, batch_data, timeframe)
                results.update(grouped_results)

        # Combiner avec les résultats du cache
        results.update(cached_results)

        return results

    def _batch_call_llm(
        self, assets: list, batch_data: dict, timeframe: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Appelle le LLM une seule fois pour plusieurs actifs.

        Args:
            assets: Liste des symboles d'actifs
            batch_data: Données préparées pour chaque actif
            timeframe: Horizon temporel

        Returns:
            Dictionnaire des prédictions par actif
        """
        # Construire un prompt combiné pour tous les actifs
        combined_prompt = f"""
Agissez en tant qu'analyste financier expert spécialisé dans les cryptomonnaies.

Votre tâche: Générer des prédictions directionnelles pour {len(assets)} actifs sur un horizon {timeframe}.

Pour chaque actif, analysez les données fournies et générez une prédiction structurée.

"""

        for i, asset in enumerate(assets, 1):
            combined_prompt += f"\n## ACTIF {i}: {asset}\n"

            # Données de marché
            data = batch_data[asset]["data"]
            data_summary = data.tail(5)[["close", "volume"]].to_string()

            # Sentiment
            news_sentiment = batch_data[asset]["news_sentiment"]
            social_sentiment = batch_data[asset]["social_sentiment"]
            news_score = news_sentiment.get("sentiment_score", 0)
            social_score = social_sentiment.get("sentiment_score", 0)

            combined_prompt += f"""
Données récentes: 
{data_summary}

Sentiment:
- News: {news_score}, Social: {social_score}

"""

        combined_prompt += f"""
## Format de réponse
Fournissez vos prédictions au format JSON EXACT suivant:

{{
  "ASSET1": {{
    "direction": "bullish|bearish|neutral",
    "confidence": x.xx,
    "analysis": "Explication concise",
    "key_factors": ["facteur 1", "facteur 2"]
  }},
  "ASSET2": {{
    ...
  }}
}}

Votre réponse doit être UNIQUEMENT ce JSON, sans texte supplémentaire avant ou après.
"""

        # Appel au LLM
        response = self._call_llm_with_retry(combined_prompt)

        # Traitement de la réponse
        try:
            # Tenter de parser la réponse comme JSON
            parsed_response = json.loads(response)

            # Traiter chaque prédiction d'actif
            results = {}
            for asset in assets:
                if asset in parsed_response:
                    asset_prediction = parsed_response[asset]

                    # Construction du résultat final
                    prediction = self._parse_prediction(
                        json.dumps(asset_prediction), asset, timeframe
                    )

                    # Mise en cache
                    prediction_id = str(uuid.uuid4())
                    prediction["id"] = prediction_id
                    prediction["timestamp"] = datetime.now().isoformat()
                    self.predictions_history[prediction_id] = prediction

                    # TTL adapté selon le timeframe
                    ttl = self._get_ttl_for_timeframe(timeframe)
                    cache_key = f"predict_market_direction:{asset}:{timeframe}"
                    self.cache.set(cache_key, prediction, ttl)

                    # Ajout d'informations GPU si applicable
                    if self.rtx_optimizer:
                        prediction["gpu_info"] = (
                            self.rtx_optimizer.get_optimization_info()
                        )

                    results[asset] = prediction
                else:
                    logger.warning(
                        f"Pas de prédiction pour {asset} dans la réponse groupée"
                    )

            return results

        except json.JSONDecodeError:
            logger.error(
                "Impossible de parser la réponse JSON du LLM pour la prédiction par lots"
            )

            # Fallback: traiter les actifs individuellement
            results = {}
            for asset in assets:
                try:
                    results[asset] = self.predict_market_direction(
                        asset, timeframe, batch_data[asset]["data"]
                    )
                except Exception as e:
                    logger.error(
                        f"Erreur lors de la prédiction individuelle pour {asset}: {e}"
                    )

            return results

    def _parse_prediction(
        self, response: str, asset: str, timeframe: str
    ) -> Dict[str, Any]:
        """
        Parse la réponse du LLM en un dictionnaire structuré.

        Args:
            response: Réponse brute du LLM
            asset: Symbole de l'actif
            timeframe: Horizon temporel

        Returns:
            Dictionnaire structuré de la prédiction
        """
        try:
            # Extraction de l'objet JSON
            prediction_json = json.loads(response)

            # Construction de l'objet de prédiction
            prediction = {
                "asset": asset,
                "timeframe": timeframe,
                "direction": prediction_json.get("direction", "neutral"),
                "confidence": prediction_json.get("confidence", "low"),
                "factors": prediction_json.get("factors", []),
                "contradictions": prediction_json.get("contradictions", []),
                "volatility": prediction_json.get("volatility", "medium"),
                "raw_response": response,
            }

            return prediction
        except Exception as e:
            logger.error(f"Erreur lors du parsing de la réponse du LLM: {e}")
            return {
                "asset": asset,
                "timeframe": timeframe,
                "direction": "neutral",
                "confidence": "low",
                "factors": ["Erreur de parsing"],
                "error": str(e),
                "raw_response": response,
            }

    def purge_cache(self) -> None:
        """
        Purge les entrées expirées du cache.
        """
        purged_count = self.cache.purge_expired()
        logger.info(f"{purged_count} entrées expirées purgées du cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache.

        Returns:
            Dictionnaire des statistiques du cache
        """
        return self.cache.get_stats()

    def cleanup_resources(self):
        """
        Nettoie les ressources utilisées (mémoire GPU, etc.).
        """
        # Libération des ressources RTX
        if self.rtx_optimizer:
            self.rtx_optimizer.clear_cache()
            logger.info("Ressources GPU RTX libérées")
        elif torch.cuda.is_available():
            # Nettoyage basique de la mémoire CUDA
            torch.cuda.empty_cache()
            logger.info("Cache CUDA vidé")
