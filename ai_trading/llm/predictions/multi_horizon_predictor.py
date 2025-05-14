#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple

from ai_trading.utils import setup_logger
from ai_trading.llm.predictions.market_predictor import MarketPredictor
from ai_trading.llm.predictions.prediction_model import PredictionModel

# Configuration du logger
logger = setup_logger("multi_horizon_predictor")

class MultiHorizonPredictor:
    """
    Classe de prédiction multi-horizons qui génère des prédictions sur différentes échelles temporelles.
    
    Cette classe étend les fonctionnalités du MarketPredictor et du PredictionModel pour fournir
    des prédictions à court, moyen et long terme pour un actif donné, avec une adaptation dynamique
    en fonction des conditions du marché.
    """
    
    # Définition des horizons temporels
    SHORT_TERM = ["15m", "30m", "1h", "4h"]
    MEDIUM_TERM = ["6h", "12h", "24h"]
    LONG_TERM = ["3d", "7d", "14d", "30d"]
    
    def __init__(self, 
                llm_model: str = "gpt-4", 
                model_save_dir: Optional[str] = None,
                use_hybrid: bool = True):
        """
        Initialise le prédicteur multi-horizons.
        
        Args:
            llm_model (str): Modèle LLM à utiliser (gpt-3.5-turbo, gpt-4, etc.)
            model_save_dir (str, optional): Répertoire de sauvegarde des modèles
            use_hybrid (bool): Utiliser le modèle hybride (True) ou uniquement LLM (False)
        """
        self.llm_model = llm_model
        self.use_hybrid = use_hybrid
        
        # Initialisation du MarketPredictor de base
        self.market_predictor = MarketPredictor(custom_config={"model_name": llm_model})
        
        # Dictionnaire pour stocker les modèles de prédiction par horizon
        self.prediction_models = {}
        
        # Répertoire de sauvegarde des modèles
        if model_save_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                   "info_retour", "models", "predictions")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            self.model_save_dir = os.path.join(base_dir, "multi_horizon")
        else:
            self.model_save_dir = model_save_dir
            
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            
        logger.info(f"MultiHorizonPredictor initialisé avec modèle: {llm_model}, mode hybride: {use_hybrid}")
    
    def initialize_prediction_models(self, timeframes: List[str]):
        """
        Initialise les modèles de prédiction pour chaque horizon temporel.
        
        Args:
            timeframes (List[str]): Liste des horizons temporels à initialiser
        """
        for timeframe in timeframes:
            if timeframe not in self.prediction_models:
                model_path = os.path.join(self.model_save_dir, f"prediction_model_{timeframe}.pkl")
                
                if os.path.exists(model_path) and self.use_hybrid:
                    # Charger un modèle existant
                    self.prediction_models[timeframe] = PredictionModel()
                    self.prediction_models[timeframe].load_model(model_path)
                    logger.info(f"Modèle chargé pour {timeframe} depuis {model_path}")
                elif self.use_hybrid:
                    # Créer un nouveau modèle
                    self.prediction_models[timeframe] = PredictionModel(
                        custom_config={
                            "model_dir": self.model_save_dir,
                            "llm_weight": 0.4,  # Donner un peu plus de poids au LLM
                            "ml_weight": 0.6
                        }
                    )
                    logger.info(f"Nouveau modèle créé pour {timeframe}")
        
        return self.prediction_models
    
    def predict_all_horizons(self, 
                           asset: str, 
                           short_term: bool = True, 
                           medium_term: bool = True, 
                           long_term: bool = True) -> Dict[str, Dict]:
        """
        Génère des prédictions pour tous les horizons temporels spécifiés.
        
        Args:
            asset (str): Symbole de l'actif (ex: 'BTC')
            short_term (bool): Inclure les prédictions à court terme
            medium_term (bool): Inclure les prédictions à moyen terme
            long_term (bool): Inclure les prédictions à long terme
            
        Returns:
            Dict[str, Dict]: Dictionnaire des prédictions par horizon temporel
        """
        horizons = []
        if short_term:
            horizons.extend(self.SHORT_TERM)
        if medium_term:
            horizons.extend(self.MEDIUM_TERM)
        if long_term:
            horizons.extend(self.LONG_TERM)
            
        if self.use_hybrid:
            self.initialize_prediction_models(horizons)
            
        predictions = {}
        
        for timeframe in horizons:
            logger.info(f"Génération de prédiction pour {asset} sur {timeframe}")
            
            if self.use_hybrid and timeframe in self.prediction_models:
                # Utiliser le modèle hybride si disponible
                try:
                    current_data = self._get_data_for_timeframe(asset, timeframe)
                    prediction = self.prediction_models[timeframe].predict(current_data, timeframe)
                    predictions[timeframe] = prediction
                except Exception as e:
                    logger.error(f"Erreur lors de la prédiction hybride pour {timeframe}: {str(e)}")
                    # Fallback sur LLM si erreur
                    prediction = self.market_predictor.predict_market_direction(asset, timeframe)
                    predictions[timeframe] = prediction
            else:
                # Utiliser uniquement le LLM
                prediction = self.market_predictor.predict_market_direction(asset, timeframe)
                predictions[timeframe] = prediction
                
        return predictions
    
    def _get_data_for_timeframe(self, asset: str, timeframe: str) -> Dict:
        """
        Récupère les données nécessaires pour un horizon temporel spécifique.
        
        Args:
            asset (str): Symbole de l'actif
            timeframe (str): Horizon temporel
            
        Returns:
            Dict: Données préparées pour la prédiction
        """
        # Cette méthode devrait être implémentée pour récupérer les données
        # appropriées pour chaque timeframe à partir d'une source de données
        # Pour l'instant, nous utilisons des données fictives
        
        # Dans une implémentation réelle, vous récupéreriez les données
        # de marché et de sentiment pour le timeframe spécifié
        
        from ai_trading.llm.predictions.test_predictions import generate_mock_market_data, generate_mock_sentiment_data
        
        # Ajuster la longueur des données en fonction du timeframe
        if timeframe in self.SHORT_TERM:
            days = 30  # Moins de données pour le court terme
        elif timeframe in self.MEDIUM_TERM:
            days = 60  # Plus de données pour le moyen terme
        else:
            days = 100  # Encore plus pour le long terme
            
        market_data = generate_mock_market_data(days=days)
        sentiment_data = generate_mock_sentiment_data(days=days)
        
        return {
            "market_data": market_data,
            "sentiment_data": sentiment_data,
            "asset": asset,
            "timeframe": timeframe
        }
    
    def train_models(self, 
                    asset: str, 
                    timeframes: List[str], 
                    historical_days: int = 365) -> Dict[str, Dict]:
        """
        Entraîne les modèles de prédiction pour chaque horizon temporel spécifié.
        
        Args:
            asset (str): Symbole de l'actif
            timeframes (List[str]): Liste des horizons temporels à entraîner
            historical_days (int): Nombre de jours de données historiques à utiliser
            
        Returns:
            Dict[str, Dict]: Résultats d'entraînement par horizon temporel
        """
        if not self.use_hybrid:
            logger.info("Mode hybride désactivé, pas d'entraînement nécessaire")
            return {}
            
        self.initialize_prediction_models(timeframes)
        
        results = {}
        
        for timeframe in timeframes:
            logger.info(f"Entraînement du modèle pour {asset} sur {timeframe}")
            try:
                # Récupérer les données d'entraînement
                data = self._get_training_data(asset, timeframe, historical_days)
                
                # Entraîner le modèle
                metrics = self.prediction_models[timeframe].train(
                    market_data=data["market_data"],
                    sentiment_data=data["sentiment_data"]
                )
                
                # Sauvegarder le modèle
                model_path = os.path.join(self.model_save_dir, f"prediction_model_{timeframe}.pkl")
                self.prediction_models[timeframe].save_model(model_path)
                
                results[timeframe] = {
                    "metrics": metrics,
                    "model_path": model_path
                }
                
                logger.info(f"Modèle pour {timeframe} entraîné avec succès. Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement pour {timeframe}: {str(e)}")
                results[timeframe] = {"error": str(e)}
                
        return results
    
    def _get_training_data(self, asset: str, timeframe: str, days: int) -> Dict:
        """
        Récupère les données d'entraînement pour un actif et un horizon temporel.
        
        Args:
            asset (str): Symbole de l'actif
            timeframe (str): Horizon temporel
            days (int): Nombre de jours de données historiques
            
        Returns:
            Dict: Données d'entraînement structurées
        """
        # Dans une implémentation réelle, vous récupéreriez les données
        # historiques appropriées pour l'entraînement
        
        from ai_trading.llm.predictions.test_predictions import generate_mock_market_data, generate_mock_sentiment_data
        
        market_data = generate_mock_market_data(days=days)
        sentiment_data = generate_mock_sentiment_data(days=days)
        
        return {
            "market_data": market_data,
            "sentiment_data": sentiment_data
        }
    
    def analyze_consistency(self, predictions: Dict[str, Dict]) -> Dict:
        """
        Analyse la cohérence des prédictions à travers différents horizons temporels.
        
        Args:
            predictions (Dict[str, Dict]): Dictionnaire des prédictions par horizon
            
        Returns:
            Dict: Analyse de cohérence et signaux de trading potentiels
        """
        if not predictions:
            return {"error": "Aucune prédiction à analyser"}
            
        directions = {}
        confidences = {}
        
        # Grouper les directions et confiances par horizon
        for timeframe, prediction in predictions.items():
            if timeframe in self.SHORT_TERM:
                horizon = "short_term"
            elif timeframe in self.MEDIUM_TERM:
                horizon = "medium_term"
            else:
                horizon = "long_term"
                
            if horizon not in directions:
                directions[horizon] = []
            if horizon not in confidences:
                confidences[horizon] = []
                
            directions[horizon].append(prediction["direction"])
            confidences[horizon].append(prediction["confidence"])
        
        # Analyser la cohérence par horizon
        results = {
            "short_term": self._analyze_horizon_consistency(directions.get("short_term", []), confidences.get("short_term", [])),
            "medium_term": self._analyze_horizon_consistency(directions.get("medium_term", []), confidences.get("medium_term", [])),
            "long_term": self._analyze_horizon_consistency(directions.get("long_term", []), confidences.get("long_term", []))
        }
        
        # Détecter les divergences et les signaux potentiels
        signals = self._generate_trading_signals(results)
        
        return {
            "horizon_analysis": results,
            "trading_signals": signals,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_horizon_consistency(self, directions: List[str], confidences: List[str]) -> Dict:
        """
        Analyse la cohérence des prédictions au sein d'un horizon temporel.
        
        Args:
            directions (List[str]): Liste des directions prédites
            confidences (List[str]): Liste des niveaux de confiance
            
        Returns:
            Dict: Statistiques de cohérence pour cet horizon
        """
        if not directions:
            return {"consistency": 0, "overall_direction": "unknown", "confidence": "unknown"}
            
        # Compter les occurrences de chaque direction
        direction_counts = {
            "bullish": directions.count("bullish"),
            "bearish": directions.count("bearish"),
            "neutral": directions.count("neutral")
        }
        
        total = len(directions)
        
        # Trouver la direction dominante
        max_direction = max(direction_counts, key=direction_counts.get)
        max_count = direction_counts[max_direction]
        
        # Calculer la cohérence (proportion de la direction dominante)
        consistency = max_count / total if total > 0 else 0
        
        # Convertir les niveaux de confiance en valeurs numériques
        confidence_values = {
            "high": 3,
            "medium": 2,
            "low": 1,
            "unknown": 0
        }
        
        conf_values = [confidence_values.get(conf, 0) for conf in confidences]
        avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0
        
        # Reconvertir la confiance moyenne en catégorie
        if avg_confidence >= 2.5:
            confidence_label = "high"
        elif avg_confidence >= 1.5:
            confidence_label = "medium"
        elif avg_confidence > 0:
            confidence_label = "low"
        else:
            confidence_label = "unknown"
            
        return {
            "consistency": consistency,
            "overall_direction": max_direction,
            "confidence": confidence_label,
            "direction_counts": direction_counts,
            "total_signals": total
        }
    
    def _generate_trading_signals(self, horizon_analysis: Dict) -> Dict:
        """
        Génère des signaux de trading basés sur l'analyse de cohérence multi-horizons.
        
        Args:
            horizon_analysis (Dict): Analyse de cohérence par horizon
            
        Returns:
            Dict: Signaux de trading potentiels
        """
        short_term = horizon_analysis.get("short_term", {})
        medium_term = horizon_analysis.get("medium_term", {})
        long_term = horizon_analysis.get("long_term", {})
        
        # Vérifier si nous avons assez de données pour générer des signaux
        if not short_term or not medium_term or not long_term:
            return {"signal": "insufficient_data"}
            
        # Vérifier la cohérence de chaque horizon
        short_consistent = short_term.get("consistency", 0) >= 0.7
        medium_consistent = medium_term.get("consistency", 0) >= 0.7
        long_consistent = long_term.get("consistency", 0) >= 0.7
        
        # Vérifier l'alignement des directions
        all_bullish = (
            short_term.get("overall_direction") == "bullish" and
            medium_term.get("overall_direction") == "bullish" and
            long_term.get("overall_direction") == "bullish"
        )
        
        all_bearish = (
            short_term.get("overall_direction") == "bearish" and
            medium_term.get("overall_direction") == "bearish" and
            long_term.get("overall_direction") == "bearish"
        )
        
        # Vérifier les divergences
        bullish_divergence = (
            short_term.get("overall_direction") == "bearish" and
            medium_term.get("overall_direction") == "neutral" and
            long_term.get("overall_direction") == "bullish"
        )
        
        bearish_divergence = (
            short_term.get("overall_direction") == "bullish" and
            medium_term.get("overall_direction") == "neutral" and
            long_term.get("overall_direction") == "bearish"
        )
        
        # Générer le signal final
        if all_bullish and short_consistent and medium_consistent and long_consistent:
            signal = "strong_buy"
            description = "Forte tendance haussière sur tous les horizons temporels"
        elif all_bearish and short_consistent and medium_consistent and long_consistent:
            signal = "strong_sell"
            description = "Forte tendance baissière sur tous les horizons temporels"
        elif bullish_divergence:
            signal = "potential_reversal_bullish"
            description = "Potentiel renversement haussier (court terme baissier, long terme haussier)"
        elif bearish_divergence:
            signal = "potential_reversal_bearish"
            description = "Potentiel renversement baissier (court terme haussier, long terme baissier)"
        elif short_term.get("overall_direction") == "bullish" and medium_term.get("overall_direction") == "bullish":
            signal = "buy"
            description = "Tendance haussière à court et moyen terme"
        elif short_term.get("overall_direction") == "bearish" and medium_term.get("overall_direction") == "bearish":
            signal = "sell"
            description = "Tendance baissière à court et moyen terme"
        else:
            signal = "neutral"
            description = "Pas de tendance claire ou signaux contradictoires"
        
        return {
            "signal": signal,
            "description": description,
            "short_term_dir": short_term.get("overall_direction"),
            "medium_term_dir": medium_term.get("overall_direction"),
            "long_term_dir": long_term.get("overall_direction"),
            "confidence": min(
                short_term.get("confidence", "low"),
                medium_term.get("confidence", "low"),
                long_term.get("confidence", "low"),
                key=lambda x: {"high": 3, "medium": 2, "low": 1, "unknown": 0}[x]
            )
        } 