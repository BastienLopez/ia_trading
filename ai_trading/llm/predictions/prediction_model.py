"""
Module de modèle de prédiction combinant données techniques et sentiment.

Ce module implémente un modèle hybride qui combine les LLM avec des modèles 
de machine learning traditionnels pour produire des prédictions de marché.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import time

# Importations internes
from ai_trading.llm.predictions.market_predictor import MarketPredictor
import ai_trading.config as config
from ai_trading.utils import setup_logger
from ai_trading.llm.predictions.parallel_processor import EnsembleParallelProcessor, ParallelProcessor
from ai_trading.llm.predictions.cache_manager import CacheManager, cached
from ai_trading.llm.predictions.performance_analysis import profile
from ai_trading.llm.predictions.rtx_optimizer import RTXOptimizer, detect_rtx_gpu, setup_rtx_environment

# Configuration du logger
logger = setup_logger("prediction_model")

# Configuration initiale de l'environnement GPU
rtx_gpu_info = None
if torch.cuda.is_available():
    # Vérifier d'abord si un GPU RTX est disponible
    rtx_gpu_info = detect_rtx_gpu()
    if rtx_gpu_info:
        # Optimisation spécifique pour RTX
        setup_rtx_environment()
        logger.info(f"Environnement GPU RTX {rtx_gpu_info['series']} configuré pour les modèles de prédiction")
    else:
        logger.info("Aucun GPU RTX détecté, exécution standard pour les modèles de prédiction")
else:
    logger.info("Aucun GPU détecté, exécution en mode CPU pour les modèles de prédiction")

# Classe pour les réseaux de neurones PyTorch 
class PredictionNN(nn.Module):
    """
    Réseau de neurones simple pour la prédiction de marché.
    
    Ce modèle PyTorch peut exploiter l'accélération GPU disponible.
    """
    
    def __init__(self, input_size, hidden_size=64, output_size=3):
        """
        Initialise le réseau de neurones.
        
        Args:
            input_size: Taille des features d'entrée
            hidden_size: Taille des couches cachées
            output_size: Nombre de classes de sortie
        """
        super(PredictionNN, self).__init__()
        self.input_shape = (1, input_size)  # Pour les optimisations RTX et TensorRT
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Propagation avant.
        
        Args:
            x: Tenseur d'entrée
        
        Returns:
            Tenseur de sortie avec probabilités
        """
        return self.model(x)

class PredictionModel:
    """
    Modèle de prédiction hybride combinant LLM et ML.
    
    Ce modèle utilise une combinaison pondérée des prédictions LLM et ML
    pour fournir des prédictions de marché plus robustes.
    """
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialise le modèle de prédiction.
        
        Args:
            custom_config: Configuration personnalisée optionnelle
        """
        # Utilisation de la configuration du projet
        self.config = custom_config or {}
        
        # Poids pour la combinaison des prédictions
        self.llm_weight = self.config.get("llm_weight", 0.4)
        self.ml_weight = self.config.get("ml_weight", 0.6)
        
        # Méthode de calibration des modèles
        self.calibration_method = self.config.get("calibration_method", "isotonic")
        
        # Répertoire de sauvegarde des modèles
        self.model_dir = self.config.get("model_dir", str(config.DATA_DIR / "models"))
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Prédictor LLM
        market_predictor_config = {
            "model_name": self.config.get("llm_model_name", "gpt-4"),
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 500),
            "cache_dir": self.config.get("cache_dir", str(config.DATA_DIR / "cache" / "predictions")),
            "use_gpu": self.config.get("use_gpu", True)
        }
        self.market_predictor = MarketPredictor(custom_config=market_predictor_config)
        
        # Modèles ML
        self.ml_model = None
        self.scaler = None
        self.processor = EnsembleParallelProcessor(
            max_workers=self.config.get("max_workers", 4)
        )
        
        # Variables pour la gestion des features
        self.feature_columns = []
        self.target_column = "direction"
        self.direction_mapping = {"bullish": 2, "neutral": 1, "bearish": 0}
        self.inverse_direction_mapping = {v: k for k, v in self.direction_mapping.items()}
        
        # Support PyTorch
        self.has_torch_models = False
        if torch.cuda.is_available():
            self.has_torch_models = True
            logger.info("Support PyTorch activé avec accélération GPU")
        
        # Initialisation des modèles PyTorch
        self.torch_models = []
        
        # Initialisation de l'optimiseur RTX si disponible
        use_gpu = self.config.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            # Vérification si RTX est disponible
            if rtx_gpu_info:
                # Utiliser l'optimiseur RTX spécifique
                self.rtx_optimizer = RTXOptimizer(
                    device_id=self.config.get("gpu_device_id", None),
                    enable_tensor_cores=self.config.get("enable_tensor_cores", True),
                    enable_half_precision=self.config.get("enable_half_precision", True),
                    optimize_memory=self.config.get("optimize_memory", True),
                    enable_tensorrt=self.config.get("enable_tensorrt", False)
                )
                logger.info(f"Accélération GPU RTX activée pour les modèles: {self.rtx_optimizer.get_optimization_info()}")
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
        
        logger.info("PredictionModel initialisé avec poids LLM: %s, ML: %s", self.llm_weight, self.ml_weight)
    
    def _prepare_pytorch_models(self, input_size: int) -> List[nn.Module]:
        """
        Prépare les modèles PyTorch pour l'entraînement.
        
        Args:
            input_size: Nombre de features d'entrée
            
        Returns:
            Liste de modèles PyTorch
        """
        models = []
        
        # Modèle de base
        base_model = PredictionNN(input_size=input_size)
        models.append(base_model)
        
        # Modèle plus large (plus de neurones)
        large_model = PredictionNN(input_size=input_size, hidden_size=128)
        models.append(large_model)
        
        # Déplacer les modèles sur GPU si disponible
        if self.rtx_optimizer:
            # Priorité à l'optimiseur RTX
            models = [self.rtx_optimizer.to_device(model) for model in models]
        elif torch.cuda.is_available():
            # Fallback direct sur CUDA
            models = [model.to("cuda:0") for model in models]
            
        return models
    
    def _optimize_pytorch_models(self, models: List[nn.Module]) -> List[nn.Module]:
        """
        Optimise les modèles PyTorch pour l'inférence.
        
        Args:
            models: Liste de modèles PyTorch
            
        Returns:
            Liste de modèles PyTorch optimisés
        """
        optimized_models = []
        
        for model in models:
            model.eval()  # Passage en mode évaluation
            
            if self.rtx_optimizer:
                # Optimisation spécifique RTX
                optimized_model = self.rtx_optimizer.optimize_for_inference(model)
                optimized_models.append(optimized_model)
            else:
                # Pas d'optimisation spécifique
                optimized_models.append(model)
                
        return optimized_models

    @profile(output_dir=str(config.DATA_DIR / "profiling" / "prediction_model"))
    def train(self, market_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entraîne les modèles de prédiction.
        
        Args:
            market_data: Données de marché (OHLCV, indicateurs techniques, etc.)
            sentiment_data: Données de sentiment (news, médias sociaux, etc.)
            
        Returns:
            Métriques d'entraînement
        """
        logger.info("Début de l'entraînement des modèles de prédiction")
        
        # Préparation des données
        X, y = self._prepare_data(market_data, sentiment_data)
        
        # Split des données pour validation temporelle
        tscv = TimeSeriesSplit(n_splits=5)
        train_indices, test_indices = list(tscv.split(X))[4]  # Utilisation du dernier split
        
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        # Scaling des features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Liste de modèles à entraîner
        models_config = [
            {
                "name": "RandomForest",
                "model": RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
            },
            {
                "name": "GradientBoosting",
                "model": GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42
                )
            }
        ]
        
        # Ajout de modèles PyTorch si disponibles
        if self.has_torch_models:
            # Préparation des tenseurs pour PyTorch
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            
            # Création et ajout des modèles PyTorch à la liste de modèles
            input_size = X_train.shape[1]
            self.torch_models = self._prepare_pytorch_models(input_size)
            
            # Optimisations spécifiques pour RTX si disponible
            train_context = None
            if self.rtx_optimizer:
                train_context = self.rtx_optimizer.autocast_context()
            else:
                from contextlib import nullcontext
                train_context = nullcontext()
                
            # Entraînement parallèle avec le contexte approprié
            with train_context:
                # Utilisation du processeur parallèle pour l'entraînement (traitement par lots)
                self.processor.train_pytorch_models(
                    self.torch_models, 
                    X_train_tensor, 
                    y_train_tensor,
                    epochs=50,
                    batch_size=32,
                    learning_rate=0.001
                )
                
            # Optimisation des modèles pour l'inférence
            self.torch_models = self._optimize_pytorch_models(self.torch_models)
        
        # Entraînement en parallèle des modèles scikit-learn
        self.ml_model = self.processor.train_models(models_config, X_train_scaled, y_train)
        
        # Évaluation des modèles
        y_pred = self._ensemble_predict(X_test_scaled)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(X.columns),
            "n_models": len(self.ml_model) + len(self.torch_models),
            "gpu_acceleration": self.rtx_optimizer is not None
        }
        
        # Ajout d'informations sur l'utilisation du GPU RTX
        if self.rtx_optimizer:
            metrics["gpu_info"] = self.rtx_optimizer.get_optimization_info()
        
        # Sauvegarde des modèles
        self._save_models()
        
        logger.info("Entraînement terminé. Accuracy: %.4f, F1: %.4f", 
                   metrics["accuracy"], metrics["f1"])
        
        return metrics
    
    @profile(output_dir=str(config.DATA_DIR / "profiling" / "prediction_model"))
    def predict(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Génère une prédiction combinée pour un actif.
        
        Args:
            asset: Symbole de l'actif (ex: "BTC", "ETH")
            timeframe: Horizon temporel (ex: "1h", "24h", "7d")
            
        Returns:
            Prédiction finale combinant LLM et ML
        """
        start_time = datetime.now()
        
        # 1. Prédiction du LLM
        llm_prediction = self.market_predictor.predict_market_direction(asset, timeframe)
        
        # 2. Récupération des données récentes pour le ML
        market_data, sentiment_data = self._fetch_recent_data(asset, timeframe)
        
        # 3. Préparation des données
        X, _ = self._prepare_data(market_data, sentiment_data, for_training=False)
        
        # Si aucun modèle ML n'est entraîné, on utilise uniquement la prédiction LLM
        if self.ml_model is None:
            return llm_prediction
        
        # 4. Scaling des features
        X_scaled = self.scaler.transform(X)
        
        # 5. Prédiction ML avec ensemble
        ml_proba = self._predict_proba(X_scaled)
        ml_direction_idx = np.argmax(ml_proba)
        ml_direction = self.inverse_direction_mapping[ml_direction_idx]
        ml_confidence = float(ml_proba[ml_direction_idx])
        
        # Création de la prédiction ML
        ml_prediction = {
            "direction": ml_direction,
            "confidence": ml_confidence,
            "probabilities": {
                self.inverse_direction_mapping[i]: float(p) 
                for i, p in enumerate(ml_proba)
            }
        }
        
        # 6. Combinaison des prédictions
        combined_prediction = self._combine_predictions(llm_prediction, ml_prediction)
        combined_prediction["asset"] = asset
        combined_prediction["timeframe"] = timeframe
        combined_prediction["llm_prediction"] = llm_prediction
        combined_prediction["ml_prediction"] = ml_prediction
        
        # Temps de prédiction
        prediction_time = (datetime.now() - start_time).total_seconds()
        combined_prediction["prediction_time_seconds"] = prediction_time
        
        # Ajout d'informations sur l'utilisation du GPU RTX
        if self.rtx_optimizer:
            combined_prediction["gpu_info"] = self.rtx_optimizer.get_optimization_info()
        
        logger.info(f"Prédiction combinée pour {asset} ({timeframe}): {combined_prediction['direction']} "
                   f"(confiance: {combined_prediction['confidence']:.2f})")
        
        return combined_prediction
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Génère des probabilités de classe en utilisant l'ensemble de modèles.
        
        Args:
            X: Features d'entrée
            
        Returns:
            Probabilités moyennes pour chaque classe
        """
        # Initialisation du tableau de probabilités
        all_probas = []
        
        # 1. Prédictions des modèles scikit-learn
        if self.ml_model:
            for model in self.ml_model:
                # Vérification que le modèle a predict_proba
                if hasattr(model, 'predict_proba'):
                    all_probas.append(model.predict_proba(X))
        
        # 2. Prédictions des modèles PyTorch
        if self.torch_models:
            # Conversion en tenseur
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Optimisations GPU RTX pour l'inférence
            inference_context = None
            if self.rtx_optimizer:
                inference_context = self.rtx_optimizer.autocast_context()
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.rtx_optimizer.device)
            else:
                from contextlib import nullcontext
                inference_context = nullcontext()
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                
            # Prédiction avec le contexte approprié
            with torch.no_grad(), inference_context:
                for model in self.torch_models:
                    proba = model(X_tensor).cpu().numpy()
                    all_probas.append(proba)
        
        # Si aucun modèle n'a donné de prédiction, retourner probabilités uniformes
        if not all_probas:
            return np.ones(len(self.direction_mapping)) / len(self.direction_mapping)
        
        # Moyenne des probabilités (ensemble)
        avg_proba = np.mean(all_probas, axis=0)
        
        return avg_proba

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
        
        # Nettoyage du MarketPredictor
        self.market_predictor.cleanup_resources()

    @profile(output_dir=str(config.DATA_DIR / "profiling" / "prediction_model"))
    def batch_predict(self, assets: List[str], timeframe: str = "24h") -> Dict[str, Dict[str, Any]]:
        """
        Génère des prédictions pour plusieurs actifs.
        
        Args:
            assets: Liste des symboles d'actifs
            timeframe: Horizon temporel
            
        Returns:
            Dictionnaire des prédictions par actif
        """
        # Si le market_predictor a une méthode batch_predict_directions, on l'utilise
        if hasattr(self.market_predictor, 'batch_predict_directions'):
            logger.info(f"Utilisation de la méthode batch_predict_directions pour {len(assets)} actifs")
            return self.market_predictor.batch_predict_directions(assets, timeframe)
        
        # Sinon, fallback sur une approche parallèle
        logger.info(f"Utilisation de la méthode parallèle pour {len(assets)} actifs")
        results = {}
        
        def predict_asset(asset):
            try:
                return asset, self.predict(asset, timeframe)
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction pour {asset}: {e}")
                return asset, {"error": str(e), "direction": "neutral", "confidence": 0.5}
        
        # Prédictions en parallèle avec un nombre de workers adapté
        results_list = self.processor.map(predict_asset, assets)
        
        # Conversion en dictionnaire
        for asset, prediction in results_list:
            results[asset] = prediction
        
        return results
    
    def preload_predictions(self, 
                          assets: List[str], 
                          timeframes: List[str] = ["24h"], 
                          async_mode: bool = True) -> None:
        """
        Précharge des prédictions en cache pour des actifs et horizons temporels spécifiques.
        
        Cette méthode peut être utilisée pour préparer le cache avant les heures de forte demande.
        
        Args:
            assets: Liste des symboles d'actifs à précharger
            timeframes: Liste des horizons temporels
            async_mode: Si True, exécute le préchargement de manière asynchrone
        """
        logger.info(f"Préchargement des prédictions pour {len(assets)} actifs sur {len(timeframes)} timeframes")
        
        def preload_worker():
            start_time = time.time()
            preloaded_count = 0
            cache_hit_count = 0
            
            for timeframe in timeframes:
                # Utiliser le traitement par lots pour chaque timeframe
                batch_size = 5  # Taille optimale pour les appels groupés
                for i in range(0, len(assets), batch_size):
                    batch_assets = assets[i:i+batch_size]
                    
                    # Vérifier quels actifs sont déjà en cache
                    assets_to_predict = []
                    for asset in batch_assets:
                        cache_key = f"predict_market_direction:{asset}:{timeframe}"
                        if self.market_predictor.cache.get(cache_key) is not None:
                            cache_hit_count += 1
                        else:
                            assets_to_predict.append(asset)
                    
                    if assets_to_predict:
                        try:
                            # Génération des prédictions manquantes
                            self.batch_predict(assets_to_predict, timeframe)
                            preloaded_count += len(assets_to_predict)
                        except Exception as e:
                            logger.error(f"Erreur lors du préchargement du lot {i//batch_size+1}: {e}")
            
            duration = time.time() - start_time
            logger.info(f"Préchargement terminé en {duration:.2f}s. "
                       f"Préchargés: {preloaded_count}, Déjà en cache: {cache_hit_count}")
        
        if async_mode:
            # Exécution asynchrone dans un thread séparé
            import threading
            preload_thread = threading.Thread(target=preload_worker, daemon=True)
            preload_thread.start()
        else:
            # Exécution synchrone
            preload_worker()
    
    def schedule_prediction_preloading(self, 
                                    assets: List[str], 
                                    timeframes: List[str] = ["24h"],
                                    interval_hours: int = 4) -> None:
        """
        Planifie le préchargement périodique des prédictions.
        
        Args:
            assets: Liste des symboles d'actifs à précharger
            timeframes: Liste des horizons temporels
            interval_hours: Intervalle entre les préchargements en heures
        """
        logger.info(f"Planification du préchargement toutes les {interval_hours} heures")
        
        import threading
        
        def scheduler_worker():
            while True:
                try:
                    # Exécuter le préchargement
                    self.preload_predictions(assets, timeframes, async_mode=False)
                    
                    # Attendre jusqu'au prochain intervalle
                    time.sleep(interval_hours * 3600)
                except Exception as e:
                    logger.error(f"Erreur dans la tâche planifiée de préchargement: {e}")
                    time.sleep(300)  # Attendre 5 minutes en cas d'erreur
        
        # Démarrer dans un thread séparé
        scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        scheduler_thread.start()
        
        logger.info("Tâche de préchargement périodique démarrée")
    
    def optimize_prediction_pipeline(self, assets: List[str], timeframe: str = "24h") -> Dict[str, Any]:
        """
        Optimise la pipeline de prédiction pour un ensemble d'actifs.
        
        Cette méthode analyse les performances et réalise des optimisations dynamiques
        pour améliorer la vitesse et la qualité des prédictions.
        
        Args:
            assets: Liste des symboles d'actifs
            timeframe: Horizon temporel
            
        Returns:
            Statistiques d'optimisation
        """
        logger.info(f"Optimisation de la pipeline pour {len(assets)} actifs")
        start_time = time.time()
        
        # 1. Tester et sélectionner la méthode de prédiction optimale
        test_batch_size = min(3, len(assets))
        test_assets = assets[:test_batch_size]
        
        # Tester la méthode batch
        batch_start = time.time()
        if hasattr(self.market_predictor, 'batch_predict_directions'):
            try:
                self.market_predictor.batch_predict_directions(test_assets, timeframe)
                batch_duration = time.time() - batch_start
                batch_per_asset = batch_duration / test_batch_size
                batch_method_available = True
            except Exception as e:
                logger.error(f"Erreur lors du test de la méthode batch: {e}")
                batch_method_available = False
                batch_per_asset = float('inf')
        else:
            batch_method_available = False
            batch_per_asset = float('inf')
        
        # Tester la méthode parallèle
        parallel_start = time.time()
        try:
            def predict_test(asset):
                return self.predict(asset, timeframe)
            
            self.processor.map(predict_test, test_assets)
            parallel_duration = time.time() - parallel_start
            parallel_per_asset = parallel_duration / test_batch_size
        except Exception as e:
            logger.error(f"Erreur lors du test de la méthode parallèle: {e}")
            parallel_per_asset = float('inf')
        
        # 2. Optimiser les tailles de batch et le niveau de parallélisme
        if batch_method_available and batch_per_asset <= parallel_per_asset:
            # La méthode batch est plus rapide
            optimal_method = "batch"
            
            # Déterminer la taille de batch optimale (entre 3 et 10)
            batch_size = min(10, max(3, len(assets) // 2))
        else:
            # La méthode parallèle est plus rapide
            optimal_method = "parallel"
            
            # Optimiser le nombre de workers
            available_cores = os.cpu_count() or 4
            optimal_workers = min(available_cores, max(2, len(assets)))
            self.processor.max_workers = optimal_workers
            
            # Batch size pour le traitement parallèle
            batch_size = 1
        
        # 3. Précharger les actifs les plus fréquemment demandés
        if len(assets) > 3:
            most_common_assets = assets[:3]  # Les 3 premiers sont supposés être les plus communs
            self.preload_predictions(most_common_assets, [timeframe], async_mode=True)
        
        # Durée totale de l'optimisation
        optimization_duration = time.time() - start_time
        
        # Statistiques d'optimisation
        optimization_stats = {
            "optimal_method": optimal_method,
            "batch_per_asset_ms": batch_per_asset * 1000 if batch_method_available else None,
            "parallel_per_asset_ms": parallel_per_asset * 1000,
            "optimal_batch_size": batch_size,
            "optimization_duration_ms": optimization_duration * 1000,
            "num_assets": len(assets),
            "timeframe": timeframe,
            "num_workers": self.processor.max_workers,
            "gpu_accelerated": self.rtx_optimizer is not None
        }
        
        logger.info(f"Optimisation terminée. Méthode: {optimal_method}, "
                   f"Durée: {optimization_duration:.2f}s")
        
        return optimization_stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache de prédiction.
        
        Returns:
            Statistiques du cache
        """
        # Statistiques du cache du market_predictor
        market_predictor_stats = self.market_predictor.get_cache_stats()
        
        return {
            "market_predictor_cache": market_predictor_stats
        }

    def _preprocess_data(self, market_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données de marché et de sentiment pour l'entraînement ou la prédiction.
        
        Args:
            market_data (pd.DataFrame): Données de marché avec colonnes OHLCV
            sentiment_data (pd.DataFrame): Données de sentiment avec colonnes de sentiment
            
        Returns:
            pd.DataFrame: Données prétraitées et fusionnées
        """
        # Fusionner les données de marché et de sentiment
        if 'date' in market_data.columns and 'date' in sentiment_data.columns:
            data = pd.merge(market_data, sentiment_data, on='date', how='inner')
        else:
            # Si les dates ne sont pas disponibles, on suppose que les indices correspondent
            data = pd.concat([market_data.reset_index(drop=True), 
                            sentiment_data.reset_index(drop=True)], axis=1)
        
        # Gestion des valeurs manquantes
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Exclure la colonne date pour le scaling
        if 'date' in data.columns:
            data = data.drop(columns=['date'])
        
        # Exclure les colonnes 'direction' ou autres colonnes texte
        for col in ['direction']:
            if col in data.columns:
                data = data.drop(columns=[col])
        
        # S'assurer que toutes les colonnes sont numériques
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col])
                except:
                    data = data.drop(columns=[col])
        
        logger.info(f"Dimensions data après prétraitement: {data.shape}")
        logger.info(f"Colonnes: {data.columns.tolist()}")
        
        return data 