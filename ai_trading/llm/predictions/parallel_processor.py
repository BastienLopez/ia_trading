"""
Module de traitement parallèle pour les prédictions de marché.

Ce module implémente des fonctionnalités de parallélisation pour accélérer 
les opérations coûteuses comme l'entraînement des modèles et les prédictions d'ensemble.
"""

import concurrent.futures
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
import os

import numpy as np
import pandas as pd

# Configuration du logger
from ai_trading.utils import setup_logger
logger = setup_logger("parallel_processor")

class ParallelProcessor:
    """
    Processeur parallèle pour les opérations de prédiction de marché.
    
    Permet la parallélisation des opérations coûteuses comme l'évaluation
    des modèles d'ensemble, les prédictions batch, etc.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = True):
        """
        Initialise le processeur parallèle.
        
        Args:
            max_workers: Nombre maximal de workers (None = auto-détection)
            use_processes: Utiliser des processus plutôt que des threads
        """
        # Détermination du nombre de workers
        self.max_workers = max_workers or self._detect_optimal_workers()
        self.use_processes = use_processes
        
        # Variables de suivi des tâches
        self.active_tasks = {}
        self.task_results = {}
        self.task_lock = threading.RLock()
        self.next_task_id = 0
        
        logger.info(f"ParallelProcessor initialisé avec {self.max_workers} workers, mode: {'processus' if use_processes else 'threads'}")
    
    def _detect_optimal_workers(self) -> int:
        """
        Détecte le nombre optimal de workers.
        
        Returns:
            Nombre optimal de workers
        """
        # Utiliser le nombre de cœurs logiques, avec une limite haute pour éviter la surcharge
        cpu_count = os.cpu_count() or 4
        return max(1, min(cpu_count - 1, 16))  # Garder 1 cœur pour l'OS, max 16
    
    def map(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        Applique une fonction à chaque élément de la liste en parallèle.
        
        Args:
            func: Fonction à appliquer
            items: Liste d'éléments à traiter
            *args, **kwargs: Arguments supplémentaires à passer à la fonction
            
        Returns:
            Liste des résultats dans le même ordre que les éléments d'entrée
        """
        if not items:
            return []
        
        # Création d'une fonction partielle avec les arguments additionnels
        if args or kwargs:
            func = partial(func, *args, **kwargs)
        
        executor_class = concurrent.futures.ProcessPoolExecutor if self.use_processes else concurrent.futures.ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Soumission des tâches
            future_to_index = {executor.submit(func, item): i for i, item in enumerate(items)}
            
            # Préparation du résultat dans le bon ordre
            results = [None] * len(items)
            
            # Récupération des résultats
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Erreur lors de l'exécution de la tâche {index}: {e}")
                    results[index] = None
        
        return results
    
    def submit_task(self, func: Callable, *args, **kwargs) -> int:
        """
        Soumet une tâche à exécuter en arrière-plan.
        
        Args:
            func: Fonction à exécuter
            *args, **kwargs: Arguments à passer à la fonction
            
        Returns:
            ID de la tâche
        """
        with self.task_lock:
            task_id = self.next_task_id
            self.next_task_id += 1
            
            # Démarrage d'un thread pour gérer la tâche
            task_thread = threading.Thread(
                target=self._execute_task,
                args=(task_id, func, args, kwargs),
                daemon=True
            )
            
            self.active_tasks[task_id] = {
                "thread": task_thread,
                "status": "pending",
                "start_time": time.time()
            }
            
            task_thread.start()
            
            logger.debug(f"Tâche {task_id} soumise: {func.__name__}")
            return task_id
    
    def _execute_task(self, task_id: int, func: Callable, args: Tuple, kwargs: Dict) -> None:
        """
        Exécute une tâche en arrière-plan.
        
        Args:
            task_id: ID de la tâche
            func: Fonction à exécuter
            args: Arguments positionnels
            kwargs: Arguments nommés
        """
        executor_class = concurrent.futures.ProcessPoolExecutor if self.use_processes else concurrent.futures.ThreadPoolExecutor
        
        with self.task_lock:
            self.active_tasks[task_id]["status"] = "running"
        
        try:
            with executor_class(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                result = future.result()
                
                with self.task_lock:
                    self.task_results[task_id] = {
                        "result": result,
                        "success": True,
                        "error": None,
                        "end_time": time.time()
                    }
                    self.active_tasks[task_id]["status"] = "completed"
                
                logger.debug(f"Tâche {task_id} terminée avec succès")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la tâche {task_id}: {e}")
            
            with self.task_lock:
                self.task_results[task_id] = {
                    "result": None,
                    "success": False,
                    "error": str(e),
                    "end_time": time.time()
                }
                self.active_tasks[task_id]["status"] = "failed"
    
    def get_task_result(self, task_id: int, wait: bool = True, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Récupère le résultat d'une tâche.
        
        Args:
            task_id: ID de la tâche
            wait: Attendre la fin de la tâche si elle est en cours
            timeout: Délai d'attente maximal en secondes
            
        Returns:
            Dictionnaire contenant le résultat et les métadonnées
        """
        if task_id not in self.active_tasks:
            return {"success": False, "error": "Tâche inconnue", "result": None}
        
        # Si la tâche est déjà terminée, renvoyer le résultat
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        # Si on ne veut pas attendre, renvoyer le statut actuel
        if not wait:
            return {"success": None, "status": self.active_tasks[task_id]["status"]}
        
        # Attente de la fin de la tâche
        task_thread = self.active_tasks[task_id]["thread"]
        task_thread.join(timeout)
        
        # Vérification si la tâche est terminée
        if task_id in self.task_results:
            return self.task_results[task_id]
        else:
            return {"success": False, "error": "Timeout", "result": None}
    
    def get_all_tasks(self) -> Dict[int, Dict[str, Any]]:
        """
        Récupère l'état de toutes les tâches.
        
        Returns:
            Dictionnaire des tâches indexé par ID
        """
        with self.task_lock:
            tasks = {}
            
            for task_id, task_info in self.active_tasks.items():
                task_data = {
                    "status": task_info["status"],
                    "start_time": task_info["start_time"],
                    "elapsed": time.time() - task_info["start_time"]
                }
                
                if task_id in self.task_results:
                    task_data.update({
                        "success": self.task_results[task_id]["success"],
                        "end_time": self.task_results[task_id]["end_time"],
                        "duration": self.task_results[task_id]["end_time"] - task_info["start_time"]
                    })
                
                tasks[task_id] = task_data
            
            return tasks
    
    def cancel_task(self, task_id: int) -> bool:
        """
        Tente d'annuler une tâche en cours.
        
        Note: L'annulation n'est pas toujours possible selon l'état de la tâche.
        
        Args:
            task_id: ID de la tâche à annuler
            
        Returns:
            True si l'annulation a été initiée, False sinon
        """
        with self.task_lock:
            if task_id not in self.active_tasks:
                return False
            
            if self.active_tasks[task_id]["status"] in ["completed", "failed"]:
                return False
            
            self.active_tasks[task_id]["status"] = "cancelling"
            
            # Note: nous ne pouvons pas vraiment arrêter un thread/processus en Python
            # sauf si la fonction implémente une vérification de l'annulation
            # Cette méthode met simplement à jour le statut
            
            return True
    
    def cleanup(self, max_age: Optional[float] = None) -> int:
        """
        Nettoie les tâches terminées.
        
        Args:
            max_age: Âge maximal en secondes des tâches à conserver
            
        Returns:
            Nombre de tâches nettoyées
        """
        with self.task_lock:
            current_time = time.time()
            to_remove = []
            
            for task_id, task_info in self.active_tasks.items():
                if task_info["status"] in ["completed", "failed"]:
                    if max_age is None or (current_time - task_info["start_time"]) > max_age:
                        to_remove.append(task_id)
            
            # Suppression des tâches
            for task_id in to_remove:
                del self.active_tasks[task_id]
                if task_id in self.task_results:
                    del self.task_results[task_id]
            
            logger.debug(f"{len(to_remove)} tâches nettoyées")
            return len(to_remove)

class EnsembleParallelProcessor:
    """
    Processeur spécialisé pour les opérations d'ensemble de modèles.
    
    Permet de paralléliser les prédictions, l'entraînement et l'évaluation
    d'ensembles de modèles de machine learning.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialise le processeur d'ensemble.
        
        Args:
            max_workers: Nombre maximal de workers
        """
        self.processor = ParallelProcessor(max_workers=max_workers, use_processes=True)
        
        logger.info(f"EnsembleParallelProcessor initialisé avec {self.processor.max_workers} workers")
    
    def train_models(self, 
                    models: List[Any], 
                    train_data: pd.DataFrame, 
                    target_column: str,
                    **train_params) -> List[Any]:
        """
        Entraîne une liste de modèles en parallèle.
        
        Args:
            models: Liste des modèles à entraîner
            train_data: Données d'entraînement
            target_column: Nom de la colonne cible
            **train_params: Paramètres d'entraînement supplémentaires
            
        Returns:
            Liste des modèles entraînés
        """
        if not models:
            return []
        
        logger.info(f"Entraînement parallèle de {len(models)} modèles")
        
        # Vérification de l'unicité des features
        X = train_data.drop(columns=[target_column])
        y = train_data[target_column]
        
        # Fonction d'entraînement
        def _train_model(model, X, y, **params):
            try:
                # Cloner le modèle pour éviter les modifications en place si nécessaire
                if hasattr(model, "clone") and callable(model.clone):
                    model_copy = model.clone()
                else:
                    model_copy = model
                
                # Entraînement
                model_copy.fit(X, y, **params)
                return model_copy
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
                return None
        
        # Entraînement parallèle
        trained_models = self.processor.map(_train_model, models, X=X, y=y, **train_params)
        
        # Filtrage des modèles échoués
        trained_models = [model for model in trained_models if model is not None]
        
        logger.info(f"{len(trained_models)} modèles entraînés avec succès")
        return trained_models
    
    def predict_parallel(self, 
                       models: List[Any], 
                       data: pd.DataFrame,
                       predict_method: str = "predict") -> np.ndarray:
        """
        Effectue des prédictions parallèles avec un ensemble de modèles.
        
        Args:
            models: Liste des modèles
            data: Données pour la prédiction
            predict_method: Nom de la méthode de prédiction ("predict", "predict_proba", etc.)
            
        Returns:
            Tableau des prédictions de chaque modèle
        """
        if not models:
            return np.array([])
        
        logger.info(f"Prédiction parallèle avec {len(models)} modèles")
        
        # Fonction de prédiction
        def _predict_with_model(model, data, method):
            try:
                predict_fn = getattr(model, method)
                if predict_fn:
                    return predict_fn(data)
                else:
                    logger.warning(f"Méthode {method} non trouvée sur le modèle")
                    return None
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction avec le modèle: {e}")
                return None
        
        # Prédiction parallèle
        predictions = self.processor.map(_predict_with_model, models, data=data, method=predict_method)
        
        # Filtrage des prédictions échouées
        valid_predictions = [pred for pred in predictions if pred is not None]
        
        if len(valid_predictions) != len(models):
            logger.warning(f"Certaines prédictions ont échoué: {len(valid_predictions)}/{len(models)} réussies")
        
        # Conversion en tableau numpy
        return np.array(valid_predictions)
    
    def evaluate_models(self, 
                       models: List[Any], 
                       test_data: pd.DataFrame, 
                       target_column: str,
                       metrics: List[Callable]) -> List[Dict[str, float]]:
        """
        Évalue une liste de modèles en parallèle.
        
        Args:
            models: Liste des modèles à évaluer
            test_data: Données de test
            target_column: Nom de la colonne cible
            metrics: Liste des fonctions de métrique (score)
            
        Returns:
            Liste des résultats d'évaluation pour chaque modèle
        """
        if not models:
            return []
        
        logger.info(f"Évaluation parallèle de {len(models)} modèles")
        
        # Préparation des données
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        # Fonction d'évaluation
        def _evaluate_model(model, X, y, metrics):
            try:
                results = {}
                
                # Prédictions
                y_pred = model.predict(X)
                
                # Calcul des métriques
                for metric in metrics:
                    metric_name = metric.__name__
                    results[metric_name] = metric(y, y_pred)
                
                # Ajout de fonctionnalités supplémentaires si disponibles
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X)
                        
                        # Métriques basées sur les probabilités si applicables
                        for metric in metrics:
                            if getattr(metric, "needs_proba", False):
                                metric_name = f"{metric.__name__}_proba"
                                results[metric_name] = metric(y, y_proba)
                    except:
                        pass
                
                return results
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation du modèle: {e}")
                return {}
        
        # Évaluation parallèle
        evaluations = self.processor.map(_evaluate_model, models, X=X_test, y=y_test, metrics=metrics)
        
        return evaluations
    
    def ensemble_predict(self, 
                        models: List[Any], 
                        weights: Optional[List[float]] = None,
                        data: pd.DataFrame = None,
                        aggregation: str = "mean") -> np.ndarray:
        """
        Produit une prédiction d'ensemble avec pondération.
        
        Args:
            models: Liste des modèles
            weights: Poids pour chaque modèle (None = poids égaux)
            data: Données pour la prédiction
            aggregation: Méthode d'agrégation ("mean", "median", "max_vote")
            
        Returns:
            Prédiction agrégée
        """
        if not models:
            return np.array([])
        
        # Normalisation des poids
        if weights is None:
            weights = np.ones(len(models)) / len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Le nombre de poids doit correspondre au nombre de modèles")
            weights = np.array(weights) / np.sum(weights)
        
        # Obtention des prédictions
        predictions = self.predict_parallel(models, data, "predict_proba" if aggregation == "mean" else "predict")
        
        if len(predictions) == 0:
            return np.array([])
        
        # Agrégation des prédictions
        if aggregation == "mean":
            # Moyenne pondérée des probabilités
            weighted_preds = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_preds += weights[i] * pred
            
            # Retourne la classe avec la probabilité maximale
            return np.argmax(weighted_preds, axis=1)
        
        elif aggregation == "median":
            # Médiane des prédictions
            return np.median(predictions, axis=0).astype(int)
        
        elif aggregation == "max_vote":
            # Vote majoritaire
            # Conversion en tableau pour vote
            votes = np.array(predictions).astype(int)
            
            # Compte des votes pour chaque classe par instance
            results = []
            for i in range(votes.shape[1]):
                unique, counts = np.unique(votes[:, i], return_counts=True)
                results.append(unique[np.argmax(counts)])
            
            return np.array(results)
        
        else:
            raise ValueError(f"Méthode d'agrégation inconnue: {aggregation}")
    
    def batch_process(self, 
                     func: Callable, 
                     data: pd.DataFrame, 
                     batch_size: int = 1000, 
                     **kwargs) -> List[Any]:
        """
        Traite les données en batch pour éviter les problèmes de mémoire.
        
        Args:
            func: Fonction à appliquer à chaque batch
            data: Données à traiter
            batch_size: Taille des batchs
            **kwargs: Arguments supplémentaires pour la fonction
            
        Returns:
            Liste des résultats pour chaque batch
        """
        if len(data) == 0:
            return []
        
        # Calcul du nombre de batchs
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(f"Traitement des données en {num_batches} batchs de taille {batch_size}")
        
        results = []
        
        # Traitement par batch
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            
            batch = data.iloc[start_idx:end_idx]
            
            # Soumission du batch comme tâche
            task_id = self.processor.submit_task(func, batch, **kwargs)
            
            # Récupération du résultat
            task_result = self.processor.get_task_result(task_id)
            
            if task_result["success"]:
                results.append(task_result["result"])
            else:
                logger.error(f"Erreur lors du traitement du batch {i}: {task_result.get('error')}")
                results.append(None)
        
        # Nettoyage des tâches terminées
        self.processor.cleanup()
        
        return results 