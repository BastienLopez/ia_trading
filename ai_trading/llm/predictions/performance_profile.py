"""
Module d'optimisation des performances pour les modèles LLM.

Ce module fournit des outils pour profiler, analyser et optimiser les performances
des modèles de langage utilisés dans les prédictions de marché.
"""

import time
import os
import json
import traceback
import functools
import threading
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Configuration du logger
logger = logging.getLogger("llm_performance_optimizer")

class PerformanceProfiler:
    """
    Classe pour profiler et analyser les performances des modèles LLM.
    """
    
    def __init__(self, output_dir: str = None, enable_gpu_profiling: bool = True):
        """
        Initialise le profileur de performances.
        
        Args:
            output_dir: Répertoire où stocker les résultats du profilage
            enable_gpu_profiling: Activer le profilage GPU si disponible
        """
        self.output_dir = output_dir or "performance_profiles"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.enable_gpu_profiling = enable_gpu_profiling and torch.cuda.is_available()
        self.profiles = {}
        self.current_profile = None
        
        # Vérifier si NVIDIA Nsight est disponible pour un profilage avancé
        self.has_nsight = False
        try:
            import torch.cuda.nvtx as nvtx
            self.has_nsight = True
            self.nvtx = nvtx
            logger.info("NVIDIA Nsight support activé pour le profilage")
        except ImportError:
            logger.info("NVIDIA Nsight non disponible, utilisant des méthodes de profilage standard")
    
    def start_profile(self, name: str) -> None:
        """Commence un nouveau profil."""
        if self.current_profile:
            logger.warning(f"Un profil est déjà en cours: {self.current_profile}. Arrêt du profil précédent.")
            self.end_profile()
            
        self.current_profile = name
        
        # Début d'un nouveau profil
        profile_data = {
            "name": name,
            "start_time": time.time(),
            "events": [],
            "gpu_stats": [],
            "memory_stats": []
        }
        
        self.profiles[name] = profile_data
        
        # Marquer dans Nsight si disponible
        if self.has_nsight:
            self.nvtx.range_push(f"Profile: {name}")
            
        # Collecter les statistiques GPU initiales
        if self.enable_gpu_profiling:
            self._collect_gpu_stats()
    
    def end_profile(self) -> Dict[str, Any]:
        """Termine le profil en cours et retourne les résultats."""
        if not self.current_profile:
            logger.warning("Aucun profil en cours. Rien à terminer.")
            return {}
            
        profile_data = self.profiles[self.current_profile]
        profile_data["end_time"] = time.time()
        profile_data["duration"] = profile_data["end_time"] - profile_data["start_time"]
        
        # Collecter les statistiques GPU finales
        if self.enable_gpu_profiling:
            self._collect_gpu_stats()
            
        # Calculer les statistiques
        self._calculate_stats(profile_data)
        
        # Sauvegarder le profil
        self._save_profile(profile_data)
        
        # Terminer le marqueur Nsight
        if self.has_nsight:
            self.nvtx.range_pop()
        
        # Réinitialiser le profil courant
        current = self.current_profile
        self.current_profile = None
        
        return self.profiles[current]
    
    def mark_event(self, name: str, metadata: Dict[str, Any] = None) -> None:
        """Marque un événement dans le profil actuel."""
        if not self.current_profile:
            logger.warning(f"Aucun profil en cours. Impossible de marquer l'événement: {name}")
            return
            
        event = {
            "name": name,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.profiles[self.current_profile]["events"].append(event)
        
        # Collecter les statistiques GPU
        if self.enable_gpu_profiling:
            self._collect_gpu_stats()
            
        # Marquer dans Nsight si disponible
        if self.has_nsight:
            self.nvtx.mark(f"Event: {name}")
    
    def _collect_gpu_stats(self) -> None:
        """Collecte les statistiques GPU actuelles."""
        if not self.current_profile or not self.enable_gpu_profiling:
            return
            
        stats = {
            "timestamp": time.time(),
            "memory_allocated": 0,
            "memory_reserved": 0,
            "utilization": 0
        }
        
        try:
            stats["memory_allocated"] = torch.cuda.memory_allocated()
            stats["memory_reserved"] = torch.cuda.memory_reserved()
            
            # Essayer de collecter l'utilisation GPU si disponible
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["utilization"] = utilization.gpu
            except (ImportError, Exception):
                pass
                
        except Exception as e:
            logger.warning(f"Erreur lors de la collecte des statistiques GPU: {e}")
            
        self.profiles[self.current_profile]["gpu_stats"].append(stats)
    
    def _calculate_stats(self, profile_data: Dict[str, Any]) -> None:
        """Calcule les statistiques globales pour le profil."""
        events = profile_data["events"]
        gpu_stats = profile_data["gpu_stats"]
        
        # Statistiques d'événements
        event_durations = []
        for i in range(1, len(events)):
            duration = events[i]["timestamp"] - events[i-1]["timestamp"]
            event_durations.append(duration)
            events[i-1]["duration"] = duration
            
        if event_durations:
            profile_data["stats"] = {
                "event_count": len(events),
                "avg_event_duration": sum(event_durations) / len(event_durations),
                "max_event_duration": max(event_durations) if event_durations else 0,
                "min_event_duration": min(event_durations) if event_durations else 0
            }
        else:
            profile_data["stats"] = {
                "event_count": len(events),
                "avg_event_duration": 0,
                "max_event_duration": 0,
                "min_event_duration": 0
            }
            
        # Statistiques GPU
        if gpu_stats:
            memory_allocated = [stat["memory_allocated"] for stat in gpu_stats]
            memory_reserved = [stat["memory_reserved"] for stat in gpu_stats]
            utilization = [stat["utilization"] for stat in gpu_stats]
            
            profile_data["gpu_summary"] = {
                "peak_memory_allocated": max(memory_allocated) / (1024 * 1024),  # MB
                "avg_memory_allocated": sum(memory_allocated) / len(memory_allocated) / (1024 * 1024),  # MB
                "peak_memory_reserved": max(memory_reserved) / (1024 * 1024),  # MB
                "avg_utilization": sum(utilization) / len(utilization) if utilization[0] else 0
            }
    
    def _save_profile(self, profile_data: Dict[str, Any]) -> None:
        """Sauvegarde le profil au format JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{profile_data['name']}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convertir les données en format JSON
        json_data = {
            "name": profile_data["name"],
            "duration": profile_data["duration"],
            "start_time": profile_data["start_time"],
            "end_time": profile_data["end_time"],
            "events": profile_data["events"],
            "stats": profile_data.get("stats", {}),
        }
        
        if "gpu_summary" in profile_data:
            json_data["gpu_summary"] = profile_data["gpu_summary"]
            
        try:
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.info(f"Profil sauvegardé: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du profil: {e}")
    
    def generate_report(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Génère un rapport détaillé sur les performances."""
        if profile_name and profile_name in self.profiles:
            profiles_to_analyze = {profile_name: self.profiles[profile_name]}
        else:
            profiles_to_analyze = self.profiles
            
        if not profiles_to_analyze:
            logger.warning("Aucun profil disponible pour générer un rapport.")
            return {}
            
        report = {
            "generated_at": datetime.now().isoformat(),
            "profile_count": len(profiles_to_analyze),
            "profiles": []
        }
        
        for name, profile in profiles_to_analyze.items():
            profile_report = {
                "name": name,
                "duration": profile.get("duration", 0),
                "event_count": len(profile.get("events", [])),
                "stats": profile.get("stats", {}),
                "gpu_summary": profile.get("gpu_summary", {})
            }
            
            # Ajouter les événements les plus lents
            events = profile.get("events", [])
            slow_events = []
            
            for i in range(len(events) - 1):
                if "duration" in events[i]:
                    slow_events.append({
                        "name": events[i]["name"],
                        "duration": events[i]["duration"]
                    })
                    
            # Trier et prendre les 5 plus lents
            slow_events.sort(key=lambda x: x["duration"], reverse=True)
            profile_report["slowest_events"] = slow_events[:5]
            
            report["profiles"].append(profile_report)
            
        return report

# Décorateur pour profiler automatiquement une fonction
def profile(output_dir: str = None):
    """
    Décorateur pour profiler automatiquement une fonction.
    
    Args:
        output_dir: Répertoire où stocker les résultats du profilage
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler(output_dir=output_dir)
            profiler.start_profile(func.__name__)
            
            try:
                # Marquer le début de l'exécution de la fonction
                profiler.mark_event("function_start")
                
                # Exécuter la fonction
                result = func(*args, **kwargs)
                
                # Marquer la fin de l'exécution de la fonction
                profiler.mark_event("function_end")
                
                return result
            except Exception as e:
                # Marquer l'erreur
                profiler.mark_event("function_error", {"error": str(e), "traceback": traceback.format_exc()})
                raise
            finally:
                # Terminer le profil
                profile_data = profiler.end_profile()
                
                # Générer un rapport
                report = profiler.generate_report(func.__name__)
                
                # Log des informations de performance
                if "gpu_summary" in profile_data:
                    gpu_info = profile_data["gpu_summary"]
                    logger.info(f"Performance {func.__name__}: {profile_data['duration']:.4f}s, "
                               f"Mémoire GPU max: {gpu_info.get('peak_memory_allocated', 0):.2f} MB")
                else:
                    logger.info(f"Performance {func.__name__}: {profile_data['duration']:.4f}s")
                
        return wrapper
    return decorator

class PerformanceOptimizer:
    """
    Classe pour optimiser les performances des modèles LLM.
    """
    
    def __init__(self, 
                 model_name: str = None, 
                 use_gpu: bool = True, 
                 enable_cache: bool = True,
                 enable_batching: bool = True,
                 batch_size: int = 8):
        """
        Initialise l'optimiseur de performances.
        
        Args:
            model_name: Nom du modèle à optimiser
            use_gpu: Utiliser le GPU si disponible
            enable_cache: Activer le cache des inférences
            enable_batching: Activer le traitement par lots
            batch_size: Taille des lots pour le traitement par lots
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_cache = enable_cache
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        
        # Cache d'inférence
        self.inference_cache = {}
        
        # Statistiques de performances
        self.perf_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_inference_time": 0,
            "inference_count": 0,
            "batches_processed": 0
        }
        
    def get_optimal_batch_size(self, model, sample_input, min_batch=1, max_batch=32):
        """
        Détermine la taille de lot optimale pour maximiser le débit.
        
        Args:
            model: Le modèle à analyser
            sample_input: Un exemple d'entrée pour le modèle
            min_batch: Taille minimale de lot à tester
            max_batch: Taille maximale de lot à tester
            
        Returns:
            Taille de lot optimale
        """
        if not self.use_gpu:
            logger.info("GPU non disponible, utilisation de la taille de lot par défaut")
            return self.batch_size
        
        logger.info(f"Recherche de la taille de lot optimale entre {min_batch} et {max_batch}...")
        
        results = []
        
        # Tester différentes tailles de lots
        for batch_size in range(min_batch, max_batch + 1, 2):
            # Créer un lot d'entrées
            if isinstance(sample_input, torch.Tensor):
                batch_input = sample_input.repeat(batch_size, *([1] * len(sample_input.shape[1:])))
            else:
                # Supposons que c'est un dictionnaire d'entrées
                batch_input = [sample_input] * batch_size
                
            # Mesurer le temps d'inférence
            with torch.cuda.amp.autocast(enabled=self.use_gpu):
                torch.cuda.synchronize()
                start_time = time.time()
                
                # Effectuer plusieurs passes pour obtenir une mesure plus stable
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(batch_input)
                    
                torch.cuda.synchronize()
                end_time = time.time()
                
            # Calculer le temps par échantillon
            total_time = end_time - start_time
            time_per_sample = total_time / (5 * batch_size)
            throughput = batch_size / total_time
            
            results.append({
                "batch_size": batch_size,
                "total_time": total_time,
                "time_per_sample": time_per_sample,
                "throughput": throughput
            })
            
            logger.info(f"Batch size {batch_size}: {time_per_sample:.6f}s par échantillon, throughput: {throughput:.2f} échantillons/s")
            
        # Trouver la taille de lot qui maximise le débit
        optimal_batch = max(results, key=lambda x: x["throughput"])
        
        logger.info(f"Taille de lot optimale: {optimal_batch['batch_size']} "
                   f"(débit: {optimal_batch['throughput']:.2f} échantillons/s)")
        
        return optimal_batch["batch_size"]
        
    def optimize_model(self, model):
        """
        Applique des optimisations au modèle.
        
        Args:
            model: Le modèle à optimiser
            
        Returns:
            Modèle optimisé
        """
        if not self.use_gpu:
            logger.info("Optimisations GPU désactivées ou GPU non disponible")
            return model
            
        logger.info(f"Optimisation du modèle {self.model_name or 'inconnu'}...")
        
        # Passage du modèle en mode d'évaluation
        model.eval()
        
        # Passage du modèle sur GPU
        model.to("cuda")
        
        # Fusion des couches batch norm si disponible
        if hasattr(torch, 'nn') and hasattr(torch.nn, 'utils') and hasattr(torch.nn.utils, 'fusion'):
            try:
                torch.nn.utils.fusion.fuse_conv_bn_eval(model)
                logger.info("Fusion des couches BatchNorm effectuée")
            except Exception as e:
                logger.warning(f"Erreur lors de la fusion des couches BatchNorm: {e}")
                
        # Optimisation torch.compile si disponible (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Compilation torch.compile appliquée")
            except Exception as e:
                logger.warning(f"Erreur lors de la compilation torch.compile: {e}")
                
        # TorchScript JIT
        try:
            # Essayer d'abord de tracer le modèle
            sample_input = torch.rand(1, 3, 224, 224).cuda()  # Exemple d'entrée, à adapter
            traced_model = torch.jit.trace(model, sample_input)
            model = traced_model
            logger.info("TorchScript trace appliqué avec succès")
        except Exception as e:
            logger.warning(f"Erreur lors du traçage TorchScript: {e}")
            
            # Essayer le scripting si le tracing échoue
            try:
                scripted_model = torch.jit.script(model)
                model = scripted_model
                logger.info("TorchScript script appliqué avec succès")
            except Exception as e:
                logger.warning(f"Erreur lors du scripting TorchScript: {e}")
                
        # Définir le modèle comme persistant pour éviter de décharger/recharger
        if hasattr(model, 'is_persistable'):
            model.is_persistable = True
            
        # Synchroniser pour s'assurer que tout est chargé sur GPU
        torch.cuda.synchronize()
            
        return model
        
    def setup_autocast(self):
        """
        Configure le contexte d'autocast pour l'inférence.
        """
        if not self.use_gpu:
            return lambda: DummyContextManager()
            
        # Vérifier si mixed precision est disponible
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            return lambda: torch.cuda.amp.autocast(enabled=True)
        else:
            return lambda: DummyContextManager()
            
    def optimize_inference(self, func: Callable) -> Callable:
        """
        Optimise une fonction d'inférence.
        
        Args:
            func: Fonction d'inférence à optimiser
            
        Returns:
            Fonction d'inférence optimisée
        """
        @functools.wraps(func)
        def optimized_func(*args, **kwargs):
            # Vérifier le cache si activé
            if self.enable_cache:
                # Créer une clé de cache basée sur les arguments
                cache_key = self._generate_cache_key(args, kwargs)
                
                # Vérifier si le résultat est déjà en cache
                if cache_key in self.inference_cache:
                    self.perf_stats["cache_hits"] += 1
                    return self.inference_cache[cache_key]
                    
                self.perf_stats["cache_misses"] += 1
                
            # Mesurer le temps d'inférence
            start_time = time.time()
            
            # Utiliser autocast si disponible
            with self.setup_autocast()():
                result = func(*args, **kwargs)
                
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Mettre à jour les statistiques
            self.perf_stats["total_inference_time"] += inference_time
            self.perf_stats["inference_count"] += 1
            
            # Mettre en cache le résultat si activé
            if self.enable_cache:
                self.inference_cache[cache_key] = result
                
            return result
            
        return optimized_func
        
    def batch_process(self, func: Callable, inputs: List[Any]) -> List[Any]:
        """
        Traite une liste d'entrées par lots.
        
        Args:
            func: Fonction à appliquer à chaque lot
            inputs: Liste d'entrées à traiter
            
        Returns:
            Liste des résultats
        """
        if not self.enable_batching or len(inputs) <= 1:
            # Traitement individuel si le batching est désactivé ou s'il n'y a qu'une entrée
            return [func(input_) for input_ in inputs]
            
        results = []
        batch_size = min(self.batch_size, len(inputs))
        
        # Traiter par lots
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            # Mesurer le temps de traitement du lot
            start_time = time.time()
            
            # Traiter le lot
            batch_results = func(batch)
            
            end_time = time.time()
            batch_time = end_time - start_time
            
            # Mettre à jour les statistiques
            self.perf_stats["batches_processed"] += 1
            
            # Ajouter les résultats du lot
            results.extend(batch_results)
            
            logger.debug(f"Lot traité en {batch_time:.4f}s (taille: {len(batch)})")
            
        return results
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            "cache_enabled": self.enable_cache,
            "cache_size": len(self.inference_cache) if self.enable_cache else 0,
            "cache_hits": self.perf_stats["cache_hits"],
            "cache_misses": self.perf_stats["cache_misses"],
            "hit_ratio": 0.0
        }
        
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["hit_ratio"] = stats["cache_hits"] / total_requests
            
        return stats
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de performance.
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = dict(self.perf_stats)
        
        # Calculer le temps moyen d'inférence
        if stats["inference_count"] > 0:
            stats["avg_inference_time"] = stats["total_inference_time"] / stats["inference_count"]
        else:
            stats["avg_inference_time"] = 0
            
        # Ajouter les statistiques GPU si disponible
        if self.use_gpu:
            stats["gpu_info"] = {
                "device": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated() / (1024 * 1024),  # MB
                "memory_reserved": torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            }
            
        return stats
    
    def _generate_cache_key(self, args: tuple, kwargs: dict) -> str:
        """
        Génère une clé de cache unique pour les arguments donnés.
        
        Args:
            args: Arguments positionnels
            kwargs: Arguments nommés
            
        Returns:
            Clé de cache
        """
        # Version simple: utiliser une représentation string des arguments
        # Pour une version plus robuste, utiliser des hash de contenu ou une sérialisation
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        return f"{args_str}_{kwargs_str}"
    
    def clear_cache(self) -> None:
        """Vide le cache d'inférence."""
        self.inference_cache.clear()
        logger.info("Cache d'inférence vidé")
        
        # Réinitialiser les statistiques de cache
        self.perf_stats["cache_hits"] = 0
        self.perf_stats["cache_misses"] = 0
        
        # Forcer la libération de mémoire GPU si possible
        if self.use_gpu:
            torch.cuda.empty_cache()
            logger.info("Cache GPU vidé")

# Classes utilitaires
class DummyContextManager:
    """Gestionnaire de contexte qui ne fait rien."""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Point d'entrée principal
def optimize_llm_pipeline(model_name: str, 
                         use_gpu: bool = True,
                         enable_cache: bool = True,
                         batch_size: int = 8) -> PerformanceOptimizer:
    """
    Point d'entrée principal pour optimiser un pipeline LLM.
    
    Args:
        model_name: Nom du modèle à optimiser
        use_gpu: Utiliser le GPU si disponible
        enable_cache: Activer le cache des inférences
        batch_size: Taille des lots pour le traitement par lots
        
    Returns:
        Optimiseur de performances configuré
    """
    logger.info(f"Configuration de l'optimisation pour le modèle {model_name}")
    
    optimizer = PerformanceOptimizer(
        model_name=model_name,
        use_gpu=use_gpu,
        enable_cache=enable_cache,
        batch_size=batch_size
    )
    
    # Configurer le logging
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    
    # Afficher la configuration
    logger.info(f"Optimiseur initialisé avec les paramètres suivants:")
    logger.info(f"  - Modèle: {model_name}")
    logger.info(f"  - GPU: {'Activé' if optimizer.use_gpu else 'Désactivé'}")
    logger.info(f"  - Cache: {'Activé' if enable_cache else 'Désactivé'}")
    logger.info(f"  - Batch size: {batch_size}")
    
    return optimizer 