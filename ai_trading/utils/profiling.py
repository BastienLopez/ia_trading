#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le profilage intensif des performances du code.
Contient des utilitaires pour profiler le code avec PyTorch Profiler, TensorFlow Profiler,
NVIDIA Nsight Systems, cProfile et scalene.
"""

import os
import time
import functools
import tempfile
import subprocess
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Union
import cProfile
import pstats
import io
import logging
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérification des dépendances optionnelles
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import scalene
    SCALENE_AVAILABLE = True
except ImportError:
    SCALENE_AVAILABLE = False

class ProfilingManager:
    """Classe pour gérer différentes méthodes de profilage."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialise le gestionnaire de profilage.
        
        Args:
            output_dir: Répertoire de sortie pour les rapports de profilage
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "ai_trading_profiles"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vérifier la disponibilité de NVIDIA Nsight
        self.nsight_available = self._check_nsight_available()
        
    def _check_nsight_available(self) -> bool:
        """Vérifie si NVIDIA Nsight Systems est disponible."""
        try:
            result = subprocess.run(["nsys", "--version"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def profile_with_cprofile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile une fonction avec cProfile.
        
        Args:
            func: Fonction à profiler
            *args, **kwargs: Arguments à passer à la fonction
            
        Returns:
            Résultats du profilage
        """
        timestamp = int(time.time())
        output_path = self.output_dir / f"cprofile_{timestamp}.prof"
        
        # Profiler l'exécution de la fonction
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Sauvegarder le profil
        profiler.dump_stats(str(output_path))
        
        # Générer un rapport texte
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 fonctions les plus lentes
        
        return {
            "result": result,
            "profile_path": str(output_path),
            "text_report": s.getvalue(),
            "top_functions": self._parse_cprofile_stats(s.getvalue())
        }
    
    def _parse_cprofile_stats(self, stats_output: str) -> List[Dict[str, Any]]:
        """Parse les statistiques cProfile en structure de données utilisable."""
        result = []
        
        # Diviser la sortie en lignes
        lines = stats_output.split("\n")
        
        # Trouver l'index de la ligne qui commence le tableau de données
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("ncalls"):
                start_idx = i + 1
                break
        
        if start_idx == -1 or start_idx >= len(lines):
            return result
        
        # Traiter chaque ligne de données
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            # Utiliser une expression régulière pour extraire les chiffres
            # Le format est généralement: ncalls tottime percall cumtime percall filename:lineno(function)
            match = re.match(r'(\d+/?\d*)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(.*)', line)
            if match:
                try:
                    result.append({
                        "ncalls": match.group(1),
                        "tottime": float(match.group(2)),
                        "percall": float(match.group(3)),
                        "cumtime": float(match.group(4)),
                        "percall_cum": float(match.group(5)),
                        "function": match.group(6)
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Erreur lors du parsing de la ligne '{line}': {e}")
        
        return result
    
    def profile_with_torch(self, model: 'torch.nn.Module', 
                           input_data: Any, 
                           warm_up: int = 3,
                           iterations: int = 10) -> Dict[str, Any]:
        """
        Profile un modèle PyTorch.
        
        Args:
            model: Modèle PyTorch à profiler
            input_data: Données d'entrée pour le modèle
            warm_up: Nombre d'itérations de préchauffage
            iterations: Nombre d'itérations pour le profilage
            
        Returns:
            Résultats du profilage
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch n'est pas disponible pour le profilage.")
            return {"error": "PyTorch n'est pas disponible"}
        
        timestamp = int(time.time())
        trace_path = self.output_dir / f"pytorch_trace_{timestamp}.json"
        
        # S'assurer que le modèle est en mode évaluation
        model.eval()
        
        # Préchauffement pour éviter les surcoûts initiaux
        with torch.no_grad():
            for _ in range(warm_up):
                _ = model(input_data)
        
        # Profilage réel avec PyTorch Profiler
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        prof.start()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_data)
        prof.stop()
        
        # Sauvegarder la trace Chrome
        prof.export_chrome_trace(str(trace_path))
        
        # Analyser les résultats du profilage
        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        
        return {
            "trace_path": str(trace_path),
            "table": table,
            "bottlenecks": self._analyze_torch_profile(prof),
            "memory_stats": self._extract_memory_stats(prof)
        }
    
    def _analyze_torch_profile(self, prof) -> List[Dict[str, Any]]:
        """Analyse les résultats du profilage PyTorch pour identifier les goulots d'étranglement."""
        result = []
        for item in prof.key_averages():
            if item.cpu_time_total > 0.01 or (hasattr(item, 'cuda_time_total') and item.cuda_time_total > 0.01):
                result.append({
                    "name": item.key,
                    "cpu_time": item.cpu_time_total,
                    "cuda_time": getattr(item, 'cuda_time_total', 0),
                    "cpu_memory": getattr(item, 'cpu_memory_usage', 0),
                    "cuda_memory": getattr(item, 'cuda_memory_usage', 0),
                })
        return sorted(result, key=lambda x: x["cpu_time"] + x["cuda_time"], reverse=True)
    
    def _extract_memory_stats(self, prof) -> Dict[str, Any]:
        """Extrait les statistiques mémoire du profilage PyTorch."""
        cpu_total = 0
        cuda_total = 0
        
        for item in prof.key_averages():
            cpu_total += getattr(item, 'cpu_memory_usage', 0)
            cuda_total += getattr(item, 'cuda_memory_usage', 0)
        
        return {
            "cpu_total_bytes": cpu_total,
            "cuda_total_bytes": cuda_total,
            "cpu_total_mb": cpu_total / (1024 * 1024),
            "cuda_total_mb": cuda_total / (1024 * 1024)
        }
    
    def profile_with_nsight(self, command: str) -> Dict[str, Any]:
        """
        Profile un programme avec NVIDIA Nsight Systems.
        
        Args:
            command: Commande à profiler
            
        Returns:
            Résultats du profilage
        """
        if not self.nsight_available:
            logger.error("NVIDIA Nsight Systems n'est pas disponible.")
            return {"error": "NVIDIA Nsight n'est pas disponible"}
        
        timestamp = int(time.time())
        output_path = self.output_dir / f"nsight_{timestamp}.qdrep"
        
        # Exécuter la commande avec Nsight
        try:
            nsys_cmd = f"nsys profile -o {output_path.with_suffix('')} {command}"
            result = subprocess.run(nsys_cmd, shell=True, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  text=True)
            
            return {
                "command": command,
                "output_path": str(output_path),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except subprocess.SubprocessError as e:
            logger.error(f"Erreur lors du profilage avec Nsight: {e}")
            return {"error": str(e)}
    
    def profile_with_scalene(self, command: str) -> Dict[str, Any]:
        """
        Profile un programme Python avec Scalene.
        
        Args:
            command: Commande Python à profiler
            
        Returns:
            Résultats du profilage
        """
        if not SCALENE_AVAILABLE:
            logger.error("Scalene n'est pas disponible pour le profilage.")
            return {"error": "Scalene n'est pas disponible"}
        
        timestamp = int(time.time())
        output_html = self.output_dir / f"scalene_{timestamp}.html"
        
        # Exécuter la commande avec Scalene
        try:
            scalene_cmd = f"python -m scalene --outfile {output_html} {command}"
            result = subprocess.run(scalene_cmd, shell=True, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  text=True)
            
            return {
                "command": command,
                "output_path": str(output_html),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except subprocess.SubprocessError as e:
            logger.error(f"Erreur lors du profilage avec Scalene: {e}")
            return {"error": str(e)}
    
    def profile_tensorflow(self, model: 'tf.keras.Model', 
                          input_data: Any,
                          warm_up: int = 3,
                          iterations: int = 10) -> Dict[str, Any]:
        """
        Profile un modèle TensorFlow.
        
        Args:
            model: Modèle TensorFlow à profiler
            input_data: Données d'entrée pour le modèle
            warm_up: Nombre d'itérations de préchauffage
            iterations: Nombre d'itérations pour le profilage
            
        Returns:
            Résultats du profilage
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow n'est pas disponible pour le profilage.")
            return {"error": "TensorFlow n'est pas disponible"}
        
        timestamp = int(time.time())
        log_dir = self.output_dir / f"tf_logs_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Préchauffement
        for _ in range(warm_up):
            _ = model(input_data)
        
        # Profilage avec TensorFlow Profiler
        tf.profiler.experimental.start(str(log_dir))
        for _ in range(iterations):
            _ = model(input_data)
        tf.profiler.experimental.stop()
        
        return {
            "log_dir": str(log_dir),
            "iterations": iterations,
            "success": True,
            "note": "Pour visualiser les résultats, exécutez: tensorboard --logdir=" + str(log_dir)
        }

def profile_function(method='cprofile'):
    """
    Décorateur pour profiler une fonction.
    
    Args:
        method: Méthode de profilage ('cprofile', 'time')
        
    Returns:
        Fonction décorée
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if method == 'cprofile':
                profiler = cProfile.Profile()
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
                
                # Imprimer les résultats
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)
                print(f"Profil de {func.__name__}:")
                print(s.getvalue())
                
                return result
            elif method == 'time':
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                print(f"Temps d'exécution pour {func.__name__}: {elapsed:.4f} secondes")
                
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Fonction utilitaire pour profiler rapidement un bloc de code
def profile_block(func=None, *, name=None, method='cprofile'):
    """
    Contexte pour profiler un bloc de code.
    
    Args:
        name: Nom du bloc de code
        method: Méthode de profilage ('cprofile', 'time')
        
    Exemple:
        with profile_block(name="Prétraitement des données", method="time"):
            # Code à profiler
            data = preprocess(raw_data)
    """
    if func is not None:
        return profile_function(method)(func)
        
    class ProfilerContext:
        def __init__(self, name, method):
            self.name = name or "Bloc anonyme"
            self.method = method
            self.profiler = None if method != 'cprofile' else cProfile.Profile()
            self.start_time = None
            
        def __enter__(self):
            if self.method == 'cprofile':
                self.profiler.enable()
            else:
                self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.method == 'cprofile':
                self.profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)
                print(f"Profil de {self.name}:")
                print(s.getvalue())
            else:
                elapsed = time.time() - self.start_time
                print(f"Temps d'exécution pour {self.name}: {elapsed:.4f} secondes")
    
    return ProfilerContext(name, method) 