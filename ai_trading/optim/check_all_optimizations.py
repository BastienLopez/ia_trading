#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour vérifier l'application de toutes les optimisations CPU et GPU dans le projet.
Ce script analyse l'ensemble du codebase pour s'assurer que toutes les optimisations 
possibles sont correctement implémentées et actives.
"""

import os
import sys
import importlib
import logging
import pkgutil
import inspect
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Répertoire racine du projet (en remontant d'un niveau depuis ce fichier)
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Liste des optimisations à vérifier
OPTIMIZATIONS = {
    "intel_optimisations": {
        "module": "ai_trading.utils.intel_optimizations",
        "functions": ["optimize_for_intel", "configure_torch", "configure_numpy", "configure_tensorflow"],
        "description": "Optimisations pour les processeurs Intel (MKL, OpenMP)"
    },
    "threading_optimisations": {
        "module": "ai_trading.utils.threading_optimizer",
        "functions": ["calculate_optimal_workers", "configure_thread_limits", "set_process_priority"],
        "description": "Optimisations de multithreading/multiprocessing"
    },
    "gpu_rtx_optimisations": {
        "module": "ai_trading.utils.gpu_rtx_optimizer",
        "functions": ["setup_rtx_optimization", "optimize_batch_size"],
        "description": "Optimisations pour les GPU RTX (Tensor Cores, allocation mémoire)"
    },
    "compression_zstd": {
        "module": "ai_trading.data.compressed_storage", 
        "classes": ["CompressedStorage", "OptimizedFinancialDataset"],
        "description": "Compression de fichiers avec zstd"
    }
}

# Fichiers importants qui devraient utiliser les optimisations
KEY_FILES = [
    "ai_trading/train.py",
    "ai_trading/data_processor.py",
    "ai_trading/rl_agent.py",
    "ai_trading/api.py",
    "ai_trading/data/financial_dataset.py",
    "ai_trading/data/optimized_dataset.py",
    "ai_trading/models/lstm_model.py",
    "ai_trading/models/transformer_model.py"
]

def check_module_imports(file_path: Path) -> Dict[str, bool]:
    """
    Vérifie si un fichier Python importe les modules d'optimisation.
    
    Args:
        file_path: Chemin du fichier à vérifier
        
    Returns:
        Dictionnaire des modules d'optimisation importés
    """
    if not file_path.exists() or file_path.suffix != '.py':
        return {}
    
    imports = {}
    for opt_name, opt_info in OPTIMIZATIONS.items():
        imports[opt_name] = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for opt_name, opt_info in OPTIMIZATIONS.items():
                module_name = opt_info["module"]
                # Vérifier les différentes formes d'import possibles
                if f"import {module_name}" in content or \
                   f"from {module_name}" in content or \
                   f"from {'.'.join(module_name.split('.')[:-1])} import {module_name.split('.')[-1]}" in content:
                    imports[opt_name] = True
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du fichier {file_path}: {e}")
    
    return imports

def check_function_usage(file_path: Path) -> Dict[str, Set[str]]:
    """
    Vérifie les fonctions d'optimisation utilisées dans un fichier.
    
    Args:
        file_path: Chemin du fichier à vérifier
        
    Returns:
        Dictionnaire des fonctions d'optimisation utilisées par module
    """
    if not file_path.exists() or file_path.suffix != '.py':
        return {}
    
    function_usage = {}
    for opt_name, opt_info in OPTIMIZATIONS.items():
        function_usage[opt_name] = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for opt_name, opt_info in OPTIMIZATIONS.items():
                # Vérifier l'utilisation des fonctions
                if "functions" in opt_info:
                    for func in opt_info["functions"]:
                        if f"{func}(" in content:
                            function_usage[opt_name].add(func)
                
                # Vérifier l'utilisation des classes
                if "classes" in opt_info:
                    for cls in opt_info["classes"]:
                        if f"{cls}(" in content or f"class {cls}" in content:
                            function_usage[opt_name].add(cls)
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du fichier {file_path}: {e}")
    
    return function_usage

def scan_directory(directory: Path) -> Dict[str, Dict[str, Any]]:
    """
    Analyse récursivement un répertoire pour vérifier l'utilisation des optimisations.
    
    Args:
        directory: Répertoire à analyser
        
    Returns:
        Statistiques sur l'utilisation des optimisations
    """
    stats = {
        "files_checked": 0,
        "optimization_usage": {opt_name: 0 for opt_name in OPTIMIZATIONS},
        "optimization_functions": {opt_name: {func: 0 for func in opt_info.get("functions", [])} 
                                  for opt_name, opt_info in OPTIMIZATIONS.items()},
        "key_files_with_optimizations": {},
        "missing_optimizations": []
    }
    
    # Ajouter les classes aux statistiques
    for opt_name, opt_info in OPTIMIZATIONS.items():
        if "classes" in opt_info:
            for cls in opt_info["classes"]:
                stats["optimization_functions"][opt_name][cls] = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(os.path.join(root, file))
                relative_path = file_path.relative_to(PROJECT_ROOT)
                
                # Vérifier les imports
                imports = check_module_imports(file_path)
                
                # Vérifier l'utilisation des fonctions
                functions = check_function_usage(file_path)
                
                # Mettre à jour les statistiques
                stats["files_checked"] += 1
                for opt_name, imported in imports.items():
                    if imported:
                        stats["optimization_usage"][opt_name] += 1
                
                for opt_name, used_funcs in functions.items():
                    for func in used_funcs:
                        if func in stats["optimization_functions"][opt_name]:
                            stats["optimization_functions"][opt_name][func] += 1
                
                # Vérifier si c'est un fichier clé
                str_path = str(relative_path)
                if str_path in KEY_FILES:
                    stats["key_files_with_optimizations"][str_path] = {
                        "imports": imports,
                        "functions": {opt: list(funcs) for opt, funcs in functions.items()}
                    }
                    
                    # Vérifier les optimisations manquantes dans les fichiers clés
                    missing = []
                    for opt_name, opt_info in OPTIMIZATIONS.items():
                        if not imports[opt_name] and not functions[opt_name]:
                            missing.append(opt_name)
                    
                    if missing:
                        stats["missing_optimizations"].append({
                            "file": str_path,
                            "missing": missing
                        })
    
    return stats

def verify_runtime_optimizations() -> Dict[str, bool]:
    """
    Vérifie si les optimisations sont actives pendant l'exécution.
    
    Returns:
        État des optimisations à l'exécution
    """
    runtime_status = {}
    
    # Vérifier les optimisations Intel
    try:
        from ai_trading.utils.intel_optimizations import get_optimization_info
        intel_info = get_optimization_info()
        runtime_status["intel_optimizations"] = {
            "active": True,
            "details": {
                "is_intel_cpu": intel_info["cpu"]["is_intel"],
                "mkl_threads": intel_info["environment"]["MKL_NUM_THREADS"],
                "omp_threads": intel_info["environment"]["OMP_NUM_THREADS"]
            }
        }
    except (ImportError, Exception) as e:
        runtime_status["intel_optimizations"] = {
            "active": False,
            "error": str(e)
        }
    
    # Vérifier les optimisations de threading
    try:
        from ai_trading.utils.threading_optimizer import ThreadingOptimizer
        optimizer = ThreadingOptimizer()
        workers = optimizer.calculate_optimal_workers()
        runtime_status["threading_optimizations"] = {
            "active": True,
            "details": {
                "optimal_workers": workers,
                "cpu_count": optimizer.cpu_count,
                "hyperthreading": optimizer.has_hyperthreading
            }
        }
    except (ImportError, Exception) as e:
        runtime_status["threading_optimizations"] = {
            "active": False,
            "error": str(e)
        }
    
    # Vérifier les optimisations GPU
    try:
        import torch
        if torch.cuda.is_available():
            from ai_trading.utils.gpu_rtx_optimizer import setup_rtx_optimization
            rtx_optimized = setup_rtx_optimization()
            runtime_status["gpu_rtx_optimizations"] = {
                "active": rtx_optimized,
                "details": {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "tensor_cores_enabled": torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends.cuda, "matmul") else False,
                    "memory_allocated": f"{torch.cuda.memory_allocated() / (1024**2):.2f} MB"
                }
            }
        else:
            runtime_status["gpu_rtx_optimizations"] = {
                "active": False,
                "details": "CUDA not available"
            }
    except (ImportError, Exception) as e:
        runtime_status["gpu_rtx_optimizations"] = {
            "active": False,
            "error": str(e)
        }
    
    # Vérifier la compression zstd
    try:
        import zstandard
        runtime_status["compression_zstd"] = {
            "active": True,
            "details": {
                "version": zstandard.__version__,
                "max_compression_level": zstandard.MAX_COMPRESSION_LEVEL
            }
        }
    except (ImportError, Exception) as e:
        runtime_status["compression_zstd"] = {
            "active": False,
            "error": str(e)
        }
    
    return runtime_status

def format_report(stats: Dict[str, Any], runtime: Dict[str, Any]) -> str:
    """
    Formate les statistiques en un rapport lisible.
    
    Args:
        stats: Statistiques d'utilisation des optimisations
        runtime: État des optimisations à l'exécution
        
    Returns:
        Rapport formaté
    """
    report = []
    report.append("=" * 80)
    report.append("RAPPORT D'OPTIMISATIONS DU SYSTÈME DE TRADING")
    report.append("=" * 80)
    report.append("")
    
    # Informations générales
    report.append(f"Fichiers Python analysés: {stats['files_checked']}")
    report.append("")
    
    # Utilisation des optimisations
    report.append("UTILISATION DES OPTIMISATIONS:")
    report.append("-" * 50)
    for opt_name, count in stats["optimization_usage"].items():
        opt_info = OPTIMIZATIONS[opt_name]
        percentage = (count / stats["files_checked"]) * 100 if stats["files_checked"] > 0 else 0
        report.append(f"{opt_info['description']}: {count} fichiers ({percentage:.1f}%)")
    report.append("")
    
    # Fonctions d'optimisation
    report.append("FONCTIONS D'OPTIMISATION UTILISÉES:")
    report.append("-" * 50)
    for opt_name, funcs in stats["optimization_functions"].items():
        report.append(f"{OPTIMIZATIONS[opt_name]['description']}:")
        for func, count in funcs.items():
            if count > 0:
                report.append(f"  - {func}: {count} fichiers")
        report.append("")
    
    # Fichiers clés
    report.append("ANALYSE DES FICHIERS CLÉS:")
    report.append("-" * 50)
    for file_path, info in stats["key_files_with_optimizations"].items():
        report.append(f"Fichier: {file_path}")
        
        # Vérifier les optimisations utilisées
        used_opts = []
        for opt_name, imported in info["imports"].items():
            funcs = info["functions"][opt_name]
            if imported or funcs:
                used_opts.append(f"{opt_name} ({', '.join(funcs) if funcs else 'importé uniquement'})")
        
        if used_opts:
            report.append(f"  Optimisations utilisées: {', '.join(used_opts)}")
        else:
            report.append("  Aucune optimisation utilisée!")
        report.append("")
    
    # Optimisations manquantes
    if stats["missing_optimizations"]:
        report.append("OPTIMISATIONS MANQUANTES DANS LES FICHIERS CLÉS:")
        report.append("-" * 50)
        for missing in stats["missing_optimizations"]:
            file_path = missing["file"]
            missing_opts = missing["missing"]
            report.append(f"Fichier: {file_path}")
            report.append(f"  Optimisations manquantes: {', '.join(missing_opts)}")
        report.append("")
    
    # État d'exécution
    report.append("ÉTAT DES OPTIMISATIONS À L'EXÉCUTION:")
    report.append("-" * 50)
    for opt_name, status in runtime.items():
        if status["active"]:
            report.append(f"{opt_name}: ACTIF")
            for key, value in status.get("details", {}).items():
                report.append(f"  - {key}: {value}")
        else:
            report.append(f"{opt_name}: INACTIF")
            if "error" in status:
                report.append(f"  - Erreur: {status['error']}")
        report.append("")
    
    # Recommandations
    report.append("RECOMMANDATIONS:")
    report.append("-" * 50)
    
    missing_count = len(stats["missing_optimizations"])
    if missing_count > 0:
        report.append(f"1. Ajouter les optimisations manquantes aux {missing_count} fichiers clés identifiés.")
    else:
        report.append("1. Toutes les optimisations sont présentes dans les fichiers clés. Excellent!")
    
    inactive_opts = [opt for opt, status in runtime.items() if not status["active"]]
    if inactive_opts:
        report.append(f"2. Activer les optimisations suivantes: {', '.join(inactive_opts)}")
    else:
        report.append("2. Toutes les optimisations sont actives à l'exécution. Excellent!")
        
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Fonction principale."""
    logger.info("Démarrage de l'analyse des optimisations...")
    start_time = time.time()
    
    # Analyser le répertoire du projet
    stats = scan_directory(PROJECT_ROOT / "ai_trading")
    
    # Vérifier les optimisations à l'exécution
    runtime = verify_runtime_optimizations()
    
    # Générer le rapport
    report = format_report(stats, runtime)
    
    # Afficher le rapport
    print(report)
    
    # Sauvegarder le rapport dans un fichier
    report_path = PROJECT_ROOT / "ai_trading" / "documentation" / "opti_CUDA" / "optimization_report.txt"
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Rapport d'optimisation sauvegardé dans {report_path}")
    logger.info(f"Analyse terminée en {time.time() - start_time:.2f} secondes")

if __name__ == "__main__":
    main() 