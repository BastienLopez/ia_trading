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
    },
    # Nouvelles optimisations ajoutées
    "profiling_tools": {
        "module": "ai_trading.utils.profiling",
        "functions": ["profile_function", "profile_block"],
        "classes": ["ProfilingManager"],
        "description": "Outils de profilage intensif (cProfile, PyTorch Profiler, etc.)"
    },
    "jit_compilation": {
        "module": "ai_trading.utils.jit_compilation",
        "functions": ["compile_model", "optimize_function", "enable_tensorflow_xla"],
        "classes": ["TorchScriptCompiler", "XLAOptimizer"],
        "description": "Compilation JIT (TorchScript, XLA)"
    },
    "system_optimization": {
        "module": "ai_trading.utils.system_optimizer",
        "functions": ["optimize_system"],
        "classes": ["SystemOptimizer"],
        "description": "Optimisations système (variables d'environnement, limites système, E/S disque, etc.)"
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

def verify_runtime_optimizations() -> Dict[str, Any]:
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
            from ai_trading.utils.gpu_rtx_optimizer import RTXOptimizer
            optimizer = RTXOptimizer()
            runtime_status["gpu_rtx_optimizations"] = {
                "active": True,
                "details": {
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "tensor_cores_available": optimizer.has_tensor_cores()
                }
            }
        else:
            runtime_status["gpu_rtx_optimizations"] = {
                "active": False,
                "details": "CUDA n'est pas disponible"
            }
    except (ImportError, Exception) as e:
        runtime_status["gpu_rtx_optimizations"] = {
            "active": False,
            "error": str(e)
        }
    
    # Vérifier la compression de fichiers
    try:
        import zstandard
        from ai_trading.data.compressed_storage import CompressedStorage
        runtime_status["compression_zstd"] = {
            "active": True,
            "details": {
                "zstd_version": zstandard.__version__,
                "max_compression_level": zstandard.MAX_COMPRESSION_LEVEL
            }
        }
    except (ImportError, Exception) as e:
        runtime_status["compression_zstd"] = {
            "active": False,
            "error": str(e)
        }
    
    # Vérifier les outils de profilage
    try:
        from ai_trading.utils.profiling import ProfilingManager
        profiler = ProfilingManager()
        runtime_status["profiling_tools"] = {
            "active": True,
            "details": {
                "output_dir": str(profiler.output_dir),
                "nsight_available": profiler.nsight_available,
                "torch_available": hasattr(profiler, "profile_with_torch"),
                "tensorflow_available": hasattr(profiler, "profile_tensorflow")
            }
        }
    except (ImportError, Exception) as e:
        runtime_status["profiling_tools"] = {
            "active": False,
            "error": str(e)
        }
    
    # Vérifier la compilation JIT
    try:
        from ai_trading.utils.jit_compilation import TORCH_AVAILABLE, TF_AVAILABLE, XLA_AVAILABLE
        runtime_status["jit_compilation"] = {
            "active": True,
            "details": {
                "torch_available": TORCH_AVAILABLE,
                "tensorflow_available": TF_AVAILABLE,
                "xla_available": XLA_AVAILABLE
            }
        }
        
        # Vérifier si TorchScript fonctionne correctement
        if TORCH_AVAILABLE:
            import torch
            
            # Créer une fonction simple pour tester TorchScript
            def test_func(x, y):
                return x + y
            
            try:
                scripted_func = torch.jit.script(test_func)
                test_x = torch.tensor([1.0, 2.0])
                test_y = torch.tensor([3.0, 4.0])
                result = scripted_func(test_x, test_y)
                runtime_status["jit_compilation"]["details"]["torchscript_works"] = True
            except Exception as e:
                runtime_status["jit_compilation"]["details"]["torchscript_works"] = False
                runtime_status["jit_compilation"]["details"]["torchscript_error"] = str(e)
        
        # Vérifier si XLA fonctionne correctement
        if TF_AVAILABLE and XLA_AVAILABLE:
            import tensorflow as tf
            
            try:
                # Configurer XLA
                tf.config.optimizer.set_jit(True)
                runtime_status["jit_compilation"]["details"]["xla_enabled"] = True
            except Exception as e:
                runtime_status["jit_compilation"]["details"]["xla_enabled"] = False
                runtime_status["jit_compilation"]["details"]["xla_error"] = str(e)
        
    except (ImportError, Exception) as e:
        runtime_status["jit_compilation"] = {
            "active": False,
            "error": str(e)
        }
    
    # Vérifier les optimisations système
    try:
        from ai_trading.utils.system_optimizer import SystemOptimizer
        optimizer = SystemOptimizer()
        status = optimizer.get_optimization_status()
        
        # Vérifier les variables d'environnement clés
        env_keys = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED"]
        env_set = any(key in os.environ for key in env_keys)
        
        runtime_status["system_optimization"] = {
            "active": True,
            "details": {
                "os": status["system_info"]["os"],
                "cpu_count": status["system_info"]["cpu_count"],
                "memory_total_gb": round(status["system_info"]["memory_total"] / (1024**3), 2),
                "env_vars_set": env_set,
                "is_admin": status["system_info"]["is_admin"]
            }
        }
    except (ImportError, Exception) as e:
        runtime_status["system_optimization"] = {
            "active": False,
            "error": str(e)
        }
    
    return runtime_status

def format_report(stats: Dict[str, Any], runtime: Dict[str, Any]) -> str:
    """
    Formate un rapport de vérification des optimisations.
    
    Args:
        stats: Statistiques d'utilisation des optimisations
        runtime: État des optimisations à l'exécution
        
    Returns:
        Rapport formaté
    """
    report = []
    
    # En-tête
    report.append("=" * 80)
    report.append("RAPPORT DE VÉRIFICATION DES OPTIMISATIONS")
    report.append("=" * 80)
    report.append("")
    
    # Résumé général
    report.append(f"Fichiers analysés: {stats['files_checked']}")
    report.append("")
    
    # Utilisation des optimisations
    report.append("Utilisation des optimisations:")
    report.append("-" * 40)
    for opt_name, opt_info in OPTIMIZATIONS.items():
        usage_count = stats["optimization_usage"][opt_name]
        usage_percent = (usage_count / stats["files_checked"]) * 100 if stats["files_checked"] > 0 else 0
        report.append(f"{opt_info['description']}: {usage_count} fichiers ({usage_percent:.1f}%)")
    report.append("")
    
    # Fonctions les plus utilisées
    report.append("Fonctions d'optimisation les plus utilisées:")
    report.append("-" * 40)
    all_funcs = []
    for opt_name, funcs in stats["optimization_functions"].items():
        for func_name, count in funcs.items():
            if count > 0:
                all_funcs.append((f"{func_name} ({opt_name})", count))
    
    # Trier par nombre d'utilisations
    all_funcs.sort(key=lambda x: x[1], reverse=True)
    
    # Afficher les 10 fonctions les plus utilisées
    for i, (func_name, count) in enumerate(all_funcs[:10], 1):
        report.append(f"{i}. {func_name}: {count} utilisations")
    report.append("")
    
    # Optimisations manquantes dans les fichiers clés
    if stats["missing_optimizations"]:
        report.append("Optimisations manquantes dans les fichiers clés:")
        report.append("-" * 40)
        for missing in stats["missing_optimizations"]:
            file_name = missing["file"]
            missing_opts = ", ".join([OPTIMIZATIONS[opt]["description"] for opt in missing["missing"]])
            report.append(f"{file_name}: {missing_opts}")
        report.append("")
    
    # État des optimisations à l'exécution
    report.append("État des optimisations à l'exécution:")
    report.append("-" * 40)
    for opt_name, opt_status in runtime.items():
        if opt_status["active"]:
            report.append(f"{opt_name}: ACTIF")
            if "details" in opt_status:
                for detail_name, detail_value in opt_status["details"].items():
                    report.append(f"  - {detail_name}: {detail_value}")
        else:
            report.append(f"{opt_name}: INACTIF ({opt_status.get('error', 'Raison inconnue')})")
    report.append("")
    
    # Recommandations
    report.append("Recommandations:")
    report.append("-" * 40)
    
    # Recommandations basées sur les optimisations manquantes
    for missing in stats["missing_optimizations"]:
        file_name = missing["file"]
        for missing_opt in missing["missing"]:
            opt_info = OPTIMIZATIONS[missing_opt]
            report.append(f"- Ajouter {opt_info['description']} à {file_name}")
    
    # Recommandations basées sur l'état d'exécution
    for opt_name, opt_status in runtime.items():
        if not opt_status["active"]:
            report.append(f"- Activer {opt_name}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Point d'entrée principal pour vérifier les optimisations."""
    
    logger.info("Vérification des optimisations CPU/GPU...")
    
    # Analyser le répertoire du projet
    stats = scan_directory(PROJECT_ROOT / "ai_trading")
    
    # Vérifier l'état des optimisations à l'exécution
    runtime = verify_runtime_optimizations()
    
    # Générer et afficher le rapport
    report = format_report(stats, runtime)
    print(report)
    
    # Sauvegarder le rapport dans un fichier
    report_path = PROJECT_ROOT / "ai_trading" / "optim" / "optimization_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Rapport sauvegardé dans {report_path}")
    
    # Vérifier si des optimisations importantes sont manquantes
    has_critical_missing = False
    for missing in stats["missing_optimizations"]:
        if "train.py" in missing["file"] or "rl_agent.py" in missing["file"]:
            has_critical_missing = True
            break
    
    # Sortir avec un code d'erreur si des optimisations critiques sont manquantes
    if has_critical_missing:
        logger.error("Des optimisations critiques sont manquantes dans les fichiers clés!")
        sys.exit(1)
    
    logger.info("Toutes les vérifications d'optimisation ont été effectuées avec succès.")
    sys.exit(0)

def check_profiling_tools():
    """Vérifie les outils de profilage disponibles."""
    
    import cProfile
    import pstats
    import io
    
    # Vérifier cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code à profiler
    total = 0
    for i in range(1000):
        total += i
    
    profiler.disable()
    
    # Analyser les résultats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(5)
    
    # Vérifier PyTorch Profiler
    try:
        import torch
        has_torch_profiler = hasattr(torch, 'profiler')
    except ImportError:
        has_torch_profiler = False
    
    # Vérifier TensorFlow Profiler
    try:
        import tensorflow as tf
        has_tf_profiler = hasattr(tf, 'profiler')
    except ImportError:
        has_tf_profiler = False
    
    # Vérifier NVIDIA Nsight
    try:
        import subprocess
        result = subprocess.run(['nsys', '--version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        has_nsight = result.returncode == 0
    except:
        has_nsight = False
    
    return {
        "cprofile": True,
        "torch_profiler": has_torch_profiler,
        "tensorflow_profiler": has_tf_profiler,
        "nsight": has_nsight
    }

def check_jit_compilation():
    """Vérifie les outils de compilation JIT disponibles."""
    
    # Vérifier TorchScript
    try:
        import torch
        has_torchscript = hasattr(torch, 'jit')
        
        if has_torchscript:
            # Test simple de torchscript
            def add(a, b):
                return a + b
            
            scripted_add = torch.jit.script(add)
            x = torch.tensor([1.0, 2.0])
            y = torch.tensor([3.0, 4.0])
            result = scripted_add(x, y)
            torchscript_works = True
        else:
            torchscript_works = False
    except (ImportError, Exception):
        has_torchscript = False
        torchscript_works = False
    
    # Vérifier TensorFlow XLA
    try:
        import tensorflow as tf
        has_xla = hasattr(tf.config.optimizer, 'set_jit')
        
        if has_xla:
            # Test simple de XLA
            try:
                tf.config.optimizer.set_jit(True)
                xla_works = True
            except:
                xla_works = False
        else:
            xla_works = False
    except (ImportError, Exception):
        has_xla = False
        xla_works = False
    
    return {
        "torchscript": {
            "available": has_torchscript,
            "works": torchscript_works
        },
        "xla": {
            "available": has_xla,
            "works": xla_works
        }
    }

def print_check_results():
    """Affiche les résultats des vérifications d'optimisation."""
    
    print("\n" + "=" * 80)
    print("VÉRIFICATION DES OPTIMISATIONS DISPONIBLES")
    print("=" * 80)
    
    # Vérifier les outils de profilage
    profiling_results = check_profiling_tools()
    print("\nOutils de profilage:")
    print("-" * 40)
    for tool, available in profiling_results.items():
        print(f"{tool}: {'✓' if available else '✗'}")
    
    # Vérifier les outils de compilation JIT
    jit_results = check_jit_compilation()
    print("\nOutils de compilation JIT:")
    print("-" * 40)
    for tool, info in jit_results.items():
        status = "✓" if info["available"] and info["works"] else "✗"
        if info["available"] and not info["works"]:
            status = "⚠ (disponible mais ne fonctionne pas)"
        print(f"{tool}: {status}")
    
    # Vérifier les optimisations système
    system_results = check_system_optimizations()
    print("\nOptimisations système:")
    print("-" * 40)
    for feature, available in system_results.items():
        print(f"{feature}: {'✓' if available else '✗'}")
    
    print("\n" + "=" * 80)

def check_system_optimizations():
    """Vérifie les optimisations système disponibles."""
    
    # Vérifier si le module est disponible
    try:
        import importlib
        system_optimizer = importlib.import_module("ai_trading.utils.system_optimizer")
        module_available = True
    except ImportError:
        module_available = False
    
    if not module_available:
        return {
            "module_available": False,
            "env_vars_optimization": False,
            "system_limits": False,
            "disk_io_optimization": False,
            "memory_optimization": False,
            "logging_setup": False
        }
    
    # Vérifier les fonctionnalités individuelles
    try:
        from ai_trading.utils.system_optimizer import SystemOptimizer
        optimizer = SystemOptimizer()
        
        # Tester l'optimisation des variables d'environnement
        try:
            optimizer.optimize_environment_variables()
            env_vars_optimization = True
        except:
            env_vars_optimization = False
        
        # Tester la configuration des limites système
        try:
            optimizer.configure_system_limits()
            system_limits = True
        except:
            system_limits = False
        
        # Tester l'optimisation des E/S disque
        try:
            optimizer.optimize_disk_io()
            disk_io_optimization = True
        except:
            disk_io_optimization = False
        
        # Tester la configuration de la mémoire
        try:
            optimizer.configure_memory_params()
            memory_optimization = True
        except:
            memory_optimization = False
        
        # Tester la configuration du logging
        try:
            optimizer.setup_logging()
            logging_setup = True
        except:
            logging_setup = False
        
        return {
            "module_available": True,
            "env_vars_optimization": env_vars_optimization,
            "system_limits": system_limits,
            "disk_io_optimization": disk_io_optimization,
            "memory_optimization": memory_optimization,
            "logging_setup": logging_setup
        }
        
    except Exception:
        return {
            "module_available": True,
            "env_vars_optimization": False,
            "system_limits": False,
            "disk_io_optimization": False,
            "memory_optimization": False,
            "logging_setup": False
        }

if __name__ == "__main__":
    # Vérifier les arguments en ligne de commande
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print_check_results()
    else:
        main() 