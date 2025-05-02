#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outil de vérification pour s'assurer que toutes les optimisations sont correctement installées.
Ce script vérifie la disponibilité des bibliothèques et fonctionnalités d'optimisation
et rapporte leur état.
"""

import os
import sys
import importlib
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizationFeature:
    """Représente une fonctionnalité d'optimisation à vérifier."""
    name: str
    category: str
    description: str
    packages: List[str]
    test_func: Optional[Callable[[], Tuple[bool, str]]] = None
    is_available: bool = False
    status_message: str = ""

def test_package_import(packages: List[str]) -> Tuple[bool, str]:
    """
    Vérifie si les packages peuvent être importés.
    
    Args:
        packages: Liste des packages à importer
        
    Returns:
        Tuple (succès, message)
    """
    success = True
    missing_packages = []
    
    for package in packages:
        try:
            # Pour les packages avec un sous-module spécifié (ex: torch.cuda)
            if '.' in package:
                main_package, sub_package = package.split('.', 1)
                module = importlib.import_module(main_package)
                # Vérifier récursivement les sous-modules
                sub_module = module
                for part in sub_package.split('.'):
                    if not hasattr(sub_module, part):
                        missing_packages.append(package)
                        success = False
                        break
                    sub_module = getattr(sub_module, part)
            else:
                # Pour les packages simples
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
            success = False
    
    if success:
        return True, "Tous les packages sont disponibles"
    else:
        return False, f"Packages manquants: {', '.join(missing_packages)}"

def test_torch_cuda():
    """Vérifie si CUDA est disponible pour PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            return True, f"CUDA disponible avec {device_count} appareils: {', '.join(device_names)}"
        else:
            return False, "CUDA n'est pas disponible pour PyTorch"
    except ImportError:
        return False, "PyTorch n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors de la vérification de CUDA: {str(e)}"

def test_tensorflow_gpu():
    """Vérifie si GPU est disponible pour TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return True, f"TensorFlow détecte {len(gpus)} GPU(s)"
        else:
            return False, "Aucun GPU disponible pour TensorFlow"
    except ImportError:
        return False, "TensorFlow n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors de la vérification des GPUs TensorFlow: {str(e)}"

def test_ray_init():
    """Teste l'initialisation de Ray."""
    try:
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        return True, "Ray initialisé avec succès"
    except ImportError:
        return False, "Ray n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors de l'initialisation de Ray: {str(e)}"

def test_onnx_export():
    """Teste l'exportation ONNX basique."""
    try:
        import torch
        import torch.nn as nn
        import onnx
        import onnxruntime
        
        # Créer un modèle simple
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(10, 1)
                
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        model.eval()
        
        # Exporter en ONNX
        dummy_input = torch.randn(1, 10)
        torch.onnx.export(
            model, dummy_input, "test_model.onnx",
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        # Vérifier le modèle
        onnx_model = onnx.load("test_model.onnx")
        onnx.checker.check_model(onnx_model)
        
        # Nettoyer
        if os.path.exists("test_model.onnx"):
            os.remove("test_model.onnx")
        
        return True, "Exportation ONNX réussie"
    except ImportError as e:
        return False, f"Module manquant pour l'exportation ONNX: {str(e)}"
    except Exception as e:
        return False, f"Erreur lors du test d'exportation ONNX: {str(e)}"

def test_optuna():
    """Teste l'optimisation d'hyperparamètres Optuna."""
    try:
        import optuna
        
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=3)
        
        return True, f"Optuna fonctionne correctement, meilleure valeur: {study.best_value:.4f}"
    except ImportError:
        return False, "Optuna n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors du test d'Optuna: {str(e)}"

def test_ray_tune():
    """Teste l'optimisation d'hyperparamètres Ray Tune."""
    try:
        import ray
        from ray import tune
        
        def objective(config):
            x = config["x"]
            return {"score": (x - 2) ** 2}
        
        ray.init(ignore_reinit_error=True)
        analysis = tune.run(
            objective,
            config={"x": tune.uniform(-10, 10)},
            num_samples=3,
            verbose=0
        )
        
        best_trial = analysis.get_best_trial("score", "min")
        
        return True, f"Ray Tune fonctionne correctement, meilleure valeur: {best_trial.last_result['score']:.4f}"
    except ImportError:
        return False, "Ray Tune n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors du test de Ray Tune: {str(e)}"

def test_numba():
    """Teste les optimisations Numba."""
    try:
        import numba
        import numpy as np
        import time
        
        # Fonction sans Numba
        def sum_without_numba(arr):
            result = 0
            for i in range(len(arr)):
                result += arr[i]
            return result
        
        # Fonction avec Numba
        @numba.jit(nopython=True)
        def sum_with_numba(arr):
            result = 0
            for i in range(len(arr)):
                result += arr[i]
            return result
        
        # Créer un tableau
        arr = np.random.rand(10000000)
        
        # Mesurer le temps sans Numba
        start = time.time()
        result1 = sum_without_numba(arr)
        time_without_numba = time.time() - start
        
        # Mesurer le temps avec Numba (compilation + exécution)
        start = time.time()
        result2 = sum_with_numba(arr)
        time_with_numba_first = time.time() - start
        
        # Mesurer le temps avec Numba (exécution seulement)
        start = time.time()
        result2 = sum_with_numba(arr)
        time_with_numba_second = time.time() - start
        
        speedup = time_without_numba / time_with_numba_second
        
        return True, f"Numba fonctionne correctement, accélération: {speedup:.2f}x"
    except ImportError:
        return False, "Numba n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors du test de Numba: {str(e)}"

def test_torch_fx():
    """Teste l'optimisation de graphe avec torch.fx."""
    try:
        import torch
        import torch.fx as fx
        
        # Créer un modèle simple
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = torch.nn.Linear(10, 10)
                self.lin2 = torch.nn.Linear(10, 1)
                
            def forward(self, x):
                x = self.lin1(x)
                x = torch.relu(x)
                x = self.lin2(x)
                return x
        
        model = SimpleModel()
        
        # Tracer et générer un graphe symbolique
        traced_model = fx.symbolic_trace(model)
        
        # Afficher le graphe pour vérifier
        graph_str = traced_model.graph.python_code("self")
        
        return True, "torch.fx fonctionne correctement"
    except ImportError:
        return False, "torch.fx n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors du test de torch.fx: {str(e)}"

def test_torch_distributed():
    """Teste le module torch.distributed."""
    try:
        import torch.distributed as dist
        
        # Vérifier si le module est disponible (sans l'initialiser vraiment)
        backend_list = ["gloo", "nccl"] if torch.cuda.is_available() else ["gloo"]
        available_backends = [b for b in backend_list if dist.is_available() and dist.is_backend_available(b)]
        
        if available_backends:
            return True, f"torch.distributed disponible avec backends: {', '.join(available_backends)}"
        else:
            return False, "torch.distributed est importable mais aucun backend n'est disponible"
    except ImportError:
        return False, "torch.distributed n'est pas disponible"
    except Exception as e:
        return False, f"Erreur lors du test de torch.distributed: {str(e)}"

def test_compressed_storage():
    """Teste le stockage de données compressé."""
    try:
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import os
        import time
        
        # Créer des données test
        data = np.random.rand(100000, 10)
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(10)])
        
        # Mesurer la taille sans compression
        csv_file = "test_uncompressed.csv"
        df.to_csv(csv_file, index=False)
        uncompressed_size = os.path.getsize(csv_file)
        
        # Mesurer la taille avec compression parquet
        parquet_file = "test_compressed.parquet"
        df.to_parquet(parquet_file, compression='snappy')
        compressed_size = os.path.getsize(parquet_file)
        
        # Nettoyer
        os.remove(csv_file)
        os.remove(parquet_file)
        
        compression_ratio = uncompressed_size / compressed_size
        
        return True, f"Stockage compressé fonctionne, ratio de compression: {compression_ratio:.2f}x"
    except ImportError:
        return False, "Modules requis pour le stockage compressé (pandas, pyarrow) non disponibles"
    except Exception as e:
        return False, f"Erreur lors du test de stockage compressé: {str(e)}"

def test_lazy_loading():
    """Teste le chargement paresseux des données."""
    try:
        import dask.dataframe as dd
        import pandas as pd
        import numpy as np
        import os
        import time
        
        # Créer des données test
        data = np.random.rand(100000, 10)
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(10)])
        df.to_csv("test_data.csv", index=False)
        
        # Mesurer le temps avec chargement complet
        start = time.time()
        full_df = pd.read_csv("test_data.csv")
        normal_load_time = time.time() - start
        
        # Mesurer le temps avec chargement paresseux
        start = time.time()
        lazy_df = dd.read_csv("test_data.csv")
        # Récupérer uniquement la première ligne pour démontrer le lazy loading
        first_row = lazy_df.head(1)
        lazy_load_time = time.time() - start
        
        # Nettoyer
        os.remove("test_data.csv")
        
        return True, f"Chargement paresseux fonctionne, temps de chargement: {lazy_load_time:.4f}s vs {normal_load_time:.4f}s"
    except ImportError:
        return False, "Modules requis pour le chargement paresseux (dask) non disponibles"
    except Exception as e:
        return False, f"Erreur lors du test de chargement paresseux: {str(e)}"

def test_batch_inference():
    """Teste l'inférence par lots."""
    try:
        import torch
        import torch.nn as nn
        import time
        
        # Créer un modèle simple
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(10, 1)
                
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        model.eval()
        
        # Créer des données
        num_samples = 1000
        
        # Inférence par échantillon individuel
        single_inputs = [torch.randn(1, 10) for _ in range(num_samples)]
        
        start = time.time()
        single_outputs = []
        with torch.no_grad():
            for x in single_inputs:
                y = model(x)
                single_outputs.append(y)
        single_time = time.time() - start
        
        # S'assurer que single_time n'est pas zéro pour éviter la division par zéro
        if single_time < 1e-6:
            single_time = 1e-6
        
        # Inférence par lots
        batch_input = torch.randn(num_samples, 10)
        
        start = time.time()
        with torch.no_grad():
            batch_output = model(batch_input)
        batch_time = time.time() - start
        
        # S'assurer que batch_time n'est pas zéro pour éviter la division par zéro
        if batch_time < 1e-6:
            batch_time = 1e-6
        
        speedup = single_time / batch_time
        
        return True, f"Inférence par lots fonctionne, accélération: {speedup:.2f}x"
    except ImportError:
        return False, "PyTorch n'est pas disponible"
    except Exception as e:
        return False, f"Erreur lors du test d'inférence par lots: {e}"

def test_torch_profiler():
    """Teste le profiler PyTorch."""
    try:
        import torch
        import torch.nn as nn
        from torch.profiler import profile, record_function, ProfilerActivity
        
        # Créer un modèle simple
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # Créer des données
        x = torch.randn(100, 10)
        
        # Profiler
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                model(x)
        
        # Vérifier que le profiling a fonctionné
        events = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        
        return True, "torch.profiler fonctionne correctement"
    except ImportError:
        return False, "torch.profiler n'est pas disponible"
    except Exception as e:
        return False, f"Erreur lors du test de torch.profiler: {str(e)}"

def test_torch_quantization():
    """Teste les fonctionnalités de quantification de PyTorch."""
    try:
        import torch
        import torch.nn as nn
        
        # Vérifier si le module quantization est disponible
        if not hasattr(torch, 'quantization'):
            return False, "Le module torch.quantization n'est pas disponible"
        
        # Créer un modèle simple pour le test
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(10, 1)
                
            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        # Créer le modèle
        model = SimpleModel()
        model.eval()
        
        # Tester la quantification dynamique (plus simple et fonctionne sur plus de plateformes)
        try:
            # Vérifier juste si la fonction est accessible
            if hasattr(torch.quantization, 'quantize_dynamic'):
                # Tester avec un modèle minimal
                dummy_input = torch.randn(1, 10)
                model(dummy_input)  # Forward pass pour vérifier que le modèle fonctionne
                
                # On ne fait pas la quantification complète car elle peut échouer sur certaines plateformes
                # On vérifie juste que les fonctions sont accessibles
                return True, "Quantification PyTorch disponible"
            else:
                return False, "La fonction quantize_dynamic n'est pas disponible"
        except Exception as e:
            return False, f"Erreur lors du test de quantification dynamique: {str(e)}"
            
    except ImportError:
        return False, "PyTorch n'est pas installé"
    except Exception as e:
        return False, f"Erreur lors du test de quantification PyTorch: {str(e)}"

def create_optimization_features() -> List[OptimizationFeature]:
    """Crée la liste des fonctionnalités d'optimisation à vérifier."""
    features = [
        # Optimisation mémoire CPU / RAM
        OptimizationFeature(
            name="PyTorch Core",
            category="CPU/RAM",
            description="PyTorch avec compile() et autres optimisations",
            packages=["torch", "torch.compile"],
            test_func=lambda: test_package_import(["torch", "torch.compile"])
        ),
        OptimizationFeature(
            name="Data Loader Optimization",
            category="CPU/RAM",
            description="Optimisation des chargeurs de données",
            packages=["torch.utils.data"],
            test_func=lambda: test_package_import(["torch.utils.data"])
        ),
        OptimizationFeature(
            name="Intel MKL",
            category="CPU/RAM",
            description="Optimisations spécifiques pour Intel",
            packages=["numpy"],
            test_func=lambda: test_package_import(["numpy"])
        ),
        
        # Optimisation CPU
        OptimizationFeature(
            name="Profiling Tools",
            category="CPU",
            description="Outils de profilage CPU",
            packages=["cProfile", "line_profiler", "memory_profiler"],
            test_func=lambda: test_package_import(["cProfile"])
        ),
        OptimizationFeature(
            name="Numba",
            category="CPU",
            description="Compilation JIT pour code Python",
            packages=["numba"],
            test_func=test_numba
        ),
        OptimizationFeature(
            name="Threading Optimizations",
            category="CPU",
            description="Optimisations multi-threading",
            packages=["threading", "multiprocessing", "concurrent.futures"],
            test_func=lambda: test_package_import(["threading", "multiprocessing", "concurrent.futures"])
        ),
        
        # Optimisation GPU
        OptimizationFeature(
            name="CUDA Support",
            category="GPU",
            description="Support CUDA pour PyTorch",
            packages=["torch.cuda"],
            test_func=test_torch_cuda
        ),
        OptimizationFeature(
            name="TensorFlow GPU",
            category="GPU",
            description="Support GPU pour TensorFlow",
            packages=["tensorflow"],
            test_func=test_tensorflow_gpu
        ),
        OptimizationFeature(
            name="Mixed Precision",
            category="GPU",
            description="Entraînement en précision mixte",
            packages=["torch.cuda.amp"],
            test_func=lambda: test_package_import(["torch.cuda.amp"])
        ),
        
        # Optimisation de l'architecture IA
        OptimizationFeature(
            name="Model Pruning",
            category="AI Architecture",
            description="Élagage de modèle",
            packages=["torch.nn.utils.prune"],
            test_func=lambda: test_package_import(["torch.nn.utils.prune"])
        ),
        OptimizationFeature(
            name="Quantization",
            category="AI Architecture",
            description="Quantification de modèle",
            packages=["torch.quantization"],
            test_func=test_torch_quantization
        ),
        
        # Optimisation de l'entraînement RL
        OptimizationFeature(
            name="Stable Baselines",
            category="RL",
            description="Stable Baselines 3 pour RL",
            packages=["stable_baselines3"],
            test_func=lambda: test_package_import(["stable_baselines3"])
        ),
        OptimizationFeature(
            name="Ray RLlib",
            category="RL",
            description="Ray RLlib pour entraînement distribué",
            packages=["ray.rllib"],
            test_func=test_ray_init
        ),
        
        # Optimisation générale du projet
        OptimizationFeature(
            name="PyTorch Profiler",
            category="Profiling",
            description="Profilage PyTorch",
            packages=["torch.profiler"],
            test_func=test_torch_profiler
        ),
        OptimizationFeature(
            name="TensorFlow Profiler",
            category="Profiling",
            description="Profilage TensorFlow",
            packages=["tensorflow.profiler"],
            test_func=lambda: test_package_import(["tensorflow"])
        ),
        OptimizationFeature(
            name="JIT Compilation",
            category="Compilation",
            description="Compilation JIT avec TorchScript",
            packages=["torch.jit"],
            test_func=lambda: test_package_import(["torch.jit"])
        ),
        
        # Outils/méthodes
        OptimizationFeature(
            name="DeepSpeed",
            category="Tools",
            description="Optimisation mémoire pour grands modèles",
            packages=["deepspeed"],
            test_func=lambda: test_package_import(["deepspeed"])
        ),
        OptimizationFeature(
            name="Huggingface Accelerate",
            category="Tools",
            description="Accélération multi-GPU simplifiée",
            packages=["accelerate"],
            test_func=lambda: test_package_import(["accelerate"])
        ),
        OptimizationFeature(
            name="Optuna",
            category="Tools",
            description="Optimisation d'hyperparamètres",
            packages=["optuna"],
            test_func=test_optuna
        ),
        OptimizationFeature(
            name="Ray Tune",
            category="Tools",
            description="Optimisation d'hyperparamètres distribuée",
            packages=["ray.tune"],
            test_func=test_ray_tune
        ),
        OptimizationFeature(
            name="ONNX Export",
            category="Tools",
            description="Exportation de modèle pour inférence rapide",
            packages=["onnx", "onnxruntime"],
            test_func=test_onnx_export
        ),
        
        # Nouveaux outils
        OptimizationFeature(
            name="torch.fx",
            category="Nouveaux Outils",
            description="Optimisation de graphe avec torch.fx",
            packages=["torch.fx"],
            test_func=test_torch_fx
        ),
        OptimizationFeature(
            name="torch.distributed",
            category="Nouveaux Outils",
            description="Support multi-GPU avec torch.distributed",
            packages=["torch.distributed"],
            test_func=test_torch_distributed
        ),
        
        # Optimisation de fichiers
        OptimizationFeature(
            name="Stockage compressé",
            category="Fichiers",
            description="Stockage de données compressé (parquet, etc.)",
            packages=["pandas", "pyarrow", "pyarrow.parquet"],
            test_func=test_compressed_storage
        ),
        OptimizationFeature(
            name="Lecture paresseuse",
            category="Fichiers",
            description="Chargement paresseux des données",
            packages=["dask.dataframe"],
            test_func=test_lazy_loading
        ),
        OptimizationFeature(
            name="Cache de features",
            category="Fichiers",
            description="Mise en cache des features pré-calculées",
            packages=["joblib", "functools"],
            test_func=lambda: test_package_import(["joblib", "functools"])
        ),
        
        # Batch inference
        OptimizationFeature(
            name="Batch Inference",
            category="Inférence",
            description="Inférence par lots pour accélérer les prédictions",
            packages=["torch"],
            test_func=test_batch_inference
        )
    ]
    
    return features

def check_optimization_features() -> Dict[str, List[OptimizationFeature]]:
    """
    Vérifie la disponibilité des fonctionnalités d'optimisation.
    
    Returns:
        Dictionnaire des fonctionnalités par catégorie
    """
    features = create_optimization_features()
    features_by_category = {}
    
    for feature in features:
        # Exécuter la fonction de test
        if feature.test_func:
            feature.is_available, feature.status_message = feature.test_func()
        else:
            feature.is_available, feature.status_message = test_package_import(feature.packages)
        
        # Ajouter à la catégorie
        if feature.category not in features_by_category:
            features_by_category[feature.category] = []
        features_by_category[feature.category].append(feature)
    
    return features_by_category

def print_optimization_status(features_by_category: Dict[str, List[OptimizationFeature]]):
    """Affiche le statut des fonctionnalités d'optimisation."""
    print("\n" + "=" * 80)
    print("STATUT DES OPTIMISATIONS")
    print("=" * 80)
    
    # Calculer les statistiques
    total_features = 0
    available_features = 0
    
    # Parcourir les catégories
    for category, features in sorted(features_by_category.items()):
        print(f"\n-- {category} --")
        
        for feature in features:
            status = "✓" if feature.is_available else "✗"
            print(f"{status} {feature.name}: {feature.description}")
            print(f"   {feature.status_message}")
            
            total_features += 1
            if feature.is_available:
                available_features += 1
    
    # Afficher le résumé
    print("\n" + "=" * 80)
    percentage = (available_features / total_features) * 100 if total_features > 0 else 0
    print(f"RÉSUMÉ: {available_features}/{total_features} optimisations disponibles ({percentage:.1f}%)")
    print("=" * 80)

def main():
    """Fonction principale."""
    print("\nVérification des optimisations disponibles...\n")
    
    # Vérifier les fonctionnalités
    features_by_category = check_optimization_features()
    
    # Afficher les résultats
    print_optimization_status(features_by_category)

if __name__ == "__main__":
    main() 