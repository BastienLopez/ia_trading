#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de l'exportateur ONNX pour convertir un modèle PyTorch
et réaliser une inférence plus rapide et portable.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import du module d'exportation ONNX
from ai_trading.utils.onnx_exporter import (
    ONNXExporter,
    export_pytorch_policy_model,
    HAVE_ONNX
)

# Définition d'un modèle de politique simple pour l'exemple
class SimplePolicyNetwork(nn.Module):
    """
    Réseau de politique simple avec des couches entièrement connectées.
    """
    
    def __init__(self, input_dim=20, hidden_dims=[64, 32], output_dim=1):
        super(SimplePolicyNetwork, self).__init__()
        
        # Construire les couches cachées
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Couche de sortie
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = nn.Tanh()  # Action entre -1 et 1
    
    def forward(self, x):
        """
        Propagation avant dans le réseau.
        
        Args:
            x: Tenseur d'entrée [batch_size, input_dim]
            
        Returns:
            Tenseur de sortie [batch_size, output_dim]
        """
        features = self.feature_layers(x)
        output = self.output_layer(features)
        return self.activation(output)

def create_pytorch_model(input_dim, hidden_dims, output_dim):
    """Crée un modèle PyTorch pour l'exemple."""
    model = SimplePolicyNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim
    )
    return model

def benchmark_inference(model_path, input_shape, num_runs=1000, batch_size=1):
    """
    Compare les performances d'inférence entre PyTorch et ONNX.
    
    Args:
        model_path: Chemin vers le modèle ONNX
        input_shape: Forme des entrées (sans le batch)
        num_runs: Nombre d'exécutions pour le benchmark
        batch_size: Taille du batch pour l'inférence
        
    Returns:
        Dictionnaire des résultats du benchmark
    """
    import onnxruntime as ort
    
    # Créer le modèle PyTorch
    pytorch_model = create_pytorch_model(input_shape[0], [64, 32], 1)
    pytorch_model.eval()
    
    # Charger le modèle ONNX
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    
    # Générer des données aléatoires
    np_data = np.random.randn(batch_size, *input_shape).astype(np.float32)
    torch_data = torch.tensor(np_data)
    
    # Benchmark PyTorch
    pytorch_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = pytorch_model(torch_data)
        pytorch_times.append((time.time() - start_time) * 1000)  # en millisecondes
    
    # Benchmark ONNX
    onnx_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = ort_session.run(None, {input_name: np_data})
        onnx_times.append((time.time() - start_time) * 1000)  # en millisecondes
    
    # Calculer les statistiques
    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    onnx_mean = np.mean(onnx_times)
    onnx_std = np.std(onnx_times)
    speedup = pytorch_mean / onnx_mean
    
    results = {
        "pytorch_mean_ms": pytorch_mean,
        "pytorch_std_ms": pytorch_std,
        "onnx_mean_ms": onnx_mean,
        "onnx_std_ms": onnx_std,
        "speedup": speedup,
        "batch_size": batch_size,
        "num_runs": num_runs
    }
    
    # Afficher les résultats
    print(f"Benchmark PyTorch vs ONNX (batch_size={batch_size}, runs={num_runs}):")
    print(f"  PyTorch: {pytorch_mean:.4f} ± {pytorch_std:.4f} ms")
    print(f"  ONNX: {onnx_mean:.4f} ± {onnx_std:.4f} ms")
    print(f"  Accélération: {speedup:.2f}x")
    
    return results

def plot_inference_comparison(results, title="Comparaison des performances d'inférence"):
    """
    Trace un graphique comparant les performances d'inférence.
    
    Args:
        results: Dictionnaire des résultats du benchmark
        title: Titre du graphique
    """
    labels = ['PyTorch', 'ONNX']
    means = [results['pytorch_mean_ms'], results['onnx_mean_ms']]
    stds = [results['pytorch_std_ms'], results['onnx_std_ms']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer le graphique à barres
    x = np.arange(len(labels))
    width = 0.35
    
    rects = ax.bar(x, means, width, yerr=stds, capsize=5,
                   color=['#1f77b4', '#ff7f0e'],
                   label=labels)
    
    # Ajouter les étiquettes et le titre
    ax.set_ylabel('Temps d\'inférence (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Ajouter les valeurs au-dessus des barres
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points de décalage vertical
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Ajouter l'accélération
    ax.text(0.5, 0.9, f"Accélération: {results['speedup']:.2f}x",
            horizontalalignment='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.5))
    
    # Enregistrer le graphique
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/onnx_inference_comparison.png")
    plt.close()
    
    print(f"Graphique sauvegardé dans 'results/onnx_inference_comparison.png'")

def main(args):
    """Fonction principale."""
    if not HAVE_ONNX:
        print("ONNX n'est pas installé. Exécutez 'pip install onnx onnxruntime' pour l'installer.")
        return

    print("=" * 80)
    print("Exemple d'exportation de modèle PyTorch vers ONNX")
    print("=" * 80)
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dimensions du modèle
    input_dim = args.input_dim
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    output_dim = args.output_dim
    
    print(f"Création d'un modèle avec:")
    print(f"  - Entrée: {input_dim}")
    print(f"  - Couches cachées: {hidden_dims}")
    print(f"  - Sortie: {output_dim}")
    
    # Créer le modèle PyTorch
    model = create_pytorch_model(input_dim, hidden_dims, output_dim)
    
    # Exporter le modèle en ONNX
    if args.method == "direct":
        # Utiliser directement la classe ONNXExporter
        print("\nExportation directe avec ONNXExporter...")
        
        exporter = ONNXExporter(
            output_dir=args.output_dir,
            device="cpu",
            opset_version=12
        )
        
        onnx_model_path = exporter.export_pytorch_model(
            model=model,
            input_shape=(input_dim,),
            model_name=args.model_name,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={"state": {0: "batch_size"}, "action": {0: "batch_size"}}
        )
        
        # Tester le modèle ONNX
        test_results = exporter.test_onnx_model(
            model_path=onnx_model_path,
            input_shape=(input_dim,),
            num_samples=5
        )
        
        print(f"\nTest d'inférence ONNX:")
        print(f"  - Formes des sorties: {test_results['output_shapes']}")
        print(f"  - Temps d'inférence: {test_results['inference_time_ms']:.4f} ms")
        print(f"  - Échantillons par seconde: {test_results['samples_per_second']:.2f}")
        
        # Comparer PyTorch et ONNX
        comparison = exporter.compare_pytorch_onnx(
            pytorch_model=model,
            onnx_model_path=onnx_model_path,
            input_shape=(input_dim,),
            num_samples=10
        )
        
        print(f"\nComparaison PyTorch vs ONNX:")
        print(f"  - Correspondance: {'Oui' if comparison['is_close'] else 'Non'}")
        print(f"  - Différence maximale: {comparison['max_difference']:.6f}")
        print(f"  - Différence moyenne: {comparison['mean_difference']:.6f}")
        
    else:
        # Utiliser la fonction d'exportation simplifiée
        print("\nExportation simplifiée avec export_pytorch_policy_model...")
        
        onnx_model_path = export_pytorch_policy_model(
            model=model,
            input_shape=(input_dim,),
            model_name=args.model_name,
            output_dir=args.output_dir
        )
    
    # Benchmark des performances d'inférence
    if args.benchmark:
        print("\nExécution du benchmark d'inférence...")
        
        # Tester différentes tailles de batch
        batch_sizes = [1, 10, 100]
        all_results = []
        
        for batch_size in batch_sizes:
            results = benchmark_inference(
                model_path=onnx_model_path,
                input_shape=(input_dim,),
                num_runs=args.benchmark_runs,
                batch_size=batch_size
            )
            all_results.append(results)
        
        # Tracer le graphique pour le batch_size=1
        plot_inference_comparison(all_results[0])
    
    print("\nExportation ONNX terminée !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemple d'exportation de modèle PyTorch vers ONNX")
    parser.add_argument("--method", type=str, default="direct", 
                        choices=["direct", "simple"],
                        help="Méthode d'exportation à utiliser")
    parser.add_argument("--input_dim", type=int, default=20, 
                        help="Dimension d'entrée du modèle")
    parser.add_argument("--hidden_dims", type=str, default="64,32", 
                        help="Dimensions des couches cachées (séparées par des virgules)")
    parser.add_argument("--output_dim", type=int, default=1, 
                        help="Dimension de sortie du modèle")
    parser.add_argument("--model_name", type=str, default="policy_model", 
                        help="Nom du modèle exporté")
    parser.add_argument("--output_dir", type=str, default="onnx_models", 
                        help="Répertoire de sortie pour les modèles ONNX")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Exécuter un benchmark d'inférence")
    parser.add_argument("--benchmark_runs", type=int, default=1000, 
                        help="Nombre d'exécutions pour le benchmark")
    
    args = parser.parse_args()
    main(args) 