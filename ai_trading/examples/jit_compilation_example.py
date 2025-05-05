#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation des outils de compilation JIT pour optimiser les modèles
PyTorch et TensorFlow.
"""

import sys
import time
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Importer les utilitaires de compilation JIT
from ai_trading.utils.jit_compilation import (
    TF_AVAILABLE,
    TORCH_AVAILABLE,
    XLA_AVAILABLE,
    TorchScriptCompiler,
    XLAOptimizer,
    compile_model,
    enable_tensorflow_xla,
    optimize_function,
)


def print_separator(title):
    """Affiche un séparateur avec un titre."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def example_pytorch_compilation():
    """Exemple de compilation d'un modèle PyTorch avec TorchScript."""

    if not TORCH_AVAILABLE:
        print("PyTorch n'est pas disponible, exemple ignoré.")
        return

    print_separator("EXEMPLE DE COMPILATION PYTORCH AVEC TORCHSCRIPT")

    import torch
    import torch.nn as nn

    # Création d'un modèle simple mais avec assez de couches pour montrer un gain
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(64, 10)

        def forward(self, x):
            features = self.features(x)
            return self.classifier(features)

    # Créer une instance du modèle
    model = ExampleModel()
    model.eval()  # Mettre en mode évaluation

    # Créer des données d'exemple
    batch_size = 64
    input_data = torch.randn(batch_size, 32)

    # 1. Compilation avec torchscript.trace
    print("\n1. COMPILATION AVEC TORCH.JIT.TRACE")
    print("-" * 50)

    # Créer un compilateur
    compiler = TorchScriptCompiler(save_dir="ai_trading/info_retour/models/jit")

    # Tracer le modèle
    traced_model = compiler.trace_model(
        model=model, example_inputs=input_data, name="example_traced_model"
    )

    print(f"Modèle tracé sauvegardé dans: {compiler.save_dir}/example_traced_model.pt")

    # 2. Benchmark
    print("\n2. BENCHMARK: ORIGINAL VS TRACED MODEL")
    print("-" * 50)

    # Exécuter le benchmark
    benchmark_results = compiler.benchmark_model(
        model=model,
        scripted_model=traced_model,
        input_data=input_data,
        num_warmup=10,
        num_iter=100,
    )

    # Afficher les résultats
    print(
        f"Temps moyen par inférence (original): {benchmark_results['original_avg_ms']:.3f} ms"
    )
    print(
        f"Temps moyen par inférence (compilé): {benchmark_results['scripted_avg_ms']:.3f} ms"
    )
    print(f"Accélération: {benchmark_results['speedup']:.2f}x")

    # 3. Optimisation d'une fonction
    print("\n3. OPTIMISATION D'UNE FONCTION AVEC TORCH.JIT.SCRIPT")
    print("-" * 50)

    # Fonction à optimiser
    def complex_function(x, y):
        result = torch.zeros_like(x)
        for i in range(x.size(0)):
            result[i] = torch.sin(x[i]) * torch.cos(y[i]) + torch.sqrt(
                torch.abs(x[i] * y[i])
            )
        return result

    # Optimiser la fonction
    optimized_func = optimize_function(complex_function)

    # Créer des données pour le test
    test_x = torch.randn(1000)
    test_y = torch.randn(1000)

    # Mesurer le temps d'exécution de la fonction originale
    start_time = time.time()
    original_result = complex_function(test_x, test_y)
    original_time = time.time() - start_time

    # Mesurer le temps d'exécution de la fonction optimisée
    start_time = time.time()
    optimized_result = optimized_func(test_x, test_y)
    optimized_time = time.time() - start_time

    # Vérifier l'égalité des résultats
    is_equal = torch.allclose(original_result, optimized_result, rtol=1e-4)

    # Afficher les résultats
    print(f"Temps d'exécution (original): {original_time * 1000:.3f} ms")
    print(f"Temps d'exécution (optimisé): {optimized_time * 1000:.3f} ms")
    print(f"Accélération: {original_time / optimized_time:.2f}x")
    print(f"Résultats identiques: {is_equal}")

    # 4. API simplifiée
    print("\n4. UTILISATION DE L'API SIMPLIFIÉE")
    print("-" * 50)

    # Compiler le modèle avec l'API simplifiée
    easy_compiled_model = compile_model(
        model=model, example_inputs=input_data, framework="torch", method="trace"
    )

    print("Modèle compilé avec l'API simplifiée.")
    print(f"Type du modèle compilé: {type(easy_compiled_model).__name__}")

    # Test d'inférence
    with torch.no_grad():
        out1 = model(input_data)
        out2 = easy_compiled_model(input_data)

    # Vérifier l'égalité des sorties
    is_equal = torch.allclose(out1, out2, rtol=1e-4)
    print(f"Les sorties sont identiques: {is_equal}")


def example_tensorflow_xla():
    """Exemple d'optimisation d'un modèle TensorFlow avec XLA."""

    if not TF_AVAILABLE:
        print("TensorFlow n'est pas disponible, exemple ignoré.")
        return

    if not XLA_AVAILABLE:
        print(
            "XLA n'est pas disponible dans votre installation TensorFlow, exemple ignoré."
        )
        return

    print_separator("EXEMPLE D'OPTIMISATION TENSORFLOW AVEC XLA")

    import tensorflow as tf

    # Créer un modèle simple
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_shape=(32,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    # Compiler le modèle
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Créer des données d'exemple
    batch_size = 64
    input_data = tf.random.normal((batch_size, 32))

    # 1. Activer XLA globalement
    print("\n1. ACTIVATION DE XLA GLOBALEMENT")
    print("-" * 50)

    result = enable_tensorflow_xla()
    print(f"XLA activé globalement: {result}")

    # 2. Optimiser le modèle avec XLA
    print("\n2. OPTIMISATION DU MODÈLE AVEC XLA")
    print("-" * 50)

    # Créer un optimiseur XLA
    xla_optimizer = XLAOptimizer()

    # Compiler le modèle avec XLA
    xla_model = xla_optimizer.compile_model(model)

    print("Modèle compilé avec XLA.")

    # 3. Benchmark
    print("\n3. BENCHMARK: ORIGINAL VS XLA MODEL")
    print("-" * 50)

    # Exécuter le benchmark
    benchmark_results = xla_optimizer.benchmark_model(
        original_model=model,
        xla_model=xla_model,
        input_data=input_data,
        num_warmup=10,
        num_iter=100,
    )

    # Afficher les résultats
    print(
        f"Temps moyen par inférence (original): {benchmark_results['original_avg_ms']:.3f} ms"
    )
    print(f"Temps moyen par inférence (XLA): {benchmark_results['xla_avg_ms']:.3f} ms")
    print(f"Accélération: {benchmark_results['speedup']:.2f}x")

    # 4. API simplifiée
    print("\n4. UTILISATION DE L'API SIMPLIFIÉE")
    print("-" * 50)

    # Compiler le modèle avec l'API simplifiée
    easy_compiled_model = compile_model(model=model, framework="tensorflow")

    print("Modèle compilé avec l'API simplifiée.")


def main():
    """Fonction principale exécutant tous les exemples."""

    print_separator("EXEMPLES DE COMPILATION JIT")

    # Exemple de compilation PyTorch
    example_pytorch_compilation()

    # Exemple d'optimisation TensorFlow avec XLA
    example_tensorflow_xla()

    print("\nExemple de compilation JIT terminé!")


if __name__ == "__main__":
    main()
