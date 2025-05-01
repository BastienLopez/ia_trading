import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import psutil

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.variable_batch import VariableBatchSampler, BatchOptimizer

# Créer un modèle simple pour les tests
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=84, hidden_size=256, output_size=3):
        super(SimpleModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Créer un buffer pour les tests
class DemoReplayBuffer:
    def __init__(self, size=100000, state_dim=84):
        self.size = size
        self.state_dim = state_dim
        self.states = torch.randn(size, state_dim)
        self.actions = torch.randint(0, 3, (size,))
        self.rewards = torch.randn(size)
        self.next_states = torch.randn(size, state_dim)
        self.dones = torch.zeros(size, dtype=torch.bool)
        self.priorities = torch.ones(size)
        
        # Optimisation: créer un index pour simuler l'échantillonnage prioritaire
        self.index = torch.arange(size)
    
    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def sample_batch(self, batch_size, beta=None):
        # Simuler l'échantillonnage prioritaire
        if beta is not None:
            probs = self.priorities / self.priorities.sum()
            indices = torch.multinomial(probs, batch_size, replacement=True)
            weights = (self.size * probs[indices]) ** (-beta)
            weights = weights / weights.max()
        else:
            indices = torch.randint(0, self.size, (batch_size,))
            weights = torch.ones(batch_size)
        
        batch = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
        
        info = {
            "weights": weights,
            "indices": indices,
            "batch_size": batch_size,
            "beta": beta
        }
        
        return batch, info

# Fonction de perte simple
def simple_loss_fn(model, states, actions, rewards, next_states, dones):
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = model(next_states).max(1)[0]
        targets = rewards + 0.99 * next_q_values * (~dones)
    
    # MSE loss
    loss = ((q_values - targets) ** 2).mean()
    return loss

# Tests de performance avec différentes tailles de batch
def benchmark_batch_sizes(model, buffer, device, min_size=32, max_size=1024, n_iterations=100):
    """
    Teste les performances (temps/débit) avec différentes tailles de batch.
    
    Args:
        model: Modèle RL à utiliser
        buffer: Tampon d'expérience
        device: Appareil (CPU/GPU)
        min_size: Taille de batch minimale
        max_size: Taille de batch maximale
        n_iterations: Nombre d'itérations pour chaque taille
        
    Returns:
        Dictionnaire des résultats
    """
    model.to(device)
    model.eval()  # Mode évaluation - pas de gradients
    
    # Calculer les tailles de batch à tester (puissances de 2)
    batch_sizes = []
    size = min_size
    while size <= max_size:
        batch_sizes.append(size)
        size *= 2
    
    # Pour chaque taille, mesurer le temps total et le débit
    results = {}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        for _ in range(n_iterations):
            # Échantillonner un batch
            batch, _ = buffer.sample_batch(batch_size)
            
            # Déplacer sur le device
            states, actions, rewards, next_states, dones = [
                x.to(device) for x in batch
            ]
            
            # Calculer la loss (sans backprop pour le benchmark)
            with torch.no_grad():
                simple_loss_fn(model, states, actions, rewards, next_states, dones)
        
        elapsed = time.time() - start_time
        
        # Calculer des métriques
        total_samples = batch_size * n_iterations
        samples_per_second = total_samples / elapsed
        
        results[batch_size] = {
            "time": elapsed,
            "samples_per_second": samples_per_second,
            "iterations": n_iterations
        }
        
        print(f"Taille de batch {batch_size}: {elapsed:.4f}s, "
              f"{samples_per_second:.1f} échantillons/s")
        
        # Libérer la mémoire
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

# Fonction pour comparer les stratégies du VariableBatchSampler
def compare_strategies(buffer, strategies, n_samples=1000, check_interval=10):
    """
    Compare différentes stratégies d'adaptation de batch.
    
    Args:
        buffer: Tampon d'expérience
        strategies: Liste des stratégies à comparer
        n_samples: Nombre d'échantillons pour chaque stratégie
        check_interval: Intervalle de vérification de l'adaptation
        
    Returns:
        Dictionnaire des résultats
    """
    results = {}
    
    for strategy in strategies:
        # Créer un sampler avec la stratégie
        sampler = VariableBatchSampler(
            buffer=buffer,
            base_batch_size=64,
            min_batch_size=16,
            max_batch_size=512,
            adaptation_speed=0.1,
            check_interval=check_interval
        )
        
        # Définir la stratégie
        sampler.set_strategy(strategy)
        
        # Tracer l'évolution de la taille de batch
        batch_sizes = []
        times = []
        iterations = []
        
        print(f"\nTest de la stratégie: {strategy}")
        
        # Échantillonner
        start_time = time.time()
        for i in range(n_samples):
            # Échantillonner
            _, info = sampler.sample()
            
            # Enregistrer les infos
            batch_sizes.append(info["batch_size"])
            iterations.append(i)
            times.append(time.time() - start_time)
            
            # Afficher la progression toutes les 100 itérations
            if i % 100 == 0:
                print(f"  Itération {i}: Taille de batch {info['batch_size']}")
        
        # Obtenir les métriques finales
        metrics = sampler.get_metrics()
        
        # Collecter les résultats
        results[strategy] = {
            "batch_sizes": batch_sizes,
            "iterations": iterations,
            "times": times,
            "metrics": metrics
        }
    
    return results

# Fonction pour trouver la taille de batch optimale automatiquement
def find_optimal_batch_size_demo(model, buffer, device):
    """
    Démontre l'utilisation du BatchOptimizer pour trouver la taille optimale.
    
    Args:
        model: Modèle RL
        buffer: Tampon d'expérience
        device: Appareil (CPU/GPU)
    """
    print("\nRecherche de la taille de batch optimale...")
    
    # Déplacer le modèle sur l'appareil
    model.to(device)
    
    # Créer l'optimiseur
    optimizer = BatchOptimizer(
        model=model,
        buffer=buffer,
        loss_fn=simple_loss_fn,
        min_batch_size=16,
        max_batch_size=1024,
        warmup_iters=2,
        test_iters=5,
        search_method="binary"
    )
    
    # Trouver la taille optimale
    optimal_size = optimizer.find_optimal_batch_size()
    
    # Afficher la conclusion
    print(f"\nTaille de batch optimale trouvée: {optimal_size}")
    
    # Afficher les résultats graphiquement
    optimizer.plot_results("optimal_batch_size.png")
    
    return optimal_size, optimizer.get_results()

# Affichage des résultats de test de batch
def plot_batch_size_results(results, title="Performance par taille de batch"):
    """
    Affiche graphiquement les résultats des tests de tailles de batch.
    
    Args:
        results: Résultats des benchmarks
        title: Titre du graphique
    """
    batch_sizes = sorted(results.keys())
    times = [results[size]["time"] for size in batch_sizes]
    samples_per_second = [results[size]["samples_per_second"] for size in batch_sizes]
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Premier graphique: Temps total
    ax1.plot(batch_sizes, times, 'o-', label='Temps total')
    ax1.set_title("Temps de traitement")
    ax1.set_xlabel("Taille de batch")
    ax1.set_ylabel("Temps (s)")
    ax1.set_xscale('log', base=2)
    ax1.grid(True)
    
    # Second graphique: Débit
    ax2.plot(batch_sizes, samples_per_second, 'o-', label='Débit')
    ax2.set_title("Débit de traitement")
    ax2.set_xlabel("Taille de batch")
    ax2.set_ylabel("Échantillons/s")
    ax2.set_xscale('log', base=2)
    ax2.grid(True)
    
    # Optimum estimé basé sur le débit
    best_idx = np.argmax(samples_per_second)
    best_batch_size = batch_sizes[best_idx]
    best_throughput = samples_per_second[best_idx]
    
    ax2.plot(best_batch_size, best_throughput, 'ro', markersize=10,
             label=f'Optimal: {best_batch_size}')
    ax2.legend()
    
    # Titre global
    fig.suptitle(title)
    fig.tight_layout()
    
    # Enregistrer et afficher
    plt.savefig("batch_size_performance.png")
    plt.show()

# Affichage des résultats de comparaison des stratégies
def plot_strategy_comparison(results, title="Comparaison des stratégies d'adaptation"):
    """
    Affiche graphiquement la comparaison des stratégies d'adaptation.
    
    Args:
        results: Résultats des comparaisons
        title: Titre du graphique
    """
    strategies = list(results.keys())
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Tracer l'évolution de la taille de batch pour chaque stratégie
    for strategy in strategies:
        iterations = results[strategy]["iterations"]
        batch_sizes = results[strategy]["batch_sizes"]
        
        ax.plot(iterations, batch_sizes, label=strategy)
    
    # Configurer le graphique
    ax.set_title(title)
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Taille de batch")
    ax.legend()
    ax.grid(True)
    
    # Enregistrer et afficher
    plt.savefig("strategy_comparison.png")
    plt.show()

def main(args):
    """
    Fonction principale pour la démonstration.
    """
    # Configurer l'appareil
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    # Créer le modèle et le buffer
    model = SimpleModel(input_size=args.state_dim)
    buffer = DemoReplayBuffer(size=args.buffer_size, state_dim=args.state_dim)
    
    if args.benchmark:
        # Benchmark des différentes tailles de batch
        print("\n=== Benchmark des différentes tailles de batch ===")
        batch_results = benchmark_batch_sizes(
            model, 
            buffer, 
            device,
            min_size=args.min_batch,
            max_size=args.max_batch,
            n_iterations=args.iterations
        )
        
        # Afficher les résultats
        plot_batch_size_results(batch_results)
    
    if args.strategies:
        # Comparer les différentes stratégies
        print("\n=== Comparaison des stratégies d'adaptation ===")
        strategies = ["auto", "gpu", "cpu", "ram", "performance"]
        strategy_results = compare_strategies(
            buffer,
            strategies,
            n_samples=args.adaptations * 10,
            check_interval=10
        )
        
        # Afficher les résultats
        plot_strategy_comparison(strategy_results)
    
    if args.optimize:
        # Trouver la taille de batch optimale
        print("\n=== Recherche de la taille de batch optimale ===")
        optimal_size, optimizer_results = find_optimal_batch_size_demo(model, buffer, device)
    
    # Démonstration en temps réel
    if args.demo:
        print("\n=== Démonstration en temps réel ===")
        
        # Créer un sampler avec adaptation
        sampler = VariableBatchSampler(
            buffer=buffer,
            base_batch_size=64,
            min_batch_size=16,
            max_batch_size=512,
            adaptation_speed=0.2,
            check_interval=5
        )
        
        # Boucle principale
        batch_sizes = []
        iterations = []
        
        try:
            for i in range(args.demo_steps):
                # Échantillonner
                batch, info = sampler.sample()
                
                # Déplacer sur le device et calculer la loss
                states, actions, rewards, next_states, dones = [
                    x.to(device) for x in batch
                ]
                
                with torch.no_grad():
                    loss = simple_loss_fn(model, states, actions, rewards, next_states, dones)
                
                # Enregistrer
                batch_sizes.append(info["batch_size"])
                iterations.append(i)
                
                # Afficher la progression toutes les 10 itérations
                if i % 10 == 0:
                    resources = info["resources"]
                    cpu_util = resources.get("cpu", 0) * 100
                    ram_util = resources.get("ram", 0) * 100
                    gpu_util = resources.get("gpu", 0) * 100 if resources.get("gpu") is not None else "N/A"
                    
                    print(f"Itération {i}: Taille={info['batch_size']}, "
                          f"CPU={cpu_util:.1f}%, RAM={ram_util:.1f}%, GPU={gpu_util}")
            
            # Tracer l'évolution de la taille de batch
            plt.figure(figsize=(10, 5))
            plt.plot(iterations, batch_sizes)
            plt.title("Évolution de la taille de batch pendant la démonstration")
            plt.xlabel("Itérations")
            plt.ylabel("Taille de batch")
            plt.grid(True)
            plt.savefig("batch_evolution.png")
            plt.show()
            
        except KeyboardInterrupt:
            print("\nInterruption par l'utilisateur")
    
    print("\nFin de la démonstration")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démonstration de Variable Batch Sampler")
    
    # Paramètres généraux
    parser.add_argument("--state-dim", type=int, default=84, help="Dimension des états")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Taille du buffer")
    parser.add_argument("--use-gpu", action="store_true", help="Utiliser le GPU si disponible")
    
    # Paramètres de benchmark
    parser.add_argument("--benchmark", action="store_true", help="Exécuter le benchmark de tailles de batch")
    parser.add_argument("--min-batch", type=int, default=32, help="Taille de batch minimale pour le benchmark")
    parser.add_argument("--max-batch", type=int, default=1024, help="Taille de batch maximale pour le benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Nombre d'itérations pour chaque taille de batch")
    
    # Paramètres de comparaison des stratégies
    parser.add_argument("--strategies", action="store_true", help="Comparer les stratégies d'adaptation")
    parser.add_argument("--adaptations", type=int, default=10, help="Nombre d'adaptations à effectuer")
    
    # Recherche de la taille optimale
    parser.add_argument("--optimize", action="store_true", help="Trouver la taille de batch optimale")
    
    # Démonstration en temps réel
    parser.add_argument("--demo", action="store_true", help="Exécuter la démonstration en temps réel")
    parser.add_argument("--demo-steps", type=int, default=200, help="Nombre d'étapes pour la démonstration")
    
    args = parser.parse_args()
    
    # Si aucune option n'est spécifiée, activer toutes les démonstrations
    if not any([args.benchmark, args.strategies, args.optimize, args.demo]):
        args.benchmark = True
        args.strategies = True
        args.optimize = True
        args.demo = True
    
    main(args) 