import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.state_cache import MultiLevelCache, StateCache


# Modèle simple pour la démonstration
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=84, hidden_size=256, output_size=3):
        super(SimpleModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


# Générateur d'états pour la démonstration
class StateGenerator:
    """
    Générateur d'états pour la démonstration.
    Simule différents types de distribution d'états:
    - Uniforme: états complètement aléatoires
    - Zipf: certains états sont beaucoup plus fréquents (loi de puissance)
    - Gaussienne: états concentrés autour de certains centres
    - Séquentielle: états qui évoluent progressivement
    """

    def __init__(self, state_dim=4, n_common_states=100, distribution="zipf", seed=42):
        """
        Initialise le générateur d'états.

        Args:
            state_dim: Dimension des états
            n_common_states: Nombre d'états communs pré-générés
            distribution: Type de distribution ('uniform', 'zipf', 'gaussian', 'sequential')
            seed: Graine aléatoire
        """
        self.state_dim = state_dim
        self.n_common_states = n_common_states
        self.distribution = distribution
        self.rng = np.random.RandomState(seed)

        # Générer des états communs
        self.common_states = []
        for _ in range(n_common_states):
            self.common_states.append(torch.FloatTensor(self.rng.randn(state_dim)))

        # Pour la distribution gaussienne, générer des centres
        if distribution == "gaussian":
            self.centers = []
            for _ in range(10):  # 10 centres
                self.centers.append(torch.FloatTensor(self.rng.randn(state_dim)))

        # Pour la distribution séquentielle
        if distribution == "sequential":
            self.current_state = torch.zeros(state_dim)
            self.step_size = 0.1

        # Compteur pour Zipf
        self.zipf_counter = 0

    def next_state(self):
        """
        Génère l'état suivant selon la distribution choisie.

        Returns:
            Un tenseur PyTorch représentant l'état
        """
        if self.distribution == "uniform":
            # État complètement aléatoire
            return torch.FloatTensor(self.rng.randn(self.state_dim))

        elif self.distribution == "zipf":
            # Distribution en loi de puissance (Zipf)
            # Les premiers états sont beaucoup plus fréquents
            self.zipf_counter += 1

            # Sélectionner un indice selon la loi de Zipf
            if self.zipf_counter % 10 == 0:  # 10% d'états aléatoires
                return torch.FloatTensor(self.rng.randn(self.state_dim))
            else:
                # Générer un indice selon Zipf (plus l'indice est petit, plus il est probable)
                x = self.rng.zipf(1.5)  # Paramètre de la loi Zipf
                index = min(
                    x - 1, self.n_common_states - 1
                )  # Ajuster à l'indice 0-based
                return self.common_states[index]

        elif self.distribution == "gaussian":
            # Sélectionner un centre aléatoirement
            center = self.rng.choice(self.centers)

            # Générer un état autour de ce centre
            noise = torch.FloatTensor(self.rng.randn(self.state_dim) * 0.1)
            return center + noise

        elif self.distribution == "sequential":
            # Ajouter un petit déplacement aléatoire
            delta = torch.FloatTensor(self.rng.randn(self.state_dim) * self.step_size)
            self.current_state = self.current_state + delta

            # Normaliser pour éviter d'exploser
            if torch.norm(self.current_state) > 5.0:
                self.current_state = (
                    self.current_state * 5.0 / torch.norm(self.current_state)
                )

            return self.current_state

        else:
            raise ValueError(f"Distribution inconnue: {self.distribution}")


# Fonctions pour benchmarker les performances
def compute_action(model, state, device):
    """
    Calcule une action pour un état donné.

    Args:
        model: Modèle de policy
        state: État
        device: Appareil (CPU/GPU)

    Returns:
        Action (tenseur)
    """
    # Déplacer sur le device et ajouter une dimension de batch si nécessaire
    if len(state.shape) == 1:
        state = state.unsqueeze(0)

    state = state.to(device)

    # Forward pass
    with torch.no_grad():
        q_values = model(state)

    return q_values.cpu()


def benchmark_without_cache(model, state_generator, n_samples=1000, device="cpu"):
    """
    Mesure les performances sans cache.

    Args:
        model: Modèle de policy
        state_generator: Générateur d'états
        n_samples: Nombre d'échantillons
        device: Appareil

    Returns:
        Dictionnaire de métriques
    """
    start_time = time.time()

    # Mesurer l'utilisation de la mémoire au début
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 * 1024)  # MB

    computation_times = []

    for i in range(n_samples):
        # Générer un état
        state = state_generator.next_state()

        # Mesurer le temps de calcul
        start_compute = time.time()
        action = compute_action(model, state, device)
        computation_time = time.time() - start_compute
        computation_times.append(computation_time)

    # Mesurer l'utilisation de la mémoire à la fin
    end_memory = process.memory_info().rss / (1024 * 1024)  # MB

    # Calculer les métriques
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / n_samples
    avg_computation_time = sum(computation_times) / n_samples
    memory_usage = end_memory - start_memory

    return {
        "total_time": total_time,
        "avg_time_per_sample": avg_time_per_sample,
        "avg_computation_time": avg_computation_time,
        "memory_usage": memory_usage,
        "hits": 0,
        "misses": n_samples,
        "hit_rate": 0.0,
    }


def benchmark_with_cache(
    model,
    state_generator,
    n_samples=1000,
    device="cpu",
    cache_config=None,
    multi_level=False,
):
    """
    Mesure les performances avec cache.

    Args:
        model: Modèle de policy
        state_generator: Générateur d'états
        n_samples: Nombre d'échantillons
        device: Appareil
        cache_config: Configuration du cache (dict)
        multi_level: Utiliser le cache multi-niveaux

    Returns:
        Dictionnaire de métriques
    """
    # Configuration par défaut
    if cache_config is None:
        cache_config = {
            "capacity": 1000,
            "similarity_threshold": 0.001,
            "enable_disk_cache": False,
        }

    # Créer le cache
    if multi_level:
        # Créer un cache multi-niveaux
        levels = {
            "frequent": {
                "capacity": cache_config["capacity"],
                "similarity_threshold": 0.01,
            },
            "rare": {
                "capacity": cache_config["capacity"] // 2,
                "similarity_threshold": 0.001,
            },
        }
        cache = MultiLevelCache(levels=levels)

        # Définir une fonction de sélection de niveau simple
        def level_selector(state):
            if state.sum().item() > 0:
                return "frequent"
            else:
                return "rare"

        cache.set_level_selector(level_selector)
    else:
        # Cache simple
        temp_dir = None
        if cache_config["enable_disk_cache"]:
            temp_dir = tempfile.mkdtemp()

        cache = StateCache(
            capacity=cache_config["capacity"],
            similarity_threshold=cache_config["similarity_threshold"],
            cache_dir=temp_dir,
            enable_disk_cache=cache_config["enable_disk_cache"],
        )

    # Fonction de calcul pour le cache
    def compute_action_fn(state):
        return compute_action(model, state, device)

    start_time = time.time()

    # Mesurer l'utilisation de la mémoire au début
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 * 1024)  # MB

    computation_times = []

    for i in range(n_samples):
        # Générer un état
        state = state_generator.next_state()

        # Mesurer le temps de calcul
        start_compute = time.time()

        # Récupérer l'action du cache ou la calculer
        action, info = cache.get(state, compute_action_fn)

        computation_time = time.time() - start_compute
        computation_times.append(computation_time)

    # Mesurer l'utilisation de la mémoire à la fin
    end_memory = process.memory_info().rss / (1024 * 1024)  # MB

    # Libérer les ressources
    if (
        cache_config["enable_disk_cache"]
        and hasattr(cache, "cache_dir")
        and cache.cache_dir
    ):
        if os.path.exists(cache.cache_dir):
            try:
                import shutil

                shutil.rmtree(cache.cache_dir)
            except:
                pass

    # Récupérer les métriques du cache
    if multi_level:
        cache_metrics = cache.get_metrics()
        hits = cache_metrics["global"]["hits"]
        total_queries = cache_metrics["global"]["total_queries"]
    else:
        cache_metrics = cache.get_metrics()
        hits = cache_metrics["hits"]
        total_queries = cache_metrics["total_queries"]

    # Calculer les métriques
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / n_samples
    avg_computation_time = sum(computation_times) / n_samples
    memory_usage = end_memory - start_memory
    hit_rate = hits / total_queries if total_queries > 0 else 0

    return {
        "total_time": total_time,
        "avg_time_per_sample": avg_time_per_sample,
        "avg_computation_time": avg_computation_time,
        "memory_usage": memory_usage,
        "hits": hits,
        "misses": total_queries - hits,
        "hit_rate": hit_rate,
        "cache_metrics": cache_metrics,
    }


def compare_distributions(model, n_samples=1000, device="cpu"):
    """
    Compare les performances avec différentes distributions d'états.

    Args:
        model: Modèle de policy
        n_samples: Nombre d'échantillons
        device: Appareil

    Returns:
        Dictionnaire de résultats
    """
    distributions = ["uniform", "zipf", "gaussian", "sequential"]
    results = {}

    print("\n=== Comparaison des distributions d'états ===")

    for dist in distributions:
        print(f"\nDistribution: {dist}")

        # Créer le générateur d'états
        state_generator = StateGenerator(
            state_dim=84, n_common_states=100, distribution=dist
        )

        # Benchmark sans cache
        print("Sans cache...")
        no_cache_results = benchmark_without_cache(
            model, state_generator, n_samples, device
        )

        # Benchmark avec cache
        print("Avec cache...")
        cache_results = benchmark_with_cache(model, state_generator, n_samples, device)

        # Calculer l'accélération
        speedup = no_cache_results["total_time"] / cache_results["total_time"]

        print(f"Temps sans cache: {no_cache_results['total_time']:.4f}s")
        print(f"Temps avec cache: {cache_results['total_time']:.4f}s")
        print(f"Accélération: {speedup:.2f}x")
        print(f"Hit rate: {cache_results['hit_rate']:.2%}")

        # Stocker les résultats
        results[dist] = {
            "no_cache": no_cache_results,
            "cache": cache_results,
            "speedup": speedup,
        }

    return results


def compare_cache_configs(model, state_generator, n_samples=1000, device="cpu"):
    """
    Compare les performances avec différentes configurations de cache.

    Args:
        model: Modèle de policy
        state_generator: Générateur d'états
        n_samples: Nombre d'échantillons
        device: Appareil

    Returns:
        Dictionnaire de résultats
    """
    # Différentes configurations à tester
    configs = [
        {
            "name": "Petit cache",
            "config": {
                "capacity": 100,
                "similarity_threshold": 0.001,
                "enable_disk_cache": False,
            },
        },
        {
            "name": "Cache moyen",
            "config": {
                "capacity": 1000,
                "similarity_threshold": 0.001,
                "enable_disk_cache": False,
            },
        },
        {
            "name": "Grand cache",
            "config": {
                "capacity": 10000,
                "similarity_threshold": 0.001,
                "enable_disk_cache": False,
            },
        },
        {
            "name": "Cache avec similarité élevée",
            "config": {
                "capacity": 1000,
                "similarity_threshold": 0.01,
                "enable_disk_cache": False,
            },
        },
        {
            "name": "Cache avec similarité faible",
            "config": {
                "capacity": 1000,
                "similarity_threshold": 0.0001,
                "enable_disk_cache": False,
            },
        },
        {
            "name": "Cache sur disque",
            "config": {
                "capacity": 1000,
                "similarity_threshold": 0.001,
                "enable_disk_cache": True,
            },
        },
        {
            "name": "Cache multi-niveaux",
            "config": {
                "capacity": 1000,
                "similarity_threshold": 0.001,
                "enable_disk_cache": False,
            },
            "multi_level": True,
        },
    ]

    results = {}

    print("\n=== Comparaison des configurations de cache ===")

    # Benchmark sans cache d'abord
    print("\nSans cache...")
    no_cache_results = benchmark_without_cache(
        model, state_generator, n_samples, device
    )

    for config_info in configs:
        name = config_info["name"]
        config = config_info["config"]
        multi_level = config_info.get("multi_level", False)

        print(f"\nConfiguration: {name}")

        # Benchmark avec cette configuration
        cache_results = benchmark_with_cache(
            model,
            state_generator,
            n_samples,
            device,
            cache_config=config,
            multi_level=multi_level,
        )

        # Calculer l'accélération
        speedup = no_cache_results["total_time"] / cache_results["total_time"]

        print(f"Temps sans cache: {no_cache_results['total_time']:.4f}s")
        print(f"Temps avec cache: {cache_results['total_time']:.4f}s")
        print(f"Accélération: {speedup:.2f}x")
        print(f"Hit rate: {cache_results['hit_rate']:.2%}")
        print(f"Utilisation mémoire: {cache_results['memory_usage']:.2f} MB")

        # Stocker les résultats
        results[name] = {
            "config": config,
            "multi_level": multi_level,
            "metrics": cache_results,
            "speedup": speedup,
        }

    return results


def plot_distribution_comparison(
    results, title="Comparaison des distributions d'états"
):
    """
    Affiche graphiquement la comparaison des performances par distribution.

    Args:
        results: Résultats des comparaisons
        title: Titre du graphique
    """
    distributions = list(results.keys())

    # Extraire les métriques
    speedups = [results[dist]["speedup"] for dist in distributions]
    hit_rates = [results[dist]["cache"]["hit_rate"] * 100 for dist in distributions]

    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Premier graphique: Accélération
    ax1.bar(distributions, speedups)
    ax1.set_title("Accélération par distribution")
    ax1.set_xlabel("Distribution")
    ax1.set_ylabel("Accélération (x fois)")
    for i, v in enumerate(speedups):
        ax1.text(i, v + 0.1, f"{v:.2f}x", ha="center")

    # Second graphique: Hit rate
    ax2.bar(distributions, hit_rates)
    ax2.set_title("Taux de hit par distribution")
    ax2.set_xlabel("Distribution")
    ax2.set_ylabel("Hit rate (%)")
    for i, v in enumerate(hit_rates):
        ax2.text(i, v + 1, f"{v:.1f}%", ha="center")

    # Titre global
    fig.suptitle(title)
    fig.tight_layout()

    # Enregistrer et afficher
    plt.savefig("distribution_comparison.png")
    plt.show()


def plot_config_comparison(results, title="Comparaison des configurations de cache"):
    """
    Affiche graphiquement la comparaison des performances par configuration.

    Args:
        results: Résultats des comparaisons
        title: Titre du graphique
    """
    configs = list(results.keys())

    # Extraire les métriques
    speedups = [results[config]["speedup"] for config in configs]
    hit_rates = [results[config]["metrics"]["hit_rate"] * 100 for config in configs]
    memory_usage = [results[config]["metrics"]["memory_usage"] for config in configs]

    # Créer la figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Premier graphique: Accélération
    ax1.bar(configs, speedups)
    ax1.set_title("Accélération par configuration")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Accélération (x fois)")
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    for i, v in enumerate(speedups):
        ax1.text(i, v + 0.1, f"{v:.2f}x", ha="center")

    # Second graphique: Hit rate
    ax2.bar(configs, hit_rates)
    ax2.set_title("Taux de hit par configuration")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Hit rate (%)")
    ax2.set_xticklabels(configs, rotation=45, ha="right")
    for i, v in enumerate(hit_rates):
        ax2.text(i, v + 1, f"{v:.1f}%", ha="center")

    # Troisième graphique: Utilisation mémoire
    ax3.bar(configs, memory_usage)
    ax3.set_title("Utilisation mémoire par configuration")
    ax3.set_xlabel("Configuration")
    ax3.set_ylabel("Mémoire (MB)")
    ax3.set_xticklabels(configs, rotation=45, ha="right")
    for i, v in enumerate(memory_usage):
        ax3.text(i, v + 1, f"{v:.1f} MB", ha="center")

    # Titre global
    fig.suptitle(title)
    fig.tight_layout()

    # Enregistrer et afficher
    plt.savefig("config_comparison.png")
    plt.show()


def main(args):
    """
    Fonction principale pour la démonstration.
    """
    # Configurer l'appareil
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    print(f"Utilisation de l'appareil: {device}")

    # Créer le modèle
    model = SimpleModel(input_size=args.state_dim)
    model.to(device)
    model.eval()

    # Exemple d'utilisation simple du cache
    if args.basic_demo:
        print("\n=== Démonstration basique du cache d'états ===")

        # Créer un cache
        cache = StateCache(capacity=1000, similarity_threshold=0.001)

        # Créer un état
        state = torch.randn(args.state_dim)
        print(f"État créé: shape={state.shape}")

        # Calculer une action et la stocker dans le cache
        action = compute_action(model, state, device)
        cache.put(state, action)
        print(f"Action calculée et mise en cache: {action.shape}")

        # Récupérer l'action du cache
        cached_action, info = cache.get(state)
        print(f"Action récupérée du cache: {cached_action.shape}")
        print(f"Info: {info}")

        # Essayer avec un état similaire
        similar_state = state + torch.randn_like(state) * 0.0005
        similar_action, info = cache.get(similar_state)
        print(f"\nTest avec un état similaire:")
        print(f"Hit du cache: {info['cache_hit']}")

        # Essayer avec un état différent
        different_state = torch.randn(args.state_dim)
        different_action, info = cache.get(different_state)
        print(f"\nTest avec un état différent:")
        print(f"Hit du cache: {info['cache_hit']}")

    # Benchmark des distributions
    if args.distributions:
        dist_results = compare_distributions(
            model, n_samples=args.n_samples, device=device
        )

        # Afficher les résultats graphiquement
        plot_distribution_comparison(dist_results)

    # Benchmark des configurations
    if args.configs:
        # Utiliser la distribution Zipf pour ce test
        state_generator = StateGenerator(
            state_dim=args.state_dim, n_common_states=100, distribution="zipf"
        )

        config_results = compare_cache_configs(
            model, state_generator, n_samples=args.n_samples, device=device
        )

        # Afficher les résultats graphiquement
        plot_config_comparison(config_results)

    # Démonstration du cache multi-niveaux
    if args.multi_level:
        print("\n=== Démonstration du cache multi-niveaux ===")

        # Créer un cache multi-niveaux
        levels = {
            "frequent": {"capacity": 500, "similarity_threshold": 0.01},
            "rare": {"capacity": 200, "similarity_threshold": 0.001},
            "precise": {"capacity": 100, "similarity_threshold": 0.0001},
        }

        multi_cache = MultiLevelCache(levels=levels)

        # Définir une fonction de sélection de niveau
        def level_selector(state):
            # Calculer une métrique simple: somme des valeurs
            sum_val = state.sum().item()

            if sum_val > 2.0:
                return "rare"  # États rares
            elif sum_val < -2.0:
                return "precise"  # États qui nécessitent une grande précision
            else:
                return "frequent"  # États fréquents

        multi_cache.set_level_selector(level_selector)

        # Générer quelques états et les mettre en cache
        generator = StateGenerator(state_dim=args.state_dim, distribution="zipf")

        for i in range(100):
            state = generator.next_state()
            action = compute_action(model, state, device)
            multi_cache.put(state, action)

        # Afficher les métriques par niveau
        metrics = multi_cache.get_metrics()
        print("\nMétriques par niveau:")
        for level, level_metrics in metrics.items():
            if level != "global":
                print(f"Niveau '{level}':")
                print(f"  Capacité: {level_metrics['capacity']}")
                print(f"  Entrées: {level_metrics['memory_entries']}")
                print(f"  Utilisation: {level_metrics['utilization']:.2%}")

        print("\nMétriques globales:")
        print(f"  Niveaux: {metrics['global']['levels']}")
        print(f"  Total des requêtes: {metrics['global']['total_queries']}")

    print("\nFin de la démonstration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démonstration du cache d'états")

    # Paramètres généraux
    parser.add_argument("--state-dim", type=int, default=84, help="Dimension des états")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Utiliser le GPU si disponible"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Nombre d'échantillons pour les benchmarks",
    )

    # Différentes démos
    parser.add_argument(
        "--basic-demo", action="store_true", help="Démonstration basique du cache"
    )
    parser.add_argument(
        "--distributions",
        action="store_true",
        help="Comparaison des distributions d'états",
    )
    parser.add_argument(
        "--configs", action="store_true", help="Comparaison des configurations de cache"
    )
    parser.add_argument(
        "--multi-level",
        action="store_true",
        help="Démonstration du cache multi-niveaux",
    )

    args = parser.parse_args()

    # Si aucune option n'est spécifiée, activer toutes les démonstrations
    if not any([args.basic_demo, args.distributions, args.configs, args.multi_level]):
        args.basic_demo = True
        args.distributions = True
        args.configs = True
        args.multi_level = True

    main(args)
