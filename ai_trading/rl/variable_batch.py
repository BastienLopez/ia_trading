import gc
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import psutil
import torch


class VariableBatchSampler:
    """
    Échantillonneur de batchs de taille variable qui ajuste dynamiquement
    la taille des batchs en fonction des ressources système disponibles.

    Permet d'optimiser l'utilisation des ressources en:
    1. Augmentant la taille des batchs quand les ressources sont disponibles
    2. Réduisant la taille des batchs en cas de contraintes
    3. Adaptant la stratégie d'échantillonnage en fonction des performances
    """

    def __init__(
        self,
        buffer: Any,
        base_batch_size: int = 64,
        min_batch_size: int = 16,
        max_batch_size: int = 512,
        target_gpu_util: float = 0.85,
        target_cpu_util: float = 0.75,
        target_ram_util: float = 0.90,
        adaptation_speed: float = 0.05,
        check_interval: int = 10,
        schedule_fn: Optional[Callable[[int], int]] = None,
    ):
        """
        Initialise l'échantillonneur de batchs variables.

        Args:
            buffer: Tampon d'expérience à échantillonner
            base_batch_size: Taille de batch de base
            min_batch_size: Taille minimale de batch
            max_batch_size: Taille maximale de batch
            target_gpu_util: Utilisation GPU cible (0-1)
            target_cpu_util: Utilisation CPU cible (0-1)
            target_ram_util: Utilisation RAM cible (0-1)
            adaptation_speed: Vitesse d'adaptation de la taille (0-1)
            check_interval: Nombre d'échantillonnages entre les vérifications
            schedule_fn: Fonction personnalisée pour calculer la taille de batch
        """
        self.buffer = buffer
        self.base_batch_size = base_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_gpu_util = target_gpu_util
        self.target_cpu_util = target_cpu_util
        self.target_ram_util = target_ram_util
        self.adaptation_speed = adaptation_speed
        self.check_interval = check_interval
        self.schedule_fn = schedule_fn

        # État interne
        self.current_batch_size = base_batch_size
        self.iterations = 0
        self.sample_times = []
        self.batch_sizes_history = []
        self.resource_history = []

        # Dernier échantillon pour analyse
        self.last_sample = None
        self.last_sample_info = None

        # Pour mesurer les performances
        self.last_check_time = time.time()

        # Mode d'adaptation
        self.strategy = "auto"  # "auto", "gpu", "cpu", "ram", "performance"

    def get_resource_utilization(self) -> Dict[str, float]:
        """
        Obtient l'utilisation actuelle des ressources.

        Returns:
            Dictionnaire d'utilisation des ressources
        """
        # Utilisation RAM
        ram = psutil.virtual_memory()
        ram_percent = ram.percent / 100.0

        # Utilisation CPU
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0

        # Utilisation GPU si disponible
        gpu_percent = None
        try:
            import torch

            if torch.cuda.is_available():
                # Obtenir l'utilisation mémoire GPU
                gpu_memory_allocated = torch.cuda.memory_allocated(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_percent = gpu_memory_allocated / gpu_memory_total
        except (ImportError, RuntimeError):
            pass

        return {"ram": ram_percent, "cpu": cpu_percent, "gpu": gpu_percent}

    def adapt_batch_size(self):
        """
        Adapte la taille du batch en fonction des ressources et des performances.
        """
        # Si une fonction de planification personnalisée est fournie, l'utiliser
        if self.schedule_fn is not None:
            self.current_batch_size = max(
                self.min_batch_size,
                min(self.max_batch_size, self.schedule_fn(self.iterations)),
            )
            return

        # Obtenir l'utilisation des ressources
        resources = self.get_resource_utilization()
        self.resource_history.append(resources)

        # Calculer le temps moyen par échantillon
        if len(self.sample_times) > 2:
            avg_time = sum(self.sample_times[-10:]) / len(self.sample_times[-10:])
        else:
            avg_time = 0

        # Base : Aucune adaptation si pas assez de données
        adjustment = 0

        # Stratégie basée sur la GPU
        if resources.get("gpu") is not None and self.strategy in ["auto", "gpu"]:
            gpu_util = resources["gpu"]
            if gpu_util < self.target_gpu_util * 0.8:
                # GPU sous-utilisé, augmenter la taille de batch
                adjustment += self.current_batch_size * self.adaptation_speed
            elif gpu_util > self.target_gpu_util * 1.1:
                # GPU sur-utilisé, réduire la taille de batch
                adjustment -= self.current_batch_size * self.adaptation_speed

        # Stratégie basée sur la RAM
        if self.strategy in ["auto", "ram"]:
            ram_util = resources["ram"]
            if ram_util > self.target_ram_util:
                # RAM sur-utilisée, réduire la taille de batch
                adjustment -= self.current_batch_size * self.adaptation_speed * 2

        # Stratégie basée sur le CPU
        if self.strategy in ["auto", "cpu"]:
            cpu_util = resources["cpu"]
            if cpu_util > self.target_cpu_util:
                # CPU sur-utilisé, réduire la taille de batch
                adjustment -= self.current_batch_size * self.adaptation_speed

        # Appliquer l'ajustement
        if adjustment != 0:
            new_size = max(
                self.min_batch_size,
                min(self.max_batch_size, int(self.current_batch_size + adjustment)),
            )

            # Arrondir à la puissance de 2 la plus proche pour une meilleure efficacité
            power = np.log2(new_size)
            rounded_power = int(power + 0.5)  # Arrondir à l'entier le plus proche
            new_size = 2 ** max(4, min(9, rounded_power))  # Limites entre 16 et 512

            self.current_batch_size = new_size

        # Enregistrer la taille de batch actuelle
        self.batch_sizes_history.append(self.current_batch_size)

    def sample(
        self, prioritized: bool = False, beta: float = None
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Échantillonne un batch du buffer en utilisant la taille actuelle.

        Args:
            prioritized: Utiliser l'échantillonnage prioritaire
            beta: Paramètre beta pour l'échantillonnage prioritaire

        Returns:
            Tuple (batch, info) où info contient des métadonnées sur l'échantillonnage
        """
        start_time = time.time()
        self.iterations += 1

        # Adapter la taille de batch si c'est le moment
        if self.iterations % self.check_interval == 0:
            self.adapt_batch_size()

        # Échantillonner du buffer
        if hasattr(self.buffer, "sample_batch"):
            # Interface personnalisée
            if prioritized and beta is not None:
                batch, batch_info = self.buffer.sample_batch(
                    batch_size=self.current_batch_size, beta=beta
                )
            else:
                batch, batch_info = self.buffer.sample_batch(
                    batch_size=self.current_batch_size
                )
        elif hasattr(self.buffer, "sample"):
            # Interface standard
            if (
                prioritized
                and beta is not None
                and "beta" in self.buffer.sample.__code__.co_varnames
            ):
                batch = self.buffer.sample(self.current_batch_size, beta=beta)
                batch_info = None
            else:
                batch = self.buffer.sample(self.current_batch_size)
                batch_info = None
        else:
            raise ValueError(
                "Buffer incompatible: doit avoir une méthode 'sample' ou 'sample_batch'"
            )

        # Mesurer le temps d'échantillonnage
        sample_time = time.time() - start_time
        self.sample_times.append(sample_time)

        # Limiter la taille de l'historique
        if len(self.sample_times) > 100:
            self.sample_times = self.sample_times[-100:]
        if len(self.batch_sizes_history) > 100:
            self.batch_sizes_history = self.batch_sizes_history[-100:]
        if len(self.resource_history) > 20:
            self.resource_history = self.resource_history[-20:]

        # Stocker le dernier échantillon
        self.last_sample = batch
        self.last_sample_info = batch_info

        # Préparer l'info pour le retour
        info = {
            "batch_size": self.current_batch_size,
            "sample_time": sample_time,
            "iterations": self.iterations,
            "avg_sample_time": sum(self.sample_times[-10:])
            / max(1, len(self.sample_times[-10:])),
            "resources": self.get_resource_utilization(),
            "strategy": self.strategy,
        }

        return batch, info

    def set_strategy(self, strategy: str):
        """
        Définit la stratégie d'adaptation.

        Args:
            strategy: "auto", "gpu", "cpu", "ram", ou "performance"
        """
        valid_strategies = ["auto", "gpu", "cpu", "ram", "performance"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Stratégie invalide. Doit être l'une de {valid_strategies}"
            )
        self.strategy = strategy

    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtient les métriques de performance de l'échantillonneur.

        Returns:
            Dictionnaire de métriques
        """
        metrics = {
            "current_batch_size": self.current_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "iterations": self.iterations,
            "strategy": self.strategy,
            "adaptation_speed": self.adaptation_speed,
        }

        # Ajouter les moyennes
        if self.sample_times:
            metrics["avg_sample_time"] = sum(self.sample_times) / len(self.sample_times)
        if self.batch_sizes_history:
            metrics["avg_batch_size"] = sum(self.batch_sizes_history) / len(
                self.batch_sizes_history
            )

        # Ajouter l'utilisation des ressources
        if self.resource_history:
            last_resources = self.resource_history[-1]
            metrics.update(
                {f"last_{k}": v for k, v in last_resources.items() if v is not None}
            )

        return metrics

    def reset(self):
        """
        Réinitialise l'état de l'échantillonneur.
        """
        self.current_batch_size = self.base_batch_size
        self.iterations = 0
        self.sample_times = []
        self.batch_sizes_history = []
        self.resource_history = []
        self.last_sample = None
        self.last_sample_info = None
        self.last_check_time = time.time()
        gc.collect()  # Nettoyer la mémoire


class BatchOptimizer:
    """
    Optimiseur de batchs qui trouve automatiquement la taille optimale
    en fonction des contraintes système et des performances.

    Utilise une recherche par grille ou binaire pour trouver la taille
    de batch optimale qui maximise le débit d'entraînement.
    """

    def __init__(
        self,
        model: Any,
        buffer: Any,
        loss_fn: Callable,
        min_batch_size: int = 16,
        max_batch_size: int = 1024,
        warmup_iters: int = 5,
        test_iters: int = 10,
        search_method: str = "binary",
    ):
        """
        Initialise l'optimiseur de batch.

        Args:
            model: Modèle à entraîner
            buffer: Tampon d'expérience
            loss_fn: Fonction de perte
            min_batch_size: Taille minimale à tester
            max_batch_size: Taille maximale à tester
            warmup_iters: Nombre d'itérations de warmup
            test_iters: Nombre d'itérations pour chaque test
            search_method: "grid" ou "binary"
        """
        self.model = model
        self.buffer = buffer
        self.loss_fn = loss_fn
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self.search_method = search_method

        # Résultats
        self.results = {}
        self.optimal_batch_size = None

    def test_batch_size(self, batch_size: int) -> float:
        """
        Teste une taille de batch spécifique et retourne le débit.

        Args:
            batch_size: Taille de batch à tester

        Returns:
            Débit en échantillons par seconde
        """
        # Récupérer le device du modèle
        if hasattr(self.model, "device"):
            device = self.model.device
        elif hasattr(next(self.model.parameters()), "device"):
            device = next(self.model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Warmup
        for _ in range(self.warmup_iters):
            # Échantillonner un batch
            if hasattr(self.buffer, "sample_batch"):
                batch, _ = self.buffer.sample_batch(batch_size=batch_size)
            else:
                batch = self.buffer.sample(batch_size)

            # Calculer la perte (sans mise à jour des poids)
            with torch.no_grad():
                if isinstance(batch, tuple) and len(batch) == 5:
                    states, actions, rewards, next_states, dones = batch

                    # Déplacer les tenseurs sur le device
                    if isinstance(states, torch.Tensor):
                        states = states.to(device)
                    if isinstance(actions, torch.Tensor):
                        actions = actions.to(device)
                    if isinstance(rewards, torch.Tensor):
                        rewards = rewards.to(device)
                    if isinstance(next_states, torch.Tensor):
                        next_states = next_states.to(device)
                    if isinstance(dones, torch.Tensor):
                        dones = dones.to(device)

                    self.loss_fn(
                        self.model, states, actions, rewards, next_states, dones
                    )
                elif (
                    hasattr(self.loss_fn, "__code__")
                    and "batch" in self.loss_fn.__code__.co_varnames
                ):
                    # Interface de perte acceptant un batch complet
                    self.loss_fn(self.model, batch)
                else:
                    # Interface simplifiée pour les tests
                    # Supposer que batch est un tuple (states, _) ou juste states
                    if isinstance(batch, tuple):
                        states = batch[0]
                        if isinstance(states, torch.Tensor):
                            states = states.to(device)
                        self.loss_fn(self.model, states)
                    else:
                        if isinstance(batch, torch.Tensor):
                            batch = batch.to(device)
                        self.loss_fn(self.model, batch)

        # Test réel
        start_time = time.time()

        for _ in range(self.test_iters):
            # Échantillonner un batch
            if hasattr(self.buffer, "sample_batch"):
                batch, _ = self.buffer.sample_batch(batch_size=batch_size)
            else:
                batch = self.buffer.sample(batch_size)

            # Calculer la perte (sans mise à jour des poids)
            with torch.no_grad():
                if isinstance(batch, tuple) and len(batch) == 5:
                    states, actions, rewards, next_states, dones = batch

                    # Déplacer les tenseurs sur le device
                    if isinstance(states, torch.Tensor):
                        states = states.to(device)
                    if isinstance(actions, torch.Tensor):
                        actions = actions.to(device)
                    if isinstance(rewards, torch.Tensor):
                        rewards = rewards.to(device)
                    if isinstance(next_states, torch.Tensor):
                        next_states = next_states.to(device)
                    if isinstance(dones, torch.Tensor):
                        dones = dones.to(device)

                    self.loss_fn(
                        self.model, states, actions, rewards, next_states, dones
                    )
                elif (
                    hasattr(self.loss_fn, "__code__")
                    and "batch" in self.loss_fn.__code__.co_varnames
                ):
                    # Interface de perte acceptant un batch complet
                    self.loss_fn(self.model, batch)
                else:
                    # Interface simplifiée pour les tests
                    # Supposer que batch est un tuple (states, _) ou juste states
                    if isinstance(batch, tuple):
                        states = batch[0]
                        if isinstance(states, torch.Tensor):
                            states = states.to(device)
                        self.loss_fn(self.model, states)
                    else:
                        if isinstance(batch, torch.Tensor):
                            batch = batch.to(device)
                        self.loss_fn(self.model, batch)

        elapsed_time = time.time() - start_time

        # Éviter la division par zéro
        if elapsed_time < 1e-6:
            elapsed_time = 1e-6  # Valeur minimale pour éviter division par zéro

        throughput = (batch_size * self.test_iters) / elapsed_time

        # Libérer la mémoire
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return throughput

    def find_optimal_batch_size(self) -> int:
        """
        Trouve la taille de batch optimale en testant différentes tailles.

        Returns:
            Taille de batch optimale
        """
        if self.search_method == "grid":
            return self._grid_search()
        else:
            return self._binary_search()

    def _grid_search(self) -> int:
        """
        Effectue une recherche par grille pour trouver la taille optimale.

        Returns:
            Taille de batch optimale
        """
        batch_sizes = []
        current = self.min_batch_size

        # Générer des tailles de batch en puissances de 2
        while current <= self.max_batch_size:
            batch_sizes.append(current)
            current *= 2

        # Tester chaque taille
        best_size = None
        best_throughput = 0

        for size in batch_sizes:
            throughput = self.test_batch_size(size)
            self.results[size] = throughput

            print(f"Taille de batch {size}: {throughput:.2f} échantillons/sec")

            if throughput > best_throughput:
                best_throughput = throughput
                best_size = size

        # Vérifier si nous avons atteint la limite supérieure
        if best_size == batch_sizes[-1]:
            print("Avertissement: La taille de batch optimale peut être plus grande")

        self.optimal_batch_size = best_size
        return best_size

    def _binary_search(self) -> int:
        """
        Effectue une recherche binaire pour trouver la taille optimale.

        Returns:
            Taille de batch optimale
        """
        left = self.min_batch_size
        right = self.max_batch_size
        best_size = left
        best_throughput = 0

        # Tester d'abord les extrêmes
        left_throughput = self.test_batch_size(left)
        self.results[left] = left_throughput
        print(f"Taille de batch {left}: {left_throughput:.2f} échantillons/sec")

        right_throughput = self.test_batch_size(right)
        self.results[right] = right_throughput
        print(f"Taille de batch {right}: {right_throughput:.2f} échantillons/sec")

        if left_throughput > right_throughput:
            best_throughput = left_throughput
            best_size = left
            # La taille optimale est probablement plus proche de la limite inférieure
            right = left * 4
        else:
            best_throughput = right_throughput
            best_size = right

        # Recherche binaire
        while right - left > self.min_batch_size / 2:
            mid = (left + right) // 2
            # Arrondir à la puissance de 2 la plus proche
            power = np.log2(mid)
            rounded_power = int(power + 0.5)
            mid = 2**rounded_power

            # Éviter de tester deux fois la même taille
            if mid in self.results:
                break

            throughput = self.test_batch_size(mid)
            self.results[mid] = throughput
            print(f"Taille de batch {mid}: {throughput:.2f} échantillons/sec")

            if throughput > best_throughput:
                best_throughput = throughput
                best_size = mid

            # Ajuster les limites de recherche
            if mid < best_size:
                left = mid
            else:
                right = mid

        self.optimal_batch_size = best_size
        return best_size

    def get_results(self) -> Dict[int, float]:
        """
        Retourne les résultats des tests.

        Returns:
            Dictionnaire {taille_batch: débit}
        """
        return self.results

    def plot_results(self, save_path: Optional[str] = None):
        """
        Affiche les résultats sous forme de graphique.

        Args:
            save_path: Chemin pour sauvegarder le graphique
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib non disponible, impossible d'afficher le graphique")
            return

        if not self.results:
            print(
                "Aucun résultat à afficher. Exécutez find_optimal_batch_size() d'abord."
            )
            return

        sizes = list(sorted(self.results.keys()))
        throughputs = [self.results[s] for s in sizes]

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, throughputs, "o-", label="Débit")

        # Marquer le meilleur
        best_idx = throughputs.index(max(throughputs))
        plt.plot(
            sizes[best_idx],
            throughputs[best_idx],
            "ro",
            markersize=10,
            label=f"Optimal: {sizes[best_idx]}",
        )

        plt.title("Performance par taille de batch")
        plt.xlabel("Taille de batch")
        plt.ylabel("Échantillons par seconde")
        plt.xscale("log", base=2)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Graphique sauvegardé: {save_path}")

        plt.show()
