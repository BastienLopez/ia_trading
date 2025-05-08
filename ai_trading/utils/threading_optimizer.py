#!/usr/bin/env python
"""
Module d'optimisation du multithreading et multiprocessing pour le système de trading.
Fournit des outils pour configurer et optimiser l'utilisation des threads et processus.
"""

import logging
import os
import platform
from typing import Callable, Dict, List, Optional, Union

import psutil

# Configuration du logger
logger = logging.getLogger(__name__)


class ThreadingOptimizer:
    """
    Classe pour optimiser l'utilisation des threads et processus dans le système de trading.
    """

    def __init__(self, reserve_cores: int = 1, memory_per_worker: float = 2.0):
        """
        Initialise l'optimiseur de threading.

        Args:
            reserve_cores: Nombre de cœurs à réserver pour le système (non utilisés pour le calcul)
            memory_per_worker: Quantité de mémoire estimée utilisée par chaque worker (en GB)
        """
        self.reserve_cores = max(1, reserve_cores)
        self.memory_per_worker = memory_per_worker

        # Détection du système
        self.system = platform.system()
        self.process = psutil.Process()

        # Détection des ressources
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False) or self.cpu_count

        # Détection de l'hyperthreading
        self.has_hyperthreading = self.cpu_count > self.physical_cores

        # Mémoire disponible
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Nombre optimal de workers basé sur les ressources
        self._calculated_optimal_workers = None

    def calculate_optimal_workers(
        self, force_recalculate: bool = False
    ) -> Dict[str, int]:
        """
        Calcule le nombre optimal de workers pour différentes tâches.

        Args:
            force_recalculate: Si True, force le recalcul même si déjà calculé

        Returns:
            Dictionnaire avec le nombre optimal de workers pour différentes tâches
        """
        if self._calculated_optimal_workers is not None and not force_recalculate:
            return self._calculated_optimal_workers

        # Recalculer la mémoire disponible actuelle
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Calculer le nombre de workers basé sur le CPU
        usable_cores = max(1, self.cpu_count - self.reserve_cores)

        # Traitement spécial pour les systèmes avec hyperthreading
        if self.has_hyperthreading:
            # Sur les systèmes avec hyperthreading, limiter à physical_cores + quelques threads logiques
            cpu_based_workers = min(
                usable_cores, self.physical_cores + self.physical_cores // 2
            )
        else:
            cpu_based_workers = usable_cores

        # Calculer le nombre de workers basé sur la mémoire
        memory_based_workers = max(
            1, int(self.available_memory_gb / self.memory_per_worker)
        )

        # Le nombre optimal est le minimum entre CPU et mémoire
        optimal_workers = min(cpu_based_workers, memory_based_workers)

        # Calculer les valeurs optimales pour différentes tâches
        self._calculated_optimal_workers = {
            "dataloader": optimal_workers,
            "training": max(
                1, min(optimal_workers, self.physical_cores)
            ),  # Pour l'entraînement, limiter aux cœurs physiques
            "preprocessing": max(
                1, optimal_workers - 1
            ),  # Garder un cœur libre pour la boucle principale
            "inference": max(
                1, min(optimal_workers, self.physical_cores * 2)
            ),  # Pour l'inférence, utiliser plus de cœurs
        }

        logger.info(
            f"Nombre optimal de workers calculé: {self._calculated_optimal_workers}"
        )
        logger.info(
            f"Basé sur {self.cpu_count} cœurs logiques, {self.physical_cores} cœurs physiques, "
            f"{self.available_memory_gb:.1f}GB de mémoire disponible"
        )

        return self._calculated_optimal_workers

    def configure_thread_limits(
        self,
        numpy_threads: Optional[int] = None,
        torch_threads: Optional[int] = None,
        omp_threads: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Configure les limites de threads pour les différentes bibliothèques.

        Args:
            numpy_threads: Nombre de threads pour NumPy, si None utilise la valeur optimale
            torch_threads: Nombre de threads pour PyTorch, si None utilise la valeur optimale
            omp_threads: Nombre de threads pour OpenMP, si None utilise la valeur optimale

        Returns:
            Dictionnaire avec les limites configurées
        """
        # Calculer les valeurs optimales si non spécifiées
        optimal = self.calculate_optimal_workers()

        if numpy_threads is None:
            numpy_threads = optimal["training"]

        if torch_threads is None:
            torch_threads = optimal["training"]

        if omp_threads is None:
            omp_threads = optimal["training"]

        # Configurer NumPy
        try:
            import numpy as np

            # Vérifier si la fonction set_num_threads existe
            if hasattr(np, "set_num_threads"):
                np.set_num_threads(numpy_threads)
                logger.info(f"NumPy configuré pour utiliser {numpy_threads} threads")
            else:
                # Alternative pour les versions de NumPy qui n'ont pas set_num_threads
                os.environ["NUMPY_NUM_THREADS"] = str(numpy_threads)
                logger.info(
                    f"NumPy configuré via variable d'environnement NUMPY_NUM_THREADS={numpy_threads}"
                )
        except ImportError:
            logger.warning("NumPy non disponible")

        # Configurer PyTorch
        try:
            import torch

            torch.set_num_threads(torch_threads)
            if torch.cuda.is_available():
                logger.info(
                    f"PyTorch configuré pour utiliser {torch_threads} threads CPU + GPU"
                )
            else:
                logger.info(
                    f"PyTorch configuré pour utiliser {torch_threads} threads CPU"
                )
        except ImportError:
            logger.warning("PyTorch non disponible")

        # Configurer OpenMP et autres via variables d'environnement
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ["MKL_NUM_THREADS"] = str(numpy_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(numpy_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(numpy_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(numpy_threads)
        os.environ["NUMPY_NUM_THREADS"] = str(numpy_threads)

        logger.info(
            f"Variables d'environnement pour threads configurées: OMP={omp_threads}, MKL/BLAS={numpy_threads}"
        )

        return {"numpy": numpy_threads, "torch": torch_threads, "omp": omp_threads}

    def get_dataloader_config(
        self,
        data_size: Optional[int] = None,
        batch_size: int = 32,
        persistent_workers: bool = True,
    ) -> Dict[str, Union[int, bool]]:
        """
        Obtient la configuration optimale pour les DataLoaders PyTorch.

        Args:
            data_size: Taille approximative du dataset, si connue
            batch_size: Taille de batch prévue
            persistent_workers: Si True, maintenir les workers entre les epochs

        Returns:
            Dictionnaire avec la configuration optimale
        """
        optimal = self.calculate_optimal_workers()
        num_workers = optimal["dataloader"]

        # Ajuster en fonction de la taille des données si spécifiée
        if data_size is not None:
            # Pour les petits datasets, réduire le nombre de workers
            if data_size < batch_size * 10:
                num_workers = min(2, num_workers)
                persistent_workers = False
            elif data_size < batch_size * 50:
                num_workers = min(4, num_workers)

        # Calculer le facteur de prefetch optimal
        if num_workers > 0:
            # Plus de prefetch pour plus de workers, mais éviter de surcharger la mémoire
            prefetch_factor = min(4, max(2, num_workers // 2))
        else:
            prefetch_factor = None  # Ignoré quand num_workers=0

        # Déterminer si pin_memory est bénéfique (généralement avec CUDA)
        try:
            import torch

            pin_memory = torch.cuda.is_available()
        except ImportError:
            pin_memory = False

        config = {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }

        logger.info(f"Configuration DataLoader optimale: {config}")
        return config

    def get_multiprocessing_config(
        self, task_type: str = "preprocessing"
    ) -> Dict[str, Union[str, int]]:
        """
        Obtient la configuration optimale pour multiprocessing.

        Args:
            task_type: Type de tâche ('preprocessing', 'training', 'inference')

        Returns:
            Dictionnaire avec la configuration optimale
        """
        optimal = self.calculate_optimal_workers()

        # Obtenir le nombre optimal de processus pour ce type de tâche
        if task_type in optimal:
            num_processes = optimal[task_type]
        else:
            num_processes = optimal["preprocessing"]  # valeur par défaut

        # Sélectionner la méthode de démarrage optimale pour ce système
        if self.system == "Windows":
            # Sur Windows, 'spawn' est plus stable mais plus lent
            start_method = "spawn"
        else:
            # Sur Unix, 'fork' est plus rapide et généralement stable
            # 'forkserver' est un bon compromis
            start_method = "fork"

        # Configuration à retourner
        config = {
            "num_processes": num_processes,
            "start_method": start_method,
            "maxtasksperchild": 10,  # Recycler les processus tous les 10 tâches pour éviter les fuites mémoire
        }

        logger.info(
            f"Configuration multiprocessing optimale pour {task_type}: {config}"
        )
        return config

    def set_process_priority(self, higher_priority: bool = False) -> bool:
        """
        Définit la priorité du processus actuel.

        Args:
            higher_priority: Si True, augmente la priorité du processus

        Returns:
            True si la priorité a été modifiée avec succès
        """
        try:
            if self.system == "Windows":
                # Sur Windows, priorités: IDLE_PRIORITY_CLASS < NORMAL_PRIORITY_CLASS < HIGH_PRIORITY_CLASS
                if higher_priority:
                    self.process.nice(psutil.HIGH_PRIORITY_CLASS)
                    logger.info("Priorité du processus augmentée à HIGH_PRIORITY_CLASS")
                else:
                    self.process.nice(psutil.NORMAL_PRIORITY_CLASS)
                    logger.info("Priorité du processus définie à NORMAL_PRIORITY_CLASS")
            else:
                # Sur Unix, priorités: 20 (le plus bas) à -20 (le plus élevé)
                if higher_priority:
                    # Valeur -10 pour augmenter la priorité sans monopoliser le système
                    self.process.nice(-10)
                    logger.info("Priorité du processus augmentée (nice = -10)")
                else:
                    # Valeur 0 pour une priorité normale
                    self.process.nice(0)
                    logger.info("Priorité du processus définie à normale (nice = 0)")

            return True
        except (psutil.AccessDenied, PermissionError):
            logger.warning(
                "Impossible de modifier la priorité du processus (permission refusée)"
            )
            return False
        except Exception as e:
            logger.error(
                f"Erreur lors de la modification de la priorité du processus: {e}"
            )
            return False


def optimize_torch_dataloader(
    dataset_size: int, batch_size: int
) -> Dict[str, Union[int, bool]]:
    """
    Fonction utilitaire pour obtenir rapidement une configuration optimale de DataLoader.

    Args:
        dataset_size: Taille du dataset
        batch_size: Taille de batch désirée

    Returns:
        Configuration optimale pour DataLoader
    """
    optimizer = ThreadingOptimizer()
    return optimizer.get_dataloader_config(
        data_size=dataset_size, batch_size=batch_size
    )


def optimize_system_for_training():
    """
    Configure le système pour les performances d'entraînement optimales.
    """
    optimizer = ThreadingOptimizer()

    # Configurer les limites de threads
    optimizer.configure_thread_limits()

    # Augmenter la priorité du processus si possible
    optimizer.set_process_priority(higher_priority=True)

    # Désactiver le garbage collector automatique pour éviter les pauses
    # mais le faire tourner explicitement de temps en temps
    import gc

    gc.disable()

    # Configurer le scheduling pour privilégier les performances
    if optimizer.system == "Windows":
        try:
            # Essayer de définir la classe de scheduling "HIGH" sur Windows
            import win32api
            import win32con
            import win32process

            win32process.SetPriorityClass(
                win32api.GetCurrentProcess(), win32con.HIGH_PRIORITY_CLASS
            )
            logger.info("Classe de scheduling Windows définie sur HIGH_PRIORITY_CLASS")
        except ImportError:
            logger.warning(
                "Module win32api non disponible, impossible de modifier la classe de scheduling"
            )
    else:
        try:
            # Sur Linux, essayer d'ajuster la politique de scheduling
            os.system("chrt -f -p 50 {}".format(os.getpid()))
            logger.info("Politique de scheduling Linux ajustée")
        except Exception:
            pass


def parallel_map(
    func: Callable,
    items: List,
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    show_progress: bool = False,
) -> List:
    """
    Exécute une fonction en parallèle sur une liste d'éléments.

    Args:
        func: Fonction à exécuter sur chaque élément
        items: Liste d'éléments à traiter
        max_workers: Nombre maximum de workers, si None utilise la valeur optimale
        use_processes: Si True, utilise des processus au lieu de threads
        show_progress: Si True, affiche une barre de progression

    Returns:
        Liste des résultats
    """
    if max_workers is None:
        optimizer = ThreadingOptimizer()
        optimal = optimizer.calculate_optimal_workers()
        max_workers = optimal["preprocessing"]

    # Limiter le nombre de workers au nombre d'éléments
    max_workers = min(max_workers, len(items))

    if max_workers <= 1 or len(items) <= 1:
        # Exécution séquentielle si un seul worker ou un seul élément
        if show_progress:
            try:
                from tqdm import tqdm

                return [func(item) for item in tqdm(items)]
            except ImportError:
                return [func(item) for item in items]
        else:
            return [func(item) for item in items]

    # Exécution parallèle
    if use_processes:
        # Utiliser des processus (utile pour CPU-bound)
        from concurrent.futures import ProcessPoolExecutor as PoolExecutor
    else:
        # Utiliser des threads (utile pour I/O-bound)
        from concurrent.futures import ThreadPoolExecutor as PoolExecutor

    with PoolExecutor(max_workers=max_workers) as executor:
        if show_progress:
            try:
                from tqdm import tqdm

                results = list(tqdm(executor.map(func, items), total=len(items)))
            except ImportError:
                results = list(executor.map(func, items))
        else:
            results = list(executor.map(func, items))

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    optimizer = ThreadingOptimizer()

    print("\n=== Détection des ressources système ===")
    print(f"Système: {optimizer.system}")
    print(
        f"CPU: {optimizer.cpu_count} cœurs logiques, {optimizer.physical_cores} cœurs physiques"
    )
    print(
        f"Hyperthreading: {'Détecté' if optimizer.has_hyperthreading else 'Non détecté'}"
    )
    print(
        f"Mémoire: {optimizer.total_memory_gb:.1f} GB total, {optimizer.available_memory_gb:.1f} GB disponible"
    )

    print("\n=== Configuration optimale ===")
    optimal_workers = optimizer.calculate_optimal_workers()
    print(f"DataLoader: {optimal_workers['dataloader']} workers")
    print(f"Entraînement: {optimal_workers['training']} threads")
    print(f"Prétraitement: {optimal_workers['preprocessing']} processus")
    print(f"Inférence: {optimal_workers['inference']} threads")

    dataloader_config = optimizer.get_dataloader_config()
    print(f"\nConfiguration DataLoader recommandée:")
    for key, value in dataloader_config.items():
        print(f"  - {key}: {value}")
