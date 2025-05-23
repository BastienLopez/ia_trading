"""
Module de tampons de replay pour l'apprentissage par renforcement.

Ce module contient plusieurs implémentations de tampons de replay :
- ReplayBuffer : Tampon de replay standard
- PrioritizedReplayBuffer : Tampon de replay prioritaire basé sur les erreurs TD
- NStepReplayBuffer : Tampon de replay avec retours sur n étapes
- DiskReplayBuffer : Tampon de replay stocké sur disque pour gérer de grands volumes de données
"""

import logging
import os
import pickle
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

# Configuration du logger
logger = logging.getLogger("ReplayBuffer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def nstep_preprocess(
    current_state: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    n_step: int = 1,
    gamma: float = 0.99,
    buffer: deque = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    """
    Prétraite une transition pour n-step returns.

    Args:
        current_state: État actuel
        action: Action prise
        reward: Récompense reçue
        next_state: État suivant
        done: Indicateur de fin d'épisode
        n_step: Nombre d'étapes pour les retours
        gamma: Facteur d'actualisation
        buffer: Tampon temporaire pour les n-step returns

    Returns:
        tuple: (état actuel traité, action, récompense cumulée, état suivant traité, done)
    """
    if n_step == 1:
        return current_state, action, reward, next_state, done

    if buffer is None:
        buffer = deque(maxlen=n_step)

    # Ajouter la transition au tampon temporaire
    buffer.append((current_state, action, reward, next_state, done))

    # Si le tampon n'est pas assez rempli, retourner None
    if len(buffer) < n_step:
        return None, None, None, None, None

    # Récupérer l'état et l'action de la première transition du tampon
    initial_state, initial_action, _, _, _ = buffer[0]

    # Calculer la récompense cumulée actualisée
    cum_reward = 0
    for i in range(n_step):
        if i >= len(buffer):
            break
        cum_reward += (gamma**i) * buffer[i][2]

    # Récupérer l'état final et l'indicateur de fin d'épisode
    final_next_state = buffer[-1][3]
    final_done = buffer[-1][4]

    # Si l'une des transitions est terminée, marquer comme terminé
    for i in range(len(buffer)):
        if buffer[i][4]:
            final_done = True
            break

    return initial_state, initial_action, cum_reward, final_next_state, final_done


class ReplayBuffer:
    """
    Tampon de replay standard pour stocker et échantillonner des transitions.
    """

    def __init__(self, buffer_size: int, n_step: int = 1, gamma: float = 0.99):
        """
        Initialise le tampon de replay.

        Args:
            buffer_size: Taille maximale du tampon
            n_step: Nombre d'étapes pour les retours
            gamma: Facteur d'actualisation
        """
        self.buffer = deque(maxlen=buffer_size)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        logger.info(
            f"Tampon de replay initialisé avec taille={buffer_size}, n_step={n_step}"
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Ajoute une transition au tampon.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        # Prétraiter pour n-step returns si nécessaire
        if self.n_step > 1:
            exp = nstep_preprocess(
                state,
                action,
                reward,
                next_state,
                done,
                self.n_step,
                self.gamma,
                self.n_step_buffer,
            )
            if exp[0] is not None:  # Vérifier que l'expérience est valide
                s, a, r, ns, d = exp
                self.buffer.append((s, a, r, ns, d))
        else:
            # Ajouter directement la transition
            self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Échantillonne un batch de transitions aléatoires.

        Args:
            batch_size: Taille du batch à échantillonner

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        # S'assurer que le tampon contient suffisamment d'éléments
        batch_size = min(batch_size, len(self.buffer))

        # Échantillonner des indices aléatoires
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # Récupérer les transitions
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # Convertir en tableaux numpy/tensors
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        """
        Retourne la taille actuelle du tampon.

        Returns:
            int: Nombre d'éléments dans le tampon
        """
        return len(self.buffer)

    def clear(self):
        """
        Vide le tampon.
        """
        self.buffer.clear()
        self.n_step_buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Tampon de replay prioritaire basé sur les erreurs TD.
    """

    def __init__(
        self,
        buffer_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        n_step: int = 1,
        gamma: float = 0.99,
        epsilon: float = 1e-6,
    ):
        """
        Initialise le tampon de replay prioritaire.

        Args:
            buffer_size: Taille maximale du tampon
            alpha: Exposant qui détermine l'intensité de la priorisation (0 = uniforme, 1 = greedy)
            beta: Exposant pour la correction du biais d'importance-sampling (0 = pas de correction)
            beta_increment: Incrément pour beta à chaque échantillonnage (pour converger vers 1)
            n_step: Nombre d'étapes pour les retours
            gamma: Facteur d'actualisation
            epsilon: Petit constante pour éviter les priorités nulles
        """
        super().__init__(buffer_size, n_step, gamma)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0
        logger.info(
            f"Tampon de replay prioritaire initialisé avec alpha={alpha}, beta={beta}"
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Ajoute une transition au tampon avec priorité maximale.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        # Utiliser la priorité maximale pour les nouvelles expériences
        max_priority = self.max_priority if len(self.buffer) > 0 else 1.0

        # Ajouter la transition au tampon
        super().add(state, action, reward, next_state, done)

        # Mettre à jour la priorité
        if len(self.buffer) <= self.priorities.shape[0]:
            self.priorities[len(self.buffer) - 1] = max_priority

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Échantillonne un batch de transitions avec priorités.

        Args:
            batch_size: Taille du batch à échantillonner

        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self) < batch_size:
            raise ValueError("Pas assez de transitions dans le tampon")

        # Calculer les probabilités d'échantillonnage
        priorities = self.priorities[:len(self)]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)  # Normaliser les probabilités

        # Échantillonner les indices
        indices = np.random.choice(len(self), batch_size, replace=False, p=probabilities)

        # Calculer les poids d'importance
        weights = (len(self) * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normaliser les poids

        # Incrémenter beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Échantillonner les transitions
        states, actions, rewards, next_states, dones = super().sample(batch_size)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Met à jour les priorités des transitions échantillonnées.

        Args:
            indices: Indices des transitions à mettre à jour
            td_errors: Erreurs TD correspondantes
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class NStepReplayBuffer(ReplayBuffer):
    """
    Tampon de replay qui stocke les transitions et calcule les retours sur n étapes.

    Cette implémentation accumule les récompenses sur n étapes et utilise un
    facteur d'actualisation pour calculer les retours multi-étapes, ce qui peut
    aider à propager les récompenses plus rapidement et améliorer l'apprentissage.

    Référence:
        "Rainbow: Combining Improvements in Deep Reinforcement Learning"
        https://arxiv.org/abs/1710.02298
    """

    def __init__(self, buffer_size: int = 100000, n_steps: int = 3, gamma: float = 0.99):
        """
        Initialise le tampon de replay avec retours multi-étapes.

        Args:
            buffer_size: Taille maximale du tampon
            n_steps: Nombre d'étapes pour accumuler les récompenses
            gamma: Facteur d'actualisation pour les récompenses futures
        """
        super().__init__(buffer_size, n_steps, gamma)
        logger.info(
            f"Tampon de replay à {n_steps} étapes initialisé: buffer_size={buffer_size}, gamma={gamma}"
        )

    def handle_episode_end(self):
        """
        Gère la fin d'un épisode en traitant les expériences restantes dans le tampon temporaire.
        Cette méthode doit être appelée à la fin de chaque épisode.
        """
        # Traiter chaque expérience restante dans le tampon temporaire
        while len(self.n_step_buffer) > 0:
            # Calculer le retour pour les expériences restantes
            reward, next_state, done = self._calculate_n_step_return()

            # Récupérer l'état et l'action initiale
            state, action, _, _, _ = self.n_step_buffer[0]

            # Ajouter l'expérience au tampon principal
            self.buffer.append((state, action, reward, next_state, done))

            # Retirer la première expérience du tampon temporaire
            self.n_step_buffer.popleft()

    def _calculate_n_step_return(self) -> Tuple[float, np.ndarray, bool]:
        """
        Calcule le retour sur n étapes pour le tampon temporaire actuel.

        Returns:
            tuple: (récompense accumulée, état final, indicateur de fin)
        """
        # Initialiser avec les valeurs du dernier état
        _, _, _, last_next_state, last_done = self.n_step_buffer[-1]

        # Calculer la récompense accumulée
        n_step_reward = 0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            # Accumuler les récompenses avec actualisation
            n_step_reward += r * (self.gamma**i)

            # Si un épisode se termine avant les n étapes, on s'arrête là
            if d:
                return n_step_reward, last_next_state, True

        # Retourner la récompense accumulée, le dernier état et l'indicateur de fin
        return n_step_reward, last_next_state, last_done


class DiskReplayBuffer:
    """
    Tampon de replay qui stocke les transitions sur disque pour économiser la RAM.

    Cette implémentation est optimisée pour les disques SSD/NVMe et utilise:
    - Un petit cache en mémoire pour les opérations fréquentes
    - Le format HDF5 pour un stockage efficace des tableaux numpy
    - L'écriture asynchrone pour minimiser la latence
    - La compression pour réduire l'espace disque utilisé
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: Union[int, Tuple[int, ...]],
        storage_path: str = "./buffer_storage",
        cache_size: int = 1000,
        n_step: int = 1,
        gamma: float = 0.99,
        compression: str = "gzip",
        compression_level: int = 4,
    ):
        """
        Initialise le tampon de replay sur disque.

        Args:
            buffer_size: Taille maximale du tampon
            state_dim: Dimensions de l'état (int ou tuple)
            action_dim: Dimensions de l'action (int ou tuple)
            storage_path: Chemin où stocker les fichiers sur disque
            cache_size: Nombre de transitions à garder en cache (RAM)
            n_step: Nombre d'étapes pour les retours
            gamma: Facteur d'actualisation
            compression: Type de compression HDF5 ('gzip', 'lzf', None)
            compression_level: Niveau de compression (1-9, 9 étant le plus compressé)
        """
        self.buffer_size = buffer_size
        self.storage_path = Path(storage_path)
        self.cache_size = min(cache_size, buffer_size)
        self.n_step = n_step
        self.gamma = gamma
        self.compression = compression
        self.compression_level = compression_level

        # S'assurer que les dimensions sont sous forme de tuple
        self.state_dim = state_dim if isinstance(state_dim, tuple) else (state_dim,)
        self.action_dim = action_dim if isinstance(action_dim, tuple) else (action_dim,)

        # Créer le répertoire de stockage s'il n'existe pas
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Nom du fichier HDF5 principal
        self.h5_path = self.storage_path / "replay_buffer.h5"

        # Cache en mémoire
        self.cache = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }

        # Cache pour n-step returns
        self.n_step_buffer = []

        # Position courante dans le tampon
        self.pos = 0
        self.full = False  # Indique si le tampon est plein

        # État du fichier
        self.file_initialized = False
        self._init_h5_file()

        # Métriques de performance
        self.metrics = {
            "write_time": 0,
            "read_time": 0,
            "total_writes": 0,
            "total_reads": 0,
        }

        logger.info(f"Tampon de replay sur disque initialisé à {self.h5_path}")
        logger.info(f"Taille du tampon: {buffer_size}, Taille du cache: {cache_size}")
        logger.info(f"Compression: {compression} niveau {compression_level}")

    def _init_h5_file(self):
        """
        Initialise le fichier HDF5 avec les datasets nécessaires.
        """
        # Création d'un fichier HDF5 vide avec les datasets nécessaires
        with h5py.File(self.h5_path, "w") as f:
            # Créer les datasets pour chaque type de données
            f.create_dataset(
                "states",
                shape=(self.buffer_size, *self.state_dim),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_level,
            )

            f.create_dataset(
                "actions",
                shape=(self.buffer_size, *self.action_dim),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_level,
            )

            f.create_dataset(
                "rewards",
                shape=(self.buffer_size, 1),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_level,
            )

            f.create_dataset(
                "next_states",
                shape=(self.buffer_size, *self.state_dim),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_level,
            )

            f.create_dataset(
                "dones",
                shape=(self.buffer_size, 1),
                dtype=np.bool_,
                compression=self.compression,
                compression_opts=self.compression_level,
            )

            # Métadonnées
            f.attrs["buffer_size"] = self.buffer_size
            f.attrs["current_size"] = 0
            f.attrs["pos"] = 0
            f.attrs["last_updated"] = time.time()

        self.file_initialized = True
        logger.info("Fichier HDF5 initialisé")

    def _flush_cache(self, force: bool = False):
        """
        Écrit le cache en mémoire sur le disque lorsque nécessaire.

        Args:
            force: Forcer l'écriture même si le cache n'est pas plein
        """
        if not self.cache["states"] or (
            len(self.cache["states"]) < self.cache_size and not force
        ):
            return

        start_time = time.time()

        with h5py.File(self.h5_path, "r+") as f:
            current_size = f.attrs["current_size"]
            pos = f.attrs["pos"]

            # Récupérer le nombre d'éléments dans le cache
            n_items = len(self.cache["states"])

            # Gérer le cas où le tampon est plein (écraser les plus anciennes entrées)
            if pos + n_items > self.buffer_size:
                # Première partie (jusqu'à la fin du buffer)
                first_part = self.buffer_size - pos

                # Écrire la première partie (jusqu'à la fin du buffer)
                f["states"][pos:] = np.array(self.cache["states"][:first_part])
                f["actions"][pos:] = np.array(self.cache["actions"][:first_part])
                f["rewards"][pos:] = np.array(
                    self.cache["rewards"][:first_part]
                ).reshape(-1, 1)
                f["next_states"][pos:] = np.array(
                    self.cache["next_states"][:first_part]
                )
                f["dones"][pos:] = np.array(self.cache["dones"][:first_part]).reshape(
                    -1, 1
                )

                # Écrire la seconde partie (retour au début du buffer)
                remaining = n_items - first_part
                if remaining > 0:
                    f["states"][:remaining] = np.array(
                        self.cache["states"][first_part:]
                    )
                    f["actions"][:remaining] = np.array(
                        self.cache["actions"][first_part:]
                    )
                    f["rewards"][:remaining] = np.array(
                        self.cache["rewards"][first_part:]
                    ).reshape(-1, 1)
                    f["next_states"][:remaining] = np.array(
                        self.cache["next_states"][first_part:]
                    )
                    f["dones"][:remaining] = np.array(
                        self.cache["dones"][first_part:]
                    ).reshape(-1, 1)

                pos = remaining
                self.full = True
            else:
                # Cas simple: assez d'espace dans le buffer
                f["states"][pos : pos + n_items] = np.array(self.cache["states"])
                f["actions"][pos : pos + n_items] = np.array(self.cache["actions"])
                f["rewards"][pos : pos + n_items] = np.array(
                    self.cache["rewards"]
                ).reshape(-1, 1)
                f["next_states"][pos : pos + n_items] = np.array(
                    self.cache["next_states"]
                )
                f["dones"][pos : pos + n_items] = np.array(self.cache["dones"]).reshape(
                    -1, 1
                )

                pos += n_items

            # Mettre à jour les métadonnées
            f.attrs["pos"] = pos
            f.attrs["current_size"] = min(current_size + n_items, self.buffer_size)
            f.attrs["last_updated"] = time.time()

        # Vider le cache
        for key in self.cache:
            self.cache[key] = []

        # Mettre à jour les métriques
        self.metrics["write_time"] += time.time() - start_time
        self.metrics["total_writes"] += 1

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Ajoute une transition au tampon.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        # Prétraiter pour n-step returns si nécessaire
        if self.n_step > 1:
            exp = nstep_preprocess(
                state,
                action,
                reward,
                next_state,
                done,
                self.n_step,
                self.gamma,
                self.n_step_buffer,
            )
            if exp[0] is not None:  # Vérifier que l'expérience est valide
                s, a, r, ns, d = exp
                self._add_to_cache(s, a, r, ns, d)
        else:
            # Ajouter directement la transition
            self._add_to_cache(state, action, reward, next_state, done)

    def _add_to_cache(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Ajoute une transition au cache en mémoire.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        self.cache["states"].append(state)
        self.cache["actions"].append(action)
        self.cache["rewards"].append(reward)
        self.cache["next_states"].append(next_state)
        self.cache["dones"].append(done)

        # Écrire sur le disque si le cache est plein
        if len(self.cache["states"]) >= self.cache_size:
            self._flush_cache()

    def sample(self, batch_size):
        """Échantillonne un batch de transitions."""
        if len(self) < batch_size:
            raise ValueError("Pas assez de transitions dans le tampon")

        # Sélectionner des indices aléatoires
        indices = np.random.choice(len(self), batch_size, replace=False)
        indices.sort()  # Trier les indices pour l'indexation HDF5

        # Lire les données du fichier HDF5
        with h5py.File(self.h5_path, "r") as f:
            states = f["states"][indices]
            actions = f["actions"][indices]
            rewards = f["rewards"][indices]
            next_states = f["next_states"][indices]
            dones = f["dones"][indices]

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Retourne la taille actuelle du tampon.

        Returns:
            int: Nombre d'éléments dans le tampon
        """
        with h5py.File(self.h5_path, "r") as f:
            return f.attrs["current_size"]

    def clear(self):
        """
        Vide le tampon.
        """
        # Vider le cache
        for key in self.cache:
            self.cache[key] = []

        # Réinitialiser le fichier HDF5
        self._init_h5_file()

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Retourne les métriques de performance du tampon.

        Returns:
            dict: Métriques de performance
        """
        return {
            "write_time": self.metrics["write_time"],
            "read_time": self.metrics["read_time"],
            "total_writes": self.metrics["total_writes"],
            "total_reads": self.metrics["total_reads"],
            "avg_write_time": self.metrics["write_time"] / max(1, self.metrics["total_writes"]),
            "avg_read_time": self.metrics["read_time"] / max(1, self.metrics["total_reads"]),
        }

    def save_metadata(self, filepath: Optional[str] = None):
        """
        Sauvegarde les métadonnées du tampon.

        Args:
            filepath: Chemin du fichier de métadonnées (optionnel)
        """
        if filepath is None:
            filepath = self.storage_path / "buffer_metadata.pkl"

        metadata = {
            "buffer_size": self.buffer_size,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "cache_size": self.cache_size,
            "metrics": self.metrics,
        }

        with open(filepath, "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, storage_path: str, metadata_path: str) -> "DiskReplayBuffer":
        """Charge un tampon de replay depuis le disque."""
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        buffer = cls(
            buffer_size=metadata["buffer_size"],
            state_dim=metadata["state_dim"],
            action_dim=metadata["action_dim"],
            storage_path=storage_path,
            cache_size=metadata["cache_size"],  # Utiliser la taille du cache sauvegardée
        )

        return buffer
