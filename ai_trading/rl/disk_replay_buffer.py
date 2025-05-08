"""
Module de tampon de replay sur disque (Disk Replay Buffer)

Ce module implémente un tampon de replay qui stocke les transitions sur disque (SSD/NVMe)
plutôt qu'en mémoire RAM, permettant de gérer des volumes de données beaucoup plus importants
tout en maintenant des performances acceptables si un disque rapide est utilisé.
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

# Configuration du logger
logger = logging.getLogger("DiskReplayBuffer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
            f.attrs["current_size"] = min(self.buffer_size, current_size + n_items)
            f.attrs["last_updated"] = time.time()

        # Vider le cache
        self.cache = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }

        # Mettre à jour les métriques
        self.metrics["write_time"] += time.time() - start_time
        self.metrics["total_writes"] += 1

        if self.metrics["total_writes"] % 10 == 0:
            logger.debug(f"Cache écrit sur disque en {time.time() - start_time:.4f}s")

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
        # Implémentation de n-step returns si nécessaire
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))

            if len(self.n_step_buffer) < self.n_step:
                return

            if len(self.n_step_buffer) > self.n_step:
                self.n_step_buffer.pop(0)

            # Calculer la récompense cumulée
            cum_reward = 0
            for i, (_, _, r, _, terminal) in enumerate(self.n_step_buffer):
                cum_reward += r * (self.gamma**i)
                if terminal:
                    break

            # Récupérer l'état initial et l'action
            initial_state = self.n_step_buffer[0][0]
            initial_action = self.n_step_buffer[0][1]

            # Récupérer l'état final et le statut de fin
            final_next_state = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]

            # Ajouter au cache
            self.cache["states"].append(initial_state)
            self.cache["actions"].append(initial_action)
            self.cache["rewards"].append(cum_reward)
            self.cache["next_states"].append(final_next_state)
            self.cache["dones"].append(final_done)
        else:
            # Ajouter directement la transition au cache
            self.cache["states"].append(state)
            self.cache["actions"].append(action)
            self.cache["rewards"].append(reward)
            self.cache["next_states"].append(next_state)
            self.cache["dones"].append(done)

        # Mettre à jour la position
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

        # Écrire le cache sur disque si nécessaire
        if len(self.cache["states"]) >= self.cache_size:
            self._flush_cache()

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
        # S'assurer que toutes les données sont sur disque
        self._flush_cache(force=True)

        start_time = time.time()

        with h5py.File(self.h5_path, "r") as f:
            current_size = f.attrs["current_size"]

            # S'assurer que le tampon contient suffisamment d'éléments
            batch_size = min(batch_size, current_size)

            # Pas assez de données
            if batch_size == 0:
                return None

            # Échantillonner des indices aléatoires
            indices = np.random.choice(current_size, batch_size, replace=False)

            # Récupérer les transitions
            states = f["states"][indices]
            actions = f["actions"][indices]
            rewards = f["rewards"][indices]
            next_states = f["next_states"][indices]
            dones = f["dones"][indices]

        # Mettre à jour les métriques
        self.metrics["read_time"] += time.time() - start_time
        self.metrics["total_reads"] += 1

        if self.metrics["total_reads"] % 100 == 0:
            logger.debug(
                f"Batch lu depuis le disque en {time.time() - start_time:.4f}s"
            )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Retourne la taille actuelle du tampon.

        Returns:
            int: Nombre d'éléments dans le tampon
        """
        with h5py.File(self.h5_path, "r") as f:
            current_size = f.attrs["current_size"]

        return current_size + len(self.cache["states"])

    def clear(self):
        """
        Vide le tampon.
        """
        # Vider le cache
        self.cache = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }

        # Réinitialiser le fichier HDF5
        self._init_h5_file()

        # Réinitialiser les compteurs
        self.pos = 0
        self.full = False
        self.n_step_buffer = []

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Retourne des métriques de performance du tampon.

        Returns:
            dict: Métriques de performance
        """
        metrics = self.metrics.copy()

        # Calculer les moyennes de temps
        if metrics["total_writes"] > 0:
            metrics["avg_write_time"] = metrics["write_time"] / metrics["total_writes"]
        else:
            metrics["avg_write_time"] = 0

        if metrics["total_reads"] > 0:
            metrics["avg_read_time"] = metrics["read_time"] / metrics["total_reads"]
        else:
            metrics["avg_read_time"] = 0

        # Ajouter la taille du tampon
        metrics["buffer_size"] = self.buffer_size
        metrics["current_size"] = len(self)

        # Taille du fichier sur disque
        if os.path.exists(self.h5_path):
            metrics["file_size_mb"] = os.path.getsize(self.h5_path) / (1024 * 1024)
        else:
            metrics["file_size_mb"] = 0

        return metrics

    def save_metadata(self, filepath: Optional[str] = None):
        """
        Sauvegarde les métadonnées du tampon pour reprise ultérieure.

        Args:
            filepath: Chemin où sauvegarder les métadonnées (si None, utilise le répertoire du tampon)
        """
        if filepath is None:
            filepath = self.storage_path / "buffer_metadata.pkl"

        metadata = {
            "buffer_size": self.buffer_size,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cache_size": self.cache_size,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "metrics": self.metrics,
            "timestamp": time.time(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Métadonnées du tampon sauvegardées à {filepath}")

    @classmethod
    def load(cls, storage_path: str, metadata_path: Optional[str] = None):
        """
        Charge un tampon existant depuis le disque.

        Args:
            storage_path: Chemin du répertoire de stockage du tampon
            metadata_path: Chemin vers le fichier de métadonnées (si None, cherche dans storage_path)

        Returns:
            DiskReplayBuffer: Instance du tampon chargé
        """
        storage_path = Path(storage_path)

        if metadata_path is None:
            metadata_path = storage_path / "buffer_metadata.pkl"

        # Vérifier que les fichiers existent
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Fichier de métadonnées non trouvé: {metadata_path}"
            )

        h5_path = storage_path / "replay_buffer.h5"
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Fichier de données non trouvé: {h5_path}")

        # Charger les métadonnées
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Créer une nouvelle instance
        buffer = cls(
            buffer_size=metadata["buffer_size"],
            state_dim=metadata["state_dim"],
            action_dim=metadata["action_dim"],
            storage_path=storage_path,
            cache_size=metadata["cache_size"],
            n_step=metadata["n_step"],
            gamma=metadata["gamma"],
            compression=metadata["compression"],
            compression_level=metadata["compression_level"],
        )

        # Restaurer les métriques
        buffer.metrics = metadata["metrics"]

        # Ne pas initialiser le fichier HDF5 puisqu'il existe déjà
        buffer.file_initialized = True

        # Lire les attributs du fichier pour s'assurer que tout est cohérent
        with h5py.File(h5_path, "r") as f:
            buffer.pos = f.attrs["pos"]
            buffer.full = f.attrs["current_size"] >= buffer.buffer_size

        logger.info(f"Tampon de replay chargé depuis {storage_path}")
        logger.info(f"Taille actuelle: {len(buffer)}/{buffer.buffer_size}")

        return buffer


# Exemple d'utilisation:
if __name__ == "__main__":
    # Initialiser un tampon avec des dimensions simples pour test
    buffer = DiskReplayBuffer(
        buffer_size=10000,
        state_dim=(4,),  # États 4D (comme CartPole)
        action_dim=(1,),  # Actions discrètes
        storage_path="./buffer_test",
        cache_size=100,  # Petite taille de cache pour tester les écritures fréquentes
    )

    # Générer quelques données aléatoires
    for i in range(1000):
        state = np.random.randn(4).astype(np.float32)
        action = np.array([np.random.randint(0, 2)]).astype(np.float32)
        reward = np.random.rand().astype(np.float32)
        next_state = np.random.randn(4).astype(np.float32)
        done = bool(np.random.rand() > 0.9)

        buffer.add(state, action, reward, next_state, done)

    # Échantillonner des données
    batch = buffer.sample(64)
    print(f"Batch sample: {[b.shape for b in batch]}")

    # Afficher les métriques de performance
    metrics = buffer.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Sauvegarder les métadonnées
    buffer.save_metadata()

    # Tester le chargement
    loaded_buffer = DiskReplayBuffer.load("./buffer_test")
    print(f"Loaded buffer size: {len(loaded_buffer)}")
