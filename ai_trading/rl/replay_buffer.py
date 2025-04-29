import logging
from collections import deque
from typing import Tuple

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
            np.array(states, dtype=np.float16),
            np.array(actions, dtype=np.float16),
            np.array(rewards, dtype=np.float16).reshape(-1, 1),
            np.array(next_states, dtype=np.float16),
            np.array(dones, dtype=np.float16).reshape(-1, 1),
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
        self.priorities = deque(maxlen=buffer_size)  # Stockage des priorités
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
        # Utiliser la priorité maximale pour les nouvelles transitions
        max_priority = max(self.priorities) if self.priorities else 1.0

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
                self.priorities.append(max_priority)
        else:
            # Ajouter directement la transition
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Échantillonne un batch de transitions basé sur les priorités.

        Args:
            batch_size: Taille du batch à échantillonner

        Returns:
            tuple: (states, actions, rewards, next_states, dones, weights, indices)
        """
        # S'assurer que le tampon contient suffisamment d'éléments
        batch_size = min(batch_size, len(self.buffer))

        # Calculer les probabilités d'échantillonnage
        priorities = np.array(self.priorities, dtype=np.float16)
        probabilities = priorities**self.alpha
        probabilities /= np.sum(probabilities)

        # Échantillonner des indices basés sur les priorités
        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities, replace=False
        )

        # Calculer les poids d'importance-sampling
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normaliser

        # Incrémenter beta pour converger vers 1
        self.beta = min(1.0, self.beta + self.beta_increment)

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
            np.array(states, dtype=np.float16),
            np.array(actions, dtype=np.float16),
            np.array(rewards, dtype=np.float16).reshape(-1, 1),
            np.array(next_states, dtype=np.float16),
            np.array(dones, dtype=np.float16).reshape(-1, 1),
            np.array(weights, dtype=np.float16).reshape(-1, 1),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Met à jour les priorités basées sur les erreurs TD.

        Args:
            indices: Indices des transitions dans le tampon
            td_errors: Erreurs TD absolues correspondantes
        """
        for idx, error in zip(indices, td_errors):
            # Ajouter une petite constante pour éviter les priorités nulles
            priority = (float(error) + self.epsilon) ** self.alpha

            # Mettre à jour la priorité
            self.priorities[idx] = priority

    def clear(self):
        """
        Vide le tampon et les priorités.
        """
        super().clear()
        self.priorities.clear()
