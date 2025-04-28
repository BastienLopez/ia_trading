import logging
from collections import namedtuple

import numpy as np

# Configuration du logger
logger = logging.getLogger("PrioritizedReplayMemory")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class SumTree:
    """
    Structure de données SumTree pour une mémoire de replay priorisée efficace.
    Une somme binaire est utilisée pour stocker les priorités et permettre un échantillonnage
    en O(log n) basé sur la priorité.
    """

    def __init__(self, capacity):
        """
        Initialise la structure SumTree.

        Args:
            capacity (int): Capacité maximale de la mémoire
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Nœuds internes + feuilles
        self.data = np.zeros(
            capacity, dtype=object
        )  # Buffer pour stocker les expériences
        self.data_pointer = 0  # Pointeur pour le remplissage circulaire
        self.size = 0  # Nombre d'éléments actuellement stockés

    def add(self, priority, data):
        """
        Ajoute une nouvelle expérience avec sa priorité.

        Args:
            priority (float): Priorité de l'expérience
            data (Experience): L'expérience à stocker
        """
        # Index de la feuille dans l'arbre
        tree_index = self.data_pointer + self.capacity - 1

        # Stocker l'expérience
        self.data[self.data_pointer] = data

        # Mettre à jour l'arbre avec la nouvelle priorité
        self.update(tree_index, priority)

        # Passer au prochain emplacement (remplissage circulaire)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        # Mettre à jour la taille actuelle
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_index, priority):
        """
        Met à jour la priorité d'une feuille et propage le changement vers la racine.

        Args:
            tree_index (int): Index de la feuille dans l'arbre
            priority (float): Nouvelle priorité
        """
        # Calculer le changement de priorité
        change = priority - self.tree[tree_index]

        # Mettre à jour la feuille
        self.tree[tree_index] = priority

        # Propager le changement à travers l'arbre (jusqu'à la racine)
        while tree_index != 0:
            # Passer au parent
            tree_index = (tree_index - 1) // 2
            # Mettre à jour la somme
            self.tree[tree_index] += change

    def get_leaf(self, value):
        """
        Récupère une feuille en fonction de la valeur de priorité.

        Args:
            value (float): Valeur de priorité cumulée recherchée

        Returns:
            tuple: (tree_index, priority, experience)
        """
        parent_index = 0

        # Descendre dans l'arbre jusqu'à atteindre une feuille
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # Si on atteint la fin de l'arbre, on est sur une feuille
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            # Sinon, vérifier dans quel sous-arbre descendre
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        # Calculer l'index dans le buffer de données
        data_index = leaf_index - (self.capacity - 1)

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        """
        Retourne la priorité totale de l'arbre.

        Returns:
            float: Somme de toutes les priorités
        """
        return self.tree[0]


class PrioritizedReplayMemory:
    """
    Mémoire de replay priorisée qui échantillonne les expériences
    en fonction de leur importance (erreur TD).
    """

    def __init__(
        self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01
    ):
        """
        Initialise la mémoire de replay priorisée.

        Args:
            capacity (int): Capacité maximale de la mémoire
            alpha (float): Exposant déterminant le degré de priorisation (0 = uniforme, 1 = complètement priorisé)
            beta (float): Exposant pour la correction de biais d'importance-sampling (0 = pas de correction, 1 = correction complète)
            beta_increment (float): Incrément pour beta après chaque échantillonnage
            epsilon (float): Petite valeur pour éviter que les priorités soient nulles
        """
        self.sum_tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Priorité maximale initiale

        logger.info(
            f"Mémoire de replay priorisée initialisée avec capacité={capacity}, alpha={alpha}, beta={beta}"
        )

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une nouvelle expérience à la mémoire.

        Args:
            state: État courant
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Booléen indiquant si l'épisode est terminé
        """
        # Créer un objet Experience
        experience = Experience(state, action, reward, next_state, done)

        # Nouvelle expérience reçoit priorité maximale
        priority = self.max_priority**self.alpha

        # Ajouter l'expérience dans l'arbre
        self.sum_tree.add(priority, experience)

    def sample(self, batch_size):
        """
        Échantillonne un batch d'expériences en fonction de leur priorité.

        Args:
            batch_size (int): Taille du batch à échantillonner

        Returns:
            tuple: (batch_indices, batch, is_weights) où:
                - batch_indices sont les indices dans l'arbre
                - batch contient les expériences échantillonnées
                - is_weights sont les poids d'importance-sampling
        """
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        batch = []
        is_weights = np.zeros(batch_size, dtype=np.float32)

        # Segment de priorité total
        priority_segment = self.sum_tree.total_priority() / batch_size

        # Incrémenter beta pour la correction d'importance-sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Calculer min_priority pour normalisation des poids IS
        min_priority = (
            np.min(self.sum_tree.tree[-self.sum_tree.capacity :])
            / self.sum_tree.total_priority()
        )
        if min_priority == 0:
            min_priority = self.epsilon

        # Échantillonner batch_size expériences
        for i in range(batch_size):
            # Échantillonner un segment de priorité aléatoire
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Récupérer l'expérience depuis l'arbre
            index, priority, experience = self.sum_tree.get_leaf(value)

            # Calculer le poids d'importance-sampling
            sampling_probability = priority / self.sum_tree.total_priority()
            is_weights[i] = np.power(sampling_probability / min_priority, -self.beta)

            # Stocker l'index et l'expérience
            batch_indices[i] = index
            batch.append(experience)

        return batch_indices, batch, is_weights

    def update_priorities(self, tree_indices, priorities):
        """
        Met à jour les priorités des expériences après apprentissage.

        Args:
            tree_indices (list): Liste des indices dans l'arbre
            priorities (list): Liste des nouvelles priorités
        """
        for index, priority in zip(tree_indices, priorities):
            # Ajouter epsilon pour éviter les priorités nulles
            priority = (priority + self.epsilon) ** self.alpha

            # Mettre à jour la priorité maximale
            self.max_priority = max(self.max_priority, priority)

            # Mettre à jour l'arbre
            self.sum_tree.update(index, priority)

    def __len__(self):
        """
        Retourne le nombre d'expériences dans la mémoire.

        Returns:
            int: Nombre d'expériences stockées
        """
        return self.sum_tree.size
