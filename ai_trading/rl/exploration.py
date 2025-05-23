import logging
import math
from collections import defaultdict

import numpy as np

# Configuration du logger
logger = logging.getLogger("AdvancedExploration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class UCBExploration:
    """
    Implémentation de l'algorithme d'exploration Upper Confidence Bound (UCB).

    UCB équilibre l'exploitation des actions connues comme bonnes et l'exploration
    des actions moins testées en utilisant la formule:
    UCB = Q(a) + c * sqrt(ln(t) / N(a))

    Où:
    - Q(a) est la valeur estimée de l'action a
    - c est un paramètre d'exploration
    - t est le nombre total d'actions prises
    - N(a) est le nombre de fois que l'action a a été choisie
    """

    def __init__(self, action_size, c=2.0):
        """
        Initialise l'explorateur UCB.

        Args:
            action_size (int): Nombre d'actions possibles
            c (float): Paramètre de confiance pour UCB (plus c est grand, plus l'exploration est forte)
        """
        self.action_size = action_size
        self.c = c
        self.action_values = np.zeros(action_size, dtype=np.float16)
        self.action_counts = np.zeros(action_size, dtype=np.int32)
        self.total_steps = 0

        logger.info(f"UCBExploration initialisé avec {action_size} actions et c={c}")

    def select_action(self, q_values, state=None):
        """
        Sélectionne une action en utilisant la stratégie UCB.

        Args:
            q_values (numpy.array): Valeurs Q estimées par le modèle
            state (numpy.array, optional): État actuel (non utilisé pour UCB)

        Returns:
            int: Action sélectionnée
        """
        self.total_steps += 1

        # Si une action n'a jamais été choisie, la sélectionner
        # (exploration pure pour les actions non-testées)
        if np.any(self.action_counts == 0):
            untried_actions = np.where(self.action_counts == 0)[0]
            action = np.random.choice(untried_actions)
            logger.debug(f"Action {action} sélectionnée (jamais essayée)")
            return action

        # Calculer le score UCB pour chaque action
        ucb_values = np.zeros(self.action_size)
        for a in range(self.action_size):
            # Q-value (exploitation)
            exploitation = q_values[a]

            # Terme UCB (exploration)
            exploration = self.c * math.sqrt(
                math.log(self.total_steps) / self.action_counts[a]
            )

            # Score UCB final
            ucb_values[a] = exploitation + exploration

        # Sélectionner l'action avec le score UCB le plus élevé
        action = np.argmax(ucb_values)

        logger.debug(
            f"Action {action} sélectionnée avec UCB score {ucb_values[action]}"
        )
        return action

    def update(self, action, reward):
        """
        Met à jour les statistiques pour l'action sélectionnée.

        Args:
            action (int): Action sélectionnée
            reward (float): Récompense reçue
        """
        # Incrémenter le compteur d'actions
        self.action_counts[action] += 1

        # Mettre à jour la valeur estimée de l'action (moyenne des récompenses)
        # Formule: Q_n = Q_{n-1} + (r_n - Q_{n-1}) / n
        n = self.action_counts[action]
        old_value = self.action_values[action]
        self.action_values[action] = old_value + (reward - old_value) / n

        logger.debug(
            f"Action {action} mise à jour: count={n}, value={self.action_values[action]}"
        )

    def reset(self):
        """
        Réinitialise l'explorateur UCB.
        """
        self.action_values = np.zeros(self.action_size, dtype=np.float16)
        self.action_counts = np.zeros(self.action_size, dtype=np.int32)
        self.total_steps = 0
        logger.info("UCBExploration réinitialisé")


class NoveltyExploration:
    """
    Exploration basée sur la nouveauté (Novelty-based exploration).

    Cette approche encourage l'exploration des états qui sont "nouveaux" ou rarement visités
    en attribuant un bonus de nouveauté aux états moins fréquemment visités.
    """

    def __init__(
        self,
        state_size,
        action_size,
        novelty_scale=1.0,
        decay_rate=0.99,
        hash_bins=10,
        max_count=1000,
    ):
        """
        Initialise l'explorateur basé sur la nouveauté.

        Args:
            state_size (int): Taille de l'espace d'état
            action_size (int): Taille de l'espace d'action
            novelty_scale (float): Échelle du bonus de nouveauté
            decay_rate (float): Taux de décroissance de la nouveauté
            hash_bins (int): Nombre de bins pour le hachage de l'état
            max_count (int): Valeur maximale du compteur d'états
        """
        self.state_size = state_size
        self.action_size = action_size
        self.novelty_scale = novelty_scale
        self.decay_rate = decay_rate
        self.hash_bins = hash_bins
        self.max_count = max_count

        # Dictionnaire pour compter les visites des états
        self.state_counts = defaultdict(int)

        logger.info(
            f"NoveltyExploration initialisé: state_size={state_size}, "
            f"action_size={action_size}, novelty_scale={novelty_scale}"
        )

    def _hash_state(self, state):
        """
        Convertit un état continu en une représentation discrète pour le comptage.

        Args:
            state (numpy.array): État à discrétiser

        Returns:
            tuple: Représentation discrète de l'état
        """
        # Discrétiser l'état en le divisant en bins
        discretized = tuple(
            min(int(s * self.hash_bins), self.hash_bins - 1) for s in state
        )
        return discretized

    def _compute_novelty_bonus(self, state):
        """
        Calcule le bonus de nouveauté pour un état.

        Args:
            state (numpy.array): État actuel

        Returns:
            float: Bonus de nouveauté
        """
        state_hash = self._hash_state(state)
        count = min(self.state_counts[state_hash], self.max_count)

        # Le bonus diminue avec le nombre de visites
        # 1 / (count + 1) donne un grand bonus pour les états jamais vus (count=0)
        # et un petit bonus pour les états souvent vus
        novelty_bonus = self.novelty_scale / (count + 1.0)

        return novelty_bonus

    def select_action(self, q_values, state):
        """
        Sélectionne une action en utilisant l'exploration basée sur la nouveauté.

        Args:
            q_values (numpy.array): Valeurs Q estimées par le modèle
            state (numpy.array): État actuel

        Returns:
            int: Action sélectionnée
        """
        # Compter la visite de cet état
        state_hash = self._hash_state(state)
        self.state_counts[state_hash] += 1

        # Calculer le bonus de nouveauté
        novelty_bonus = self._compute_novelty_bonus(state)

        # Ajouter le bonus aux valeurs Q
        novelty_q_values = q_values + novelty_bonus

        # Sélectionner l'action avec la valeur Q + bonus la plus élevée
        action = np.argmax(novelty_q_values)

        logger.debug(
            f"Action {action} sélectionnée avec bonus de nouveauté {novelty_bonus}"
        )
        return action

    def decay_novelty(self):
        """
        Diminue progressivement l'importance de la nouveauté au fil du temps.
        """
        self.novelty_scale *= self.decay_rate
        logger.debug(f"Échelle de nouveauté réduite à {self.novelty_scale}")

    def reset(self):
        """
        Réinitialise l'explorateur basé sur la nouveauté.
        """
        self.state_counts.clear()
        logger.info("NoveltyExploration réinitialisé")


class HybridExploration:
    """
    Exploration hybride combinant UCB et exploration basée sur la nouveauté.
    """

    def __init__(
        self,
        state_size,
        action_size,
        ucb_c=2.0,
        novelty_scale=1.0,
        decay_rate=0.99,
        hash_bins=10,
        max_count=1000,
    ):
        """
        Initialise l'explorateur hybride.

        Args:
            state_size (int): Taille de l'espace d'état
            action_size (int): Taille de l'espace d'action
            ucb_c (float): Paramètre de confiance pour UCB
            novelty_scale (float): Échelle du bonus de nouveauté
            decay_rate (float): Taux de décroissance de la nouveauté
            hash_bins (int): Nombre de bins pour le hachage de l'état
            max_count (int): Valeur maximale du compteur d'états
        """
        self.ucb = UCBExploration(action_size, ucb_c)
        self.novelty = NoveltyExploration(
            state_size, action_size, novelty_scale, decay_rate, hash_bins, max_count
        )

        logger.info(
            f"HybridExploration initialisé: state_size={state_size}, "
            f"action_size={action_size}"
        )

    def select_action(self, q_values, state):
        """
        Sélectionne une action en combinant UCB et exploration basée sur la nouveauté.

        Args:
            q_values (numpy.array): Valeurs Q estimées par le modèle
            state (numpy.array): État actuel

        Returns:
            int: Action sélectionnée
        """
        # Obtenir les scores UCB
        ucb_action = self.ucb.select_action(q_values, state)

        # Obtenir les scores de nouveauté
        novelty_action = self.novelty.select_action(q_values, state)

        # Combiner les deux approches (ici, on utilise simplement UCB)
        # On pourrait implémenter une stratégie plus sophistiquée
        action = ucb_action

        logger.debug(
            f"Action {action} sélectionnée (UCB: {ucb_action}, Novelty: {novelty_action})"
        )
        return action

    def update(self, action, reward):
        """
        Met à jour les statistiques pour l'action sélectionnée.

        Args:
            action (int): Action sélectionnée
            reward (float): Récompense reçue
        """
        self.ucb.update(action, reward)

    def decay_novelty(self):
        """
        Diminue progressivement l'importance de la nouveauté.
        """
        self.novelty.decay_novelty()

    def reset(self):
        """
        Réinitialise l'explorateur hybride.
        """
        self.ucb.reset()
        self.novelty.reset()
        logger.info("HybridExploration réinitialisé")


class AdaptiveExploration:
    """
    Stratégie d'exploration adaptative pour l'agent RL.
    Combine epsilon-greedy avec UCB et s'adapte à la volatilité du marché.
    """

    def __init__(self, initial_epsilon=0.1, min_epsilon=0.01, decay=0.995):
        """
        Initialise l'explorateur adaptatif.

        Args:
            initial_epsilon (float): Valeur initiale d'epsilon pour epsilon-greedy
            min_epsilon (float): Valeur minimale d'epsilon
            decay (float): Taux de décroissance d'epsilon
        """
        self.initial_epsilon = initial_epsilon  # Stocker la valeur initiale
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.action_counts = {}  # Pour UCB
        self.total_steps = 0

        logger.info(
            f"AdaptiveExploration initialisé: epsilon={initial_epsilon}, "
            f"min_epsilon={min_epsilon}, decay={decay}"
        )

    def should_explore(self, state, market_volatility=None):
        """
        Détermine si l'agent doit explorer en fonction des conditions actuelles.

        Args:
            state: État actuel de l'environnement
            market_volatility: Mesure de la volatilité du marché (optionnel)

        Returns:
            bool: True si l'agent doit explorer, False sinon
        """
        self.total_steps += 1

        # Adapter epsilon selon la volatilité du marché
        if market_volatility is not None:
            # Augmenter l'exploration dans les marchés plus volatils
            adjusted_epsilon = self.epsilon * (1 + market_volatility)
        else:
            adjusted_epsilon = self.epsilon

        # Décider d'explorer ou d'exploiter
        should_explore = np.random.random() < adjusted_epsilon

        # Décrémenter epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        logger.debug(
            f"Exploration décision: {should_explore}, "
            f"epsilon={adjusted_epsilon:.4f}"
        )
        return should_explore

    def get_ucb_action(self, state_str, q_values, c=2.0):
        """
        Sélectionne une action selon la stratégie Upper Confidence Bound.

        Args:
            state_str: Représentation de l'état sous forme de chaîne
            q_values: Valeurs Q pour chaque action
            c: Paramètre d'exploration UCB

        Returns:
            int: Action sélectionnée
        """
        if state_str not in self.action_counts:
            self.action_counts[state_str] = np.zeros(len(q_values))

        # Calculer les scores UCB
        exploration_term = np.sqrt(
            np.log(self.total_steps + 1) / (self.action_counts[state_str] + 1e-6)
        )
        ucb_scores = q_values + c * exploration_term

        # Sélectionner l'action avec le score UCB le plus élevé
        action = np.argmax(ucb_scores)

        # Mettre à jour le compteur d'actions
        self.action_counts[state_str][action] += 1

        logger.debug(
            f"Action UCB {action} sélectionnée pour l'état {state_str}"
        )
        return action

    def reset(self):
        """
        Réinitialise l'explorateur adaptatif.
        """
        self.epsilon = self.initial_epsilon  # Utiliser la valeur initiale stockée
        self.action_counts.clear()
        self.total_steps = 0
        logger.info("AdaptiveExploration réinitialisé") 