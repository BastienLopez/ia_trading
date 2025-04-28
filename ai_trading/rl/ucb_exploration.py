import logging
import numpy as np
import math
from collections import defaultdict

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
        self.action_values = np.zeros(action_size, dtype=np.float32)
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
            exploration = self.c * math.sqrt(math.log(self.total_steps) / self.action_counts[a])
            
            # Score UCB final
            ucb_values[a] = exploitation + exploration
        
        # Sélectionner l'action avec le score UCB le plus élevé
        action = np.argmax(ucb_values)
        
        logger.debug(f"Action {action} sélectionnée avec UCB score {ucb_values[action]}")
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
        
        logger.debug(f"Action {action} mise à jour: count={n}, value={self.action_values[action]}")
    
    def reset(self):
        """
        Réinitialise l'explorateur UCB.
        """
        self.action_values = np.zeros(self.action_size, dtype=np.float32)
        self.action_counts = np.zeros(self.action_size, dtype=np.int32)
        self.total_steps = 0
        logger.info("UCBExploration réinitialisé")


class NoveltyExploration:
    """
    Exploration basée sur la nouveauté (Novelty-based exploration).
    
    Cette approche encourage l'exploration des états qui sont "nouveaux" ou rarement visités
    en attribuant un bonus de nouveauté aux états moins fréquemment visités.
    """
    
    def __init__(self, state_size, action_size, novelty_scale=1.0, decay_rate=0.99, 
                 hash_bins=10, max_count=1000):
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
        
        logger.info(f"NoveltyExploration initialisé: state_size={state_size}, "
                   f"action_size={action_size}, novelty_scale={novelty_scale}")
    
    def _hash_state(self, state):
        """
        Convertit un état continu en une représentation discrète pour le comptage.
        
        Args:
            state (numpy.array): État à discrétiser
            
        Returns:
            tuple: Représentation discrète de l'état
        """
        # Discrétiser l'état en le divisant en bins
        discretized = tuple(min(int(s * self.hash_bins), self.hash_bins - 1) 
                           for s in state)
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
        
        logger.debug(f"Action {action} sélectionnée avec bonus de nouveauté {novelty_bonus}")
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
    
    Cette approche prend en compte à la fois:
    - L'équilibre exploitation/exploration de UCB
    - L'encouragement à explorer des états nouveaux
    """
    
    def __init__(self, state_size, action_size, ucb_c=2.0, novelty_scale=1.0, 
                decay_rate=0.99, hash_bins=10, max_count=1000):
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
        # Initialiser les deux explorateurs
        self.ucb = UCBExploration(action_size, c=ucb_c)
        self.novelty = NoveltyExploration(state_size, action_size, 
                                         novelty_scale=novelty_scale,
                                         decay_rate=decay_rate,
                                         hash_bins=hash_bins,
                                         max_count=max_count)
        
        logger.info(f"HybridExploration initialisé: UCB(c={ucb_c}) + "
                   f"Novelty(scale={novelty_scale})")
    
    def select_action(self, q_values, state):
        """
        Sélectionne une action en utilisant les deux stratégies d'exploration.
        
        Args:
            q_values (numpy.array): Valeurs Q estimées par le modèle
            state (numpy.array): État actuel
            
        Returns:
            int: Action sélectionnée
        """
        # Obtenir les valeurs UCB (remplacer les q_values originaux)
        # Ceci implique que la méthode UCB.select_action() n'est pas appelée directement
        
        # Compter la visite de cet état pour la nouveauté
        state_hash = self.novelty._hash_state(state)
        self.novelty.state_counts[state_hash] += 1
        
        # Calculer le bonus de nouveauté
        novelty_bonus = self.novelty._compute_novelty_bonus(state)
        
        # Mise à jour du compteur total pour UCB
        self.ucb.total_steps += 1
        
        # Calculer les scores UCB pour chaque action
        ucb_values = np.zeros(self.ucb.action_size)
        
        # Si une action n'a jamais été choisie, lui donner une forte préférence
        if np.any(self.ucb.action_counts == 0):
            untried_actions = np.where(self.ucb.action_counts == 0)[0]
            action = np.random.choice(untried_actions)
            logger.debug(f"Action {action} sélectionnée (jamais essayée)")
            return action
        
        # Calculer le score UCB pour chaque action avec bonus de nouveauté
        for a in range(self.ucb.action_size):
            # Q-value (exploitation)
            exploitation = q_values[a]
            
            # Terme UCB (exploration)
            exploration = self.ucb.c * math.sqrt(
                math.log(self.ucb.total_steps) / self.ucb.action_counts[a])
            
            # Score UCB + bonus de nouveauté
            ucb_values[a] = exploitation + exploration + novelty_bonus
        
        # Sélectionner l'action avec le score combiné le plus élevé
        action = np.argmax(ucb_values)
        
        logger.debug(f"Action {action} sélectionnée avec score hybride {ucb_values[action]}")
        return action
    
    def update(self, action, reward):
        """
        Met à jour les statistiques pour l'action sélectionnée.
        
        Args:
            action (int): Action sélectionnée
            reward (float): Récompense reçue
        """
        # Mettre à jour les statistiques UCB
        self.ucb.update(action, reward)
    
    def decay_novelty(self):
        """
        Diminue progressivement l'importance de la nouveauté.
        """
        self.novelty.decay_novelty()
    
    def reset(self):
        """
        Réinitialise les deux explorateurs.
        """
        self.ucb.reset()
        self.novelty.reset()
        logger.info("HybridExploration réinitialisé") 