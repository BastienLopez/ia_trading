"""
Module d'Experience Replay Prioritaire Amélioré (Enhanced Prioritized Experience Replay)

Ce module implémente une version optimisée du Prioritized Experience Replay
avec des mécanismes avancés pour réduire la taille du buffer tout en maintenant
ou améliorant la qualité de l'apprentissage.
"""

import logging
import numpy as np
import torch
from collections import namedtuple, deque
from typing import Dict, List, Optional, Tuple, Any, Union

# Configuration du logger
logger = logging.getLogger("EnhancedPrioritizedReplay")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Structure pour stocker les expériences
Experience = namedtuple("Experience", 
                       ["state", "action", "reward", "next_state", "done", "info"])


class SumTree:
    """
    Structure de données SumTree optimisée pour l'échantillonnage prioritaire.
    Implémente une somme binaire pour une recherche efficace basée sur la priorité.
    """
    
    def __init__(self, capacity):
        """
        Initialise la structure SumTree.
        
        Args:
            capacity: Capacité maximale du buffer
        """
        self.capacity = capacity  # Nombre de feuilles
        self.tree = np.zeros(2 * capacity - 1)  # Arbre complet (nœuds + feuilles)
        self.data = np.zeros(capacity, dtype=object)  # Buffer de données
        self.data_pointer = 0  # Position actuelle
        self.size = 0  # Nombre d'éléments actuellement stockés
        
    def add(self, priority, data):
        """
        Ajoute une nouvelle expérience avec sa priorité.
        
        Args:
            priority: Priorité de l'expérience
            data: L'expérience à stocker
        """
        # Calculer l'index dans l'arbre
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Stocker les données et mettre à jour la priorité
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        # Passer à la position suivante
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Mettre à jour la taille actuelle
        if self.size < self.capacity:
            self.size += 1
            
    def update(self, tree_idx, priority):
        """
        Met à jour la priorité d'une expérience.
        
        Args:
            tree_idx: Index dans l'arbre
            priority: Nouvelle priorité
        """
        # Calculer le changement de priorité
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propager le changement jusqu'à la racine
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            
    def get_leaf(self, v):
        """
        Récupère une feuille en fonction de la valeur.
        
        Args:
            v: Valeur entre 0 et sum(priorities)
            
        Returns:
            tuple: (tree_idx, priority, data)
        """
        parent_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            # Si on atteint une feuille, on s'arrête
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # Sinon, on descend à gauche ou à droite selon la valeur
            if v <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                v -= self.tree[left_idx]
                parent_idx = right_idx
                
        data_idx = leaf_idx - (self.capacity - 1)
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total_priority(self):
        """
        Retourne la somme totale des priorités.
        
        Returns:
            float: Somme des priorités
        """
        return self.tree[0]


class EnhancedPrioritizedReplay:
    """
    Version améliorée du tampon de replay prioritaire avec des optimisations pour:
    - Réduire la taille utile du buffer sans perdre en qualité
    - Identifier et retenir les expériences les plus importantes
    - Éliminer les expériences redondantes
    - Combiner des expériences similaires
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 0.01,
        n_step: int = 1,
        gamma: float = 0.99,
        redundancy_threshold: float = 0.95,
        importance_threshold: float = 0.5,
        importance_decay: float = 0.99,
        cluster_threshold: float = 0.1,
        state_encoder: Optional[Any] = None,
    ):
        """
        Initialise le tampon de replay prioritaire amélioré.
        
        Args:
            capacity: Capacité maximale du buffer
            alpha: Exposant de prioritisation (0 = uniforme, 1 = greedy)
            beta: Exposant pour la correction d'importance sampling
            beta_increment: Incrément de beta à chaque échantillonnage
            epsilon: Petite valeur pour éviter les priorités nulles
            n_step: Nombre d'étapes pour les retours
            gamma: Facteur d'actualisation
            redundancy_threshold: Seuil pour détecter les expériences redondantes
            importance_threshold: Seuil pour conserver les expériences importantes
            importance_decay: Facteur de décroissance de l'importance
            cluster_threshold: Seuil pour le clustering des expériences similaires
            state_encoder: Fonction optionnelle pour encoder les états (feature extraction)
        """
        self.sum_tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Paramètres d'optimisation
        self.redundancy_threshold = redundancy_threshold
        self.importance_threshold = importance_threshold
        self.importance_decay = importance_decay
        self.cluster_threshold = cluster_threshold
        
        # Encodeur d'état (pour extraire des features)
        self.state_encoder = state_encoder
        
        # Trackers pour l'analyse du buffer
        self.importance_scores = {}  # Historique d'importance des expériences
        self.state_clusters = {}     # Clusters d'états similaires
        self.episode_boundaries = [] # Marquer les débuts/fins d'épisodes
        self.duplicate_count = 0     # Compteur d'expériences redondantes
        
        # Métriques
        self.metrics = {
            "redundant_experiences": 0,
            "important_experiences": 0,
            "clustered_experiences": 0,
            "buffer_reduction": 0.0,
            "sample_efficiency": 0.0
        }
        
        logger.info(f"Tampon de replay prioritaire amélioré initialisé avec capacité={capacity}")
        logger.info(f"Paramètres d'optimisation: redundancy={redundancy_threshold}, "
                   f"importance={importance_threshold}, clusters={cluster_threshold}")
    
    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """
        Prétraite un état en utilisant l'encodeur si disponible.
        
        Args:
            state: État à prétraiter
            
        Returns:
            np.ndarray: État prétraité
        """
        if self.state_encoder is not None:
            # Convertir en tensor pour l'encodeur puis reconvertir en numpy
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state_t = torch.FloatTensor(state)
                else:
                    state_t = state
                
                # Ajouter une dimension batch si nécessaire
                if state_t.dim() == 1:
                    state_t = state_t.unsqueeze(0)
                
                encoded = self.state_encoder(state_t)
                
                if isinstance(encoded, torch.Tensor):
                    return encoded.cpu().numpy()
                return encoded
        return state
    
    def _compute_state_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calcule la similarité entre deux états.
        
        Args:
            state1: Premier état
            state2: Deuxième état
            
        Returns:
            float: Score de similarité entre 0 et 1
        """
        # Prétraiter les états
        state1_proc = self._preprocess_state(state1)
        state2_proc = self._preprocess_state(state2)
        
        # Calcul de similarité cosinus
        dot_product = np.dot(state1_proc.flatten(), state2_proc.flatten())
        norm1 = np.linalg.norm(state1_proc.flatten())
        norm2 = np.linalg.norm(state2_proc.flatten())
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, (similarity + 1) / 2))  # Normaliser entre 0 et 1
    
    def _is_redundant(self, state: np.ndarray, action: np.ndarray) -> bool:
        """
        Vérifie si une expérience est redondante (très similaire à une existante).
        
        Args:
            state: État à vérifier
            action: Action à vérifier
            
        Returns:
            bool: True si l'expérience est redondante
        """
        if self.sum_tree.size < 10:  # Pas assez d'expériences pour comparer
            return False
        
        # Échantillonner quelques expériences pour comparaison
        sample_size = min(10, self.sum_tree.size)
        indices = np.random.choice(self.sum_tree.size, sample_size, replace=False)
        
        for i in indices:
            exp = self.sum_tree.data[i]
            if exp is None:
                continue
                
            # Vérifier similarité d'état
            state_similarity = self._compute_state_similarity(state, exp.state)
            
            # Vérifier identité d'action (pour actions discrètes ou continues)
            if isinstance(action, int) and isinstance(exp.action, int):
                action_match = (action == exp.action)
            else:
                action_match = np.array_equal(action, exp.action)
            
            # Si très similaire, considérer comme redondant
            if state_similarity > self.redundancy_threshold and action_match:
                self.duplicate_count += 1
                self.metrics["redundant_experiences"] += 1
                return True
                
        return False
    
    def _update_importance_score(self, tree_idx: int, td_error: float):
        """
        Met à jour le score d'importance d'une expérience.
        
        Args:
            tree_idx: Index dans l'arbre
            td_error: Erreur TD
        """
        data_idx = tree_idx - (self.sum_tree.capacity - 1)
        
        # Initialiser le score s'il n'existe pas
        if data_idx not in self.importance_scores:
            self.importance_scores[data_idx] = abs(td_error)
        else:
            # Mettre à jour avec décroissance
            self.importance_scores[data_idx] = (
                self.importance_decay * self.importance_scores[data_idx] +
                (1 - self.importance_decay) * abs(td_error)
            )
    
    def _n_step_processing(
        self, state, action, reward, next_state, done, info=None
    ) -> tuple:
        """
        Traite une transition pour retours n-step.
        
        Args:
            state: État actuel
            action: Action choisie
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
            info: Informations supplémentaires
            
        Returns:
            tuple: (processé?, état, action, récompense, état suivant, done, info)
        """
        if self.n_step == 1:
            return True, state, action, reward, next_state, done, info
            
        # Ajouter à la file
        self.n_step_buffer.append((state, action, reward, next_state, done, info))
        
        # Pas assez d'éléments
        if len(self.n_step_buffer) < self.n_step:
            return False, None, None, None, None, None, None
            
        # Récupérer l'état et l'action initiaux
        init_state, init_action, _, _, _, init_info = self.n_step_buffer[0]
        
        # Calculer la récompense multi-étapes
        n_step_reward = 0
        for i, (_, _, r, _, terminal, _) in enumerate(self.n_step_buffer):
            n_step_reward += r * (self.gamma ** i)
            if terminal:
                break
                
        # État final et indicateur de fin
        final_next_state = self.n_step_buffer[-1][3]
        final_done = self.n_step_buffer[-1][4]
        final_info = self.n_step_buffer[-1][5]
        
        return (
            True, init_state, init_action, n_step_reward, 
            final_next_state, final_done, 
            init_info if init_info is not None else final_info
        )
    
    def add(
        self, state, action, reward, next_state, done, info=None, td_error=None
    ):
        """
        Ajoute une expérience au buffer.
        
        Args:
            state: État actuel
            action: Action choisie
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
            info: Informations supplémentaires
            td_error: Erreur TD initiale (si connue)
        """
        # Marquer les limites d'épisodes
        if done:
            self.episode_boundaries.append(self.sum_tree.data_pointer)
            
        # Traitement n-step
        processed, state, action, reward, next_state, done, info = self._n_step_processing(
            state, action, reward, next_state, done, info
        )
        
        if not processed:
            return
            
        # Vérifier si l'expérience est redondante
        if self._is_redundant(state, action):
            return
            
        # Créer l'objet Experience
        experience = Experience(state, action, reward, next_state, done, info)
        
        # Déterminer la priorité (max priorité par défaut)
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = self.max_priority
            
        # Ajouter au buffer
        self.sum_tree.add(priority, experience)
    
    def sample(self, batch_size, beta=None):
        """
        Échantillonne un batch d'expériences prioritaires.
        
        Args:
            batch_size: Taille du batch
            beta: Exposant pour la correction d'importance sampling
            
        Returns:
            tuple: (batch_indices, experiences, is_weights)
        """
        beta = beta if beta is not None else self.beta
        
        # Incrémenter beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Échantillonner un batch
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        batch = []
        is_weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculer le segment de priorité
        priority_segment = self.sum_tree.total_priority() / batch_size
        
        # Déterminer le poids minimum pour la normalisation
        min_priority = np.min(self.sum_tree.tree[-self.sum_tree.capacity:][
            np.nonzero(self.sum_tree.tree[-self.sum_tree.capacity:])
        ]) / self.sum_tree.total_priority()
        
        if min_priority == 0:
            min_priority = self.epsilon
        
        # Échantillonner uniformément dans chaque segment
        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # Récupérer l'expérience
            idx, priority, data = self.sum_tree.get_leaf(value)
            
            # Calculer le poids d'importance sampling
            sampling_prob = priority / self.sum_tree.total_priority()
            is_weights[i] = (sampling_prob / min_priority) ** (-beta)
            
            batch_indices[i] = idx
            batch.append(data)
        
        # Normaliser les poids
        is_weights /= np.max(is_weights)
        
        return batch_indices, batch, is_weights
    
    def update_priorities(self, indices, td_errors):
        """
        Met à jour les priorités des expériences.
        
        Args:
            indices: Indices des expériences
            td_errors: Erreurs TD correspondantes
        """
        for idx, td_error in zip(indices, td_errors):
            # Mettre à jour l'importance
            self._update_importance_score(idx, td_error)
            
            # Mettre à jour la priorité
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.sum_tree.update(idx, priority)
    
    def optimize_buffer(self):
        """
        Optimise le buffer pour éliminer les expériences redondantes
        et favoriser les expériences importantes.
        """
        if self.sum_tree.size < 100:  # Pas assez d'expériences
            return
            
        logger.info("Optimisation du buffer...")
        
        # 1. Identifier les expériences importantes à conserver
        important_indices = []
        for idx, score in self.importance_scores.items():
            if score > self.importance_threshold:
                tree_idx = idx + (self.sum_tree.capacity - 1)
                important_indices.append(tree_idx)
                self.metrics["important_experiences"] += 1
                
        # 2. Identifier les expériences uniques (non redondantes)
        unique_experiences = set()
        redundant_indices = []
        
        # Parcourir toutes les expériences
        for i in range(self.sum_tree.size):
            exp = self.sum_tree.data[i]
            if exp is None:
                continue
                
            # Encoder l'état
            state_encoded = self._preprocess_state(exp.state).tobytes()
            
            # Vérifier si déjà vue
            if state_encoded in unique_experiences:
                tree_idx = i + (self.sum_tree.capacity - 1)
                redundant_indices.append(tree_idx)
            else:
                unique_experiences.add(state_encoded)
        
        # 3. Mettre à jour les priorités
        # Augmenter la priorité des expériences importantes
        for idx in important_indices:
            current_priority = self.sum_tree.tree[idx]
            self.sum_tree.update(idx, current_priority * 1.5)
            
        # Réduire la priorité des expériences redondantes
        for idx in redundant_indices:
            current_priority = self.sum_tree.tree[idx]
            self.sum_tree.update(idx, current_priority * 0.5)
            
        # Mise à jour des métriques
        self.metrics["buffer_reduction"] = len(redundant_indices) / max(1, self.sum_tree.size)
        
        logger.info(f"Buffer optimisé: {len(important_indices)} expériences importantes, "
                   f"{len(redundant_indices)} expériences redondantes")
    
    def get_metrics(self):
        """
        Retourne les métriques du buffer.
        
        Returns:
            dict: Métriques du buffer
        """
        # Calculer l'efficacité d'échantillonnage
        if hasattr(self, 'sample_count') and self.sample_count > 0:
            self.metrics["sample_efficiency"] = self.metrics["important_experiences"] / max(1, self.sample_count)
            
        return self.metrics
    
    def __len__(self):
        """
        Retourne la taille actuelle du buffer.
        
        Returns:
            int: Nombre d'expériences stockées
        """
        return self.sum_tree.size
    
    def clear(self):
        """
        Vide le buffer.
        """
        self.sum_tree = SumTree(self.sum_tree.capacity)
        self.max_priority = 1.0
        self.importance_scores = {}
        self.state_clusters = {}
        self.episode_boundaries = []
        self.duplicate_count = 0
        self.n_step_buffer.clear()


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un buffer
    buffer = EnhancedPrioritizedReplay(
        capacity=10000,
        alpha=0.6,
        beta=0.4,
        n_step=3,
        redundancy_threshold=0.95
    )
    
    # Générer des données aléatoires
    for i in range(1000):
        state = np.random.randn(4).astype(np.float32)
        action = np.array([np.random.randint(0, 2)]).astype(np.float32)
        reward = np.random.rand().astype(np.float32)
        next_state = np.random.randn(4).astype(np.float32)
        done = bool(np.random.rand() > 0.9)
        
        buffer.add(state, action, reward, next_state, done)
        
    # Échantillonner des données
    indices, batch, weights = buffer.sample(64)
    
    # Mise à jour des priorités (simulation d'erreurs TD)
    td_errors = np.random.randn(64)
    buffer.update_priorities(indices, td_errors)
    
    # Optimiser le buffer
    buffer.optimize_buffer()
    
    # Afficher les métriques
    metrics = buffer.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}") 