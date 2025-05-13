"""
Apprentissage inverse par renforcement (Inverse Reinforcement Learning) pour l'extraction de fonctions de récompense.

Ce module implémente des algorithmes d'apprentissage inverse par renforcement qui permettent
d'extraire des fonctions de récompense à partir de démonstrations d'experts en trading.
Les algorithmes disponibles sont:
- Maximum Entropy IRL: Extrait une fonction de récompense qui maximise l'entropie des trajectoires
- Apprenticed Learning: Apprend directement une politique imitant un expert

Ces techniques sont utiles pour:
1. Inférer les objectifs sous-jacents des traders experts
2. Développer des agents qui imitent les stratégies de trading humaines
3. Combiner l'expertise humaine avec l'optimisation par RL
"""

import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RewardNetwork(nn.Module):
    """
    Réseau de neurones pour modéliser une fonction de récompense.

    Cette architecture prend en entrée un état et une action et prédit la récompense associée.
    Elle est utilisée par les algorithmes d'IRL pour apprendre les fonctions de récompense
    sous-jacentes aux comportements d'experts.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialise le réseau de récompense.

        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Dimension de l'espace d'action
            hidden_dims: Dimensions des couches cachées
        """
        super(RewardNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Construire les couches du réseau
        layers = []
        input_dim = state_dim + action_dim  # Concaténation de l'état et de l'action

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Couche de sortie (une seule valeur de récompense)
        layers.append(nn.Linear(input_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calcule la récompense pour un état et une action donnés.

        Args:
            state: Tensor d'état [batch_size, state_dim]
            action: Tensor d'action [batch_size, action_dim]

        Returns:
            Récompense prédite [batch_size, 1]
        """
        # Préparation des actions (one-hot si actions discrètes)
        if action.dim() == 1:
            if isinstance(action, torch.LongTensor) or isinstance(
                action, torch.cuda.LongTensor
            ):
                action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
            else:
                action_one_hot = F.one_hot(
                    action.long(), num_classes=self.action_dim
                ).float()
        else:
            action_one_hot = action

        # Concaténer l'état et l'action
        x = torch.cat([state, action_one_hot], dim=1)

        # Calculer la récompense
        reward = self.model(x)

        return reward


class MaximumEntropyIRL:
    """
    Apprentissage inverse par renforcement par maximum d'entropie.

    Cet algorithme apprend une fonction de récompense qui maximise l'entropie
    des trajectoires tout en faisant correspondre les caractéristiques des
    trajectoires d'experts.

    Référence:
    Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008).
    Maximum entropy inverse reinforcement learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        regularization: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise l'algorithme MaxEnt IRL.

        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Dimension de l'espace d'action
            gamma: Facteur d'actualisation
            learning_rate: Taux d'apprentissage
            regularization: Coefficient de régularisation
            device: Appareil sur lequel exécuter les calculs ("cuda" ou "cpu")
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.device = device

        # Réseau de récompense
        self.reward_network = RewardNetwork(state_dim, action_dim).to(device)

        # Optimiseur
        self.optimizer = torch.optim.Adam(
            self.reward_network.parameters(), lr=learning_rate
        )

        logger.info(
            f"MaxEnt IRL initialisé avec state_dim={state_dim}, action_dim={action_dim}"
        )

    def load_expert_demonstrations(self, demos: List[List[Tuple]]) -> None:
        """
        Charge des démonstrations d'experts.

        Args:
            demos: Liste de trajectoires, chaque trajectoire étant une liste de tuples (état, action)
        """
        self.expert_demos = demos

        # Convertir en tensors
        self.expert_states = []
        self.expert_actions = []

        for demo in demos:
            for state, action in demo:
                self.expert_states.append(state)
                self.expert_actions.append(action)

        self.expert_states = torch.FloatTensor(self.expert_states).to(self.device)
        self.expert_actions = torch.LongTensor(self.expert_actions).to(self.device)

        logger.info(f"Chargé {len(demos)} démonstrations d'experts")

    def compute_state_visitation_freq(
        self, trajectories: List[List[Tuple]]
    ) -> torch.Tensor:
        """
        Calcule la fréquence de visite des états à partir des trajectoires.

        Args:
            trajectories: Liste de trajectoires, chaque trajectoire étant une liste de tuples (état, action)

        Returns:
            Fréquence de visite des états
        """
        # Initialiser la fréquence de visite
        state_visitation = {}

        # Compter les occurrences
        for trajectory in trajectories:
            for state, _ in trajectory:
                state_tuple = tuple(state)
                if state_tuple in state_visitation:
                    state_visitation[state_tuple] += 1
                else:
                    state_visitation[state_tuple] = 1

        # Normaliser
        total_visits = sum(state_visitation.values())
        for state in state_visitation:
            state_visitation[state] /= total_visits

        return state_visitation

    def predict_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Prédit la récompense pour un état et une action donnés.

        Args:
            state: Tensor d'état
            action: Tensor d'action

        Returns:
            Récompense prédite
        """
        return self.reward_network(state, action)

    def train(
        self,
        policy_optimizer: Callable,
        env: Any,
        epochs: int = 100,
        policy_update_steps: int = 100,
        batch_size: int = 64,
    ) -> Dict[str, List[float]]:
        """
        Entraîne l'algorithme MaxEnt IRL.

        Args:
            policy_optimizer: Fonction qui optimise une politique étant donné une fonction de récompense
            env: Environnement pour l'optimisation de la politique
            epochs: Nombre d'époques d'entraînement
            policy_update_steps: Nombre d'étapes d'optimisation de la politique à chaque époque
            batch_size: Taille du batch pour l'entraînement

        Returns:
            Historique d'entraînement
        """
        history = {"loss": [], "expert_reward": [], "policy_reward": []}

        for epoch in range(epochs):
            logger.info(f"Époque {epoch+1}/{epochs}")

            # Étape 1: Optimiser la politique actuelle avec la fonction de récompense apprise
            policy = policy_optimizer(env, self.reward_network, policy_update_steps)

            # Étape 2: Collecter des trajectoires avec la politique optimisée
            policy_trajectories = self._collect_trajectories(env, policy, 20)

            # Étape 3: Calculer les fréquences de visite pour les trajectoires de l'expert et de la politique
            expert_freq = self.compute_state_visitation_freq(self.expert_demos)
            policy_freq = self.compute_state_visitation_freq(policy_trajectories)

            # Étape 4: Mettre à jour la fonction de récompense
            for _ in range(10):  # Mini-batch updates
                # Échantillonner un batch d'exemples d'experts
                indices = np.random.choice(
                    len(self.expert_states),
                    min(batch_size, len(self.expert_states)),
                    replace=False,
                )
                expert_batch_states = self.expert_states[indices]
                expert_batch_actions = self.expert_actions[indices]

                # Calculer la récompense pour les exemples d'experts
                expert_rewards = self.reward_network(
                    expert_batch_states, expert_batch_actions
                )

                # Échantillonner un batch d'exemples de la politique
                policy_batch_states = []
                policy_batch_actions = []
                for trajectory in policy_trajectories:
                    for state, action in trajectory:
                        policy_batch_states.append(state)
                        policy_batch_actions.append(action)

                if len(policy_batch_states) > 0:
                    # Convertir en tensors
                    policy_batch_states = torch.FloatTensor(policy_batch_states).to(
                        self.device
                    )
                    policy_batch_actions = torch.LongTensor(policy_batch_actions).to(
                        self.device
                    )

                    # Sous-échantillonnage si nécessaire
                    if len(policy_batch_states) > batch_size:
                        indices = np.random.choice(
                            len(policy_batch_states), batch_size, replace=False
                        )
                        policy_batch_states = policy_batch_states[indices]
                        policy_batch_actions = policy_batch_actions[indices]

                    # Calculer la récompense pour les exemples de la politique
                    policy_rewards = self.reward_network(
                        policy_batch_states, policy_batch_actions
                    )

                    # Calculer la perte (maximiser la récompense des experts, minimiser celle de la politique)
                    loss = -torch.mean(expert_rewards) + torch.mean(policy_rewards)

                    # Ajouter la régularisation L2
                    l2_reg = 0
                    for param in self.reward_network.parameters():
                        l2_reg += torch.norm(param)
                    loss += self.regularization * l2_reg

                    # Optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Enregistrer les métriques
                    history["loss"].append(loss.item())
                    history["expert_reward"].append(torch.mean(expert_rewards).item())
                    history["policy_reward"].append(torch.mean(policy_rewards).item())

                    if _ == 0:  # Uniquement pour la première mini-batch
                        logger.info(f"  Loss: {loss.item():.4f}")
                        logger.info(
                            f"  Expert reward: {torch.mean(expert_rewards).item():.4f}"
                        )
                        logger.info(
                            f"  Policy reward: {torch.mean(policy_rewards).item():.4f}"
                        )

        return history

    def _collect_trajectories(
        self, env: Any, policy: Any, num_trajectories: int
    ) -> List[List[Tuple]]:
        """
        Collecte des trajectoires en utilisant une politique donnée.

        Args:
            env: Environnement
            policy: Politique à utiliser
            num_trajectories: Nombre de trajectoires à collecter

        Returns:
            Liste de trajectoires
        """
        trajectories = []

        for _ in range(num_trajectories):
            trajectory = []
            state = env.reset()
            done = False

            while not done:
                # Sélectionner une action avec la politique
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = policy(state_tensor).argmax().item()

                # Exécuter l'action
                next_state, _, done, _ = env.step(action)

                # Enregistrer l'état et l'action
                trajectory.append((state, action))

                # Passer à l'état suivant
                state = next_state

            trajectories.append(trajectory)

        return trajectories

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle.

        Args:
            path: Chemin de sauvegarde
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "reward_network": self.reward_network.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "regularization": self.regularization,
            },
            path,
        )
        logger.info(f"Modèle sauvegardé dans {path}")

    def load(self, path: str) -> None:
        """
        Charge le modèle.

        Args:
            path: Chemin du modèle à charger
        """
        checkpoint = torch.load(path)
        self.state_dim = checkpoint["state_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.gamma = checkpoint["gamma"]
        self.regularization = checkpoint["regularization"]

        self.reward_network = RewardNetwork(self.state_dim, self.action_dim).to(
            self.device
        )
        self.reward_network.load_state_dict(checkpoint["reward_network"])
        logger.info(f"Modèle chargé depuis {path}")


class ApprenticeshipLearning:
    """
    Apprenticeship Learning (Apprentissage par apprentissage) pour l'imitation d'experts.

    Cette approche apprend directement une politique qui imite les démonstrations d'experts
    en minimisant la différence entre les caractéristiques des trajectoires générées et
    celles des experts.

    Référence:
    Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_extractor: Callable = None,
        feature_dim: int = None,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise l'algorithme d'Apprenticeship Learning.

        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Dimension de l'espace d'action
            feature_extractor: Fonction qui extrait des caractéristiques des états
            feature_dim: Dimension des caractéristiques extraites
            gamma: Facteur d'actualisation
            device: Appareil sur lequel exécuter les calculs ("cuda" ou "cpu")
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_extractor = feature_extractor
        self.feature_dim = (
            feature_dim or state_dim
        )  # Par défaut, utilise l'état comme caractéristique
        self.gamma = gamma
        self.device = device

        # Poids de la fonction de récompense (pour les caractéristiques)
        self.reward_weights = torch.zeros(
            self.feature_dim, device=device, requires_grad=True
        )

        logger.info(
            f"Apprenticeship Learning initialisé avec state_dim={state_dim}, action_dim={action_dim}"
        )

    def extract_features(self, state: np.ndarray) -> np.ndarray:
        """
        Extrait des caractéristiques d'un état.

        Args:
            state: État à partir duquel extraire des caractéristiques

        Returns:
            Caractéristiques extraites
        """
        if self.feature_extractor is not None:
            return self.feature_extractor(state)
        else:
            return state  # Par défaut, utilise l'état comme caractéristique

    def compute_expected_feature_counts(
        self, trajectories: List[List[Tuple]]
    ) -> np.ndarray:
        """
        Calcule les comptes moyens des caractéristiques à partir des trajectories.

        Args:
            trajectories: Liste de trajectoires

        Returns:
            Vecteur de comptes de caractéristiques moyens
        """
        feature_counts = np.zeros(self.feature_dim)
        num_trajectories = len(trajectories)

        for trajectory in trajectories:
            discounted_sum = np.zeros(self.feature_dim)
            for t, (state, _) in enumerate(trajectory):
                features = self.extract_features(state)
                discounted_sum += (self.gamma**t) * features

            feature_counts += discounted_sum / num_trajectories

        return feature_counts

    def load_expert_demonstrations(self, demos: List[List[Tuple]]) -> None:
        """
        Charge des démonstrations d'experts.

        Args:
            demos: Liste de trajectoires d'experts
        """
        self.expert_demos = demos

        # Calculer les comptes de caractéristiques de l'expert
        self.expert_feature_counts = self.compute_expected_feature_counts(demos)

        logger.info(f"Chargé {len(demos)} démonstrations d'experts")

    def compute_reward(self, state: np.ndarray) -> float:
        """
        Calcule la récompense pour un état donné.

        Args:
            state: État pour lequel calculer la récompense

        Returns:
            Récompense calculée
        """
        features = self.extract_features(state)
        features_tensor = torch.FloatTensor(features).to(self.device)
        reward = torch.dot(features_tensor, self.reward_weights)
        return reward.item()

    def train(
        self,
        policy_optimizer: Callable,
        env: Any,
        max_iterations: int = 20,
        policy_update_steps: int = 100,
        epsilon: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Entraîne l'algorithme d'Apprenticeship Learning.

        Args:
            policy_optimizer: Fonction qui optimise une politique étant donné une fonction de récompense
            env: Environnement pour l'optimisation de la politique
            max_iterations: Nombre maximal d'itérations
            policy_update_steps: Nombre d'étapes d'optimisation de la politique à chaque itération
            epsilon: Seuil de convergence

        Returns:
            Historique d'entraînement
        """
        history = {"feature_difference": []}

        # Calculer les comptes de caractéristiques de l'expert
        expert_features = torch.FloatTensor(self.expert_feature_counts).to(self.device)

        # Initialiser la liste des comptes de caractéristiques
        all_feature_counts = []

        for iteration in range(max_iterations):
            logger.info(f"Itération {iteration+1}/{max_iterations}")

            # Étape 1: Optimiser la politique avec la fonction de récompense actuelle
            def reward_function(state):
                return self.compute_reward(state)

            policy = policy_optimizer(env, reward_function, policy_update_steps)

            # Étape 2: Collecter des trajectoires avec la politique optimisée
            policy_trajectories = self._collect_trajectories(env, policy, 20)

            # Étape 3: Calculer les comptes de caractéristiques pour la politique
            policy_feature_counts = self.compute_expected_feature_counts(
                policy_trajectories
            )
            all_feature_counts.append(policy_feature_counts)

            # Convertir en tensor
            policy_features = torch.FloatTensor(policy_feature_counts).to(self.device)

            # Étape 4: Vérifier la convergence
            feature_difference = torch.norm(expert_features - policy_features).item()
            history["feature_difference"].append(feature_difference)

            logger.info(f"  Différence de caractéristiques: {feature_difference:.4f}")

            if feature_difference < epsilon:
                logger.info(f"Convergence atteinte à l'itération {iteration+1}")
                break

            # Étape 5: Mettre à jour les poids de la fonction de récompense
            # Résoudre un problème d'optimisation quadratique pour trouver les meilleurs poids
            # Ici, nous utilisons une approximation plus simple
            all_policy_features = torch.stack(
                [torch.FloatTensor(fc).to(self.device) for fc in all_feature_counts]
            )

            # Trouver les poids qui maximisent la différence entre l'expert et la pire politique
            best_weights = None
            best_margin = -float("inf")

            # Essayer différentes directions aléatoires
            for _ in range(100):
                # Générer des poids aléatoires unitaires
                w = torch.randn(self.feature_dim, device=self.device)
                w = w / torch.norm(w)

                # Calculer la marge minimale
                margins = torch.matmul(expert_features - all_policy_features, w)
                min_margin = torch.min(margins).item()

                if min_margin > best_margin:
                    best_margin = min_margin
                    best_weights = w.clone()

            # Mettre à jour les poids
            self.reward_weights = best_weights

        return history

    def _collect_trajectories(
        self, env: Any, policy: Any, num_trajectories: int
    ) -> List[List[Tuple]]:
        """
        Collecte des trajectoires en utilisant une politique donnée.

        Args:
            env: Environnement
            policy: Politique à utiliser
            num_trajectories: Nombre de trajectoires à collecter

        Returns:
            Liste de trajectoires
        """
        trajectories = []

        for _ in range(num_trajectories):
            trajectory = []
            state = env.reset()
            done = False

            while not done:
                # Sélectionner une action avec la politique
                action = policy(state)

                # Exécuter l'action
                next_state, _, done, _ = env.step(action)

                # Enregistrer l'état et l'action
                trajectory.append((state, action))

                # Passer à l'état suivant
                state = next_state

            trajectories.append(trajectory)

        return trajectories

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle.

        Args:
            path: Chemin de sauvegarde
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "reward_weights": self.reward_weights.cpu().detach().numpy(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "feature_dim": self.feature_dim,
            "gamma": self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Modèle sauvegardé dans {path}")

    def load(self, path: str) -> None:
        """
        Charge le modèle.

        Args:
            path: Chemin du modèle à charger
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.state_dim = data["state_dim"]
        self.action_dim = data["action_dim"]
        self.feature_dim = data["feature_dim"]
        self.gamma = data["gamma"]

        self.reward_weights = torch.FloatTensor(data["reward_weights"]).to(self.device)
        logger.info(f"Modèle chargé depuis {path}")
