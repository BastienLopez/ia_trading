import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Configuration du logger
logger = logging.getLogger(__name__)


class PPOActorCritic(nn.Module):
    """
    Réseau ActorCritic pour PPO avec architecture partagée.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Initialise le réseau ActorCritic.

        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Dimension de l'espace d'action
            hidden_size: Taille des couches cachées
            log_std_min: Valeur minimum pour le log de l'écart-type
            log_std_max: Valeur maximum pour le log de l'écart-type
            action_bounds: Limites de l'espace d'action (min, max)
        """
        super(PPOActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_low, self.action_high = action_bounds

        # Couches partagées
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Couches de la politique (acteur)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

        # Couches de valeur (critique)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """
        Propage l'état à travers le réseau.

        Args:
            state: État d'entrée

        Returns:
            Tuple (moyenne, log_std, valeur)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.mean.weight.device)

        features = self.shared_network(state)

        # Calcul de la moyenne et log_std pour la politique
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Calcul de la valeur
        value = self.value(features)

        return mean, log_std, value

    def get_action_and_log_prob(self, state, deterministic=False):
        """
        Obtient une action et sa log-probabilité à partir d'un état.

        Args:
            state: État d'entrée
            deterministic: Si True, retourne l'action moyenne (déterministe)

        Returns:
            Tuple (action, log_prob)
        """
        mean, log_std, _ = self.forward(state)

        if deterministic:
            # Action déterministe (moyenne)
            action = torch.tanh(mean)
            return action, None

        # Échantillonnage stochastique
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparametrization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Calcul du log_prob avec correction pour tanh
        log_prob = normal.log_prob(x_t)

        # Correction pour la transformation tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_value(self, state):
        """Obtient la valeur d'un état."""
        _, _, value = self.forward(state)
        return value

    def evaluate_actions(self, states, actions):
        """
        Évalue les actions données par rapport aux états.

        Args:
            states: Batch d'états
            actions: Batch d'actions à évaluer

        Returns:
            Tuple (log_probs, values, entropy)
        """
        mean, log_std, values = self.forward(states)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Transformation inverse de tanh pour obtenir les actions avant transformation
        # clipping pour éviter les problèmes numériques
        actions_tanh = torch.clamp(actions, -0.999, 0.999)
        x_t = torch.atanh(actions_tanh)

        # Calcul des log_probs
        log_probs = normal.log_prob(x_t)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        # Calcul de l'entropie
        entropy = normal.entropy().sum(dim=-1, keepdim=True)

        return log_probs, values, entropy


class PPOAgent:
    """
    Agent PPO (Proximal Policy Optimization) avec support pour les actions continues.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        critic_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        mini_batch_size: int = 64,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise l'agent PPO.

        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Dimension de l'espace d'action
            hidden_size: Taille des couches cachées
            learning_rate: Taux d'apprentissage
            gamma: Facteur d'actualisation
            gae_lambda: Paramètre λ pour l'estimation de l'avantage généralisé
            clip_epsilon: Paramètre de clipping pour PPO
            critic_loss_coef: Coefficient pour la perte de la fonction valeur
            entropy_coef: Coefficient pour le terme d'entropie
            max_grad_norm: Valeur maximale de la norme du gradient
            update_epochs: Nombre d'époques pour mettre à jour les paramètres
            mini_batch_size: Taille des mini-batchs pour l'entraînement
            action_bounds: Bornes de l'espace d'action (min, max)
            device: Dispositif sur lequel exécuter les calculs
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.action_bounds = action_bounds
        self.device = device

        # Initialiser le réseau ActorCritic
        self.ac_network = PPOActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            action_bounds=action_bounds,
        ).to(device)

        # Initialiser l'optimiseur
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)

        # Historiques des pertes
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []

        logger.info(
            f"Agent PPO initialisé avec state_dim={state_dim}, action_dim={action_dim}"
        )

        # Tampons pour stocker les expériences
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def get_action(self, state, deterministic=False):
        """
        Sélectionne une action à partir d'un état.

        Args:
            state: État courant
            deterministic: Si True, sélectionne l'action de manière déterministe

        Returns:
            tuple: Action et log-probabilité associée
        """
        # Conversion en tensor si nécessaire
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action, log_prob = self.ac_network.get_action_and_log_prob(
                state, deterministic
            )

        # Stocker l'expérience si non déterministe (mode entraînement)
        if not deterministic:
            value = self.ac_network.get_value(state)
            self.log_probs.append(log_prob)
            self.values.append(value)

        return action.cpu().numpy(), (
            log_prob.cpu().numpy() if log_prob is not None else None
        )

    def compute_gae(self, next_value, rewards, masks, values):
        """
        Calcule les avantages avec Generalized Advantage Estimation (GAE).

        Args:
            next_value: Valeur de l'état suivant
            rewards: Liste de récompenses
            masks: Liste de masques (1 - done)
            values: Liste de valeurs estimées

        Returns:
            Tensors des retours et avantages
        """
        values = values + [next_value]
        advantages = []
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step]
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return returns, advantages

    def update(self, states, actions, rewards, next_states, dones):
        """
        Met à jour les paramètres du réseau avec PPO.

        Args:
            states: Liste des états
            actions: Liste des actions
            rewards: Liste des récompenses
            next_states: Liste des états suivants
            dones: Liste des indicateurs de fin d'épisode

        Returns:
            dict: Dictionnaire des statistiques d'entraînement
        """
        # Convertir en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calculer les valeurs des états
        with torch.no_grad():
            values = [self.ac_network.get_value(state).detach() for state in states]
            next_value = self.ac_network.get_value(next_states[-1]).detach()

        # Convertir les valeurs en liste
        values_list = [v.cpu().numpy()[0, 0] for v in values]

        # Calculer les retours et avantages
        masks = 1 - dones.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        returns, advantages = self.compute_gae(
            next_value.cpu().numpy()[0, 0], rewards_np, masks, values_list
        )

        # Convertir en tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Récupérer les log_probs et valeurs originales
        with torch.no_grad():
            old_log_probs, old_values, _ = self.ac_network.evaluate_actions(
                states, actions
            )

        # Variables pour les statistiques
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_epoch = 0

        # Mise à jour sur plusieurs époques
        for _ in range(self.update_epochs):
            # Générer les indices aléatoires
            permutation = torch.randperm(states.size(0))

            # Parcourir les mini-batches
            for start_idx in range(0, states.size(0), self.mini_batch_size):
                # Extraire les indices du mini-batch
                idx = permutation[start_idx : start_idx + self.mini_batch_size]

                # Extraire les données du mini-batch
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_old_log_probs = old_log_probs[idx]

                # Évaluer les actions actuelles
                new_log_probs, new_values, entropy = self.ac_network.evaluate_actions(
                    mb_states, mb_actions
                )

                # Calcul du ratio pour PPO
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Calcul des deux termes de la perte de PPO
                term1 = ratio * mb_advantages
                term2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )

                # Perte de l'acteur (sens négatif car on veut maximiser)
                actor_loss = -torch.min(term1, term2).mean()

                # Perte du critique (MSE)
                critic_loss = F.mse_loss(new_values, mb_returns)

                # Perte totale
                loss = (
                    actor_loss
                    + self.critic_loss_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Mise à jour des paramètres
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.ac_network.parameters(), self.max_grad_norm
                )

                self.optimizer.step()

                # Accumuler les statistiques
                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.mean().item()

        # Moyenner les pertes sur toutes les époques
        n_updates = states.size(0) // self.mini_batch_size
        actor_loss_epoch /= n_updates * self.update_epochs
        critic_loss_epoch /= n_updates * self.update_epochs
        entropy_epoch /= n_updates * self.update_epochs

        # Mettre à jour les historiques
        self.actor_loss_history.append(actor_loss_epoch)
        self.critic_loss_history.append(critic_loss_epoch)
        self.entropy_history.append(entropy_epoch)

        # Retourner les statistiques
        return {
            "actor_loss": actor_loss_epoch,
            "critic_loss": critic_loss_epoch,
            "entropy": entropy_epoch,
        }

    def save(self, path):
        """Sauvegarde le modèle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "ac_network": self.ac_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "actor_loss_history": self.actor_loss_history,
                "critic_loss_history": self.critic_loss_history,
                "entropy_history": self.entropy_history,
            },
            path,
        )
        logger.info(f"Modèle sauvegardé à {path}")

    def load(self, path):
        """Charge le modèle."""
        if not os.path.exists(path):
            logger.error(f"Le fichier {path} n'existe pas")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.ac_network.load_state_dict(checkpoint["ac_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.actor_loss_history = checkpoint["actor_loss_history"]
            self.critic_loss_history = checkpoint["critic_loss_history"]
            self.entropy_history = checkpoint["entropy_history"]
            logger.info(f"Modèle chargé depuis {path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False
