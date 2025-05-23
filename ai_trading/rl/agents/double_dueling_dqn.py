import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai_trading.rl.replay_buffer import PrioritizedReplayBuffer

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DQNNetwork(nn.Module):
    """Réseau de neurones pour l'agent DQN."""

    def __init__(self, state_size, action_size, use_dueling=False):
        super(DQNNetwork, self).__init__()
        self.use_dueling = use_dueling

        # Couches partagées
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        if use_dueling:
            # Architecture Dueling
            self.value_stream = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_size)
            )
        else:
            # Architecture DQN standard
            self.output_layer = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.shared_layers(x)

        if self.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)

            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            advantage_mean = advantage.mean(dim=1, keepdim=True)
            q_values = value + (advantage - advantage_mean)
        else:
            q_values = self.output_layer(x)

        return q_values


class DoubleDQNAgent:
    """
    Agent Double DQN pour le trading de cryptomonnaies.
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        update_target_every=5,
        buffer_size=10000,
        use_prioritized_replay=True,
        use_dueling=False,
        device="cpu",
    ):
        """
        Initialise l'agent Double DQN.

        Args:
            state_size (int): Taille de l'espace d'état
            action_size (int): Taille de l'espace d'action
            learning_rate (float): Taux d'apprentissage pour l'optimiseur
            discount_factor (float): Facteur d'actualisation pour les récompenses futures
            epsilon (float): Valeur initiale d'epsilon pour l'exploration
            epsilon_decay (float): Taux de décroissance d'epsilon
            epsilon_min (float): Valeur minimale d'epsilon
            batch_size (int): Taille du lot pour l'entraînement
            update_target_every (int): Nombre d'étapes entre chaque mise à jour du réseau cible
            buffer_size (int): Taille du tampon de replay
            use_prioritized_replay (bool): Utiliser un tampon de replay prioritaire
            use_dueling (bool): Utiliser l'architecture Dueling DQN
            device (str): Appareil à utiliser pour les calculs ('cpu' ou 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.use_prioritized_replay = use_prioritized_replay
        self.use_dueling = use_dueling
        self.device = device

        # Compteur d'étapes pour la mise à jour du réseau cible
        self.target_update_counter = 0

        # Initialiser le tampon de replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size=buffer_size)
        else:
            self.memory = deque(maxlen=buffer_size)

        # Créer les réseaux principal et cible
        self.model = DQNNetwork(state_size, action_size, use_dueling).to(device)
        self.target_model = DQNNetwork(state_size, action_size, use_dueling).to(device)
        self.update_target_model()

        # Optimiseur et fonction de perte
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        logger.info(
            f"Agent initialisé avec state_size={state_size}, action_size={action_size}, "
            f"use_dueling={use_dueling}, use_prioritized_replay={use_prioritized_replay}"
        )

    def update_target_model(self):
        """Met à jour le modèle cible avec les poids du modèle principal."""
        self.target_model.load_state_dict(self.model.state_dict())
        logger.debug("Modèle cible mis à jour")

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke une expérience dans le tampon de replay.

        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        if self.use_prioritized_replay:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Choisit une action en fonction de l'état actuel.

        Args:
            state: État actuel
            training (bool): Si True, utilise la politique d'exploration

        Returns:
            int: Action choisie
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Convertir l'état en tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Obtenir les Q-values
        with torch.no_grad():
            q_values = self.model(state)

        # Retourner l'action avec la Q-value la plus élevée
        return q_values.argmax().item()

    def train(self):
        """
        Entraîne le modèle sur un batch d'expériences.

        Returns:
            float: Perte moyenne sur le batch
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Échantillonner un batch
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = (
                self.memory.sample(self.batch_size)
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.array([t[0] for t in minibatch])
            actions = np.array([t[1] for t in minibatch])
            rewards = np.array([t[2] for t in minibatch])
            next_states = np.array([t[3] for t in minibatch])
            dones = np.array([t[4] for t in minibatch])
            weights = torch.ones(self.batch_size).to(self.device)

        # Convertir en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calculer les Q-values actuelles
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Calculer les Q-values cibles (Double DQN)
        with torch.no_grad():
            # Sélectionner les actions avec le modèle principal
            next_actions = self.model(next_states).argmax(1)
            # Évaluer les Q-values avec le modèle cible
            next_q_values = self.target_model(next_states).gather(
                1, next_actions.unsqueeze(1)
            )
            target_q_values = (
                rewards + (1 - dones) * self.discount_factor * next_q_values
            )

        # Calculer la perte
        loss = self.criterion(current_q_values, target_q_values)
        loss = (loss * weights).mean()

        # Mettre à jour le modèle
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mettre à jour les priorités si on utilise le replay prioritaire
        if self.use_prioritized_replay:
            with torch.no_grad():
                td_errors = torch.abs(current_q_values - target_q_values).cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        # Mettre à jour le modèle cible périodiquement
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.update_target_model()
            self.target_update_counter = 0

        # Décroître epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filepath):
        """
        Sauvegarde les poids du modèle.

        Args:
            filepath (str): Chemin du fichier de sauvegarde
        """
        try:
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"Modèle sauvegardé dans {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")

    def load(self, filepath):
        """
        Charge les poids du modèle.

        Args:
            filepath (str): Chemin du fichier à charger
        """
        try:
            self.model.load_state_dict(torch.load(filepath))
            self.update_target_model()
            logger.info(f"Modèle chargé depuis {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")


class DuelingDQNAgent(DoubleDQNAgent):
    """Agent Dueling DQN qui hérite de DoubleDQNAgent."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        update_target_every=5,
        buffer_size=10000,
        use_prioritized_replay=True,
        device="cpu",
    ):
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            update_target_every=update_target_every,
            buffer_size=buffer_size,
            use_prioritized_replay=use_prioritized_replay,
            use_dueling=True,
            device=device,
        )


class DoubleDuelingDQNAgent(DoubleDQNAgent):
    """Agent Double Dueling DQN qui hérite de DoubleDQNAgent."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        update_target_every=5,
        buffer_size=10000,
        use_prioritized_replay=True,
        device="cpu",
    ):
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            update_target_every=update_target_every,
            buffer_size=buffer_size,
            use_prioritized_replay=use_prioritized_replay,
            use_dueling=True,
            device=device,
        )
        logger.info("Agent Double-Dueling DQN initialisé")
