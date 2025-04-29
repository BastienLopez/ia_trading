import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Configuration du logger
logger = logging.getLogger("DQNAgent")
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

    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    def get_weights(self):
        """Retourne les poids du modèle sous forme de listes numpy."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy())
        return weights

    def set_weights(self, weights):
        """Définit les poids du modèle à partir d'une liste de tableaux numpy."""
        for param, weight in zip(self.parameters(), weights):
            param.data = torch.from_numpy(weight).to(param.device)


class DQNAgent:
    """
    Agent d'apprentissage par renforcement utilisant l'algorithme Deep Q-Network (DQN).
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=2000,
        device="cpu",
    ):
        """
        Initialise l'agent DQN.

        Args:
            state_size (int): Taille de l'espace d'état
            action_size (int): Taille de l'espace d'action
            learning_rate (float): Taux d'apprentissage pour l'optimiseur
            gamma (float): Facteur d'actualisation pour les récompenses futures
            epsilon (float): Taux d'exploration initial
            epsilon_decay (float): Facteur de décroissance d'epsilon
            epsilon_min (float): Valeur minimale d'epsilon
            batch_size (int): Taille du batch pour l'apprentissage
            memory_size (int): Taille de la mémoire de replay
            device (str): Appareil à utiliser pour les calculs ('cpu' ou 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device

        # Mémoire de replay
        self.memory = deque(maxlen=memory_size)

        # Modèles
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork(state_size, action_size).to(device)
        self.update_target_model()

        # Optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Métriques de suivi
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []

        logger.info(
            f"Agent DQN initialisé avec state_size={state_size}, action_size={action_size}"
        )

    def update_target_model(self):
        """Met à jour le modèle cible avec les poids du modèle principal."""
        self.target_model.load_state_dict(self.model.state_dict())
        logger.debug("Modèle cible mis à jour")

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke une expérience dans la mémoire.

        Args:
            state (numpy.array or tuple): État avant l'action
            action (int): Action effectuée
            reward (float): Récompense obtenue
            next_state (numpy.array or tuple): État après l'action
            done (bool): Si l'épisode est terminé
        """
        # Traiter l'état si c'est un tuple
        if isinstance(state, tuple):
            if len(state) > 0:
                if hasattr(state[0], "shape"):
                    state = state[0]
                else:
                    try:
                        state = np.array([state[0]])
                    except:
                        logger.error(
                            f"Impossible de convertir state en array numpy, type: {type(state)}"
                        )
                        return

        # Traiter next_state si c'est un tuple
        if isinstance(next_state, tuple):
            if len(next_state) > 0:
                if hasattr(next_state[0], "shape"):
                    next_state = next_state[0]
                else:
                    try:
                        next_state = np.array([next_state[0]])
                    except:
                        logger.error(
                            f"Impossible de convertir next_state en array numpy, type: {type(next_state)}"
                        )
                        return

        # Assurons-nous que state et next_state sont des arrays numpy
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state)
            except:
                logger.error(
                    f"Impossible de convertir state en array numpy dans remember, type: {type(state)}"
                )
                return

        if not isinstance(next_state, np.ndarray):
            try:
                next_state = np.array(next_state)
            except:
                logger.error(
                    f"Impossible de convertir next_state en array numpy dans remember, type: {type(next_state)}"
                )
                return

        # Vérifier si la taille de l'état correspond à celle attendue par le modèle
        if len(state.shape) == 1 and state.shape[0] != self.state_size:
            old_size = self.state_size
            self.state_size = state.shape[0]
            logger.warning(
                f"La taille de l'état ({self.state_size}) ne correspond pas à celle attendue par le modèle ({old_size}). Reconstruction des modèles..."
            )
            self.model = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_model = DQNNetwork(self.state_size, self.action_size).to(
                self.device
            )
            self.update_target_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Vider la mémoire pour éviter les incohérences de tailles d'état
            self.memory.clear()
            logger.warning(
                "Mémoire d'expérience vidée pour éviter les incohérences de tailles d'état."
            )

        # Vérifier que state et next_state ont la même taille
        if state.shape != next_state.shape:
            logger.error(
                f"Les tailles de state ({state.shape}) et next_state ({next_state.shape}) ne correspondent pas"
            )
            return

        # Stocker l'expérience dans la mémoire
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choisit une action en fonction de l'état actuel.

        Args:
            state (numpy.array): État actuel

        Returns:
            int: Action choisie
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Convertir l'état en tensor
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            elif len(state.shape) == 2 and state.shape[0] != 1:
                state = state.reshape(1, -1)
        state = torch.FloatTensor(state).to(self.device)

        # Obtenir les Q-values
        self.model.eval()  # Passer en mode évaluation
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()  # Repasser en mode entraînement

        # Retourner l'action avec la Q-value la plus élevée
        return q_values.argmax(dim=1).item()

    def select_action(self, state):
        """
        Sélectionne une action en fonction de l'état actuel.

        Args:
            state (numpy.array): État actuel

        Returns:
            int: Action sélectionnée
        """
        return self.act(state)

    def replay(self, batch_size=None):
        """Effectue une étape d'apprentissage sur un batch d'expériences."""
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return 0.0

        # Échantillonner un batch d'expériences
        minibatch = random.sample(self.memory, batch_size)

        # Convertir les expériences en tenseurs
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Reshape les états si nécessaire
        if len(states.shape) == 3:  # Si la forme est (batch_size, 1, state_size)
            states = states.squeeze(1)
            next_states = next_states.squeeze(1)
        elif len(states.shape) == 1:  # Si la forme est (state_size,)
            states = states.reshape(-1, self.state_size)
            next_states = next_states.reshape(-1, self.state_size)

        # Convertir en tenseurs PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calculer les Q-values actuelles
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculer les Q-values cibles
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculer la perte
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimiser le modèle
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Mettre à jour l'historique des pertes
        self.loss_history.append(loss.item())

        return loss.item()

    def learn(self, batch_size=None):
        """
        Alias pour la méthode replay.

        Args:
            batch_size (int, optional): Taille du batch. Si None, utilise self.batch_size.

        Returns:
            float: Perte moyenne sur le batch
        """
        return self.replay()

    def load(self, name):
        """
        Charge les poids du modèle depuis un fichier.

        Args:
            name (str): Nom du fichier à charger
        """
        try:
            self.model.load_state_dict(torch.load(name))
            self.update_target_model()
            logger.info(f"Modèle chargé depuis {name}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")

    def save(self, name):
        """
        Sauvegarde les poids du modèle dans un fichier.

        Args:
            name (str): Nom du fichier de sauvegarde
        """
        try:
            torch.save(self.model.state_dict(), name)
            logger.info(f"Modèle sauvegardé dans {name}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")

    def get_metrics(self):
        """
        Retourne les métriques de l'agent.

        Returns:
            dict: Dictionnaire contenant les métriques
        """
        return {
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
            "epsilon_history": self.epsilon_history,
        }

    def decay_epsilon(self):
        """Décroît la valeur d'epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.epsilon_history.append(self.epsilon)
