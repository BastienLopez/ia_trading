import logging
import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # facteur d'actualisation
        self.epsilon = epsilon  # taux d'exploration
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Mémoire de replay
        self.memory = deque(maxlen=memory_size)

        # Modèle principal (pour la prédiction)
        self.model = self._build_model()

        # Modèle cible (pour la stabilité de l'apprentissage)
        self.target_model = self._build_model()
        self.update_target_model()

        # Métriques de suivi
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []

        logger.info(
            f"Agent DQN initialisé avec state_size={state_size}, action_size={action_size}"
        )

    def _build_model(self):
        """
        Construit le réseau de neurones pour l'approximation de Q-value.

        Returns:
            model: Modèle Keras compilé
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Met à jour le modèle cible avec les poids du modèle principal.
        """
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Modèle cible mis à jour")

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke l'expérience dans la mémoire de replay.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Si l'épisode est terminé
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        """
        Choisit une action en fonction de l'état actuel.

        Args:
            state (np.array): État actuel
            use_epsilon (bool): Utiliser epsilon-greedy ou non (pour l'évaluation)

        Returns:
            int: Action choisie
        """
        # Exploration aléatoire avec probabilité epsilon
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: choisir la meilleure action selon le modèle
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=None):
        """
        Entraîne le modèle sur un batch d'expériences.

        Args:
            batch_size (int, optional): Taille du batch. Si None, utilise self.batch_size

        Returns:
            float: Perte moyenne du batch
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Vérifier si la mémoire contient assez d'expériences
        if len(self.memory) < batch_size:
            return 0

        # Échantillonner un batch aléatoire de la mémoire
        minibatch = random.sample(self.memory, batch_size)

        # Préparer les données d'entraînement
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # Prédire les Q-values pour l'état actuel
            target = self.model.predict(state, verbose=0)[0]

            if done:
                # Si l'épisode est terminé, la cible est simplement la récompense
                target[action] = reward
            else:
                # Sinon, la cible est la récompense plus la Q-value future actualisée
                # Utiliser le modèle cible pour la stabilité
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)

            states[i] = state
            targets[i] = target

        # Entraîner le modèle
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        self.loss_history.append(loss)

        # Décroître epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.epsilon_history.append(self.epsilon)

        return loss

    def load(self, name):
        """
        Charge les poids du modèle à partir d'un fichier.

        Args:
            name (str): Chemin du fichier
        """
        if not name.endswith(".weights.h5"):
            name += ".weights.h5"
        self.model.load_weights(name)
        self.update_target_model()
        logger.info(f"Modèle chargé depuis {name}")

    def save(self, name):
        """
        Sauvegarde les poids du modèle dans un fichier.

        Args:
            name (str): Chemin du fichier
        """
        if not name.endswith(".weights.h5"):
            name += ".weights.h5"
        self.model.save_weights(name)
        logger.info(f"Modèle sauvegardé dans {name}")

    def get_metrics(self):
        """
        Retourne les métriques de l'agent.

        Returns:
            dict: Métriques de l'agent
        """
        return {
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
            "epsilon_history": self.epsilon_history,
        }

    def decay_epsilon(self):
        """Décroît epsilon selon le facteur de décroissance"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.epsilon_history.append(self.epsilon)
        return self.epsilon
