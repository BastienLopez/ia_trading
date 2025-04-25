import logging
import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense, Dropout
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
        self.model = self.build_model()

        # Modèle cible (pour la stabilité de l'apprentissage)
        self.target_model = self.build_model()
        self.update_target_model()

        # Métriques de suivi
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []

        logger.info(
            f"Agent DQN initialisé avec state_size={state_size}, action_size={action_size}"
        )

    def build_model(self):
        """
        Construit le modèle de réseau neuronal.

        Returns:
            keras.Model: Modèle compilé.
        """
        model = Sequential()

        # Déterminer dynamiquement la taille de l'état d'entrée
        input_shape = (
            (self.state_size,) if isinstance(self.state_size, int) else self.state_size
        )

        # Couche d'entrée
        model.add(Dense(64, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.2))

        # Couches cachées
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        # Couche de sortie
        model.add(Dense(self.action_size, activation="linear"))

        # Compilation du modèle
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
                # Si le premier élément est un array, on utilise celui-là
                if hasattr(state[0], "shape"):
                    state = state[0]
                else:
                    # Sinon, on essaie de prendre le premier élément du tuple
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
                # Si le premier élément est un array, on utilise celui-là
                if hasattr(next_state[0], "shape"):
                    next_state = next_state[0]
                else:
                    # Sinon, on essaie de prendre le premier élément du tuple
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
            # Mettre à jour la taille de l'état et reconstruire le modèle
            old_size = self.state_size
            self.state_size = state.shape[0]
            logger.warning(
                f"La taille de l'état ({self.state_size}) ne correspond pas à celle attendue par le modèle ({old_size}). Reconstruction des modèles..."
            )
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.update_target_model()
            
            # Vider la mémoire pour éviter les incohérences de tailles d'état
            self.memory.clear()
            logger.warning("Mémoire d'expérience vidée pour éviter les incohérences de tailles d'état.")
            
        # Vérifier que state et next_state ont la même taille
        if len(state.shape) == 1 and len(next_state.shape) == 1 and state.shape[0] != next_state.shape[0]:
            logger.warning(
                f"Les tailles de state ({state.shape[0]}) et next_state ({next_state.shape[0]}) diffèrent. Cette expérience sera ignorée."
            )
            return

        # Ajouter l'expérience à la mémoire
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choisit une action en fonction de l'état actuel.

        Args:
            state (numpy.array or tuple): État actuel.

        Returns:
            int: Action choisie.
        """
        # Gérer le cas où state est un tuple
        if isinstance(state, tuple):
            if len(state) > 0:
                # Si le premier élément est un array, on utilise celui-là
                if hasattr(state[0], "shape"):
                    state = state[0]
                else:
                    # Sinon, on essaie de prendre le premier élément du tuple
                    state = np.array([state[0]])

        # Assurons-nous que state est un array numpy
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state)
            except:
                logger.error(
                    f"Impossible de convertir state en array numpy, type: {type(state)}"
                )
                # Fallback en cas d'échec: action aléatoire
                return random.randrange(self.action_size)

        # S'assurer que l'état a la bonne forme pour le modèle
        if len(state.shape) == 1:
            state = np.reshape(state, [1, len(state)])
            
        # Vérifier si la taille de l'état correspond à ce que le modèle attend
        if state.shape[1] != self.state_size:
            logger.warning(
                f"La taille de l'état ({state.shape[1]}) ne correspond pas à celle attendue par le modèle ({self.state_size}). Reconstruction du modèle..."
            )
            self.state_size = state.shape[1]
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.update_target_model()

        # Exploration aléatoire
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: choisir la meilleure action
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def select_action(self, state):
        """
        Alias pour act().
        Choisit une action en fonction de l'état actuel.

        Args:
            state (numpy.array): État actuel.

        Returns:
            int: Action choisie.
        """
        return self.act(state)

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

        # Vérifier si la taille des états est constante dans le minibatch
        first_state = minibatch[0][0]
        if len(first_state.shape) > 1:
            first_state = first_state[0]
        
        # Si la taille d'état a changé, reconstruire les modèles
        if len(first_state) != self.state_size:
            logger.warning(
                f"La taille de l'état dans la mémoire ({len(first_state)}) ne correspond pas à celle attendue par le modèle ({self.state_size}). Reconstruction des modèles..."
            )
            self.state_size = len(first_state)
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.update_target_model()

        # Préparer les données d'entraînement
        states = np.zeros((batch_size, self.state_size), dtype=np.float32)
        targets = np.zeros((batch_size, self.action_size), dtype=np.float32)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # S'assurer que state est dans le bon format et en float32
            if len(state.shape) > 1:
                state = state[0]  # Prendre le premier élément si c'est un batch
            state = state.astype(np.float32)

            # S'assurer que next_state est dans le bon format et en float32
            if len(next_state.shape) > 1:
                next_state = next_state[0]
            next_state = next_state.astype(np.float32)
            
            # Vérifier si la taille de next_state correspond à celle attendue par le modèle
            if len(next_state) != self.state_size:
                logger.warning(
                    f"La taille de next_state ({len(next_state)}) ne correspond pas à celle attendue par le modèle ({self.state_size}). Cet exemple sera ignoré."
                )
                continue

            # Prédire les Q-values pour l'état actuel
            state_reshaped = np.reshape(state, [1, len(state)])
            target = self.model.predict(state_reshaped, verbose=0)[0]

            if done:
                # Si l'épisode est terminé, la cible est simplement la récompense
                target[action] = reward
            else:
                # Sinon, la cible est la récompense plus la Q-value future actualisée
                # Utiliser le modèle cible pour la stabilité
                next_state_reshaped = np.reshape(next_state, [1, len(next_state)])
                t = self.target_model.predict(next_state_reshaped, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)

            # Stocker pour l'entraînement par lot
            states[i] = state
            targets[i] = target

        # Entraîner le modèle
        history = self.model.fit(states, targets, batch_size=batch_size, verbose=0)

        # Enregistrer la perte
        loss = history.history["loss"][0]
        self.loss_history.append(loss)

        # Décroître epsilon
        self.decay_epsilon()

        return loss

    def learn(self, batch_size=None):
        """
        Alias pour replay().
        Entraîne le modèle sur un batch d'expériences.

        Args:
            batch_size (int, optional): Taille du batch. Si None, utilise self.batch_size

        Returns:
            float: Perte moyenne du batch
        """
        return self.replay(batch_size)

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
