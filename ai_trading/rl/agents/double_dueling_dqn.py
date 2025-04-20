import logging
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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


class PrioritizedReplayBuffer:
    """
    Tampon de replay à priorité pour stocker et échantillonner des expériences.
    """

    def __init__(self, buffer_size=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialise le tampon de replay prioritaire.

        Args:
            buffer_size (int): Taille maximale du tampon
            alpha (float): Facteur d'exponentiation pour les priorités (0=uniforme, 1=totalement prioritaire)
            beta (float): Facteur de correction du biais d'importance-sampling
            beta_increment (float): Incrément de beta à chaque échantillonnage pour atteindre 1.0
        """
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_size = buffer_size
        self.position = 0
        self.size = 0

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience au tampon.

        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        # Utiliser la priorité maximale pour les nouvelles expériences
        max_priority = self.max_priority if self.size > 0 else 1.0

        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        Échantillonne un lot d'expériences selon leurs priorités.

        Args:
            batch_size (int): Taille du lot à échantillonner

        Returns:
            tuple: Contient (états, actions, récompenses, états suivants, indicateurs de fin, indices, poids IS)
        """
        if self.size < batch_size:
            idx = range(self.size)
        else:
            # Calculer les probabilités d'échantillonnage basées sur les priorités
            priorities = self.priorities[: self.size] ** self.alpha
            probabilities = priorities / np.sum(priorities)

            # Échantillonner selon les probabilités
            idx = np.random.choice(
                self.size, batch_size, replace=False, p=probabilities
            )

        # Calculer les poids d'importance-sampling
        weights = (self.size * probabilities[idx]) ** (-self.beta)
        weights /= np.max(weights)  # Normaliser pour stabilité

        # Incrémenter beta vers 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Extraire les échantillons
        states = np.array([self.buffer[i][0] for i in idx])
        actions = np.array([self.buffer[i][1] for i in idx])
        rewards = np.array([self.buffer[i][2] for i in idx])
        next_states = np.array([self.buffer[i][3] for i in idx])
        dones = np.array([self.buffer[i][4] for i in idx])

        return states, actions, rewards, next_states, dones, idx, weights

    def update_priorities(self, indices, priorities):
        """
        Met à jour les priorités des expériences.

        Args:
            indices (list): Indices des expériences à mettre à jour
            priorities (list): Nouvelles priorités
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


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

        # Compteur d'étapes pour la mise à jour du réseau cible
        self.target_update_counter = 0

        # Initialiser le tampon de replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size=buffer_size)
        else:
            self.memory = deque(maxlen=buffer_size)

        # Créer les réseaux principal et cible
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        logger.info(
            f"Agent initialisé avec state_size={state_size}, action_size={action_size}, "
            f"use_dueling={use_dueling}, use_prioritized_replay={use_prioritized_replay}"
        )

    def _build_model(self):
        """
        Construit le réseau de neurones pour l'agent.

        Returns:
            Model: Modèle Keras
        """
        inputs = Input(shape=(self.state_size,))

        # Couches partagées
        x = Dense(64, activation="relu")(inputs)
        x = Dense(64, activation="relu")(x)

        if self.use_dueling:
            # Architecture Dueling: séparer les flux de valeur et d'avantage
            state_value = Dense(32, activation="relu")(x)
            state_value = Dense(1)(state_value)

            action_advantages = Dense(32, activation="relu")(x)
            action_advantages = Dense(self.action_size)(action_advantages)

            # Combinaison de la valeur et des avantages
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
            action_advantages_mean = Lambda(
                lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
            )(action_advantages)
            action_advantages = Lambda(lambda x: x[0] - x[1])(
                [action_advantages, action_advantages_mean]
            )

            outputs = Lambda(lambda x: x[0] + x[1])([state_value, action_advantages])
        else:
            # Architecture DQN standard
            outputs = Dense(self.action_size)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")

        return model

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
            training (bool): Si True, utilise une politique epsilon-greedy

        Returns:
            int: Action choisie
        """
        # Exploration avec probabilité epsilon
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: choisir l'action avec la valeur Q maximale
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)

    def train(self):
        """
        Entraîne l'agent sur un lot d'expériences.

        Returns:
            float: Perte moyenne sur le lot
        """
        # Vérifier si le tampon contient suffisamment d'échantillons
        if self.use_prioritized_replay and self.memory.size < self.batch_size:
            return 0
        elif not self.use_prioritized_replay and len(self.memory) < self.batch_size:
            return 0

        # Échantillonner un lot du tampon de replay
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, is_weights = (
                self.memory.sample(self.batch_size)
            )
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.array([experience[0] for experience in minibatch])
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.array([experience[3] for experience in minibatch])
            dones = np.array([experience[4] for experience in minibatch])
            indices = None
            is_weights = np.ones(self.batch_size)

        # Double DQN: utiliser le modèle principal pour sélectionner l'action
        # et le modèle cible pour estimer sa valeur
        next_q_values = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_values, axis=1)

        # Obtenir les valeurs Q du modèle cible pour les prochains états
        target_next_q_values = self.target_model.predict(next_states, verbose=0)

        # Calculer les valeurs Q cibles
        targets = (
            rewards
            + (1 - dones)
            * self.discount_factor
            * target_next_q_values[np.arange(self.batch_size), next_actions]
        )

        # Obtenir les valeurs Q actuelles
        current_q = self.model.predict(states, verbose=0)

        # Calculer les erreurs TD pour la mise à jour des priorités
        td_errors = np.abs(targets - current_q[np.arange(self.batch_size), actions])

        # Mettre à jour les valeurs Q pour les actions prises
        target_q = current_q.copy()
        target_q[np.arange(self.batch_size), actions] = targets

        # Appliquer les poids d'importance-sampling pour le calcul de la perte
        sample_weights = is_weights

        # Entraîner le modèle
        history = self.model.fit(
            states,
            target_q,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
            sample_weight=sample_weights,
        )

        # Mettre à jour les priorités dans le tampon à priorité
        if self.use_prioritized_replay and indices is not None:
            self.memory.update_priorities(
                indices, td_errors + 1e-6
            )  # Ajouter un petit epsilon pour éviter les priorités nulles

        # Mettre à jour epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Incrémenter le compteur et mettre à jour le réseau cible si nécessaire
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            logger.debug("Mise à jour du réseau cible")

        return history.history["loss"][0]

    def save(self, filepath):
        """
        Sauvegarde le modèle.

        Args:
            filepath (str): Chemin du fichier pour la sauvegarde
        """
        self.model.save_weights(filepath)
        logger.info(f"Modèle sauvegardé à {filepath}")

    def load(self, filepath):
        """
        Charge le modèle.

        Args:
            filepath (str): Chemin du fichier à charger
        """
        self.model.load_weights(filepath)
        self.target_model.load_weights(filepath)
        logger.info(f"Modèle chargé depuis {filepath}")


class DuelingDQNAgent(DoubleDQNAgent):
    """
    Agent Dueling DQN pour le trading de cryptomonnaies.
    Hérite de Double DQN et active l'architecture Dueling.
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
    ):
        """
        Initialise l'agent Dueling DQN.

        Args:
            Voir DoubleDQNAgent pour la description des paramètres.
        """
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
            use_dueling=True,  # Activer l'architecture Dueling
        )


class DoubleDuelingDQNAgent(DoubleDQNAgent):
    """
    Agent combinant Double DQN et Dueling DQN pour le trading de cryptomonnaies.
    Cette combinaison offre les avantages des deux architectures.
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
    ):
        """
        Initialise l'agent Double-Dueling DQN.

        Args:
            Voir DoubleDQNAgent pour la description des paramètres.
        """
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
            use_dueling=True,  # Activer l'architecture Dueling
        )
        logger.info("Agent Double-Dueling DQN initialisé")
