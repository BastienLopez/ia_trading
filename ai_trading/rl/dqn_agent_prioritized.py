import logging
import random
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from .prioritized_replay_memory import PrioritizedReplayMemory

# Configuration du logger
logger = logging.getLogger("DQNAgentPrioritized")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DQNAgentPrioritized:
    """
    Agent d'apprentissage par renforcement utilisant DQN avec mémoire de replay priorisée.
    Cette version avancée de l'agent DQN échantillonne les expériences en fonction de leurs
    erreurs TD, en donnant priorité aux expériences qui ont plus à enseigner à l'agent.
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
        alpha=0.6,  # Facteur de priorisation (0 = uniforme, 1 = complètement priorisé)
        beta=0.4,   # Facteur de correction de biais (0 = pas de correction, 1 = correction complète)
        beta_increment=0.001,  # Incrément de beta après chaque échantillonnage
    ):
        """
        Initialise l'agent DQN avec mémoire priorisée.

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
            alpha (float): Facteur de priorisation
            beta (float): Facteur de correction de biais
            beta_increment (float): Incrément de beta
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Mémoire de replay priorisée
        self.memory = PrioritizedReplayMemory(
            capacity=memory_size,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment
        )

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
            f"Agent DQN avec mémoire priorisée initialisé: state_size={state_size}, action_size={action_size}, alpha={alpha}, beta={beta}"
        )

    def build_model(self):
        """
        Construit le modèle de réseau neuronal.
        
        Returns:
            keras.Model: Modèle compilé.
        """
        model = Sequential()
        
        # Déterminer dynamiquement la taille de l'état d'entrée
        input_shape = (self.state_size,) if isinstance(self.state_size, int) else self.state_size
        
        # Couche d'entrée
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.2))
        
        # Couches cachées
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        # Couche de sortie
        model.add(Dense(self.action_size, activation='linear'))
        
        # Compilation du modèle
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model

    def update_target_model(self):
        """
        Met à jour le modèle cible avec les poids du modèle principal.
        """
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Modèle cible mis à jour")

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke l'expérience dans la mémoire de replay priorisée.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Si l'épisode est terminé
        """
        # Format state et next_state si nécessaire
        if len(state.shape) == 1:
            state = np.reshape(state, [1, len(state)])
        if len(next_state.shape) == 1:
            next_state = np.reshape(next_state, [1, len(next_state)])
            
        # Ajouter à la mémoire priorisée
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        """
        Choisit une action en fonction de l'état actuel.
        
        Args:
            state (numpy.array): État actuel.
            
        Returns:
            int: Action choisie.
        """
        # S'assurer que l'état a la bonne forme pour le modèle
        if len(state.shape) == 1:
            state = np.reshape(state, [1, len(state)])
        
        # Exploration aléatoire
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: choisir la meilleure action
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=None):
        """
        Entraîne le modèle sur un batch d'expériences prioritaires.

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

        # Échantillonner un batch priorisé de la mémoire
        tree_indices, batch, is_weights = self.memory.sample(batch_size)
        
        # Préparer les arrays pour l'entraînement
        states = np.zeros((batch_size, self.state_size), dtype=np.float32)
        targets = np.zeros((batch_size, self.action_size), dtype=np.float32)
        
        # Calculer les Q-target pour chaque expérience et les erreurs TD
        td_errors = np.zeros(batch_size, dtype=np.float32)
        
        for i, experience in enumerate(batch):
            state, action, reward, next_state, done = experience
            
            # S'assurer que state est dans le bon format et en float32
            if len(state.shape) > 1:
                state = state[0]  # Prendre le premier élément si c'est un batch
            state = state.astype(np.float32)
            
            # S'assurer que next_state est dans le bon format et en float32
            if len(next_state.shape) > 1:
                next_state = next_state[0]
            next_state = next_state.astype(np.float32)
            
            # Prédire Q-values pour l'état actuel
            state_reshaped = np.reshape(state, [1, len(state)])
            target = self.model.predict(state_reshaped, verbose=0)[0]
            old_val = target[action]
            
            if done:
                # Si l'épisode est terminé, la cible est simplement la récompense
                target[action] = reward
            else:
                # Sinon, calculer la Q-target avec le modèle cible
                next_state_reshaped = np.reshape(next_state, [1, len(next_state)])
                t = self.target_model.predict(next_state_reshaped, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            
            # Calculer l'erreur TD (pour mettre à jour les priorités)
            td_errors[i] = abs(old_val - target[action])
            
            # Stocker les données d'entraînement
            states[i] = state
            targets[i] = target
        
        # Entraîner le modèle avec les poids d'importance-sampling
        history = self.model.fit(states, targets, sample_weight=is_weights, 
                                 epochs=1, verbose=0, batch_size=batch_size)
        loss = history.history["loss"][0]
        self.loss_history.append(loss)
        
        # Mettre à jour les priorités dans la mémoire
        self.memory.update_priorities(tree_indices, td_errors)
        
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