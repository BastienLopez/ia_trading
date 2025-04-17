import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, GRU, Reshape, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp
from collections import deque
import random
import traceback

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ReplayBuffer:
    """
    Tampon de replay simple pour stocker et échantillonner des expériences.
    """
    def __init__(self, buffer_size=100000):
        """
        Initialise le tampon de replay.
        
        Args:
            buffer_size (int): Taille maximale du tampon
        """
        self.buffer = deque(maxlen=buffer_size)
        
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
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Échantillonne un lot d'expériences aléatoirement.
        
        Args:
            batch_size (int): Taille du lot à échantillonner
            
        Returns:
            tuple: Contient (états, actions, récompenses, états suivants, indicateurs de fin)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def size(self):
        """Retourne le nombre d'expériences stockées dans le tampon."""
        return len(self.buffer)

class SequenceReplayBuffer:
    """
    Tampon de replay pour stocker et échantillonner des séquences d'expériences.
    Utilisé pour les architectures avec GRU qui nécessitent des séquences temporelles.
    """
    def __init__(self, buffer_size=100000, sequence_length=10):
        """
        Initialise le tampon de replay pour séquences.
        
        Args:
            buffer_size (int): Taille maximale du tampon
            sequence_length (int): Longueur des séquences temporelles
        """
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
        # Buffer temporaire pour construire les séquences
        self.temp_buffer = deque(maxlen=sequence_length)
        
    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience au tampon temporaire et construit des séquences.
        
        Args:
            state: État actuel ou séquence d'états
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant ou séquence d'états suivants
            done: Indicateur de fin d'épisode
        """
        # Conversion explicite en float32 pour éviter les problèmes de type
        if isinstance(state, np.ndarray) and state.dtype == np.dtype('O'):
            state = state.astype(np.float32)
        
        if isinstance(next_state, np.ndarray) and next_state.dtype == np.dtype('O'):
            next_state = next_state.astype(np.float32)
        
        # Vérifier et corriger les NaN dans les entrées
        if isinstance(state, np.ndarray) and np.any(np.isnan(state)):
            logging.warning("NaN détecté dans l'état ajouté au buffer. Remplacé par 0.")
            state = np.nan_to_num(state, nan=0.0)
        
        if isinstance(next_state, np.ndarray) and np.any(np.isnan(next_state)):
            logging.warning("NaN détecté dans le next_state ajouté au buffer. Remplacé par 0.")
            next_state = np.nan_to_num(next_state, nan=0.0)
            
        # Vérifier et corriger les problèmes de forme
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state)
            except:
                logging.error(f"Impossible de convertir state de type {type(state)} en numpy.ndarray")
                return
            
        if not isinstance(next_state, np.ndarray):
            try:
                next_state = np.array(next_state)
            except:
                logging.error(f"Impossible de convertir next_state de type {type(next_state)} en numpy.ndarray")
                return
        
        # Corriger la forme des séquences
        if len(state.shape) > 1:
            state = self._correct_sequence_shape(state)
        
        if len(next_state.shape) > 1:
            next_state = self._correct_sequence_shape(next_state)
        
        # Vérifier si state est déjà une séquence
        if len(state.shape) >= 2 and state.shape[0] == self.sequence_length:
            # C'est déjà une séquence complète, l'ajouter directement au buffer principal
            # Vérifier si next_state est également une séquence
            if len(next_state.shape) >= 2 and next_state.shape[0] == self.sequence_length:
                # Utiliser next_state tel quel car c'est déjà une séquence
                next_states_seq = next_state
            else:
                # Créer une séquence de next_states répétant le même next_state
                next_states_seq = np.array([next_state] * self.sequence_length)
            
            # Ajouter directement au buffer principal
            self.buffer.append((state, action, reward, next_states_seq, done))
            
            # Si l'épisode est terminé, vider le buffer temporaire
            if done:
                self.temp_buffer.clear()
        
        else:
            # Ajouter au buffer temporaire
            self.temp_buffer.append((state, action, reward, next_state, done))
            
            # Si le buffer temporaire est plein, créer une séquence
            if len(self.temp_buffer) == self.sequence_length:
                # Extraire les séquences d'états
                states_seq = np.array([exp[0] for exp in self.temp_buffer])
                
                # Extraire les séquences d'états suivants
                next_states_seq = np.array([exp[3] for exp in self.temp_buffer])
                
                # Utiliser l'action, la récompense et le done du dernier élément de la séquence
                last_exp = self.temp_buffer[-1]
                action = last_exp[1]
                reward = last_exp[2]
                done = last_exp[4]
                
                # Ajouter la séquence complète au buffer principal
                self.buffer.append((states_seq, action, reward, next_states_seq, done))
                
                # Si l'épisode est terminé, vider le buffer temporaire
                if done:
                    self.temp_buffer.clear()

    def _correct_sequence_shape(self, sequence):
        """
        Corrige la forme d'une séquence pour s'assurer qu'elle est correcte.
        
        Args:
            sequence: La séquence à corriger
            
        Returns:
            La séquence avec la forme corrigée
        """
        # Convertir en tableau NumPy si ce n'est pas déjà le cas
        if not isinstance(sequence, np.ndarray):
            try:
                sequence = np.array(sequence)
            except:
                logging.error(f"Impossible de convertir {type(sequence)} en numpy.ndarray")
                # Créer une séquence de zéros
                return np.zeros((self.sequence_length, self.state_size))
        
        # Si c'est déjà une séquence de la bonne forme, la retourner telle quelle
        if len(sequence.shape) == 2 and sequence.shape[0] == self.sequence_length:
            return sequence
        
        # Si c'est un batch de séquences avec batch_size=1
        if len(sequence.shape) == 3 and sequence.shape[0] == 1 and sequence.shape[1] == self.sequence_length:
            return sequence[0]  # Retirer la dimension de batch
        
        # Si c'est une séquence mais pas avec la longueur correcte
        if len(sequence.shape) == 2 and sequence.shape[0] != self.sequence_length:
            # Trop courte: remplir avec des zéros
            if sequence.shape[0] < self.sequence_length:
                padding = np.zeros((self.sequence_length - sequence.shape[0], sequence.shape[1]))
                return np.vstack([padding, sequence])
            # Trop longue: tronquer
            else:
                return sequence[-self.sequence_length:]
            
        # Si c'est une séquence 3D avec une forme incorrecte
        if len(sequence.shape) == 3:
            # Essayer de déterminer la bonne dimension
            if sequence.shape[1] == self.sequence_length:
                # Prendre le premier batch
                return sequence[0]
            elif sequence.shape[0] == self.sequence_length:
                # Prendre la première "feature" de chaque timestep
                return sequence[:, 0, :]
        
        # Si c'est un état unique (1D), le répéter pour former une séquence
        if len(sequence.shape) == 1:
            repeated = np.array([sequence] * self.sequence_length)
            return repeated
        
        # Si on ne peut pas déterminer comment corriger, retourner une séquence de zéros
        logging.warning(f"Impossible de corriger la forme de la séquence {sequence.shape}. Utilisation de zéros.")
        return np.zeros((self.sequence_length, sequence.shape[-1] if len(sequence.shape) > 1 else sequence.shape[0]))
        
    def sample(self, batch_size):
        """
        Échantillonne un lot de séquences aléatoirement.
        
        Args:
            batch_size (int): Taille du lot à échantillonner
            
        Returns:
            tuple: Contient (séquences d'états, actions, récompenses, séquences d'états suivants, indicateurs de fin)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states_seqs = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        next_states_seqs = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch]).reshape(-1, 1)
        
        # Vérification et correction des formes pour states_seqs
        if len(states_seqs.shape) > 3:
            logging.warning(f"Forme incorrecte détectée dans states_seqs échantillonnés: {states_seqs.shape}")
            # Si la forme est (batch_size, 1, sequence_length, state_size)
            if states_seqs.shape[1] == 1:
                states_seqs = states_seqs.reshape(batch_size, self.sequence_length, -1)
        
        # Vérification et correction des formes pour next_states_seqs
        if len(next_states_seqs.shape) > 3:
            logging.warning(f"Forme incorrecte détectée dans next_states_seqs échantillonnés: {next_states_seqs.shape}")
            # Si la forme est (batch_size, 1, sequence_length, state_size)
            if next_states_seqs.shape[1] == 1:
                next_states_seqs = next_states_seqs.reshape(batch_size, self.sequence_length, -1)
        
        # Vérifier si next_states_seqs est un ensemble d'états simples (sans séquence)
        # et dans ce cas, créer des séquences par répétition
        if len(next_states_seqs.shape) == 2:
            logging.warning(f"next_states_seqs n'est pas une séquence: {next_states_seqs.shape}. Création de séquences.")
            # Créer une séquence pour chaque état
            repeated_next_states = []
            for next_state in next_states_seqs:
                repeated_next_states.append(np.array([next_state] * self.sequence_length))
            next_states_seqs = np.array(repeated_next_states)
        
        # Vérification des NaN
        if np.any(np.isnan(states_seqs)):
            logging.warning("NaN détectés dans les états échantillonnés. Remplacés par 0.")
            states_seqs = np.nan_to_num(states_seqs, nan=0.0)
        
        if np.any(np.isnan(next_states_seqs)):
            logging.warning("NaN détectés dans les états suivants échantillonnés. Remplacés par 0.")
            next_states_seqs = np.nan_to_num(next_states_seqs, nan=0.0)
        
        return states_seqs, actions, rewards, next_states_seqs, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def size(self):
        """Retourne le nombre d'expériences stockées dans le tampon."""
        return len(self.buffer)

class SACAgent:
    """
    Soft Actor-Critic agent for reinforcement learning, based on maximum entropy.
    Cette implémentation inclut:
    - Gradient clipping pour stabiliser l'apprentissage
    - Régularisation d'entropie pour encourager l'exploration
    - Support optionnel pour les couches GRU pour traiter des séquences temporelles
    """
    
    def __init__(
        self, 
        state_size, 
        action_size, 
        action_bounds=[-1, 1],
        actor_learning_rate=3e-4, 
        critic_learning_rate=3e-4, 
        alpha_learning_rate=3e-4, 
        discount_factor=0.99, 
        tau=0.005, 
        batch_size=64, 
        buffer_size=100000,
        hidden_size=256, 
        train_alpha=True,
        target_entropy=None,
        grad_clip_value=None,
        entropy_regularization=0.0,
        use_gru=False,
        sequence_length=None,
        gru_units=64
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bounds = action_bounds
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.grad_clip_value = grad_clip_value
        self.entropy_regularization = entropy_regularization
        self.use_gru = use_gru
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        
        # Initialiser les historiques de métriques
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []
        
        # Vérifier si nous utilisons GRU et si la longueur de séquence est spécifiée
        if self.use_gru and self.sequence_length is None:
            raise ValueError("sequence_length doit être spécifié lorsque use_gru=True")
        
        # Mémorisation des expériences
        if self.use_gru:
            self.memory = SequenceReplayBuffer(
                buffer_size=buffer_size,
                sequence_length=sequence_length
            )
        else:
            self.memory = ReplayBuffer(
                buffer_size=buffer_size
            )
        
        # Alpha (température): contrôle l'importance de l'entropie
        self.train_alpha = train_alpha
        if target_entropy is None:
            self.target_entropy = -self.action_size  # -dim(A)
        else:
            self.target_entropy = target_entropy
        
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tf.exp(self.log_alpha)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_learning_rate)
        
        # Créer les réseaux
        self._build_networks()
        
        logging.info(f"SACAgent initialisé avec state_size={state_size}, action_size={action_size}, "
                    f"use_gru={use_gru}, sequence_length={sequence_length}, "
                    f"grad_clip_value={grad_clip_value}, entropy_regularization={entropy_regularization}")

    def _build_networks(self):
        """Construit les réseaux d'acteur et de critique."""
        if self.use_gru:
            # Modèles avec couches GRU pour traiter des séquences
            self.actor = self._build_gru_actor_network()
            self.critic_1 = self._build_gru_critic_network()
            self.critic_2 = self._build_gru_critic_network()
            self.target_critic_1 = self._build_gru_critic_network()
            self.target_critic_2 = self._build_gru_critic_network()
        else:
            # Modèles standards pour états simples
            self.actor = self._build_actor_network()
            self.critic_1 = self._build_critic_network()
            self.critic_2 = self._build_critic_network()
            self.target_critic_1 = self._build_critic_network()
            self.target_critic_2 = self._build_critic_network()
        
        # Copier les poids initiaux aux réseaux cibles
        self._update_target_networks(tau=1.0)
        
        # Optimiseurs
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate)
    
    def _build_actor_network(self):
        """Construit le réseau d'acteur standard."""
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(self.hidden_size, activation='relu')(inputs)
        x = tf.keras.layers.Dense(self.hidden_size, activation='relu')(x)
        
        mean = tf.keras.layers.Dense(self.action_size, activation=None)(x)
        log_std = tf.keras.layers.Dense(self.action_size, activation=None)(x)
        # Limiter log_std pour éviter des valeurs extrêmes
        log_std = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])
    
    def _build_critic_network(self):
        """Construit le réseau de critique standard."""
        state_inputs = tf.keras.layers.Input(shape=(self.state_size,))
        action_inputs = tf.keras.layers.Input(shape=(self.action_size,))
        
        x = tf.keras.layers.Concatenate()([state_inputs, action_inputs])
        x = tf.keras.layers.Dense(self.hidden_size, activation='relu')(x)
        x = tf.keras.layers.Dense(self.hidden_size, activation='relu')(x)
        q_value = tf.keras.layers.Dense(1, activation=None)(x)
        
        return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=q_value)
    
    def _build_gru_actor_network(self):
        """Construit le réseau d'acteur avec couches GRU."""
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.state_size))
        x = tf.keras.layers.GRU(units=self.gru_units, return_sequences=False)(inputs)
        x = tf.keras.layers.Dense(self.hidden_size, activation='relu')(x)
        
        mean = tf.keras.layers.Dense(self.action_size, activation=None)(x)
        log_std = tf.keras.layers.Dense(self.action_size, activation=None)(x)
        log_std = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])
    
    def _build_gru_critic_network(self):
        """Construit le réseau de critique avec couches GRU."""
        state_inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.state_size))
        action_inputs = tf.keras.layers.Input(shape=(self.action_size,))
        
        # Traiter d'abord la séquence d'états avec GRU
        state_features = tf.keras.layers.GRU(units=self.gru_units, return_sequences=False)(state_inputs)
        
        # Concaténer avec l'action (qui est ponctuelle, pas une séquence)
        x = tf.keras.layers.Concatenate()([state_features, action_inputs])
        x = tf.keras.layers.Dense(self.hidden_size, activation='relu')(x)
        q_value = tf.keras.layers.Dense(1, activation=None)(x)
        
        return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=q_value)
    
    def _normalize_sequence_states(self, state_sequence):
        """
        Normalise une séquence d'états pour l'entrée GRU.
        
        Args:
            state_sequence: Séquence d'états (batch_size, seq_length, state_size)
            
        Returns:
            tf.Tensor: Séquence d'états normalisée
        """
        if isinstance(state_sequence, np.ndarray):
            # Convertir en tensor si c'est un tableau NumPy
            state_sequence = tf.convert_to_tensor(state_sequence, dtype=tf.float32)
        
        # Vérifier les NaN et les remplacer par zéro
        if tf.reduce_any(tf.math.is_nan(state_sequence)):
            logging.warning("NaNs détectés dans la séquence d'états pendant la normalisation")
            state_sequence = tf.where(
                tf.math.is_nan(state_sequence), 
                tf.zeros_like(state_sequence), 
                state_sequence
            )
        
        # Normalisation par batch pour chaque dimension d'état
        # Reshape pour manipuler chaque dimension séparément
        batch_size, seq_length, state_size = tf.shape(state_sequence)[0], tf.shape(state_sequence)[1], tf.shape(state_sequence)[2]
        
        # Reshape pour fusionner batch et séquence (traiter chaque séquence indépendamment)
        flat_states = tf.reshape(state_sequence, [batch_size, seq_length * state_size])
        
        # Calcul des moyennes et écarts-types par batch
        means = tf.reduce_mean(flat_states, axis=1, keepdims=True)
        stddevs = tf.math.reduce_std(flat_states, axis=1, keepdims=True)
        
        # Gérer les écarts-types nuls ou proches de zéro
        epsilon = 1e-8
        safe_stddevs = tf.maximum(stddevs, epsilon)
        
        # Normalisation
        norm_flat_states = (flat_states - means) / safe_stddevs
        
        # Reshape à la forme originale
        normalized_sequence = tf.reshape(norm_flat_states, [batch_size, seq_length, state_size])
        
        return normalized_sequence

    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire."""
        self.memory.add(state, action, reward, next_state, done)
    
    def _get_action_and_log_prob(self, state, is_evaluation=False):
        """
        Récupère une action et sa log-probabilité en fonction de l'état actuel.
        
        Args:
            state: État actuel
            is_evaluation (bool): Si True, pas d'exploration (utilisé pour l'évaluation)
            
        Returns:
            tuple: (action, log_prob)
        """
        # Vérification des valeurs NaN dans l'état
        if isinstance(state, np.ndarray) and np.any(np.isnan(state)):
            print("[WARNING] NaN détecté dans l'état. Remplacement par zéro.")
            state = np.nan_to_num(state, nan=0.0)
        
        # Si nous utilisons GRU, ajuster la forme de l'état en conséquence
        if self.use_gru:
            # Pour les séquences d'états dans GRU
            if len(state.shape) == 2:  # état unique (batch_size, seq_len*features)
                # Reshape: (batch_size, seq_len, features)
                state = np.reshape(state, (1, self.sequence_length, -1))
            elif len(state.shape) == 1:  # état unique non batché
                # Reshape: (batch_size, seq_len, features)
                state = np.reshape(state, (1, self.sequence_length, -1))
        
        # Convertir en tensor en s'assurant qu'il n'y a pas de NaN
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # Vérifier encore les NaN après conversion (par précaution)
        if tf.reduce_any(tf.math.is_nan(state_tensor)):
            print("[WARNING] NaN détecté dans state_tensor. Remplacement par zéro.")
            state_tensor = tf.where(tf.math.is_nan(state_tensor), tf.zeros_like(state_tensor), state_tensor)
            
        # Obtenir la distribution normale des actions
        mean, log_std = self.actor(state_tensor, training=False)
        
        # Vérifier la présence de NaN dans mean et log_std
        if tf.reduce_any(tf.math.is_nan(mean)):
            print("[WARNING] NaN détecté dans mean. Remplacement par zéro.")
            mean = tf.where(tf.math.is_nan(mean), tf.zeros_like(mean), mean)
            
        if tf.reduce_any(tf.math.is_nan(log_std)):
            print("[WARNING] NaN détecté dans log_std. Remplacement par -20.")
            log_std = tf.where(tf.math.is_nan(log_std), -20.0 * tf.ones_like(log_std), log_std)
            
        # Créer la distribution normale
        std = tf.math.exp(log_std)
        normal_dist = tfp.distributions.Normal(mean, std)
        
        # En évaluation, utiliser directement la moyenne (action déterministe)
        if is_evaluation:
            action = mean
        else:
            # Échantillonner de la distribution pour l'action
            action = normal_dist.sample()
        
        # Vérifier si l'action contient des NaN
        if tf.reduce_any(tf.math.is_nan(action)):
            print("[WARNING] NaN détecté dans l'action. Remplacement par zéro.")
            action = tf.where(tf.math.is_nan(action), tf.zeros_like(action), action)
            
        # Appliquer tanh pour limiter l'action entre -1 et 1
        squashed_action = tf.math.tanh(action)
        
        # Vérifier si squashed_action contient des NaN
        if tf.reduce_any(tf.math.is_nan(squashed_action)):
            print("[WARNING] NaN détecté dans squashed_action. Remplacement par zéro.")
            squashed_action = tf.where(tf.math.is_nan(squashed_action), tf.zeros_like(squashed_action), squashed_action)
        
        # Calculer le log prob de l'action en tenant compte de tanh
        log_prob = normal_dist.log_prob(action)
        log_prob -= tf.math.log(1.0 - tf.math.square(squashed_action) + 1e-6)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        
        # Vérifier si log_prob contient des NaN
        if tf.reduce_any(tf.math.is_nan(log_prob)):
            print("[WARNING] NaN détecté dans log_prob. Remplacement par -1000.")
            log_prob = tf.where(tf.math.is_nan(log_prob), -1000.0 * tf.ones_like(log_prob), log_prob)
            
        return squashed_action, log_prob
        
    def act(self, state, evaluate=False):
        """
        Sélectionne une action à partir de l'état actuel.
        
        Args:
            state: État actuel
            evaluate (bool): Si True, pas d'exploration (utilisé pour l'évaluation)
            
        Returns:
            np.ndarray: Action mise à l'échelle pour l'environnement
        """
        try:
            # Vérification des valeurs NaN dans l'état
            if isinstance(state, np.ndarray) and np.any(np.isnan(state)):
                logging.warning("NaN détectés dans l'état. Remplacés par zéros.")
                state = np.nan_to_num(state, nan=0.0)
            
            # Prétraiter l'état et s'assurer qu'il a la bonne forme
            state_input = self._preprocess_state(state)
            
            # Obtenir l'action en utilisant la méthode _get_action_and_log_prob
            action, _ = self._get_action_and_log_prob(state_input, is_evaluation=evaluate)
            
            # Vérifier si la sortie contient des NaN
            if isinstance(action, tf.Tensor) and tf.reduce_any(tf.math.is_nan(action)):
                logging.error("NaN détectés dans les sorties du réseau acteur.")
                return np.zeros(self.action_size)
            
            # Convertir en numpy si nécessaire et appliquer la mise à l'échelle
            if isinstance(action, tf.Tensor):
                # Extraire le premier élément si c'est un batch
                if len(action.shape) > 1:
                    action = action[0]
                action = action.numpy()
            
            # Mettre à l'échelle l'action pour l'environnement
            scaled_action = self._scale_action(action)
            
            return scaled_action
            
        except Exception as e:
            logging.error(f"Erreur lors de la sélection d'action: {str(e)}")
            logging.error(traceback.format_exc())
            # Retourner une action nulle en cas d'erreur
            return np.zeros(self.action_size)

    def _scale_action(self, action):
        """
        Met à l'échelle une action normalisée (entre -1 et 1) vers la plage d'action réelle.
        
        Args:
            action (numpy.array): Action normalisée entre -1 et 1
        
        Returns:
            numpy.array: Action mise à l'échelle
        """
        # Vérification des valeurs NaN avant la mise à l'échelle
        if np.any(np.isnan(action)):
            print("[WARNING] NaN détecté dans l'action avant mise à l'échelle. Remplacement par zéro.")
            action = np.nan_to_num(action, nan=0.0)
        
        scaled = 0.5 * (action + 1.0) * (self.action_high - self.action_low) + self.action_low
        
        # Vérification après la mise à l'échelle
        if np.any(np.isnan(scaled)):
            print("[WARNING] NaN détecté après mise à l'échelle. Remplacement par la limite inférieure.")
            scaled = np.nan_to_num(scaled, nan=self.action_low)
        
        return scaled
    
    def _unscale_action(self, scaled_action):
        """
        Inverse la mise à l'échelle d'une action réelle vers une action normalisée (entre -1 et 1).
        
        Args:
            scaled_action (numpy.array): Action mise à l'échelle
        
        Returns:
            numpy.array: Action normalisée entre -1 et 1
        """
        # Vérification des valeurs NaN avant l'inversion de mise à l'échelle
        if np.any(np.isnan(scaled_action)):
            print("[WARNING] NaN détecté dans l'action mise à l'échelle. Remplacement par la limite inférieure.")
            scaled_action = np.nan_to_num(scaled_action, nan=self.action_low)
        
        unscaled = 2.0 * (scaled_action - self.action_low) / (self.action_high - self.action_low) - 1.0
        
        # Vérification après l'inversion de mise à l'échelle
        if np.any(np.isnan(unscaled)):
            print("[WARNING] NaN détecté après inversion de mise à l'échelle. Remplacement par zéro.")
            unscaled = np.nan_to_num(unscaled, nan=0.0)
        
        return unscaled

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """
        Effectue une étape d'entraînement pour tous les réseaux.
        
        Args:
            states: Lot d'états
            actions: Lot d'actions
            rewards: Lot de récompenses
            next_states: Lot d'états suivants
            dones: Lot d'indicateurs de fin d'épisode
            
        Returns:
            tuple: (critic_loss, actor_loss, alpha_loss, entropy)
        """
        with tf.GradientTape(persistent=True) as tape:
            # Échantillonner des actions pour l'état suivant
            next_means, next_log_stds = self.actor(next_states)
            next_stds = tf.exp(next_log_stds)
            next_normal_dists = tfp.distributions.Normal(next_means, next_stds)
            next_actions_raw = next_normal_dists.sample()
            next_actions = tf.tanh(next_actions_raw)
            
            # Calculer log-prob pour les actions suivantes
            # log_prob = log_prob_raw - log(1 - tanh(action_raw)²)
            log_probs_next = next_normal_dists.log_prob(next_actions_raw) - \
                          tf.math.log(1.0 - tf.square(next_actions) + 1e-6)
            log_probs_next = tf.reduce_sum(log_probs_next, axis=1, keepdims=True)
            
            # Valeurs Q cibles pour l'apprentissage du critique
            next_q1 = self.target_critic_1([next_states, next_actions])
            next_q2 = self.target_critic_2([next_states, next_actions])
            next_q_min = tf.minimum(next_q1, next_q2)
            
            # Soustraction du terme d'entropie pour SAC
            next_q_value = next_q_min - self.alpha * log_probs_next
            target_q = rewards + (1 - dones) * self.discount_factor * next_q_value
            
            # Valeurs Q actuelles
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])
            
            # Pertes des critiques (MSE)
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))
            
            # Échantillonner des actions pour l'état actuel (pour l'apprentissage de l'acteur)
            means, log_stds = self.actor(states)
            stds = tf.exp(log_stds)
            normal_dists = tfp.distributions.Normal(means, stds)
            actions_raw = normal_dists.sample()
            actions_policy = tf.tanh(actions_raw)
            
            # Calculer log-prob pour les actions de la politique
            log_probs = normal_dists.log_prob(actions_raw) - \
                        tf.math.log(1.0 - tf.square(actions_policy) + 1e-6)
            log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
            
            # Valeurs Q pour les actions de la politique
            q1 = self.critic_1([states, actions_policy])
            q2 = self.critic_2([states, actions_policy])
            q_min = tf.minimum(q1, q2)
            
            # Calcul de l'entropie
            entropy = -tf.reduce_mean(log_probs)
            
            # Perte de l'acteur = valeur Q attendue - entropie
            # Ajout d'une régularisation d'entropie supplémentaire
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_min) - self.entropy_regularization * entropy
            
            # Perte pour l'adaptation d'alpha (si activée)
            if self.train_alpha:
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
                )
            else:
                alpha_loss = tf.constant(0.0)
            
        # Calculer les gradients et mettre à jour les paramètres
        critic1_gradients = tape.gradient(critic1_loss, self.critic_1.trainable_variables)
        critic2_gradients = tape.gradient(critic2_loss, self.critic_2.trainable_variables)
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        # Appliquer le gradient clipping seulement si grad_clip_value est spécifié
        if self.grad_clip_value is not None:
            critic1_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                               for g in critic1_gradients]
            critic2_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                               for g in critic2_gradients]
            actor_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                               for g in actor_gradients]
        
        self.critic_1_optimizer.apply_gradients(zip(critic1_gradients, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic2_gradients, self.critic_2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        if self.train_alpha:
            alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
            # Appliquer le gradient clipping pour alpha aussi, seulement si grad_clip_value est spécifié
            if self.grad_clip_value is not None:
                alpha_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                                  for g in alpha_gradients]
            self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        del tape
        
        # Mettre à jour les réseaux cibles avec des mises à jour douces
        for target_param, param in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
            
        for target_param, param in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
        
        return (critic1_loss + critic2_loss) / 2.0, actor_loss, alpha_loss, entropy
        
    @tf.function
    def _train_step_gru(self, states_seqs, actions, rewards, next_states_seqs, dones):
        """
        Effectue une étape d'entraînement pour tous les réseaux avec des séquences temporelles.
        
        Args:
            states_seqs: Lot de séquences d'états (batch_size, sequence_length, state_size)
            actions: Lot d'actions
            rewards: Lot de récompenses
            next_states_seqs: Lot de séquences d'états suivants
            dones: Lot d'indicateurs de fin d'épisode
            
        Returns:
            tuple: (critic_loss, actor_loss, alpha_loss, entropy)
        """
        with tf.GradientTape(persistent=True) as tape:
            # Échantillonner des actions pour l'état suivant en utilisant la politique
            next_means, next_log_stds = self.actor(next_states_seqs)
            next_stds = tf.exp(next_log_stds)
            next_normal_dists = tfp.distributions.Normal(next_means, next_stds)
            next_actions_raw = next_normal_dists.sample()
            next_actions = tf.tanh(next_actions_raw)
            
            # Calculer log-prob pour les actions suivantes
            # log_prob = log_prob_raw - log(1 - tanh(action_raw)²)
            log_probs_next = next_normal_dists.log_prob(next_actions_raw) - \
                          tf.math.log(1.0 - tf.square(next_actions) + 1e-6)
            log_probs_next = tf.reduce_sum(log_probs_next, axis=1, keepdims=True)
            
            # Valeurs Q cibles pour l'apprentissage du critique
            next_q1 = self.target_critic_1([next_states_seqs, next_actions])
            next_q2 = self.target_critic_2([next_states_seqs, next_actions])
            next_q_min = tf.minimum(next_q1, next_q2)
            
            # Soustraction du terme d'entropie pour SAC
            next_q_value = next_q_min - self.alpha * log_probs_next
            target_q = rewards + (1 - dones) * self.discount_factor * next_q_value
            
            # Valeurs Q actuelles
            current_q1 = self.critic_1([states_seqs, actions])
            current_q2 = self.critic_2([states_seqs, actions])
            
            # Pertes des critiques (MSE)
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))
            
            # Échantillonner des actions pour l'état actuel (pour l'apprentissage de l'acteur)
            means, log_stds = self.actor(states_seqs)
            stds = tf.exp(log_stds)
            normal_dists = tfp.distributions.Normal(means, stds)
            actions_raw = normal_dists.sample()
            actions_policy = tf.tanh(actions_raw)
            
            # Calculer log-prob pour les actions de la politique
            log_probs = normal_dists.log_prob(actions_raw) - \
                        tf.math.log(1.0 - tf.square(actions_policy) + 1e-6)
            log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
            
            # Valeurs Q pour les actions de la politique
            q1 = self.critic_1([states_seqs, actions_policy])
            q2 = self.critic_2([states_seqs, actions_policy])
            q_min = tf.minimum(q1, q2)
            
            # Calcul de l'entropie
            entropy = -tf.reduce_mean(log_probs)
            
            # Perte de l'acteur = valeur Q attendue - entropie
            # Ajout d'une régularisation d'entropie supplémentaire
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_min) - self.entropy_regularization * entropy
            
            # Perte pour l'adaptation d'alpha (si activée)
            if self.train_alpha:
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
                )
            else:
                alpha_loss = tf.constant(0.0)
            
        # Calculer les gradients et mettre à jour les paramètres
        critic1_gradients = tape.gradient(critic1_loss, self.critic_1.trainable_variables)
        critic2_gradients = tape.gradient(critic2_loss, self.critic_2.trainable_variables)
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        # Appliquer le gradient clipping seulement si grad_clip_value est spécifié
        if self.grad_clip_value is not None:
            critic1_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                               for g in critic1_gradients]
            critic2_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                               for g in critic2_gradients]
            actor_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                               for g in actor_gradients]
        
        self.critic_1_optimizer.apply_gradients(zip(critic1_gradients, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic2_gradients, self.critic_2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        if self.train_alpha:
            alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
            # Appliquer le gradient clipping pour alpha aussi, seulement si grad_clip_value est spécifié
            if self.grad_clip_value is not None:
                alpha_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                                  for g in alpha_gradients]
            self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        del tape
        
        # Mettre à jour les réseaux cibles avec des mises à jour douces
        for target_param, param in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
            
        for target_param, param in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
        
        return (critic1_loss + critic2_loss) / 2.0, actor_loss, alpha_loss, entropy
        
    def train(self):
        """
        Entraîne l'agent sur un lot d'expériences du tampon de replay.
        
        Returns:
            dict: Dictionnaire contenant les métriques d'entraînement
        """
        if self.use_gru:
            # Vérifier si le tampon de séquences contient assez d'échantillons
            if len(self.memory) < self.batch_size:
                return {
                    "critic_loss": 0.0,
                    "actor_loss": 0.0,
                    "alpha_loss": 0.0,
                    "entropy": 0.0,
                    "alpha": float(self.alpha)
                }
            
            # Échantillonner un lot depuis le tampon de séquences
            states_seqs, actions, rewards, next_states_seqs, dones = self.memory.sample(self.batch_size)
            
            # Vérifier les formes
            logging.debug(f"Formes échantillonnées - states_seqs: {states_seqs.shape}, next_states_seqs: {next_states_seqs.shape}")
            
            # Normaliser les récompenses pour la stabilité
            rewards = np.clip(rewards, -50, 50)
            
            # Conversion en tenseurs
            states_seqs = tf.convert_to_tensor(states_seqs, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states_seqs = tf.convert_to_tensor(next_states_seqs, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            # Ajouter une dimension si nécessaire pour les récompenses et les indicateurs de fin
            rewards = tf.reshape(rewards, [-1, 1])
            dones = tf.reshape(dones, [-1, 1])
            
            # Effectuer une étape d'entraînement avec les séquences
            critic_loss, actor_loss, alpha_loss, entropy = self._train_step_gru(
                states_seqs, actions, rewards, next_states_seqs, dones
            )
            
        else:
            # Code original pour l'apprentissage sans GRU
            if len(self.memory) < self.batch_size:
                return {
                    "critic_loss": 0.0,
                    "actor_loss": 0.0,
                    "alpha_loss": 0.0,
                    "entropy": 0.0,
                    "alpha": float(self.alpha)
                }
            
            # Échantillonner un lot depuis le tampon de replay
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Normaliser les récompenses pour la stabilité
            rewards = np.clip(rewards, -50, 50)
            
            # Conversion en tenseurs
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            # Ajouter une dimension si nécessaire
            rewards = tf.reshape(rewards, [-1, 1])
            dones = tf.reshape(dones, [-1, 1])
            
            # Effectuer une étape d'entraînement
            critic_loss, actor_loss, alpha_loss, entropy = self._train_step(
                states, actions, rewards, next_states, dones
            )
        
        # Enregistrer les métriques
        self.critic_loss_history.append(float(critic_loss))
        self.actor_loss_history.append(float(actor_loss))
        self.alpha_loss_history.append(float(alpha_loss))
        self.entropy_history.append(float(entropy))
        
        # Récupérer la valeur d'alpha, qui peut être un tenseur TensorFlow
        try:
            alpha_value = float(self.alpha)
        except (TypeError, ValueError):
            # Si c'est un DeferredTensor ou autre type spécial, récupérer sa valeur évaluée
            alpha_value = float(tf.keras.backend.get_value(self.alpha))
        
        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha_loss": float(alpha_loss),
            "entropy": float(entropy),
            "alpha": alpha_value
        }
    
    def save(self, filepath):
        """
        Sauvegarde les poids des modèles.
        
        Args:
            filepath (str): Chemin de base pour la sauvegarde
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        self.actor.save_weights(f"{filepath}/actor.h5")
        self.critic_1.save_weights(f"{filepath}/critic_1.h5")
        self.critic_2.save_weights(f"{filepath}/critic_2.h5")
        
        # Sauvegarder alpha
        np.save(f"{filepath}/log_alpha.npy", self.log_alpha.numpy())
        
        logger.info(f"Modèles sauvegardés dans {filepath}")
    
    def load(self, filepath):
        """
        Charge les poids des modèles.
        
        Args:
            filepath (str): Chemin de base pour le chargement
        """
        self.actor.load_weights(f"{filepath}/actor.h5")
        self.critic_1.load_weights(f"{filepath}/critic_1.h5")
        self.critic_2.load_weights(f"{filepath}/critic_2.h5")
        
        # Charger alpha
        log_alpha_value = np.load(f"{filepath}/log_alpha.npy")
        self.log_alpha.assign(log_alpha_value)
        
        # Mettre à jour les réseaux cibles
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        logger.info(f"Modèles chargés depuis {filepath}")
    
    def get_metrics(self):
        """
        Retourne les métriques d'entraînement de l'agent.
        
        Returns:
            dict: Dictionnaire contenant les métriques
        """
        return {
            "critic_loss_history": self.critic_loss_history,
            "actor_loss_history": self.actor_loss_history,
            "alpha_loss_history": self.alpha_loss_history,
            "entropy_history": self.entropy_history,
            "current_alpha": float(self.alpha)
        }

    def _update_target_networks(self, tau):
        """
        Met à jour les réseaux cibles avec des mises à jour douces.
        
        Args:
            tau: Facteur pour les mises à jour douces
        """
        for target_param, param in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_param.assign(target_param * (1 - tau) + param * tau)
            
        for target_param, param in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_param.assign(target_param * (1 - tau) + param * tau)

    def _preprocess_state(self, state):
        """
        Prétraite l'état pour le rendre compatible avec le réseau.
        Gère les différents formats d'entrée et la normalisation.
        
        Args:
            state: État brut (NumPy array, Tensor ou autre)
            
        Returns:
            tf.Tensor: État prétraité prêt pour l'inférence
        """
        # Conversion explicite en float32 pour éviter les problèmes de type
        if isinstance(state, np.ndarray):
            if state.dtype == np.dtype('O'):  # Type 'object'
                logging.debug(f"Conversion d'un état de type {state.dtype} en float32")
                state = state.astype(np.float32)
                
            # Vérifier à nouveau si l'état contient des NaN après conversion
            if np.any(np.isnan(state)):
                state = np.nan_to_num(state, nan=0.0)
        elif isinstance(state, tf.Tensor) and tf.reduce_any(tf.math.is_nan(state)):
            state = tf.where(tf.math.is_nan(state), tf.zeros_like(state), state)
        
        # Préparer l'entrée du réseau selon le type d'agent (GRU ou standard)
        if self.use_gru:
            if isinstance(state, np.ndarray):
                # Si c'est un état unique, le répéter pour former une séquence
                if len(state.shape) == 1:
                    state_input = np.array([state] * self.sequence_length)
                    state_input = state_input.reshape(1, self.sequence_length, -1)
                # Si c'est déjà une séquence 2D, ajouter la dimension du batch
                elif len(state.shape) == 2 and state.shape[0] == self.sequence_length:
                    state_input = state.reshape(1, self.sequence_length, -1)
                # Si c'est déjà un batch de séquences, l'utiliser tel quel
                elif len(state.shape) == 3:
                    state_input = state
                else:
                    logging.error(f"Forme d'état incorrecte pour GRU: {state.shape}")
                    # Créer un état par défaut de la forme attendue
                    state_input = np.zeros((1, self.sequence_length, self.state_size))
                
                # Normaliser la séquence
                state_input = self._normalize_sequence_states(state_input)
                
            elif isinstance(state, tf.Tensor):
                # Ajuster la forme si nécessaire
                if len(state.shape) == 1:
                    repeated_state = tf.repeat(tf.expand_dims(state, axis=0), self.sequence_length, axis=0)
                    state_input = tf.expand_dims(repeated_state, axis=0)
                elif len(state.shape) == 2 and state.shape[0] == self.sequence_length:
                    state_input = tf.expand_dims(state, axis=0)
                elif len(state.shape) == 3:
                    state_input = state
                else:
                    logging.error(f"Forme d'état incorrecte pour GRU: {state.shape}")
                    # Créer un état par défaut de la forme attendue
                    state_input = tf.zeros((1, self.sequence_length, self.state_size))
                
                # Normaliser la séquence
                state_input = self._normalize_sequence_states(state_input)
            
            else:
                # Si ce n'est ni un tableau NumPy ni un tenseur TensorFlow
                logging.error(f"Type d'état non pris en charge: {type(state)}")
                state_input = np.zeros((1, self.sequence_length, self.state_size))
                state_input = tf.convert_to_tensor(state_input, dtype=tf.float32)
            
        else:
            # Pour un agent standard (sans GRU)
            if isinstance(state, np.ndarray):
                # Si c'est une séquence, prendre le dernier état
                if len(state.shape) > 1:
                    state_input = state[-1] if len(state.shape) == 2 else state[0, -1]
                else:
                    state_input = state
                
                # Convertir en tensor TensorFlow et ajouter la dimension du batch
                state_input = np.expand_dims(state_input, axis=0)
                state_input = tf.convert_to_tensor(state_input, dtype=tf.float32)
            
            elif isinstance(state, tf.Tensor):
                # Ajuster la forme si nécessaire
                if len(state.shape) > 1:
                    state_input = state[-1] if len(state.shape) == 2 else state[0, -1]
                    state_input = tf.expand_dims(state_input, axis=0)
                else:
                    state_input = tf.expand_dims(state, axis=0)
            
            else:
                # Si ce n'est ni un tableau NumPy ni un tenseur TensorFlow
                logging.error(f"Type d'état non pris en charge: {type(state)}")
                state_input = np.zeros((1, self.state_size))
                state_input = tf.convert_to_tensor(state_input, dtype=tf.float32)
        
        return state_input 