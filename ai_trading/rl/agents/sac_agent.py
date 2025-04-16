import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp
from collections import deque
import random

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


class SACAgent:
    """
    Agent Soft Actor-Critic (SAC), une méthode d'apprentissage par renforcement basée sur l'entropie maximale.
    Implémente l'algorithme SAC avec adaptation automatique du paramètre d'entropie.
    """
    
    def __init__(
        self, 
        state_size,
        action_size=1,  # Par défaut pour l'espace Box(low=-1, high=1, shape=(1,))
        action_bounds=(-1, 1),  # Limites de l'espace d'action
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        discount_factor=0.99,
        tau=0.005,  # Facteur de mise à jour douce des réseaux cibles
        batch_size=256,
        buffer_size=100000,
        hidden_size=256,
        train_alpha=True,  # Si True, le paramètre alpha (température) est appris
        target_entropy=None,  # Entropie cible, si None, sera fixée à -action_size
        grad_clip_value=1.0,  # Valeur maximale pour le gradient clipping
        entropy_regularization=0.001  # Coefficient pour la régularisation d'entropie supplémentaire
    ):
        """
        Initialise l'agent Soft Actor-Critic.
        
        Args:
            state_size: Dimension de l'espace d'état
            action_size: Dimension de l'espace d'action
            action_bounds: Tuple (min, max) des limites de l'espace d'action
            actor_learning_rate: Taux d'apprentissage pour le réseau de politique
            critic_learning_rate: Taux d'apprentissage pour les réseaux critiques
            alpha_learning_rate: Taux d'apprentissage pour le paramètre d'entropie
            discount_factor: Facteur d'actualisation pour les récompenses futures
            tau: Facteur pour les mises à jour douces des réseaux cibles
            batch_size: Taille des lots pour l'entraînement
            buffer_size: Taille du tampon de replay
            hidden_size: Nombre de neurones dans les couches cachées
            train_alpha: Si True, le paramètre d'entropie est appris automatiquement
            target_entropy: Entropie cible pour l'adaptation de alpha
            grad_clip_value: Valeur maximale pour le gradient clipping
            entropy_regularization: Coefficient pour la régularisation d'entropie supplémentaire
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low, self.action_high = action_bounds
        self.discount_factor = discount_factor
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.train_alpha = train_alpha
        self.grad_clip_value = grad_clip_value
        self.entropy_regularization = entropy_regularization
        
        # Définir l'entropie cible (par défaut: -dim(A))
        if target_entropy is None:
            self.target_entropy = -action_size
        else:
            self.target_entropy = target_entropy
        
        # Vérifier si l'initialisation du tampon de replay doit être sautée
        # (peut être utilisé par les classes dérivées qui implémentent leur propre tampon)
        self._skip_buffer_init = getattr(self, '_skip_buffer_init', False)
        
        # Initialiser le tampon de replay (sauf si spécifié autrement)
        if not self._skip_buffer_init:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Paramètre d'entropie (température)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_optimizer = Adam(learning_rate=alpha_learning_rate)
        
        # Construire les réseaux
        # Acteur (politique)
        self.actor = self._build_actor(state_size, action_size, hidden_size)
        self.actor_optimizer = Adam(learning_rate=actor_learning_rate)
        
        # Critiques (Q-functions)
        self.critic_1 = self._build_critic(state_size, action_size, hidden_size)
        self.critic_2 = self._build_critic(state_size, action_size, hidden_size)
        self.critic_1_target = self._build_critic(state_size, action_size, hidden_size)
        self.critic_2_target = self._build_critic(state_size, action_size, hidden_size)
        
        # Copier les poids des critiques vers les cibles
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())
        
        self.critic_optimizer_1 = Adam(learning_rate=critic_learning_rate)
        self.critic_optimizer_2 = Adam(learning_rate=critic_learning_rate)
        
        # Métriques
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []
        
        logger.info(
            f"Agent SAC initialisé: state_size={state_size}, action_size={action_size}, "
            f"train_alpha={train_alpha}, target_entropy={self.target_entropy}"
        )
    
    def _build_actor(self, state_size, action_size, hidden_size):
        """
        Construit le réseau de politique (acteur) qui génère une distribution de probabilité sur les actions.
        
        Returns:
            Model: Modèle Keras pour l'acteur
        """
        inputs = Input(shape=(state_size,))
        x = Dense(hidden_size, activation='relu')(inputs)
        x = Dense(hidden_size, activation='relu')(x)
        
        # Sortie pour la moyenne de la distribution gaussienne
        mean = Dense(action_size, activation='tanh')(x)
        
        # Sortie pour le log de l'écart-type (variance) de la distribution
        log_std = Dense(action_size, activation='linear')(x)
        
        # Contraindre log_std à un intervalle raisonnable
        log_std = Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        # Créer le modèle
        model = Model(inputs=inputs, outputs=[mean, log_std])
        
        return model
    
    def _build_critic(self, state_size, action_size, hidden_size):
        """
        Construit un réseau critique (Q-function) qui prédit la valeur Q d'une paire état-action.
        
        Returns:
            Model: Modèle Keras pour le critique
        """
        state_input = Input(shape=(state_size,))
        action_input = Input(shape=(action_size,))
        
        # Concaténer l'état et l'action
        merged = Concatenate()([state_input, action_input])
        
        x = Dense(hidden_size, activation='relu')(merged)
        x = Dense(hidden_size, activation='relu')(x)
        q_value = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input, action_input], outputs=q_value)
        
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
        # S'assurer que l'action est un numpy array
        if isinstance(action, (int, float)):
            action = np.array([action])
            
        # Ajouter l'expérience au tampon
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def act(self, state, deterministic=False):
        """
        Sélectionne une action selon la politique actuelle.
        
        Args:
            state: État actuel
            deterministic (bool): Si True, retourne l'action moyenne sans bruit
            
        Returns:
            numpy.array: Action sélectionnée
        """
        # S'assurer que l'état est au bon format
        state = np.reshape(state, [1, self.state_size]).astype(np.float32)
        
        # Obtenir la moyenne et l'écart-type de la distribution d'actions
        mean, log_std = self.actor.predict(state, verbose=0)
        
        if deterministic:
            # Mode déterministe : retourner la moyenne
            action = mean
        else:
            # Mode stochastique : échantillonner depuis la distribution
            std = tf.exp(log_std)
            normal_dist = tfp.distributions.Normal(mean, std)
            
            # Échantillonner et appliquer tanh pour borner entre -1 et 1
            action = normal_dist.sample()
            action = tf.tanh(action)
        
        # Mettre à l'échelle l'action à l'intervalle correct
        # Vérifier si l'action est un tenseur TensorFlow ou un tableau NumPy
        if hasattr(action, 'numpy'):
            action_numpy = action.numpy()
        else:
            action_numpy = action  # Déjà un tableau NumPy
            
        scaled_action = self._scale_action(action_numpy)
        
        return scaled_action[0]  # Retirer la dimension du lot
    
    def _scale_action(self, action):
        """
        Met à l'échelle l'action de [-1, 1] vers [low, high].
        
        Args:
            action (numpy.array): Action normalisée entre -1 et 1
            
        Returns:
            numpy.array: Action mise à l'échelle
        """
        return 0.5 * (action + 1.0) * (self.action_high - self.action_low) + self.action_low
    
    def _unscale_action(self, scaled_action):
        """
        Convertit une action mise à l'échelle vers la plage [-1, 1].
        
        Args:
            scaled_action (numpy.array): Action mise à l'échelle
            
        Returns:
            numpy.array: Action normalisée entre -1 et 1
        """
        return 2.0 * (scaled_action - self.action_low) / (self.action_high - self.action_low) - 1.0

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
            next_q1 = self.critic_1_target([next_states, next_actions])
            next_q2 = self.critic_2_target([next_states, next_actions])
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
        
        # Appliquer le gradient clipping
        critic1_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                           for g in critic1_gradients]
        critic2_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                           for g in critic2_gradients]
        actor_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                         for g in actor_gradients]
        
        self.critic_optimizer_1.apply_gradients(zip(critic1_gradients, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(critic2_gradients, self.critic_2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        if self.train_alpha:
            alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
            # Appliquer le gradient clipping pour alpha aussi
            alpha_gradients = [tf.clip_by_norm(g, self.grad_clip_value) if g is not None else g 
                             for g in alpha_gradients]
            self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        del tape
        
        # Mettre à jour les réseaux cibles avec des mises à jour douces
        for target_param, param in zip(self.critic_1_target.variables, self.critic_1.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
            
        for target_param, param in zip(self.critic_2_target.variables, self.critic_2.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
        
        return (critic1_loss + critic2_loss) / 2.0, actor_loss, alpha_loss, entropy
        
    def train(self):
        """
        Entraîne l'agent sur un lot d'expériences du tampon de replay.
        
        Returns:
            dict: Dictionnaire contenant les métriques d'entraînement
        """
        if len(self.replay_buffer) < self.batch_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "entropy": 0.0,
                "alpha": float(self.alpha)
            }
        
        # Échantillonner un lot depuis le tampon de replay
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Normaliser les récompenses pour la stabilité
        rewards = np.clip(rewards, -50, 50)
        
        # Conversion en tenseurs
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Ajouter une dimension si nécessaire
        rewards = tf.expand_dims(rewards, axis=1)
        dones = tf.expand_dims(dones, axis=1)
        
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
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())
        
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