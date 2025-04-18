import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
import os
import datetime
from collections import deque
import random

from ai_trading.models.transformer_hybrid import create_transformer_hybrid_model
from ai_trading.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, nstep_preprocess

# Configuration du logger
logger = logging.getLogger("TransformerSACAgent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TransformerSACAgent:
    """
    Agent SAC (Soft Actor-Critic) utilisant une architecture hybride Transformer.
    
    Cette implémentation combine:
    - L'architecture Transformer pour capturer les dépendances à long terme
    - Une architecture hybride avec GRU ou LSTM pour les séries temporelles
    - L'algorithme SAC pour l'apprentissage par renforcement
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        action_bounds=(-1.0, 1.0),
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=100000,
        use_prioritized_replay=False,
        alpha=0.2,
        auto_alpha_tuning=True,
        sequence_length=20,
        n_step_returns=1,
        model_type="gru",
        embed_dim=64,
        num_heads=4,
        ff_dim=64,
        num_transformer_blocks=2,
        rnn_units=64,
        dropout_rate=0.1,
        recurrent_dropout=0.0,
        checkpoints_dir="./checkpoints/transformer_sac"
    ):
        """
        Initialise l'agent Transformer-SAC.
        
        Args:
            state_dim: Dimension de l'état (forme)
            action_dim: Dimension de l'action
            action_bounds: Limites des actions (min, max)
            actor_learning_rate: Taux d'apprentissage pour l'acteur
            critic_learning_rate: Taux d'apprentissage pour le critique
            alpha_learning_rate: Taux d'apprentissage pour le paramètre d'entropie
            gamma: Facteur d'actualisation
            tau: Taux de mise à jour des réseaux cibles
            batch_size: Taille du batch pour l'entraînement
            buffer_size: Taille du tampon de replay
            use_prioritized_replay: Utiliser le replay prioritaire
            alpha: Coefficient d'entropie
            auto_alpha_tuning: Ajuster automatiquement alpha
            sequence_length: Longueur de la séquence
            n_step_returns: Nombre d'étapes pour les retours (n-step returns)
            model_type: Type de modèle hybride ('gru' ou 'lstm')
            embed_dim: Dimension d'embedding pour le Transformer
            num_heads: Nombre de têtes d'attention
            ff_dim: Dimension du réseau feed-forward dans le Transformer
            num_transformer_blocks: Nombre de blocs Transformer à empiler
            rnn_units: Nombre d'unités dans la couche RNN
            dropout_rate: Taux de dropout
            recurrent_dropout: Taux de dropout récurrent
            checkpoints_dir: Répertoire pour les points de contrôle
        """
        # Stocker les hyperparamètres
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low, self.action_high = action_bounds
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.auto_alpha_tuning = auto_alpha_tuning
        self.sequence_length = sequence_length
        self.n_step_returns = n_step_returns
        self.model_type = model_type
        self.checkpoints_dir = checkpoints_dir
        
        # Paramètres du modèle
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        
        # Créer le répertoire des checkpoints
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=buffer_size,
                alpha=0.6,  # Priorité basée sur l'erreur TD
                beta=0.4,   # Correction des biais 
                n_step=n_step_returns,
                gamma=gamma
            )
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size=buffer_size,
                n_step=n_step_returns,
                gamma=gamma
            )
        
        # Tampon d'état pour gérer les séquences
        self.state_buffer = deque(maxlen=sequence_length)
        
        # Créer les réseaux
        self.actor_optimizer = Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = Adam(learning_rate=critic_learning_rate)
        self.alpha_optimizer = Adam(learning_rate=alpha_learning_rate)
        
        # Initialiser les réseaux
        self._init_networks()
        
        # Initialiser les historiques de perte
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_loss_history = []
        
        # Initialiser target_entropy et log_alpha si auto_alpha_tuning est activé
        if self.auto_alpha_tuning:
            self.target_entropy = -np.prod(action_dim)
            self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
            self.alpha = tf.exp(self.log_alpha)
        else:
            self.alpha = tf.constant(alpha)
        
        logger.info(f"Agent TransformerSAC initialisé avec le modèle {model_type}")
        
    def _init_networks(self):
        """
        Initialise les réseaux d'acteur et de critique.
        """
        # Forme d'entrée ajustée pour les séquences
        input_shape = (self.sequence_length, self.state_dim[-1] if isinstance(self.state_dim, tuple) else self.state_dim)
        
        # Réseau d'acteur
        self.actor = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=input_shape,
            output_dim=self.action_dim * 2,  # Moyenne et log_std pour chaque action
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length
        )
        
        # Réseaux de critique (Q1 et Q2)
        self.critic_1 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length
        )
        
        self.critic_2 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length
        )
        
        # Réseaux de critique cibles
        self.target_critic_1 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length
        )
        
        self.target_critic_2 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length
        )
        
        # Construire les modèles avec un input fictif
        dummy_state = tf.random.normal((1, input_shape[0], input_shape[1]))
        dummy_state_action = tf.random.normal((1, input_shape[0], input_shape[1] + self.action_dim))
        
        # Construire l'acteur
        _ = self.actor(dummy_state)
        
        # Construire les critiques
        _ = self.critic_1(dummy_state_action)
        _ = self.critic_2(dummy_state_action)
        _ = self.target_critic_1(dummy_state_action)
        _ = self.target_critic_2(dummy_state_action)
        
        # Synchroniser les poids des réseaux cibles avec les réseaux principaux
        self.update_target_networks(tau=1.0)
        
    def update_target_networks(self, tau=None):
        """
        Met à jour les réseaux cibles avec le taux spécifié.
        
        Args:
            tau: Taux de mise à jour (si None, utilise self.tau)
        """
        if tau is None:
            tau = self.tau
            
        # Mise à jour des poids du critique 1
        for target_weight, weight in zip(self.target_critic_1.weights, self.critic_1.weights):
            target_weight.assign(weight * tau + target_weight * (1 - tau))
            
        # Mise à jour des poids du critique 2
        for target_weight, weight in zip(self.target_critic_2.weights, self.critic_2.weights):
            target_weight.assign(weight * tau + target_weight * (1 - tau))
    
    def update_state_buffer(self, state):
        """
        Met à jour le tampon d'état avec un nouvel état.
        
        Args:
            state: Nouvel état à ajouter au tampon
        """
        self.state_buffer.append(state)
    
    def get_sequence_state(self):
        """
        Récupère l'état de séquence actuel du tampon.
        
        Returns:
            np.ndarray: Séquence d'états
        """
        # Si le tampon n'est pas rempli, répliquer le premier état
        if len(self.state_buffer) < self.sequence_length:
            padding = [self.state_buffer[0]] * (self.sequence_length - len(self.state_buffer))
            sequence = padding + list(self.state_buffer)
        else:
            sequence = list(self.state_buffer)
            
        return np.array(sequence)
    
    def reset_state_buffer(self):
        """
        Réinitialise le tampon d'état.
        """
        self.state_buffer.clear()
        
    def sample_action(self, state, evaluate=False):
        """
        Échantillonne une action à partir de la politique actuelle.
        
        Args:
            state: État actuel
            evaluate: Si True, retourne l'action moyenne (sans bruit)
            
        Returns:
            np.ndarray: Action échantillonnée
        """
        # Mettre à jour le tampon d'état
        self.update_state_buffer(state)
        
        # Obtenir la séquence d'états
        sequence_state = self.get_sequence_state()
        sequence_state = np.expand_dims(sequence_state, axis=0)  # Ajouter la dimension du batch
        
        # Prédire les paramètres de la distribution d'actions
        action_params = self.actor(sequence_state, training=False)
        
        # Extraire les moyennes et les log_stds
        means, log_stds = tf.split(action_params, 2, axis=-1)
        log_stds = tf.clip_by_value(log_stds, -20, 2)  # Éviter les valeurs extrêmes
        
        # En mode évaluation, utiliser directement la moyenne
        if evaluate:
            actions = means
        else:
            # Échantillonner à partir de la distribution normale
            stds = tf.exp(log_stds)
            normal_dist = tf.random.normal(shape=means.shape)
            actions = means + normal_dist * stds
        
        # Appliquer tanh pour limiter les actions
        actions = tf.tanh(actions)
        
        # Mettre à l'échelle les actions dans l'intervalle [action_low, action_high]
        scaled_actions = (actions + 1.0) / 2.0  # [0, 1]
        scaled_actions = scaled_actions * (self.action_high - self.action_low) + self.action_low
        
        return scaled_actions[0].numpy()  # Retirer la dimension du batch
    
    def remember(self, state, action, reward, next_state, done):
        """
        Stocke une transition dans le tampon de replay.
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indique si l'épisode est terminé
        """
        # Normaliser l'action pour qu'elle soit dans l'intervalle [-1, 1]
        normalized_action = 2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0
        
        # Stocker la transition
        self.replay_buffer.add(state, normalized_action, reward, next_state, done)
    
    def train(self):
        """
        Entraîne l'agent en utilisant un mini-batch du tampon de replay.
        
        Returns:
            dict: Historique des pertes
        """
        # Vérifier s'il y a suffisamment d'échantillons
        if len(self.replay_buffer) < self.batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": float(self.alpha),
                "q_value": 0.0
            }
        
        # Échantillonner un batch
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones_like(rewards)
            indices = None
        
        # Préparer les séquences d'états
        sequence_states = self._prepare_sequence_states(states)
        sequence_next_states = self._prepare_sequence_states(next_states)
        
        # Entraîner les critiques
        critic_loss, td_errors = self._train_critics(sequence_states, actions, rewards, sequence_next_states, dones, weights)
        
        # Mettre à jour les priorités si nécessaire
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.numpy())
        
        # Entraîner l'acteur et alpha
        actor_loss, alpha_loss, q_value = self._train_actor_and_alpha(sequence_states, weights)
        
        # Mettre à jour les réseaux cibles
        self.update_target_networks()
        
        # Enregistrer les pertes
        self.actor_loss_history.append(float(actor_loss))
        self.critic_loss_history.append(float(critic_loss))
        self.alpha_loss_history.append(float(alpha_loss))
        
        return {
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(self.alpha),
            "q_value": float(q_value)
        }
    
    def _prepare_sequence_states(self, states):
        """
        Prépare les séquences d'états pour l'entraînement.
        
        Args:
            states: Batch d'états
            
        Returns:
            tf.Tensor: Batch de séquences d'états
        """
        batch_size = states.shape[0]
        
        # Initialiser les séquences avec des zéros
        sequences = tf.zeros((batch_size, self.sequence_length, self.state_dim[-1] if isinstance(self.state_dim, tuple) else self.state_dim))
        
        # Remplir chaque séquence avec l'état à la fin
        sequences = tf.tensor_scatter_nd_update(
            sequences,
            indices=[[i, self.sequence_length - 1] for i in range(batch_size)],
            updates=states
        )
        
        return sequences
    
    @tf.function
    def _train_critics(self, states, actions, rewards, next_states, dones, weights):
        """
        Entraîne les réseaux critiques.
        
        Args:
            states: Batch d'états
            actions: Batch d'actions
            rewards: Batch de récompenses
            next_states: Batch d'états suivants
            dones: Batch d'indicateurs de fin
            weights: Poids d'importance-sampling pour le replay prioritaire
            
        Returns:
            tuple: (critique_loss, td_errors)
        """
        with tf.GradientTape(persistent=True) as tape:
            # Obtenir les actions suivantes et les log_probs à partir de l'acteur
            next_action_params = self.actor(next_states, training=False)
            next_means, next_log_stds = tf.split(next_action_params, 2, axis=-1)
            next_log_stds = tf.clip_by_value(next_log_stds, -20, 2)
            next_stds = tf.exp(next_log_stds)
            
            # Échantillonner à partir de la distribution normale
            normal_dist = tf.random.normal(shape=next_means.shape)
            next_actions_raw = next_means + normal_dist * next_stds
            next_actions = tf.tanh(next_actions_raw)
            
            # Calculer les log_probs
            log_probs = self._log_probs(next_actions_raw, next_actions, next_log_stds)
            
            # Préparer les entrées pour les critiques (concaténer état et action)
            next_state_actions_1 = tf.concat([next_states, next_actions], axis=-1)
            next_state_actions_2 = tf.concat([next_states, next_actions], axis=-1)
            
            # Obtenir les valeurs Q des réseaux cibles
            next_q1 = self.target_critic_1(next_state_actions_1, training=False)
            next_q2 = self.target_critic_2(next_state_actions_2, training=False)
            
            # Prendre le minimum des deux valeurs Q
            next_q_min = tf.minimum(next_q1, next_q2)
            
            # Calculer la cible de valeur Q (avec terme d'entropie)
            next_q_target = next_q_min - self.alpha * log_probs
            q_target = rewards + (1 - dones) * self.gamma * next_q_target
            
            # Calculer les valeurs Q actuelles
            state_actions_1 = tf.concat([states, actions], axis=-1)
            state_actions_2 = tf.concat([states, actions], axis=-1)
            
            q1 = self.critic_1(state_actions_1, training=True)
            q2 = self.critic_2(state_actions_2, training=True)
            
            # Calculer les pertes des critiques
            q1_loss = 0.5 * tf.reduce_mean(weights * tf.square(q_target - q1))
            q2_loss = 0.5 * tf.reduce_mean(weights * tf.square(q_target - q2))
            critic_loss = q1_loss + q2_loss
            
            # Calculer les erreurs TD pour le replay prioritaire
            td_errors = tf.abs(q_target - q1)
        
        # Appliquer les gradients aux critiques
        critic_1_grads = tape.gradient(q1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(q2_loss, self.critic_2.trainable_variables)
        
        self.critic_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        
        del tape
        
        return critic_loss, td_errors
    
    @tf.function
    def _train_actor_and_alpha(self, states, weights):
        """
        Entraîne le réseau d'acteur et ajuste alpha.
        
        Args:
            states: Batch d'états
            weights: Poids d'importance-sampling
            
        Returns:
            tuple: (actor_loss, alpha_loss, q_value)
        """
        with tf.GradientTape() as tape:
            # Obtenir les paramètres de la distribution d'actions
            action_params = self.actor(states, training=True)
            means, log_stds = tf.split(action_params, 2, axis=-1)
            log_stds = tf.clip_by_value(log_stds, -20, 2)
            stds = tf.exp(log_stds)
            
            # Échantillonner les actions à partir de la distribution normale
            normal_dist = tf.random.normal(shape=means.shape)
            actions_raw = means + normal_dist * stds
            actions = tf.tanh(actions_raw)
            
            # Calculer les log_probs
            log_probs = self._log_probs(actions_raw, actions, log_stds)
            
            # Préparer les entrées pour les critiques
            state_actions = tf.concat([states, actions], axis=-1)
            
            # Calculer la valeur Q
            q1 = self.critic_1(state_actions, training=False)
            q2 = self.critic_2(state_actions, training=False)
            q = tf.minimum(q1, q2)
            
            # Calculer la perte de l'acteur (maximiser Q - alpha * log_prob)
            actor_loss = tf.reduce_mean(weights * (self.alpha * log_probs - q))
        
        # Appliquer les gradients à l'acteur
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Ajuster alpha si auto_alpha_tuning est activé
        if self.auto_alpha_tuning:
            with tf.GradientTape() as tape:
                self.alpha = tf.exp(self.log_alpha)
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
                )
            
            # Appliquer les gradients à alpha
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        else:
            alpha_loss = tf.constant(0.0)
        
        return actor_loss, alpha_loss, tf.reduce_mean(q)
    
    def _log_probs(self, raw_actions, actions, log_stds):
        """
        Calcule les logarithmes des probabilités.
        
        Args:
            raw_actions: Actions avant tanh
            actions: Actions après tanh
            log_stds: Logarithmes des écarts-types
            
        Returns:
            tf.Tensor: Logarithmes des probabilités
        """
        # Calculer log_probs pour la distribution normale
        log_probs_normal = -0.5 * (
            tf.square((raw_actions - 0.0) / (tf.exp(log_stds) + 1e-8))
            + 2.0 * log_stds
            + tf.math.log(2.0 * np.pi)
        )
        log_probs_normal = tf.reduce_sum(log_probs_normal, axis=-1, keepdims=True)
        
        # Correction pour la transformation tanh
        log_probs_tanh = tf.reduce_sum(
            tf.math.log(1 - tf.square(actions) + 1e-8), axis=-1, keepdims=True
        )
        
        return log_probs_normal - log_probs_tanh
    
    def save_models(self, suffix=""):
        """
        Sauvegarde les modèles de l'agent.
        
        Args:
            suffix: Suffixe pour les noms de fichiers
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.checkpoints_dir, f"{timestamp}_{suffix}")
        os.makedirs(path, exist_ok=True)
        
        self.actor.save_weights(os.path.join(path, "actor"))
        self.critic_1.save_weights(os.path.join(path, "critic_1"))
        self.critic_2.save_weights(os.path.join(path, "critic_2"))
        self.target_critic_1.save_weights(os.path.join(path, "target_critic_1"))
        self.target_critic_2.save_weights(os.path.join(path, "target_critic_2"))
        
        logger.info(f"Modèles sauvegardés dans {path}")
        
        return path
    
    def load_models(self, path):
        """
        Charge les modèles de l'agent.
        
        Args:
            path: Chemin vers les modèles sauvegardés
        """
        try:
            self.actor.load_weights(os.path.join(path, "actor"))
            self.critic_1.load_weights(os.path.join(path, "critic_1"))
            self.critic_2.load_weights(os.path.join(path, "critic_2"))
            self.target_critic_1.load_weights(os.path.join(path, "target_critic_1"))
            self.target_critic_2.load_weights(os.path.join(path, "target_critic_2"))
            
            logger.info(f"Modèles chargés depuis {path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            return False 