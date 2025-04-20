import datetime
import logging
import os
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam

from ai_trading.models.transformer_hybrid import create_transformer_hybrid_model
from ai_trading.rl.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

# Configuration du logger
logger = logging.getLogger("TransformerSACAgent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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
        checkpoints_dir="./checkpoints/transformer_sac",
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
                beta=0.4,  # Correction des biais
                n_step=n_step_returns,
                gamma=gamma,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size=buffer_size, n_step=n_step_returns, gamma=gamma
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
        input_shape = (
            self.sequence_length,
            self.state_dim[-1] if isinstance(self.state_dim, tuple) else self.state_dim,
        )

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
            sequence_length=self.sequence_length,
        )

        # Réseaux de critique (Q1 et Q2)
        self.critic_1 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(
                input_shape[0],
                input_shape[1] + self.action_dim,
            ),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        )

        self.critic_2 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(
                input_shape[0],
                input_shape[1] + self.action_dim,
            ),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        )

        # Réseaux de critique cibles
        self.target_critic_1 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(
                input_shape[0],
                input_shape[1] + self.action_dim,
            ),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        )

        self.target_critic_2 = create_transformer_hybrid_model(
            model_type=self.model_type,
            input_shape=(
                input_shape[0],
                input_shape[1] + self.action_dim,
            ),  # state + action
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        )

        # Construire les modèles avec un input fictif
        dummy_state = tf.random.normal((1, input_shape[0], input_shape[1]))
        dummy_state_action = tf.random.normal(
            (1, input_shape[0], input_shape[1] + self.action_dim)
        )

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
        for target_weight, weight in zip(
            self.target_critic_1.weights, self.critic_1.weights
        ):
            target_weight.assign(weight * tau + target_weight * (1 - tau))

        # Mise à jour des poids du critique 2
        for target_weight, weight in zip(
            self.target_critic_2.weights, self.critic_2.weights
        ):
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
            padding = [self.state_buffer[0]] * (
                self.sequence_length - len(self.state_buffer)
            )
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
        Échantillonne une action à partir de l'état actuel.

        Args:
            state: État actuel
            evaluate: Si True, l'action est déterministe

        Returns:
            np.ndarray: Action échantillonnée
        """
        # S'assurer que l'état est un array numpy
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        # Vérifier s'il y a des NaN dans l'état et les remplacer par zéro
        if np.isnan(state).any():
            state = np.nan_to_num(state, nan=0.0)

        # Mettre à jour le tampon d'état
        self.update_state_buffer(state)

        # Obtenir l'état de séquence
        sequence_state = self.get_sequence_state()

        # Convertir en tensor et ajouter dimension batch
        sequence_state = tf.convert_to_tensor(sequence_state, dtype=tf.float32)
        sequence_state = tf.expand_dims(
            sequence_state, axis=0
        )  # Ajouter dimension batch

        # Obtenir les paramètres de l'action depuis l'acteur
        action_params = self.actor(sequence_state, training=False)
        mean, log_std = tf.split(action_params, 2, axis=-1)

        # Si en mode d'évaluation, retourner simplement la moyenne
        if evaluate:
            actions = tf.tanh(mean)
        else:
            # Échantillonner à partir de la distribution normale
            log_std = tf.clip_by_value(log_std, -20, 2)
            std = tf.exp(log_std)

            normal_noise = tf.random.normal(shape=mean.shape)
            raw_actions = mean + normal_noise * std
            actions = tf.tanh(raw_actions)

        # Convertir en numpy et retirer dimension batch
        actions = actions.numpy().squeeze(0)

        # Vérifier s'il y a des NaN dans les actions et les remplacer par zéro
        if np.isnan(actions).any():
            actions = np.nan_to_num(actions, nan=0.0)

        return actions

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
        normalized_action = (
            2.0 * (action - self.action_low) / (self.action_high - self.action_low)
            - 1.0
        )

        # Stocker la transition
        self.replay_buffer.add(state, normalized_action, reward, next_state, done)

    def train(self, batch_size=64):
        """
        Entraîne l'agent sur un mini-batch d'expériences.

        Args:
            batch_size: Taille du mini-batch

        Returns:
            dict: Statistiques d'entraînement
        """
        # Vérifier s'il y a assez d'échantillons
        if len(self.replay_buffer) < batch_size:
            return {"actor_loss": 0, "critic_loss": 0, "alpha_loss": 0}

        # Récupérer un mini-batch d'expériences
        if self.use_prioritized_replay:
            batch, indices, weights = self.replay_buffer.sample(batch_size)
            s, a, r, s_, d = batch
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            batch = self.replay_buffer.sample(batch_size)
            s, a, r, s_, d = batch
            weights = tf.ones(batch_size, dtype=tf.float32)

        # Préparer les états de séquence pour chaque exemple dans le mini-batch
        seq_states = []
        next_seq_states = []

        # Traiter chaque exemple individuellement
        for i in range(batch_size):
            # Mettre à jour le tampon d'état temporaire
            for j in range(self.sequence_length - 1, -1, -1):
                self.state_buffer[j] = (
                    np.zeros_like(self.state_buffer[0]) if j > 0 else s[i]
                )

            # Obtenir l'état de séquence actuel
            seq_states.append(np.array(self.state_buffer))

            # Mettre à jour le tampon d'état temporaire pour l'état suivant
            for j in range(self.sequence_length - 1, -1, -1):
                self.state_buffer[j] = (
                    np.zeros_like(self.state_buffer[0]) if j > 0 else s_[i]
                )

            # Obtenir l'état de séquence suivant
            next_seq_states.append(np.array(self.state_buffer))

        seq_states = np.array(seq_states)
        next_seq_states = np.array(next_seq_states)

        # Convertir en tensors TensorFlow
        seq_states = tf.convert_to_tensor(seq_states, dtype=tf.float32)
        next_seq_states = tf.convert_to_tensor(next_seq_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(a, dtype=tf.float32)
        rewards = tf.convert_to_tensor(r, dtype=tf.float32)
        dones = tf.convert_to_tensor(d, dtype=tf.float32)

        # Entraîner les réseaux critiques
        critic_loss, td_errors = self._train_critics(
            seq_states, actions, rewards, next_seq_states, dones, weights
        )

        # Mettre à jour les priorités si on utilise le replay prioritisé
        if self.use_prioritized_replay:
            self.replay_buffer.update_priorities(
                indices, np.abs(td_errors.numpy()) + 1e-6
            )

        # Entraîner l'acteur et alpha
        actor_loss, alpha_loss, log_probs = self._train_actor_and_alpha(seq_states)

        # Mettre à jour les réseaux cibles
        self.update_target_networks()

        # Suivre les métriques
        self.metrics["critic_loss"].append(critic_loss.numpy())
        self.metrics["actor_loss"].append(actor_loss.numpy())
        self.metrics["alpha_loss"].append(alpha_loss.numpy())
        self.metrics["alpha"].append(self.alpha.numpy())
        self.metrics["log_probs"].append(tf.reduce_mean(log_probs).numpy())

        return {
            "critic_loss": critic_loss.numpy(),
            "actor_loss": actor_loss.numpy(),
            "alpha_loss": alpha_loss.numpy(),
            "alpha": self.alpha.numpy(),
            "log_probs": tf.reduce_mean(log_probs).numpy(),
        }

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
        # Calculer l'alpha actuel en dehors des blocs GradientTape
        alpha = tf.exp(self.log_alpha) if self.auto_alpha_tuning else self.alpha

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
            batch_size = tf.shape(next_states)[0]
            seq_length = tf.shape(next_states)[1]

            # Ajuster la forme de next_actions pour correspondre à la forme des états
            next_actions_reshaped = tf.reshape(next_actions, [batch_size, 1, -1])
            next_actions_expanded = tf.repeat(next_actions_reshaped, seq_length, axis=1)

            next_state_actions = tf.concat(
                [next_states, next_actions_expanded], axis=-1
            )

            # Obtenir les valeurs Q des réseaux cibles
            next_q1 = self.target_critic_1(next_state_actions, training=False)
            next_q2 = self.target_critic_2(next_state_actions, training=False)

            # Prendre le minimum des deux valeurs Q
            next_q_min = tf.minimum(next_q1, next_q2)

            # Calculer la cible de valeur Q (avec terme d'entropie)
            next_q_target = next_q_min - alpha * log_probs
            q_target = rewards + (1 - dones) * self.gamma * next_q_target

            # Calculer les valeurs Q actuelles
            # Ajuster la forme de actions pour correspondre à la forme des états
            actions_reshaped = tf.reshape(actions, [batch_size, 1, -1])
            actions_expanded = tf.repeat(actions_reshaped, seq_length, axis=1)

            state_actions = tf.concat([states, actions_expanded], axis=-1)

            q1 = self.critic_1(state_actions, training=True)
            q2 = self.critic_2(state_actions, training=True)

            # Calculer les pertes des critiques
            td_errors1 = q_target - q1
            td_errors2 = q_target - q2
            critic_loss1 = tf.reduce_mean(weights * tf.square(td_errors1))
            critic_loss2 = tf.reduce_mean(weights * tf.square(td_errors2))
            critic_loss = critic_loss1 + critic_loss2

        # Calculer les gradients et appliquer à critic_1
        critic1_gradients = tape.gradient(
            critic_loss1, self.critic_1.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic1_gradients, self.critic_1.trainable_variables)
        )

        # Calculer les gradients et appliquer à critic_2
        critic2_gradients = tape.gradient(
            critic_loss2, self.critic_2.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic2_gradients, self.critic_2.trainable_variables)
        )

        del tape

        # Retourner la perte combinée des critiques et les erreurs TD pour le replay prioritaire
        return critic_loss, tf.abs(td_errors1)

    def _train_actor_and_alpha(self, states):
        """
        Entraîne l'acteur et ajuste le paramètre alpha.

        Args:
            states: États de séquence

        Returns:
            tuple: (actor_loss, alpha_loss, log_probs)
        """
        with tf.GradientTape() as tape:
            # Échantillonner des actions et calculer log_probs
            action_params = self.actor(states, training=True)
            mean, log_std = tf.split(action_params, 2, axis=-1)
            log_std = tf.clip_by_value(log_std, -20, 2)
            std = tf.exp(log_std)

            # Échantillonner à partir de la distribution normale
            normal_dist = tf.random.normal(shape=mean.shape)
            raw_actions = mean + normal_dist * std
            actions = tf.tanh(raw_actions)

            # Calculer les log_probs
            log_probs = self._log_probs(raw_actions, actions, log_std)

            # Préparer les entrées pour les critiques
            batch_size = tf.shape(states)[0]
            seq_length = tf.shape(states)[1]

            # Ajuster la forme de actions pour correspondre à la forme des états
            actions_reshaped = tf.reshape(actions, [batch_size, 1, -1])
            actions_expanded = tf.repeat(actions_reshaped, seq_length, axis=1)

            state_actions = tf.concat([states, actions_expanded], axis=-1)

            # Calculer la valeur Q
            q1 = self.critic_1(state_actions, training=False)
            q2 = self.critic_2(state_actions, training=False)
            q = tf.minimum(q1, q2)

            # Calculer la perte de l'acteur (maximiser Q - alpha * log_prob)
            actor_loss = tf.reduce_mean(log_probs * (q - self.alpha * log_probs))

        # Appliquer les gradients à l'acteur
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )

        # Ajuster alpha si auto_alpha_tuning est activé
        if self.auto_alpha_tuning:
            with tf.GradientTape() as tape:
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
                )

            # Appliquer les gradients à alpha
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

            # Mettre à jour self.alpha après l'optimisation
            self.alpha = tf.exp(self.log_alpha)
        else:
            alpha_loss = tf.constant(0.0)

        return actor_loss, alpha_loss, log_probs

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
