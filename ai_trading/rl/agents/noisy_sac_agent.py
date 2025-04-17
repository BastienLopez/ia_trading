import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

from ai_trading.rl.agents.sac_agent import SACAgent, ReplayBuffer
from ai_trading.rl.agents.layers.noisy_dense import NoisyDense

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class NoisySACAgent(SACAgent):
    """
    Agent Soft Actor-Critic (SAC) avec exploration paramétrique via Noisy Networks.
    Cette implémentation remplace les couches Dense standard par des couches NoisyDense
    qui incorporent l'exploration directement dans les poids et biais du réseau.
    """
    
    def __init__(
        self, 
        state_size,
        action_size=1,
        action_bounds=(-1, 1),
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        discount_factor=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=100000,
        hidden_size=256,
        train_alpha=True,
        target_entropy=None,
        sigma_init=0.5  # Valeur initiale pour le paramètre sigma des couches bruitées
    ):
        """
        Initialise l'agent SAC avec des réseaux bruités.
        
        Args:
            state_size (int): Taille de l'espace d'état
            action_size (int): Dimension de l'espace d'action continue
            action_bounds (tuple): Bornes min et max de l'action (min, max)
            actor_learning_rate (float): Taux d'apprentissage pour l'acteur
            critic_learning_rate (float): Taux d'apprentissage pour le critique
            alpha_learning_rate (float): Taux d'apprentissage pour le paramètre d'entropie
            discount_factor (float): Facteur d'actualisation pour les récompenses futures
            tau (float): Taux pour les mises à jour douces
            batch_size (int): Taille du lot pour l'entraînement
            buffer_size (int): Taille du tampon de replay
            hidden_size (int): Taille des couches cachées dans les réseaux
            train_alpha (bool): Si True, adapte automatiquement le coefficient d'entropie
            target_entropy (float): Entropie cible pour l'adaptation automatique d'alpha
            sigma_init (float): Valeur initiale pour le paramètre sigma des couches bruitées
        """
        # Initialisation des attributs spécifiques avant l'appel à super()
        self.sigma_init = sigma_init
        
        # Appel au constructeur parent (on ne construit pas encore les réseaux)
        super(NoisySACAgent, self).__init__(
            state_size=state_size,
            action_size=action_size,
            action_bounds=action_bounds,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            alpha_learning_rate=alpha_learning_rate,
            discount_factor=discount_factor,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            hidden_size=hidden_size,
            train_alpha=train_alpha,
            target_entropy=target_entropy
        )
        
        # Reconstruire les réseaux spécifiques de NoisySAC
        # (cela remplace les réseaux construits dans le constructeur parent)
        self.actor = self._build_actor(state_size, action_size, hidden_size)
        self.critic_1 = self._build_critic(state_size, action_size, hidden_size)
        self.critic_2 = self._build_critic(state_size, action_size, hidden_size)
        self.critic_1_target = self._build_critic(state_size, action_size, hidden_size)
        self.critic_2_target = self._build_critic(state_size, action_size, hidden_size)
        
        # Copier les poids des critiques vers les cibles
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())
        
        # Réinitialiser les optimiseurs avec les paramètres des nouveaux réseaux
        self.actor_optimizer = Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer_1 = Adam(learning_rate=critic_learning_rate)
        self.critic_optimizer_2 = Adam(learning_rate=critic_learning_rate)
        
        logger.info(
            f"Agent NoisySAC initialisé: state_size={state_size}, action_size={action_size}, "
            f"sigma_init={sigma_init}, train_alpha={train_alpha}, target_entropy={self.target_entropy}"
        )
    
    def _build_actor(self, state_size, action_size, hidden_size):
        """
        Construit le réseau de politique (acteur) qui génère une distribution de probabilité sur les actions.
        Utilise des couches NoisyDense au lieu des couches Dense standard.
        
        Returns:
            Model: Modèle Keras pour l'acteur
        """
        inputs = Input(shape=(state_size,))
        x = NoisyDense(hidden_size, activation='relu', sigma_init=self.sigma_init)(inputs)
        x = NoisyDense(hidden_size, activation='relu', sigma_init=self.sigma_init)(x)
        
        # Sortie pour la moyenne de la distribution gaussienne
        mean = NoisyDense(action_size, activation='tanh', sigma_init=self.sigma_init)(x)
        
        # Sortie pour le log de l'écart-type (variance) de la distribution
        log_std = NoisyDense(action_size, activation='linear', sigma_init=self.sigma_init)(x)
        
        # Contraindre log_std à un intervalle raisonnable
        log_std = Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        # Créer le modèle
        model = Model(inputs=inputs, outputs=[mean, log_std])
        
        return model
    
    def _build_critic(self, state_size, action_size, hidden_size):
        """
        Construit un réseau critique (Q-function) qui prédit la valeur Q d'une paire état-action.
        Utilise des couches NoisyDense au lieu des couches Dense standard.
        
        Returns:
            Model: Modèle Keras pour le critique
        """
        state_input = Input(shape=(state_size,))
        action_input = Input(shape=(action_size,))
        
        # Concaténer l'état et l'action
        merged = Concatenate()([state_input, action_input])
        
        x = NoisyDense(hidden_size, activation='relu', sigma_init=self.sigma_init)(merged)
        x = NoisyDense(hidden_size, activation='relu', sigma_init=self.sigma_init)(x)
        q_value = NoisyDense(1, activation='linear', sigma_init=self.sigma_init)(x)
        
        model = Model(inputs=[state_input, action_input], outputs=q_value)
        
        return model
    
    def act(self, state, deterministic=False):
        """
        Sélectionne une action selon la politique actuelle.
        Le déterminisme est géré différemment pour NoisySAC car l'exploration 
        est intégrée directement dans les poids du réseau.
        
        Args:
            state: État actuel
            deterministic (bool): Si True, désactive le bruit dans les couches NoisyDense
            
        Returns:
            numpy.array: Action sélectionnée
        """
        # S'assurer que l'état est au bon format
        state = np.reshape(state, [1, self.state_size]).astype(np.float32)
        
        # Obtenir la moyenne et l'écart-type de la distribution d'actions
        # Pour le mode déterministe, on passe training=False aux couches NoisyDense
        mean, log_std = self.actor(state, training=not deterministic)
        
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
        if hasattr(action, 'numpy'):
            action_numpy = action.numpy()
        else:
            action_numpy = action  # Déjà un tableau NumPy
            
        scaled_action = self._scale_action(action_numpy)
        
        return scaled_action[0]  # Retirer la dimension du lot
    
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """
        Effectue une étape d'entraînement pour tous les réseaux.
        Cette méthode est partiellement réécrite pour gérer correctement
        les couches bruitées pendant l'entraînement.
        
        Args:
            states: Lot d'états
            actions: Lot d'actions
            rewards: Lot de récompenses
            next_states: Lot d'états suivants
            dones: Lot d'indicateurs de fin d'épisode
            
        Returns:
            tuple: (critic_loss, actor_loss, alpha_loss, entropy)
        """
        # La méthode de formation est similaire à celle de la classe parente,
        # mais avec le paramètre training=True explicitement passé aux appels de réseau
        with tf.GradientTape(persistent=True) as tape:
            # Échantillonner des actions pour l'état suivant avec bruit activé
            next_means, next_log_stds = self.actor(next_states, training=True)
            next_stds = tf.exp(next_log_stds)
            next_normal_dists = tfp.distributions.Normal(next_means, next_stds)
            next_actions_raw = next_normal_dists.sample()
            next_actions = tf.tanh(next_actions_raw)
            
            # Calculer log-prob pour les actions suivantes
            log_probs_next = next_normal_dists.log_prob(next_actions_raw) - \
                          tf.math.log(1.0 - tf.square(next_actions) + 1e-6)
            log_probs_next = tf.reduce_sum(log_probs_next, axis=1, keepdims=True)
            
            # Valeurs Q cibles pour l'apprentissage du critique (avec bruit)
            next_q1 = self.critic_1_target([next_states, next_actions], training=True)
            next_q2 = self.critic_2_target([next_states, next_actions], training=True)
            next_q_min = tf.minimum(next_q1, next_q2)
            
            # Soustraction du terme d'entropie pour SAC
            next_q_value = next_q_min - self.alpha * log_probs_next
            target_q = rewards + (1 - dones) * self.discount_factor * next_q_value
            
            # Valeurs Q actuelles (avec bruit)
            current_q1 = self.critic_1([states, actions], training=True)
            current_q2 = self.critic_2([states, actions], training=True)
            
            # Pertes des critiques (MSE)
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))
            
            # Échantillonner des actions pour l'état actuel (avec bruit)
            means, log_stds = self.actor(states, training=True)
            stds = tf.exp(log_stds)
            normal_dists = tfp.distributions.Normal(means, stds)
            actions_raw = normal_dists.sample()
            actions_policy = tf.tanh(actions_raw)
            
            # Calculer log-prob pour les actions de la politique
            log_probs = normal_dists.log_prob(actions_raw) - \
                        tf.math.log(1.0 - tf.square(actions_policy) + 1e-6)
            log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
            
            # Valeurs Q pour les actions de la politique (avec bruit)
            q1 = self.critic_1([states, actions_policy], training=True)
            q2 = self.critic_2([states, actions_policy], training=True)
            q_min = tf.minimum(q1, q2)
            
            # Perte de l'acteur = valeur Q attendue - entropie
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_min)
            
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
        
        self.critic_optimizer_1.apply_gradients(zip(critic1_gradients, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(critic2_gradients, self.critic_2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        if self.train_alpha:
            alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        del tape
        
        # Mettre à jour les réseaux cibles avec des mises à jour douces
        for target_param, param in zip(self.critic_1_target.variables, self.critic_1.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
            
        for target_param, param in zip(self.critic_2_target.variables, self.critic_2.variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)
        
        # Calculer l'entropie moyenne
        entropy = -tf.reduce_mean(log_probs)
        
        return (critic1_loss + critic2_loss) / 2.0, actor_loss, alpha_loss, entropy 