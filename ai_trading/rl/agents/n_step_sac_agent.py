import logging
import os

import numpy as np
import tensorflow as tf

from ai_trading.rl.agents.n_step_replay_buffer import NStepReplayBuffer
from ai_trading.rl.agents.sac_agent import SACAgent

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


class NStepSACAgent(SACAgent):
    """
    Extension de l'agent SAC (Soft Actor-Critic) qui utilise des retours multi-étapes.

    Cette implémentation remplace le tampon de replay standard par un tampon qui
    calcule des retours sur n étapes, ce qui peut aider à propager les récompenses
    plus rapidement et améliorer l'apprentissage, notamment dans les environnements
    avec des récompenses éparses.
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
        n_steps=3,  # Paramètre spécifique: nombre d'étapes pour les retours multi-étapes
    ):
        """
        Initialise l'agent SAC avec retours multi-étapes.

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
            n_steps (int): Nombre d'étapes pour calculer les retours multi-étapes
        """
        # Stocker n_steps avant d'appeler le constructeur parent
        self.n_steps = n_steps
        self.discount_factor = discount_factor  # Stocker le facteur d'actualisation

        # Initialiser la classe parente sans créer le tampon de replay standard
        # Nous allons créer notre propre tampon de replay avec retours multi-étapes
        self._skip_buffer_init = True

        # Initialiser les historiques des pertes (pour éviter les problèmes avec le code de test)
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []

        # Utiliser le même taux d'apprentissage pour tous les réseaux pour la compatibilité
        learning_rate = actor_learning_rate

        # Appel au constructeur parent
        super(NStepSACAgent, self).__init__(
            state_dim=state_size,
            action_dim=action_size,
            action_bounds=action_bounds,
            learning_rate=learning_rate,
            gamma=discount_factor,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            target_entropy=target_entropy,
        )

        # Créer le tampon de replay avec retours multi-étapes
        self.replay_buffer = NStepReplayBuffer(
            buffer_size=buffer_size, n_steps=n_steps, gamma=discount_factor
        )

        # Facteur de correction pour l'actualisation dans le calcul de la cible Q
        # Puisque nous avons déjà appliqué gamma^(n-1) dans le tampon, nous devons ajuster
        self.n_step_discount_factor = discount_factor**n_steps

        logger.info(
            f"Agent N-Step SAC initialisé: state_size={state_size}, action_size={action_size}, "
            f"n_steps={n_steps}, discount_factor={discount_factor}"
        )

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """
        Effectue une étape d'entraînement pour tous les réseaux.
        Cette méthode est adaptée pour les retours multi-étapes.

        Args:
            states: Lot d'états
            actions: Lot d'actions
            rewards: Lot de récompenses (déjà accumulées sur n étapes)
            next_states: Lot d'états suivants (après n étapes)
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
            log_probs_next = next_normal_dists.log_prob(next_actions_raw) - tf.math.log(
                1.0 - tf.square(next_actions) + 1e-6
            )
            log_probs_next = tf.reduce_sum(log_probs_next, axis=1, keepdims=True)

            # Valeurs Q cibles pour l'apprentissage du critique
            next_q1 = self.critic_1_target([next_states, next_actions])
            next_q2 = self.critic_2_target([next_states, next_actions])
            next_q_min = tf.minimum(next_q1, next_q2)

            # Soustraction du terme d'entropie pour SAC
            next_q_value = next_q_min - self.alpha * log_probs_next

            # Utiliser le facteur d'actualisation n-step ajusté pour la valeur Q cible
            # Les récompenses sont déjà accumulées sur n étapes dans le tampon de replay
            target_q = (
                rewards + (1 - dones) * self.n_step_discount_factor * next_q_value
            )

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
            log_probs = normal_dists.log_prob(actions_raw) - tf.math.log(
                1.0 - tf.square(actions_policy) + 1e-6
            )
            log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)

            # Valeurs Q pour les actions de la politique
            q1 = self.critic_1([states, actions_policy])
            q2 = self.critic_2([states, actions_policy])
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
        critic1_gradients = tape.gradient(
            critic1_loss, self.critic_1.trainable_variables
        )
        critic2_gradients = tape.gradient(
            critic2_loss, self.critic_2.trainable_variables
        )
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)

        self.critic_optimizer_1.apply_gradients(
            zip(critic1_gradients, self.critic_1.trainable_variables)
        )
        self.critic_optimizer_2.apply_gradients(
            zip(critic2_gradients, self.critic_2.trainable_variables)
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        if self.train_alpha:
            alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))

        del tape

        # Mettre à jour les réseaux cibles avec des mises à jour douces
        for target_param, param in zip(
            self.critic_1_target.variables, self.critic_1.variables
        ):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)

        for target_param, param in zip(
            self.critic_2_target.variables, self.critic_2.variables
        ):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)

        # Calculer l'entropie moyenne
        entropy = -tf.reduce_mean(log_probs)

        return (critic1_loss + critic2_loss) / 2.0, actor_loss, alpha_loss, entropy

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke une expérience dans le tampon de replay à n étapes.

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

        # Ajouter l'expérience au tampon de replay à n étapes
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Si l'épisode est terminé, gérer les expériences restantes
        if done:
            self.replay_buffer.handle_episode_end()

    def episode_end(self):
        """
        Méthode à appeler à la fin d'un épisode pour traiter les expériences restantes
        dans le tampon temporaire.
        """
        self.replay_buffer.handle_episode_end()

    def train(self):
        """
        Entraîne l'agent sur un lot d'expériences du tampon de replay.

        Returns:
            dict: Dictionnaire contenant les métriques d'entraînement
        """
        # Méthode identique à la classe parente, mais gardée par cohérence
        return super(NStepSACAgent, self).train()

    def save(self, filepath):
        """
        Sauvegarde les poids des modèles.

        Args:
            filepath (str): Chemin de base pour la sauvegarde
        """
        # Créer le répertoire si nécessaire
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        super(NStepSACAgent, self).save(filepath)

        # Sauvegarder les informations sur les étapes multiples
        np.savez(
            f"{filepath}/n_step_info.npz",
            n_steps=np.array([self.n_steps]),
            n_step_discount_factor=np.array([self.n_step_discount_factor]),
        )

        logger.info(
            f"Informations sur les retours multi-étapes sauvegardées dans {filepath}"
        )

    def load(self, filepath):
        """
        Charge les poids des modèles.

        Args:
            filepath (str): Chemin de base pour le chargement
        """
        super(NStepSACAgent, self).load(filepath)

        # Charger les informations sur les étapes multiples si elles existent
        n_step_info_path = f"{filepath}/n_step_info.npz"
        if os.path.exists(n_step_info_path):
            n_step_info = np.load(n_step_info_path)
            self.n_steps = int(n_step_info["n_steps"][0])
            self.n_step_discount_factor = float(
                n_step_info["n_step_discount_factor"][0]
            )

            # Recréer le tampon de replay avec les paramètres chargés
            self.replay_buffer = NStepReplayBuffer(
                buffer_size=self.replay_buffer.buffer.maxlen,
                n_steps=self.n_steps,
                gamma=self.discount_factor,
            )

            logger.info(
                f"Informations sur les retours multi-étapes chargées depuis {filepath}"
            )
        else:
            logger.warning(
                f"Aucune information sur les retours multi-étapes trouvée dans {filepath}"
            )


# Importation requise pour la méthode _train_step
import tensorflow_probability as tfp
