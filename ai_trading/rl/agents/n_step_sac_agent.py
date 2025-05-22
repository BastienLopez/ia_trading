import logging
import os

import numpy as np
import tensorflow as tf
import torch

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
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self.state_size = state_size
        self.action_size = action_size
        self.train_alpha = train_alpha

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
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Échantillonner un lot d'expériences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convertir les numpy arrays en tenseurs PyTorch et s'assurer qu'ils sont sur le bon device
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions).to(self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(rewards).to(self.device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.FloatTensor(next_states).to(self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.FloatTensor(dones).to(self.device)

        # S'assurer que les états ont la bonne forme pour le transformer
        if len(states.shape) == 2:
            states = states.unsqueeze(1)
        if len(next_states.shape) == 2:
            next_states = next_states.unsqueeze(1)

        # --- Logique d'entraînement SAC en PyTorch (copiée/adaptée de SACAgent) ---
        # 1. Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action_and_log_prob(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.n_step_discount_factor * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = torch.nn.functional.mse_loss(q1, target_q)
        critic2_loss = torch.nn.functional.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 2. Actor update
        new_actions, log_probs = self.actor.get_action_and_log_prob(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Alpha (entropy) update
        if self.train_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)

        # 4. Soft update des cibles
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # 5. Historique pour suivi
        self.critic_loss_history.append(((critic1_loss.item() + critic2_loss.item()) / 2.0))
        self.actor_loss_history.append(actor_loss.item())
        self.alpha_loss_history.append(alpha_loss.item())
        self.entropy_history.append(-log_probs.mean().item())

        return {
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2.0,
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'entropy': -log_probs.mean().item(),
        }

    def save(self, filepath):
        """
        Sauvegarde les poids des modèles.

        Args:
            filepath (str): Chemin de base pour la sauvegarde
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Sauvegarder les modèles
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'n_steps': self.n_steps,
            'n_step_discount_factor': self.n_step_discount_factor
        }

        # Sauvegarder le state_dict
        torch.save(state_dict, filepath)

        # Sauvegarder les informations sur les étapes multiples
        n_step_info_path = os.path.join(os.path.dirname(filepath), "n_step_info.npz")
        np.savez(
            n_step_info_path,
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
        # Charger le state_dict
        state_dict = torch.load(filepath)
        
        # Charger les poids des modèles
        self.actor.load_state_dict(state_dict['actor'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.target_critic1.load_state_dict(state_dict['target_critic1'])
        self.target_critic2.load_state_dict(state_dict['target_critic2'])
        self.log_alpha.data = torch.tensor(state_dict['log_alpha'])
        
        # Charger les paramètres n-step
        self.n_steps = state_dict['n_steps']
        self.n_step_discount_factor = state_dict['n_step_discount_factor']

        # Recréer le tampon de replay avec les paramètres chargés
        self.replay_buffer = NStepReplayBuffer(
            buffer_size=self.replay_buffer.buffer.maxlen,
            n_steps=self.n_steps,
            gamma=self.discount_factor,
        )

        logger.info(
            f"Informations sur les retours multi-étapes chargées depuis {filepath}"
        )


# Importation requise pour la méthode _train_step
import tensorflow_probability as tfp
