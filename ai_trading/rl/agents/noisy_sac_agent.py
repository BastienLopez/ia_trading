import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

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


class NoisyLinear(nn.Module):
    """
    Implémentation PyTorch d'une couche linéaire bruitée pour l'exploration.
    Basée sur le papier 'Noisy Networks for Exploration'.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Paramètres de la couche linéaire standard
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initialisation des paramètres de la couche
        """
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """
        Génère un nouveau bruit pour la couche
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Produit externe pour générer le bruit des poids
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """
        Génère un bruit factoriel
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x, training=True):
        """
        Forward pass avec ou sans bruit selon le mode d'entraînement
        """
        # Assurer que x est en float32 pour éviter les incompatibilités de type
        if x.dtype != torch.float32:
            x = x.float()

        if training:
            # Reset du bruit à chaque forward
            self.reset_noise()

            # Calcul avec bruit
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Mode déterministe (évaluation)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisySACAgent(SACAgent):
    """
    Agent Soft Actor-Critic (SAC) avec exploration paramétrique via Noisy Networks.
    Cette implémentation remplace les couches Linear standard par des couches NoisyLinear
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
        sigma_init=0.5,  # Valeur initiale pour le paramètre sigma des couches bruitées
        device="cuda" if torch.cuda.is_available() else "cpu",
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
            device (str): Périphérique de calcul ('cuda' ou 'cpu')
        """
        # Initialisation des attributs spécifiques avant l'appel à super()
        self.sigma_init = sigma_init
        self.device = device

        # Appel au constructeur parent avec les paramètres appropriés
        super(NoisySACAgent, self).__init__(
            state_size=state_size,
            action_size=action_size,
            action_bounds=action_bounds,
            learning_rate=actor_learning_rate,
            gamma=discount_factor,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            hidden_size=hidden_size,
            train_alpha=train_alpha,
            target_entropy=target_entropy,
            device=device,
        )

        # Reconstruire les réseaux spécifiques de NoisySAC
        self.actor = self._build_actor_network()
        self.critic_1 = self._build_critic_network()
        self.critic_2 = self._build_critic_network()
        self.critic_target_1 = self._build_critic_network()
        self.critic_target_2 = self._build_critic_network()

        # Déplacer les réseaux sur le bon périphérique
        self.actor = self.actor.to(device)
        self.critic_1 = self.critic_1.to(device)
        self.critic_2 = self.critic_2.to(device)
        self.critic_target_1 = self.critic_target_1.to(device)
        self.critic_target_2 = self.critic_target_2.to(device)

        # Copier les poids des critiques vers les cibles
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Réinitialiser les optimiseurs avec les paramètres des nouveaux réseaux
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_learning_rate,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_learning_rate)

        # Initialiser les attributs d'historique des pertes
        self.critic_1_loss_history = []
        self.critic_2_loss_history = []
        self.actor_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []

        logger.info(
            f"Agent NoisySAC initialisé: state_size={state_size}, action_size={action_size}, "
            f"sigma_init={sigma_init}, train_alpha={train_alpha}, target_entropy={self.target_entropy}"
        )

    def _build_actor_network(self):
        """
        Construit le réseau de politique (acteur) avec des couches bruitées.

        Returns:
            nn.Module: Réseau d'acteur bruité
        """

        class NoisyActorNetwork(nn.Module):
            def __init__(self, state_size, action_size, hidden_size, sigma_init):
                super(NoisyActorNetwork, self).__init__()
                self.noisy1 = NoisyLinear(state_size, hidden_size, sigma_init)
                self.noisy2 = NoisyLinear(hidden_size, hidden_size, sigma_init)
                self.noisy_mean = NoisyLinear(hidden_size, action_size, sigma_init)
                self.noisy_log_std = NoisyLinear(hidden_size, action_size, sigma_init)

            def forward(self, x, training=True):
                x = F.relu(self.noisy1(x, training))
                x = F.relu(self.noisy2(x, training))
                mean = self.noisy_mean(x, training)
                log_std = self.noisy_log_std(x, training)
                log_std = torch.clamp(log_std, -20, 2)
                return mean, log_std

        return NoisyActorNetwork(
            self.state_size, self.action_size, self.hidden_size, self.sigma_init
        )

    def _build_critic_network(self):
        """
        Construit le réseau critique avec des couches bruitées.

        Returns:
            nn.Module: Réseau critique bruité
        """

        class NoisyCriticNetwork(nn.Module):
            def __init__(self, state_size, action_size, hidden_size, sigma_init):
                super(NoisyCriticNetwork, self).__init__()
                self.noisy1 = NoisyLinear(
                    state_size + action_size, hidden_size, sigma_init
                )
                self.noisy2 = NoisyLinear(hidden_size, hidden_size, sigma_init)
                self.noisy3 = NoisyLinear(hidden_size, 1, sigma_init)

            def forward(self, state, action, training=True):
                x = torch.cat([state, action], dim=1)
                x = F.relu(self.noisy1(x, training))
                x = F.relu(self.noisy2(x, training))
                x = self.noisy3(x, training)
                return x

        return NoisyCriticNetwork(
            self.state_size, self.action_size, self.hidden_size, self.sigma_init
        )

    def select_action(self, state, deterministic=False):
        """
        Sélectionne une action selon la politique actuelle.

        Args:
            state: État actuel
            deterministic (bool): Si True, désactive le bruit dans les couches NoisyLinear

        Returns:
            np.array: Action sélectionnée
        """
        # Convertir l'état en tensor PyTorch
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)

        # S'assurer que l'état a la bonne forme
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Passer en mode évaluation
        self.actor.eval()

        with torch.no_grad():
            # Obtenir la moyenne et l'écart-type de la distribution d'actions
            # Le paramètre training contrôle si le bruit est utilisé dans les couches NoisyLinear
            mean, log_std = self.actor(state, training=not deterministic)

            if deterministic:
                # Mode déterministe : retourner la moyenne
                action = mean
            else:
                # Mode stochastique : échantillonner depuis la distribution
                std = torch.exp(log_std)
                normal = Normal(mean, std)
                x_t = normal.rsample()  # Reparametrization trick
                action = torch.tanh(x_t)

        # Retourner à l'entraînement
        self.actor.train()

        # Mettre à l'échelle l'action et convertir en numpy
        action = action.cpu().numpy()
        scaled_action = self._scale_action(action)

        return scaled_action[0]  # Retirer la dimension du lot

    # Alias pour rester compatible avec l'API existante
    act = select_action

    def train(self, batch_size=None):
        """
        Entraîne l'agent sur un batch d'expériences.

        Args:
            batch_size (int, optional): Taille du batch d'entraînement. Si None, utilise self.batch_size

        Returns:
            dict: Dictionnaire contenant les métriques d'entraînement
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.replay_buffer) < batch_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "entropy": 0.0,
            }

        # Échantillonner du tampon de replay
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        # Convertir les tenseurs numpy en tenseurs PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        alpha = self.log_alpha.exp().item()

        # --- Mise à jour des critiques ---
        with torch.no_grad():
            # Échantillonner les actions suivantes de la politique cible
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_distribution = Normal(next_mean, next_std)
            next_actions = torch.tanh(next_distribution.rsample())

            # Calculer l'entropie
            next_log_probs = next_distribution.log_prob(next_actions) - torch.log(
                1 - next_actions.pow(2) + 1e-6
            )
            next_log_probs = next_log_probs.sum(dim=1, keepdim=True)

            # Calculer les valeurs Q cibles
            q1_next = self.critic_target_1(next_states, next_actions)
            q2_next = self.critic_target_2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Calculer les valeurs Q actuelles
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        # Calculer la perte des critiques (MSE)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic_1_loss + critic_2_loss

        # Mettre à jour les critiques
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip_value:
            nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip_value)
            nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip_value)
        self.critic_optimizer.step()

        # --- Mise à jour de l'acteur ---
        mean, log_std = self.actor(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        actions_pred = torch.tanh(x_t)

        # Calculer les log probabilités des actions
        log_probs = normal.log_prob(x_t) - torch.log(1 - actions_pred.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=1, keepdim=True)

        # Calculer les valeurs Q pour les actions prédites
        q1 = self.critic_1(states, actions_pred)
        q2 = self.critic_2(states, actions_pred)
        min_q = torch.min(q1, q2)

        # Calculer la perte de l'acteur
        actor_loss = (alpha * log_probs - min_q).mean()

        # Mettre à jour l'acteur
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip_value:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
        self.actor_optimizer.step()

        # --- Mise à jour d'alpha ---
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.train_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs.detach() + self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # --- Mise à jour des réseaux cibles ---
        self._update_target_networks()

        # Stocker les métriques
        entropy = -log_probs.mean().item()
        self.critic_1_loss_history.append(critic_1_loss.item())
        self.critic_2_loss_history.append(critic_2_loss.item())
        self.actor_loss_history.append(actor_loss.item())
        self.alpha_loss_history.append(alpha_loss.item())
        self.entropy_history.append(entropy)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": entropy,
        }

    def _update_target_networks(self):
        """Met à jour les réseaux cibles avec un soft update"""
        for target_param, param in zip(
            self.critic_target_1.parameters(), self.critic_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target_2.parameters(), self.critic_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath):
        """
        Sauvegarde les modèles complets.

        Args:
            filepath (str): Chemin de base pour sauvegarder les modèles
        """
        torch.save(self.actor.state_dict(), f"{filepath}_actor.pt")
        torch.save(self.critic_1.state_dict(), f"{filepath}_critic1.pt")
        torch.save(self.critic_2.state_dict(), f"{filepath}_critic2.pt")
        torch.save(self.critic_target_1.state_dict(), f"{filepath}_critic_target1.pt")
        torch.save(self.critic_target_2.state_dict(), f"{filepath}_critic_target2.pt")
        torch.save(self.log_alpha, f"{filepath}_log_alpha.pt")

    def load(self, filepath):
        """
        Charge les modèles complets.

        Args:
            filepath (str): Chemin de base pour charger les modèles
        """
        self.actor.load_state_dict(
            torch.load(f"{filepath}_actor.pt", map_location=self.device)
        )
        self.critic_1.load_state_dict(
            torch.load(f"{filepath}_critic1.pt", map_location=self.device)
        )
        self.critic_2.load_state_dict(
            torch.load(f"{filepath}_critic2.pt", map_location=self.device)
        )
        self.critic_target_1.load_state_dict(
            torch.load(f"{filepath}_critic_target1.pt", map_location=self.device)
        )
        self.critic_target_2.load_state_dict(
            torch.load(f"{filepath}_critic_target2.pt", map_location=self.device)
        )
        self.log_alpha = torch.load(
            f"{filepath}_log_alpha.pt", map_location=self.device
        )
