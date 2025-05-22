import logging
import os
import random
import sys
from collections import deque
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

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


class SequenceReplayBuffer:
    """
    Tampon de replay pour stocker et échantillonner des séquences d'expériences.
    """

    def __init__(
        self,
        buffer_size: int,
        sequence_length: int,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """
        Initialise le tampon de replay.

        Args:
            buffer_size: Taille maximale du tampon
            sequence_length: Longueur des séquences à stocker
            n_step: Nombre d'étapes pour les retours
            gamma: Facteur d'actualisation
        """
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        logger.info(
            f"Tampon de replay de séquences initialisé: buffer_size={buffer_size}, sequence_length={sequence_length}"
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Ajoute une transition au tampon.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
        """
        # Prétraiter pour n-step returns si nécessaire
        if self.n_step > 1:
            exp = self._nstep_preprocess(
                state,
                action,
                reward,
                next_state,
                done,
                self.n_step,
                self.gamma,
                self.n_step_buffer,
            )
            if exp[0] is not None:  # Vérifier que l'expérience est valide
                s, a, r, ns, d = exp
                self.buffer.append((s, a, r, ns, d))
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def _nstep_preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        n_step: int,
        gamma: float,
        n_step_buffer: deque,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[np.ndarray], bool]:
        """
        Prétraite une transition pour les retours sur n étapes.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indicateur de fin d'épisode
            n_step: Nombre d'étapes
            gamma: Facteur d'actualisation
            n_step_buffer: Tampon pour les n étapes

        Returns:
            Tuple contenant (state, action, reward, next_state, done)
        """
        n_step_buffer.append((state, action, reward, next_state, done))

        if len(n_step_buffer) < n_step:
            return None, None, 0.0, None, False

        # Calculer la récompense cumulée
        cum_reward = 0
        for i, (_, _, r, _, terminal) in enumerate(n_step_buffer):
            cum_reward += r * (gamma**i)
            if terminal:
                break

        # Récupérer l'état initial et l'action
        initial_state = n_step_buffer[0][0]
        initial_action = n_step_buffer[0][1]

        # Récupérer l'état final et le statut de fin
        final_next_state = n_step_buffer[-1][3]
        final_done = n_step_buffer[-1][4]

        return initial_state, initial_action, cum_reward, final_next_state, final_done

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Échantillonne un batch de transitions aléatoires.

        Args:
            batch_size: Taille du batch à échantillonner

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        # S'assurer que le tampon contient suffisamment d'éléments
        batch_size = min(batch_size, len(self.buffer))

        # Échantillonner des indices aléatoires
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # Récupérer les transitions
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # Convertir en tableaux numpy
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        """
        Retourne la taille actuelle du tampon.

        Returns:
            int: Nombre d'éléments dans le tampon
        """
        return len(self.buffer)

    def clear(self):
        """
        Vide le tampon.
        """
        self.buffer.clear()
        self.n_step_buffer.clear()


class SACAgent:
    """
    Agent SAC (Soft Actor-Critic) classique avec réseaux MLP pour le trading.
    """
    def __init__(
        self,
        state_size: int,
        action_size: int = 1,
        action_bounds: tuple = (-1, 1),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1000000,
        hidden_size: int = 256,
        train_alpha: bool = True,
        target_entropy: float = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low, self.action_high = action_bounds
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.train_alpha = train_alpha
        self.target_entropy = (
            -float(action_size) if target_entropy is None else target_entropy
        )
        # Réseaux
        self.actor = self._build_actor().to(device)
        self.critic_1 = self._build_critic().to(device)
        self.critic_2 = self._build_critic().to(device)
        self.critic_target_1 = self._build_critic().to(device)
        self.critic_target_2 = self._build_critic().to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=learning_rate,
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.alpha = self.log_alpha.exp().item()
        # Buffer de replay
        self.replay_buffer = SequenceReplayBuffer(
            buffer_size=buffer_size, sequence_length=1, gamma=gamma
        )
    def _build_actor(self):
        class Actor(nn.Module):
            def __init__(self, state_size, action_size, hidden_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.mean = nn.Linear(hidden_size, action_size)
                self.log_std = nn.Linear(hidden_size, action_size)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                mean = self.mean(x)
                log_std = torch.clamp(self.log_std(x), -20, 2)
                return mean, log_std
        return Actor(self.state_size, self.action_size, self.hidden_size)
    def _build_critic(self):
        class Critic(nn.Module):
            def __init__(self, state_size, action_size, hidden_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size + action_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.out = nn.Linear(hidden_size, 1)
            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.out(x)
        return Critic(self.state_size, self.action_size, self.hidden_size)
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(state)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
        action = action.cpu().numpy()
        scaled_action = self._scale_action(action)
        return scaled_action[0]
    def _scale_action(self, action):
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
    def train(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if len(self.replay_buffer) < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0, "entropy": 0.0}
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_normal = Normal(next_mean, next_std)
            next_x_t = next_normal.rsample()
            next_action = torch.tanh(next_x_t)
            next_log_prob = next_normal.log_prob(next_x_t) - torch.log(1 - next_action.pow(2) + 1e-6)
            next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
            target_q1 = self.critic_target_1(next_states, next_action)
            target_q2 = self.critic_target_2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_prob)
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        mean, log_std = self.actor(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        q1 = self.critic_1(states, action)
        q2 = self.critic_2(states, action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.train_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)
        self._update_target_networks()
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            "entropy": -log_prob.mean().item(),
        }
    def _update_target_networks(self):
        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic_1.state_dict(),
                "critic2_state_dict": self.critic_2.state_dict(),
                "critic_target_1_state_dict": self.critic_target_1.state_dict(),
                "critic_target_2_state_dict": self.critic_target_2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha": self.alpha,
            },
            path,
        )
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic2_state_dict"])
        self.critic_target_1.load_state_dict(checkpoint["critic_target_1_state_dict"])
        self.critic_target_2.load_state_dict(checkpoint["critic_target_2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha = checkpoint["alpha"]


class TransformerActor(nn.Module):
    """
    Réseau de politique (acteur) basé sur Transformer pour l'agent SAC.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 200,
        sequence_length: int = 50,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.action_low, self.action_high = action_bounds

        # Toujours prendre le plus grand entre max_seq_len et sequence_length
        self.max_seq_len = max(max_seq_len, sequence_length)

        # Transformer pour extraire les caractéristiques temporelles
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Couche d'entrée pour projeter les états dans l'espace du Transformer
        self.input_projection = nn.Linear(state_dim, d_model)

        # Couches pour la moyenne et l'écart-type de la politique
        self.mean_layer = nn.Linear(d_model, action_dim)
        self.log_std_layer = nn.Linear(d_model, action_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, self.max_seq_len, d_model), requires_grad=True
        )

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialise les poids des couches."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant du réseau.

        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_len, state_dim)

        Returns:
            Tuple (mean, log_std) où mean et log_std sont les paramètres de la distribution
        """
        # Projeter les états dans l'espace du Transformer
        # Si la séquence est trop longue, on tronque
        seq_len = x.size(1)
        max_seq = self.pos_encoder.size(1)
        if seq_len > max_seq:
            x = x[:, -max_seq:, :]
            seq_len = max_seq
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x[:, -1, :]
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std

    def get_action_and_log_prob(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Échantillonne une action à partir de l'état avec sa log-probabilité.

        Args:
            states: Séquence d'états [batch_size, seq_len, state_dim]
            deterministic: Si True, retourne l'action moyenne (déterministe)

        Returns:
            Tuple (action, log_prob) ou (action, None) si deterministic=True
        """
        mean, log_std = self.forward(states)

        if deterministic:
            # Action déterministe (moyenne)
            action = torch.tanh(mean)
            return action, None

        # Échantillonnage stochastique
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparametrization trick pour permettre le backprop
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Calcul du log_prob avec correction pour la transformation tanh
        log_prob = normal.log_prob(x_t)
        # Correction pour tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class TransformerCritic(nn.Module):
    """
    Réseau de valeur (critique) basé sur Transformer pour l'agent SAC.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 200,
        sequence_length: int = 50,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.max_seq_len = max(max_seq_len, sequence_length)

        # Transformer pour extraire les caractéristiques temporelles
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Couche d'entrée pour projeter les états dans l'espace du Transformer
        self.input_projection = nn.Linear(state_dim, d_model)

        # Couche pour projeter les actions
        self.action_projection = nn.Linear(action_dim, d_model)

        # Couche de sortie pour la valeur Q
        self.output_layer = nn.Linear(d_model, 1)

        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, self.max_seq_len, d_model), requires_grad=True
        )

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialise les poids des couches."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du réseau.

        Args:
            states: Tenseur d'états de forme (batch_size, seq_len, state_dim)
            actions: Tenseur d'actions de forme (batch_size, action_dim)

        Returns:
            Tenseur de valeurs Q de forme (batch_size, 1)
        """
        seq_len = states.size(1)
        max_seq = self.pos_encoder.size(1)
        if seq_len > max_seq:
            states = states[:, -max_seq:, :]
            seq_len = max_seq
        x = self.input_projection(states)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x[:, -1, :]
        action_features = self.action_projection(actions)
        x = x + action_features
        q_value = self.output_layer(x)
        return q_value


class TransformerSACAgent:
    """
    Agent SAC (Soft Actor-Critic) avec architecture Transformer pour le trading.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 200,
        sequence_length: int = 50,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        # Toujours prendre le plus grand entre max_seq_len et sequence_length
        self.max_seq_len = max(max_seq_len, sequence_length)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.action_low, self.action_high = action_bounds

        # Initialisation des réseaux
        self.actor = TransformerActor(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=self.max_seq_len,
            sequence_length=sequence_length,
            action_bounds=action_bounds,
        ).to(device)

        self.critic1 = TransformerCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=self.max_seq_len,
            sequence_length=sequence_length,
        ).to(device)

        self.critic2 = TransformerCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=self.max_seq_len,
            sequence_length=sequence_length,
        ).to(device)

        # Initialisation des réseaux cibles
        self.target_critic1 = TransformerCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=self.max_seq_len,
            sequence_length=sequence_length,
        ).to(device)

        self.target_critic2 = TransformerCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=self.max_seq_len,
            sequence_length=sequence_length,
        ).to(device)

        # Copier les poids des critiques vers les cibles
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Buffer de replay
        self.replay_buffer = SequenceReplayBuffer(
            buffer_size=buffer_size,
            sequence_length=sequence_length,
            gamma=gamma
        )

        # Entropie automatique
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

    def _pad_or_stack_state(self, state: np.ndarray) -> np.ndarray:
        """
        Transforme un état plat en séquence si besoin (padding ou duplication).
        Tronque ou sélectionne les bonnes dimensions si besoin.
        """
        # Si l'état est plus grand que state_dim, on tronque
        if state.ndim == 1:
            if state.shape[0] > self.state_dim:
                import warnings
                warnings.warn(f"L'état d'entrée ({state.shape[0]}) est plus grand que state_dim ({self.state_dim}), on tronque.")
                state = state[:self.state_dim]
            # état plat -> dupliquer pour former une séquence
            return np.tile(state, (self.sequence_length, 1))
        elif state.shape[-1] > self.state_dim:
            import warnings
            warnings.warn(f"L'état d'entrée ({state.shape[-1]}) est plus grand que state_dim ({self.state_dim}), on tronque.")
            state = state[..., :self.state_dim]
        # séquence de mauvaise longueur -> padding ou troncature
        if state.shape[0] != self.sequence_length:
            if state.shape[0] > self.sequence_length:
                return state[-self.sequence_length:]
        else:
                pad = np.zeros((self.sequence_length - state.shape[0], self.state_dim), dtype=state.dtype)
                return np.vstack([pad, state])
        return state

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Sélectionne une action en fonction de l'état actuel.

        Args:
            state: État actuel [sequence_length, state_dim] ou [state_dim]
            deterministic: Si True, retourne l'action moyenne (déterministe)

        Returns:
            Action sélectionnée [action_dim]
        """
        # Toujours fournir un batch de séquences à l'acteur
        state_seq = self._pad_or_stack_state(state)
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.get_action_and_log_prob(state_tensor, deterministic)
            return action.cpu().numpy()[0]

    def train(self, batch_size: Optional[int] = None) -> dict:
        """
        Entraîne l'agent sur un batch d'expériences.

        Args:
            batch_size: Taille du batch (utilise self.batch_size si None)

        Returns:
            Dictionnaire contenant les métriques d'entraînement
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Échantillonner un batch du buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        # Tronquer les états à la bonne dimension
        if states.shape[-1] > self.state_dim:
            import warnings
            warnings.warn(f"Les états du buffer ({states.shape[-1]}) sont plus grands que state_dim ({self.state_dim}), on tronque.")
            states = states[..., :self.state_dim]
        if next_states.shape[-1] > self.state_dim:
            import warnings
            warnings.warn(f"Les next_states du buffer ({next_states.shape[-1]}) sont plus grands que state_dim ({self.state_dim}), on tronque.")
            next_states = next_states[..., :self.state_dim]
        # Si les états sont en 2D, les transformer en 3D (batch, seq_len, state_dim)
        if states.ndim == 2:
            states = np.tile(states[:, None, :], (1, self.sequence_length, 1))
        if next_states.ndim == 2:
            next_states = np.tile(next_states[:, None, :], (1, self.sequence_length, 1))
        # Tronquer la séquence si trop longue (seulement si 3D)
        if states.ndim == 3 and states.shape[1] > self.sequence_length:
            states = states[:, -self.sequence_length:, :]
        if next_states.ndim == 3 and next_states.shape[1] > self.sequence_length:
            next_states = next_states[:, -self.sequence_length:, :]

        # Convertir en tenseurs PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Mise à jour des critiques
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action_and_log_prob(
                next_states
            )
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (
                target_q - self.alpha * next_log_probs
            )

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Mise à jour de l'acteur
        new_actions, log_probs = self.actor.get_action_and_log_prob(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Mise à jour de l'entropie
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Mise à jour des réseaux cibles
        self._update_target_networks()

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }

    def _update_target_networks(self):
        """Met à jour les réseaux cibles avec une moyenne mobile exponentielle."""
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """
        Sauvegarde les poids du modèle.

        Args:
            path: Chemin où sauvegarder le modèle
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "target_critic1_state_dict": self.target_critic1.state_dict(),
                "target_critic2_state_dict": self.target_critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha": self.alpha,
            },
            path,
        )

    def load(self, path: str):
        """
        Charge les poids du modèle.

        Args:
            path: Chemin vers le fichier de sauvegarde
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha = checkpoint["alpha"]
