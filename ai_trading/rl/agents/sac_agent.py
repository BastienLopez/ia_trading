import logging
import os
import random
import sys
from collections import deque
from typing import Optional, Tuple, Dict, Any

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
    Tampon de replay optimisé pour stocker et échantillonner des séquences d'expériences.
    """

    def __init__(
        self,
        buffer_size: int,
        sequence_length: int,
        n_step: int = 1,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
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
        if self.n_step > 1:
            exp = self._nstep_preprocess(
                state, action, reward, next_state, done,
                self.n_step, self.gamma, self.n_step_buffer
            )
            if exp[0] is not None:
                self.buffer.append(exp)
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
        n_step_buffer.append((state, action, reward, next_state, done))

        if len(n_step_buffer) < n_step:
            return None, None, 0.0, None, False

        cum_reward = 0
        for i, (_, _, r, _, terminal) in enumerate(n_step_buffer):
            cum_reward += r * (gamma**i)
            if terminal:
                break

        initial_state = n_step_buffer[0][0]
        initial_action = n_step_buffer[0][1]
        final_next_state = n_step_buffer[-1][3]
        final_done = n_step_buffer[-1][4]

        return initial_state, initial_action, cum_reward, final_next_state, final_done

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.FloatTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).reshape(-1, 1).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(dones)).reshape(-1, 1).to(self.device),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.n_step_buffer.clear()


class TransformerActor(nn.Module):
    """
    Réseau de politique (acteur) basé sur Transformer optimisé pour le trading.
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
        self.max_seq_len = max(max_seq_len, sequence_length)

        # Transformer optimisé
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

        self.input_projection = nn.Linear(state_dim, d_model)
        self.mean_layer = nn.Linear(d_model, action_dim)
        self.log_std_layer = nn.Linear(d_model, action_dim)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, self.max_seq_len, d_model), requires_grad=True
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialiser log_std_layer avec une valeur raisonnable (ex: -0.5)
        if hasattr(self, 'log_std_layer'):
            nn.init.constant_(self.log_std_layer.bias, -0.5)
            nn.init.constant_(self.log_std_layer.weight, 0.01)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ajouter une dimension de séquence si nécessaire
        if len(states.shape) == 2:
            states = states.unsqueeze(1)
            
        seq_len = states.size(1)
        max_seq = self.pos_encoder.size(1)
        if seq_len > max_seq:
            states = states[:, -max_seq:, :]
            seq_len = max_seq

        # S'assurer que les dimensions sont correctes
        if states.size(-1) != self.state_dim:
            states = states[..., :self.state_dim]

        x = self.input_projection(states)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x[:, -1, :]
        
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std

    def get_action_and_log_prob(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean, log_std = self.forward(states)

        if deterministic:
            return torch.tanh(mean), None

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class TransformerCritic(nn.Module):
    """
    Réseau de valeur (critique) basé sur Transformer optimisé pour le trading.
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

        self.input_projection = nn.Linear(state_dim, d_model)
        self.action_projection = nn.Linear(action_dim, d_model)
        self.output_layer = nn.Linear(d_model, 1)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, self.max_seq_len, d_model), requires_grad=True
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Ajouter une dimension de séquence si nécessaire
        if len(states.shape) == 2:
            states = states.unsqueeze(1)
            
        seq_len = states.size(1)
        max_seq = self.pos_encoder.size(1)
        if seq_len > max_seq:
            states = states[:, -max_seq:, :]
            seq_len = max_seq

        # S'assurer que les dimensions sont correctes
        if states.size(-1) != self.state_dim:
            states = states[..., :self.state_dim]

        x = self.input_projection(states)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x[:, -1, :]
        
        # Correction : broadcast des actions si besoin
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).expand(-1, seq_len, -1)
        action_features = self.action_projection(actions[:, -1, :])
        x = x + action_features
        return self.output_layer(x)


class OptimizedSACAgent:
    """
    Agent SAC optimisé avec architecture Transformer pour le trading.
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
        grad_clip_value: float = 1.0,
        entropy_regularization: float = 0.2,
    ):
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
        self.grad_clip_value = grad_clip_value
        self.entropy_regularization = entropy_regularization
        # Ajuster l'échelle du bruit en fonction de la régularisation d'entropie
        self.noise_scale = 0.5 if entropy_regularization > 0 else 0.1
        # Augmenter l'impact de la régularisation d'entropie
        self.entropy_scale = 2.0 if entropy_regularization > 0 else 1.0

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

        # Réseaux cibles
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

        # Copier les poids
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimiseurs avec gradient clipping
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Buffer de replay optimisé
        self.replay_buffer = SequenceReplayBuffer(
            buffer_size=buffer_size,
            sequence_length=sequence_length,
            gamma=gamma,
            device=device
        )

        # Entropie automatique
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

    def _pad_or_stack_state(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            if state.shape[0] > self.state_dim:
                import warnings
                warnings.warn(f"L'état d'entrée ({state.shape[0]}) est plus grand que state_dim ({self.state_dim}), on tronque.")
                state = state[:self.state_dim]
            return np.tile(state, (self.sequence_length, 1))
        elif state.shape[-1] > self.state_dim:
            import warnings
            warnings.warn(f"L'état d'entrée ({state.shape[-1]}) est plus grand que state_dim ({self.state_dim}), on tronque.")
            state = state[..., :self.state_dim]
        
        if state.shape[0] != self.sequence_length:
            if state.shape[0] > self.sequence_length:
                return state[-self.sequence_length:]
            else:
                pad = np.zeros((self.sequence_length - state.shape[0], self.state_dim), dtype=state.dtype)
                return np.vstack([pad, state])
        return state

    def select_action(self, state, deterministic=False):
        self.actor.eval()  # Toujours en mode eval pour l'inférence
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_seq = self._pad_or_stack_state(state)
                state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            else:
                if state.ndim == 2:
                    state_tensor = state.unsqueeze(0).to(self.device)
                elif state.ndim == 3:
                    state_tensor = state.to(self.device)
                else:
                    raise ValueError("Format d'état non supporté")

            mean, log_std = self.actor(state_tensor)
            std = torch.exp(torch.clamp(log_std, min=-20, max=2))

            if deterministic:
                action = torch.tanh(mean)
            else:
                # Ajuster l'échelle du bruit
                noise = torch.randn_like(mean) * self.noise_scale
                action = torch.tanh(mean + std * noise)

            action = torch.clamp(action, -1.0, 1.0)
            action_np = action.cpu().numpy().squeeze()
            return np.atleast_1d(action_np)

    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        if batch_size is None:
            batch_size = self.batch_size

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Mise à jour des critiques
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action_and_log_prob(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Ajuster l'alpha en fonction de la régularisation d'entropie
            if self.entropy_regularization > 0:
                alpha = self.entropy_regularization * self.entropy_scale
                # Augmenter l'impact de l'entropie sur la valeur cible
                target_q = rewards + (1 - dones) * self.gamma * (target_q - alpha * next_log_probs)
            else:
                alpha = self.log_alpha.exp().item()
                target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Mise à jour des critiques avec gradient clipping
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip_value)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_value)
        self.critic2_optimizer.step()

        # Mise à jour de l'acteur
        new_actions, log_probs = self.actor.get_action_and_log_prob(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)

        if self.entropy_regularization > 0:
            # Avec régularisation d'entropie, maximiser l'entropie (SAC classique)
            actor_loss = (alpha * log_probs - q).mean()
            # Augmenter l'impact de la régularisation d'entropie
            actor_loss = actor_loss - self.entropy_regularization * self.entropy_scale * log_probs.mean()
            alpha_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # Sans régularisation, utiliser l'alpha automatique
            alpha = self.log_alpha.exp()
            actor_loss = (alpha * log_probs - q).mean()
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
        self.actor_optimizer.step()

        if not self.entropy_regularization > 0:
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
            "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            "alpha": self.alpha,
        }

    def _update_target_networks(self):
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

    def save(self, path: str) -> None:
        """Sauvegarde les poids du modèle."""
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.entropy_regularization > 0 else None
        }
        torch.save(state_dict, path)

    def load(self, path: str) -> None:
        """Charge les poids du modèle."""
        state_dict = torch.load(path)
        self.actor.load_state_dict(state_dict['actor'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.target_critic1.load_state_dict(state_dict['target_critic1'])
        self.target_critic2.load_state_dict(state_dict['target_critic2'])
        self.log_alpha = state_dict['log_alpha']
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(state_dict['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(state_dict['critic2_optimizer'])
        if self.entropy_regularization > 0 and state_dict['alpha_optimizer'] is not None:
            self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.add(state, action, reward, next_state, done)

# Alias pour maintenir la compatibilité
SACAgent = OptimizedSACAgent
TransformerSACAgent = OptimizedSACAgent
