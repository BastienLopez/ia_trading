import datetime
import logging
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ai_trading.rl.prioritized_replay import PrioritizedReplayBuffer
from ai_trading.rl.replay_buffer import ReplayBuffer
from ai_trading.rl.transformer_models import TransformerHybridModel

# Configuration du logger
logger = logging.getLogger("TransformerSAC")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

INFO_RETOUR_DIR = Path(__file__).parent.parent / "info_retour"
INFO_RETOUR_DIR.mkdir(exist_ok=True)


class TransformerSACAgent:
    """
    Agent d'apprentissage par renforcement combinant:
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
        checkpoints_dir=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise l'agent Transformer-SAC.
        """
        self.device = device
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

        if checkpoints_dir is None:
            self.checkpoints_dir = INFO_RETOUR_DIR / "checkpoints"
        else:
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
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=buffer_size,
                alpha=0.6,
                beta=0.4,
                n_step=n_step_returns,
                gamma=gamma,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size=buffer_size, n_step=n_step_returns, gamma=gamma
            )

        # Tampon d'état pour gérer les séquences
        self.state_buffer = deque(maxlen=sequence_length)

        # Initialiser target_entropy et log_alpha si auto_alpha_tuning est activé
        if self.auto_alpha_tuning:
            self.target_entropy = -np.prod(action_dim)
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(0.0, requires_grad=True, device=self.device)
            )
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=self.device)

        # Initialiser les réseaux
        self._init_networks()

        # Initialiser les historiques de perte
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_loss_history = []

        logger.info(f"Agent TransformerSAC initialisé avec le modèle {model_type}")

    def _init_networks(self):
        """Initialise les réseaux d'acteur et de critique."""
        input_shape = (
            self.sequence_length,
            self.state_dim[-1] if isinstance(self.state_dim, tuple) else self.state_dim,
        )

        # Réseau d'acteur
        self.actor = TransformerHybridModel(
            model_type=self.model_type,
            input_shape=input_shape,
            output_dim=self.action_dim * 2,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        ).to(self.device)

        # Réseaux de critique (Q1 et Q2)
        self.critic_1 = TransformerHybridModel(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        ).to(self.device)

        self.critic_2 = TransformerHybridModel(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        ).to(self.device)

        # Réseaux cibles
        self.critic_1_target = TransformerHybridModel(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        ).to(self.device)

        self.critic_2_target = TransformerHybridModel(
            model_type=self.model_type,
            input_shape=(input_shape[0], input_shape[1] + self.action_dim),
            output_dim=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            rnn_units=self.rnn_units,
            dropout_rate=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            sequence_length=self.sequence_length,
        ).to(self.device)

        # Copier les poids initiaux
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimiseurs
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=3e-4)
        self.alpha_optimizer = Adam([self.log_alpha], lr=3e-4)

        # Compiler les modèles uniquement si on n'est pas sous Windows
        if not sys.platform.startswith("win"):
            self.actor = torch.compile(self.actor)
            self.critic_1 = torch.compile(self.critic_1)
            self.critic_2 = torch.compile(self.critic_2)
            self.critic_1_target = torch.compile(self.critic_1_target)
            self.critic_2_target = torch.compile(self.critic_2_target)

    def _update_target_networks(self):
        """Met à jour les réseaux cibles avec les poids des réseaux principaux."""
        with torch.no_grad():
            for target_param, param in zip(
                self.critic_1_target.parameters(), self.critic_1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

            for target_param, param in zip(
                self.critic_2_target.parameters(), self.critic_2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def update_state_buffer(self, state):
        """Met à jour le tampon d'état avec le nouvel état."""
        self.state_buffer.append(state)

    def get_sequence_state(self):
        """Retourne l'état séquentiel actuel."""
        if len(self.state_buffer) < self.sequence_length:
            # Remplir avec le premier état si le tampon n'est pas plein
            while len(self.state_buffer) < self.sequence_length:
                self.state_buffer.appendleft(self.state_buffer[0])
        return np.array(self.state_buffer)

    def reset_state_buffer(self):
        """Réinitialise le tampon d'état."""
        self.state_buffer.clear()

    def sample_action(self, state, evaluate=False):
        """Échantillonne une action à partir de l'état actuel."""
        if len(self.state_buffer) < self.sequence_length:
            self.update_state_buffer(state)
            return np.random.uniform(
                self.action_low, self.action_high, size=self.action_dim
            )

        sequence_state = self.get_sequence_state()
        sequence_state = torch.FloatTensor(sequence_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(sequence_state).chunk(2, dim=-1)
                action = torch.tanh(mean)
            else:
                mean, log_std = self.actor(sequence_state).chunk(2, dim=-1)
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)

        action = action.cpu().numpy()[0]
        return np.clip(action, self.action_low, self.action_high)

    def remember(self, state, action, reward, next_state, done):
        """Stocke une transition dans le replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self, batch_size=None):
        """Effectue une étape d'entraînement."""
        batch_size = batch_size or self.batch_size

        if len(self.replay_buffer) < batch_size:
            return

        # Échantillonner un batch
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            batch, indices, weights = self.replay_buffer.sample(batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = self.replay_buffer.sample(batch_size)
            weights = torch.ones(batch_size).to(self.device)

        states, actions, rewards, next_states, dones = batch

        # Convertir en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Entraîner les critiques
        critic_loss = self._train_critics(
            states, actions, rewards, next_states, dones, weights
        )
        self.critic_loss_history.append(critic_loss)

        # Entraîner l'acteur et alpha
        actor_loss, alpha_loss = self._train_actor_and_alpha(states)
        self.actor_loss_history.append(actor_loss)
        if alpha_loss is not None:
            self.alpha_loss_history.append(alpha_loss)

        # Mettre à jour les réseaux cibles
        self._update_target_networks()

        # Mettre à jour les priorités si on utilise le replay prioritaire
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            with torch.no_grad():
                next_actions, next_log_probs = self._get_action_and_log_prob(
                    next_states
                )
                next_q1 = self.critic_1_target(
                    torch.cat([next_states, next_actions], dim=-1)
                )
                next_q2 = self.critic_2_target(
                    torch.cat([next_states, next_actions], dim=-1)
                )
                next_q = torch.min(next_q1, next_q2)
                target_q = rewards + (1 - dones) * self.gamma * (
                    next_q - self.alpha * next_log_probs
                )
                current_q1 = self.critic_1(torch.cat([states, actions], dim=-1))
                current_q2 = self.critic_2(torch.cat([states, actions], dim=-1))
                td_errors = torch.abs(target_q - torch.min(current_q1, current_q2))
                self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        return critic_loss, actor_loss, alpha_loss

    def _train_critics(self, states, actions, rewards, next_states, dones, weights):
        """Entraîne les réseaux de critique."""
        with torch.no_grad():
            next_actions, next_log_probs = self._get_action_and_log_prob(next_states)

            # Reshape next_actions to match next_states dimensions
            next_actions = next_actions.unsqueeze(1).expand(-1, next_states.size(1), -1)

            next_q1 = self.critic_1_target(
                torch.cat([next_states, next_actions], dim=-1)
            )
            next_q2 = self.critic_2_target(
                torch.cat([next_states, next_actions], dim=-1)
            )
            next_q = torch.min(next_q1, next_q2)
            next_q = next_q - self.alpha * next_log_probs.unsqueeze(-1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Reshape actions to match states dimensions
        actions = actions.unsqueeze(1).expand(-1, states.size(1), -1)

        current_q1 = self.critic_1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic_2(torch.cat([states, actions], dim=-1))

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        critic_loss = (critic_loss * weights).mean()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        return critic_loss.item()

    def _train_actor_and_alpha(self, states):
        """Entraîne le réseau d'acteur et le paramètre alpha."""
        actions, log_probs = self._get_action_and_log_prob(states)

        # Reshape actions to match states dimensions
        actions = actions.unsqueeze(1).expand(-1, states.size(1), -1)

        q1 = self.critic_1(torch.cat([states, actions], dim=-1))
        q2 = self.critic_2(torch.cat([states, actions], dim=-1))
        q = torch.min(q1, q2)

        # Entraînement de l'acteur
        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Entraînement de alpha si auto_alpha_tuning est activé
        alpha_loss = None
        if self.auto_alpha_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        return actor_loss.item(), alpha_loss.item() if alpha_loss is not None else None

    def _get_action_and_log_prob(self, states):
        """Retourne l'action et sa log-probabilité."""
        mean, log_std = self.actor(states).chunk(2, dim=-1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        actions = torch.tanh(x_t)
        log_probs = normal.log_prob(x_t) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(-1, keepdim=True)
        return actions, log_probs

    def save_models(self, suffix=""):
        """Sauvegarde les modèles."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.checkpoints_dir / f"checkpoint_{timestamp}{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_1_state_dict": self.critic_1.state_dict(),
                "critic_2_state_dict": self.critic_2.state_dict(),
                "critic_1_target_state_dict": self.critic_1_target.state_dict(),
                "critic_2_target_state_dict": self.critic_2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha": self.alpha,
            },
            checkpoint_dir / "models.pt",
        )

        logger.info(f"Modèles sauvegardés dans {checkpoint_dir}")

    def load_models(self, path):
        """Charge les modèles."""
        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.critic_1_target.load_state_dict(checkpoint["critic_1_target_state_dict"])
        self.critic_2_target.load_state_dict(checkpoint["critic_2_target_state_dict"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        self.log_alpha = checkpoint["log_alpha"]
        self.alpha = checkpoint["alpha"]

        logger.info(f"Modèles chargés depuis {path}")

    def update_target_networks(self):
        """Met à jour les réseaux cibles avec polyak averaging."""
        self._update_target_networks()
