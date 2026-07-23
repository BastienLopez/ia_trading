"""
Module DQN (Deep Q-Network) amélioré avec UCB et replay priorisé.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Union
import logging
from ai_trading.rl.agents.layers.noisy_linear import NoisyLinear

logger = logging.getLogger(__name__)

# Structure pour stocker les transitions
Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward', 'done', 'next_action_mask'))
Transition.__new__.__defaults__ = (None,)


class NoisyQNetwork(nn.Module):
    """Réseau DQN à bruit paramétrique pour l'exploration GPU."""

    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NoisyLinear(state_size, hidden_size),
                NoisyLinear(hidden_size, hidden_size),
                NoisyLinear(hidden_size, action_size),
            ]
        )

    def forward(self, state, deterministic=False):
        state = torch.relu(self.layers[0](state, deterministic=deterministic))
        state = torch.relu(self.layers[1](state, deterministic=deterministic))
        return self.layers[2](state, deterministic=deterministic)

    def reset_noise(self):
        for layer in self.layers:
            layer.reset_noise()

class PrioritizedReplayBuffer:
    """Buffer de replay priorisé pour DQN."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialise le buffer de replay priorisé.
        
        Args:
            capacity: Taille maximale du buffer
            alpha: Paramètre de priorité (0 = uniforme, 1 = priorité maximale)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.position = 0
        
    def __len__(self) -> int:
        """Retourne la taille actuelle du buffer."""
        return len(self.memory)
        
    def push(self, state, action, next_state, reward, done, next_action_mask=None):
        """Ajoute une transition au buffer."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        transition = Transition(state, action, next_state, reward, done, next_action_mask)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = transition
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """
        Échantillonne un batch de transitions avec priorité.
        
        Args:
            batch_size: Taille du batch
            beta: Paramètre d'importance sampling
            
        Returns:
            Tuple contenant (states, actions, next_states, rewards, dones, indices, weights)
        """
        if batch_size > len(self.memory):
            raise ValueError("batch_size doit être <= à la taille du buffer")
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:len(self.memory)])
            
        # Calcul des probabilités de sélection
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sélection des indices
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        assert len(set(indices)) == len(indices), "Des indices dupliqués ont été sélectionnés !"
        
        # Calcul des poids d'importance sampling
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Extraction des transitions
        transitions = [self.memory[idx] for idx in indices]
        batch = Transition(*zip(*transitions))
        
        return batch, indices, weights
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Met à jour les priorités des transitions."""
        for idx, priority in zip(indices, priorities):
            # Les erreurs TD peuvent être des tableaux (batch, 1) : le replay
            # stocke toujours une priorité scalaire pour rester échantillonnable.
            self.priorities[idx] = float(np.asarray(priority).reshape(-1)[0])

class DQNAgent:
    """Agent DQN avec UCB et replay priorisé."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_ucb: bool = True,
        use_prioritized_replay: bool = True,
        ucb_c: float = 2.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        # Aliases de compatibilité avec l'ancienne API publique.
        epsilon: Optional[float] = None,
        epsilon_min: Optional[float] = None,
        memory_size: Optional[int] = None,
        use_noisy_network: bool = False,
        n_step: int = 1,
    ):
        """
        Initialise l'agent DQN.
        
        Args:
            state_size: Dimension de l'état
            action_size: Nombre d'actions possibles
            hidden_size: Taille des couches cachées
            learning_rate: Taux d'apprentissage
            gamma: Facteur d'actualisation
            epsilon_start: Valeur initiale d'epsilon
            epsilon_end: Valeur minimale d'epsilon
            epsilon_decay: Taux de décroissance d'epsilon
            buffer_size: Taille du buffer de replay
            batch_size: Taille des batches
            target_update: Fréquence de mise à jour du réseau cible
            device: Device pour les calculs (CPU/GPU)
            use_ucb: Utiliser l'exploration UCB
            use_prioritized_replay: Utiliser le replay priorisé
            ucb_c: Paramètre d'exploration UCB
            alpha: Paramètre de priorité pour le replay
            beta: Paramètre d'importance sampling
        """
        if epsilon is not None:
            epsilon_start = epsilon
        if epsilon_min is not None:
            epsilon_end = epsilon_min
        if memory_size is not None:
            buffer_size = memory_size

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.use_ucb = use_ucb
        self.use_prioritized_replay = use_prioritized_replay
        self.ucb_c = ucb_c
        self.use_noisy_network = use_noisy_network
        self.n_step = max(1, int(n_step))
        self._n_step_buffer = deque(maxlen=self.n_step)
        
        # Compteurs pour UCB
        self.action_counts = np.zeros(action_size)
        self.action_values = np.zeros(action_size)
        
        # Réseaux
        self.policy_net = self._build_network().to(device)
        self.target_net = self._build_network().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Le chemin foreach/fused de Torch 2.13 déclenche un crash natif Triton
        # dans l'image CUDA du projet. Ce réglage conserve l'optimisation GPU
        # tout en utilisant le noyau Adam stable.
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            foreach=False,
            fused=False,
        )
        
        # Buffer de replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
            self.beta = beta
        else:
            self.memory = deque(maxlen=buffer_size)
            
        self.steps_done = 0

    @property
    def model(self) -> nn.Module:
        """Alias historique du réseau de politique."""
        return self.policy_net

    @property
    def target_model(self) -> nn.Module:
        """Alias historique du réseau cible."""
        return self.target_net
        
    def _build_network(self) -> nn.Module:
        """Construit le réseau de neurones."""
        if self.use_noisy_network:
            return NoisyQNetwork(self.state_size, self.hidden_size, self.action_size)
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )

    def _q_values(self, network, state, deterministic=False):
        if self.use_noisy_network:
            return network(state, deterministic=deterministic)
        return network(state)

    def _reset_noise(self):
        if self.use_noisy_network:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Sélectionne une action selon la politique epsilon-greedy ou UCB.
        
        Args:
            state: État actuel
            training: Mode entraînement ou évaluation
            
        Returns:
            Action sélectionnée
        """
        valid_actions = self._valid_action_indices(action_mask)
        if not training:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self._q_values(self.policy_net, state, deterministic=True)
                invalid_actions = np.setdiff1d(np.arange(self.action_size), valid_actions)
                if len(invalid_actions):
                    q_values[:, invalid_actions] = -torch.inf
                return q_values.max(1)[1].item()
                
        if self.use_ucb:
            # Exploration UCB
            if self.steps_done < len(valid_actions):
                action = int(valid_actions[self.steps_done])
            else:
                ucb_values = self.action_values + self.ucb_c * np.sqrt(
                    np.log(self.steps_done) / (self.action_counts + 1e-6)
                )
                action = int(valid_actions[np.argmax(ucb_values[valid_actions])])
        else:
            # Epsilon-greedy
            if random.random() < self.epsilon:
                action = int(random.choice(valid_actions.tolist()))
            else:
                with torch.no_grad():
                    self._reset_noise()
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self._q_values(self.policy_net, state)
                    invalid_actions = np.setdiff1d(np.arange(self.action_size), valid_actions)
                    if len(invalid_actions):
                        q_values[:, invalid_actions] = -torch.inf
                    action = q_values.max(1)[1].item()
                    
        self.steps_done += 1
        self.action_counts[action] += 1
        
        return action

    def _valid_action_indices(self, action_mask: Optional[np.ndarray]) -> np.ndarray:
        """Valide un masque booléen et retourne les actions exécutables.

        Le masque est appliqué aussi bien à l'exploration qu'à l'inférence : une
        vente sans position ou un achat sans solde ne peut donc pas être choisi
        par la politique DQN.
        """
        if action_mask is None:
            return np.arange(self.action_size, dtype=int)
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.action_size:
            raise ValueError("Le masque d'action DQN a une taille incompatible")
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("Le masque d'action DQN interdit toutes les actions")
        return valid
        
    def update_ucb(self, action: int, reward: float):
        """
        Met à jour les statistiques UCB.
        
        Args:
            action: Action effectuée
            reward: Récompense obtenue
        """
        self.action_values[action] = (
            (self.action_values[action] * (self.action_counts[action] - 1) + reward)
            / self.action_counts[action]
        )

    def act(self, state: np.ndarray, use_epsilon: bool = True) -> int:
        """Compatibilité de l'ancienne API DQN sans dupliquer l'agent."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return self.select_action(state, training=use_epsilon)

    def predict(self, state: np.ndarray) -> int:
        """Action déterministe pour l'inférence."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return self.select_action(state, training=False)

    def remember(self, state, action, reward, next_state, done, next_action_mask=None) -> None:
        """Stocke une transition issue de l'environnement Gymnasium."""
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device
        )
        next_state_tensor = torch.as_tensor(
            np.asarray(next_state, dtype=np.float32).reshape(1, -1), device=self.device
        )
        action_tensor = torch.tensor([[int(action)]], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([[float(reward)]], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[bool(done)]], dtype=torch.bool, device=self.device)
        mask_tensor = None
        if next_action_mask is not None:
            mask = np.asarray(next_action_mask, dtype=bool).reshape(-1)
            if mask.shape[0] != self.action_size or not mask.any():
                raise ValueError("Le masque de l'état suivant DQN est invalide")
            mask_tensor = torch.as_tensor(mask.reshape(1, -1), dtype=torch.bool, device=self.device)
        self._n_step_buffer.append(
            (state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor, mask_tensor)
        )
        if len(self._n_step_buffer) == self.n_step:
            self._store_n_step_transition()
            self._n_step_buffer.popleft()
        if done:
            self.end_episode()

    def _store_n_step_transition(self):
        state, action, _, _, _, _ = self._n_step_buffer[0]
        accumulated_reward = torch.zeros_like(self._n_step_buffer[0][3])
        next_state, done, next_action_mask = self._n_step_buffer[-1][2], self._n_step_buffer[-1][4], self._n_step_buffer[-1][5]
        for offset, (_, _, candidate_next, reward, candidate_done, candidate_mask) in enumerate(self._n_step_buffer):
            accumulated_reward += (self.gamma ** offset) * reward
            next_state, done, next_action_mask = candidate_next, candidate_done, candidate_mask
            if bool(candidate_done.item()):
                break
        self.memory.push(state, action, next_state, accumulated_reward, done, next_action_mask)

    def end_episode(self):
        """Ajoute les transitions partielles restantes en fin d'épisode."""
        while self._n_step_buffer:
            self._store_n_step_transition()
            self._n_step_buffer.popleft()

    def replay(self, batch_size: Optional[int] = None) -> float:
        """Compatibilité de l'ancienne boucle d'entraînement."""
        if batch_size is not None and batch_size != self.batch_size:
            previous_batch_size = self.batch_size
            self.batch_size = batch_size
            try:
                loss = self.optimize_model()
            finally:
                self.batch_size = previous_batch_size
        else:
            loss = self.optimize_model()
        return 0.0 if loss is None else float(loss)

    def update_target_model(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def optimize_model(self) -> Optional[float]:
        """
        Optimise le modèle sur un batch d'expériences.
        
        Returns:
            Perte moyenne si l'optimisation a été effectuée, None sinon
        """
        if len(self.memory) < self.batch_size:
            return None
            
        if self.use_prioritized_replay:
            batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = random.sample(self.memory, self.batch_size)
            batch = Transition(*zip(*transitions))
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Calcul des masques pour les états terminaux
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = [s for s in batch.next_state if s is not None]
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        self._reset_noise()
        # Calcul des Q-values actuelles
        state_action_values = self._q_values(self.policy_net, state_batch).gather(1, action_batch)
        
        # Calcul des Q-values cibles
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states:
            next_states = torch.cat(non_final_next_states)
            next_q_values = self._q_values(self.target_net, next_states)
            masks = [mask for state, mask in zip(batch.next_state, batch.next_action_mask) if state is not None]
            if any(mask is not None for mask in masks):
                normalized_masks = [
                    torch.ones((1, self.action_size), dtype=torch.bool, device=self.device)
                    if mask is None else mask.to(self.device).reshape(1, -1)
                    for mask in masks
                ]
                allowed = torch.cat(normalized_masks, dim=0)
                next_q_values = next_q_values.masked_fill(~allowed, -torch.inf)
            next_state_values[non_final_mask] = next_q_values.max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calcul de la perte avec importance sampling
        td_errors = state_action_values.squeeze(1) - expected_state_action_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Mise à jour des priorités si replay priorisé
        if self.use_prioritized_replay:
            priorities = torch.abs(td_errors).detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        
        # Mise à jour d'epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Mise à jour du réseau cible
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
        
    def save(self, path: str):
        """Sauvegarde le modèle."""
        torch.save({
            'agent_type': 'dqn',
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
            'device': self.device,
            'environment_config': getattr(self, 'environment_config', None),
            'use_noisy_network': self.use_noisy_network,
            'n_step': self.n_step,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'action_counts': self.action_counts,
            'action_values': self.action_values
        }, path)
        
    def load(self, path: str):
        """Charge le modèle."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Les checkpoints historiques ne contenaient que le state_dict du modèle.
        if 'policy_net_state_dict' not in checkpoint:
            self.policy_net.load_state_dict(checkpoint)
            self.update_target_model()
            return
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.action_counts = checkpoint['action_counts']
        self.action_values = checkpoint['action_values']
        self.environment_config = checkpoint.get('environment_config')
