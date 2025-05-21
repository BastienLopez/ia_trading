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

logger = logging.getLogger(__name__)

# Structure pour stocker les transitions
Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward', 'done'))

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
        
    def push(self, state, action, next_state, reward, done):
        """Ajoute une transition au buffer."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
            
        self.memory[self.position] = Transition(state, action, next_state, reward, done)
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
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:len(self.memory)])
            
        # Calcul des probabilités de sélection
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sélection des indices
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
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
            self.priorities[idx] = priority

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
        beta: float = 0.4
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
        
        # Compteurs pour UCB
        self.action_counts = np.zeros(action_size)
        self.action_values = np.zeros(action_size)
        
        # Réseaux
        self.policy_net = self._build_network().to(device)
        self.target_net = self._build_network().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Buffer de replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
            self.beta = beta
        else:
            self.memory = deque(maxlen=buffer_size)
            
        self.steps_done = 0
        
    def _build_network(self) -> nn.Module:
        """Construit le réseau de neurones."""
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Sélectionne une action selon la politique epsilon-greedy ou UCB.
        
        Args:
            state: État actuel
            training: Mode entraînement ou évaluation
            
        Returns:
            Action sélectionnée
        """
        if not training:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state).max(1)[1].item()
                
        if self.use_ucb:
            # Exploration UCB
            if self.steps_done < self.action_size:
                action = self.steps_done
            else:
                ucb_values = self.action_values + self.ucb_c * np.sqrt(
                    np.log(self.steps_done) / (self.action_counts + 1e-6)
                )
                action = np.argmax(ucb_values)
        else:
            # Epsilon-greedy
            if random.random() < self.epsilon:
                action = random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.policy_net(state).max(1)[1].item()
                    
        self.steps_done += 1
        self.action_counts[action] += 1
        
        return action
        
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
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None
        ])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Calcul des Q-values actuelles
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calcul des Q-values cibles
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Calcul de la perte avec importance sampling
        loss = (weights * (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2)).mean()
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Mise à jour des priorités si replay priorisé
        if self.use_prioritized_replay:
            td_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)
        
        # Mise à jour d'epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Mise à jour du réseau cible
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
        
    def save(self, path: str):
        """Sauvegarde le modèle."""
        torch.save({
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
        checkpoint = torch.load(path, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.action_counts = checkpoint['action_counts']
        self.action_values = checkpoint['action_values'] 