import logging
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from ai_trading.rl.models.multi_horizon_transformer import MultiHorizonTemporalTransformer

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Répertoire pour sauvegarder les modèles
INFO_RETOUR_DIR = Path(__file__).parent.parent.parent / "info_retour"
MODEL_DIR = INFO_RETOUR_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


class TransformerPPOActor(nn.Module):
    """
    Réseau de politique (acteur) basé sur Transformer pour l'agent PPO.
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
        log_std_min: float = -20,
        log_std_max: float = 2,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_low, self.action_high = action_bounds
        
        # Transformer pour extraire les caractéristiques temporelles
        self.transformer = MultiHorizonTemporalTransformer(
            input_dim=state_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=max_seq_len,
            forecast_horizons=[1],  # Nous utilisons seulement la prédiction à l'horizon 1
            output_dim=d_model,
        )
        
        # Couches pour la moyenne et l'écart-type de la politique
        self.mean_layer = nn.Linear(d_model, action_dim)
        self.log_std_layer = nn.Linear(d_model, action_dim)
        
        # Initialisation des poids
        self._init_weights()
        
    def _init_weights(self):
        """Initialisation des poids optimisée pour PPO."""
        # Initialisation de la couche moyenne
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)
        
        # Initialisation de la couche log_std
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)
        
    def forward(self, states_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass de l'acteur Transformer pour obtenir la moyenne et le log_std.
        
        Args:
            states_seq: Séquence d'états [batch_size, seq_len, state_dim]
            
        Returns:
            Tuple (moyenne, log_std) des actions
        """
        # Obtenir les caractéristiques du transformeur
        # Le transformeur retourne un dictionnaire {horizon: prédictions}
        transformer_features = self.transformer(states_seq)
        
        # Nous utilisons seulement la prédiction à l'horizon 1
        features = transformer_features[1]  # [batch_size, d_model]
        
        # Calcul de la moyenne et log_std des actions
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action_and_log_prob(
        self, 
        states_seq: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Échantillonne une action à partir de l'état avec sa log-probabilité.
        
        Args:
            states_seq: Séquence d'états [batch_size, seq_len, state_dim]
            deterministic: Si True, retourne l'action moyenne (déterministe)
            
        Returns:
            Tuple (action, log_prob) ou (action, None) si deterministic=True
        """
        mean, log_std = self.forward(states_seq)
        
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


class TransformerPPOCritic(nn.Module):
    """
    Réseau de valeur (critique) basé sur Transformer pour l'agent PPO.
    """
    
    def __init__(
        self,
        state_dim: int,
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
        self.sequence_length = sequence_length
        
        # Transformer pour extraire les caractéristiques temporelles
        self.transformer = MultiHorizonTemporalTransformer(
            input_dim=state_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=max_seq_len,
            forecast_horizons=[1],  # Nous utilisons seulement la prédiction à l'horizon 1
            output_dim=d_model,
        )
        
        # Couche de sortie pour la valeur
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialisation des poids
        self._init_weights()
        
    def _init_weights(self):
        """Initialisation des poids optimisée pour le critique."""
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, states_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du critique Transformer.
        
        Args:
            states_seq: Séquence d'états [batch_size, seq_len, state_dim]
            
        Returns:
            Valeur d'état estimée [batch_size, 1]
        """
        # Obtenir les caractéristiques du transformeur
        transformer_features = self.transformer(states_seq)
        
        # Utiliser la prédiction à l'horizon 1
        features = transformer_features[1]  # [batch_size, d_model]
        
        # Calcul de la valeur
        value = self.value_head(features)
        
        return value


class TransformerPPOAgent:
    """
    Agent PPO (Proximal Policy Optimization) avec architecture Transformer
    pour les actions continues et la modélisation temporelle avancée.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Paramètres Transformer
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 200,
        sequence_length: int = 50,
        # Paramètres PPO
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip_epsilon: float = 0.2,
        critic_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        mini_batch_size: int = 64,
        # Autres paramètres
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_dir: Optional[str] = None,
    ):
        """
        Initialise l'agent PPO avec des réseaux Transformer.
        
        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Dimension de l'espace d'action
            d_model: Dimension du modèle Transformer
            n_heads: Nombre de têtes d'attention
            num_layers: Nombre de couches Transformer
            dim_feedforward: Dimension de la couche feed-forward
            dropout: Taux de dropout
            activation: Fonction d'activation ('relu' ou 'gelu')
            max_seq_len: Longueur maximale de séquence supportée
            sequence_length: Longueur effective de la séquence d'états
            learning_rate: Taux d'apprentissage
            gamma: Facteur d'actualisation
            gae_lambda: Paramètre λ pour l'estimation de l'avantage généralisé
            clip_epsilon: Paramètre de clipping pour PPO
            value_clip_epsilon: Paramètre de clipping pour la fonction valeur
            critic_loss_coef: Coefficient pour la perte de la fonction valeur
            entropy_coef: Coefficient pour le terme d'entropie
            max_grad_norm: Valeur maximale de la norme du gradient
            update_epochs: Nombre d'époques pour mettre à jour les paramètres
            mini_batch_size: Taille des mini-batchs pour l'entraînement
            action_bounds: Bornes de l'espace d'action (min, max)
            device: Dispositif sur lequel exécuter les calculs
            model_dir: Répertoire pour sauvegarder les modèles (None = utiliser le défaut)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.action_bounds = action_bounds
        self.device = device
        
        # Répertoire pour sauvegarder les modèles
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = MODEL_DIR
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialiser les réseaux d'acteur et de critique
        self.actor = TransformerPPOActor(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=max_seq_len,
            sequence_length=sequence_length,
            action_bounds=action_bounds,
        ).to(device)
        
        self.critic = TransformerPPOCritic(
            state_dim=state_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_len=max_seq_len,
            sequence_length=sequence_length,
        ).to(device)
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Buffer pour stocker les séquences d'états
        self.state_buffer = deque(maxlen=sequence_length)
        
        # Historiques des pertes
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        
        logger.info(
            f"Agent TransformerPPO initialisé avec state_dim={state_dim}, "
            f"action_dim={action_dim}, d_model={d_model}, n_heads={n_heads}, "
            f"num_layers={num_layers}, sequence_length={sequence_length}"
        )
    
    def update_state_buffer(self, state: np.ndarray) -> None:
        """
        Met à jour le buffer d'états avec un nouvel état.
        
        Args:
            state: Nouvel état à ajouter
        """
        self.state_buffer.append(state)
    
    def get_padded_state_sequence(self) -> torch.Tensor:
        """
        Crée une séquence d'états avec padding pour l'entrée du Transformer.
        
        Returns:
            Tenseur contenant la séquence d'états [1, seq_len, state_dim]
        """
        # Copier le buffer actuel
        states = list(self.state_buffer)
        
        # Vérifier si le buffer est vide
        if len(states) == 0:
            # Créer un état nul comme fallback
            zero_state = np.zeros(self.state_dim)
            states = [zero_state]
        
        # Vérifier et adapter les dimensions des états si nécessaire
        for i in range(len(states)):
            if len(states[i]) != self.state_dim:
                # Si la dimension ne correspond pas, adapter l'état
                if len(states[i]) < self.state_dim:
                    # Padding si l'état est trop petit
                    padding = np.zeros(self.state_dim - len(states[i]))
                    states[i] = np.concatenate([states[i], padding])
                else:
                    # Troncature si l'état est trop grand
                    states[i] = states[i][:self.state_dim]
                
        # Ajouter du padding si nécessaire
        if len(states) < self.sequence_length:
            padding_needed = self.sequence_length - len(states)
            # Utiliser le premier état pour le padding
            first_state = states[0]
            padding_states = [first_state] * padding_needed
            states = padding_states + states
        
        # Convertir en array numpy avant de convertir en tenseur pour éviter l'avertissement
        states_array = np.array(states)
        states_tensor = torch.FloatTensor(states_array).to(self.device)
        
        # Ajouter la dimension de batch
        states_tensor = states_tensor.unsqueeze(0)  # [1, seq_len, state_dim]
        
        return states_tensor
    
    def reset_state_buffer(self) -> None:
        """Réinitialise le buffer d'états."""
        self.state_buffer.clear()
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Sélectionne une action à partir d'un état.
        
        Args:
            state: État courant
            deterministic: Si True, sélectionne l'action de manière déterministe
            
        Returns:
            Tuple (action, log_prob) ou (action, None) si deterministic=True
        """
        # Mettre à jour le buffer d'états
        self.update_state_buffer(state)
        
        # Obtenir la séquence d'états
        states_seq = self.get_padded_state_sequence()
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(states_seq, deterministic)
            
        return action.cpu().numpy(), log_prob.cpu().numpy() if log_prob is not None else None
    
    def compute_gae(
        self, 
        next_values: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule les avantages avec Generalized Advantage Estimation (GAE).
        
        Args:
            next_values: Valeurs des états suivants
            rewards: Récompenses obtenues
            masks: Masques (1 - done) pour les états terminaux
            values: Valeurs des états actuels
            
        Returns:
            Tuple (returns, advantages)
        """
        values = torch.cat([values, next_values], dim=0)
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
            
        returns = advantages + values[:-1]
        
        return returns, advantages
    
    def prepare_state_sequences(
        self, 
        states: torch.Tensor, 
        sequence_length: int
    ) -> torch.Tensor:
        """
        Prépare les séquences d'états pour les entrées du Transformer.
        
        Args:
            states: Tenseur d'états [taille_batch, state_dim]
            sequence_length: Longueur de séquence à créer
            
        Returns:
            Tenseur de séquences [taille_batch, sequence_length, state_dim]
        """
        batch_size = states.size(0)
        sequences = []
        
        # Pour chaque état, créer une séquence en utilisant les états précédents
        for i in range(batch_size):
            if i < sequence_length - 1:
                # Pour les premiers états, utiliser du padding
                padding = torch.zeros(
                    sequence_length - i - 1, 
                    self.state_dim, 
                    dtype=states.dtype, 
                    device=states.device
                )
                sequence = torch.cat([padding, states[:i+1]], dim=0)
            else:
                # Utiliser une fenêtre glissante pour les autres états
                sequence = states[i-sequence_length+1:i+1]
            
            sequences.append(sequence)
            
        return torch.stack(sequences)
    
    def update(
        self, 
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict[str, float]:
        """
        Met à jour les réseaux de l'agent avec PPO.
        
        Args:
            states: Tenseur des états [batch_size, state_dim]
            actions: Tenseur des actions [batch_size, action_dim]
            rewards: Tenseur des récompenses [batch_size]
            next_states: Tenseur des états suivants [batch_size, state_dim]
            dones: Tenseur des indicateurs de fin d'épisode [batch_size]
            
        Returns:
            Dictionnaire des statistiques d'entraînement
        """
        # Convertir en tenseurs PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Convertir les masques (1 - done)
        masks = 1.0 - dones
        
        # Estimer les valeurs des états
        with torch.no_grad():
            # Préparer les séquences d'états
            state_sequences = self.prepare_state_sequences(states, self.sequence_length)
            next_state_sequences = self.prepare_state_sequences(
                torch.cat([states[1:], next_states[-1:]], dim=0),
                self.sequence_length
            )
            
            values = self.critic(state_sequences)
            next_values = self.critic(next_state_sequences[-1:])
            
        # Calculer les retours et avantages
        returns, advantages = self.compute_gae(next_values, rewards, masks, values)
        
        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Récupérer les log_probs et valeurs originales
        with torch.no_grad():
            old_actions, old_log_probs = self.actor.get_action_and_log_prob(state_sequences)
            old_values = self.critic(state_sequences)
            
        # Variables pour les statistiques
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_epoch = 0
        
        # Mise à jour sur plusieurs époques
        for _ in range(self.update_epochs):
            # Générer les indices aléatoires
            batch_size = states.size(0)
            indices = torch.randperm(batch_size).to(self.device)
            
            # Parcourir les mini-batches
            for start_idx in range(0, batch_size, self.mini_batch_size):
                # Extraire les indices du mini-batch
                batch_indices = indices[start_idx:start_idx + self.mini_batch_size]
                
                # Sélectionner les données du mini-batch
                mb_states = state_sequences[batch_indices]
                mb_actions = actions[batch_indices]
                mb_returns = returns[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_old_values = old_values[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                
                # Évaluer les actions et valeurs actuelles
                new_actions, new_log_probs = self.actor.get_action_and_log_prob(mb_states)
                new_values = self.critic(mb_states)
                
                # Calculer le ratio pour PPO
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Calculer les deux termes de la perte de PPO
                term1 = ratio * mb_advantages
                term2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * mb_advantages
                
                # Perte de l'acteur (sens négatif car on veut maximiser)
                actor_loss = -torch.min(term1, term2).mean()
                
                # Entropie pour encourager l'exploration
                entropy = -new_log_probs.mean()
                
                # Perte du critique (MSE avec clipping)
                value_loss1 = F.mse_loss(new_values, mb_returns)
                value_pred_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values,
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon
                )
                value_loss2 = F.mse_loss(value_pred_clipped, mb_returns)
                critic_loss = torch.max(value_loss1, value_loss2)
                
                # Perte totale
                loss = (
                    actor_loss 
                    + self.critic_loss_coef * critic_loss 
                    - self.entropy_coef * entropy
                )
                
                # Mise à jour de l'acteur et du critique
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Accumuler les statistiques
                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.item()
        
        # Moyenner les pertes sur toutes les époques et mini-batchs
        n_updates = (len(states) + self.mini_batch_size - 1) // self.mini_batch_size
        actor_loss_epoch /= (n_updates * self.update_epochs)
        critic_loss_epoch /= (n_updates * self.update_epochs)
        entropy_epoch /= (n_updates * self.update_epochs)
        
        # Mettre à jour les historiques
        self.actor_loss_history.append(actor_loss_epoch)
        self.critic_loss_history.append(critic_loss_epoch)
        self.entropy_history.append(entropy_epoch)
        
        # Retourner les statistiques
        return {
            "actor_loss": actor_loss_epoch,
            "critic_loss": critic_loss_epoch,
            "entropy": entropy_epoch,
        }
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin de sauvegarde (optionnel)
        """
        if path is None:
            path = self.model_dir / "transformer_ppo_agent.pt"
        else:
            path = Path(path)
            
        path.parent.mkdir(exist_ok=True, parents=True)
        
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_loss_history": self.actor_loss_history,
                "critic_loss_history": self.critic_loss_history,
                "entropy_history": self.entropy_history,
            },
            path
        )
        
        logger.info(f"Modèle sauvegardé à {path}")
    
    def load(self, path: str) -> bool:
        """
        Charge le modèle.
        
        Args:
            path: Chemin du modèle
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            self.actor_loss_history = checkpoint["actor_loss_history"]
            self.critic_loss_history = checkpoint["critic_loss_history"]
            self.entropy_history = checkpoint["entropy_history"]
            
            logger.info(f"Modèle chargé depuis {path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False 