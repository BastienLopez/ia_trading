"""
Module de meta-learning pour l'adaptation rapide à de nouveaux marchés.

Ce module implémente :
- MAML (Model-Agnostic Meta-Learning)
- Adaptation rapide aux changements de marché
- Apprentissage de stratégies généralisables
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from .agents.sac_agent import SACAgent
from .trading_environment import TradingEnvironment

logger = logging.getLogger(__name__)

class MAMLNetwork(nn.Module):
    """Réseau adaptatif pour MAML."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialise le réseau MAML.
        
        Args:
            input_dim: Dimension d'entrée
            hidden_dims: Dimensions des couches cachées
            output_dim: Dimension de sortie
            activation: Fonction d'activation
        """
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Construire les couches cachées
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                activation,
                nn.LayerNorm(dims[i+1])
            ])
        
        # Couche de sortie
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

class MAML:
    """Implémentation de Model-Agnostic Meta-Learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialise MAML.
        
        Args:
            input_dim: Dimension d'entrée
            hidden_dims: Dimensions des couches cachées
            output_dim: Dimension de sortie
            inner_lr: Taux d'apprentissage interne (adaptation)
            meta_lr: Taux d'apprentissage méta (mise à jour des poids)
            num_inner_steps: Nombre d'étapes d'adaptation
            device: Dispositif de calcul
        """
        self.model = MAMLNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        ).to(device)
        
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.device = device
        
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        
        logger.info(
            f"MAML initialisé avec {num_inner_steps} étapes internes, "
            f"lr_inner={inner_lr}, lr_meta={meta_lr}"
        )
    
    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        model: Optional[nn.Module] = None
    ) -> nn.Module:
        """
        Adapte le modèle aux données de support.
        
        Args:
            support_data: Tuple (entrées, étiquettes) pour l'adaptation
            model: Modèle à adapter (utilise self.model si None)
            
        Returns:
            nn.Module: Modèle adapté
        """
        if model is None:
            model = deepcopy(self.model)
        
        inputs, labels = support_data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Adaptation rapide avec gradient descent
        for _ in range(self.num_inner_steps):
            predictions = model(inputs)
            loss = F.mse_loss(predictions, labels)
            
            grads = torch.autograd.grad(loss, model.parameters())
            
            # Mise à jour manuelle des paramètres
            for param, grad in zip(model.parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        return model
    
    def meta_train_step(
        self,
        tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Dict[str, float]:
        """
        Effectue une étape d'entraînement méta.
        
        Args:
            tasks: Liste de tuples ((support_x, support_y), (query_x, query_y))
            
        Returns:
            Dict[str, float]: Métriques d'entraînement
        """
        meta_loss = 0
        meta_accuracy = 0
        
        for support_data, query_data in tasks:
            # Adaptation rapide sur les données de support
            adapted_model = self.adapt(support_data)
            
            # Évaluation sur les données de requête
            query_x, query_y = query_data
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            predictions = adapted_model(query_x)
            task_loss = F.mse_loss(predictions, query_y)
            
            meta_loss += task_loss
            
            # Calcul de la précision (pour les tâches de classification)
            if len(query_y.shape) == 1:  # Classification
                accuracy = (predictions.argmax(dim=1) == query_y).float().mean()
                meta_accuracy += accuracy
        
        # Moyenne sur toutes les tâches
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_accuracy / len(tasks) if len(query_y.shape) == 1 else 0
        
        # Mise à jour méta
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            "meta_loss": meta_loss.item(),
            "meta_accuracy": meta_accuracy.item() if isinstance(meta_accuracy, torch.Tensor) else meta_accuracy
        }
    
    def meta_train(
        self,
        task_generator: callable,
        num_epochs: int,
        tasks_per_batch: int,
        eval_interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Entraîne le méta-apprenant.
        
        Args:
            task_generator: Fonction générant des tâches d'entraînement
            num_epochs: Nombre d'époques
            tasks_per_batch: Nombre de tâches par lot
            eval_interval: Intervalle d'évaluation
            
        Returns:
            Dict[str, List[float]]: Historique d'entraînement
        """
        history = {
            "meta_loss": [],
            "meta_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": []
        }
        
        for epoch in range(num_epochs):
            # Générer un lot de tâches
            train_tasks = [task_generator() for _ in range(tasks_per_batch)]
            
            # Entraînement méta
            metrics = self.meta_train_step(train_tasks)
            
            history["meta_loss"].append(metrics["meta_loss"])
            history["meta_accuracy"].append(metrics["meta_accuracy"])
            
            # Évaluation périodique
            if (epoch + 1) % eval_interval == 0:
                eval_tasks = [task_generator() for _ in range(tasks_per_batch)]
                eval_metrics = self.meta_evaluate(eval_tasks)
                
                history["eval_loss"].append(eval_metrics["eval_loss"])
                history["eval_accuracy"].append(eval_metrics["eval_accuracy"])
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Meta Loss: {metrics['meta_loss']:.4f} - "
                    f"Meta Acc: {metrics['meta_accuracy']:.4f} - "
                    f"Eval Loss: {eval_metrics['eval_loss']:.4f} - "
                    f"Eval Acc: {eval_metrics['eval_accuracy']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Meta Loss: {metrics['meta_loss']:.4f} - "
                    f"Meta Acc: {metrics['meta_accuracy']:.4f}"
                )
        
        return history
    
    def meta_evaluate(
        self,
        tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Dict[str, float]:
        """
        Évalue le méta-apprenant.
        
        Args:
            tasks: Liste de tâches d'évaluation
            
        Returns:
            Dict[str, float]: Métriques d'évaluation
        """
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for support_data, query_data in tasks:
                # Adaptation aux données de support
                adapted_model = self.adapt(support_data)
                
                # Évaluation sur les données de requête
                query_x, query_y = query_data
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)
                
                predictions = adapted_model(query_x)
                loss = F.mse_loss(predictions, query_y)
                
                total_loss += loss.item()
                
                # Calcul de la précision pour les tâches de classification
                if len(query_y.shape) == 1:
                    accuracy = (predictions.argmax(dim=1) == query_y).float().mean()
                    total_accuracy += accuracy.item()
        
        num_tasks = len(tasks)
        return {
            "eval_loss": total_loss / num_tasks,
            "eval_accuracy": total_accuracy / num_tasks if len(query_y.shape) == 1 else 0
        }
    
    def save_model(self, path: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin de sauvegarde
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Modèle MAML sauvegardé dans {path}")
    
    def load_model(self, path: str) -> None:
        """
        Charge le modèle.
        
        Args:
            path: Chemin du modèle
        """
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Modèle MAML chargé depuis {path}")

class MarketMAML(MAML):
    """Extension de MAML spécifique au trading."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialise MarketMAML.
        
        Args:
            Voir MAML.__init__
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            num_inner_steps=num_inner_steps,
            device=device
        )
    
    def generate_market_task(
        self,
        env: TradingEnvironment,
        support_size: int = 100,
        query_size: int = 50
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Génère une tâche de trading.
        
        Args:
            env: Environnement de trading
            support_size: Taille des données de support
            query_size: Taille des données de requête
            
        Returns:
            Tuple: ((support_x, support_y), (query_x, query_y))
        """
        # Collecter les données
        states = []
        actions = []
        
        state = env.reset()[0]
        done = False
        
        while len(states) < (support_size + query_size) and not done:
            states.append(state)
            action = env.action_space.sample()  # Action aléatoire pour la collecte
            actions.append(action)
            
            state, _, done, _, _ = env.step(action)
        
        # Convertir en tenseurs
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        
        # Diviser en support et requête
        support_x = states[:support_size]
        support_y = actions[:support_size]
        query_x = states[support_size:support_size+query_size]
        query_y = actions[support_size:support_size+query_size]
        
        return (support_x, support_y), (query_x, query_y)
    
    def adapt_to_market(
        self,
        env: TradingEnvironment,
        num_episodes: int = 10
    ) -> None:
        """
        Adapte le modèle à un nouveau marché.
        
        Args:
            env: Environnement de trading
            num_episodes: Nombre d'épisodes d'adaptation
        """
        logger.info(f"Adaptation au nouveau marché sur {num_episodes} épisodes")
        
        for episode in range(num_episodes):
            # Générer une tâche d'adaptation
            support_data, query_data = self.generate_market_task(env)
            
            # Adapter le modèle
            adapted_model = self.adapt(support_data)
            
            # Évaluer sur les données de requête
            query_x, query_y = query_data
            with torch.no_grad():
                predictions = adapted_model(query_x.to(self.device))
                loss = F.mse_loss(predictions, query_y.to(self.device))
            
            logger.info(f"Épisode {episode+1}/{num_episodes} - Loss: {loss.item():.4f}")
        
        # Mettre à jour le modèle principal avec le modèle adapté
        self.model.load_state_dict(adapted_model.state_dict())
        logger.info("Adaptation terminée") 