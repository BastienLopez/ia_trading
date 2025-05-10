"""
Module de network distillation pour la compression et le transfert de connaissances entre modèles.

Ce module implémente :
- Transfert de connaissances du modèle professeur vers l'élève
- Compression de modèle via distillation
- Fine-tuning du modèle distillé
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .agents.sac_agent import SACAgent
from .trading_environment import TradingEnvironment

logger = logging.getLogger(__name__)

class DistillationNetwork(nn.Module):
    """Réseau de l'élève pour la distillation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        temperature: float = 2.0
    ):
        """
        Initialise le réseau de l'élève.
        
        Args:
            input_dim: Dimension d'entrée
            hidden_dims: Dimensions des couches cachées
            output_dim: Dimension de sortie
            activation: Fonction d'activation
            temperature: Température pour la distillation (softmax)
        """
        super().__init__()
        
        self.temperature = temperature
        layers = []
        
        # Construire les couches cachées
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                activation,
                nn.BatchNorm1d(dims[i+1])
            ])
        
        # Couche de sortie
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec température."""
        logits = self.network(x)
        return torch.softmax(logits / self.temperature, dim=-1)

class NetworkDistillation:
    """Gestionnaire de la distillation de réseau."""
    
    def __init__(
        self,
        teacher_model: SACAgent,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temperature: float = 2.0,
        alpha: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialise le processus de distillation.
        
        Args:
            teacher_model: Modèle professeur (pré-entraîné)
            input_dim: Dimension d'entrée
            hidden_dims: Dimensions des couches cachées de l'élève
            output_dim: Dimension de sortie
            temperature: Température pour la distillation
            alpha: Coefficient de pondération entre les pertes
            device: Dispositif de calcul
        """
        self.teacher = teacher_model
        self.student = DistillationNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            temperature=temperature
        ).to(device)
        
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.optimizer = torch.optim.Adam(self.student.parameters())
        
        logger.info(f"Network Distillation initialisée avec température {temperature}")
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte de distillation.
        
        Args:
            student_logits: Sorties de l'élève
            teacher_logits: Sorties du professeur
            labels: Étiquettes réelles
            
        Returns:
            torch.Tensor: Perte totale
        """
        # Perte de distillation (KL divergence)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Perte sur les étiquettes réelles
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Perte totale
        total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
        return total_loss
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Effectue une étape d'entraînement.
        
        Args:
            batch: Tuple (entrées, étiquettes)
            
        Returns:
            Dict[str, float]: Métriques d'entraînement
        """
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass du professeur
        with torch.no_grad():
            teacher_logits = self.teacher.get_action(inputs)
        
        # Forward pass de l'élève
        student_logits = self.student(inputs)
        
        # Calcul de la perte
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "teacher_confidence": torch.max(teacher_logits, dim=1)[0].mean().item(),
            "student_confidence": torch.max(student_logits, dim=1)[0].mean().item()
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        eval_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Entraîne le modèle élève.
        
        Args:
            train_loader: DataLoader d'entraînement
            num_epochs: Nombre d'époques
            eval_loader: DataLoader d'évaluation
            
        Returns:
            Dict[str, List[float]]: Historique d'entraînement
        """
        history = {
            "train_loss": [],
            "eval_loss": [],
            "teacher_confidence": [],
            "student_confidence": []
        }
        
        for epoch in range(num_epochs):
            # Entraînement
            self.student.train()
            epoch_metrics = []
            
            for batch in train_loader:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
            
            # Moyennes des métriques
            avg_metrics = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()
            }
            
            history["train_loss"].append(avg_metrics["loss"])
            history["teacher_confidence"].append(avg_metrics["teacher_confidence"])
            history["student_confidence"].append(avg_metrics["student_confidence"])
            
            # Évaluation
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                history["eval_loss"].append(eval_metrics["loss"])
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {avg_metrics['loss']:.4f} - "
                f"Teacher conf: {avg_metrics['teacher_confidence']:.4f} - "
                f"Student conf: {avg_metrics['student_confidence']:.4f}"
            )
        
        return history
    
    def evaluate(
        self,
        eval_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Évalue le modèle élève.
        
        Args:
            eval_loader: DataLoader d'évaluation
            
        Returns:
            Dict[str, float]: Métriques d'évaluation
        """
        self.student.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                teacher_logits = self.teacher.get_action(inputs)
                student_logits = self.student(inputs)
                
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                total_loss += loss.item()
                num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    def save_student(self, path: str) -> None:
        """
        Sauvegarde le modèle élève.
        
        Args:
            path: Chemin de sauvegarde
        """
        torch.save(self.student.state_dict(), path)
        logger.info(f"Modèle élève sauvegardé dans {path}")
    
    def load_student(self, path: str) -> None:
        """
        Charge le modèle élève.
        
        Args:
            path: Chemin du modèle
        """
        self.student.load_state_dict(torch.load(path))
        logger.info(f"Modèle élève chargé depuis {path}") 