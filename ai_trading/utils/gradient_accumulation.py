"""
Module d'utilitaires pour l'accumulation de gradient.

L'accumulation de gradient est une technique qui permet de simuler des batchs de grande taille
en accumulant les gradients sur plusieurs mini-batchs avant de mettre à jour les poids du modèle.
Cela est particulièrement utile lorsque la mémoire GPU est limitée.
"""

import logging
import torch
from typing import Dict, Optional, Union, Callable

logger = logging.getLogger(__name__)

def train_with_gradient_accumulation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 2,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gradient_clip: Optional[float] = None,
    fp16: bool = False,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    process_batch_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Fonction d'entraînement avec accumulation de gradient.
    
    Args:
        model: Le modèle PyTorch à entraîner
        dataloader: Le DataLoader contenant les données d'entraînement
        criterion: La fonction de perte
        optimizer: L'optimiseur
        device: Le périphérique sur lequel effectuer les calculs (CPU ou GPU)
        accumulation_steps: Nombre d'étapes sur lesquelles accumuler les gradients avant la mise à jour
        scheduler: Planificateur de taux d'apprentissage (optionnel)
        gradient_clip: Valeur pour le clipping de gradient (optionnel)
        fp16: Utiliser la précision mixte (float16) pour l'entraînement
        grad_scaler: Scaler pour la précision mixte, requis si fp16=True
        process_batch_fn: Fonction personnalisée pour traiter un batch (optionnel)
        
    Returns:
        Un dictionnaire contenant les métriques d'entraînement
    """
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)
    effective_batch_size = dataloader.batch_size * accumulation_steps
    
    logger.info(f"Entraînement avec accumulation de gradient: "
               f"{dataloader.batch_size} (batch réel) × {accumulation_steps} (étapes) "
               f"= {effective_batch_size} (batch effectif)")
    
    # Réinitialiser le gradient au début de l'accumulation
    optimizer.zero_grad()
    
    # Si fp16 est activé, vérifier si grad_scaler est fourni
    if fp16 and grad_scaler is None:
        grad_scaler = torch.cuda.amp.GradScaler()
        logger.info("Création automatique d'un GradScaler pour l'entraînement en précision mixte")
    
    for i, batch in enumerate(dataloader):
        # Utiliser la fonction de traitement personnalisée si fournie
        if process_batch_fn is not None:
            inputs, targets = process_batch_fn(batch)
        # Sinon, supposer que batch est un tuple (inputs, targets)
        else:
            inputs, targets = batch
        
        # Déplacer les données sur le périphérique approprié
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]
        else:
            inputs = inputs.to(device)
        
        if isinstance(targets, (list, tuple)):
            targets = [y.to(device) if torch.is_tensor(y) else y for y in targets]
        else:
            targets = targets.to(device)
        
        # Forward pass avec précision mixte si activée
        if fp16:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Normaliser la perte par le nombre d'étapes d'accumulation
                loss = loss / accumulation_steps
            
            # Backward pass avec scaler pour précision mixte
            grad_scaler.scale(loss).backward()
        else:
            # Forward pass standard
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Normaliser la perte par le nombre d'étapes d'accumulation
            loss = loss / accumulation_steps
            
            # Backward pass standard
            loss.backward()
        
        # Accumuler la perte pour le reporting
        total_loss += loss.item() * accumulation_steps
        
        # Mettre à jour les poids seulement après accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batches:
            # Appliquer le gradient clipping si configuré
            if gradient_clip is not None:
                if fp16:
                    grad_scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Mettre à jour les poids avec précision mixte si activée
            if fp16:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            
            # Planificateur de taux d'apprentissage
            if scheduler is not None:
                scheduler.step()
            
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Log périodique
            if (i + 1) % (accumulation_steps * 10) == 0:
                logger.info(f"Batch {i+1}/{n_batches}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Calculer la perte moyenne
    avg_loss = total_loss / n_batches
    
    return {"loss": avg_loss}


class GradientAccumulator:
    """
    Classe utilitaire pour faciliter l'accumulation de gradient.
    Permet d'encapsuler la logique d'accumulation de gradient dans un objet.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 2,
        fp16: bool = False,
        gradient_clip: Optional[float] = None,
    ):
        """
        Initialise l'accumulateur de gradient.
        
        Args:
            model: Le modèle PyTorch
            optimizer: L'optimiseur à utiliser
            accumulation_steps: Nombre d'étapes sur lesquelles accumuler les gradients
            fp16: Utiliser la précision mixte
            gradient_clip: Valeur pour le clipping de gradient
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.fp16 = fp16
        self.gradient_clip = gradient_clip
        self.current_step = 0
        self.grad_scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Réinitialiser les gradients au début
        self.optimizer.zero_grad()
        
        logger.info(f"GradientAccumulator initialisé avec {accumulation_steps} étapes")
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Effectue la passe arrière (backward) avec accumulation.
        
        Args:
            loss: La perte calculée
        """
        # Normaliser la perte
        normalized_loss = loss / self.accumulation_steps
        
        # Backward pass avec ou sans précision mixte
        if self.fp16:
            self.grad_scaler.scale(normalized_loss).backward()
        else:
            normalized_loss.backward()
        
        self.current_step += 1
    
    def step(self) -> bool:
        """
        Met à jour les poids si nous avons atteint le nombre d'étapes d'accumulation.
        
        Returns:
            True si une mise à jour des poids a été effectuée, False sinon
        """
        # Vérifier si nous avons atteint le nombre d'étapes d'accumulation
        if self.current_step % self.accumulation_steps == 0:
            # Appliquer le gradient clipping si configuré
            if self.gradient_clip is not None:
                if self.fp16:
                    self.grad_scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            # Mettre à jour les poids
            if self.fp16:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            
            # Réinitialiser les gradients
            self.optimizer.zero_grad()
            
            return True
        
        return False
    
    def reset(self) -> None:
        """Réinitialise l'accumulateur."""
        self.current_step = 0
        self.optimizer.zero_grad() 