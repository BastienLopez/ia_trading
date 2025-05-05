#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module wrapper pour DeepSpeed qui fonctionne même sans le package installé.
Ce module permet d'avoir un comportement dégradé mais fonctionnel en cas d'absence
du package DeepSpeed, tout en maintenant une interface cohérente.
"""

import os
import logging
import platform
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import torch.nn as nn

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérifier si DeepSpeed est disponible
HAVE_DEEPSPEED = False
try:
    import deepspeed
    HAVE_DEEPSPEED = True
    logger.info("DeepSpeed est disponible et sera utilisé pour l'optimisation")
except ImportError:
    logger.warning("DeepSpeed n'est pas installé. Un mode de compatibilité limité sera utilisé.")
    logger.warning("Pour des performances optimales, installez DeepSpeed: pip install deepspeed")

class DeepSpeedCompatModel:
    """
    Classe de compatibilité qui émule l'interface de DeepSpeed sans avoir besoin du package.
    Cela permet au code de fonctionner même si DeepSpeed n'est pas installé.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer = None, **kwargs):
        """
        Initialise le modèle de compatibilité.
        
        Args:
            model: Modèle PyTorch
            optimizer: Optimiseur PyTorch
            **kwargs: Arguments supplémentaires (ignorés en mode compatibilité)
        """
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.device = next(model.parameters()).device
        
        logger.warning("Utilisation du mode de compatibilité DeepSpeed (performances non optimisées)")
    
    def backward(self, loss):
        """Émule la méthode backward de DeepSpeed."""
        loss.backward()
    
    def step(self):
        """Émule la méthode step de DeepSpeed."""
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def __call__(self, *args, **kwargs):
        """Appel du modèle sous-jacent."""
        return self.model(*args, **kwargs)
    
    @property
    def module(self):
        """Accès au module sous-jacent."""
        return self.model
    
    def eval(self):
        """Passe le modèle en mode évaluation."""
        self.model.eval()
    
    def train(self):
        """Passe le modèle en mode entraînement."""
        self.model.train()
    
    def save_checkpoint(self, save_dir, client_state=None):
        """Sauvegarde un checkpoint du modèle."""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, "model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'client_state': client_state
        }, checkpoint_path)
        logger.info(f"Checkpoint sauvegardé dans {checkpoint_path} (mode compatibilité)")
    
    def load_checkpoint(self, load_dir):
        """Charge un checkpoint du modèle."""
        checkpoint_path = os.path.join(load_dir, "model.pt")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        client_state = checkpoint.get('client_state', None)
        logger.info(f"Checkpoint chargé depuis {checkpoint_path} (mode compatibilité)")
        return self.model, client_state


def initialize_deepspeed(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    fp16: bool = True,
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
    **kwargs
) -> Union[Any, DeepSpeedCompatModel]:
    """
    Initialise un modèle avec DeepSpeed ou retourne un modèle de compatibilité.
    
    Args:
        model: Modèle PyTorch
        optimizer: Optimiseur PyTorch
        config: Configuration DeepSpeed (prioritaire sur les autres paramètres)
        fp16: Activer la précision mixte (float16)
        zero_stage: Niveau d'optimisation ZeRO (0, 1, 2, 3)
        offload_optimizer: Décharger l'état de l'optimiseur sur le CPU
        offload_parameters: Décharger les paramètres sur le CPU
        **kwargs: Arguments supplémentaires pour DeepSpeed
    
    Returns:
        Modèle optimisé avec DeepSpeed ou modèle de compatibilité
    """
    # Si DeepSpeed n'est pas disponible, retourner le modèle de compatibilité
    if not HAVE_DEEPSPEED:
        return DeepSpeedCompatModel(model, optimizer)
    
    # Créer une configuration si non fournie
    if config is None:
        config = {
            "train_batch_size": kwargs.get("train_batch_size", 32),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
            
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0
                }
            },
            
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": 1000
                }
            },
            
            "gradient_clipping": kwargs.get("max_grad_norm", 1.0),
            "wall_clock_breakdown": False,
            "zero_allow_untested_optimizer": True
        }
        
        # Configuration de ZeRO
        if zero_stage > 0:
            config["zero_optimization"] = {
                "stage": zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8
            }
            
            # Configuration du offload (ZeRO-Offload)
            if offload_optimizer or offload_parameters:
                config["zero_optimization"]["offload_optimizer"] = offload_optimizer
                if zero_stage >= 3 and offload_parameters:
                    config["zero_optimization"]["offload_param"] = offload_parameters
        
        # Configuration de la précision mixte (fp16)
        if fp16:
            config["fp16"] = {
                "enabled": True,
                "auto_cast": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
    
    # Initialiser le modèle DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
        model_parameters=list(model.parameters())
    )
    
    logger.info("Modèle initialisé avec DeepSpeed")
    return model_engine


def is_deepspeed_available():
    """
    Vérifie si DeepSpeed est disponible.
    
    Returns:
        bool: True si DeepSpeed est disponible
    """
    return HAVE_DEEPSPEED


def get_recommended_config() -> Dict[str, Any]:
    """
    Retourne une configuration DeepSpeed recommandée pour le système actuel.
    
    Returns:
        Configuration recommandée
    """
    # Détecter le système
    is_linux = platform.system() == "Linux"
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    # Configuration de base
    config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 1000
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": cuda_available,
            "auto_cast": True,
            "loss_scale": 0
        }
    }
    
    # ZeRO configuration optimisée pour le nombre de GPUs
    if cuda_available:
        # Sur des systèmes avec plusieurs GPUs, utiliser ZeRO-2
        if gpu_count > 1:
            config["zero_optimization"] = {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True
            }
        # Sur des systèmes avec un seul GPU, utiliser ZeRO-1
        else:
            config["zero_optimization"] = {
                "stage": 1,
                "contiguous_gradients": True
            }
        
        # Si nous sommes sur Linux, nous pouvons utiliser l'offload
        if is_linux:
            config["zero_optimization"]["offload_optimizer"] = True
    
    return config


if __name__ == "__main__":
    # Test simple
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Initialiser avec notre wrapper
    ds_model = initialize_deepspeed(model)
    
    # Afficher le statut
    print(f"DeepSpeed disponible: {HAVE_DEEPSPEED}")
    print(f"Type du modèle wrapper: {type(ds_model)}")
    
    # Test d'inférence
    test_input = torch.randn(5, 10)
    output = ds_model(test_input)
    print(f"Forme de sortie: {output.shape}") 