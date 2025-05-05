#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module wrapper pour DeepSpeed qui fonctionne même sans le package installé.

Ce module fournit une interface simplifiée qui peut fonctionner avec ou sans l'installation
du package DeepSpeed, tout en maintenant une interface cohérente.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérifier si DeepSpeed est disponible
HAVE_DEEPSPEED = False
try:
    import deepspeed

    HAVE_DEEPSPEED = True
    logger.info("DeepSpeed est disponible et sera utilisé pour l'optimisation")
except ImportError:
    logger.warning(
        "DeepSpeed n'est pas installé. Un mode de compatibilité limité sera utilisé."
    )
    logger.warning(
        "Pour des performances optimales, installez DeepSpeed: pip install deepspeed"
    )

# Créer le répertoire de configuration DeepSpeed
CONFIG_DIR = Path("ai_trading/info_retour/config/deepspeed")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Créer un fichier de configuration par défaut s'il n'existe pas déjà
DEFAULT_CONFIG_PATH = CONFIG_DIR / "ds_config_default.json"
if not DEFAULT_CONFIG_PATH.exists() and HAVE_DEEPSPEED:
    try:
        # Importer la fonction create_deepspeed_config depuis le même répertoire
        # Utiliser une importation relative pour éviter les erreurs d'importation circulaire
        from .deepspeed_optimizer import create_deepspeed_config
        
        # Créer la configuration par défaut
        create_deepspeed_config(
            zero_stage=2,
            fp16=True,
            offload_optimizer=False,
            offload_parameters=False,
            output_file=str(DEFAULT_CONFIG_PATH)
        )
        logger.info(f"Configuration DeepSpeed par défaut créée dans {DEFAULT_CONFIG_PATH}")
    except ImportError:
        logger.warning("Impossible de créer le fichier de configuration DeepSpeed par défaut")

class DeepSpeedCompatModel:
    """
    Classe de compatibilité qui émule l'interface de DeepSpeed sans avoir besoin du package.
    Cela permet au code de fonctionner même si DeepSpeed n'est pas installé.
    """

    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Initialise un modèle compatible DeepSpeed.

        Args:
            model: Modèle PyTorch
            optimizer: Optimiseur PyTorch (optionnel)
        """
        self.model = model
        self.optimizer = optimizer
        logger.warning(
            "Utilisation du mode de compatibilité DeepSpeed (performances non optimisées)"
        )

    def backward(self, loss: torch.Tensor):
        """Émule la méthode backward de DeepSpeed."""
        loss.backward()

    def step(self):
        """Émule la méthode step de DeepSpeed."""
        if self.optimizer:
            self.optimizer.step()

    def __call__(self, *args, **kwargs):
        """Appelle le modèle sous-jacent."""
        return self.model(*args, **kwargs)

    def train(self):
        """Mets le modèle en mode entraînement."""
        self.model.train()

    def eval(self):
        """Mets le modèle en mode évaluation."""
        self.model.eval()

    def to(self, device):
        """Déplace le modèle sur le périphérique spécifié."""
        self.model.to(device)
        return self

    def load_checkpoint(self, checkpoint_dir, **kwargs):
        """
        Émule le chargement d'un checkpoint.
        Retourne None, {} pour simuler le comportement de DeepSpeed.
        """
        logger.warning(
            "Chargement de checkpoint en mode de compatibilité DeepSpeed (ignoré)"
        )
        return None, {}

    def save_checkpoint(self, save_dir, **kwargs):
        """Émule la sauvegarde d'un checkpoint."""
        logger.warning(
            "Sauvegarde de checkpoint en mode de compatibilité DeepSpeed (ignorée)"
        )

    def module(self):
        """Retourne le modèle sous-jacent."""
        return self.model


def initialize_deepspeed(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    fp16: bool = True,
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
    **kwargs,
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
                    "weight_decay": 0,
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": 1000,
                },
            },
            "gradient_clipping": kwargs.get("max_grad_norm", 1.0),
            "wall_clock_breakdown": False,
            "zero_allow_untested_optimizer": True,
        }

        # Configuration de ZeRO
        if zero_stage > 0:
            config["zero_optimization"] = {
                "stage": zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
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
                "min_loss_scale": 1,
            }

    # Initialiser le modèle DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
        model_parameters=list(model.parameters()),
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
                "weight_decay": 0,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "fp16": {"enabled": cuda_available, "auto_cast": True, "loss_scale": 0},
    }

    # ZeRO configuration optimisée pour le nombre de GPUs
    if cuda_available:
        # Sur des systèmes avec plusieurs GPUs, utiliser ZeRO-2
        if gpu_count > 1:
            config["zero_optimization"] = {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
            }
        # Sur des systèmes avec un seul GPU, utiliser ZeRO-1
        else:
            config["zero_optimization"] = {"stage": 1, "contiguous_gradients": True}

        # Si nous sommes sur Linux, nous pouvons utiliser l'offload
        if is_linux:
            config["zero_optimization"]["offload_optimizer"] = True

    return config


if __name__ == "__main__":
    # Test simple
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

    # Initialiser avec notre wrapper
    ds_model = initialize_deepspeed(model)

    # Afficher le statut
    print(f"DeepSpeed disponible: {HAVE_DEEPSPEED}")
    print(f"Type du modèle wrapper: {type(ds_model)}")

    # Test d'inférence
    test_input = torch.randn(5, 10)
    output = ds_model(test_input)
    print(f"Forme de sortie: {output.shape}")
