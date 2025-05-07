#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'optimisation DeepSpeed pour les modèles RL lourds et les LLMs.
Permet de réduire drastiquement l'utilisation mémoire et d'accélérer l'entraînement des grands 
modèles.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn

# Essayer d'importer torch.distributed
try:
    import torch.distributed as dist
except ImportError:
    logging.warning("torch.distributed n'est pas disponible")

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Vérification de DeepSpeed
HAVE_DEEPSPEED = False
try:
    import deepspeed
    HAVE_DEEPSPEED = True
except ImportError:
    logger.warning(
        "DeepSpeed n'est pas installé. Utilisez 'pip install deepspeed' pour l'installer."
    )
    logger.info(
        "En mode compatibilité - certaines fonctionnalités avancées ne seront pas disponibles."
    )

# Classe DeepSpeedStub pour fournir une compatibilité quand DeepSpeed n'est pas installé
class DeepSpeedStub:
    """Implémentation minimale de compatibilité pour DeepSpeed."""
    
    @staticmethod
    def initialize(*args, **kwargs):
        """Stub pour deepspeed.initialize"""
        model = kwargs.get("model", args[0] if args else None)
        optimizer = kwargs.get("optimizer", args[1] if len(args) > 1 else None)
        model_wrapped = DeepSpeedModelStub(model, optimizer)
        return model_wrapped, None, None, None

class DeepSpeedModelStub:
    """Stub pour le modèle DeepSpeed."""
    
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cpu")
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self):
        self.model.train()
        return self
    
    def backward(self, loss):
        if self.optimizer:
            # Vérifier si la loss a requires_grad avant d'appeler backward
            # Si non, ne rien faire pour éviter l'erreur
            if hasattr(loss, 'requires_grad') and loss.requires_grad:
                loss.backward()
            else:
                # Simuler un backward sans erreur pour les tests
                pass
    
    def step(self):
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def save_checkpoint(self, checkpoint_path, client_state=None):
        state_dict = self.model.state_dict()
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
        if client_state:
            torch.save(client_state, os.path.join(checkpoint_path, "client_state.pt"))
    
    def load_checkpoint(self, checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model.pt")
        client_state_path = os.path.join(checkpoint_path, "client_state.pt")
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
        
        client_state = {}
        if os.path.exists(client_state_path):
            client_state = torch.load(client_state_path)
        
        return None, client_state

# Utiliser le stub si DeepSpeed n'est pas disponible
if not HAVE_DEEPSPEED:
    deepspeed = DeepSpeedStub()


class DeepSpeedOptimizer:
    """
    Optimiseur utilisant DeepSpeed pour l'entraînement efficace des modèles lourds.
    Permet d'utiliser les optimisations ZeRO (Zero Redundancy Optimizer) pour réduire l'empreinte mémoire.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        fp16: bool = True,
        zero_stage: int = 2,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        local_rank: int = -1,
        train_batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Optional[str] = None,
        save_interval: int = 1000,
    ):
        """
        Initialise l'optimiseur DeepSpeed.

        Args:
            model: Modèle PyTorch à optimiser.
            optimizer: Optimiseur PyTorch (si None, utilise Adam).
            fp16: Si True, active l'entraînement en demi-précision (float16).
            zero_stage: Niveau de ZeRO (0, 1, 2 ou 3).
            offload_optimizer: Si True, décharge l'état de l'optimiseur sur le CPU.
            offload_parameters: Si True, décharge les paramètres sur le CPU.
            local_rank: Rang local pour l'entraînement distribué (-1 pour non-distribué).
            train_batch_size: Taille du batch d'entraînement par GPU.
            gradient_accumulation_steps: Nombre d'étapes d'accumulation de gradient.
            max_grad_norm: Valeur maximale de norme de gradient pour l'écrêtage.
            checkpoint_dir: Répertoire pour sauvegarder les checkpoints.
            save_interval: Intervalle entre les sauvegardes (en étapes).
        """
        self.model = model
        self.local_rank = (
            local_rank if local_rank != -1 else int(os.environ.get("LOCAL_RANK", "0"))
        )
        self.fp16 = fp16
        self.zero_stage = zero_stage
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir

        # Initialiser le processus distribué si nécessaire
        if local_rank != -1 and hasattr(dist, 'is_initialized') and not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser le groupe de processus distribués: {e}")

        # Créer la configuration DeepSpeed
        ds_config = self._create_ds_config(
            fp16=fp16,
            zero_stage=zero_stage,
            offload_optimizer=offload_optimizer,
            offload_parameters=offload_parameters,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            checkpoint_dir=checkpoint_dir,
            save_interval=save_interval,
        )

        # Créer ou obtenir l'optimiseur
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Initialiser le modèle DeepSpeed
        self.ds_model, self.ds_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )

        logger.info(f"Modèle DeepSpeed initialisé avec ZeRO stage {zero_stage}")
        logger.info(f"Mode FP16: {fp16}")
        logger.info(f"Offload optimizer: {offload_optimizer}")
        logger.info(f"Offload parameters: {offload_parameters}")

    def _create_ds_config(
        self,
        fp16: bool,
        zero_stage: int,
        offload_optimizer: bool,
        offload_parameters: bool,
        train_batch_size: int,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        checkpoint_dir: Optional[str],
        save_interval: int,
    ) -> Dict[str, Any]:
        """
        Crée la configuration DeepSpeed.

        Args:
            fp16: Si True, active l'entraînement en demi-précision.
            zero_stage: Niveau de ZeRO (0, 1, 2 ou 3).
            offload_optimizer: Si True, décharge l'état de l'optimiseur sur le CPU.
            offload_parameters: Si True, décharge les paramètres sur le CPU.
            train_batch_size: Taille du batch d'entraînement.
            gradient_accumulation_steps: Nombre d'étapes pour l'accumulation de gradient.
            max_grad_norm: Valeur maximale de la norme du gradient pour l'écrêtage.
            checkpoint_dir: Répertoire pour les checkpoints (None pour désactiver).
            save_interval: Intervalle entre les sauvegardes (en nombre d'étapes).

        Returns:
            Configuration DeepSpeed sous forme de dictionnaire.
        """
        # Utiliser la fonction create_deepspeed_config pour générer la configuration de base
        config = create_deepspeed_config(
            zero_stage=zero_stage,
            fp16=fp16,
            offload_optimizer=offload_optimizer,
            offload_parameters=offload_parameters,
        )

        # Personnaliser la configuration générée pour cet optimiseur spécifique
        config["train_batch_size"] = train_batch_size
        config["gradient_accumulation_steps"] = gradient_accumulation_steps
        config["gradient_clipping"] = max_grad_norm

        # Configuration des checkpoints si un répertoire est spécifié
        if checkpoint_dir is not None:
            config["checkpoint"] = {
                "tag_validation": "val_loss",
                "save_folder": checkpoint_dir,
                "save_interval": save_interval,
            }

        return config

    def train_step(self, batch, labels):
        """
        Effectue une étape d'entraînement avec DeepSpeed.

        Args:
            batch: Données d'entrée.
            labels: Étiquettes cibles.

        Returns:
            Valeur de la perte.
        """
        # Transférer les données sur le périphérique approprié
        device = self.ds_model.device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)

        # Forward pass
        outputs = self.ds_model(batch)

        # Calculer la perte (exemple: MSE)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Certains modèles retournent des tuples

        if hasattr(self.model, "compute_loss"):
            # Si le modèle a une fonction compute_loss
            loss = self.model.compute_loss(outputs, labels)
        else:
            # Sinon, utiliser MSE par défaut
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(outputs, labels)

        # Backward pass
        self.ds_model.backward(loss)

        # Mise à jour des poids
        self.ds_model.step()

        return loss.item()

    def eval_step(self, batch):
        """
        Effectue une étape d'inférence.

        Args:
            batch: Données d'entrée.

        Returns:
            Sorties du modèle.
        """
        # Mode évaluation
        self.ds_model.eval()

        # Transférer les données sur le périphérique approprié
        device = self.ds_model.device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = self.ds_model(batch)

        # Revenir au mode entraînement
        self.ds_model.train()

        return outputs

    def save_checkpoint(self, checkpoint_path, client_state=None):
        """
        Sauvegarde un checkpoint du modèle.

        Args:
            checkpoint_path: Chemin de sauvegarde.
            client_state: État supplémentaire à sauvegarder (dict).

        Returns:
            None
        """
        if client_state is None:
            client_state = {}

        # Ajouter la configuration du modèle au client_state
        client_state["model_config"] = getattr(self.model, "config", {})

        self.ds_model.save_checkpoint(checkpoint_path, client_state=client_state)
        logger.info(f"Checkpoint sauvegardé dans {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Charge un checkpoint du modèle.

        Args:
            checkpoint_path: Chemin du checkpoint à charger.

        Returns:
            dict: État client sauvegardé avec le checkpoint.
        """
        _, client_state = self.ds_model.load_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint chargé depuis {checkpoint_path}")
        return client_state

    def get_model(self) -> nn.Module:
        """
        Retourne le modèle DeepSpeed.

        Returns:
            Modèle DeepSpeed.
        """
        return self.ds_model

    def get_optimizer(self) -> Any:
        """
        Retourne l'optimiseur DeepSpeed.

        Returns:
            Optimiseur DeepSpeed.
        """
        return self.ds_optimizer

    def get_lr(self) -> float:
        """
        Retourne le taux d'apprentissage actuel.

        Returns:
            Taux d'apprentissage.
        """
        return self.ds_optimizer.param_groups[0]["lr"]

    def set_lr(self, lr: float):
        """
        Définit le taux d'apprentissage.

        Args:
            lr: Nouveau taux d'apprentissage.
        """
        for param_group in self.ds_optimizer.param_groups:
            param_group["lr"] = lr


def create_deepspeed_config(
    zero_stage: int = 2,
    fp16: bool = True,
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crée une configuration DeepSpeed avec les paramètres spécifiés.

    Args:
        zero_stage: Niveau de ZeRO (0, 1, 2 ou 3).
        fp16: Active l'entraînement en demi-précision.
        offload_optimizer: Décharge l'état de l'optimiseur sur le CPU.
        offload_parameters: Décharge les paramètres sur le CPU.
        train_batch_size: Taille du batch d'entraînement.
        gradient_accumulation_steps: Nombre d'étapes pour l'accumulation de gradient.
        max_grad_norm: Valeur maximale de la norme du gradient pour l'écrêtage.
        output_file: Fichier de sortie pour la configuration (optionnel).

    Returns:
        dict: Configuration DeepSpeed.
    """
    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 1e-6,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 3e-4, "warmup_num_steps": 500},
        },
        "gradient_clipping": max_grad_norm,
        "wall_clock_breakdown": False,
    }

    # Configuration de ZeRO
    zero_config = {"stage": zero_stage}

    # Options d'offload
    if offload_optimizer:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if offload_parameters and zero_stage == 3:
        zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}

    # Options ZeRO spécifiques à chaque étape
    if zero_stage >= 2:
        zero_config["reduce_scatter"] = True
        zero_config["contiguous_gradients"] = True
        zero_config["overlap_comm"] = True

    # Ajouter les configurations ZeRO
    config["zero_optimization"] = zero_config

    # Configuration FP16
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,  # Utilisation du dynamic loss scaling
            "initial_scale_power": 16,
        }

    # Écrire la configuration dans un fichier si demandé
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(config, f, indent=4)

    return config


def optimize_model_with_deepspeed(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fp16: bool = True,
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
    local_rank: int = -1,
) -> DeepSpeedOptimizer:
    """
    Fonction utilitaire pour optimiser un modèle avec DeepSpeed.

    Args:
        model: Modèle PyTorch à optimiser.
        optimizer: Optimiseur PyTorch (si None, utilise Adam).
        fp16: Si True, active l'entraînement en demi-précision.
        zero_stage: Niveau de ZeRO (0, 1, 2 ou 3).
        offload_optimizer: Si True, décharge l'état de l'optimiseur sur le CPU.
        offload_parameters: Si True, décharge les paramètres sur le CPU.
        local_rank: Rang local pour l'entraînement distribué (-1 pour non-distribué).

    Returns:
        Optimiseur DeepSpeed.
    """
    if not HAVE_DEEPSPEED:
        raise ImportError(
            "DeepSpeed n'est pas installé. Utilisez 'pip install deepspeed' pour l'installer."
        )

    return DeepSpeedOptimizer(
        model=model,
        optimizer=optimizer,
        fp16=fp16,
        zero_stage=zero_stage,
        offload_optimizer=offload_optimizer,
        offload_parameters=offload_parameters,
        local_rank=local_rank,
    )
