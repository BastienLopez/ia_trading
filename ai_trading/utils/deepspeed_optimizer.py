#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'optimisation DeepSpeed pour les modèles RL lourds et les LLMs.
Permet de réduire drastiquement l'utilisation mémoire et d'accélérer l'entraînement des grands modèles.
"""

import json
import logging
import os
from typing import Any, Dict, Optional
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
        if not HAVE_DEEPSPEED:
            raise ImportError(
                "DeepSpeed n'est pas installé. Utilisez 'pip install deepspeed' pour l'installer."
            )

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
        if local_rank != -1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

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

        # Inference sans calcul de gradient
        with torch.no_grad():
            outputs = self.ds_model(batch)

        # Revenir au mode entraînement
        self.ds_model.train()

        return outputs

    def save_checkpoint(self, path: Optional[str] = None):
        """
        Sauvegarde le modèle et l'état de l'optimiseur.

        Args:
            path: Chemin de sauvegarde (utilise checkpoint_dir si None).
        """
        client_state = {"model_config": {}}
        path = path or self.checkpoint_dir

        if path is not None:
            os.makedirs(path, exist_ok=True)
            self.ds_model.save_checkpoint(path, client_state=client_state)
            logger.info(f"Checkpoint sauvegardé dans {path}")
        else:
            logger.warning("Aucun chemin de sauvegarde spécifié.")

    def load_checkpoint(self, path: str):
        """
        Charge un checkpoint sauvegardé.

        Args:
            path: Chemin vers le checkpoint.
        """
        _, client_state = self.ds_model.load_checkpoint(path)
        logger.info(f"Checkpoint chargé depuis {path}")
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
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crée un fichier de configuration DeepSpeed.

    Args:
        zero_stage: Niveau de ZeRO (0, 1, 2 ou 3).
        fp16: Si True, active l'entraînement en demi-précision.
        offload_optimizer: Si True, décharge l'état de l'optimiseur sur le CPU.
        offload_parameters: Si True, décharge les paramètres sur le CPU.
        output_file: Chemin du fichier de sortie (si None, utilise un chemin par défaut).

    Returns:
        Configuration DeepSpeed sous forme de dictionnaire.
    """
    if not HAVE_DEEPSPEED:
        logger.warning(
            "DeepSpeed n'est pas installé. Utilisez 'pip install deepspeed' pour l'installer."
        )

    config = {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
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

    # Utiliser le répertoire par défaut si output_file n'est pas spécifié
    if output_file is None:
        # Créer le répertoire de configuration DeepSpeed s'il n'existe pas
        config_dir = Path("ai_trading/info_retour/config/deepspeed")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Générer un nom de fichier basé sur les paramètres
        config_name = f"ds_config_z{zero_stage}_fp{16 if fp16 else 32}"
        if offload_optimizer:
            config_name += "_offload_opt"
        if offload_parameters:
            config_name += "_offload_param"
        
        output_file = config_dir / f"{config_name}.json"

    # Écrire dans un fichier
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration DeepSpeed écrite dans {output_file}")

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
