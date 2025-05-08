"""
Module d'utilitaires pour l'entraînement distribué multi-GPU.

Ce module fournit des fonctions et des classes pour répartir l'entraînement d'un modèle
sur plusieurs GPUs en utilisant DistributedDataParallel (DDP) de PyTorch.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialise le processus de distribution pour DDP.

    Args:
        rank: L'ID du processus actuel
        world_size: Le nombre total de processus
        backend: Le backend de communication à utiliser ('nccl' pour GPU, 'gloo' pour CPU)
    """
    # Configuration de l'environnement
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialiser le groupe de processus
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    logger.info(f"Processus {rank} initialisé (sur {world_size} au total)")


def cleanup_ddp() -> None:
    """Nettoie le groupe de processus DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Groupe de processus DDP nettoyé")


def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
) -> DataLoader:
    """
    Prépare un DataLoader pour l'entraînement distribué.

    Args:
        dataset: Le dataset PyTorch
        batch_size: Taille du batch par GPU
        shuffle: Si True, mélange les données
        num_workers: Nombre de workers pour le chargement des données
        pin_memory: Si True, utilise la mémoire épinglée pour accélérer le transfert CPU->GPU
        rank: L'ID du processus actuel
        world_size: Le nombre total de processus
        distributed: Si True, utilise DistributedSampler

    Returns:
        Un DataLoader configuré pour l'entraînement distribué
    """
    # Utiliser DistributedSampler si en mode distribué
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        shuffle = False  # Le sampler gère déjà le mélange

    # Créer le DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not distributed else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=True if distributed else False,
    )

    return dataloader


def train_ddp_model(
    model_fn: Callable[[], torch.nn.Module],
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    epochs: int = 10,
    optimizer_fn: Optional[Callable[[torch.nn.Module], torch.optim.Optimizer]] = None,
    criterion_fn: Optional[Callable[[], torch.nn.Module]] = None,
    lr_scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
    num_workers: int = 4,
    world_size: Optional[int] = None,
    backend: str = "nccl",
    find_unused_parameters: bool = False,
    mixed_precision: bool = False,
    checkpoint_dir: Optional[str] = None,
    train_step_fn: Optional[Callable] = None,
    val_step_fn: Optional[Callable] = None,
    save_best_only: bool = True,
) -> Dict[str, Any]:
    """
    Entraîne un modèle avec DistributedDataParallel sur tous les GPUs disponibles.

    Args:
        model_fn: Fonction qui retourne une instance du modèle
        train_dataset: Dataset d'entraînement
        val_dataset: Dataset de validation (optionnel)
        batch_size: Taille du batch par GPU
        epochs: Nombre d'époques d'entraînement
        optimizer_fn: Fonction qui prend le modèle et retourne un optimiseur
        criterion_fn: Fonction qui retourne une fonction de perte
        lr_scheduler_fn: Fonction qui prend l'optimiseur et retourne un scheduler
        num_workers: Nombre de workers pour le DataLoader
        world_size: Nombre de GPUs à utiliser (défaut: tous disponibles)
        backend: Backend de communication ('nccl' pour GPU, 'gloo' pour CPU)
        find_unused_parameters: Si True, cherche les paramètres non utilisés dans le forward pass
        mixed_precision: Si True, utilise l'entraînement en précision mixte
        checkpoint_dir: Répertoire pour sauvegarder les checkpoints
        train_step_fn: Fonction personnalisée pour l'étape d'entraînement
        val_step_fn: Fonction personnalisée pour l'étape de validation
        save_best_only: Si True, sauvegarde uniquement le meilleur modèle

    Returns:
        Dictionnaire contenant les métriques d'entraînement
    """
    # Déterminer le nombre de GPUs à utiliser
    if world_size is None:
        world_size = torch.cuda.device_count()

    if world_size <= 1:
        logger.warning(
            "Un seul GPU détecté ou spécifié. L'entraînement distribué n'est pas nécessaire."
        )
        # Ici on pourrait entraîner de manière non-distribuée
        return {}

    logger.info(f"Lancement de l'entraînement distribué sur {world_size} GPUs")

    # Fonction d'entraînement qui sera exécutée par chaque processus
    def train_worker(rank, world_size, results_queue=None):
        # Initialiser le processus DDP
        setup_ddp(rank, world_size, backend)

        # Définir le device pour ce processus
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Créer le modèle et le déplacer sur le GPU
        model = model_fn().to(device)

        # Wrapper le modèle avec DDP
        ddp_model = DDP(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=rank if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused_parameters,
        )

        # Créer l'optimiseur
        optimizer = (
            optimizer_fn(ddp_model.parameters())
            if optimizer_fn
            else torch.optim.Adam(ddp_model.parameters())
        )

        # Créer la fonction de perte
        criterion = criterion_fn() if criterion_fn else torch.nn.MSELoss()

        # Créer le scheduler
        scheduler = lr_scheduler_fn(optimizer) if lr_scheduler_fn else None

        # Créer un GradScaler pour la précision mixte si nécessaire
        scaler = (
            torch.cuda.amp.GradScaler()
            if mixed_precision and torch.cuda.is_available()
            else None
        )

        # Créer les dataloaders
        train_dataloader = prepare_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            rank=rank,
            world_size=world_size,
            distributed=True,
        )

        val_dataloader = None
        if val_dataset:
            val_dataloader = prepare_dataloader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                rank=rank,
                world_size=world_size,
                distributed=True,
            )

        # Historique d'entraînement
        history = {"train_loss": [], "val_loss": [] if val_dataloader else None}

        best_val_loss = float("inf")

        # Entraînement principal
        for epoch in range(epochs):
            # Réinitialiser le sampler à chaque époque pour un nouveau mélange
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            # Phase d'entraînement
            ddp_model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_dataloader):
                # Utiliser une fonction d'étape personnalisée si fournie
                if train_step_fn:
                    batch_loss = train_step_fn(
                        ddp_model, batch, optimizer, criterion, device, scaler=scaler
                    )
                else:
                    # Étape d'entraînement standard
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    else:
                        inputs = batch.to(device)
                        targets = None

                    # Réinitialiser les gradients
                    optimizer.zero_grad()

                    # Forward pass avec précision mixte si activée
                    if mixed_precision and scaler:
                        with torch.cuda.amp.autocast():
                            outputs = ddp_model(inputs)
                            loss = (
                                criterion(outputs, targets)
                                if targets is not None
                                else outputs
                            )

                        # Backward et optimize avec scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Forward pass standard
                        outputs = ddp_model(inputs)
                        loss = (
                            criterion(outputs, targets)
                            if targets is not None
                            else outputs
                        )

                        # Backward et optimize
                        loss.backward()
                        optimizer.step()

                    batch_loss = loss.item()

                train_loss += batch_loss

                # Logs pour le processus principal uniquement
                if rank == 0 and (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Époque {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {batch_loss:.4f}"
                    )

            # Calculer la perte moyenne d'entraînement
            train_loss = train_loss / len(train_dataloader)
            history["train_loss"].append(train_loss)

            # Phase de validation si un dataset de validation est fourni
            val_loss = None
            if val_dataloader:
                ddp_model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_dataloader:
                        # Utiliser une fonction d'étape personnalisée si fournie
                        if val_step_fn:
                            batch_loss = val_step_fn(
                                ddp_model, batch, criterion, device
                            )
                        else:
                            # Étape de validation standard
                            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                                inputs, targets = batch
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                            else:
                                inputs = batch.to(device)
                                targets = None

                            # Forward pass
                            outputs = ddp_model(inputs)
                            loss = (
                                criterion(outputs, targets)
                                if targets is not None
                                else outputs
                            )
                            batch_loss = loss.item()

                        val_loss += batch_loss

                # Calculer la perte moyenne de validation
                val_loss = val_loss / len(val_dataloader)
                history["val_loss"].append(val_loss)

                # Logs pour le processus principal uniquement
                if rank == 0:
                    logger.info(
                        f"Époque {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                # Sauvegarder le checkpoint si c'est le meilleur modèle
                if rank == 0 and checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    if not save_best_only or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_file = os.path.join(
                            checkpoint_dir,
                            f"model_epoch_{epoch+1}_valloss_{val_loss:.4f}.pt",
                        )

                        # Sauvegarder uniquement les poids du modèle
                        torch.save(ddp_model.module.state_dict(), checkpoint_file)
                        logger.info(f"Checkpoint sauvegardé: {checkpoint_file}")
            else:
                # Logs pour le processus principal uniquement (sans validation)
                if rank == 0:
                    logger.info(
                        f"Époque {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
                    )

            # Mettre à jour le scheduler si fourni
            if scheduler:
                scheduler.step()

        # Nettoyer le processus DDP
        cleanup_ddp()

        # Retourner l'historique d'entraînement via la queue si fournie
        if rank == 0 and results_queue is not None:
            results_queue.put(history)

        return history

    # Lancer le processus d'entraînement distribué
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

    # Note: on ne peut pas facilement récupérer l'historique de tous les processus
    # Une solution pourrait être d'utiliser une queue multiprocessing ou de sauvegarder l'historique
    # dans des fichiers et de les lire après la fin de l'entraînement

    return {}


class DDPModelWrapper:
    """
    Wrapper pour faciliter l'utilisation de DistributedDataParallel.
    """

    def __init__(
        self,
        model_fn: Callable[[], torch.nn.Module],
        optimizer_fn: Optional[
            Callable[[torch.nn.Module], torch.optim.Optimizer]
        ] = None,
        criterion_fn: Optional[Callable[[], torch.nn.Module]] = None,
        lr_scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
        world_size: Optional[int] = None,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        mixed_precision: bool = False,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialise le wrapper DDP.

        Args:
            model_fn: Fonction qui retourne une instance du modèle
            optimizer_fn: Fonction qui prend le modèle et retourne un optimiseur
            criterion_fn: Fonction qui retourne une fonction de perte
            lr_scheduler_fn: Fonction qui prend l'optimiseur et retourne un scheduler
            world_size: Nombre de GPUs à utiliser (défaut: tous disponibles)
            backend: Backend de communication ('nccl' pour GPU, 'gloo' pour CPU)
            find_unused_parameters: Si True, cherche les paramètres non utilisés dans le forward pass
            mixed_precision: Si True, utilise l'entraînement en précision mixte
            checkpoint_dir: Répertoire pour sauvegarder les checkpoints
        """
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.criterion_fn = criterion_fn
        self.lr_scheduler_fn = lr_scheduler_fn

        # Déterminer le nombre de GPUs à utiliser
        self.world_size = (
            world_size if world_size is not None else torch.cuda.device_count()
        )
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = checkpoint_dir

        # Vérifier si le multi-GPU est disponible
        available_gpus = count_available_gpus()
        if self.world_size <= 1 or available_gpus == 0:
            logger.warning(
                "Un seul GPU détecté ou spécifié, ou aucun GPU disponible. L'entraînement distribué n'est pas utilisé."
            )
            self.use_ddp = False
        else:
            self.use_ddp = True
            logger.info(
                f"Configuration pour l'entraînement distribué sur {self.world_size} GPUs"
            )

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        epochs: int = 10,
        num_workers: int = 4,
        train_step_fn: Optional[Callable] = None,
        val_step_fn: Optional[Callable] = None,
        save_best_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle sur plusieurs GPUs en utilisant DDP.

        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation (optionnel)
            batch_size: Taille du batch par GPU
            epochs: Nombre d'époques d'entraînement
            num_workers: Nombre de workers pour le DataLoader
            train_step_fn: Fonction personnalisée pour l'étape d'entraînement
            val_step_fn: Fonction personnalisée pour l'étape de validation
            save_best_only: Si True, sauvegarde uniquement le meilleur modèle

        Returns:
            Dictionnaire contenant les métriques d'entraînement
        """
        if not self.use_ddp:
            logger.warning("Basculement sur l'entraînement standard (non-distribué)")
            return self._train_single_gpu(
                train_dataset,
                val_dataset,
                batch_size,
                epochs,
                num_workers,
                train_step_fn,
                val_step_fn,
                save_best_only,
            )

        # Utiliser la fonction d'entraînement DDP
        return train_ddp_model(
            model_fn=self.model_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_fn=self.optimizer_fn,
            criterion_fn=self.criterion_fn,
            lr_scheduler_fn=self.lr_scheduler_fn,
            num_workers=num_workers,
            world_size=self.world_size,
            backend=self.backend,
            find_unused_parameters=self.find_unused_parameters,
            mixed_precision=self.mixed_precision,
            checkpoint_dir=self.checkpoint_dir,
            train_step_fn=train_step_fn,
            val_step_fn=val_step_fn,
            save_best_only=save_best_only,
        )

    def _train_single_gpu(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        epochs: int = 10,
        num_workers: int = 4,
        train_step_fn: Optional[Callable] = None,
        val_step_fn: Optional[Callable] = None,
        save_best_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Implémentation d'entraînement sur un seul GPU (fallback).

        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation (optionnel)
            batch_size: Taille du batch
            epochs: Nombre d'époques d'entraînement
            num_workers: Nombre de workers pour le DataLoader
            train_step_fn: Fonction personnalisée pour l'étape d'entraînement
            val_step_fn: Fonction personnalisée pour l'étape de validation
            save_best_only: Si True, sauvegarde uniquement le meilleur modèle

        Returns:
            Dictionnaire contenant les métriques d'entraînement
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Créer le modèle et le déplacer sur le GPU
        model = self.model_fn().to(device)

        # Créer l'optimiseur
        optimizer = (
            self.optimizer_fn(model.parameters())
            if self.optimizer_fn
            else torch.optim.Adam(model.parameters())
        )

        # Créer la fonction de perte
        criterion = self.criterion_fn() if self.criterion_fn else torch.nn.MSELoss()

        # Créer le scheduler
        scheduler = self.lr_scheduler_fn(optimizer) if self.lr_scheduler_fn else None

        # Créer un GradScaler pour la précision mixte si nécessaire
        scaler = (
            torch.cuda.amp.GradScaler()
            if self.mixed_precision and torch.cuda.is_available()
            else None
        )

        # Créer les dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        # Historique d'entraînement
        history = {"train_loss": [], "val_loss": [] if val_dataloader else None}

        best_val_loss = float("inf")

        # Entraînement principal
        for epoch in range(epochs):
            # Phase d'entraînement
            model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_dataloader):
                # Utiliser une fonction d'étape personnalisée si fournie
                if train_step_fn:
                    batch_loss = train_step_fn(
                        model, batch, optimizer, criterion, device, scaler=scaler
                    )
                else:
                    # Étape d'entraînement standard
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    else:
                        inputs = batch.to(device)
                        targets = None

                    # Réinitialiser les gradients
                    optimizer.zero_grad()

                    # Forward pass avec précision mixte si activée
                    if self.mixed_precision and scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = (
                                criterion(outputs, targets)
                                if targets is not None
                                else outputs
                            )

                        # Backward et optimize avec scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Forward pass standard
                        outputs = model(inputs)
                        loss = (
                            criterion(outputs, targets)
                            if targets is not None
                            else outputs
                        )

                        # Backward et optimize
                        loss.backward()
                        optimizer.step()

                    batch_loss = loss.item()

                train_loss += batch_loss

                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Époque {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {batch_loss:.4f}"
                    )

            # Calculer la perte moyenne d'entraînement
            train_loss = train_loss / len(train_dataloader)
            history["train_loss"].append(train_loss)

            # Phase de validation si un dataset de validation est fourni
            val_loss = None
            if val_dataloader:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_dataloader:
                        # Utiliser une fonction d'étape personnalisée si fournie
                        if val_step_fn:
                            batch_loss = val_step_fn(model, batch, criterion, device)
                        else:
                            # Étape de validation standard
                            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                                inputs, targets = batch
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                            else:
                                inputs = batch.to(device)
                                targets = None

                            # Forward pass
                            outputs = model(inputs)
                            loss = (
                                criterion(outputs, targets)
                                if targets is not None
                                else outputs
                            )
                            batch_loss = loss.item()

                        val_loss += batch_loss

                # Calculer la perte moyenne de validation
                val_loss = val_loss / len(val_dataloader)
                history["val_loss"].append(val_loss)

                logger.info(
                    f"Époque {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Sauvegarder le checkpoint si c'est le meilleur modèle
                if self.checkpoint_dir:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)

                    if not save_best_only or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_file = os.path.join(
                            self.checkpoint_dir,
                            f"model_epoch_{epoch+1}_valloss_{val_loss:.4f}.pt",
                        )

                        # Sauvegarder uniquement les poids du modèle
                        torch.save(model.state_dict(), checkpoint_file)
                        logger.info(f"Checkpoint sauvegardé: {checkpoint_file}")
            else:
                logger.info(f"Époque {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

            # Mettre à jour le scheduler si fourni
            if scheduler:
                scheduler.step()

        return history


def count_available_gpus() -> int:
    """
    Compte le nombre de GPUs disponibles pour l'entraînement.

    Returns:
        Nombre de GPUs disponibles
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_gpu_memory_usage() -> List[Dict[str, Union[int, str]]]:
    """
    Obtient l'utilisation mémoire actuelle de tous les GPUs.

    Returns:
        Liste de dictionnaires contenant les informations d'utilisation mémoire
    """
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB

        gpu_info.append(
            {
                "device": i,
                "name": torch.cuda.get_device_name(i),
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved,
            }
        )

    return gpu_info
