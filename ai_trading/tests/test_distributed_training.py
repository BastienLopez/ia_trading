"""
Tests unitaires pour le module d'entraînement distribué multi-GPU.

Ces tests vérifient que les fonctionnalités d'entraînement distribué avec
DistributedDataParallel (DDP) fonctionnent correctement.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.utils.distributed_training import (
    DDPModelWrapper,
    count_available_gpus,
    get_gpu_memory_usage,
    prepare_dataloader,
)


class SimpleModel(nn.Module):
    """Un modèle simple pour les tests."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def test_gpu_detection():
    """Teste la détection des GPUs."""
    num_gpus = count_available_gpus()

    # Vérifier que la fonction renvoie un entier
    assert isinstance(num_gpus, int)

    # Vérifier que le nombre est cohérent
    if torch.cuda.is_available():
        assert num_gpus == torch.cuda.device_count()
    else:
        assert num_gpus == 0


def test_gpu_memory_info():
    """Teste l'obtention des informations sur la mémoire GPU."""
    gpu_info = get_gpu_memory_usage()

    # Vérifier que la fonction renvoie une liste
    assert isinstance(gpu_info, list)

    # Vérifier que les informations sont cohérentes
    if torch.cuda.is_available():
        assert len(gpu_info) == torch.cuda.device_count()
        for gpu in gpu_info:
            assert "device" in gpu
            assert "name" in gpu
            assert "allocated_gb" in gpu
            assert "reserved_gb" in gpu
            assert "total_gb" in gpu
            assert "free_gb" in gpu
    else:
        assert len(gpu_info) == 0


def test_dataloader_preparation():
    """Teste la préparation du DataLoader."""
    # Créer un dataset de test
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100, 1)).float()
    dataset = TensorDataset(inputs, targets)

    # Tester le DataLoader standard
    dataloader = prepare_dataloader(
        dataset, batch_size=16, shuffle=True, num_workers=0, distributed=False
    )

    # Vérifier les propriétés du DataLoader
    assert dataloader.batch_size == 16
    assert dataloader.num_workers == 0
    assert dataloader.pin_memory == True

    # Vérifier que ce n'est pas un DistributedSampler
    from torch.utils.data.distributed import DistributedSampler

    assert not isinstance(dataloader.sampler, DistributedSampler)

    # Tester une itération dans le DataLoader
    batch = next(iter(dataloader))
    assert len(batch) == 2  # inputs, targets
    assert batch[0].shape[0] <= 16  # batch_size
    assert batch[0].shape[1] == 10  # input_dim


# Ce test ne s'exécute que si CUDA est disponible et qu'il y a au moins un GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA n'est pas disponible")
def test_ddp_model_wrapper_single_gpu():
    """Teste le wrapper DDP en mode GPU unique."""
    # Créer un dataset de test
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100, 1)).float()
    dataset = TensorDataset(inputs, targets)

    # Créer le wrapper DDP avec une seule GPU
    ddp_wrapper = DDPModelWrapper(
        model_fn=lambda: SimpleModel(input_dim=10),
        optimizer_fn=lambda params: optim.Adam(params, lr=0.001),
        criterion_fn=lambda: nn.BCELoss(),
        world_size=1,  # Forcer l'utilisation d'un seul GPU
    )

    # Vérifier que le wrapper est configuré pour fallback sur single-GPU
    assert ddp_wrapper.use_ddp == False

    # Entraîner le modèle sur une époque
    history = ddp_wrapper.train(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=16,
        epochs=1,
        num_workers=0,
    )

    # Vérifier que l'entraînement a fonctionné
    assert "train_loss" in history
    assert len(history["train_loss"]) == 1  # Une époque
    assert history["train_loss"][0] > 0  # La perte est positive

    # Vérifier que la validation a fonctionné
    assert "val_loss" in history
    assert len(history["val_loss"]) == 1  # Une époque


# Ce test ne s'exécute que s'il y a au moins 2 GPUs
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Au moins 2 GPUs sont nécessaires"
)
def test_ddp_model_wrapper_multi_gpu():
    """Teste le wrapper DDP en mode multi-GPU."""
    # Créer un dataset de test
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100, 1)).float()
    dataset = TensorDataset(inputs, targets)

    # Créer le wrapper DDP avec plusieurs GPUs
    ddp_wrapper = DDPModelWrapper(
        model_fn=lambda: SimpleModel(input_dim=10),
        optimizer_fn=lambda params: optim.Adam(params, lr=0.001),
        criterion_fn=lambda: nn.BCELoss(),
        world_size=2,  # Utiliser 2 GPUs
    )

    # Vérifier que le wrapper est configuré pour DDP
    assert ddp_wrapper.use_ddp == True
    assert ddp_wrapper.world_size == 2

    # Entraîner le modèle sur une époque
    # Note: train() retourne un dict vide en mode DDP car l'historique
    # est géré dans chaque processus et n'est pas facilement récupérable
    history = ddp_wrapper.train(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=16,
        epochs=1,
        num_workers=0,
    )

    # Vérifier que l'entraînement s'est terminé sans erreur
    assert isinstance(history, dict)


if __name__ == "__main__":
    # Exécuter directement les tests si le script est appelé directement
    test_gpu_detection()
    print("Test de détection GPU réussi!")

    test_gpu_memory_info()
    print("Test d'info mémoire GPU réussi!")

    test_dataloader_preparation()
    print("Test de préparation DataLoader réussi!")

    if torch.cuda.is_available():
        test_ddp_model_wrapper_single_gpu()
        print("Test de DDPModelWrapper en mode single-GPU réussi!")

        if torch.cuda.device_count() >= 2:
            test_ddp_model_wrapper_multi_gpu()
            print("Test de DDPModelWrapper en mode multi-GPU réussi!")
        else:
            print("Test multi-GPU ignoré (nombre insuffisant de GPUs)")
    else:
        print("Tests GPU ignorés (CUDA non disponible)")
