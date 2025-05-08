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

from utils.distributed_training import (
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


# Test de simulation multi-GPU, ne nécessite pas réellement plusieurs GPUs
def test_multi_gpu_simulation():
    """Teste les configurations du wrapper DDP pour multi-GPU sans exécuter l'entraînement."""
    # Créer le wrapper DDP avec paramètres multi-GPU
    ddp_wrapper = DDPModelWrapper(
        model_fn=lambda: SimpleModel(input_dim=10),
        optimizer_fn=lambda params: optim.Adam(params, lr=0.001),
        criterion_fn=lambda: nn.BCELoss(),
        world_size=2,  # Simuler 2 GPUs
        backend="gloo",  # Utiliser gloo au lieu de nccl
    )

    # Vérifier les configurations multi-GPU
    assert ddp_wrapper.world_size == 2
    assert ddp_wrapper.backend == "gloo"

    # Vérifier le comportement de fallback avec world_size > GPU disponibles
    available_gpus = count_available_gpus()
    if available_gpus < 2:
        # En mode simulation, il devrait utiliser un mode non distribué (use_ddp=False)
        # si aucun GPU n'est disponible
        assert ddp_wrapper.use_ddp == (available_gpus >= 1)

    # Vérifier que la préparation des dataloaders distribués fonctionne
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100, 1)).float()
    dataset = TensorDataset(inputs, targets)

    # Test de la préparation du dataloader (pas d'exécution réelle)
    dataloader = prepare_dataloader(
        dataset, batch_size=16, shuffle=True, num_workers=0, distributed=True
    )

    # Vérifier que c'est un DistributedSampler si demandé
    from torch.utils.data.distributed import DistributedSampler

    assert isinstance(dataloader.sampler, DistributedSampler)


if __name__ == "__main__":
    # Exécuter directement les tests si le script est appelé directement
    test_gpu_detection()
    print("Test de détection GPU réussi!")

    test_gpu_memory_info()
    print("Test d'info mémoire GPU réussi!")

    test_dataloader_preparation()
    print("Test de préparation DataLoader réussi!")

    test_multi_gpu_simulation()
    print("Test de simulation multi-GPU réussi!")

    if torch.cuda.is_available():
        test_ddp_model_wrapper_single_gpu()
        print("Test de DDPModelWrapper en mode single-GPU réussi!")
    else:
        print("Tests GPU ignorés (CUDA non disponible)")
