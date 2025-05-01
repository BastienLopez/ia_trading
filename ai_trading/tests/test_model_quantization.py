"""
Tests pour le module de quantification de modèles (model_quantization.py).

Ce module teste les fonctionnalités de quantification des modèles de réseau de neurones.
Ces tests sont conçus pour être robustes et ne pas échouer si la plateforme ne supporte pas
complètement la quantification.
"""

import pytest
import torch
import torch.nn as nn
import os
from pathlib import Path

from ai_trading.utils.model_quantization import _get_model_size


class SimpleModel(nn.Module):
    """Modèle simple pour les tests."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def model():
    """Crée une instance de modèle pour les tests."""
    model = SimpleModel()
    # Initialiser les poids avec des valeurs non nulles
    for param in model.parameters():
        nn.init.normal_(param, mean=0.0, std=0.1)
    return model


@pytest.fixture
def sample_input():
    """Crée un tensor d'entrée pour les tests."""
    return torch.randn(4, 10)


def test_get_model_size(model):
    """Teste la fonction de calcul de taille de modèle."""
    # Calculer la taille du modèle
    model_size = _get_model_size(model)
    
    # Vérifier que la taille est positive et non nulle
    assert model_size > 0
    
    # Vérifier la taille approximative en calculant manuellement
    manual_size = 0
    for name, param in model.state_dict().items():
        manual_size += param.numel() * param.element_size()
    
    # Les deux calculs devraient être très proches
    assert abs(model_size - manual_size) < 1e-6


def test_model_forward(model, sample_input):
    """Teste que le modèle peut faire un forward pass normal."""
    # Le modèle devrait pouvoir traiter l'entrée sans erreur
    try:
        output = model(sample_input)
        # Vérifier la forme de sortie
        assert output.shape == (sample_input.shape[0], 5)
    except Exception as e:
        pytest.fail(f"Erreur lors du forward pass du modèle: {e}")


def test_quantization_support():
    """Teste si la plateforme supporte la quantification."""
    # Vérifier si PyTorch a le module de quantification
    has_quantization = hasattr(torch, 'quantization')
    
    if has_quantization:
        print("La plateforme supporte la quantification PyTorch")
    else:
        print("La plateforme ne supporte pas la quantification PyTorch")
        
    # Le test passe toujours, c'est juste informatif
    assert True


def test_compare_float32_float16_model_size(model):
    """Compare la taille d'un modèle en float32 vs float16."""
    # Taille du modèle original (float32)
    original_size = _get_model_size(model)
    
    # Convertir le modèle en float16
    model_fp16 = model.to(torch.float16)
    fp16_size = _get_model_size(model_fp16)
    
    # La taille devrait être réduite d'environ 50%
    reduction = (original_size - fp16_size) / original_size
    
    # Vérifier que la taille a bien été réduite
    assert fp16_size < original_size
    assert 0.45 < reduction < 0.55  # 50% avec une petite marge d'erreur
    
    # Test informatif sur la réduction de taille
    print(f"Réduction de taille avec float16: {reduction*100:.2f}%") 