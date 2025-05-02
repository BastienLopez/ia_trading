#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de DeepSpeed pour l'optimisation mémoire d'un modèle RL lourd.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ai_trading.utils.deepspeed_optimizer import (
    optimize_model_with_deepspeed,
    create_deepspeed_config,
    DeepSpeedOptimizer
)

# Définition d'un modèle de RL lourd (simulé)
class LargeRLModel(nn.Module):
    """Modèle de RL artificiel lourd pour démontrer les optimisations DeepSpeed."""
    
    def __init__(self, input_dim=128, hidden_dims=[512, 1024, 2048, 1024, 512], output_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Couches du modèle
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Têtes pour l'actor et le critic (PPO style)
        self.actor_head = nn.Linear(prev_dim, output_dim)
        self.critic_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value
    
    def compute_loss(self, outputs, targets):
        """Calcule la perte combinée (actor + critic)."""
        action_logits, value = outputs
        actions, returns = targets
        
        # Perte de l'actor (cross-entropy)
        actor_loss = F.cross_entropy(action_logits, actions)
        
        # Perte du critic (MSE)
        critic_loss = F.mse_loss(value.squeeze(-1), returns)
        
        # Perte combinée
        total_loss = actor_loss + 0.5 * critic_loss
        return total_loss


def generate_dummy_data(batch_size=32, input_dim=128, output_dim=64, num_batches=10):
    """Génère des données d'entraînement factices."""
    all_inputs = []
    all_actions = []
    all_returns = []
    
    for _ in range(num_batches):
        # Générer des features aléatoires
        inputs = torch.randn(batch_size, input_dim)
        
        # Générer des actions aléatoires (indices de classe pour la cross-entropy)
        actions = torch.randint(0, output_dim, (batch_size,))
        
        # Générer des retours aléatoires
        returns = torch.randn(batch_size)
        
        all_inputs.append(inputs)
        all_actions.append(actions)
        all_returns.append(returns)
    
    return all_inputs, all_actions, all_returns


def train_with_deepspeed(
    model,
    inputs,
    actions,
    returns,
    zero_stage=2,
    fp16=True,
    offload_optimizer=False,
    offload_parameters=False,
    epochs=3
):
    """Entraîne le modèle avec DeepSpeed."""
    # Initialiser l'optimiseur DeepSpeed
    ds_optimizer = optimize_model_with_deepspeed(
        model=model,
        zero_stage=zero_stage,
        fp16=fp16,
        offload_optimizer=offload_optimizer,
        offload_parameters=offload_parameters
    )
    
    # Entraîner le modèle
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for i in range(len(inputs)):
            # Préparer les données
            batch_inputs = inputs[i]
            batch_actions = actions[i]
            batch_returns = returns[i]
            
            # Calculer la perte et mettre à jour les poids
            loss = ds_optimizer.train_step(
                batch=batch_inputs,
                labels=(batch_actions, batch_returns)
            )
            
            epoch_loss += loss
        
        # Calculer le temps d'exécution
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(inputs):.4f} | Time: {epoch_time:.2f}s")
    
    return ds_optimizer


def train_without_deepspeed(
    model,
    inputs,
    actions,
    returns,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    epochs=3
):
    """Entraîne le modèle sans DeepSpeed pour comparaison."""
    # Déplacer le modèle sur le périphérique
    model = model.to(device)
    
    # Créer l'optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Entraîner le modèle
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for i in range(len(inputs)):
            # Préparer les données
            batch_inputs = inputs[i].to(device)
            batch_actions = actions[i].to(device)
            batch_returns = returns[i].to(device)
            
            # Forward pass
            outputs = model(batch_inputs)
            
            # Calculer la perte
            loss = model.compute_loss(outputs, (batch_actions, batch_returns))
            
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculer le temps d'exécution
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(inputs):.4f} | Time: {epoch_time:.2f}s")


def measure_memory_usage(train_func, *args, **kwargs):
    """Mesure l'utilisation mémoire d'une fonction d'entraînement."""
    # Libérer la mémoire GPU
    torch.cuda.empty_cache()
    
    # Mesurer l'utilisation mémoire avant
    torch.cuda.synchronize()
    memory_before = torch.cuda.memory_allocated()
    
    # Exécuter la fonction d'entraînement
    result = train_func(*args, **kwargs)
    
    # Mesurer l'utilisation mémoire après
    torch.cuda.synchronize()
    memory_after_peak = torch.cuda.max_memory_allocated()
    
    memory_used = memory_after_peak - memory_before
    memory_used_mb = memory_used / (1024 * 1024)
    
    print(f"Utilisation mémoire: {memory_used_mb:.2f} MB")
    
    return result, memory_used_mb


def main(args):
    # Créer le modèle
    model = LargeRLModel(
        input_dim=args.input_dim,
        hidden_dims=[args.hidden_dim] * args.num_layers,
        output_dim=args.output_dim
    )
    
    # Générer des données factices
    inputs, actions, returns = generate_dummy_data(
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_batches=args.num_batches
    )
    
    # Créer une copie du modèle pour la comparaison
    model_for_baseline = LargeRLModel(
        input_dim=args.input_dim,
        hidden_dims=[args.hidden_dim] * args.num_layers,
        output_dim=args.output_dim
    )
    model_for_baseline.load_state_dict(model.state_dict())  # Mêmes poids initiaux
    
    print("=" * 50)
    print("Entraînement avec DeepSpeed:")
    print("=" * 50)
    
    # Créer un répertoire pour les checkpoints
    os.makedirs("deepspeed_checkpoints", exist_ok=True)
    
    # Entraîner avec DeepSpeed et mesurer l'utilisation mémoire
    _, ds_memory_usage = measure_memory_usage(
        train_with_deepspeed,
        model=model,
        inputs=inputs,
        actions=actions,
        returns=returns,
        zero_stage=args.zero_stage,
        fp16=args.fp16,
        offload_optimizer=args.offload_optimizer,
        offload_parameters=args.offload_parameters,
        epochs=args.epochs
    )
    
    if args.run_baseline:
        print("\n" + "=" * 50)
        print("Entraînement sans DeepSpeed (baseline):")
        print("=" * 50)
        
        # Entraîner sans DeepSpeed et mesurer l'utilisation mémoire
        _, baseline_memory_usage = measure_memory_usage(
            train_without_deepspeed,
            model=model_for_baseline,
            inputs=inputs,
            actions=actions,
            returns=returns,
            epochs=args.epochs
        )
        
        # Afficher la comparaison
        memory_reduction = (1 - ds_memory_usage / baseline_memory_usage) * 100
        print("\n" + "=" * 50)
        print(f"Réduction mémoire: {memory_reduction:.2f}%")
        print("=" * 50)
    
    print("\nExample DeepSpeed terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemple d'utilisation de DeepSpeed pour l'optimisation mémoire")
    parser.add_argument("--input_dim", type=int, default=128, help="Dimension d'entrée")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Dimension des couches cachées")
    parser.add_argument("--num_layers", type=int, default=5, help="Nombre de couches cachées")
    parser.add_argument("--output_dim", type=int, default=64, help="Dimension de sortie")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille du batch")
    parser.add_argument("--num_batches", type=int, default=10, help="Nombre de batches")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'époques")
    parser.add_argument("--zero_stage", type=int, default=2, help="Stage ZeRO (0, 1, 2, ou 3)")
    parser.add_argument("--fp16", action="store_true", help="Utiliser la précision mixte (FP16)")
    parser.add_argument("--offload_optimizer", action="store_true", help="Décharger l'optimiseur sur CPU")
    parser.add_argument("--offload_parameters", action="store_true", help="Décharger les paramètres sur CPU")
    parser.add_argument("--run_baseline", action="store_true", help="Exécuter l'entraînement sans DeepSpeed pour comparaison")
    
    args = parser.parse_args()
    main(args) 