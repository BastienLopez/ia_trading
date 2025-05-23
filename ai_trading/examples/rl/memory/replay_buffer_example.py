"""
Exemples d'utilisation des différents types de replay buffers.

Ce fichier contient des exemples d'utilisation pour :
1. ReplayBuffer standard
2. PrioritizedReplayBuffer
3. NStepReplayBuffer
4. DiskReplayBuffer
"""

import numpy as np
import torch
import os
import tempfile
from ai_trading.rl.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    NStepReplayBuffer,
    DiskReplayBuffer
)

def example_basic_replay():
    """Exemple d'utilisation du ReplayBuffer standard."""
    print("\n=== Exemple ReplayBuffer Standard ===")
    
    # Création du buffer
    buffer = ReplayBuffer(buffer_size=1000)
    
    # Ajout de quelques transitions
    for i in range(5):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        print(f"Transition {i} ajoutée")
    
    # Échantillonnage
    batch_size = 3
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print("\nRésultats de l'échantillonnage :")
    print(f"États : {states.shape}")
    print(f"Actions : {actions.shape}")
    print(f"Récompenses : {rewards.shape}")
    print(f"États suivants : {next_states.shape}")
    print(f"Dones : {dones.shape}")

def example_prioritized_replay():
    """Exemple d'utilisation du PrioritizedReplayBuffer."""
    print("\n=== Exemple PrioritizedReplayBuffer ===")
    
    # Création du buffer
    buffer = PrioritizedReplayBuffer(buffer_size=1000, alpha=0.6)
    
    # Ajout de transitions avec différentes priorités
    for i in range(5):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = False
        
        # Priorité plus élevée pour les transitions avec des récompenses plus grandes
        # priority = abs(reward) + 1.0  # On ne l'utilise plus ici
        buffer.add(state, action, reward, next_state, done)
        print(f"Transition {i} ajoutée")
    
    # Échantillonnage
    batch_size = 3
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size)
    
    print("\nRésultats de l'échantillonnage :")
    print(f"États : {states.shape}")
    print(f"Actions : {actions.shape}")
    print(f"Récompenses : {rewards.shape}")
    print(f"États suivants : {next_states.shape}")
    print(f"Dones : {dones.shape}")
    print(f"Indices : {indices}")
    print(f"Poids : {weights}")

def example_nstep_replay():
    """Exemple d'utilisation du NStepReplayBuffer."""
    print("\n=== Exemple NStepReplayBuffer ===")
    
    # Création du buffer
    buffer = NStepReplayBuffer(buffer_size=1000, n_steps=3, gamma=0.99)
    
    # Ajout de transitions
    for i in range(10):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        print(f"Transition {i} ajoutée")
    
    # Échantillonnage
    batch_size = 3
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print("\nRésultats de l'échantillonnage :")
    print(f"États : {states.shape}")
    print(f"Actions : {actions.shape}")
    print(f"Récompenses : {rewards.shape}")
    print(f"États suivants : {next_states.shape}")
    print(f"Dones : {dones.shape}")

def example_disk_replay():
    """Exemple d'utilisation du DiskReplayBuffer."""
    print("\n=== Exemple DiskReplayBuffer ===")
    
    # Création d'un répertoire temporaire pour le stockage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "replay_buffer.h5")
        metadata_path = os.path.join(temp_dir, "metadata.json")
        
        # Création du buffer
        buffer = DiskReplayBuffer(
            buffer_size=1000,
            state_dim=4,
            action_dim=1,
            storage_path=storage_path,
            cache_size=100
        )
        
        # Ajout de transitions
        for i in range(100):
            state = np.random.randn(4)
            action = np.array([np.random.randint(0, 2)])
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = False
            
            buffer.add(state, action, reward, next_state, done)
            if i < 5 or i == 99:
                print(f"Transition {i} ajoutée")
        
        # Échantillonnage
        batch_size = 3
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        print("\nRésultats de l'échantillonnage :")
        print(f"États : {states.shape}")
        print(f"Actions : {actions.shape}")
        print(f"Récompenses : {rewards.shape}")
        print(f"États suivants : {next_states.shape}")
        print(f"Dones : {dones.shape}")
        
        # Sauvegarde et chargement
        buffer.save_metadata(metadata_path)
        new_buffer = DiskReplayBuffer.load(storage_path, metadata_path)
        
        print("\nVérification après chargement :")
        print(f"Taille du nouveau buffer : {len(new_buffer)}")
        print(f"Taille du cache : {new_buffer.cache_size}")

def main():
    """Fonction principale exécutant tous les exemples."""
    print("Démonstration des différents types de replay buffers")
    print("=" * 50)
    
    # Exécution des exemples
    example_basic_replay()
    example_prioritized_replay()
    example_nstep_replay()
    example_disk_replay()

if __name__ == "__main__":
    main() 