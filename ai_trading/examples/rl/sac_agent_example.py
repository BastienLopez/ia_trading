import logging
import os
import sys
import numpy as np
import pandas as pd
import torch

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data
from ai_trading.rl.agents.sac_agent import OptimizedSACAgent
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SACAgentExample")

def example_sac_continuous():
    """Exemple d'utilisation du SAC agent en mode continu."""
    # Générer des données de marché synthétiques
    df = generate_synthetic_market_data(n_samples=1000)
    
    # Créer l'environnement de trading
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        window_size=10,
        transaction_fee=0.001,
        action_type="continuous"
    )
    
    # Initialiser l'agent SAC
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = OptimizedSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        sequence_length=10,
        d_model=128,
        n_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        max_seq_len=200,
        hidden_dim=256,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_size=100000,
        batch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        action_bounds=(-1.0, 1.0)
    )
    
    # Entraînement
    n_episodes = 100
    max_steps = 200
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Sélectionner une action
            action = agent.select_action(state)
            
            # Exécuter l'action
            next_state, reward, done, _, info = env.step(action)
            
            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Entraîner l'agent
            if len(agent.replay_buffer) > agent.batch_size:
                metrics = agent.train()
                if step % 10 == 0:
                    logger.info(f"Episode {episode}, Step {step}, Metrics: {metrics}")
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        logger.info(f"Episode {episode} finished with reward {episode_reward}")
        
        # Sauvegarder le modèle périodiquement
        if episode % 10 == 0:
            save_path = os.path.join("models", f"sac_agent_episode_{episode}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)

def example_transformer_policy():
    """Exemple d'utilisation du SAC agent avec politique Transformer."""
    # Générer des données de marché synthétiques
    df = generate_synthetic_market_data(n_samples=1000)
    
    # Créer l'environnement de trading
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        window_size=50,  # Plus grande fenêtre pour le Transformer
        transaction_fee=0.001,
        action_type="continuous"
    )
    
    # Initialiser l'agent SAC avec une architecture Transformer plus grande
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = OptimizedSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        sequence_length=50,
        d_model=256,  # Plus grand modèle
        n_heads=8,    # Plus de têtes d'attention
        num_layers=6, # Plus de couches
        dim_feedforward=512,
        dropout=0.1,
        activation="gelu",
        max_seq_len=200,
        hidden_dim=512,
        learning_rate=1e-4,  # Learning rate plus petit pour la stabilité
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_size=200000,  # Buffer plus grand
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
        action_bounds=(-1.0, 1.0)
    )
    
    # Entraînement
    n_episodes = 100
    max_steps = 200
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Sélectionner une action
            action = agent.select_action(state)
            
            # Exécuter l'action
            next_state, reward, done, _, info = env.step(action)
            
            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Entraîner l'agent
            if len(agent.replay_buffer) > agent.batch_size:
                metrics = agent.train()
                if step % 10 == 0:
                    logger.info(f"Episode {episode}, Step {step}, Metrics: {metrics}")
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        logger.info(f"Episode {episode} finished with reward {episode_reward}")
        
        # Sauvegarder le modèle périodiquement
        if episode % 10 == 0:
            save_path = os.path.join("models", f"transformer_sac_agent_episode_{episode}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)

if __name__ == "__main__":
    # Exécuter les exemples
    logger.info("Running SAC continuous example...")
    example_sac_continuous()
    
    logger.info("Running Transformer policy example...")
    example_transformer_policy() 