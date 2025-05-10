import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "ai_trading" / "models"
RESULTS_DIR = BASE_DIR / "ai_trading" / "info_retour" / "visualisations" / "rl"
DATA_DIR = BASE_DIR / "ai_trading" / "info_retour" / "data" / "processed"

# Assurer que les répertoires existent
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import des modules
from ai_trading.rl.agents.ppo_agent import PPOAgent
from ai_trading.rl.trading_environment import TradingEnvironment


def generate_test_data(n_samples=1000):
    """Génère des données de prix synthétiques pour le test."""
    
    # Simuler un mouvement de prix avec une tendance et du bruit
    trend = np.linspace(0, 30, n_samples)
    noise = np.random.normal(0, 5, n_samples)
    sine = 10 * np.sin(np.linspace(0, 5, n_samples))
    
    prices = 100 + trend + noise + sine
    
    # Générer d'autres colonnes nécessaires
    data = {
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_samples),
    }
    
    # Créer le DataFrame avec une colonne de timestamp
    df = pd.DataFrame(data)
    df["timestamp"] = pd.date_range(start="2023-01-01", periods=n_samples, freq="1H")
    
    return df


def test_ppo_continuous(use_synthetic_data=True, n_episodes=100):
    """Teste l'agent PPO avec des actions continues."""
    
    # Charger ou générer les données
    if use_synthetic_data:
        df = generate_test_data()
    else:
        # Charger des données réelles
        data_path = DATA_DIR / "btc_usd_1h.csv"
        if not os.path.exists(data_path):
            logger.error(f"Fichier de données introuvable: {data_path}")
            return
        
        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # S'assurer que le timestamp est l'index
    if "timestamp" in df.columns:
        df.set_index("timestamp", inplace=True)
    
    # Créer l'environnement avec des actions continues
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        transaction_fee=0.001,
        window_size=20,
        action_type="continuous",  # Utiliser des actions continues
        reward_function="sharpe",  # Utiliser le ratio de Sharpe comme récompense
    )
    
    # Dimensions de l'état et de l'action
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Créer l'agent PPO
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        critic_loss_coef=0.5,
        entropy_coef=0.01,
        update_epochs=10,
        mini_batch_size=64,
    )
    
    # Variables pour suivre les performances
    episode_returns = []
    episode_sharpe_ratios = []
    episode_max_drawdowns = []
    
    # Entraînement
    logger.info(f"Début de l'entraînement sur {n_episodes} épisodes")
    
    for episode in range(n_episodes):
        # Réinitialiser l'environnement
        state, _ = env.reset()
        episode_return = 0.0
        done = False
        truncated = False
        
        # Listes pour stocker les expériences de l'épisode
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        # Jouer un épisode
        while not (done or truncated):
            # Sélectionner une action
            action, _ = agent.get_action(state)
            
            # Exécuter l'action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Stocker l'expérience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            
            # Mettre à jour pour la prochaine étape
            state = next_state
            episode_return += reward
        
        # Mettre à jour l'agent à la fin de l'épisode
        if len(states) > 0:
            losses = agent.update(
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
            )
            
            # Enregistrer les métriques
            episode_returns.append(episode_return)
            episode_sharpe_ratios.append(env.calculate_sharpe_ratio())
            episode_max_drawdowns.append(env.calculate_max_drawdown())
            
            # Afficher les résultats périodiquement
            if (episode + 1) % 10 == 0 or episode == 0:
                logger.info(
                    f"Épisode {episode + 1}/{n_episodes} - "
                    f"Rendement: {episode_return:.2f}, "
                    f"Sharpe: {episode_sharpe_ratios[-1]:.2f}, "
                    f"Drawdown: {episode_max_drawdowns[-1]:.2%}, "
                    f"Perte Acteur: {losses['actor_loss']:.4f}, "
                    f"Perte Critique: {losses['critic_loss']:.4f}"
                )
    
    # Sauvegarder le modèle entrainé
    model_path = MODELS_DIR / "ppo_continuous_agent.pth"
    agent.save(model_path)
    logger.info(f"Modèle sauvegardé à {model_path}")
    
    # Visualiser les résultats
    plot_results(episode_returns, episode_sharpe_ratios, episode_max_drawdowns)
    
    return agent, env, {
        "returns": episode_returns,
        "sharpe_ratios": episode_sharpe_ratios,
        "max_drawdowns": episode_max_drawdowns,
    }


def plot_results(returns, sharpe_ratios, max_drawdowns):
    """Visualise les résultats de l'entraînement."""
    
    plt.figure(figsize=(15, 10))
    
    # Rendements
    plt.subplot(3, 1, 1)
    plt.plot(returns)
    plt.title("Rendements par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Rendement")
    plt.grid(True)
    
    # Ratio de Sharpe
    plt.subplot(3, 1, 2)
    plt.plot(sharpe_ratios)
    plt.title("Ratio de Sharpe par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Sharpe")
    plt.grid(True)
    
    # Maximum Drawdown
    plt.subplot(3, 1, 3)
    plt.plot(max_drawdowns)
    plt.title("Maximum Drawdown par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Max Drawdown")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ppo_continuous_results.png")
    plt.close()


def evaluate_agent(agent, env, n_episodes=10):
    """Évalue un agent entraîné."""
    
    returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Action déterministe en mode évaluation
            action, _ = agent.get_action(state, deterministic=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_return += reward
            state = next_state
        
        returns.append(episode_return)
        sharpe_ratios.append(env.calculate_sharpe_ratio())
        max_drawdowns.append(env.calculate_max_drawdown())
    
    # Afficher les statistiques d'évaluation
    logger.info("\nRésultats de l'évaluation:")
    logger.info(f"Rendement moyen: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    logger.info(f"Sharpe moyen: {np.mean(sharpe_ratios):.2f} ± {np.std(sharpe_ratios):.2f}")
    logger.info(f"Drawdown moyen: {np.mean(max_drawdowns):.2%} ± {np.std(max_drawdowns):.2%}")
    
    return {
        "returns": returns,
        "sharpe_ratios": sharpe_ratios,
        "max_drawdowns": max_drawdowns,
    }


if __name__ == "__main__":
    # Test avec données synthétiques
    agent, env, train_metrics = test_ppo_continuous(use_synthetic_data=True, n_episodes=100)
    
    # Évaluer l'agent entraîné
    eval_metrics = evaluate_agent(agent, env, n_episodes=20)
    
    # Pour charger et évaluer un modèle existant
    # loaded_agent = PPOAgent(state_dim, action_dim)
    # loaded_agent.load(MODELS_DIR / "ppo_continuous_agent.pth")
    # evaluate_agent(loaded_agent, env) 