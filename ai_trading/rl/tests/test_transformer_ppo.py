import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ajuster les chemins d'importation
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import des modules
from ai_trading.rl.agents.transformer_ppo_agent import TransformerPPOAgent
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "ai_trading" / "info_retour" / "visualisations" / "rl"
DATA_DIR = BASE_DIR / "ai_trading" / "info_retour" / "data" / "processed"

# Assurer que le répertoire d'output existe
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=1000):
    """
    Génère des données de prix synthétiques pour les tests.
    """
    # Simuler un mouvement de prix avec tendance, bruit et cycles
    trend = np.linspace(0, 30, n_samples)
    noise = np.random.normal(0, 5, n_samples)
    sine = 10 * np.sin(np.linspace(0, 5, n_samples))
    
    # Pour simuler des changements de régime
    regime_changes = np.zeros(n_samples)
    change_points = np.random.choice(range(n_samples), size=5, replace=False)
    for point in change_points:
        regime_changes[point:] += np.random.uniform(-10, 10)
    
    # Combinaison des composantes
    prices = 100 + trend + noise + sine + regime_changes
    
    # Générer les données OHLCV
    data = {
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_samples),
    }
    
    # Créer un DataFrame avec timestmap
    df = pd.DataFrame(data)
    df["timestamp"] = pd.date_range(start="2023-01-01", periods=n_samples, freq="1H")
    df.set_index("timestamp", inplace=True)
    
    return df


def setup_environment(df=None, use_synthetic_data=True):
    """
    Configure l'environnement de trading pour les tests.
    """
    if df is None and use_synthetic_data:
        df = generate_synthetic_data()
    elif df is None:
        # Charger des données réelles si disponibles
        data_path = DATA_DIR / "btc_usd_1h.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
        else:
            logger.warning(f"Données réelles non trouvées à {data_path}. Utilisation de données synthétiques.")
            df = generate_synthetic_data()
    
    # Créer l'environnement
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        transaction_fee=0.001,
        window_size=30,
        action_type="continuous",  # Actions continues pour Transformer PPO
        reward_function="sharpe",  # Utiliser le ratio de Sharpe comme récompense
        include_technical_indicators=True,  # Inclure les indicateurs techniques
        # Contrôle plus précis des features pour assurer la compatibilité des dimensions
        features=["open", "high", "low", "close", "volume"],
        technical_indicators={
            "sma": [5, 10, 20, 50],
            "ema": [5, 10, 20, 50],
            "rsi": [14],
            "macd": [(12, 26, 9)],
            "bbands": [(20, 2)],
            "atr": [14],
            "adx": [14],
            "obv": [],
            "vwap": []
        }
    )
    
    return env


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calcule le ratio de Sharpe à partir d'une liste de rendements.
    """
    returns_array = np.array(returns)
    if len(returns_array) < 2:
        return 0.0
    
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array) + 1e-8  # Éviter division par zéro
    
    # Annualiser le ratio (supposant des données quotidiennes)
    sharpe = (mean_return - risk_free_rate) / std_return
    
    return sharpe


def calculate_max_drawdown(portfolio_values):
    """
    Calcule le drawdown maximum à partir d'une série de valeurs de portefeuille.
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Calculer le drawdown pour chaque point
    peak = portfolio_values[0]
    max_dd = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return max_dd


def train_transformer_ppo(env, n_episodes=100, sequence_length=50):
    """
    Entraîne l'agent TransformerPPO dans l'environnement spécifié.
    """
    # Dimensions de l'état et de l'action
    state_dim = env.observation_space.shape[0]
    logger.info(f"Dimension de l'état : {state_dim}")
    action_dim = env.action_space.shape[0]
    
    # Créer l'agent
    agent = TransformerPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=128,  # Dimension du modèle Transformer
        n_heads=4,    # Nombre de têtes d'attention
        num_layers=3, # Nombre de couches Transformer
        dim_feedforward=256,  # Dimension feedforward
        dropout=0.1,
        activation="gelu",
        sequence_length=sequence_length,  # Longueur de la séquence temporelle
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
    episode_portfolio_values = []
    episode_rewards = []
    
    logger.info(f"Début de l'entraînement sur {n_episodes} épisodes")
    
    for episode in range(n_episodes):
        # Réinitialiser l'environnement et le buffer d'états de l'agent
        state, _ = env.reset()
        agent.reset_state_buffer()
        
        # Variables pour l'épisode
        episode_return = 0.0
        done = False
        truncated = False
        episode_reward_list = []
        portfolio_values = []
        
        # Listes pour stocker les transitions
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        # Jouer l'épisode
        step = 0
        while not (done or truncated):
            # S'assurer que l'état a la bonne dimension
            if len(state) != state_dim:
                # Adapter la dimension de l'état si nécessaire
                if len(state) < state_dim:
                    padding = np.zeros(state_dim - len(state))
                    state = np.concatenate([state, padding])
                else:
                    state = state[:state_dim]
            
            # Sélectionner une action
            action, _ = agent.get_action(state)
            
            # Exécuter l'action
            next_state, reward, done, truncated, info = env.step(action)
            
            # S'assurer que next_state a la bonne dimension
            if len(next_state) != state_dim:
                if len(next_state) < state_dim:
                    padding = np.zeros(state_dim - len(next_state))
                    next_state = np.concatenate([next_state, padding])
                else:
                    next_state = next_state[:state_dim]
            
            # Stocker la transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            
            # Collecter des informations pour les métriques
            episode_reward_list.append(reward)
            if 'portfolio_value' in info:
                portfolio_values.append(info['portfolio_value'])
            
            # Mettre à jour pour la prochaine étape
            state = next_state
            episode_return += reward
            step += 1
        
        # Entraîner l'agent à la fin de l'épisode
        if len(states) > 0:
            # Convertir les listes en tableaux numpy avec les bonnes dimensions
            states_array = np.array(states, dtype=np.float32)
            actions_array = np.array(actions, dtype=np.float32)
            rewards_array = np.array(rewards, dtype=np.float32)
            next_states_array = np.array(next_states, dtype=np.float32)
            dones_array = np.array(dones, dtype=np.float32)
            
            losses = agent.update(
                states_array,
                actions_array,
                rewards_array,
                next_states_array,
                dones_array
            )
            
            # Enregistrer les statistiques
            episode_returns.append(episode_return)
            episode_rewards.append(episode_reward_list)
            if portfolio_values:
                episode_portfolio_values.append(portfolio_values)
            
            # Calculer le ratio de Sharpe et le drawdown maximum
            sharpe_ratio = calculate_sharpe_ratio(episode_reward_list) if episode_reward_list else 0
            max_drawdown = calculate_max_drawdown(portfolio_values) if portfolio_values else 0
            
            # Afficher périodiquement les résultats
            if (episode + 1) % 10 == 0 or episode == 0:
                logger.info(
                    f"Épisode {episode + 1}/{n_episodes} "
                    f"({step} étapes) - "
                    f"Rendement: {episode_return:.2f}, "
                    f"Sharpe: {sharpe_ratio:.2f}, "
                    f"Drawdown: {max_drawdown:.2%}, "
                    f"Actor Loss: {losses['actor_loss']:.4f}, "
                    f"Critic Loss: {losses['critic_loss']:.4f}"
                )
    
    # Sauvegarder le modèle entraîné
    agent.save()
    
    # Calculer les métriques finales
    all_rewards = [item for sublist in episode_rewards for item in sublist]
    final_sharpe = calculate_sharpe_ratio(all_rewards)
    
    final_max_drawdown = 0.0
    if episode_portfolio_values:
        all_portfolio_values = [item for sublist in episode_portfolio_values for item in sublist]
        final_max_drawdown = calculate_max_drawdown(all_portfolio_values)
    
    # Visualiser les résultats
    visualize_results(
        episode_returns, 
        [calculate_sharpe_ratio(rewards) for rewards in episode_rewards if rewards], 
        [calculate_max_drawdown(values) for values in episode_portfolio_values if values]
    )
    
    return agent, env, {
        "returns": episode_returns,
        "sharpe_ratio": final_sharpe,
        "max_drawdown": final_max_drawdown,
        "actor_losses": agent.actor_loss_history,
        "critic_losses": agent.critic_loss_history,
        "entropy": agent.entropy_history,
    }


def evaluate_agent(agent, env, n_episodes=10):
    """
    Évalue l'agent entraîné.
    """
    returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        agent.reset_state_buffer()
        episode_return = 0.0
        done = False
        truncated = False
        
        # Pour le calcul des métriques
        rewards = []
        portfolio_values = []
        
        while not (done or truncated):
            # S'assurer que l'état a la bonne dimension
            if len(state) != agent.state_dim:
                if len(state) < agent.state_dim:
                    padding = np.zeros(agent.state_dim - len(state))
                    state = np.concatenate([state, padding])
                else:
                    state = state[:agent.state_dim]
            
            # Action déterministe en mode évaluation
            action, _ = agent.get_action(state, deterministic=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Collecter les données pour les métriques
            rewards.append(reward)
            if 'portfolio_value' in info:
                portfolio_values.append(info['portfolio_value'])
                
            episode_return += reward
            state = next_state
        
        returns.append(episode_return)
        sharpe_ratios.append(calculate_sharpe_ratio(rewards))
        max_drawdowns.append(calculate_max_drawdown(portfolio_values) if portfolio_values else 0.0)
    
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


def visualize_results(returns, sharpe_ratios, max_drawdowns):
    """
    Visualise les résultats de l'entraînement.
    """
    plt.figure(figsize=(15, 15))
    
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
    plt.savefig(RESULTS_DIR / "transformer_ppo_results.png")
    plt.close()


def analyze_model_predictions(agent, env, n_steps=200):
    """
    Analyse la capacité de prédiction du modèle Transformer pour différents horizons.
    """
    # Réinitialiser l'environnement et l'agent
    state, _ = env.reset()
    agent.reset_state_buffer()
    
    # Collecter des données pour l'analyse
    true_returns = []
    predicted_actions = []
    portfolio_values = []
    
    for _ in range(n_steps):
        # Obtenir l'action du modèle
        action, _ = agent.get_action(state, deterministic=True)
        predicted_actions.append(action)
        
        # Exécuter l'action
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Collecter les données
        true_returns.append(reward)
        portfolio_values.append(env.get_portfolio_value())
        
        # Mettre à jour l'état
        state = next_state
        
        if done or truncated:
            break
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 8))
    
    # Valeur du portefeuille
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title("Valeur du portefeuille")
    plt.xlabel("Étape")
    plt.ylabel("Valeur")
    plt.grid(True)
    
    # Actions prédites
    plt.subplot(2, 1, 2)
    plt.plot(np.array(predicted_actions).squeeze())
    plt.title("Actions prédites (continu)")
    plt.xlabel("Étape")
    plt.ylabel("Action")
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "transformer_ppo_predictions.png")
    plt.close()
    
    return {
        "portfolio_values": portfolio_values,
        "predicted_actions": predicted_actions,
        "true_returns": true_returns,
    }


def main():
    """Fonction principale de test."""
    # Préparation de l'environnement
    env = setup_environment(use_synthetic_data=True)
    
    # Entraînement de l'agent
    agent, env, train_metrics = train_transformer_ppo(env, n_episodes=100, sequence_length=50)
    
    # Évaluation de l'agent
    eval_metrics = evaluate_agent(agent, env, n_episodes=20)
    
    # Analyse des prédictions
    prediction_metrics = analyze_model_predictions(agent, env, n_steps=200)
    
    logger.info("Test terminé avec succès!")
    
    # Retourner les résultats pour d'éventuelles analyses supplémentaires
    return agent, env, train_metrics, eval_metrics, prediction_metrics


if __name__ == "__main__":
    main() 