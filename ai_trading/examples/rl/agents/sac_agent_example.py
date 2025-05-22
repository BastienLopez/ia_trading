import argparse
import datetime
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.examples.rl.agents.multi_asset_trading_example import generate_synthetic_data
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.agents.transformer_sac_agent import TransformerSACAgent
from ai_trading.rl.environments.trading_environment import TradingEnvironment
from ai_trading.rl.environments.multi_asset_trading_environment import MultiAssetTradingEnvironment

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def visualize_training_results(history: Dict[str, List[float]], save_path: Optional[str] = None) -> str:
    """
    Visualise les résultats d'entraînement de l'agent SAC.

    Args:
        history: Historique des métriques d'entraînement
        save_path: Chemin pour sauvegarder les visualisations

    Returns:
        str: Chemin où les visualisations ont été sauvegardées
    """
    # Créer un répertoire pour les visualisations
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(INFO_RETOUR_DIR, "models", "sac", f"sac_{timestamp}")

    os.makedirs(save_path, exist_ok=True)

    # Extraire les métriques
    episodes = np.arange(1, len(history["portfolio_values"]) + 1)
    episode_rewards = history["episode_rewards"]
    portfolio_returns = history["portfolio_returns"]

    # Plot des récompenses par épisode
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, marker="o", linestyle="-", color="blue")
    plt.title("Récompenses cumulées par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulée")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "episode_rewards.png"))
    plt.close()

    # Plot des rendements du portefeuille
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, portfolio_returns, marker="o", linestyle="-", color="green")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    plt.title("Rendement du portefeuille par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Rendement (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "portfolio_returns.png"))
    plt.close()

    # Plot des pertes d'entraînement si disponibles
    if "actor_losses" in history and len(history["actor_losses"]) > 0:
        plt.figure(figsize=(15, 10))

        # Plot de la perte de l'acteur
        plt.subplot(3, 1, 1)
        plt.plot(history["actor_losses"], color="blue")
        plt.title("Perte de l'acteur")
        plt.xlabel("Étape d'entraînement")
        plt.ylabel("Perte")
        plt.grid(True, alpha=0.3)

        # Plot de la perte du critique
        plt.subplot(3, 1, 2)
        plt.plot(history["critic_losses"], color="red")
        plt.title("Perte du critique")
        plt.xlabel("Étape d'entraînement")
        plt.ylabel("Perte")
        plt.grid(True, alpha=0.3)

        # Plot de la valeur d'alpha
        plt.subplot(3, 1, 3)
        plt.plot(history["alphas"], color="green")
        plt.title("Valeur d'alpha")
        plt.xlabel("Étape d'entraînement")
        plt.ylabel("Alpha")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "training_losses.png"))
        plt.close()

    return save_path

def create_environment(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    args: argparse.Namespace,
    is_multi_asset: bool = False
) -> Union[TradingEnvironment, MultiAssetTradingEnvironment]:
    """
    Crée l'environnement de trading approprié.

    Args:
        data: Données de trading (DataFrame unique ou dictionnaire de DataFrames)
        args: Arguments de configuration
        is_multi_asset: Si True, crée un environnement multi-actifs

    Returns:
        Environnement de trading configuré
    """
    if is_multi_asset:
        env = MultiAssetTradingEnvironment(
            data_dict=data,
            initial_balance=args.initial_balance,
            window_size=args.window_size,
            include_technical_indicators=True,
            allocation_method="smart",
            rebalance_frequency=5,
            max_active_positions=len(data) // 2,
            action_type="continuous",
        )
    else:
        env = TradingEnvironment(
            df=data,
            initial_balance=args.initial_balance,
            transaction_fee=0.001,
            window_size=args.window_size,
            include_position=True,
            include_balance=True,
            include_technical_indicators=True,
            risk_management=True,
            normalize_observation=True,
            reward_function=args.reward_function,
            action_type="continuous",
        )

    logger.info(f"Environnement créé avec espace d'action continu")
    logger.info(f"Taille de l'espace d'observation: {env.observation_space.shape[0]}")

    return env

def create_agent(
    env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
    args: argparse.Namespace,
    load_path: Optional[str] = None
) -> Union[SACAgent, TransformerSACAgent]:
    """
    Crée l'agent SAC approprié.

    Args:
        env: Environnement de trading
        args: Arguments de configuration
        load_path: Chemin vers un modèle pré-entraîné

    Returns:
        Agent SAC configuré
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = (env.action_space.low[0], env.action_space.high[0])

    if args.use_transformer:
        agent = TransformerSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            actor_learning_rate=args.learning_rate,
            critic_learning_rate=args.learning_rate,
            alpha_learning_rate=args.learning_rate * 0.5,
            batch_size=args.batch_size,
            buffer_size=100000,
            sequence_length=args.sequence_length,
            model_type=args.model_type,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim=args.embed_dim * 4,
            num_transformer_blocks=args.transformer_blocks,
            rnn_units=args.embed_dim,
            checkpoints_dir=args.checkpoints_dir,
        )
    else:
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            batch_size=args.batch_size,
            buffer_size=100000,
            actor_learning_rate=args.learning_rate,
            critic_learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            train_alpha=True,
        )

    if load_path:
        agent.load(load_path)
        logger.info(f"Agent chargé depuis {load_path}")

    return agent

def train_agent(
    env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
    agent: Union[SACAgent, TransformerSACAgent],
    args: argparse.Namespace
) -> Dict[str, List[float]]:
    """
    Entraîne l'agent SAC.

    Args:
        env: Environnement de trading
        agent: Agent SAC
        args: Arguments de configuration

    Returns:
        Historique d'entraînement
    """
    history = {
        "episode_rewards": [],
        "portfolio_returns": [],
        "portfolio_values": [],
        "actor_losses": [],
        "critic_losses": [],
        "alpha_losses": [],
        "alphas": [],
        "q_values": [],
    }

    for episode in range(args.episodes):
        state, _ = env.reset()
        if isinstance(agent, TransformerSACAgent):
            agent.reset_state_buffer()
            agent.update_state_buffer(state)

        episode_reward = 0
        done = False
        step = 0

        while not done and step < args.max_steps_per_episode:
            # Sélectionner une action
            if isinstance(agent, TransformerSACAgent):
                action = agent.sample_action(state)
            else:
                action = agent.act(state)

            # Exécuter l'action
            next_state, reward, done, _, info = env.step(action)

            # Stocker la transition
            if isinstance(agent, TransformerSACAgent):
                agent.remember(state, action, reward, next_state, done)
            else:
                agent.store_experience(state, action, reward, next_state, done)

            # Entraîner l'agent
            train_info = agent.train()

            # Mettre à jour l'état
            state = next_state
            if isinstance(agent, TransformerSACAgent):
                agent.update_state_buffer(state)

            # Accumuler la récompense
            episode_reward += reward

            # Enregistrer les métriques
            if step % 10 == 0:
                history["actor_losses"].append(train_info.get("actor_loss", 0))
                history["critic_losses"].append(train_info.get("critic_loss", 0))
                history["alpha_losses"].append(train_info.get("alpha_loss", 0))
                history["alphas"].append(train_info.get("alpha", 0))
                history["q_values"].append(train_info.get("q_value", 0))

            # Log périodique
            if step % 20 == 0:
                portfolio_value = info.get("portfolio_value", 0)
                logger.info(
                    f"Épisode {episode+1}/{args.episodes}, Étape {step}, "
                    f"Portefeuille: ${portfolio_value:.2f}, "
                    f"Récompense: {reward:.4f}"
                )

            step += 1

        # Calculer le rendement du portefeuille
        final_portfolio_value = env.get_portfolio_value()
        portfolio_return = (final_portfolio_value / env.initial_balance - 1) * 100

        # Enregistrer les résultats
        history["episode_rewards"].append(episode_reward)
        history["portfolio_returns"].append(portfolio_return)
        history["portfolio_values"].append(env.portfolio_value_history)

        logger.info(f"Épisode {episode+1}/{args.episodes} terminé:")
        logger.info(f"  Récompense totale: {episode_reward:.4f}")
        logger.info(f"  Valeur finale du portefeuille: ${final_portfolio_value:.2f}")
        logger.info(f"  Rendement: {portfolio_return:.2f}%")

        # Sauvegarder périodiquement
        if (episode + 1) % args.save_every == 0:
            save_path = os.path.join(args.checkpoints_dir, f"episode_{episode+1}")
            agent.save(save_path)

    return history

def evaluate_agent(
    env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
    agent: Union[SACAgent, TransformerSACAgent],
    num_episodes: int = 5
) -> Tuple[List[float], List[float]]:
    """
    Évalue l'agent après l'entraînement.

    Args:
        env: Environnement de trading
        agent: Agent SAC
        num_episodes: Nombre d'épisodes d'évaluation

    Returns:
        Tuple contenant les récompenses et valeurs du portefeuille
    """
    returns = []
    portfolio_values = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        if isinstance(agent, TransformerSACAgent):
            agent.reset_state_buffer()
            agent.update_state_buffer(state)

        episode_reward = 0
        done = False

        while not done:
            # En évaluation, utiliser la politique déterministe
            if isinstance(agent, TransformerSACAgent):
                action = agent.sample_action(state, evaluate=True)
            else:
                action = agent.act(state, deterministic=True)

            next_state, reward, done, _, info = env.step(action)
            state = next_state
            if isinstance(agent, TransformerSACAgent):
                agent.update_state_buffer(state)

            episode_reward += reward

        returns.append(episode_reward)
        portfolio_values.append(env.get_portfolio_value())

        logger.info(
            f"Évaluation, épisode {episode+1}: "
            f"Récompense={episode_reward:.2f}, "
            f"Portfolio={env.get_portfolio_value():.2f}"
        )

    logger.info(
        f"Résultat moyen sur {num_episodes} épisodes: "
        f"Récompense={np.mean(returns):.2f}, "
        f"Portfolio={np.mean(portfolio_values):.2f}"
    )

    return returns, portfolio_values

def main():
    """Fonction principale pour exécuter l'exemple."""
    parser = argparse.ArgumentParser(description="Exemple d'entraînement avec SAC")
    
    # Arguments communs
    parser.add_argument("--use_transformer", action="store_true", help="Utiliser l'architecture Transformer")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "LTC", "ADA", "SOL"], help="Symboles des crypto-monnaies")
    parser.add_argument("--episodes", type=int, default=10, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Nombre maximum d'étapes par épisode")
    parser.add_argument("--batch_size", type=int, default=64, help="Taille du batch")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Taux d'apprentissage")
    parser.add_argument("--window_size", type=int, default=20, help="Taille de la fenêtre d'observation")
    parser.add_argument("--initial_balance", type=float, default=10000.0, help="Balance initiale")
    parser.add_argument("--reward_function", type=str, default="sharpe", help="Fonction de récompense")
    parser.add_argument("--save_every", type=int, default=10, help="Sauvegarder tous les N épisodes")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/sac", help="Répertoire des checkpoints")
    parser.add_argument("--visualize", action="store_true", help="Visualiser les résultats")
    parser.add_argument("--eval_only", action="store_true", help="Seulement évaluer un agent existant")
    parser.add_argument("--model_path", type=str, default=None, help="Chemin vers un modèle pré-entraîné")
    
    # Arguments spécifiques à Transformer
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "lstm"], help="Type de modèle RNN")
    parser.add_argument("--embed_dim", type=int, default=64, help="Dimension d'embedding")
    parser.add_argument("--num_heads", type=int, default=4, help="Nombre de têtes d'attention")
    parser.add_argument("--sequence_length", type=int, default=20, help="Longueur de la séquence")
    parser.add_argument("--transformer_blocks", type=int, default=2, help="Nombre de blocs Transformer")
    
    # Arguments spécifiques à SAC standard
    parser.add_argument("--hidden_size", type=int, default=256, help="Taille des couches cachées")

    args = parser.parse_args()

    # Générer des données synthétiques
    logger.info(f"Génération de données synthétiques pour {args.symbols}...")
    data = generate_synthetic_data(args.symbols, days=200)

    # Créer l'environnement
    env = create_environment(data, args, is_multi_asset=args.use_transformer)

    if args.eval_only and args.model_path:
        # Créer l'agent pour l'évaluation
        agent = create_agent(env, args, load_path=args.model_path)
        
        # Évaluer l'agent
        evaluate_agent(agent, env)
    else:
        # Créer l'agent pour l'entraînement
        agent = create_agent(env, args)
        
        # Entraîner l'agent
        history = train_agent(env, agent, args)
        
        # Visualiser les résultats si demandé
        if args.visualize:
            visualize_training_results(history)
        
        # Évaluer l'agent final
        evaluate_agent(env, agent)

    logger.info("Exemple terminé.")

if __name__ == "__main__":
    main() 