import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from ai_trading.examples.multi_asset_trading_example import generate_synthetic_data
from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment
from ai_trading.rl.transformer_sac_agent import TransformerSACAgent

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_training_results(history, save_path=None):
    """
    Visualise les résultats d'entraînement de l'agent TransformerSAC.

    Args:
        history: Historique des métriques d'entraînement
        save_path: Chemin pour sauvegarder les visualisations
    """
    # Créer un répertoire pour les visualisations
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "visualizations",
            f"transformer_sac_{timestamp}",
        )

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

    # Plot des valeurs de portefeuille pour chaque épisode (si pas trop nombreux)
    if len(history["portfolio_values"]) <= 10:
        plt.figure(figsize=(15, 8))
        for i, values in enumerate(history["portfolio_values"]):
            plt.plot(values, label=f"Épisode {i+1}")

        plt.axhline(
            y=10000, color="r", linestyle="--", alpha=0.5, label="Capital initial"
        )
        plt.title("Évolution de la valeur du portefeuille")
        plt.xlabel("Étape")
        plt.ylabel("Valeur du portefeuille ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, "portfolio_evolution.png"))
        plt.close()
    else:
        # Sélectionner quelques épisodes à visualiser
        selected_episodes = [
            0,
            len(history["portfolio_values"]) // 2,
            len(history["portfolio_values"]) - 1,
        ]
        plt.figure(figsize=(15, 8))
        for i in selected_episodes:
            plt.plot(history["portfolio_values"][i], label=f"Épisode {i+1}")

        plt.axhline(
            y=10000, color="r", linestyle="--", alpha=0.5, label="Capital initial"
        )
        plt.title("Évolution de la valeur du portefeuille (épisodes sélectionnés)")
        plt.xlabel("Étape")
        plt.ylabel("Valeur du portefeuille ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, "portfolio_evolution_selected.png"))
        plt.close()

    logger.info(f"Visualisations sauvegardées dans {save_path}")
    return save_path


def train_transformer_sac_agent(
    data_dict,
    episodes=50,
    max_steps_per_episode=1000,
    batch_size=64,
    learning_rate=3e-4,
    model_type="gru",
    embed_dim=64,
    num_heads=4,
    sequence_length=20,
    num_transformer_blocks=2,
    save_every=10,
    eval_every=5,
    visualize=True,
    checkpoints_dir="./checkpoints/transformer_sac",
):
    """
    Entraîne un agent TransformerSAC sur l'environnement de trading multi-actifs.

    Args:
        data_dict: Dictionnaire de DataFrames avec les données à utiliser
        episodes: Nombre d'épisodes d'entraînement
        max_steps_per_episode: Nombre maximum d'étapes par épisode
        batch_size: Taille du batch pour l'entraînement
        learning_rate: Taux d'apprentissage
        model_type: Type de modèle hybride ('gru' ou 'lstm')
        embed_dim: Dimension d'embedding pour le Transformer
        num_heads: Nombre de têtes d'attention
        sequence_length: Longueur de la séquence
        num_transformer_blocks: Nombre de blocs Transformer
        save_every: Nombre d'épisodes entre chaque sauvegarde
        eval_every: Nombre d'épisodes entre chaque évaluation
        visualize: Si True, visualise les résultats
        checkpoints_dir: Répertoire pour les points de contrôle

    Returns:
        dict: Historique d'entraînement
    """
    # Créer l'environnement
    env = MultiAssetTradingEnvironment(
        data_dict=data_dict,
        initial_balance=10000.0,
        window_size=sequence_length,
        include_technical_indicators=True,
        allocation_method="smart",
        rebalance_frequency=5,
        max_active_positions=len(data_dict) // 2,
        action_type="continuous",
    )

    # Obtenir les dimensions de l'état et de l'action
    observation, _ = env.reset()
    state_dim = observation.shape
    action_dim = env.action_space.shape[0]
    action_bounds = (env.action_space.low[0], env.action_space.high[0])

    logger.info(f"Dimension de l'état: {state_dim}")
    logger.info(f"Dimension de l'action: {action_dim}")
    logger.info(f"Limites des actions: {action_bounds}")

    # Créer l'agent
    agent = TransformerSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=action_bounds,
        actor_learning_rate=learning_rate,
        critic_learning_rate=learning_rate,
        alpha_learning_rate=learning_rate * 0.5,
        batch_size=batch_size,
        buffer_size=100000,
        sequence_length=sequence_length,
        model_type=model_type,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=embed_dim * 4,
        num_transformer_blocks=num_transformer_blocks,
        rnn_units=embed_dim,
        checkpoints_dir=checkpoints_dir,
    )

    # Historique d'entraînement
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

    # Entraîner l'agent
    for episode in range(episodes):
        observation, _ = env.reset()
        agent.reset_state_buffer()

        # Ajouter l'état initial au tampon de l'agent
        agent.update_state_buffer(observation)

        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            # Sélectionner une action
            action = agent.sample_action(observation)

            # Exécuter l'action dans l'environnement
            next_observation, reward, done, _, info = env.step(action)

            # Stocker la transition
            agent.remember(observation, action, reward, next_observation, done)

            # Entraîner l'agent
            train_info = agent.train()

            # Mettre à jour l'observation
            observation = next_observation

            # Accumuler la récompense
            episode_reward += reward

            # Enregistrer les métriques d'entraînement
            if (
                step % 10 == 0
            ):  # Enregistrer périodiquement pour économiser de la mémoire
                history["actor_losses"].append(train_info["actor_loss"])
                history["critic_losses"].append(train_info["critic_loss"])
                history["alpha_losses"].append(train_info["alpha_loss"])
                if "alpha" in train_info:
                    history["alphas"].append(train_info["alpha"])
                history["q_values"].append(train_info.get("q_value", 0))

            # Log périodique
            if step % 20 == 0:
                portfolio_value = info["portfolio_value"]
                portfolio_weights = info["portfolio_weights"]
                logger.info(
                    f"Épisode {episode+1}/{episodes}, Étape {step}, "
                    f"Portefeuille: ${portfolio_value:.2f}, "
                    f"Récompense: {reward:.4f}"
                )
                logger.info(f"Allocation: {portfolio_weights}")

            step += 1

        # Calculer le rendement du portefeuille pour cet épisode
        final_portfolio_value = env.get_portfolio_value()
        portfolio_return = (final_portfolio_value / env.initial_balance - 1) * 100

        # Enregistrer les résultats de l'épisode
        history["episode_rewards"].append(episode_reward)
        history["portfolio_returns"].append(portfolio_return)
        history["portfolio_values"].append(env.portfolio_value_history)

        logger.info(f"Épisode {episode+1}/{episodes} terminé:")
        logger.info(f"  Récompense totale: {episode_reward:.4f}")
        logger.info(f"  Valeur finale du portefeuille: ${final_portfolio_value:.2f}")
        logger.info(f"  Rendement: {portfolio_return:.2f}%")

        # Sauvegarder périodiquement l'agent
        if (episode + 1) % save_every == 0:
            agent.save_models(suffix=f"episode_{episode+1}")

        # Évaluer périodiquement l'agent
        if (episode + 1) % eval_every == 0:
            evaluate_agent(
                agent, data_dict, model_type=model_type, sequence_length=sequence_length
            )

    # Sauvegarder l'agent final
    agent.save_models(suffix="final")

    # Visualiser les résultats si demandé
    if visualize:
        visualize_path = visualize_training_results(history)

    return history


def evaluate_agent(
    agent, data_dict, episodes=3, model_type="gru", sequence_length=20, visualize=True
):
    """
    Évalue un agent TransformerSAC sur l'environnement de trading multi-actifs.

    Args:
        agent: Agent à évaluer
        data_dict: Dictionnaire de DataFrames avec les données à utiliser
        episodes: Nombre d'épisodes d'évaluation
        model_type: Type de modèle hybride ('gru' ou 'lstm')
        sequence_length: Longueur de la séquence
        visualize: Si True, visualise les résultats d'évaluation

    Returns:
        dict: Résultats d'évaluation
    """
    # Créer l'environnement
    env = MultiAssetTradingEnvironment(
        data_dict=data_dict,
        initial_balance=10000.0,
        window_size=sequence_length,
        include_technical_indicators=True,
        allocation_method="smart",
        rebalance_frequency=5,
        max_active_positions=len(data_dict) // 2,
        action_type="continuous",
    )

    # Résultats d'évaluation
    results = {"episode_rewards": [], "portfolio_returns": [], "portfolio_values": []}

    # Évaluer l'agent
    for episode in range(episodes):
        observation, _ = env.reset()
        agent.reset_state_buffer()

        # Ajouter l'état initial au tampon de l'agent
        agent.update_state_buffer(observation)

        episode_reward = 0
        done = False

        while not done:
            # Sélectionner une action (en mode évaluation)
            action = agent.sample_action(observation, evaluate=True)

            # Exécuter l'action dans l'environnement
            next_observation, reward, done, _, info = env.step(action)

            # Mettre à jour l'observation
            observation = next_observation

            # Accumuler la récompense
            episode_reward += reward

            # Log périodique
            if env.current_step % 20 == 0:
                portfolio_value = info["portfolio_value"]
                portfolio_weights = info["portfolio_weights"]
                logger.info(
                    f"Évaluation - Épisode {episode+1}/{episodes}, Étape {env.current_step}, "
                    f"Portefeuille: ${portfolio_value:.2f}"
                )

        # Calculer le rendement du portefeuille pour cet épisode
        final_portfolio_value = env.get_portfolio_value()
        portfolio_return = (final_portfolio_value / env.initial_balance - 1) * 100

        # Enregistrer les résultats de l'épisode
        results["episode_rewards"].append(episode_reward)
        results["portfolio_returns"].append(portfolio_return)
        results["portfolio_values"].append(env.portfolio_value_history)

        logger.info(f"Évaluation - Épisode {episode+1}/{episodes} terminé:")
        logger.info(f"  Récompense totale: {episode_reward:.4f}")
        logger.info(f"  Valeur finale du portefeuille: ${final_portfolio_value:.2f}")
        logger.info(f"  Rendement: {portfolio_return:.2f}%")

        # Visualiser l'allocation du portefeuille
        if visualize:
            env.visualize_portfolio_allocation()

    # Calculer les moyennes
    avg_reward = np.mean(results["episode_rewards"])
    avg_return = np.mean(results["portfolio_returns"])

    logger.info(f"Évaluation terminée:")
    logger.info(f"  Récompense moyenne: {avg_reward:.4f}")
    logger.info(f"  Rendement moyen: {avg_return:.2f}%")

    return results


def main():
    """Fonction principale pour exécuter l'exemple."""
    parser = argparse.ArgumentParser(
        description="Exemple d'entraînement avec TransformerSAC"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH", "LTC", "ADA", "SOL"],
        help="Symboles des crypto-monnaies à trader",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Taille du batch pour l'entraînement"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Taux d'apprentissage")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gru",
        choices=["gru", "lstm"],
        help="Type de modèle RNN dans l'architecture hybride",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="Dimension d'embedding pour le Transformer",
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Nombre de têtes d'attention"
    )
    parser.add_argument(
        "--seq_length", type=int, default=20, help="Longueur de la séquence"
    )
    parser.add_argument(
        "--transformer_blocks", type=int, default=2, help="Nombre de blocs Transformer"
    )
    parser.add_argument(
        "--days", type=int, default=200, help="Nombre de jours de données synthétiques"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualiser les résultats"
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Seulement évaluer un agent existant"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Chemin vers un modèle pré-entraîné",
    )

    args = parser.parse_args()

    # Générer des données synthétiques
    logger.info(f"Génération de données synthétiques pour {args.symbols}...")
    data_dict = generate_synthetic_data(args.symbols, days=args.days)

    if args.eval_only and args.model_path:
        # Créer un agent temporaire pour obtenir les dimensions
        temp_env = MultiAssetTradingEnvironment(
            data_dict=data_dict, window_size=args.seq_length, action_type="continuous"
        )
        temp_observation, _ = temp_env.reset()
        state_dim = temp_observation.shape
        action_dim = temp_env.action_space.shape[0]
        action_bounds = (temp_env.action_space.low[0], temp_env.action_space.high[0])

        # Créer l'agent
        agent = TransformerSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            sequence_length=args.seq_length,
            model_type=args.model_type,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_transformer_blocks=args.transformer_blocks,
        )

        # Charger le modèle
        if agent.load_models(args.model_path):
            # Évaluer l'agent
            evaluate_agent(
                agent=agent,
                data_dict=data_dict,
                model_type=args.model_type,
                sequence_length=args.seq_length,
                visualize=args.visualize,
            )
        else:
            logger.error(f"Impossible de charger le modèle depuis {args.model_path}")
    else:
        # Entraîner l'agent
        train_transformer_sac_agent(
            data_dict=data_dict,
            episodes=args.episodes,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_type=args.model_type,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            sequence_length=args.seq_length,
            num_transformer_blocks=args.transformer_blocks,
            visualize=args.visualize,
        )

    logger.info("Exemple terminé.")


if __name__ == "__main__":
    main()
    main()
