import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

# Ajouter le répertoire parent au chemin pour importer les modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.trading_environment import TradingEnvironment

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sac_training_example")


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Exemple d'entraînement avec SAC")

    parser.add_argument(
        "--episodes", type=int, default=100, help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC", help="Symbole de la cryptomonnaie"
    )
    parser.add_argument(
        "--days", type=int, default=60, help="Nombre de jours de données à utiliser"
    )
    parser.add_argument(
        "--interval", type=str, default="1h", help="Intervalle de temps (1h, 4h, 1d)"
    )
    parser.add_argument(
        "--initial_balance", type=float, default=10000.0, help="Solde initial"
    )
    parser.add_argument(
        "--save", action="store_true", help="Sauvegarder le modèle entraîné"
    )
    parser.add_argument("--load", type=str, help="Charger un modèle existant")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualiser les résultats"
    )
    parser.add_argument(
        "--window_size", type=int, default=20, help="Taille de la fenêtre d'observation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Taille du batch pour l'entraînement"
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default="sharpe",
        choices=["simple", "sharpe", "transaction_penalty", "drawdown"],
        help="Fonction de récompense à utiliser",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=5,
        help="Nombre d'épisodes d'évaluation après l'entraînement",
    )

    return parser.parse_args()


def create_environment(df, args):
    """Crée l'environnement de trading avec actions continues."""
    env = TradingEnvironment(
        df=df,
        initial_balance=args.initial_balance,
        transaction_fee=0.001,
        window_size=args.window_size,
        include_position=True,
        include_balance=True,
        include_technical_indicators=True,
        risk_management=True,
        normalize_observation=True,
        reward_function=args.reward_function,
        action_type="continuous",  # Important pour utiliser SAC
    )

    logger.info(
        f"Environnement créé avec {len(df)} points de données et espace d'action continu"
    )
    logger.info(f"Taille de l'espace d'observation: {env.observation_space.shape[0]}")

    return env


def create_agent(env, args, load_path=None):
    """Crée l'agent SAC."""
    state_size = env.observation_space.shape[0]
    action_size = 1  # Pour l'espace d'action Box(-1, 1)

    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        action_bounds=(-1, 1),
        batch_size=args.batch_size,
        buffer_size=100000,  # Taille du tampon de replay
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        hidden_size=256,
        train_alpha=True,  # Adaptation automatique du paramètre d'entropie
    )

    if load_path:
        agent.load(load_path)
        logger.info(f"Agent chargé depuis {load_path}")

    return agent


def train_agent(env, agent, args):
    """Entraîne l'agent dans l'environnement de trading."""
    rewards_history = []
    portfolio_values_history = []
    actor_losses = []
    critic_losses = []
    alpha_history = []

    # Mesurer le temps d'entraînement
    start_time = datetime.now()

    # Boucle d'entraînement
    for episode in range(args.episodes):
        # Initialiser l'environnement
        state, _ = env.reset()
        episode_reward = 0

        terminated = False
        truncated = False
        step = 0

        # Boucle d'un épisode
        while not (terminated or truncated):
            # Sélectionner une action
            action = agent.act(state)

            # Appliquer l'action à l'environnement
            next_state, reward, terminated, truncated, info = env.step(action)

            # Stocker l'expérience dans le tampon de replay
            agent.remember(state, action, reward, next_state, terminated)

            # Mettre à jour l'état actuel
            state = next_state

            # Entraîner l'agent
            if step % 4 == 0:  # Entraîner tous les 4 pas pour être plus efficace
                metrics = agent.train()
                if step % 40 == 0:  # Journaliser moins fréquemment
                    actor_losses.append(metrics["actor_loss"])
                    critic_losses.append(metrics["critic_loss"])
                    alpha_history.append(metrics["alpha"])

            episode_reward += reward
            step += 1

        # Enregistrer la valeur finale du portefeuille et la récompense totale
        rewards_history.append(episode_reward)
        portfolio_values_history.append(env.get_portfolio_value())

        # Journaliser les progrès
        if (episode + 1) % 5 == 0 or episode == 0:
            elapsed_time = datetime.now() - start_time
            logger.info(
                f"Épisode {episode+1}/{args.episodes}: "
                f"Récompense={episode_reward:.2f}, "
                f"Portfolio={env.get_portfolio_value():.2f}, "
                f"Alpha={metrics['alpha']:.4f}, "
                f"Temps écoulé={elapsed_time}"
            )

    logger.info(f"Entraînement terminé en {datetime.now() - start_time}")

    # Retourner les métriques d'entraînement
    return {
        "rewards": rewards_history,
        "portfolio_values": portfolio_values_history,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "alpha_history": alpha_history,
    }


def evaluate_agent(env, agent, num_episodes=5):
    """Évalue l'agent après l'entraînement."""
    returns = []
    portfolio_values = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # En évaluation, nous utilisons la politique déterministe (sans exploration)
            action = agent.act(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
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


def visualize_results(metrics):
    """Visualise les résultats de l'entraînement."""
    plt.figure(figsize=(15, 12))

    # Rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title("Récompenses par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense")

    # Portfolio Value
    plt.subplot(3, 2, 2)
    plt.plot(metrics["portfolio_values"])
    plt.title("Valeur du portefeuille")
    plt.xlabel("Épisode")
    plt.ylabel("Valeur ($)")

    # Actor Loss
    plt.subplot(3, 2, 3)
    plt.plot(metrics["actor_losses"])
    plt.title("Perte de l'acteur")
    plt.xlabel("Itération")
    plt.ylabel("Perte")

    # Critic Loss
    plt.subplot(3, 2, 4)
    plt.plot(metrics["critic_losses"])
    plt.title("Perte du critique")
    plt.xlabel("Itération")
    plt.ylabel("Perte")

    # Alpha
    plt.subplot(3, 2, 5)
    plt.plot(metrics["alpha_history"])
    plt.title("Paramètre d'entropie (Alpha)")
    plt.xlabel("Itération")
    plt.ylabel("Alpha")

    plt.tight_layout()
    plt.savefig("sac_training_results.png")
    plt.show()


def main():
    """Fonction principale."""
    args = parse_arguments()

    # Récupérer les données
    data_integrator = RLDataIntegrator()

    try:
        # Calculer les dates basées sur le nombre de jours
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

        # Essayer de récupérer des données réelles avec la bonne méthode
        df = data_integrator.collect_market_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
        )
        logger.info(
            f"Données récupérées pour {args.symbol} sur {args.days} jours avec un intervalle de {args.interval}"
        )

    except Exception as e:
        # En cas d'erreur, utiliser des données synthétiques
        logger.warning(f"Erreur lors de la récupération des données: {e}")
        logger.info("Utilisation de données synthétiques à la place")

        df = data_integrator.generate_synthetic_data(
            n_samples=500, trend="random", volatility=0.02, with_sentiment=True
        )

    # Prétraiter les données, ce qui inclut l'ajout d'indicateurs techniques
    df = data_integrator.preprocess_market_data(df)

    # Créer l'environnement
    env = create_environment(df, args)

    # Créer l'agent
    agent = create_agent(env, args, args.load)

    # Entraîner l'agent si demandé
    if not args.load or args.episodes > 0:
        logger.info(f"Début de l'entraînement pour {args.episodes} épisodes...")
        metrics = train_agent(env, agent, args)

        # Sauvegarder le modèle si demandé
        if args.save:
            save_dir = (
                INFO_RETOUR_DIR
                / "models"
                / "sac"
                / f"{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            agent.save(save_dir)
            logger.info(f"Modèle sauvegardé dans {save_dir}")

        # Visualiser les résultats si demandé
        if args.visualize:
            visualize_results(metrics)

    # Évaluer l'agent
    logger.info(f"Évaluation de l'agent sur {args.eval_episodes} épisodes...")
    evaluate_agent(env, agent, args.eval_episodes)


if __name__ == "__main__":
    main()
