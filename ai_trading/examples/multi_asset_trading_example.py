import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.rl.multi_asset_trading_environment import MultiAssetTradingEnvironment

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(symbols, days=100, seed=42):
    """
    Génère des données synthétiques pour les tests.

    Args:
        symbols (list): Liste des symboles pour lesquels générer des données
        days (int): Nombre de jours de données à générer
        seed (int): Graine aléatoire pour la reproductibilité

    Returns:
        dict: Dictionnaire de DataFrames avec les données générées
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=days, freq="D")

    data_dict = {}

    # Paramètres pour différents actifs
    parameters = {
        "BTC": {"trend": 0.2, "volatility": 0.05, "start_price": 10000},
        "ETH": {"trend": 0.15, "volatility": 0.07, "start_price": 300},
        "LTC": {"trend": 0.10, "volatility": 0.04, "start_price": 50},
        "XRP": {"trend": 0.05, "volatility": 0.06, "start_price": 0.25},
        "ADA": {"trend": 0.25, "volatility": 0.08, "start_price": 0.10},
        "SOL": {"trend": 0.30, "volatility": 0.10, "start_price": 20},
        "DOT": {"trend": 0.20, "volatility": 0.09, "start_price": 5},
        "LINK": {"trend": 0.15, "volatility": 0.06, "start_price": 15},
    }

    for symbol in symbols:
        if symbol not in parameters:
            # Utiliser des paramètres par défaut pour les symboles non listés
            params = {"trend": 0.1, "volatility": 0.05, "start_price": 10}
        else:
            params = parameters[symbol]

        # Générer une tendance avec volatilité
        prices = np.zeros(days)
        prices[0] = params["start_price"]

        for i in range(1, days):
            # Tendance + bruit aléatoire
            daily_return = np.random.normal(
                params["trend"] / days, params["volatility"] / np.sqrt(days)
            )
            prices[i] = prices[i - 1] * (1 + daily_return)

        # Créer un DataFrame avec les prix
        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, days)),
                "high": prices * (1 + np.random.normal(0.01, 0.01, days)),
                "low": prices * (1 + np.random.normal(-0.01, 0.01, days)),
                "close": prices,
                "volume": np.random.lognormal(10, 1, days),
            },
            index=dates,
        )

        # Corriger les prix high/low par rapport à open/close
        for i in range(days):
            high = max(df.iloc[i]["open"], df.iloc[i]["close"], df.iloc[i]["high"])
            low = min(df.iloc[i]["open"], df.iloc[i]["close"], df.iloc[i]["low"])
            df.iloc[i, df.columns.get_loc("high")] = high
            df.iloc[i, df.columns.get_loc("low")] = low

        data_dict[symbol] = df

    return data_dict


def random_agent(observation, action_space):
    """
    Agent aléatoire pour le trading multi-actifs.

    Args:
        observation: Observation actuelle de l'environnement
        action_space: Espace d'action de l'environnement

    Returns:
        np.ndarray: Action aléatoire
    """
    return action_space.sample()


def test_multi_asset_environment(data_dict, episodes=5, visualize=True):
    """
    Teste l'environnement de trading multi-actifs avec un agent aléatoire.

    Args:
        data_dict (dict): Dictionnaire de DataFrames avec les données à utiliser
        episodes (int): Nombre d'épisodes à exécuter
        visualize (bool): Si True, visualise les résultats

    Returns:
        dict: Résultats du test
    """
    # Créer l'environnement
    env = MultiAssetTradingEnvironment(
        data_dict=data_dict,
        initial_balance=10000.0,
        window_size=20,
        include_technical_indicators=True,
        allocation_method="smart",
        rebalance_frequency=5,
        max_active_positions=3,
    )

    # Stocker les résultats
    results = {"episode_returns": [], "portfolio_values": [], "final_allocations": []}

    # Exécuter plusieurs épisodes
    for episode in range(episodes):
        logger.info(f"Épisode {episode+1}/{episodes}")

        # Réinitialiser l'environnement
        obs, _ = env.reset()
        done = False
        total_reward = 0

        # Boucle principale de l'épisode
        while not done:
            # Générer une action aléatoire
            action = random_agent(obs, env.action_space)

            # Exécuter l'action dans l'environnement
            obs, reward, done, _, info = env.step(action)

            # Accumuler la récompense
            total_reward += reward

            # Afficher des informations sur la progression
            if env.current_step % 20 == 0:
                portfolio_value = info["portfolio_value"]
                portfolio_weights = info["portfolio_weights"]
                logger.info(
                    f"Étape {env.current_step}, Portefeuille: ${portfolio_value:.2f}"
                )
                logger.info(f"Allocation: {portfolio_weights}")

        # Enregistrer les résultats de l'épisode
        final_portfolio_value = env.get_portfolio_value()
        final_return = (final_portfolio_value / env.initial_balance - 1) * 100

        logger.info(f"Épisode {episode+1} terminé:")
        logger.info(f"  Valeur finale du portefeuille: ${final_portfolio_value:.2f}")
        logger.info(f"  Rendement: {final_return:.2f}%")
        logger.info(f"  Récompense totale: {total_reward:.4f}")

        # Sauvegarder les résultats
        results["episode_returns"].append(final_return)
        results["portfolio_values"].append(env.portfolio_value_history)

        if env.allocation_history:
            results["final_allocations"].append(env.allocation_history[-1])

        # Visualiser les allocations de portefeuille si demandé
        if visualize:
            env.visualize_portfolio_allocation()

    # Afficher les résultats moyens
    avg_return = np.mean(results["episode_returns"])
    logger.info(f"Rendement moyen sur {episodes} épisodes: {avg_return:.2f}%")

    # Visualiser les résultats agrégés si demandé
    if visualize and episodes > 1:
        visualize_results(results, env.symbols)

    return results


def visualize_results(results, symbols):
    """
    Visualise les résultats des tests sur l'environnement de trading multi-actifs.

    Args:
        results (dict): Résultats à visualiser
        symbols (list): Liste des symboles des actifs
    """
    # Créer un répertoire pour les visualisations
    visualization_dir = VISUALIZATION_DIR / "multi_asset"
    os.makedirs(visualization_dir, exist_ok=True)

    # 1. Graphique des rendements par épisode
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(results["episode_returns"]) + 1), results["episode_returns"])
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Épisode")
    plt.ylabel("Rendement (%)")
    plt.title("Rendement par épisode")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(visualization_dir, "episode_returns.png"))
    plt.close()

    # 2. Graphique de l'évolution du portefeuille pour chaque épisode
    plt.figure(figsize=(12, 8))
    for i, values in enumerate(results["portfolio_values"]):
        plt.plot(values, label=f"Épisode {i+1}")
    plt.axhline(y=10000, color="r", linestyle="--", alpha=0.5, label="Capital initial")
    plt.xlabel("Étape")
    plt.ylabel("Valeur du portefeuille ($)")
    plt.title("Évolution de la valeur du portefeuille")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(visualization_dir, "portfolio_evolution.png"))
    plt.close()

    # 3. Graphique des allocations finales moyennes
    if results["final_allocations"]:
        avg_allocation = {}
        for symbol in symbols:
            avg_allocation[symbol] = np.mean(
                [alloc.get(symbol, 0) for alloc in results["final_allocations"]]
            )

        plt.figure(figsize=(10, 8))
        plt.pie(
            [avg_allocation[symbol] for symbol in symbols],
            labels=[f"{symbol}: {avg_allocation[symbol]:.1%}" for symbol in symbols],
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.tab10.colors[: len(symbols)],
        )
        plt.axis("equal")
        plt.title("Allocation moyenne du portefeuille (final)")
        plt.savefig(os.path.join(visualization_dir, "average_allocation.png"))
        plt.close()


def main():
    """Fonction principale pour exécuter l'exemple."""
    parser = argparse.ArgumentParser(description="Exemple de trading multi-actifs")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH", "LTC", "ADA", "SOL"],
        help="Symboles des crypto-monnaies à trader",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Nombre d'épisodes à exécuter"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualiser les résultats"
    )
    parser.add_argument(
        "--days", type=int, default=100, help="Nombre de jours de données"
    )

    args = parser.parse_args()

    # Générer des données synthétiques
    logger.info(f"Génération de données synthétiques pour {args.symbols}...")
    data_dict = generate_synthetic_data(args.symbols, days=args.days)

    # Tester l'environnement
    test_multi_asset_environment(
        data_dict, episodes=args.episodes, visualize=args.visualize
    )

    logger.info("Test terminé.")


if __name__ == "__main__":
    main()
