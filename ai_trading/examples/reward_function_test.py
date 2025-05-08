#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test des différentes fonctions de récompense dans l'environnement de trading.
Ce script permet de comparer les performances des différentes stratégies de récompense.
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configurer le logger pour enregistrer dans un fichier dans web_app/info_retour
os.makedirs(os.path.join(os.getcwd(), "web_app", "info_retour"), exist_ok=True)
log_file = os.path.join(os.getcwd(), "web_app", "info_retour", "reward_test.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.data.market_data import MarketDataFetcher
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.trading_environment import TradingEnvironment


# Configuration des arguments en ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Test des fonctions de récompense")
    parser.add_argument(
        "--symbol", type=str, default="BTC", help="Symbole de la crypto-monnaie"
    )
    parser.add_argument(
        "--days", type=int, default=60, help="Nombre de jours de données historiques"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualiser les résultats"
    )
    parser.add_argument(
        "--initial_balance",
        type=float,
        default=10000.0,
        help="Solde initial du portefeuille",
    )
    return parser.parse_args()


def fetch_market_data(symbol, days):
    """
    Récupère les données de marché pour le symbole spécifié.

    Args:
        symbol (str): Symbole de la crypto-monnaie
        days (int): Nombre de jours de données historiques

    Returns:
        pd.DataFrame: Données de marché
    """
    logger.info(f"Récupération des données pour {symbol}...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    fetcher = MarketDataFetcher()
    data = fetcher.fetch_crypto_data(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1d",
    )

    logger.info(f"Données récupérées: {len(data)} entrées")
    return data


def preprocess_data(data):
    """
    Prétraite les données pour ajouter des indicateurs techniques.

    Args:
        data (pd.DataFrame): Données brutes

    Returns:
        pd.DataFrame: Données prétraitées
    """
    from ai_trading.data.technical_indicators import TechnicalIndicators

    logger.info("Ajout des indicateurs techniques...")

    # S'assurer que les données ont suffisamment de points
    if len(data) < 30:
        logger.info(
            "Données insuffisantes, génération de données synthétiques supplémentaires."
        )
        data = generate_synthetic_data(days=60)

    # S'assurer que toutes les colonnes nécessaires sont présentes
    data_prepared = data.copy()

    # Utiliser 'close' ou 'price' comme base si certaines colonnes sont manquantes
    price_col = "close" if "close" in data_prepared.columns else "price"

    if "open" not in data_prepared.columns and price_col in data_prepared.columns:
        logger.info(
            "Colonne 'open' manquante, utilisation de la colonne de prix avec variation."
        )
        data_prepared["open"] = data_prepared[price_col] * np.random.uniform(
            0.995, 1.005, len(data_prepared)
        )

    if "high" not in data_prepared.columns and price_col in data_prepared.columns:
        logger.info(
            "Colonne 'high' manquante, utilisation de la colonne de prix avec augmentation."
        )
        data_prepared["high"] = data_prepared[price_col] * np.random.uniform(
            1.001, 1.01, len(data_prepared)
        )

    if "low" not in data_prepared.columns and price_col in data_prepared.columns:
        logger.info(
            "Colonne 'low' manquante, utilisation de la colonne de prix avec diminution."
        )
        data_prepared["low"] = data_prepared[price_col] * np.random.uniform(
            0.99, 0.999, len(data_prepared)
        )

    if "volume" not in data_prepared.columns:
        logger.info("Colonne 'volume' manquante, ajout de données synthétiques.")
        # Générer un volume arbitraire basé sur le prix
        base_price = data_prepared[price_col].mean() if len(data_prepared) > 0 else 100
        data_prepared["volume"] = (
            np.random.uniform(1000, 10000, len(data_prepared)) * base_price / 100
        )

    logger.info(f"Colonnes après préparation: {data_prepared.columns.tolist()}")

    # Ajouter des indicateurs techniques
    indicators = TechnicalIndicators()
    data_with_indicators = indicators.add_all_indicators(data_prepared)

    # Combler les valeurs manquantes si possible au lieu de supprimer les lignes
    logger.info("Comblement des valeurs manquantes...")

    # Remplissage des valeurs manquantes
    data_filled = (
        data_with_indicators.copy()
    )  # Pour éviter les avertissements de SettingWithCopyWarning
    data_filled = data_filled.fillna(data_filled.shift())  # Forward fill
    data_filled = data_filled.fillna(data_filled.shift(-1))  # Backward fill

    # Vérifier s'il reste des NaN
    na_count = data_filled.isna().sum().sum()
    if na_count > 0:
        logger.warning(f"Il reste {na_count} valeurs manquantes après comblement.")

    logger.info(f"Indicateurs ajoutés. Données finales: {len(data_filled)} lignes")
    return data_filled


def generate_synthetic_data(days=60, symbol="BTC"):
    """
    Génère des données synthétiques avec une tendance réaliste pour simuler des données de marché.

    Args:
        days (int): Nombre de jours de données à générer
        symbol (str): Symbole de la crypto-monnaie

    Returns:
        pd.DataFrame: DataFrame contenant les données synthétiques
    """
    logger.info(f"Génération de données synthétiques pour {symbol} sur {days} jours...")

    # Date de fin : aujourd'hui
    end_date = datetime.now()
    # Date de début : jours avant
    start_date = end_date - timedelta(days=days)

    # Créer une série de dates pour l'index
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Déterminer le prix de base en fonction du symbole
    base_price = (
        30000 if symbol.upper() == "BTC" else (2000 if symbol.upper() == "ETH" else 100)
    )

    # Générer les tendances et les fluctuations
    # Une tendance générale
    trend = np.linspace(-0.1, 0.2, len(date_range))
    # Ajouter une composante cyclique
    cycles = 0.1 * np.sin(np.linspace(0, 4 * np.pi, len(date_range)))
    # Ajouter du bruit aléatoire
    noise = np.random.normal(0, 0.01, len(date_range))

    # Combinaison pour créer les variations de prix
    changes = trend + cycles + noise

    # Calculer les prix
    prices = [base_price]
    for i in range(1, len(date_range)):
        next_price = prices[-1] * (1 + changes[i])
        prices.append(next_price)

    prices = np.array(prices)

    # Créer le DataFrame avec toutes les colonnes nécessaires
    df = pd.DataFrame(
        {
            "open": prices * np.random.uniform(0.997, 1.003, len(date_range)),
            "high": prices * np.random.uniform(1.002, 1.01, len(date_range)),
            "low": prices * np.random.uniform(0.99, 0.998, len(date_range)),
            "close": prices,
            "price": prices,
            "volume": np.random.uniform(
                base_price * 10, base_price * 100, len(date_range)
            ),
            "market_cap": prices
            * np.random.uniform(1e9, 1e10, len(date_range))
            / base_price,
            "source": ["synthetic"] * len(date_range),
        },
        index=date_range,
    )

    logger.info(f"Données synthétiques générées: {len(df)} points pour {symbol}")
    return df


def train_agent_with_reward(env, reward_function, episodes, visualize=False):
    """
    Entraîne un agent DQN avec une fonction de récompense spécifique.

    Args:
        env (TradingEnvironment): Environnement de trading
        reward_function (str): Fonction de récompense à utiliser
        episodes (int): Nombre d'épisodes d'entraînement
        visualize (bool): Visualiser l'entraînement

    Returns:
        dict: Résultats de l'entraînement
    """
    # Mettre à jour la fonction de récompense
    env.reward_function = reward_function

    # Créer l'agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=2000,
        batch_size=32,
        gamma=0.95,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Variables pour le suivi
    rewards_history = []
    portfolio_values_history = []
    transaction_counts = []

    # Entraînement
    logger.info(f"\nEntraînement avec fonction de récompense: {reward_function}")
    for episode in range(episodes):
        # Réinitialiser l'environnement
        observation, _ = env.reset()

        # Variables d'épisode
        episode_reward = 0
        done = False

        while not done:
            # Sélectionner une action
            action = agent.select_action(observation)

            # Exécuter l'action
            next_observation, reward, done, _, info = env.step(action)

            # Stocker l'expérience
            agent.remember(observation, action, reward, next_observation, done)

            # Apprendre de l'expérience
            agent.learn()

            # Mettre à jour l'observation
            observation = next_observation

            # Cumuler la récompense
            episode_reward += reward

        # Enregistrer les résultats
        rewards_history.append(episode_reward)
        portfolio_values_history.append(env.get_portfolio_value_history())
        transaction_counts.append(env.transaction_count)

        # Afficher les progrès
        logger.info(
            f"Épisode {episode+1}/{episodes} | "
            f"Récompense: {episode_reward:.2f} | "
            f"Valeur finale: {env.get_portfolio_value():.2f} | "
            f"Transactions: {env.transaction_count}"
        )

    # Visualiser les résultats si demandé
    if visualize:
        env.render()

    # Collecter les métriques
    final_portfolio_value = env.get_portfolio_value()
    max_portfolio_value = max(env.portfolio_value_history)
    max_drawdown = (
        max_portfolio_value - min(env.portfolio_value_history)
    ) / max_portfolio_value

    return {
        "reward_function": reward_function,
        "rewards_history": rewards_history,
        "portfolio_values_history": portfolio_values_history,
        "transaction_counts": transaction_counts,
        "final_portfolio_value": final_portfolio_value,
        "max_portfolio_value": max_portfolio_value,
        "max_drawdown": max_drawdown,
        "transaction_count": env.transaction_count,
    }


def compare_reward_functions(data, args):
    """
    Compare les différentes fonctions de récompense.

    Args:
        data (pd.DataFrame): Données prétraitées
        args: Arguments en ligne de commande
    """
    # Liste des fonctions de récompense à tester
    reward_functions = ["simple", "sharpe", "transaction_penalty", "drawdown"]

    # Résultats pour chaque fonction de récompense
    results = []

    # Tester chaque fonction de récompense
    for reward_function in reward_functions:
        # Créer l'environnement
        env = TradingEnvironment(
            df=data,
            initial_balance=args.initial_balance,
            transaction_fee=0.001,
            window_size=20,
            include_technical_indicators=True,
            risk_management=True,
            normalize_observation=True,
            reward_function=reward_function,
            risk_aversion=0.1,
            transaction_penalty=0.001,
            lookback_window=20,
            action_type="discrete",  # Type d'action: discrete ou continuous
            n_discrete_actions=5,  # Nombre d'actions discrètes par catégorie (achat/vente)
        )

        # Ajout manuel des attributs manquants (corriger le bug)
        if not hasattr(env, "action_type"):
            env.action_type = "discrete"
        if not hasattr(env, "use_risk_manager"):
            env.use_risk_manager = env.risk_management
        if not hasattr(env, "n_discrete_actions"):
            env.n_discrete_actions = 5

        # Entraîner l'agent
        result = train_agent_with_reward(
            env=env,
            reward_function=reward_function,
            episodes=args.episodes,
            visualize=args.visualize,
        )

        results.append(result)

    # Afficher les résultats comparatifs
    logger.info("\n==== Comparaison des fonctions de récompense ====")
    logger.info(
        f"{'Fonction':<20} | {'Valeur finale':<15} | {'Rendement':<15} | {'Max Drawdown':<15} | {'Transactions':<10}"
    )
    logger.info("-" * 80)

    for result in results:
        reward_function = result["reward_function"]
        final_value = result["final_portfolio_value"]
        return_pct = (final_value - args.initial_balance) / args.initial_balance * 100
        max_drawdown = result["max_drawdown"] * 100
        transactions = result["transaction_count"]

        logger.info(
            f"{reward_function:<20} | {final_value:<15.2f} | {return_pct:<15.2f}% | {max_drawdown:<15.2f}% | {transactions:<10}"
        )

    # Visualiser les résultats
    if args.visualize:
        visualize_comparison(results, args.initial_balance)


def visualize_comparison(results, initial_balance):
    """
    Visualise la comparaison des différentes fonctions de récompense.

    Args:
        results (list): Résultats pour chaque fonction de récompense
        initial_balance (float): Solde initial du portefeuille
    """
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Couleurs pour les différentes fonctions de récompense
    colors = ["blue", "green", "red", "purple"]

    # 1. Évolution de la valeur du portefeuille
    for i, result in enumerate(results):
        # Obtenir la dernière évolution du portefeuille
        portfolio_values = result["portfolio_values_history"][-1]
        reward_function = result["reward_function"]

        # Tracer l'évolution
        axs[0].plot(portfolio_values, color=colors[i], label=reward_function)

    # Ajouter une ligne pour la valeur initiale
    axs[0].axhline(y=initial_balance, color="black", linestyle="--", label="Initial")

    axs[0].set_title("Évolution de la valeur du portefeuille")
    axs[0].set_ylabel("Valeur ($)")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Récompenses par épisode
    for i, result in enumerate(results):
        rewards = result["rewards_history"]
        reward_function = result["reward_function"]

        axs[1].plot(rewards, color=colors[i], label=reward_function)

    axs[1].set_title("Récompenses par épisode")
    axs[1].set_ylabel("Récompense")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Nombre de transactions par épisode
    for i, result in enumerate(results):
        transactions = result["transaction_counts"]
        reward_function = result["reward_function"]

        axs[2].plot(transactions, color=colors[i], label=reward_function)

    axs[2].set_title("Nombre de transactions par épisode")
    axs[2].set_xlabel("Épisode")
    axs[2].set_ylabel("Transactions")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

    # Créer le répertoire des visualisations s'il n'existe pas
    visualization_dir = VISUALIZATION_DIR / "reward_functions"
    os.makedirs(visualization_dir, exist_ok=True)

    # Enregistrer la figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        visualization_dir, f"reward_functions_comparison_{timestamp}.png"
    )
    plt.savefig(filename)

    logger.info(f"\nVisualisations enregistrées dans: {filename}")
    plt.show()


def main():
    """
    Fonction principale.
    """
    try:
        # Analyser les arguments
        logger.info("Analyse des arguments...")
        args = parse_args()
        logger.info(f"Arguments: {args}")

        # Récupérer les données de marché
        logger.info(f"Récupération des données pour {args.symbol}...")
        data = fetch_market_data(args.symbol, args.days)
        logger.info(
            f"Données récupérées: {len(data)} entrées avec colonnes: {data.columns.tolist()}"
        )

        # Prétraiter les données
        logger.info("Prétraitement des données...")
        data_clean = preprocess_data(data)
        logger.info(
            f"Données prétraitées: {len(data_clean)} entrées avec colonnes: {data_clean.columns.tolist()}"
        )

        # Comparer les fonctions de récompense
        logger.info("Comparaison des fonctions de récompense...")
        compare_reward_functions(data_clean, args)

        logger.info("Script terminé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
        traceback.print_exc()
        print(f"Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
