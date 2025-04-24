#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test comparatif de fonctions de récompense avancées pour le trading.

Ce script compare différentes fonctions de récompense pour l'apprentissage par renforcement
dans un contexte de trading algorithmique:
1. Ratio de Sharpe
2. Pénalisation des transactions fréquentes
3. Récompense basée sur le drawdown

Usage:
    python advanced_reward_test.py --symbol BTC --days 60 --episodes 10 --visualize
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta

import colorlog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ajouter le répertoire parent au path pour pouvoir importer les modules personnalisés
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.rl.adaptive_normalization import AdaptiveNormalizer
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.risk_manager import RiskManager
from ai_trading.rl.technical_indicators import TechnicalIndicators
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector

# Configuration du logger
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(name)s:%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

logger = colorlog.getLogger("advanced_reward_test")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Utiliser le VISUALIZATION_DIR de la configuration
VISUALIZATION_DIR = os.path.join(
    os.path.dirname(__file__),
    "info_retour",
    "visualizations",
    "evaluation",
)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


def parse_args():
    """
    Parse les arguments de ligne de commande.
    """
    parser = argparse.ArgumentParser(
        description="Test de fonctions de récompense avancées pour le trading RL"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC", help="Symbole de la crypto-monnaie"
    )
    parser.add_argument(
        "--days", type=int, default=60, help="Nombre de jours de données historiques"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Intervalle des données (1h, 1d, etc.)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualiser les résultats"
    )
    return parser.parse_args()


def fetch_market_data(symbol, days, interval="1d"):
    """
    Récupère les données de marché pour une crypto-monnaie.

    Args:
        symbol (str): Symbole de la crypto-monnaie
        days (int): Nombre de jours de données historiques
        interval (str): Intervalle des données (1h, 1d, etc.)

    Returns:
        pd.DataFrame: DataFrame contenant les données de marché
    """
    logger.info(
        f"Récupération des données pour {symbol} sur {days} jours avec intervalle {interval}..."
    )

    # Utiliser EnhancedDataCollector pour récupérer les données
    collector = EnhancedDataCollector()

    try:
        # Tenter de récupérer les données réelles
        # Convertir le symbole en identifiant pour CoinGecko (ex: BTC -> bitcoin)
        coin_id = symbol.lower()
        if symbol == "BTC":
            coin_id = "bitcoin"
        elif symbol == "ETH":
            coin_id = "ethereum"

        data = collector.get_merged_price_data(
            coin_id=coin_id, days=days, include_fear_greed=True
        )

        # Vérifier si les données sont valides
        if data is not None and not data.empty and len(data) > 5:
            logger.info(f"Données récupérées avec succès: {len(data)} entrées")
            return data
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données: {e}")

    # Si les données réelles ne sont pas disponibles, utiliser des données synthétiques
    logger.warning(f"Utilisation de données synthétiques pour {symbol}")
    return generate_synthetic_data(symbol, days)


def generate_synthetic_data(symbol, days):
    """
    Génère des données synthétiques si les données réelles ne sont pas disponibles.

    Args:
        symbol (str): Symbole de la crypto-monnaie
        days (int): Nombre de jours de données historiques

    Returns:
        pd.DataFrame: DataFrame contenant les données synthétiques
    """
    logger.info(f"Génération de données synthétiques pour {symbol} sur {days} jours...")

    # Créer une série temporelle
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Paramètres de base
    initial_price = 10000 if symbol == "BTC" else 1000
    volatility = 0.03 if symbol == "BTC" else 0.05

    # Générer des prix avec une marche aléatoire
    np.random.seed(42)  # Pour la reproductibilité
    returns = np.random.normal(0.001, volatility, len(date_range))
    prices = [initial_price]

    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    prices = prices[1:]  # Retirer le prix initial

    # Créer le DataFrame
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            "low": [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            "close": prices,
            "volume": [p * np.random.uniform(500, 5000) for p in prices],
            # Ajouter des données de sentiment synthétiques
            "compound_score": np.random.uniform(-1, 1, len(prices)),
            "positive_score": np.random.uniform(0, 1, len(prices)),
            "negative_score": np.random.uniform(0, 1, len(prices)),
            "neutral_score": np.random.uniform(0, 1, len(prices)),
            "fear_greed_value": np.random.uniform(0, 100, len(prices)),
            "fear_greed_classification": np.random.choice(
                ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                len(prices),
            ),
        },
        index=date_range[: len(prices)],
    )

    # Normaliser les scores pour qu'ils soient cohérents
    for i in range(len(df)):
        total = (
            df.iloc[i]["positive_score"]
            + df.iloc[i]["negative_score"]
            + df.iloc[i]["neutral_score"]
        )
        df.iloc[i, df.columns.get_loc("positive_score")] /= total
        df.iloc[i, df.columns.get_loc("negative_score")] /= total
        df.iloc[i, df.columns.get_loc("neutral_score")] /= total

    logger.info(f"Données synthétiques générées: {len(df)} entrées")
    return df


def preprocess_data(df):
    """
    Prétraite les données pour l'apprentissage par renforcement.

    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes

    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    logger.info("Prétraitement des données...")

    # Copier le DataFrame pour éviter de modifier l'original
    data = df.copy()

    # S'assurer que l'index est un DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except:
            logger.warning("Impossible de convertir l'index en DatetimeIndex")

    # S'assurer que les colonnes OHLCV sont présentes
    if "price" in data.columns and "close" not in data.columns:
        data["close"] = data["price"]
    if "close" in data.columns and "price" not in data.columns:
        data["price"] = data["close"]
    if "close" in data.columns and "open" not in data.columns:
        data["open"] = data["close"]
    if "close" in data.columns and "high" not in data.columns:
        # Si high n'existe pas, on utilise close + une légère variation
        data["high"] = data["close"] * (1 + 0.01)  # +1% par rapport au close
    if "close" in data.columns and "low" not in data.columns:
        # Si low n'existe pas, on utilise close - une légère variation
        data["low"] = data["close"] * (1 - 0.01)  # -1% par rapport au close
    if "volume" not in data.columns:
        data["volume"] = 1000000  # Volume par défaut

    # Ajouter des indicateurs techniques
    if len(data) >= 20:  # Vérifier qu'il y a suffisamment de données
        indicators = TechnicalIndicators(data)
        data = indicators.add_all_indicators(data)
        logger.info(
            f"Indicateurs techniques ajoutés: {len(data.columns) - len(df.columns)} nouveaux indicateurs"
        )
    else:
        logger.warning(
            f"Pas assez de données pour calculer les indicateurs techniques ({len(data)} points de données)"
        )

    # Gérer les valeurs manquantes
    # Plutôt que de supprimer les lignes avec NaN, on va les interpoler
    data = data.interpolate(method="linear")

    # Si après interpolation il reste des NaN (au début des séries), on les remplace par des zéros
    data = data.fillna(0)

    logger.info(
        f"Données prétraitées: {len(data)} entrées avec {data.columns.tolist()[:5]}... et {len(data.columns) - 5} autres colonnes"
    )
    return data


def train_agent_with_reward(
    data,
    reward_function,
    episodes,
    window_size=20,
    initial_balance=10000,
    transaction_fee=0.001,
    action_type="discrete",
    n_discrete_actions=5,
):
    """
    Entraîne un agent DQN avec une fonction de récompense spécifique.

    Args:
        data (pd.DataFrame): Données prétraitées
        reward_function (str): Nom de la fonction de récompense
        episodes (int): Nombre d'épisodes d'entraînement
        window_size (int): Taille de la fenêtre d'observation
        initial_balance (float): Solde initial
        transaction_fee (float): Frais de transaction

    Returns:
        dict: Résultats de l'entraînement
    """
    logger.info(f"Entraînement avec la fonction de récompense '{reward_function}'...")

    # Paramètres du gestionnaire de risque
    risk_config = {
        "stop_loss_atr_factor": 3.0,
        "take_profit_atr_factor": 4.0,
        "volatility_lookback": 14,
    }
    risk_manager = RiskManager(config=risk_config)

    # Paramètres de normalisation adaptative
    normalize_observation = True
    normalizer = AdaptiveNormalizer(window_size=100, method="minmax")

    # Créer l'environnement de trading
    env = TradingEnvironment(
        df=data,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        reward_function=reward_function,
        include_technical_indicators=True,
        include_portfolio_info=True,
        normalize_observation=normalize_observation,
        normalizer=normalizer,
        risk_manager=risk_manager,
        action_type=action_type,
        n_discrete_actions=n_discrete_actions,
        use_risk_manager=True,
    )

    # Définir les paramètres de l'agent DQN
    state_size = env.observation_space.shape[0]
    action_size = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )

    # Créer l'agent DQN
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=10000,
    )

    # Variables pour stocker les résultats
    episode_rewards = []
    portfolio_values = []

    # Entraînement de l'agent
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Sélectionner une action
            action = agent.act(state)

            # Exécuter l'action
            # La méthode step peut retourner 5 valeurs dans les nouvelles versions de gym
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            # Stocker l'expérience dans la mémoire de l'agent
            agent.remember(state, action, reward, next_state, done)

            # Apprendre de l'expérience
            agent.replay()

            # Mettre à jour l'état
            state = next_state

            # Accumuler la récompense
            total_reward += reward

        # Stocker les résultats de l'épisode
        episode_rewards.append(total_reward)
        portfolio_values.append(env.get_portfolio_value())

        # Afficher les progrès
        if (episode + 1) % max(1, episodes // 10) == 0:
            logger.info(
                f"Épisode {episode + 1}/{episodes}, Récompense: {total_reward:.2f}, "
                f"Portfolio: ${env.get_portfolio_value():.2f}, "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    # Calculer les métriques finales
    final_portfolio = env.get_portfolio_value()
    roi = (final_portfolio - initial_balance) / initial_balance * 100

    # Calcul du ratio de Sharpe
    if len(env.returns_history) > 0:
        returns = np.array(env.returns_history)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)  # Annualisé
    else:
        sharpe = 0

    # Calcul du drawdown maximal
    if len(env.portfolio_value_history) > 0:
        peak = np.maximum.accumulate(env.portfolio_value_history)
        drawdown = (peak - env.portfolio_value_history) / peak
        max_drawdown = drawdown.max() * 100 if len(drawdown) > 0 else 0
    else:
        max_drawdown = 0

    # Résultats de l'entraînement
    results = {
        "reward_function": reward_function,
        "final_portfolio": final_portfolio,
        "roi": roi,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "total_reward": sum(episode_rewards),
        "episode_rewards": episode_rewards,
        "portfolio_values": portfolio_values,
    }

    logger.info(f"Résultats pour '{reward_function}':")
    logger.info(f"  Portfolio final: ${final_portfolio:.2f}")
    logger.info(f"  ROI: {roi:.2f}%")
    logger.info(f"  Ratio de Sharpe: {sharpe:.4f}")
    logger.info(f"  Drawdown maximal: {max_drawdown:.2f}%")

    return results


def compare_reward_functions(data, args):
    """
    Compare différentes fonctions de récompense.

    Args:
        data (pd.DataFrame): Données prétraitées
        args: Arguments de ligne de commande
    """
    logger.info("Comparaison des différentes fonctions de récompense...")

    # Liste des fonctions de récompense à comparer
    reward_functions = [
        "simple",  # Récompense de base (changement de valeur du portefeuille)
        "sharpe",  # Récompense basée sur le ratio de Sharpe
        "transaction_penalty",  # Récompense avec pénalité pour les transactions fréquentes
        "drawdown",  # Récompense avec pénalité pour les drawdowns importants
    ]

    # Entraîner un agent avec chaque fonction de récompense
    results = {}
    for reward_function in reward_functions:
        results[reward_function] = train_agent_with_reward(
            data=data,
            reward_function=reward_function,
            episodes=args.episodes,
            window_size=20,
            initial_balance=10000,
            transaction_fee=0.001,
            action_type="discrete",
            n_discrete_actions=5,
        )

    # Afficher les résultats récapitulatifs
    logger.info("\nRésultats comparatifs des fonctions de récompense:")
    logger.info("-" * 80)
    logger.info(
        f"{'Fonction':20} | {'Portfolio':12} | {'ROI':8} | {'Sharpe':8} | {'Drawdown':10}"
    )
    logger.info("-" * 80)

    for reward_function, result in results.items():
        logger.info(
            f"{reward_function:20} | ${result['final_portfolio']:<11.2f} | {result['roi']:<7.2f}% | "
            f"{result['sharpe_ratio']:<7.4f} | {result['max_drawdown']:<9.2f}%"
        )

    # Visualiser les résultats si demandé
    if args.visualize:
        visualize_comparison(results, args.symbol)

    return results


def visualize_comparison(results, symbol):
    """
    Visualise les résultats de la comparaison des fonctions de récompense.

    Args:
        results (dict): Résultats de l'entraînement pour chaque fonction de récompense
        symbol (str): Symbole de la crypto-monnaie
    """
    logger.info("Visualisation des résultats...")

    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Comparaison des fonctions de récompense pour {symbol}", fontsize=16)

    # Palette de couleurs
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # 1. Évolution de la valeur du portefeuille
    ax = axs[0, 0]
    for i, (reward_function, result) in enumerate(results.items()):
        ax.plot(result["portfolio_values"], label=reward_function, color=colors[i])
    ax.set_title("Évolution de la valeur du portefeuille")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Valeur ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Récompenses cumulées
    ax = axs[0, 1]
    for i, (reward_function, result) in enumerate(results.items()):
        cumulative_rewards = np.cumsum(result["episode_rewards"])
        ax.plot(cumulative_rewards, label=reward_function, color=colors[i])
    ax.set_title("Récompenses cumulées")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Récompense cumulée")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Comparaison des ROI finaux
    ax = axs[1, 0]
    reward_functions = list(results.keys())
    roi_values = [result["roi"] for result in results.values()]
    ax.bar(reward_functions, roi_values, color=colors)
    ax.set_title("Comparaison des ROI finaux")
    ax.set_xlabel("Fonction de récompense")
    ax.set_ylabel("ROI (%)")
    for i, v in enumerate(roi_values):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Comparaison des ratios de Sharpe et drawdowns
    ax = axs[1, 1]
    ax2 = ax.twinx()

    sharpe_values = [result["sharpe_ratio"] for result in results.values()]
    drawdown_values = [result["max_drawdown"] for result in results.values()]

    ax.bar(
        [i - 0.2 for i in range(len(reward_functions))],
        sharpe_values,
        width=0.4,
        color="green",
        alpha=0.7,
        label="Sharpe",
    )
    ax2.bar(
        [i + 0.2 for i in range(len(reward_functions))],
        drawdown_values,
        width=0.4,
        color="red",
        alpha=0.7,
        label="Drawdown",
    )

    ax.set_title("Ratio de Sharpe et Drawdown maximal")
    ax.set_xlabel("Fonction de récompense")
    ax.set_ylabel("Ratio de Sharpe", color="green")
    ax2.set_ylabel("Drawdown maximal (%)", color="red")
    ax.set_xticks(range(len(reward_functions)))
    ax.set_xticklabels(reward_functions)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(sharpe_values):
        ax.text(i - 0.2, v + 0.1, f"{v:.2f}", ha="center", color="green")
    for i, v in enumerate(drawdown_values):
        ax2.text(i + 0.2, v + 1, f"{v:.2f}%", ha="center", color="red")

    # Légende
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.grid(True, alpha=0.3)

    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Sauvegarder la figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_reward_comparison_{timestamp}.png"
    filepath = os.path.join(VISUALIZATION_DIR, filename)
    plt.savefig(filepath)
    logger.info(f"Visualisation sauvegardée dans {filepath}")

    plt.close()


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
        logger.info("Comparaison des fonctions de récompense avancées...")
        compare_reward_functions(data_clean, args)

        logger.info("Script terminé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
        traceback.print_exc()
        print(f"Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
