#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de la validation croisée temporelle pour optimiser et évaluer
des agents de trading sur des données historiques de crypto-monnaies.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configurer le logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.data.data_collector import EnhancedDataCollector
from ai_trading.data.data_processor import EnhancedMarketDataPreprocessor
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.utils.temporal_cross_validation import (
    TemporalCrossValidator,
    WalkForwardOptimizer,
)


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Exemple de validation croisée temporelle pour l'optimisation d'agents de trading"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Symbole de la crypto-monnaie (ex: BTC, ETH)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Nombre de jours de données historiques à utiliser",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Intervalle des données (ex: 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--initial_train_size",
        type=float,
        default=0.6,
        help="Proportion initiale pour l'entraînement",
    )
    parser.add_argument(
        "--n_splits", type=int, default=3, help="Nombre de divisions temporelles"
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=5,
        help="Écart entre l'entraînement et la validation (en périodes)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualiser les résultats"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Exécuter l'optimisation des hyperparamètres",
    )

    return parser.parse_args()


def collect_data(symbol, days, interval):
    """Collecte des données historiques pour une crypto-monnaie."""
    logger.info(
        f"Collecte des données pour {symbol} sur {days} jours avec un intervalle de {interval}"
    )

    try:
        # Utiliser notre collecteur de données amélioré
        collector = EnhancedDataCollector()
        data = collector.get_historical_market_data(
            symbol=symbol, days=days, interval=interval
        )

        # Si aucune donnée n'est disponible, utiliser des données synthétiques
        if data is None or len(data) == 0:
            logger.warning(
                "Aucune donnée n'a pu être récupérée. Génération de données synthétiques."
            )
            return generate_synthetic_data(days)

        return data
    except Exception as e:
        logger.error(f"Erreur lors de la collecte des données: {str(e)}")
        logger.info("Génération de données synthétiques à la place.")
        return generate_synthetic_data(days)


def generate_synthetic_data(days):
    """Génère des données synthétiques pour les tests."""
    np.random.seed(42)  # Pour la reproductibilité

    # Générer une tendance avec des cycles
    t = np.linspace(0, 2 * np.pi, days)
    price = 10000 + 2000 * np.sin(t) + 50 * np.cumsum(np.random.randn(days))

    # Créer un dataframe
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    df = pd.DataFrame(
        {
            "open": price * np.random.uniform(0.99, 1.01, days),
            "high": price * np.random.uniform(1.01, 1.03, days),
            "low": price * np.random.uniform(0.97, 0.99, days),
            "close": price * np.random.uniform(0.98, 1.02, days),
            "volume": np.random.uniform(1000, 5000, days) * price / 10000,
        },
        index=dates,
    )

    return df


def preprocess_data(data):
    """Prétraite les données et ajoute des fonctionnalités techniques."""
    logger.info("Prétraitement des données et ajout d'indicateurs techniques")

    preprocessor = EnhancedMarketDataPreprocessor()

    # Ajouter des indicateurs techniques
    data = preprocessor.add_technical_indicators(data)

    # Normaliser les données
    normalized_data = preprocessor.normalize_data(data)

    return normalized_data


def create_agent(state_size, action_size, params=None):
    """Crée un agent SAC avec les paramètres spécifiés."""
    default_params = {
        "hidden_size": 128,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "discount_factor": 0.99,
        "tau": 0.005,
        "entropy_regularization": 0.1,
    }

    # Mettre à jour les paramètres par défaut avec ceux fournis
    if params:
        default_params.update(params)

    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        action_bounds=[-1, 1],
        hidden_size=default_params["hidden_size"],
        actor_learning_rate=default_params["learning_rate"],
        critic_learning_rate=default_params["learning_rate"],
        alpha_learning_rate=default_params["learning_rate"],
        discount_factor=default_params["discount_factor"],
        tau=default_params["tau"],
        batch_size=default_params["batch_size"],
        entropy_regularization=default_params["entropy_regularization"],
        train_alpha=True,
    )

    return agent


def train_agent(agent, env, episodes=10, max_steps_per_episode=None):
    """Entraîne un agent sur un environnement de trading."""
    rewards = []
    portfolio_values = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # Sélectionner une action
            action = agent.act(state)

            # Exécuter l'action dans l'environnement
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            # Mémoriser l'expérience et entraîner l'agent
            agent.remember(state, action, reward, next_state, done)
            agent.train()

            # Mettre à jour l'état
            state = next_state
            total_reward += reward
            step += 1

            # Si un nombre maximum d'étapes est spécifié, arrêter si atteint
            if max_steps_per_episode and step >= max_steps_per_episode:
                break

        # Enregistrer les métriques de l'épisode
        portfolio_value = env.portfolio_value
        rewards.append(total_reward)
        portfolio_values.append(portfolio_value)

        logger.info(
            f"Épisode {episode + 1}/{episodes} - Récompense: {total_reward:.2f}, "
            f"Valeur du portefeuille: {portfolio_value:.2f}"
        )

    return {"rewards": rewards, "portfolio_values": portfolio_values}


def evaluate_agent(agent, env, max_steps=None):
    """Évalue un agent sur un environnement de trading."""
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    portfolio_history = [env.portfolio_value]

    while not done:
        # Sélectionner une action (mode évaluation)
        action = agent.act(state, evaluate=True)

        # Exécuter l'action dans l'environnement
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated

        # Mettre à jour l'état
        state = next_state
        total_reward += reward
        step += 1

        # Enregistrer la valeur du portefeuille
        portfolio_history.append(env.portfolio_value)

        # Si un nombre maximum d'étapes est spécifié, arrêter si atteint
        if max_steps and step >= max_steps:
            break

    # Calculer les métriques d'évaluation
    initial_value = portfolio_history[0]
    final_value = portfolio_history[-1]
    profit_pct = (final_value - initial_value) / initial_value * 100

    # Calculer le ratio de Sharpe (version simplifiée)
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

    logger.info(
        f"Évaluation - Récompense: {total_reward:.2f}, Profit: {profit_pct:.2f}%, "
        f"Ratio de Sharpe: {sharpe_ratio:.4f}"
    )

    return {
        "total_reward": total_reward,
        "profit_pct": profit_pct,
        "sharpe_ratio": sharpe_ratio,
        "portfolio_history": portfolio_history,
    }


def model_fn(train_data, params):
    """Fonction pour créer et entraîner un modèle à partir des données d'entraînement."""
    # Créer un environnement de trading
    env = TradingEnvironment(
        df=train_data,
        initial_balance=10000.0,
        window_size=params.get("window_size", 10),
        transaction_fee=params.get("transaction_fee", 0.001),
        reward_scaling=params.get("reward_scaling", 1.0),
    )

    # Créer un agent
    agent_params = {
        "hidden_size": params.get("hidden_size", 128),
        "learning_rate": params.get("learning_rate", 0.0003),
        "batch_size": params.get("batch_size", 64),
        "discount_factor": params.get("discount_factor", 0.99),
        "entropy_regularization": params.get("entropy_regularization", 0.1),
    }

    agent = create_agent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        params=agent_params,
    )

    # Entraîner l'agent
    train_agent(
        agent=agent,
        env=env,
        episodes=params.get("train_episodes", 5),
        max_steps_per_episode=None,  # Utiliser toutes les données disponibles
    )

    # Retourner un dictionnaire contenant l'agent et l'environnement
    return {
        "agent": agent,
        "env_params": {
            "window_size": params.get("window_size", 10),
            "transaction_fee": params.get("transaction_fee", 0.001),
            "reward_scaling": params.get("reward_scaling", 1.0),
        },
    }


def eval_fn(model, data):
    """Fonction pour évaluer un modèle sur des données de validation."""
    # Récupérer l'agent et les paramètres de l'environnement
    agent = model["agent"]
    env_params = model["env_params"]

    # Créer un environnement de trading pour l'évaluation
    env = TradingEnvironment(
        df=data,
        initial_balance=10000.0,
        window_size=env_params["window_size"],
        transaction_fee=env_params["transaction_fee"],
        reward_scaling=env_params["reward_scaling"],
    )

    # Évaluer l'agent
    eval_results = evaluate_agent(agent, env)

    # On peut retourner différentes métriques selon les besoins
    # Ici, on utilise le profit_pct comme métrique principale
    return eval_results["profit_pct"]


def validate_with_temporal_cv(data, args):
    """Effectue une validation croisée temporelle sur les données."""
    logger.info("Exécution de la validation croisée temporelle")

    # Définir les paramètres par défaut pour le modèle
    default_params = {
        "window_size": 10,
        "transaction_fee": 0.001,
        "reward_scaling": 1.0,
        "hidden_size": 128,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "discount_factor": 0.99,
        "entropy_regularization": 0.1,
        "train_episodes": 5,
    }

    # Créer un validateur croisé temporel
    cv = TemporalCrossValidator(
        data=data,
        n_splits=args.n_splits,
        initial_train_size=args.initial_train_size,
        gap=args.gap,
    )

    # Définir une fonction modèle partielle avec les paramètres par défaut
    def model_fn_with_default_params(train_data):
        return model_fn(train_data, default_params)

    # Évaluer avec la validation croisée temporelle
    results = cv.evaluate(model_fn_with_default_params, eval_fn)

    logger.info(f"Résultats de la validation croisée:")
    logger.info(f"  Train metrics: {results['train_metrics']}")
    logger.info(f"  Validation metrics: {results['val_metrics']}")
    logger.info(f"  Moyenne train: {np.mean(results['train_metrics']):.2f}%")
    logger.info(f"  Moyenne validation: {np.mean(results['val_metrics']):.2f}%")

    # Visualiser les résultats si demandé
    if args.visualize:
        visualize_cv_results(results)

    return results


def optimize_with_walk_forward(data, args):
    """Optimise les hyperparamètres avec une approche walk-forward."""
    logger.info("Exécution de l'optimisation walk-forward")

    # Définir une grille de paramètres à explorer
    param_grid = {
        "window_size": [5, 10, 20],
        "reward_scaling": [0.5, 1.0, 2.0],
        "learning_rate": [0.0001, 0.0003, 0.001],
        "entropy_regularization": [0.01, 0.1, 0.5],
        "train_episodes": [3],  # Limité pour la démonstration
    }

    # Créer un optimiseur walk-forward
    optimizer = WalkForwardOptimizer(
        data=data,
        param_grid=param_grid,
        n_splits=args.n_splits,
        initial_train_size=args.initial_train_size,
        gap=args.gap,
        evaluation_metric="profit",  # Maximiser le profit
    )

    # Optimiser les hyperparamètres
    optimization_results = optimizer.optimize(model_fn, eval_fn)

    # Afficher les meilleurs hyperparamètres trouvés
    best_params = optimization_results["best_params"]
    best_metric = optimization_results["best_metric"]

    logger.info(f"Meilleurs hyperparamètres trouvés:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Métrique moyenne sur validation: {best_metric:.2f}%")

    # Visualiser les résultats si demandé
    if args.visualize:
        visualize_optimization_results(optimization_results)

    return optimization_results


def visualize_cv_results(results):
    """Visualise les résultats de la validation croisée."""
    plt.figure(figsize=(12, 6))

    # Créer un graphique à barres pour les métriques
    train_metrics = results["train_metrics"]
    val_metrics = results["val_metrics"]
    x = np.arange(len(train_metrics))

    plt.bar(x - 0.2, train_metrics, width=0.4, label="Train")
    plt.bar(x + 0.2, val_metrics, width=0.4, label="Validation")

    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Division")
    plt.ylabel("Profit (%)")
    plt.title("Résultats de la validation croisée temporelle")
    plt.xticks(x, [f"Fold {i+1}" for i in range(len(train_metrics))])
    plt.legend()

    plt.savefig("cv_results.png")
    plt.close()


def visualize_optimization_results(results):
    """Visualise les résultats de l'optimisation des hyperparamètres."""
    # Créer une figure pour les métriques de validation moyennes
    plt.figure(figsize=(12, 6))

    # Extraire les métriques et trier par ordre décroissant
    all_results = results["all_results"]
    metrics = [r["avg_val_metric"] for r in all_results]
    param_labels = [str(r["params"]) for r in all_results]

    # Trier par métrique
    sorted_indices = np.argsort(metrics)[::-1]
    sorted_metrics = [metrics[i] for i in sorted_indices]
    sorted_labels = [param_labels[i] for i in sorted_indices]

    # Limiter le nombre de paramètres affichés pour la lisibilité
    max_display = 10
    if len(sorted_metrics) > max_display:
        sorted_metrics = sorted_metrics[:max_display]
        sorted_labels = sorted_labels[:max_display]

    # Créer un graphique à barres horizontal
    y_pos = np.arange(len(sorted_metrics))
    plt.barh(y_pos, sorted_metrics, align="center")
    plt.yticks(
        y_pos,
        [label[:30] + "..." if len(label) > 30 else label for label in sorted_labels],
    )
    plt.xlabel("Profit moyen (%)")
    plt.title("Résultats de l'optimisation des hyperparamètres")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimization_results.png")
    plt.close()


def main():
    """Fonction principale pour exécuter l'exemple de validation croisée temporelle."""
    args = parse_arguments()

    # Collecter les données
    data = collect_data(args.symbol, args.days, args.interval)

    # Prétraiter les données
    preprocessed_data = preprocess_data(data)

    # Effectuer une simple validation croisée temporelle
    cv_results = validate_with_temporal_cv(preprocessed_data, args)

    # Si l'optimisation est demandée, exécuter l'optimisation walk-forward
    if args.optimize:
        optimization_results = optimize_with_walk_forward(preprocessed_data, args)


if __name__ == "__main__":
    main()
