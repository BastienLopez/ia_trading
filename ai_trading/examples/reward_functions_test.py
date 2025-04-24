import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.rl.adaptive_normalization import AdaptiveNormalizer
from ai_trading.rl.risk_manager import RiskManager
from ai_trading.rl.technical_indicators import TechnicalIndicators

# Imports de notre projet
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector


def preprocess_data(df, window_size=20):
    """
    Prétraite les données pour les rendre utilisables par l'environnement.

    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes
        window_size (int): Taille de la fenêtre d'observation

    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    # Vérifier que le DataFrame contient les colonnes nécessaires
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing}")

    # Convertir la timestamp en datetime si ce n'est pas déjà fait
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Trier par ordre chronologique
    df = df.sort_values("timestamp")

    # Définir timestamp comme index
    df = df.set_index("timestamp")

    # Ajouter les indicateurs techniques
    tech_indicators = TechnicalIndicators()
    df = tech_indicators.add_all_indicators(df)

    # Supprimer les lignes avec des valeurs NaN
    df_clean = df.interpolate(method="linear").ffill().bfill()

    # Vérifier qu'il reste suffisamment de données après le nettoyage
    if len(df_clean) < window_size:
        raise ValueError(
            f"Après nettoyage, il ne reste que {len(df_clean)} points de données, ce qui est inférieur à window_size={window_size}"
        )

    return df_clean


def generate_synthetic_data(symbol, days=30, volatility=0.02):
    """
    Génère des données synthétiques pour les tests.

    Args:
        symbol (str): Symbole de la cryptomonnaie
        days (int): Nombre de jours à générer
        volatility (float): Volatilité des prix

    Returns:
        pd.DataFrame: DataFrame contenant les données synthétiques
    """
    np.random.seed(42)  # Pour la reproductibilité

    # Générer les timestamps
    end_date = datetime.now()
    date_range = pd.date_range(end=end_date, periods=days + 1, freq="D")

    # Générer les prix avec une marche aléatoire
    initial_price = 10000.0  # Prix initial
    returns = np.random.normal(0, volatility, days + 1)

    # S'assurer que les rendements sont entre -10% et +10%
    returns = np.clip(returns, -0.1, 0.1)

    # Calculer les prix
    prices = initial_price * (1 + np.cumsum(returns))

    # Générer les autres colonnes
    open_prices = prices * (1 + np.random.normal(0, volatility / 2, days + 1))
    high_prices = prices * (1 + np.abs(np.random.normal(0, volatility, days + 1)))
    low_prices = prices * (1 - np.abs(np.random.normal(0, volatility, days + 1)))
    volumes = np.random.lognormal(10, 1, days + 1) * 1000

    # S'assurer que high >= open/close >= low
    for i in range(days + 1):
        high_prices[i] = max(high_prices[i], open_prices[i], prices[i])
        low_prices[i] = min(low_prices[i], open_prices[i], prices[i])

    # Créer le DataFrame
    data = {
        "timestamp": date_range,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": prices,
        "volume": volumes,
        "symbol": symbol,
    }

    return pd.DataFrame(data)


def compare_reward_functions(df, episodes=5, initial_balance=10000.0, window_size=20):
    """
    Compare les différentes fonctions de récompense.

    Args:
        df (pd.DataFrame): DataFrame contenant les données prétraitées
        episodes (int): Nombre d'épisodes à exécuter
        initial_balance (float): Solde initial
        window_size (int): Taille de la fenêtre d'observation

    Returns:
        dict: Dictionnaire contenant les résultats pour chaque fonction de récompense
    """
    reward_functions = ["simple", "sharpe", "transaction_penalty", "drawdown"]
    results = {}

    # Initialiser le risk manager
    risk_manager = RiskManager()

    # Initialiser le normaliseur adaptatif
    normalizer = AdaptiveNormalizer()

    for reward_function in reward_functions:
        print(f"\nTest de la fonction de récompense: {reward_function}")

        portfolio_values = []
        rewards = []

        for episode in range(episodes):
            print(f"Épisode {episode+1}/{episodes}")

            # Créer l'environnement avec la fonction de récompense spécifiée
            env = TradingEnvironment(
                df=df,
                initial_balance=initial_balance,
                transaction_fee=0.001,
                window_size=window_size,
                include_position=True,
                include_balance=True,
                include_technical_indicators=True,
                risk_management=True,
                normalize_observation=True,
                reward_function=reward_function,
                risk_aversion=0.1,
                transaction_penalty=0.001,
                lookback_window=20,
                action_type="discrete",
                n_discrete_actions=5,
            )

            # Réinitialiser l'environnement
            obs, _ = env.reset()
            done = False
            episode_rewards = []

            # Simuler un agent aléatoire pour les tests
            step_count = 0
            with tqdm(total=len(df) - window_size, desc=f"Simulation") as pbar:
                while not done:
                    # Choisir une action aléatoire
                    action = env.action_space.sample()

                    # Exécuter l'action
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    episode_rewards.append(reward)
                    step_count += 1
                    pbar.update(1)

                    if step_count >= len(df) - window_size - 1:
                        break

            # Récupérer les valeurs du portefeuille pour cet épisode
            episode_portfolio_values = env.get_portfolio_value_history()
            portfolio_values.append(episode_portfolio_values)
            rewards.append(episode_rewards)

        # Calculer les statistiques pour cette fonction de récompense
        all_final_values = [values[-1] for values in portfolio_values]
        avg_final_value = np.mean(all_final_values)
        max_final_value = np.max(all_final_values)
        min_final_value = np.min(all_final_values)
        std_final_value = np.std(all_final_values)

        all_rewards = [r for episode_rewards in rewards for r in episode_rewards]
        avg_reward = np.mean(all_rewards)
        cumulative_reward = np.sum(all_rewards)

        results[reward_function] = {
            "avg_final_value": avg_final_value,
            "max_final_value": max_final_value,
            "min_final_value": min_final_value,
            "std_final_value": std_final_value,
            "avg_reward": avg_reward,
            "cumulative_reward": cumulative_reward,
            "portfolio_values": portfolio_values,
            "rewards": rewards,
        }

        print(f"Valeur finale moyenne du portefeuille: {avg_final_value:.2f}")
        print(f"Récompense moyenne: {avg_reward:.4f}")
        print(f"Récompense cumulative: {cumulative_reward:.4f}")

    return results


def plot_results(results):
    """
    Trace les graphiques comparatifs des différentes fonctions de récompense.

    Args:
        results (dict): Dictionnaire contenant les résultats
    """
    reward_functions = list(results.keys())

    # Créer le dossier de destination s'il n'existe pas
    visualization_dir = VISUALIZATION_DIR / "reward_functions"
    os.makedirs(visualization_dir, exist_ok=True)

    # 1. Comparaison des valeurs finales du portefeuille
    plt.figure(figsize=(12, 6))
    final_values = [results[rf]["avg_final_value"] for rf in reward_functions]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.bar(reward_functions, final_values, color=colors)
    plt.title("Valeur finale moyenne du portefeuille par fonction de récompense")
    plt.xlabel("Fonction de récompense")
    plt.ylabel("Valeur du portefeuille")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(final_values):
        plt.text(i, v + 100, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(
        os.path.join(visualization_dir, "reward_functions_portfolio_comparison.png")
    )

    # 2. Évolution de la valeur du portefeuille au fil du temps (moyenne sur tous les épisodes)
    plt.figure(figsize=(12, 6))

    for i, rf in enumerate(reward_functions):
        # Calculer la valeur moyenne du portefeuille à chaque étape
        # Trouver la longueur minimale parmi tous les épisodes
        min_length = min(len(values) for values in results[rf]["portfolio_values"])

        # Tronquer toutes les séries à cette longueur minimale
        truncated_values = [
            values[:min_length] for values in results[rf]["portfolio_values"]
        ]

        # Calculer la moyenne pour chaque étape
        avg_values = np.mean(truncated_values, axis=0)

        # Tracer la courbe
        plt.plot(avg_values, label=rf, color=colors[i], linewidth=2)

    plt.title("Évolution moyenne de la valeur du portefeuille")
    plt.xlabel("Pas de temps")
    plt.ylabel("Valeur du portefeuille")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(visualization_dir, "reward_functions_portfolio_evolution.png")
    )

    # 3. Distribution des récompenses
    plt.figure(figsize=(12, 6))

    for i, rf in enumerate(reward_functions):
        all_rewards = [
            r for episode_rewards in results[rf]["rewards"] for r in episode_rewards
        ]

        plt.subplot(2, 2, i + 1)
        plt.hist(all_rewards, bins=30, color=colors[i], alpha=0.7)
        plt.title(f"Distribution des récompenses - {rf}")
        plt.xlabel("Récompense")
        plt.ylabel("Fréquence")
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(visualization_dir, "reward_functions_rewards_distribution.png")
    )

    # 4. Comparaison des récompenses cumulatives
    plt.figure(figsize=(12, 6))
    cumulative_rewards = [results[rf]["cumulative_reward"] for rf in reward_functions]

    plt.bar(reward_functions, cumulative_rewards, color=colors)
    plt.title("Récompense cumulative par fonction de récompense")
    plt.xlabel("Fonction de récompense")
    plt.ylabel("Récompense cumulative")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(cumulative_rewards):
        plt.text(
            i,
            v
            + (
                0.1 * max(cumulative_rewards)
                if v > 0
                else -0.1 * max(cumulative_rewards)
            ),
            f"{v:.2f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(visualization_dir, "reward_functions_cumulative_rewards.png")
    )

    print(f"Graphiques sauvegardés dans le répertoire {visualization_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Test des fonctions de récompense pour l'environnement de trading"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Nombre d'épisodes à exécuter"
    )
    parser.add_argument(
        "--days", type=int, default=60, help="Nombre de jours de données à utiliser"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC", help="Symbole de la cryptomonnaie"
    )
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        help="Utiliser des données réelles au lieu de données synthétiques",
    )

    args = parser.parse_args()

    print(f"Test des fonctions de récompense pour {args.symbol}")
    print(f"Paramètres: {args.episodes} épisodes, {args.days} jours")

    try:
        if args.use_real_data:
            print("Récupération des données réelles...")
            # Initialiser le collecteur de données
            data_collector = EnhancedDataCollector()

            # Récupérer les données pour le symbole spécifié
            df = data_collector.get_merged_price_data(
                symbol=args.symbol,
                days=args.days,
                include_fear_greed=True,
                use_mock_data_if_needed=True,
            )

            if df is None or len(df) == 0:
                print(
                    "Aucune donnée réelle disponible. Utilisation de données synthétiques à la place."
                )
                df = generate_synthetic_data(args.symbol, args.days)
            else:
                print(f"{len(df)} points de données récupérés avec succès.")
        else:
            print("Génération de données synthétiques...")
            df = generate_synthetic_data(args.symbol, args.days)
            print(f"{len(df)} points de données synthétiques générés.")

        # Prétraiter les données
        print("Prétraitement des données...")
        df_processed = preprocess_data(df)
        print(f"Données prétraitées: {len(df_processed)} points après nettoyage.")

        # Comparer les fonctions de récompense
        print("\nComparaison des fonctions de récompense...")
        results = compare_reward_functions(
            df=df_processed,
            episodes=args.episodes,
            initial_balance=10000.0,
            window_size=20,
        )

        # Tracer les résultats
        print("\nTracé des graphiques comparatifs...")
        plot_results(results)

        # Identifier la meilleure fonction de récompense basée sur les résultats
        best_reward_function = max(
            results.keys(), key=lambda rf: results[rf]["avg_final_value"]
        )
        best_avg_value = results[best_reward_function]["avg_final_value"]

        print("\n====== Résultats finaux ======")
        print(f"Meilleure fonction de récompense: {best_reward_function}")
        print(f"Valeur finale moyenne du portefeuille: {best_avg_value:.2f}")
        print(
            f"Récompense cumulative: {results[best_reward_function]['cumulative_reward']:.4f}"
        )
        print("==============================")

    except Exception as e:
        print(f"Erreur lors de l'exécution du test: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
