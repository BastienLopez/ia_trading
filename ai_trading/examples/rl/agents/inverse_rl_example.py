"""
Exemple d'utilisation de l'apprentissage inverse par renforcement (Inverse RL).

Ce script démontre comment:
1. Charger des trajectoires d'experts (traders professionnels)
2. Utiliser MaximumEntropyIRL pour extraire une fonction de récompense
3. Utiliser ApprenticeshipLearning pour imiter directement la politique de l'expert
4. Évaluer et comparer les performances

L'apprentissage inverse par renforcement est utile pour:
- Découvrir les objectifs implicites des traders experts
- Créer des agents d'IA qui imitent les stratégies humaines réussies
- Combiner l'expertise humaine avec l'optimisation par RL
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inverse_rl_example")

# Ajout du répertoire parent au path pour les imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.inverse_rl import ApprenticeshipLearning, MaximumEntropyIRL
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.utils.data_collector import DataCollector


def load_sample_data():
    """
    Charge des données d'exemple pour l'entraînement et l'évaluation.

    Returns:
        DataFrame contenant les données de marché
    """
    # Tenter de charger depuis des fichiers existants d'abord
    try:
        data_path = os.path.join(parent_dir, "data", "sample_btc_data.csv")
        if os.path.exists(data_path):
            logger.info(f"Chargement des données depuis {data_path}")
            df = pd.read_csv(data_path)
            return df
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des données existantes: {e}")

    # Si pas de données existantes, collecter un petit ensemble de données
    logger.info("Collecte d'un nouvel ensemble de données d'exemple")
    collector = DataCollector()
    df = collector.fetch_historical_data(
        "BTC", start_date="2022-01-01", end_date="2022-03-01", interval="1d"
    )

    # Sauvegarder pour réutilisation future
    os.makedirs(os.path.join(parent_dir, "data"), exist_ok=True)
    df.to_csv(os.path.join(parent_dir, "data", "sample_btc_data.csv"), index=False)

    return df


def simulate_expert_demonstrations(env, num_demos=10, demo_length=100):
    """
    Simule des démonstrations d'experts pour l'apprentissage par IRL.

    Dans un cas réel, ces données proviendraient de traders humains experts,
    mais pour cet exemple, nous simulons un expert avec une stratégie heuristique.

    Args:
        env: Environnement de trading
        num_demos: Nombre de démonstrations à générer
        demo_length: Longueur maximale de chaque démonstration

    Returns:
        Liste de trajectoires, chaque trajectoire étant une liste de tuples (état, action)
    """
    demos = []

    for _ in range(num_demos):
        state = env.reset()
        trajectory = []

        for step in range(demo_length):
            # Stratégie d'expert simulée (basée sur des heuristiques simples)
            # Dans un cas réel, ces actions proviendraient de données historiques de trading réelles

            # Extraire quelques indicateurs de l'état pour notre heuristique
            price_change = (
                env.normalized_prices[-5:].diff().iloc[-1]
                if len(env.normalized_prices) > 5
                else 0
            )
            rsi = (
                env.indicators["rsi"].iloc[-1]
                if hasattr(env, "indicators") and "rsi" in env.indicators
                else 50
            )

            # Stratégie simulée: acheter quand prix en hausse et RSI < 70, vendre quand prix en baisse et RSI > 30
            if price_change > 0 and rsi < 70:
                action = 1  # Acheter
            elif price_change < 0 and rsi > 30:
                action = 2  # Vendre
            else:
                action = 0  # Attendre

            next_state, reward, done, _ = env.step(action)

            # Enregistrer l'état et l'action
            trajectory.append((state, action))

            if done:
                break

            state = next_state

        demos.append(trajectory)

    logger.info(f"Généré {len(demos)} démonstrations simulées d'experts")
    return demos


def optimize_policy(env, reward_function, num_steps=1000):
    """
    Optimise une politique en utilisant une fonction de récompense donnée.

    Cette fonction est utilisée dans les algorithmes d'IRL pour trouver
    une politique optimale étant donné une fonction de récompense.

    Args:
        env: Environnement de trading
        reward_function: Fonction de récompense à utiliser
        num_steps: Nombre d'étapes d'optimisation

    Returns:
        La politique optimisée
    """

    # Créer un wrapper d'environnement qui utilise la fonction de récompense donnée
    class RewardWrapper:
        def __init__(self, env, reward_function):
            self.env = env
            self.reward_function = reward_function

        def reset(self):
            return self.env.reset()

        def step(self, action):
            next_state, _, done, info = self.env.step(action)
            reward = self.reward_function(next_state, torch.tensor([action]))
            return next_state, reward, done, info

    wrapped_env = RewardWrapper(env, reward_function)

    # Créer un agent DQN pour optimiser la politique
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update=10,
    )

    # Entraîner l'agent
    for step in range(num_steps):
        state = wrapped_env.reset()
        done = False
        total_reward = 0

        while not done:
            # Sélectionner une action
            action = agent.select_action(state)

            # Exécuter l'action
            next_state, reward, done, _ = wrapped_env.step(action)

            # Stocker l'expérience
            agent.store_experience(state, action, reward, next_state, done)

            # Apprendre
            agent.learn()

            total_reward += reward
            state = next_state

        if step % 100 == 0:
            logger.info(f"Étape {step}, Récompense totale: {total_reward}")

    # Retourner la politique optimisée
    def policy(state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return agent.q_network(state_tensor)

    return policy


def evaluate_policy(env, policy, num_episodes=10):
    """
    Évalue une politique sur un environnement donné.

    Args:
        env: Environnement de trading
        policy: Politique à évaluer
        num_episodes: Nombre d'épisodes d'évaluation

    Returns:
        Performance moyenne (rendement total)
    """
    total_returns = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0

        while not done:
            # Sélectionner une action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy(state_tensor).argmax().item()

            # Exécuter l'action
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            state = next_state

        total_returns.append(episode_return)

    avg_return = sum(total_returns) / len(total_returns)
    logger.info(f"Performance moyenne: {avg_return:.4f}")
    return avg_return


def main():
    """
    Fonction principale démontrant l'utilisation de l'apprentissage inverse par renforcement.
    """
    # 1. Charger les données
    logger.info("Chargement des données...")
    data = load_sample_data()

    # 2. Créer un environnement de trading
    logger.info("Création de l'environnement de trading...")
    env = TradingEnvironment(
        data=data,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        reward_scaling=0.01,
    )

    # 3. Simuler des démonstrations d'experts
    logger.info("Génération de démonstrations d'experts simulées...")
    expert_demos = simulate_expert_demonstrations(env, num_demos=10, demo_length=100)

    # 4. Initialiser l'algorithme MaxEnt IRL
    logger.info("Initialisation de l'algorithme MaxEnt IRL...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    max_ent_irl = MaximumEntropyIRL(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        learning_rate=0.001,
        regularization=0.01,
    )

    # 5. Charger les démonstrations d'experts
    logger.info("Chargement des démonstrations d'experts dans MaxEnt IRL...")
    max_ent_irl.load_expert_demonstrations(expert_demos)

    # 6. Entraîner l'algorithme MaxEnt IRL
    logger.info("Entraînement de l'algorithme MaxEnt IRL...")
    history = max_ent_irl.train(
        policy_optimizer=optimize_policy,
        env=env,
        epochs=10,
        policy_update_steps=500,
        batch_size=32,
    )

    # 7. Optimiser une politique finale avec la fonction de récompense apprise
    logger.info(
        "Optimisation de la politique finale avec la fonction de récompense apprise..."
    )
    final_policy = optimize_policy(env, max_ent_irl.predict_reward, num_steps=1000)

    # 8. Initialiser l'algorithme Apprenticeship Learning
    logger.info("Initialisation de l'algorithme Apprenticeship Learning...")
    al = ApprenticeshipLearning(state_dim=state_dim, action_dim=action_dim, gamma=0.99)

    # 9. Charger les démonstrations d'experts
    logger.info(
        "Chargement des démonstrations d'experts dans Apprenticeship Learning..."
    )
    al.load_expert_demonstrations(expert_demos)

    # 10. Entraîner l'algorithme Apprenticeship Learning
    logger.info("Entraînement de l'algorithme Apprenticeship Learning...")
    al_history = al.train(
        policy_optimizer=optimize_policy,
        env=env,
        max_iterations=10,
        policy_update_steps=500,
    )

    # 11. Obtenir la politique finale de l'algorithme Apprenticeship Learning
    logger.info("Optimisation de la politique finale avec Apprenticeship Learning...")
    al_policy = optimize_policy(env, al.compute_reward, num_steps=1000)

    # 12. Évaluer les différentes politiques
    logger.info("Évaluation des différentes politiques...")

    # Politique aléatoire (baseline)
    logger.info("Évaluation de la politique aléatoire...")

    def random_policy(state):
        return torch.randn(1, action_dim)

    random_performance = evaluate_policy(env, random_policy)

    # Politique MaxEnt IRL
    logger.info("Évaluation de la politique MaxEnt IRL...")
    maxent_performance = evaluate_policy(env, final_policy)

    # Politique Apprenticeship Learning
    logger.info("Évaluation de la politique Apprenticeship Learning...")
    al_performance = evaluate_policy(env, al_policy)

    # 13. Afficher les résultats
    logger.info("\nRésultats d'évaluation:")
    logger.info(f"  Politique aléatoire: {random_performance:.4f}")
    logger.info(f"  Politique MaxEnt IRL: {maxent_performance:.4f}")
    logger.info(f"  Politique Apprenticeship Learning: {al_performance:.4f}")

    # 14. Visualiser l'historique d'entraînement
    plt.figure(figsize=(15, 5))

    # MaxEnt IRL
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Perte")
    plt.plot(history["expert_reward"], label="Récompense expert")
    plt.plot(history["policy_reward"], label="Récompense politique")
    plt.title("Apprentissage MaxEnt IRL")
    plt.xlabel("Mise à jour")
    plt.ylabel("Valeur")
    plt.legend()

    # Apprenticeship Learning
    plt.subplot(1, 2, 2)
    plt.plot(al_history["feature_difference"], label="Différence de caractéristiques")
    plt.title("Apprentissage Apprenticeship Learning")
    plt.xlabel("Itération")
    plt.ylabel("Différence")
    plt.legend()

    plt.tight_layout()

    # Sauvegarder la figure
    plot_dir = os.path.join(parent_dir, "results")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "inverse_rl_results.png"))

    logger.info(
        f"Figure sauvegardée dans {os.path.join(plot_dir, 'inverse_rl_results.png')}"
    )

    # 15. Sauvegarder les modèles
    models_dir = os.path.join(parent_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    max_ent_irl.save(os.path.join(models_dir, "max_ent_irl_model.pt"))
    al.save(os.path.join(models_dir, "apprenticeship_learning_model.pkl"))

    logger.info(f"Modèles sauvegardés dans {models_dir}")


if __name__ == "__main__":
    main()
