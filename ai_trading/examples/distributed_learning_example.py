"""
Exemple d'apprentissage distribué pour le trading algorithmique.

Ce script démontre comment:
1. Distribuer la collecte d'expériences sur plusieurs machines/processeurs
2. Entraîner un modèle central en utilisant les expériences collectées
3. Synchroniser les politiques entre les travailleurs

L'apprentissage distribué permet:
- Une collecte de données plus efficace
- Un entraînement accéléré
- Une meilleure exploration de l'environnement
"""

import logging
import multiprocessing as mp
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("distributed_learning_example")

# Ajout du répertoire parent au path pour les imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ai_trading.rl.distributed_experience import (
    DistributedExperienceManager,
    ExperienceMaster,
    ExperienceWorker,
)
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.replay_buffer import ReplayBuffer
from ai_trading.rl.technical_indicators import add_indicators
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.utils.data_collector import DataCollector


def load_data():
    """
    Charge les données de marché pour l'exemple.

    Returns:
        DataFrame contenant les données de marché
    """
    # Tenter de charger depuis des fichiers existants d'abord
    try:
        data_path = os.path.join(parent_dir, "data", "sample_data_distributed.csv")
        if os.path.exists(data_path):
            logger.info(f"Chargement des données depuis {data_path}")
            df = pd.read_csv(data_path)
            return df
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des données existantes: {e}")

    # Si pas de données existantes, collecter un petit ensemble de données
    logger.info("Collecte d'un nouvel ensemble de données")
    collector = DataCollector()
    df = collector.fetch_historical_data(
        "BTC",
        start_date="2022-01-01",
        end_date="2022-03-01",
        interval="1h",  # Données horaires pour avoir plus d'échantillons
    )

    # Ajouter des indicateurs techniques
    df = add_indicators(df)

    # Sauvegarder pour réutilisation future
    os.makedirs(os.path.join(parent_dir, "data"), exist_ok=True)
    df.to_csv(data_path, index=False)

    return df


def create_environment(data):
    """
    Crée un environnement de trading.

    Args:
        data: DataFrame contenant les données de marché

    Returns:
        Environnement de trading
    """
    return TradingEnvironment(
        data=data,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        reward_scaling=0.01,
    )


def create_policy(state_dim, action_dim):
    """
    Crée une politique/modèle pour l'agent.

    Args:
        state_dim: Dimension de l'espace d'état
        action_dim: Dimension de l'espace d'action

    Returns:
        Modèle de politique
    """
    # Utiliser un modèle DQN simple
    model = torch.nn.Sequential(
        torch.nn.Linear(state_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, action_dim),
    )

    return model


def train_with_local_workers(data, num_workers=4, collect_steps=10000, batch_size=32):
    """
    Entraîne un modèle en utilisant des travailleurs locaux pour collecter des expériences.

    Args:
        data: DataFrame contenant les données de marché
        num_workers: Nombre de travailleurs
        collect_steps: Nombre d'étapes à collecter par travailleur
        batch_size: Taille du lot pour l'apprentissage

    Returns:
        Tuple (agent, metrics): L'agent entraîné et les métriques de performance
    """
    logger.info(f"Entraînement avec {num_workers} travailleurs locaux")

    # Créer un environnement pour déterminer les dimensions
    env = create_environment(data)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Créer un replay buffer partagé
    buffer_size = (
        num_workers * collect_steps
    )  # Assez grand pour stocker toutes les expériences
    replay_buffer = ReplayBuffer(buffer_size, state_dim)

    # Créer un agent DQN central
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        memory_size=buffer_size,
        batch_size=batch_size,
        target_update=10,
    )

    # Fonction pour créer un environnement
    def env_creator():
        return create_environment(data)

    # Créer un policy provider qui retourne la politique actuelle
    def policy_provider():
        return agent.q_network

    # Initialiser le gestionnaire d'expérience distribuée
    manager = DistributedExperienceManager(
        env_creator=env_creator,
        policy=agent.q_network,
        replay_buffer=agent.replay_memory,
        n_local_workers=num_workers,
        batch_size=batch_size,
        policy_provider=policy_provider,
    )

    # Initialiser le système distribué
    manager.initialize()

    # Démarrer la collecte d'expériences
    logger.info("Démarrage de la collecte d'expériences distribuée")
    manager.start_collection(steps_per_worker=collect_steps // num_workers)

    # Pendant que les travailleurs collectent des expériences, entraîner l'agent
    start_time = time.time()
    training_steps = 0
    rewards = []

    # Attendre que le buffer commence à se remplir
    time.sleep(2)

    # Boucle d'entraînement
    while training_steps < collect_steps:
        # Vérifier si tous les travailleurs ont terminé
        if manager.all_workers_done():
            break

        # Entraîner sur plusieurs mini-lots
        for _ in range(10):
            loss = agent.learn()
            training_steps += 1

            if training_steps % 100 == 0:
                # Mettre à jour le réseau cible périodiquement
                agent.update_target_network()

                # Évaluer l'agent actuel
                eval_reward = evaluate_agent(agent, env)
                rewards.append(eval_reward)

                # Afficher les métriques
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Étape {training_steps}, Récompense: {eval_reward:.2f}, Temps écoulé: {elapsed_time:.2f}s"
                )

        # Petite pause pour éviter de surcharger le CPU
        time.sleep(0.1)

    # Attendre que tous les travailleurs terminent
    logger.info("Attente de la fin de tous les travailleurs...")
    manager.wait_until_done()

    # Arrêter le système distribué
    manager.stop()

    # Collecter les métriques finales
    metrics = manager.get_metrics()

    return agent, rewards, metrics


def simulate_distributed_training(
    data, num_workers=4, collect_steps=10000, batch_size=32
):
    """
    Simule un entraînement distribué sur plusieurs nœuds.

    Dans un système réel, les travailleurs seraient sur différentes machines.
    Ici, nous simulons le comportement avec des processus locaux.

    Args:
        data: DataFrame contenant les données de marché
        num_workers: Nombre de travailleurs
        collect_steps: Nombre d'étapes à collecter par travailleur
        batch_size: Taille du lot pour l'apprentissage

    Returns:
        Tuple (agent, metrics): L'agent entraîné et les métriques de performance
    """
    logger.info(f"Simulation d'entraînement distribué avec {num_workers} nœuds")

    # Créer un environnement pour déterminer les dimensions
    env = create_environment(data)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Créer un replay buffer partagé
    buffer_size = num_workers * collect_steps
    replay_buffer = ReplayBuffer(buffer_size, state_dim)

    # Créer un agent DQN central
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        memory_size=buffer_size,
        batch_size=batch_size,
        target_update=10,
    )

    # Créer le maître d'expérience
    master = ExperienceMaster(
        replay_buffer=agent.replay_memory,
        max_workers=num_workers,
        update_freq=1000,
        local_mode=True,  # Mode local pour la simulation
    )

    # Fonction pour créer un environnement
    def env_creator():
        return create_environment(data)

    # Créer les travailleurs
    workers = []
    for i in range(num_workers):
        worker = ExperienceWorker(
            env_creator=env_creator,
            policy=agent.q_network,
            worker_id=f"worker-{i}",
            batch_size=batch_size,
            send_freq=5,
            local_mode=True,  # Mode local pour la simulation
        )

        # Ajouter au maître
        master.add_local_worker(worker)
        workers.append(worker)

    # Démarrer le maître
    master.start()

    # Démarrer tous les travailleurs
    logger.info("Démarrage des travailleurs d'expérience")
    for worker in workers:
        worker.run_in_process(collect_steps // num_workers)

    # Pendant que les travailleurs collectent des expériences, entraîner l'agent
    start_time = time.time()
    training_steps = 0
    rewards = []

    # Attendre que le buffer commence à se remplir
    time.sleep(2)

    # Boucle d'entraînement
    while training_steps < collect_steps and not all(
        worker.worker_process is None or not worker.worker_process.is_alive()
        for worker in workers
    ):
        # Entraîner sur plusieurs mini-lots
        for _ in range(10):
            loss = agent.learn()
            training_steps += 1

            if training_steps % 100 == 0:
                # Mettre à jour le réseau cible périodiquement
                agent.update_target_network()

                # Évaluer l'agent actuel
                eval_reward = evaluate_agent(agent, env)
                rewards.append(eval_reward)

                # Afficher les métriques
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Étape {training_steps}, Récompense: {eval_reward:.2f}, Temps écoulé: {elapsed_time:.2f}s"
                )

        # Petite pause pour éviter de surcharger le CPU
        time.sleep(0.1)

    # Arrêter tous les travailleurs
    logger.info("Arrêt des travailleurs...")
    for worker in workers:
        worker.stop()

    # Arrêter le maître
    master.stop()

    # Collecter les métriques finales
    metrics = master.get_metrics()

    return agent, rewards, metrics


def evaluate_agent(agent, env, num_episodes=5):
    """
    Évalue un agent sur un environnement donné.

    Args:
        agent: Agent à évaluer
        env: Environnement d'évaluation
        num_episodes: Nombre d'épisodes d'évaluation

    Returns:
        Récompense moyenne
    """
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Sélectionner la meilleure action (sans exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.q_network(state_tensor)
                action = q_values.argmax().item()

            # Exécuter l'action
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def evaluate_standard_training(data, total_steps=10000, batch_size=32):
    """
    Évalue l'entraînement standard (non distribué) pour comparaison.

    Args:
        data: DataFrame contenant les données de marché
        total_steps: Nombre total d'étapes d'entraînement
        batch_size: Taille du lot pour l'apprentissage

    Returns:
        Tuple (agent, rewards): L'agent entraîné et la liste des récompenses
    """
    logger.info("Évaluation de l'entraînement standard (non distribué)")

    # Créer un environnement
    env = create_environment(data)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Créer un agent DQN
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        memory_size=total_steps,
        batch_size=batch_size,
        target_update=10,
    )

    # Entraîner l'agent de manière standard
    start_time = time.time()
    step = 0
    rewards = []

    # Collecter des expériences et entraîner
    while step < total_steps:
        state = env.reset()
        done = False
        episode_reward = 0

        while not done and step < total_steps:
            # Sélectionner une action
            action = agent.select_action(state)

            # Exécuter l'action
            next_state, reward, done, _ = env.step(action)

            # Stocker l'expérience
            agent.store_experience(state, action, reward, next_state, done)

            # Apprendre
            agent.learn()

            episode_reward += reward
            state = next_state
            step += 1

            if step % 100 == 0:
                # Mettre à jour le réseau cible périodiquement
                agent.update_target_network()

                # Évaluer l'agent actuel
                eval_reward = evaluate_agent(agent, env)
                rewards.append(eval_reward)

                # Afficher les métriques
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Étape {step}, Récompense: {eval_reward:.2f}, Temps écoulé: {elapsed_time:.2f}s"
                )

    return agent, rewards


def compare_methods(standard_rewards, local_rewards, distributed_rewards):
    """
    Compare les performances des différentes méthodes d'entraînement.

    Args:
        standard_rewards: Liste des récompenses pour l'entraînement standard
        local_rewards: Liste des récompenses pour l'entraînement avec travailleurs locaux
        distributed_rewards: Liste des récompenses pour l'entraînement distribué simulé
    """
    # Créer un graphique de comparaison
    plt.figure(figsize=(12, 6))

    # Tracer les récompenses pour chaque méthode
    x_standard = np.arange(len(standard_rewards)) * 100
    x_local = np.arange(len(local_rewards)) * 100
    x_distributed = np.arange(len(distributed_rewards)) * 100

    plt.plot(x_standard, standard_rewards, label="Standard", marker="o", markersize=3)
    plt.plot(
        x_local, local_rewards, label="Travailleurs locaux", marker="s", markersize=3
    )
    plt.plot(
        x_distributed,
        distributed_rewards,
        label="Distribué simulé",
        marker="^",
        markersize=3,
    )

    plt.xlabel("Étapes d'entraînement")
    plt.ylabel("Récompense moyenne")
    plt.title("Comparaison des méthodes d'entraînement")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Ajouter des lignes moyennes mobiles
    window_size = min(5, len(standard_rewards) // 2)
    if window_size > 0:
        standard_avg = np.convolve(
            standard_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        local_avg = np.convolve(
            local_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        distributed_avg = np.convolve(
            distributed_rewards, np.ones(window_size) / window_size, mode="valid"
        )

        x_standard_avg = x_standard[window_size - 1 :][: len(standard_avg)]
        x_local_avg = x_local[window_size - 1 :][: len(local_avg)]
        x_distributed_avg = x_distributed[window_size - 1 :][: len(distributed_avg)]

        plt.plot(x_standard_avg, standard_avg, "--", color="blue", alpha=0.5)
        plt.plot(x_local_avg, local_avg, "--", color="orange", alpha=0.5)
        plt.plot(x_distributed_avg, distributed_avg, "--", color="green", alpha=0.5)

    # Calculer et afficher les statistiques
    print("\nStatistiques de performance:")
    print(
        f"Standard: Moyenne={np.mean(standard_rewards):.2f}, Max={np.max(standard_rewards):.2f}"
    )
    print(
        f"Travailleurs locaux: Moyenne={np.mean(local_rewards):.2f}, Max={np.max(local_rewards):.2f}"
    )
    print(
        f"Distribué simulé: Moyenne={np.mean(distributed_rewards):.2f}, Max={np.max(distributed_rewards):.2f}"
    )

    # Sauvegarder la figure
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "distributed_learning_comparison.png"))

    logger.info(
        f"Figure sauvegardée dans {os.path.join(results_dir, 'distributed_learning_comparison.png')}"
    )
    plt.close()


def display_metrics(
    standard_time, local_time, distributed_time, local_metrics, distributed_metrics
):
    """
    Affiche et compare les métriques de performance des différentes méthodes.

    Args:
        standard_time: Temps total pour l'entraînement standard
        local_time: Temps total pour l'entraînement avec travailleurs locaux
        distributed_time: Temps total pour l'entraînement distribué simulé
        local_metrics: Métriques des travailleurs locaux
        distributed_metrics: Métriques de l'entraînement distribué
    """
    # Créer un graphique des temps d'exécution
    plt.figure(figsize=(8, 6))
    methods = ["Standard", "Travailleurs locaux", "Distribué simulé"]
    times = [standard_time, local_time, distributed_time]

    plt.bar(methods, times, color=["blue", "orange", "green"])
    plt.xlabel("Méthode")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.title("Comparaison des temps d'exécution")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Afficher les valeurs exactes
    for i, v in enumerate(times):
        plt.text(i, v + 0.5, f"{v:.1f}s", ha="center")

    # Sauvegarder la figure
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "distributed_learning_time_comparison.png"))

    logger.info(
        f"Figure sauvegardée dans {os.path.join(results_dir, 'distributed_learning_time_comparison.png')}"
    )
    plt.close()

    # Afficher des statistiques supplémentaires
    print("\nStatistiques d'efficacité:")
    print(f"Accélération (travailleurs locaux): {standard_time / local_time:.2f}x")
    print(f"Accélération (distribué simulé): {standard_time / distributed_time:.2f}x")

    print("\nMétriques des travailleurs locaux:")
    for key, value in local_metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")

    print("\nMétriques de l'entraînement distribué:")
    for key, value in distributed_metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")


def main():
    """
    Fonction principale démontrant l'apprentissage distribué.
    """
    # Définir le nombre de processus pour multiprocessing
    mp.set_start_method("spawn", force=True)

    # 1. Charger les données
    logger.info("Chargement des données...")
    data = load_data()

    # 2. Configuration des expériences
    total_steps = 5000  # Réduire pour l'exemple
    batch_size = 32
    num_workers = min(4, mp.cpu_count())  # Limiter au nombre de cœurs disponibles

    # 3. Évaluer l'entraînement standard
    logger.info("Démarrage de l'entraînement standard...")
    start_time = time.time()
    standard_agent, standard_rewards = evaluate_standard_training(
        data, total_steps, batch_size
    )
    standard_time = time.time() - start_time
    logger.info(f"Entraînement standard terminé en {standard_time:.2f} secondes")

    # 4. Évaluer l'entraînement avec des travailleurs locaux
    logger.info("Démarrage de l'entraînement avec travailleurs locaux...")
    start_time = time.time()
    local_agent, local_rewards, local_metrics = train_with_local_workers(
        data, num_workers, total_steps, batch_size
    )
    local_time = time.time() - start_time
    logger.info(
        f"Entraînement avec travailleurs locaux terminé en {local_time:.2f} secondes"
    )

    # 5. Simuler un entraînement distribué
    logger.info("Démarrage de la simulation d'entraînement distribué...")
    start_time = time.time()
    distributed_agent, distributed_rewards, distributed_metrics = (
        simulate_distributed_training(data, num_workers, total_steps, batch_size)
    )
    distributed_time = time.time() - start_time
    logger.info(
        f"Simulation d'entraînement distribué terminée en {distributed_time:.2f} secondes"
    )

    # 6. Comparer les performances des différentes méthodes
    logger.info("Comparaison des performances...")
    compare_methods(standard_rewards, local_rewards, distributed_rewards)

    # 7. Afficher les métriques de performance
    logger.info("Affichage des métriques...")
    display_metrics(
        standard_time, local_time, distributed_time, local_metrics, distributed_metrics
    )

    logger.info("Exemple d'apprentissage distribué terminé avec succès!")


if __name__ == "__main__":
    main()
