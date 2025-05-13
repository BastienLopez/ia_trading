"""
Exemple d'utilisation du transfert d'apprentissage entre différents marchés et actifs financiers.

Ce script démontre comment:
1. Entraîner un modèle de trading sur un marché source (BTC)
2. Transférer les connaissances vers un marché cible différent (ETH)
3. Comparer les performances avant et après le transfert d'apprentissage

Le transfert d'apprentissage est particulièrement utile quand:
- On dispose de peu de données sur le marché cible
- Les marchés partagent des caractéristiques communes
- On veut accélérer l'apprentissage sur de nouveaux marchés
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("market_transfer_example")

# Ajout du répertoire parent au path pour les imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.models.transfer_learning import (
    DomainAdaptation,
    MarketTransferLearning,
)
from ai_trading.rl.technical_indicators import add_indicators
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.utils.data_collector import DataCollector


def load_market_data(
    symbol: str, start_date: str, end_date: str, interval: str = "1d"
) -> pd.DataFrame:
    """
    Charge des données de marché pour un actif donné.

    Args:
        symbol: Symbole de l'actif (ex: 'BTC', 'ETH')
        start_date: Date de début au format 'YYYY-MM-DD'
        end_date: Date de fin au format 'YYYY-MM-DD'
        interval: Intervalle de temps ('1d', '1h', etc.)

    Returns:
        DataFrame contenant les données de marché
    """
    try:
        # Vérifier si les données existent déjà
        data_path = os.path.join(
            parent_dir, "data", f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        )
        if os.path.exists(data_path):
            logger.info(f"Chargement des données depuis {data_path}")
            df = pd.read_csv(data_path)
            return df
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des données existantes: {e}")

    # Si pas de données existantes, collecter un ensemble de données
    logger.info(f"Collecte de données pour {symbol} de {start_date} à {end_date}")
    collector = DataCollector()
    df = collector.fetch_historical_data(
        symbol, start_date=start_date, end_date=end_date, interval=interval
    )

    # Ajouter des indicateurs techniques
    df = add_indicators(df)

    # Sauvegarder pour réutilisation future
    os.makedirs(os.path.join(parent_dir, "data"), exist_ok=True)
    df.to_csv(data_path, index=False)

    return df


def prepare_datasets() -> (
    Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, TradingEnvironment]]]
):
    """
    Prépare les ensembles de données et les environnements pour les marchés source et cible.

    Returns:
        Tuple contenant:
        - Dictionnaire des DataFrames (train/val/test pour chaque marché)
        - Dictionnaire des environnements (train/val/test pour chaque marché)
    """
    # Paramètres communs
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    val_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    test_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Collecter les données pour BTC (marché source)
    btc_data = load_market_data("BTC", start_date, end_date)

    # Collecter les données pour ETH (marché cible)
    eth_data = load_market_data("ETH", start_date, end_date)

    # Diviser les données en train/val/test
    data_dict = {
        "btc": {
            "train": btc_data.loc[
                (btc_data["timestamp"] >= start_date)
                & (btc_data["timestamp"] < val_date)
            ],
            "val": btc_data.loc[
                (btc_data["timestamp"] >= val_date)
                & (btc_data["timestamp"] < test_date)
            ],
            "test": btc_data.loc[btc_data["timestamp"] >= test_date],
        },
        "eth": {
            "train": eth_data.loc[
                (eth_data["timestamp"] >= start_date)
                & (eth_data["timestamp"] < val_date)
            ],
            "val": eth_data.loc[
                (eth_data["timestamp"] >= val_date)
                & (eth_data["timestamp"] < test_date)
            ],
            "test": eth_data.loc[eth_data["timestamp"] >= test_date],
        },
    }

    # Créer les environnements de trading
    envs = {}
    for market in data_dict:
        envs[market] = {}
        for split in data_dict[market]:
            envs[market][split] = TradingEnvironment(
                data=data_dict[market][split],
                initial_balance=10000,
                transaction_fee=0.001,
                window_size=20,
                reward_scaling=0.01,
            )

    return data_dict, envs


def create_data_loaders(
    envs: Dict[str, Dict[str, TradingEnvironment]],
) -> Dict[str, Dict[str, torch.utils.data.DataLoader]]:
    """
    Crée des DataLoaders pour l'entraînement et l'évaluation.

    Args:
        envs: Dictionnaire d'environnements

    Returns:
        Dictionnaire de DataLoaders
    """
    loaders = {}

    for market in envs:
        loaders[market] = {}
        for split in envs[market]:
            # Collecter des exemples d'états
            states = []
            for _ in range(
                5000 if split == "train" else 1000
            ):  # Plus d'exemples pour l'entraînement
                state = envs[market][split].reset()
                states.append(state)
                for _ in range(
                    np.random.randint(1, 5)
                ):  # Faire quelques étapes aléatoires
                    action = np.random.randint(0, envs[market][split].action_space.n)
                    next_state, _, done, _ = envs[market][split].step(action)
                    if done:
                        break
                    states.append(next_state)

            # Convertir en tenseurs
            states = torch.FloatTensor(np.array(states))

            # Créer un DataLoader
            dataset = torch.utils.data.TensorDataset(
                states, states
            )  # Entrée = Sortie pour l'auto-encodeur
            loaders[market][split] = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=(split == "train")
            )

    return loaders


def train_source_agent(envs: Dict[str, Dict[str, TradingEnvironment]]) -> DQNAgent:
    """
    Entraîne un agent DQN sur le marché source (BTC).

    Args:
        envs: Dictionnaire d'environnements

    Returns:
        Agent DQN entraîné
    """
    logger.info("Entraînement de l'agent sur le marché source (BTC)...")

    # Environnement source
    env = envs["btc"]["train"]

    # Créer l'agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
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
    num_episodes = 100
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
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

        rewards.append(episode_reward)

        if episode % 10 == 0:
            logger.info(
                f"Épisode {episode+1}/{num_episodes}, Récompense: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
            )

    logger.info(
        f"Entraînement terminé. Récompense moyenne: {np.mean(rewards[-10:]):.2f}"
    )

    # Évaluer l'agent sur l'ensemble de validation
    val_rewards = evaluate_agent(agent, envs["btc"]["val"])
    logger.info(f"Performance sur BTC validation: {np.mean(val_rewards):.2f}")

    return agent


def apply_transfer_learning(
    source_agent: DQNAgent,
    envs: Dict[str, Dict[str, TradingEnvironment]],
    data_loaders: Dict[str, Dict[str, torch.utils.data.DataLoader]],
) -> DQNAgent:
    """
    Applique le transfert d'apprentissage du marché source au marché cible.

    Args:
        source_agent: Agent entraîné sur le marché source
        envs: Dictionnaire d'environnements
        data_loaders: Dictionnaire de DataLoaders

    Returns:
        Agent adapté au marché cible
    """
    logger.info("Application du transfert d'apprentissage de BTC à ETH...")

    # Créer un nouvel agent pour le marché cible avec les mêmes paramètres
    target_env = envs["eth"]["train"]
    target_agent = DQNAgent(
        state_dim=target_env.observation_space.shape[0],
        action_dim=target_env.action_space.n,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=0.5,  # Commencer avec une exploration réduite
        epsilon_end=0.1,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update=10,
    )

    # Copier les poids de l'agent source à l'agent cible
    target_agent.q_network.load_state_dict(source_agent.q_network.state_dict())
    target_agent.target_network.load_state_dict(
        source_agent.target_network.state_dict()
    )

    # Initialiser le système de transfert d'apprentissage
    # Fine-tuner uniquement les dernières couches pour l'adaptation
    transfer_learning = MarketTransferLearning(
        base_model=target_agent.q_network,
        fine_tune_layers=["fc2", "output"],  # Adapter uniquement les dernières couches
        learning_rate=0.0001,
        feature_mapping=True,  # Ajouter une couche d'adaptation
    )

    # Fine-tuner sur les données du marché cible
    logger.info("Fine-tuning du modèle sur le marché cible (ETH)...")
    history = transfer_learning.fine_tune(
        train_loader=data_loaders["eth"]["train"],
        val_loader=data_loaders["eth"]["val"],
        epochs=10,
        criterion=torch.nn.MSELoss(),
        early_stopping_patience=3,
    )

    # Mettre à jour le modèle de l'agent avec le modèle fine-tuné
    target_agent.q_network = transfer_learning.base_model

    # Mettre à jour le réseau cible avec les poids du nouveau réseau Q
    target_agent.target_network.load_state_dict(target_agent.q_network.state_dict())

    # Entraîner l'agent sur quelques épisodes pour adapter la politique
    logger.info("Ajustement final de la politique...")
    num_episodes = 50
    rewards = []

    for episode in range(num_episodes):
        state = target_env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Sélectionner une action
            action = target_agent.select_action(state)

            # Exécuter l'action
            next_state, reward, done, _ = target_env.step(action)

            # Stocker l'expérience
            target_agent.store_experience(state, action, reward, next_state, done)

            # Apprendre
            target_agent.learn()

            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)

        if episode % 10 == 0:
            logger.info(
                f"Épisode {episode+1}/{num_episodes}, Récompense: {episode_reward:.2f}, Epsilon: {target_agent.epsilon:.2f}"
            )

    logger.info(f"Ajustement terminé. Récompense moyenne: {np.mean(rewards[-10:]):.2f}")

    return target_agent


def use_domain_adaptation(
    envs: Dict[str, Dict[str, TradingEnvironment]],
    data_loaders: Dict[str, Dict[str, torch.utils.data.DataLoader]],
) -> DQNAgent:
    """
    Utilise l'adaptation de domaine pour créer un agent qui fonctionne bien sur plusieurs marchés.

    Args:
        envs: Dictionnaire d'environnements
        data_loaders: Dictionnaire de DataLoaders

    Returns:
        Agent adapté à plusieurs marchés
    """
    logger.info("Application de l'adaptation de domaine entre BTC et ETH...")

    # Environnements source et cible
    source_env = envs["btc"]["train"]
    target_env = envs["eth"]["train"]

    # Créer un agent pour l'adaptation de domaine
    state_dim = source_env.observation_space.shape[0]
    action_dim = source_env.action_space.n

    # Créer un réseau de base pour l'agent
    class BaseNetwork(torch.nn.Module):
        def __init__(self, state_dim, action_dim):
            super(BaseNetwork, self).__init__()
            self.feature_extractor = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
            )
            self.value_head = torch.nn.Linear(64, action_dim)

        def forward(self, x):
            features = self.feature_extractor(x)
            return self.value_head(features)

        def get_features(self, x):
            return self.feature_extractor(x)

    # Instancier le réseau de base
    base_model = BaseNetwork(state_dim, action_dim)

    # Initialiser l'adaptation de domaine
    domain_adaptation = DomainAdaptation(
        source_model=base_model,
        adaptation_type="dann",  # Domain-Adversarial Neural Network
        lambda_param=0.1,
    )

    # Entraîner avec adaptation de domaine
    logger.info("Entraînement avec adaptation de domaine...")
    history = domain_adaptation.train(
        source_loader=data_loaders["btc"]["train"],
        target_loader=data_loaders["eth"]["train"],
        val_loader=data_loaders["eth"]["val"],
        epochs=10,
    )

    # Créer un agent qui utilise le modèle adapté
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=0.3,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update=10,
    )

    # Remplacer le réseau Q par le modèle adapté
    agent.q_network = domain_adaptation.source_model
    agent.target_network.load_state_dict(agent.q_network.state_dict())

    return agent


def evaluate_agent(
    agent: DQNAgent, env: TradingEnvironment, num_episodes: int = 10
) -> np.ndarray:
    """
    Évalue un agent sur un environnement donné.

    Args:
        agent: Agent à évaluer
        env: Environnement d'évaluation
        num_episodes: Nombre d'épisodes d'évaluation

    Returns:
        Tableau des récompenses par épisode
    """
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Sélectionner une action (sans exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.q_network(state_tensor)
                action = q_values.argmax().item()

            # Exécuter l'action
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)

    return np.array(rewards)


def compare_performance(
    agents_dict: Dict[str, DQNAgent], envs: Dict[str, Dict[str, TradingEnvironment]]
) -> None:
    """
    Compare les performances des différents agents sur différents marchés.

    Args:
        agents_dict: Dictionnaire des agents à comparer
        envs: Dictionnaire des environnements
    """
    logger.info("Comparaison des performances des différents agents...")

    # Marchés à évaluer
    markets = ["btc", "eth"]

    # Pour chaque agent, évaluer sur les deux marchés
    results = {}

    for agent_name, agent in agents_dict.items():
        results[agent_name] = {}

        for market in markets:
            # Évaluer sur l'ensemble de test
            rewards = evaluate_agent(agent, envs[market]["test"], num_episodes=20)
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            results[agent_name][market] = {
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "rewards": rewards,
            }

            logger.info(
                f"Agent {agent_name} sur {market.upper()}: {avg_reward:.2f} ± {std_reward:.2f}"
            )

    # Visualiser les résultats
    plt.figure(figsize=(12, 6))

    # Barplot pour les récompenses moyennes
    plt.subplot(1, 2, 1)

    x = np.arange(len(markets))
    width = 0.2
    offset = -width * (len(agents_dict) - 1) / 2

    for i, (agent_name, agent_results) in enumerate(results.items()):
        means = [agent_results[market]["avg_reward"] for market in markets]
        stds = [agent_results[market]["std_reward"] for market in markets]

        plt.bar(x + offset + i * width, means, width, label=agent_name, yerr=stds)

    plt.xticks(x, [m.upper() for m in markets])
    plt.ylabel("Récompense moyenne")
    plt.title("Performance sur différents marchés")
    plt.legend()

    # Boxplot pour la distribution des récompenses
    plt.subplot(1, 2, 2)

    for i, market in enumerate(markets):
        market_data = []
        labels = []

        for agent_name, agent_results in results.items():
            market_data.append(agent_results[market]["rewards"])
            labels.append(f"{agent_name} ({market.upper()})")

        plt.boxplot(
            market_data,
            positions=np.arange(i * len(agents_dict), (i + 1) * len(agents_dict)),
            labels=labels,
            widths=0.6,
        )

    plt.ylabel("Récompense")
    plt.title("Distribution des récompenses")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Sauvegarder la figure
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "market_transfer_results.png"))

    logger.info(
        f"Figure sauvegardée dans {os.path.join(results_dir, 'market_transfer_results.png')}"
    )


def main():
    """
    Fonction principale démontrant le transfert d'apprentissage entre marchés.
    """
    # 1. Préparer les ensembles de données et les environnements
    logger.info("Préparation des données et des environnements...")
    data_dict, envs = prepare_datasets()

    # 2. Créer des DataLoaders pour l'entraînement
    logger.info("Création des DataLoaders...")
    data_loaders = create_data_loaders(envs)

    # 3. Entraîner un agent sur le marché source (BTC)
    source_agent = train_source_agent(envs)

    # 4. Créer un agent pour le marché cible sans transfert (base de comparaison)
    logger.info("Entraînement d'un agent sur ETH sans transfert (référence)...")
    target_env = envs["eth"]["train"]

    baseline_agent = DQNAgent(
        state_dim=target_env.observation_space.shape[0],
        action_dim=target_env.action_space.n,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update=10,
    )

    # Entraîner l'agent de référence
    num_episodes = 100
    for episode in range(num_episodes):
        state = target_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = baseline_agent.select_action(state)
            next_state, reward, done, _ = target_env.step(action)
            baseline_agent.store_experience(state, action, reward, next_state, done)
            baseline_agent.learn()
            episode_reward += reward
            state = next_state

        if episode % 10 == 0:
            logger.info(
                f"Épisode {episode+1}/{num_episodes}, Récompense: {episode_reward:.2f}"
            )

    # 5. Appliquer le transfert d'apprentissage
    transfer_agent = apply_transfer_learning(source_agent, envs, data_loaders)

    # 6. Utiliser l'adaptation de domaine
    domain_agent = use_domain_adaptation(envs, data_loaders)

    # 7. Comparer les performances
    agents = {
        "Source (BTC)": source_agent,
        "Baseline (ETH)": baseline_agent,
        "Transfer": transfer_agent,
        "Domain Adaptation": domain_agent,
    }

    compare_performance(agents, envs)


if __name__ == "__main__":
    main()
