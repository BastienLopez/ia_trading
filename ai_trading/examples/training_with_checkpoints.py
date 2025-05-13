"""
Exemple d'entraînement d'agent RL avec journalisation avancée et sauvegarde/reprise.

Ce script montre comment utiliser les modules de journalisation avancée et de gestion
des checkpoints pour entraîner un agent RL avec sauvegarde et reprise automatiques.
"""

import argparse
import os
import random
import sys

import numpy as np

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim

from ai_trading.utils.advanced_logging import (
    get_logger,
    log_exceptions,
    log_execution_time,
)
from ai_trading.utils.checkpoint_manager import CheckpointType, get_checkpoint_manager
from ai_trading.utils.performance_logger import (
    get_performance_tracker,
    start_metrics_collection,
    stop_metrics_collection,
)

# Configurer le logger
logger = get_logger("ai_trading.examples.training")


# Définir un modèle d'agent RL simple
class SimpleRLAgent(nn.Module):
    """Modèle d'agent RL simple pour la démonstration."""

    def __init__(self, state_size=10, hidden_size=64, action_size=3):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Réseau de l'acteur (politique)
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

        # Réseau du critique (valeur)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        """
        Calcule la distribution de probabilité des actions et la valeur d'état.

        Args:
            state: Tenseur d'état de forme (batch_size, state_size)

        Returns:
            Tuple (action_probs, state_value)
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


# Environnement de trading simplifié
class SimpleTradingEnvironment:
    """Environnement de trading simplifié pour la démonstration."""

    def __init__(self, data_length=1000, state_size=10):
        self.data_length = data_length
        self.state_size = state_size
        self.current_step = 0
        self.max_steps = 200

        # Générer des données de prix aléatoires
        self.price_data = np.cumsum(np.random.normal(0, 1, data_length))
        self.price_data = 100 + 20 * (self.price_data - np.min(self.price_data)) / (
            np.max(self.price_data) - np.min(self.price_data)
        )

        # Générer des indicateurs aléatoires
        self.indicators = np.random.randn(data_length, state_size - 1)

        # État initial
        self.position = 0  # -1: short, 0: neutre, 1: long
        self.balance = 10000.0
        self.initial_balance = self.balance

        # Historique
        self.history = []

    def reset(self):
        """
        Réinitialise l'environnement.

        Returns:
            État initial
        """
        self.current_step = random.randint(
            self.state_size, self.data_length - self.max_steps
        )
        self.position = 0
        self.balance = self.initial_balance
        self.history = []

        return self._get_state()

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action: Indice de l'action (0: conserver, 1: acheter, 2: vendre)

        Returns:
            Tuple (next_state, reward, done, info)
        """
        # Obtenir le prix actuel
        current_price = self.price_data[self.current_step]

        # Déterminer la nouvelle position
        new_position = self.position
        if action == 1:  # Acheter
            new_position = 1
        elif action == 2:  # Vendre
            new_position = -1

        # Calculer la récompense
        price_change = self.price_data[self.current_step + 1] - current_price

        # Récompense basée sur le changement de prix et la position
        if self.position == 1:  # Long
            reward = price_change
        elif self.position == -1:  # Short
            reward = -price_change
        else:  # Neutre
            reward = 0

        # Pénalité pour les changements de position (frais de transaction)
        if new_position != self.position:
            reward -= 0.1 * current_price

        # Mettre à jour la balance
        self.balance += reward

        # Mettre à jour la position
        self.position = new_position

        # Avancer d'un pas
        self.current_step += 1

        # Vérifier si l'épisode est terminé
        done = (self.current_step >= self.data_length - 1) or (
            self.current_step - (self.current_step - self.state_size) >= self.max_steps
        )

        # Enregistrer l'historique
        self.history.append(
            {
                "step": self.current_step,
                "price": current_price,
                "position": self.position,
                "reward": reward,
                "balance": self.balance,
            }
        )

        # Informations supplémentaires
        info = {
            "price": current_price,
            "balance": self.balance,
            "position": self.position,
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """
        Retourne l'état actuel.

        Returns:
            Tenseur d'état
        """
        # Inclure le prix actuel et les indicateurs
        price = self.price_data[self.current_step]
        indicators = self.indicators[self.current_step]

        # Normaliser le prix
        normalized_price = (price - 90) / 20

        # Combiner le prix et les indicateurs
        state = np.concatenate(([normalized_price], indicators))

        return torch.FloatTensor(state)


# Fonction d'entraînement avec sauvegarde et journalisation
@log_exceptions()
@log_execution_time(level=20)  # INFO
def train_agent(
    agent,
    env,
    num_episodes=100,
    gamma=0.99,
    learning_rate=0.001,
    checkpoint_interval=20,
    resume_from=None,
    max_steps_per_episode=None,
):
    """
    Entraîne un agent RL avec sauvegarde et journalisation.

    Args:
        agent: Agent RL (modèle PyTorch)
        env: Environnement
        num_episodes: Nombre d'épisodes
        gamma: Facteur de réduction des récompenses futures
        learning_rate: Taux d'apprentissage
        checkpoint_interval: Intervalle de sauvegarde des checkpoints
        resume_from: ID du checkpoint pour reprendre l'entraînement
        max_steps_per_episode: Nombre maximum d'étapes par épisode

    Returns:
        Dictionnaire contenant les métriques d'entraînement
    """
    logger.info(f"Démarrage de l'entraînement pour {num_episodes} épisodes")

    # Configurer le tracker de performance
    perf_tracker = get_performance_tracker("training")

    # Configurer l'optimiseur
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # Obtenir le gestionnaire de checkpoints
    checkpoint_manager = get_checkpoint_manager()

    # Reprendre l'entraînement si demandé
    start_episode = 0
    all_rewards = []
    episode_durations = []

    if resume_from:
        logger.info(f"Reprise de l'entraînement depuis le checkpoint {resume_from}")

        try:
            # Charger le checkpoint
            checkpoint_data = checkpoint_manager.load_checkpoint(resume_from, agent)

            # Restaurer l'état de l'optimiseur
            if "optimizer_state" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state"])

            # Restaurer les métriques
            if "metrics" in checkpoint_data:
                metrics = checkpoint_data["metrics"]
                if "episode" in metrics:
                    start_episode = metrics["episode"] + 1
                if "all_rewards" in metrics:
                    all_rewards = metrics["all_rewards"]
                if "episode_durations" in metrics:
                    episode_durations = metrics["episode_durations"]

            logger.info(f"Entraînement repris à l'épisode {start_episode}")

        except Exception as e:
            logger.error(f"Erreur lors de la reprise de l'entraînement: {str(e)}")
            logger.info("Démarrage d'un nouvel entraînement")

    # Boucle d'entraînement
    for episode in range(start_episode, num_episodes):
        perf_tracker.start(f"episode_{episode}")

        # Réinitialiser l'environnement
        state = env.reset()

        # Variables pour cet épisode
        episode_reward = 0
        episode_log_probs = []
        episode_values = []
        episode_rewards = []
        episode_masks = []

        # Boucle d'étapes
        step = 0
        done = False

        while not done:
            # Obtenir l'action et la valeur d'état
            action_probs, value = agent(state.unsqueeze(0))

            # Échantillonner une action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            # Calculer le log de la probabilité de l'action
            log_prob = action_dist.log_prob(action)

            # Exécuter l'action dans l'environnement
            next_state, reward, done, info = env.step(action.item())

            # Enregistrer les données
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            episode_rewards.append(reward)
            episode_masks.append(1 - int(done))

            # Mettre à jour l'état et la récompense cumulée
            state = next_state
            episode_reward += reward

            # Limiter le nombre d'étapes si nécessaire
            step += 1
            if max_steps_per_episode and step >= max_steps_per_episode:
                done = True

        # Enregistrer la récompense de l'épisode
        all_rewards.append(episode_reward)
        episode_durations.append(step)

        # Calculer les avantages et les rendements
        returns = []
        advantages = []
        R = 0

        # Calculer les rendements et les avantages
        for i in reversed(range(len(episode_rewards))):
            R = episode_rewards[i] + gamma * R * episode_masks[i]
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)

        # Normaliser les rendements
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculer la perte
        policy_loss = 0
        value_loss = 0

        for log_prob, value, R in zip(episode_log_probs, episode_values, returns):
            advantage = R - value.item()

            # Perte de politique
            policy_loss -= log_prob * advantage

            # Perte de valeur
            value_loss += 0.5 * (value - R) ** 2

        # Perte totale
        loss = policy_loss + value_loss

        # Mettre à jour les paramètres
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Mesurer la durée de l'épisode
        episode_duration = perf_tracker.stop(f"episode_{episode}")

        # Journaliser les résultats
        logger.info(
            f"Épisode {episode+1}/{num_episodes}, "
            f"Récompense: {episode_reward:.2f}, "
            f"Étapes: {step}, "
            f"Durée: {episode_duration:.2f}s"
        )

        # Sauvegarder un checkpoint périodiquement
        if (episode + 1) % checkpoint_interval == 0 or episode == num_episodes - 1:
            # Métriques pour le checkpoint
            metrics = {
                "episode": episode,
                "total_steps": sum(episode_durations),
                "mean_reward": np.mean(all_rewards[-checkpoint_interval:]),
                "mean_duration": np.mean(episode_durations[-checkpoint_interval:]),
                "all_rewards": all_rewards,
                "episode_durations": episode_durations,
                "learning_rate": learning_rate,
                "gamma": gamma,
            }

            # Données supplémentaires
            custom_data = {
                "optimizer_state": optimizer.state_dict(),
            }

            # Sauvegarder le checkpoint
            checkpoint_id = checkpoint_manager.save_checkpoint(
                obj=agent,
                type=CheckpointType.MODEL,
                prefix="rl_agent",
                description=f"Agent RL après {episode+1} épisodes",
                metrics=metrics,
                custom_data=custom_data,
            )

            if checkpoint_id:
                logger.info(f"Checkpoint sauvegardé: {checkpoint_id}")

    # Métriques finales
    final_metrics = {
        "num_episodes": num_episodes,
        "total_steps": sum(episode_durations),
        "mean_reward": np.mean(all_rewards),
        "mean_duration": np.mean(episode_durations),
        "max_reward": np.max(all_rewards),
        "min_reward": np.min(all_rewards),
        "final_reward": all_rewards[-1] if all_rewards else 0,
        "all_rewards": all_rewards,
        "episode_durations": episode_durations,
    }

    logger.info("Entraînement terminé")
    logger.info(f"Récompense moyenne: {final_metrics['mean_reward']:.2f}")
    logger.info(f"Récompense finale: {final_metrics['final_reward']:.2f}")

    return final_metrics


# Fonction principale
def main():
    """Fonction principale pour la démonstration."""
    # Analyser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(
        description="Entraînement d'agent RL avec sauvegarde et journalisation"
    )
    parser.add_argument("--episodes", type=int, default=100, help="Nombre d'épisodes")
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Facteur de réduction"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Intervalle de sauvegarde des checkpoints",
    )
    parser.add_argument(
        "--resume-from", type=str, help="ID du checkpoint pour reprendre l'entraînement"
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Nombre maximum d'étapes par épisode"
    )
    args = parser.parse_args()

    logger.info(
        "Démarrage de la démonstration d'entraînement avec sauvegarde et journalisation"
    )

    # Démarrer la collecte des métriques système
    metrics_collector = start_metrics_collection(interval=10.0)

    try:
        # Créer l'environnement
        env = SimpleTradingEnvironment(data_length=10000, state_size=10)

        # Créer l'agent
        agent = SimpleRLAgent(state_size=10, hidden_size=64, action_size=3)

        # Entraîner l'agent
        train_agent(
            agent=agent,
            env=env,
            num_episodes=args.episodes,
            learning_rate=args.lr,
            gamma=args.gamma,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume_from,
            max_steps_per_episode=args.max_steps,
        )

    finally:
        # Arrêter la collecte des métriques système
        stop_metrics_collection()

    logger.info("Démonstration terminée")


if __name__ == "__main__":
    main()
