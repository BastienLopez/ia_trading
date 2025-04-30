import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
matplotlib.use("Agg")  # Forcer le backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ai_trading.config import LOGS_DIR, MODELS_DIR, VISUALIZATION_DIR
from ai_trading.rl.dqn_agent import DQNAgent

# Configuration du logger
logger = logging.getLogger("TrainRL")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

INFO_RETOUR_DIR = Path(__file__).parent.parent / "info_retour"
INFO_RETOUR_DIR.mkdir(exist_ok=True)


class TrainingMonitor:
    """
    Classe pour surveiller et visualiser l'entraînement en temps réel.
    """

    def __init__(self, initial_balance, save_dir=None, plot_interval=10):
        """
        Initialise le moniteur d'entraînement.

        Args:
            initial_balance (float): Solde initial du portefeuille
            save_dir (str): Répertoire de sauvegarde des graphiques
            plot_interval (int): Intervalle de mise à jour des graphiques
        """
        self.initial_balance = initial_balance

        # Utiliser VISUALIZATION_DIR si aucun répertoire n'est spécifié
        if save_dir is None:
            self.save_dir = VISUALIZATION_DIR
        else:
            self.save_dir = save_dir

        # Créer le répertoire si nécessaire
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.figs = {}  # Initialiser les dictionnaires de figures
        self.axes = {}  # Initialiser les dictionnaires d'axes

        if self.save_dir:
            self._init_figures()  # Forcer la création des figures pour les sauvegardes
        elif plt.isinteractive():
            self._init_figures()

        self.plot_interval = plot_interval
        self.history = {
            "episode_rewards": [],
            "episode_portfolio_values": [],
            "episode_returns": [],
            "losses": [],
            "epsilon": [],
        }

    def _init_figures(self):
        """Initialise les figures pour la visualisation en temps réel."""
        # Figure pour les récompenses
        self.figs["rewards"] = plt.figure(figsize=(10, 6))
        self.axes["rewards"] = self.figs["rewards"].add_subplot(111)
        self.axes["rewards"].set_title("Récompenses par épisode")
        self.axes["rewards"].set_xlabel("Épisode")
        self.axes["rewards"].set_ylabel("Récompense totale")
        self.axes["rewards"].grid(True)

        # Figure pour la valeur du portefeuille
        self.figs["portfolio"] = plt.figure(figsize=(10, 6))
        self.axes["portfolio"] = self.figs["portfolio"].add_subplot(111)
        self.axes["portfolio"].set_title("Valeur du portefeuille par épisode")
        self.axes["portfolio"].set_xlabel("Épisode")
        self.axes["portfolio"].set_ylabel("Valeur ($)")
        self.axes["portfolio"].grid(True)

        # Figure pour les rendements
        self.figs["returns"] = plt.figure(figsize=(10, 6))
        self.axes["returns"] = self.figs["returns"].add_subplot(111)
        self.axes["returns"].set_title("Rendements par épisode")
        self.axes["returns"].set_xlabel("Épisode")
        self.axes["returns"].set_ylabel("Rendement (%)")
        self.axes["returns"].grid(True)

        # Figure pour les pertes
        self.figs["losses"] = plt.figure(figsize=(10, 6))
        self.axes["losses"] = self.figs["losses"].add_subplot(111)
        self.axes["losses"].set_title("Perte moyenne par épisode")
        self.axes["losses"].set_xlabel("Épisode")
        self.axes["losses"].set_ylabel("Perte")
        self.axes["losses"].grid(True)

        # Figure pour epsilon
        self.figs["epsilon"] = plt.figure(figsize=(10, 6))
        self.axes["epsilon"] = self.figs["epsilon"].add_subplot(111)
        self.axes["epsilon"].set_title("Epsilon par épisode")
        self.axes["epsilon"].set_xlabel("Épisode")
        self.axes["epsilon"].set_ylabel("Epsilon")
        self.axes["epsilon"].grid(True)

        # Afficher les figures
        for fig in self.figs.values():
            fig.tight_layout()

    def update(
        self,
        episode,
        total_reward,
        portfolio_value,
        epsilon,
        avg_loss,
        elapsed_time,
        step=None,
        returns=None,
        loss=None,
    ):
        """
        Met à jour l'historique d'entraînement et les visualisations.

        Args:
            episode (int): Numéro de l'épisode
            total_reward (float): Récompense totale de l'épisode
            portfolio_value (float): Valeur finale du portefeuille
            epsilon (float): Valeur actuelle d'epsilon
            avg_loss (float): Perte moyenne de l'épisode
            elapsed_time (float): Temps écoulé pour l'épisode
            step (int, optional): Étape actuelle (peut être ignorée)
            returns (float, optional): Rendement de l'épisode
            loss (float, optional): Perte de l'épisode
        """
        # Mettre à jour l'historique
        self.history["episode_rewards"].append(total_reward)
        self.history["episode_portfolio_values"].append(portfolio_value)
        self.history["episode_returns"].append(
            (portfolio_value / self.initial_balance) - 1
        )
        self.history["losses"].append(avg_loss)
        self.history["epsilon"].append(epsilon)

        # Mettre à jour les visualisations en temps réel si l'intervalle est atteint
        if plt.isinteractive() and episode % self.plot_interval == 0:
            self._update_plots()

    def _update_plots(self):
        """Met à jour les graphiques en temps réel."""
        # Mettre à jour le graphique des récompenses
        self.axes["rewards"].clear()
        self.axes["rewards"].plot(self.history["episode_rewards"])
        self.axes["rewards"].set_title("Récompenses par épisode")
        self.axes["rewards"].set_xlabel("Épisode")
        self.axes["rewards"].set_ylabel("Récompense totale")
        self.axes["rewards"].grid(True)

        # Mettre à jour le graphique de la valeur du portefeuille
        self.axes["portfolio"].clear()
        self.axes["portfolio"].plot(self.history["episode_portfolio_values"])
        self.axes["portfolio"].set_title("Valeur du portefeuille par épisode")
        self.axes["portfolio"].set_xlabel("Épisode")
        self.axes["portfolio"].set_ylabel("Valeur ($)")
        self.axes["portfolio"].grid(True)

        # Mettre à jour le graphique des rendements
        self.axes["returns"].clear()
        self.axes["returns"].plot(self.history["episode_returns"])
        self.axes["returns"].set_title("Rendements par épisode")
        self.axes["returns"].set_xlabel("Épisode")
        self.axes["returns"].set_ylabel("Rendement (%)")
        self.axes["returns"].grid(True)

        # Mettre à jour le graphique des pertes
        self.axes["losses"].clear()
        self.axes["losses"].plot(self.history["losses"])
        self.axes["losses"].set_title("Perte moyenne par épisode")
        self.axes["losses"].set_xlabel("Épisode")
        self.axes["losses"].set_ylabel("Perte")
        self.axes["losses"].grid(True)

        # Mettre à jour le graphique d'epsilon
        self.axes["epsilon"].clear()
        self.axes["epsilon"].plot(self.history["epsilon"])
        self.axes["epsilon"].set_title("Epsilon par épisode")
        self.axes["epsilon"].set_xlabel("Épisode")
        self.axes["epsilon"].set_ylabel("Epsilon")
        self.axes["epsilon"].grid(True)

        # Rafraîchir les figures
        for fig in self.figs.values():
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except NotImplementedError:
                # Ignorer les erreurs sur les backends qui ne supportent pas flush_events
                pass

    def save_plots(self):
        """Sauvegarde tous les graphiques dans le répertoire spécifié."""
        if not self.save_dir:
            logger.warning("Aucun répertoire de sauvegarde spécifié.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Mettre à jour les graphiques avant la sauvegarde
        self._update_plots()

        # Sauvegarder chaque figure avec un timestamp unique
        for name, fig in self.figs.items():
            save_path = os.path.join(self.save_dir, f"{name}_{timestamp}.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        logger.info(f"Figures sauvegardées dans {self.save_dir}")

    def get_history(self):
        """
        Retourne l'historique d'entraînement.

        Returns:
            dict: Historique d'entraînement
        """
        return self.history


def train_agent(
    agent,
    env,
    episodes=1000,
    batch_size=32,
    update_target_every=5,
    save_path=None,
    visualize=True,
    checkpoint_interval=100,
    early_stopping=None,
    max_steps_per_episode=None,
    use_tensorboard=False,
    tensorboard_log_dir=None,
):
    """
    Entraîne un agent d'apprentissage par renforcement.

    Args:
        agent: Agent d'apprentissage par renforcement
        env: Environnement de trading
        episodes (int): Nombre d'épisodes d'entraînement
        batch_size (int): Taille du batch pour l'apprentage
        update_target_every (int): Fréquence de mise à jour du modèle cible
        save_path (str): Chemin pour sauvegarder le modèle
        visualize (bool): Afficher les visualisations
        checkpoint_interval (int): Intervalle pour sauvegarder les checkpoints
        early_stopping (dict): Paramètres pour l'arrêt anticipé
        max_steps_per_episode (int): Nombre maximum d'étapes par épisode
        use_tensorboard (bool): Utiliser TensorBoard pour le suivi
        tensorboard_log_dir (str): Répertoire pour les logs TensorBoard

    Returns:
        dict: Historique d'entraînement
    """
    # Vérifier la taille de l'état
    state_size = env.observation_space.shape[0]
    if state_size != agent.state_size:
        logger.warning(
            f"Taille de l'état de l'environnement ({state_size}) différente de celle de l'agent ({agent.state_size})"
        )
        logger.warning("Réinitialisation de l'agent avec la bonne taille d'état")
        # Réinitialiser l'agent avec la bonne taille d'état
        new_agent = DQNAgent(
            state_size=state_size,
            action_size=agent.action_size,
            learning_rate=agent.learning_rate,
            gamma=agent.gamma,
            epsilon=agent.epsilon,
            epsilon_decay=agent.epsilon_decay,
            epsilon_min=agent.epsilon_min,
            batch_size=agent.batch_size,
            memory_size=len(agent.memory),
        )
        # Copier les autres attributs si nécessaire
        agent = new_agent

    # Configurer TensorBoard si demandé
    if use_tensorboard:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if tensorboard_log_dir is None:
            log_dir = LOGS_DIR / f"tensorboard/{current_time}"
        else:
            log_dir = tensorboard_log_dir

        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(str(log_dir))
        logger.info(f"Logs TensorBoard disponibles dans {log_dir}")

    # Créer le dossier de sauvegarde si nécessaire
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        # Utiliser MODELS_DIR par défaut si aucun chemin n'est spécifié
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = MODELS_DIR / f"rl_agent_{timestamp}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialiser le moniteur d'entraînement
    viz_dir = VISUALIZATION_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    monitor = TrainingMonitor(
        initial_balance=env.initial_balance, save_dir=viz_dir, plot_interval=5
    )

    # Paramètres pour l'arrêt anticipé
    if early_stopping:
        best_value = float("-inf")
        patience_counter = 0
        patience = early_stopping.get("patience", 10)
        min_delta = early_stopping.get("min_delta", 0.0)
        metric = early_stopping.get("metric", "returns")

    # Temps de début de l'entraînement
    start_time = time.time()

    # Boucle d'entraînement avec barre de progression
    for e in tqdm(range(episodes), desc="Entraînement"):
        # Réinitialiser l'environnement
        state = env.reset()

        # Redimensionner l'état en utilisant la taille de l'état de l'agent
        # qui a été mise à jour si nécessaire
        state = np.reshape(state, [1, agent.state_size])

        # Variables pour suivre les performances de l'épisode
        total_reward = 0
        episode_losses = []
        step_count = 0

        done = False
        while not done:
            # Vérifier si le nombre maximum d'étapes est atteint
            if max_steps_per_episode and step_count >= max_steps_per_episode:
                break

            # Choisir une action
            action = agent.act(state)

            # Exécuter l'action
            next_state, reward, done, info = env.step(action)

            # Redimensionner l'état suivant en utilisant la taille de l'état de l'agent
            next_state = np.reshape(next_state, [1, agent.state_size])

            # Stocker l'expérience dans la mémoire
            agent.remember(state, action, reward, next_state, done)

            # Passer à l'état suivant
            state = next_state

            # Accumuler la récompense
            total_reward += reward

            # Entraîner l'agent
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                episode_losses.append(loss)

            step_count += 1

        # Mettre à jour le modèle cible périodiquement
        if e % update_target_every == 0:
            agent.update_target_model()

        # Calculer les métriques de l'épisode
        portfolio_value = env.get_portfolio_value()
        returns = (portfolio_value / env.initial_balance) - 1
        avg_loss = np.mean(episode_losses) if episode_losses else 0

        # Mettre à jour le moniteur
        monitor.update(
            e,
            total_reward,
            portfolio_value,
            agent.epsilon,
            avg_loss,
            time.time() - start_time,
        )

        # Ajouter la récompense à l'historique de l'agent
        agent.reward_history.append(total_reward)

        # Afficher les progrès
        if (e + 1) % 10 == 0 or e == 0:
            elapsed_time = time.time() - start_time
            logger.info(
                f"Épisode {e+1}/{episodes} - Récompense: {total_reward:.4f}, "
                f"Valeur du portefeuille: ${portfolio_value:.2f}, "
                f"Rendement: {returns*100:.2f}%, "
                f"Epsilon: {agent.epsilon:.4f}, "
                f"Perte moyenne: {avg_loss:.4f}, "
                f"Temps écoulé: {elapsed_time:.1f}s"
            )

        # Enregistrer les métriques dans TensorBoard
        if use_tensorboard:
            with summary_writer.as_default():
                tf.summary.scalar("reward", total_reward, step=e)
                tf.summary.scalar("portfolio_value", portfolio_value, step=e)
                tf.summary.scalar("returns", returns, step=e)
                tf.summary.scalar("loss", avg_loss, step=e)
                tf.summary.scalar("epsilon", agent.epsilon, step=e)

        # Sauvegarder le modèle périodiquement
        if save_path and (e + 1) % checkpoint_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"{save_path}_ep{e+1}_{timestamp}.h5"
            agent.save(save_file)
            logger.info(f"Modèle sauvegardé dans {save_file}")

        # Vérifier l'arrêt anticipé
        if early_stopping:
            current_value = (
                total_reward
                if metric == "reward"
                else portfolio_value if metric == "portfolio_value" else returns
            )

            if current_value > best_value + min_delta:
                best_value = current_value
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                if save_path:
                    best_model_path = f"{save_path}_best.h5"
                    agent.save(best_model_path)
                    logger.info(f"Meilleur modèle sauvegardé dans {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Arrêt anticipé à l'épisode {e+1} après {patience} épisodes sans amélioration"
                    )
                    break

    # Temps total d'entraînement
    total_time = time.time() - start_time
    logger.info(f"Entraînement terminé en {total_time:.1f} secondes")

    # Sauvegarder le modèle final
    if save_path:
        final_save_path = f"{save_path}_final.h5"
        agent.save(final_save_path)
        logger.info(f"Modèle final sauvegardé dans {final_save_path}")

    # Générer des visualisations finales
    if visualize:
        monitor.save_plots()

    return monitor.get_history()
