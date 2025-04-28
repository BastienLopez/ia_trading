import logging
import os
import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from ai_trading.rl.ucb_exploration import (
    HybridExploration,
    NoveltyExploration,
    UCBExploration,
)

# Configuration du logger
logger = logging.getLogger("DQNAgentAdvancedExploration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DQNAgentAdvancedExploration:
    """
    Agent DQN utilisant des stratégies d'exploration avancées (UCB, nouveauté, ou hybride)
    au lieu de l'approche epsilon-greedy traditionnelle.
    """

    def __init__(
        self,
        state_size,
        action_size,
        exploration_strategy="ucb",  # 'ucb', 'novelty', 'hybrid'
        learning_rate=0.001,
        gamma=0.95,  # Facteur d'actualisation
        batch_size=32,
        memory_size=2000,
        target_update_freq=100,
        ucb_c=2.0,
        novelty_scale=1.0,
        decay_rate=0.99,
        hash_bins=10,
        max_count=1000,
        verbose=False,
    ):
        """
        Initialise l'agent DQN avec une stratégie d'exploration avancée.

        Args:
            state_size (int): Taille de l'espace d'état
            action_size (int): Taille de l'espace d'action
            exploration_strategy (str): Stratégie d'exploration ('ucb', 'novelty', 'hybrid')
            learning_rate (float): Taux d'apprentissage pour le modèle
            gamma (float): Facteur d'actualisation pour les récompenses futures
            batch_size (int): Taille du batch pour l'apprentissage
            memory_size (int): Taille maximale de la mémoire de replay
            target_update_freq (int): Fréquence de mise à jour du modèle cible (en étapes)
            ucb_c (float): Paramètre de confiance pour UCB
            novelty_scale (float): Échelle du bonus de nouveauté
            decay_rate (float): Taux de décroissance de la nouveauté
            hash_bins (int): Nombre de bins pour le hachage d'état (nouveauté)
            max_count (int): Valeur maximale du compteur d'états (nouveauté)
            verbose (bool): Afficher les logs détaillés
        """
        self.state_size = state_size
        self.action_size = action_size
        self.exploration_strategy = exploration_strategy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.verbose = verbose

        # Paramètres d'exploration
        self.ucb_c = ucb_c
        self.novelty_scale = novelty_scale
        self.decay_rate = decay_rate
        self.hash_bins = hash_bins
        self.max_count = max_count

        # Initialiser les explorateurs en fonction de la stratégie choisie
        if self.exploration_strategy == "ucb":
            self.explorer = UCBExploration(action_size, c=ucb_c)
        elif self.exploration_strategy == "novelty":
            self.explorer = NoveltyExploration(
                state_size,
                action_size,
                novelty_scale=novelty_scale,
                decay_rate=decay_rate,
                hash_bins=hash_bins,
                max_count=max_count,
            )
        elif self.exploration_strategy == "hybrid":
            self.explorer = HybridExploration(
                state_size,
                action_size,
                ucb_c=ucb_c,
                novelty_scale=novelty_scale,
                decay_rate=decay_rate,
                hash_bins=hash_bins,
                max_count=max_count,
            )
        else:
            raise ValueError(
                f"Stratégie d'exploration '{exploration_strategy}' non reconnue. "
                "Utilisez 'ucb', 'novelty' ou 'hybrid'."
            )

        # Métriques de performance
        self.rewards_history = []
        self.loss_history = []

        # Construire les modèles (principal et cible)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # Synchroniser les poids au départ

        logger.info(
            f"Agent DQN initialisé avec la stratégie d'exploration {exploration_strategy}"
        )
        logger.info(f"Taille état: {state_size}, Taille action: {action_size}")

    def _build_model(self):
        """
        Construit le réseau de neurones pour l'approximation de la fonction Q.

        Returns:
            keras.Model: Le modèle compilé
        """
        model = Sequential()
        # Couche d'entrée et première couche cachée
        model.add(
            Dense(
                64,
                input_dim=self.state_size,
                activation="relu",
                kernel_initializer="he_uniform",
            )
        )
        # Deuxième couche cachée
        model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        # Couche de sortie (une valeur Q pour chaque action)
        model.add(
            Dense(
                self.action_size, activation="linear", kernel_initializer="he_uniform"
            )
        )

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        if self.verbose:
            model.summary()

        return model

    def update_target_model(self):
        """
        Met à jour le modèle cible avec les poids du modèle principal.
        """
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Modèle cible mis à jour")

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke une expérience dans la mémoire de replay.

        Args:
            state (numpy.array): État courant
            action (int): Action choisie
            reward (float): Récompense reçue
            next_state (numpy.array): État suivant
            done (bool): Indique si l'épisode est terminé
        """
        self.memory.append((state, action, reward, next_state, done))

        # Mettre à jour l'explorateur avec l'action prise et la récompense reçue
        if hasattr(self.explorer, "update"):
            self.explorer.update(action, reward)

    def act(self, state):
        """
        Choisit une action en fonction de l'état actuel.

        Args:
            state (numpy.array): État courant

        Returns:
            int: Action choisie
        """
        # Normaliser l'état pour le réseau neuronal
        state = np.reshape(state, [1, self.state_size])

        # Obtenir les valeurs Q prédites par le modèle
        q_values = self.model.predict(state, verbose=0)[0]

        # Appliquer la stratégie d'exploration pour choisir une action
        action = self.explorer.select_action(q_values, state[0])

        if self.verbose:
            logger.debug(f"Action choisie: {action}, Q-values: {q_values}")

        return action

    def replay(self, batch_size=None):
        """
        Entraîne le modèle sur un échantillon d'expériences.

        Args:
            batch_size (int, optional): Taille du batch. Par défaut, utilise self.batch_size.

        Returns:
            float: La perte moyenne (loss) du batch
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Vérifier si la mémoire contient suffisamment d'expériences
        if len(self.memory) < batch_size:
            return 0

        # Échantillonner un batch d'expériences
        minibatch = random.sample(self.memory, batch_size)

        # Préparer les données d'entraînement
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))

        # Calculer les cibles pour chaque expérience
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state

            # Prédire les valeurs Q actuelles
            target = self.model.predict(state.reshape(1, self.state_size), verbose=0)[0]

            if done:
                # Si l'épisode est terminé, pas de récompense future
                target[action] = reward
            else:
                # Utiliser le modèle cible pour une estimation stable des valeurs Q futures
                next_q_values = self.target_model.predict(
                    next_state.reshape(1, self.state_size), verbose=0
                )[0]

                # Appliquer l'équation de Bellman
                target[action] = reward + self.gamma * np.amax(next_q_values)

            targets[i] = target

        # Entraîner le modèle
        history = self.model.fit(
            states, targets, epochs=1, verbose=0, batch_size=batch_size
        )

        # Mettre à jour le compteur pour le modèle cible
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()

        # Appliquer la décroissance de la nouveauté si nécessaire
        if hasattr(self.explorer, "decay_novelty"):
            self.explorer.decay_novelty()

        # Stocker la perte pour le suivi
        loss = history.history["loss"][0]
        self.loss_history.append(loss)

        return loss

    def end_episode(self, episode_reward):
        """
        Effectue les opérations de fin d'épisode.

        Args:
            episode_reward (float): Récompense totale pour l'épisode
        """
        # Enregistrer la récompense pour le suivi
        self.rewards_history.append(episode_reward)

        # Afficher des statistiques périodiques
        if len(self.rewards_history) % 10 == 0 and self.verbose:
            logger.info(
                f"Épisode {len(self.rewards_history)}, "
                f"Récompense moyenne sur 10 épisodes: "
                f"{np.mean(self.rewards_history[-10:]):.2f}"
            )

    def load(self, path):
        """
        Charge les poids du modèle à partir d'un fichier.

        Args:
            path (str): Chemin vers le fichier de poids
        """
        if os.path.exists(path):
            self.model.load_weights(path)
            self.update_target_model()
            logger.info(f"Poids chargés depuis {path}")
        else:
            logger.warning(f"Fichier de poids {path} non trouvé")

    def save(self, path):
        """
        Sauvegarde les poids du modèle dans un fichier.

        Args:
            path (str): Chemin pour sauvegarder les poids
        """
        self.model.save_weights(path)
        logger.info(f"Poids sauvegardés dans {path}")

    def get_metrics(self):
        """
        Retourne les métriques de performance de l'agent.

        Returns:
            dict: Dictionnaire contenant les métriques de performance
        """
        return {
            "rewards": self.rewards_history,
            "losses": self.loss_history,
            "memory_size": len(self.memory),
        }

    def reset_explorer(self):
        """
        Réinitialise l'explorateur (utile entre les épisodes).
        """
        if hasattr(self.explorer, "reset"):
            self.explorer.reset()
            logger.debug("Explorateur réinitialisé")
