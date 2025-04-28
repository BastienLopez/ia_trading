import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class AdaptiveEntropyRegularization:
    """
    Classe pour la régularisation d'entropie adaptative dans les algorithmes d'apprentissage par renforcement.

    Cette classe gère le coefficient alpha qui contrôle l'importance de la maximisation d'entropie
    dans les algorithmes comme SAC (Soft Actor-Critic).
    """

    def __init__(
        self,
        action_size,
        initial_alpha=0.2,
        update_interval=1,
        learning_rate=3e-4,
        reward_scaling=5.0,
        target_entropy_ratio=1.0,
    ):
        """
        Initialise le mécanisme de régularisation d'entropie adaptative.

        Args:
            action_size (int): Dimension de l'espace d'action
            initial_alpha (float): Valeur initiale du coefficient alpha
            update_interval (int): Nombre d'étapes entre les mises à jour d'alpha
            learning_rate (float): Taux d'apprentissage pour l'optimisation d'alpha
            reward_scaling (float): Facteur d'échelle pour les récompenses (influence l'échelle d'alpha)
            target_entropy_ratio (float): Ratio pour calculer l'entropie cible (-action_size * ratio)
        """
        self.action_size = action_size
        self.initial_alpha = initial_alpha
        self.update_interval = update_interval
        self.learning_rate = learning_rate
        self.reward_scaling = reward_scaling
        self.target_entropy_ratio = target_entropy_ratio

        # Entropie cible (heuristique : -dim(A))
        self.target_entropy = -self.action_size * self.target_entropy_ratio

        # Variable log_alpha pour la stabilité numérique (alpha = exp(log_alpha))
        self.log_alpha = tf.Variable(np.log(initial_alpha), dtype=tf.float32)

        # Optimiseur pour la mise à jour d'alpha
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compteur d'étapes pour la mise à jour périodique
        self.steps_counter = 0

        logger.info(
            f"Régularisation d'entropie adaptative initialisée avec: "
            f"action_size={action_size}, initial_alpha={initial_alpha}, "
            f"target_entropy={self.target_entropy}"
        )

    def get_alpha(self):
        """Retourne la valeur actuelle d'alpha."""
        return tf.exp(self.log_alpha)

    def update(self, log_probs):
        """
        Met à jour la valeur d'alpha en fonction des probabilités logarithmiques des actions.

        Args:
            log_probs (tf.Tensor): Logarithmes des probabilités des actions échantillonnées

        Returns:
            float: La perte d'alpha après la mise à jour
        """
        self.steps_counter += 1

        # Ne mettre à jour alpha qu'à intervalles réguliers
        if self.steps_counter % self.update_interval != 0:
            return 0.0

        with tf.GradientTape() as tape:
            # Calculer la perte pour alpha
            # L'objectif est de minimiser: -alpha * (log_prob + target_entropy)
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
            )

        # Calculer et appliquer les gradients
        gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.optimizer.apply_gradients(zip(gradients, [self.log_alpha]))

        # Journaliser la valeur actuelle d'alpha et sa perte
        alpha = self.get_alpha()
        logger.debug(
            f"Alpha mis à jour: {alpha.numpy():.4f}, loss: {alpha_loss.numpy():.4f}"
        )

        return alpha_loss.numpy()

    def reset(self):
        """Réinitialise alpha à sa valeur initiale."""
        self.log_alpha.assign(np.log(self.initial_alpha))
        self.steps_counter = 0
        logger.info(f"Alpha réinitialisé à {self.initial_alpha}")
