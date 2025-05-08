"""
Module de méta-apprentissage (Meta-Learning) pour l'adaptation rapide entre marchés financiers.

Ce module implémente:
- MAML (Model-Agnostic Meta-Learning) pour l'adaptation rapide entre différents marchés
- Adaptation de MAML pour séries temporelles financières
- Fonctions d'utilitaires pour le meta-apprentissage dans le contexte du trading
"""

import copy
import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ai_trading.config import MODELS_DIR

logger = logging.getLogger(__name__)


class MAML:
    """
    Implémentation de Model-Agnostic Meta-Learning (MAML) pour le trading.

    MAML est une approche qui permet d'apprendre rapidement à partir de peu d'exemples
    sur de nouvelles tâches, ce qui est particulièrement utile pour s'adapter à
    différents marchés ou conditions de marché.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise l'algorithme MAML.

        Args:
            model: Modèle de base à méta-entraîner
            inner_lr: Taux d'apprentissage pour l'adaptation rapide (inner loop)
            outer_lr: Taux d'apprentissage pour le méta-apprentissage (outer loop)
            inner_steps: Nombre d'étapes d'adaptation rapide
            first_order: Si True, utilise l'approximation au premier ordre (plus rapide)
            device: Appareil sur lequel exécuter les calculs ("cuda" ou "cpu")
        """
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device

        # Initialiser l'optimiseur pour le méta-apprentissage (outer loop)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.outer_lr
        )

        # Fonction de perte pour les tâches financières (MSE par défaut)
        self.loss_fn = nn.MSELoss()

        logger.info(
            f"MAML initialisé avec device={device}, inner_lr={inner_lr}, outer_lr={outer_lr}"
        )

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapte le modèle à une nouvelle tâche en utilisant le support set.

        Args:
            support_x: Données d'entrée du support set
            support_y: Cibles du support set
            steps: Nombre d'étapes d'adaptation (utilise inner_steps par défaut)

        Returns:
            Le modèle adapté à la nouvelle tâche
        """
        if steps is None:
            steps = self.inner_steps

        # Créer une copie du modèle pour l'adapter à cette tâche
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        # Déplacer les données vers le bon device
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        # Adaptation rapide
        for _ in range(steps):
            # Forward pass
            predictions = adapted_model(support_x)
            loss = self.loss_fn(predictions, support_y)

            # Calculer les gradients
            grads = torch.autograd.grad(
                loss, adapted_model.parameters(), create_graph=not self.first_order
            )

            # Mettre à jour les paramètres manuellement
            updated_params = OrderedDict()
            for (name, param), grad in zip(adapted_model.named_parameters(), grads):
                updated_params[name] = param - self.inner_lr * grad

            # Remplacer les paramètres du modèle
            adapted_model = self._replace_params(adapted_model, updated_params)

        return adapted_model

    def _replace_params(self, model: nn.Module, params_dict: OrderedDict) -> nn.Module:
        """
        Remplace les paramètres d'un modèle par ceux fournis dans params_dict.

        Args:
            model: Le modèle à mettre à jour
            params_dict: Dictionnaire des nouveaux paramètres

        Returns:
            Le modèle avec les paramètres mis à jour
        """
        state_dict = model.state_dict()
        for name, param in params_dict.items():
            state_dict[name] = param

        model.load_state_dict(state_dict)
        return model

    def meta_train(
        self,
        task_generator: Callable,
        num_tasks: int,
        num_epochs: int,
        support_size: int,
        query_size: int,
        batch_size: int = 4,
    ) -> Dict[str, List[float]]:
        """
        Entraîne le modèle avec l'algorithme MAML sur un ensemble de tâches.

        Args:
            task_generator: Fonction qui génère des tâches (support_x, support_y, query_x, query_y)
            num_tasks: Nombre de tâches à générer par epoch
            num_epochs: Nombre d'epochs d'entraînement
            support_size: Taille des ensembles de support (adaptation)
            query_size: Taille des ensembles de query (évaluation)
            batch_size: Nombre de tâches à traiter en parallèle

        Returns:
            Historique d'entraînement avec les pertes meta-train et meta-val
        """
        history = {"meta_train_loss": [], "meta_val_loss": []}

        self.model.train()

        for epoch in range(num_epochs):
            meta_train_loss = 0.0

            # Échantillonner des tâches pour cet epoch
            tasks = [task_generator(support_size, query_size) for _ in range(num_tasks)]

            # Traiter les tâches par batch
            for i in range(0, num_tasks, batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_loss = self._process_task_batch(batch_tasks)
                meta_train_loss += batch_loss

            # Moyenne de la perte sur toutes les tâches
            meta_train_loss /= num_tasks
            history["meta_train_loss"].append(meta_train_loss)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Meta-train loss: {meta_train_loss:.6f}"
            )

        return history

    def _process_task_batch(
        self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Traite un batch de tâches pour le méta-apprentissage.

        Args:
            tasks: Liste de tuples (support_x, support_y, query_x, query_y)

        Returns:
            Perte moyenne sur le batch
        """
        batch_loss = 0.0
        self.meta_optimizer.zero_grad()

        for support_x, support_y, query_x, query_y in tasks:
            # Déplacer les données vers le bon device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Adapter le modèle à cette tâche
            adapted_model = self.adapt(support_x, support_y)

            # Évaluer sur l'ensemble query
            predictions = adapted_model(query_x)
            task_loss = self.loss_fn(predictions, query_y)
            batch_loss += task_loss

            # Rétropropager la perte pour cette tâche
            task_loss.backward()

        # Moyenne de la perte sur le batch
        batch_loss /= len(tasks)

        # Mettre à jour les paramètres du modèle méta-appris
        self.meta_optimizer.step()

        return batch_loss.item()

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle méta-appris.

        Args:
            path: Chemin où sauvegarder le modèle
        """
        save_path = MODELS_DIR / path if not path.startswith("/") else path
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "inner_lr": self.inner_lr,
                "outer_lr": self.outer_lr,
                "inner_steps": self.inner_steps,
                "first_order": self.first_order,
            },
            save_path,
        )
        logger.info(f"Modèle MAML sauvegardé: {save_path}")

    def load(self, path: str) -> None:
        """
        Charge un modèle méta-appris.

        Args:
            path: Chemin du modèle à charger
        """
        load_path = MODELS_DIR / path if not path.startswith("/") else path
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.inner_lr = checkpoint["inner_lr"]
        self.outer_lr = checkpoint["outer_lr"]
        self.inner_steps = checkpoint["inner_steps"]
        self.first_order = checkpoint["first_order"]
        logger.info(f"Modèle MAML chargé: {load_path}")
