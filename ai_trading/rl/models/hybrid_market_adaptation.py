"""
Module hybride combinant meta-learning et adaptation de domaine pour le trading inter-marchés.

Cette implémentation fusionne les avantages de:
- MAML (Model-Agnostic Meta-Learning) pour l'adaptation rapide
- Adaptation de domaine pour réduire l'écart entre distributions de marchés
"""

import logging
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from ai_trading.config import MODELS_DIR
from ai_trading.rl.models.meta_learning import MAML
from ai_trading.rl.models.transfer_learning import DomainAdaptation

logger = logging.getLogger(__name__)


class HybridMarketAdaptation:
    """
    Approche hybride combinant meta-learning et adaptation de domaine.

    Cette classe permet d'adapter efficacement un modèle entre différents marchés
    en utilisant à la fois les capacités d'adaptation rapide du meta-learning et
    l'alignement des distributions fournies par l'adaptation de domaine.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_inner_lr: float = 0.01,
        meta_outer_lr: float = 0.001,
        meta_inner_steps: int = 3,
        domain_lambda: float = 0.1,
        adaptation_type: str = "dann",  # "dann", "coral", "mmd"
        first_order: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise le modèle hybride.

        Args:
            model: Modèle de base à adapter
            meta_inner_lr: Taux d'apprentissage pour l'adaptation rapide (inner loop)
            meta_outer_lr: Taux d'apprentissage pour le méta-apprentissage (outer loop)
            meta_inner_steps: Nombre d'étapes d'adaptation rapide
            domain_lambda: Poids du terme d'adaptation de domaine
            adaptation_type: Type d'adaptation de domaine
            first_order: Si True, utilise l'approximation au premier ordre pour MAML
            device: Appareil sur lequel exécuter les calculs
        """
        self.device = device
        self.model = model.to(device)

        # Initialiser MAML
        self.maml = MAML(
            model=model,
            inner_lr=meta_inner_lr,
            outer_lr=meta_outer_lr,
            inner_steps=meta_inner_steps,
            first_order=first_order,
            device=device,
        )

        # Initialiser l'adaptation de domaine
        self.domain_adaptation = DomainAdaptation(
            source_model=model,
            adaptation_type=adaptation_type,
            lambda_param=domain_lambda,
            device=device,
        )

        # Paramètres pour suivre la meilleure performance
        self.best_model_state = None
        self.best_val_performance = float("inf")

        logger.info(
            f"Modèle hybride initialisé: meta_inner_lr={meta_inner_lr}, "
            f"meta_outer_lr={meta_outer_lr}, domain_lambda={domain_lambda}, "
            f"adaptation_type={adaptation_type}"
        )

    def adapt_to_market(
        self,
        source_market_data: Tuple[torch.Tensor, torch.Tensor],
        target_market_data: Tuple[torch.Tensor, torch.Tensor],
        adaptation_steps: int = 20,
        target_support_size: int = 10,
    ) -> nn.Module:
        """
        Adapte le modèle du marché source au marché cible.

        Args:
            source_market_data: Tuple (x, y) avec données du marché source
            target_market_data: Tuple (x, y) avec données du marché cible
            adaptation_steps: Nombre d'étapes d'adaptation
            target_support_size: Nombre d'exemples du marché cible à utiliser pour l'adaptation

        Returns:
            Modèle adapté au marché cible
        """
        source_x, source_y = source_market_data
        target_x, target_y = target_market_data

        # Déplacer les données vers le bon device
        source_x = source_x.to(self.device)
        source_y = source_y.to(self.device)
        target_x = target_x.to(self.device)
        target_y = target_y.to(self.device)

        # 1. Appliquer d'abord l'adaptation de domaine
        logger.info("Phase 1: Adaptation de domaine entre marchés")
        self.domain_adaptation.source_model = (
            self.model
        )  # Assurer l'utilisation du modèle actuel

        for step in range(adaptation_steps // 2):
            metrics = self.domain_adaptation.train_step(
                source_data=source_x, source_labels=source_y, target_data=target_x
            )

            if step % 5 == 0:
                logger.info(
                    f"Étape {step+1}/{adaptation_steps//2} - "
                    f"Task Loss: {metrics['task_loss']:.6f}, "
                    f"Domain Loss: {metrics['domain_loss']:.6f}"
                )

        # Mettre à jour le modèle principal avec le modèle adapté
        self.model.load_state_dict(self.domain_adaptation.source_model.state_dict())
        self.maml.model.load_state_dict(self.model.state_dict())

        # 2. Ensuite, appliquer l'adaptation rapide avec quelques exemples du marché cible
        logger.info("Phase 2: Meta-adaptation au marché cible")

        # Sélectionner un sous-ensemble du marché cible pour l'adaptation
        indices = torch.randperm(len(target_x))[:target_support_size]
        support_x = target_x[indices]
        support_y = target_y[indices]

        # Adapter avec MAML
        adapted_model = self.maml.adapt(
            support_x, support_y, steps=self.maml.inner_steps
        )

        return adapted_model

    def meta_domain_train(
        self,
        task_generator: Callable,
        num_tasks: int,
        num_epochs: int,
        support_size: int,
        query_size: int,
        domain_weight: float = 0.3,
        batch_size: int = 4,
    ) -> Dict[str, List[float]]:
        """
        Entraîne le modèle avec une combinaison de meta-learning et adaptation de domaine.

        Args:
            task_generator: Fonction qui génère des tâches
            num_tasks: Nombre de tâches pour l'entraînement
            num_epochs: Nombre d'epochs d'entraînement
            support_size: Taille des ensembles de support
            query_size: Taille des ensembles de query
            domain_weight: Poids relatif de l'adaptation de domaine vs meta-learning
            batch_size: Taille des batchs de tâches

        Returns:
            Historique d'entraînement
        """
        history = {"meta_loss": [], "domain_loss": [], "total_loss": []}

        for epoch in range(num_epochs):
            epoch_meta_loss = 0.0
            epoch_domain_loss = 0.0
            epoch_total_loss = 0.0

            # Générer des tâches pour cet epoch
            tasks = []
            for _ in range(num_tasks):
                task = task_generator(support_size, query_size)
                tasks.append(task)

            # Traiter les tâches par batch
            for i in range(0, num_tasks, batch_size):
                batch_tasks = tasks[i : i + batch_size]

                # 1. Étape de meta-learning
                self.maml.meta_optimizer.zero_grad()
                meta_batch_loss = 0.0

                for task in batch_tasks:
                    support_x, support_y, query_x, query_y = task

                    # Adapter le modèle à cette tâche
                    adapted_model = self.maml.adapt(support_x, support_y)

                    # Évaluer sur l'ensemble query
                    query_x = query_x.to(self.device)
                    query_y = query_y.to(self.device)
                    predictions = adapted_model(query_x)
                    task_loss = self.maml.loss_fn(predictions, query_y)

                    meta_batch_loss += task_loss / len(batch_tasks)

                # Rétropropager la perte meta-learning
                meta_batch_loss.backward()
                self.maml.meta_optimizer.step()

                # 2. Étape d'adaptation de domaine
                # Pour chaque tâche, considérer le support set comme source et query set comme cible
                domain_batch_loss = 0.0

                if domain_weight > 0:
                    for task in batch_tasks:
                        support_x, support_y, query_x, query_y = task

                        # Considérer support comme source et query comme cible
                        domain_metrics = self.domain_adaptation.train_step(
                            source_data=support_x,
                            source_labels=support_y,
                            target_data=query_x,
                        )

                        domain_batch_loss += domain_metrics["domain_loss"] / len(
                            batch_tasks
                        )

                # 3. Synchroniser les modèles
                # Mettre à jour le modèle MAML avec l'état du modèle d'adaptation de domaine
                alpha = 1.0 - domain_weight
                beta = domain_weight

                # Interpolation des poids entre les deux modèles
                with torch.no_grad():
                    for p_maml, p_domain in zip(
                        self.maml.model.parameters(),
                        self.domain_adaptation.source_model.parameters(),
                    ):
                        if p_maml.requires_grad and p_domain.requires_grad:
                            p_shared = alpha * p_maml + beta * p_domain
                            p_maml.copy_(p_shared)
                            p_domain.copy_(p_shared)

                # Mettre à jour les pertes pour cet epoch
                epoch_meta_loss += meta_batch_loss.item()
                epoch_domain_loss += (
                    domain_batch_loss
                    if isinstance(domain_batch_loss, float)
                    else domain_batch_loss.item()
                )
                epoch_total_loss += epoch_meta_loss + epoch_domain_loss

            # Calculer les moyennes pour l'epoch
            epoch_meta_loss /= num_tasks / batch_size
            epoch_domain_loss /= num_tasks / batch_size
            epoch_total_loss = epoch_meta_loss + epoch_domain_loss

            # Enregistrer dans l'historique
            history["meta_loss"].append(epoch_meta_loss)
            history["domain_loss"].append(epoch_domain_loss)
            history["total_loss"].append(epoch_total_loss)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Meta Loss: {epoch_meta_loss:.6f}, "
                f"Domain Loss: {epoch_domain_loss:.6f}, "
                f"Total Loss: {epoch_total_loss:.6f}"
            )

        return history

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module = nn.MSELoss(),
    ) -> float:
        """
        Évalue le modèle sur un ensemble de données.

        Args:
            data_loader: DataLoader pour les données d'évaluation
            criterion: Fonction de perte à utiliser

        Returns:
            Perte moyenne sur l'ensemble de données
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()

        return total_loss / len(data_loader)

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle hybride.

        Args:
            path: Chemin où sauvegarder le modèle
        """
        save_path = MODELS_DIR / path if not path.startswith("/") else path
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "maml_state": {
                    "inner_lr": self.maml.inner_lr,
                    "outer_lr": self.maml.outer_lr,
                    "inner_steps": self.maml.inner_steps,
                    "first_order": self.maml.first_order,
                },
                "domain_state": {
                    "adaptation_type": self.domain_adaptation.adaptation_type,
                    "lambda_param": self.domain_adaptation.lambda_param,
                    "discriminator_state_dict": (
                        self.domain_adaptation.domain_discriminator.state_dict()
                        if self.domain_adaptation.domain_discriminator
                        else None
                    ),
                },
            },
            save_path,
        )
        logger.info(f"Modèle hybride sauvegardé: {save_path}")

    def load(self, path: str) -> None:
        """
        Charge un modèle hybride.

        Args:
            path: Chemin du modèle à charger
        """
        load_path = MODELS_DIR / path if not path.startswith("/") else path
        checkpoint = torch.load(load_path, map_location=self.device)

        # Charger l'état du modèle
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Mettre à jour MAML
        maml_state = checkpoint["maml_state"]
        self.maml.model.load_state_dict(checkpoint["model_state_dict"])
        self.maml.inner_lr = maml_state["inner_lr"]
        self.maml.outer_lr = maml_state["outer_lr"]
        self.maml.inner_steps = maml_state["inner_steps"]
        self.maml.first_order = maml_state["first_order"]

        # Mettre à jour Domain Adaptation
        domain_state = checkpoint["domain_state"]
        self.domain_adaptation.source_model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.domain_adaptation.adaptation_type = domain_state["adaptation_type"]
        self.domain_adaptation.lambda_param = domain_state["lambda_param"]

        if (
            self.domain_adaptation.domain_discriminator
            and domain_state["discriminator_state_dict"]
        ):
            self.domain_adaptation.domain_discriminator.load_state_dict(
                domain_state["discriminator_state_dict"]
            )

        logger.info(f"Modèle hybride chargé: {load_path}")
