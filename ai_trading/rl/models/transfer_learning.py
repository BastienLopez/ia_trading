"""
Module de transfer learning inter-marchés pour l'adaptation entre différents actifs financiers.

Ce module implémente:
- Techniques de fine-tuning pour adapter les modèles préentraînés à de nouveaux marchés
- Feature mapping pour aligner les caractéristiques entre différents marchés
- Domain adaptation pour gérer les différences de distribution entre marchés
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_trading.config import MODELS_DIR

logger = logging.getLogger(__name__)


class MarketTransferLearning:
    """
    Classe pour le transfer learning entre différents marchés financiers.

    Cette approche permet d'adapter un modèle entraîné sur un marché source
    à un nouveau marché cible en préservant les connaissances générales tout
    en adaptant le modèle aux spécificités du nouveau marché.
    """

    def __init__(
        self,
        base_model: nn.Module,
        fine_tune_layers: Optional[List[str]] = None,
        learning_rate: float = 0.0001,
        feature_mapping: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise le système de transfer learning.

        Args:
            base_model: Modèle préentraîné sur le marché source
            fine_tune_layers: Liste des noms des couches à fine-tuner (None = toutes)
            learning_rate: Taux d'apprentissage pour le fine-tuning
            feature_mapping: Si True, ajoute une couche d'adaptation pour mapper les features
            device: Appareil sur lequel exécuter les calculs ("cuda" ou "cpu")
        """
        self.base_model = base_model.to(device)
        self.fine_tune_layers = fine_tune_layers
        self.learning_rate = learning_rate
        self.feature_mapping = feature_mapping
        self.device = device

        # Geler les couches qui ne sont pas fine-tunées
        if fine_tune_layers is not None:
            self._freeze_layers(fine_tune_layers)

        # Ajouter une couche d'adaptation si nécessaire
        self.feature_mapper = None
        if feature_mapping:
            # Détecter la dimension d'entrée du modèle
            input_dim = self._detect_input_dim()
            if input_dim is not None:
                self.feature_mapper = nn.Linear(input_dim, input_dim).to(device)
                logger.info(f"Couche d'adaptation créée avec dimension {input_dim}")

        # Initialiser l'optimiseur
        self._setup_optimizer()

        logger.info(
            f"Transfer Learning initialisé avec device={device}, learning_rate={learning_rate}"
        )

    def _detect_input_dim(self) -> Optional[int]:
        """
        Détecte la dimension d'entrée du modèle.

        Returns:
            Dimension d'entrée ou None si impossible à détecter
        """
        # Essayer de trouver la première couche linéaire ou conv
        for module in self.base_model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                return module.in_channels

        logger.warning("Impossible de détecter automatiquement la dimension d'entrée")
        return None

    def _freeze_layers(self, layers_to_fine_tune: List[str]) -> None:
        """
        Gèle toutes les couches sauf celles spécifiées dans layers_to_fine_tune.

        Args:
            layers_to_fine_tune: Liste des noms des couches à fine-tuner
        """
        # Par défaut, geler tous les paramètres
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Dégeler les couches spécifiées
        for name, param in self.base_model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_fine_tune):
                param.requires_grad = True
                logger.debug(f"Couche dégelée pour fine-tuning: {name}")

    def _setup_optimizer(self) -> None:
        """
        Configure l'optimiseur pour le fine-tuning.
        """
        # Collecter les paramètres à optimiser
        parameters = []

        # Ajouter les paramètres du modèle de base qui sont dégelés
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                parameters.append(param)

        # Ajouter les paramètres du feature mapper si présent
        if self.feature_mapper is not None:
            parameters.extend(self.feature_mapper.parameters())

        # Créer l'optimiseur
        if parameters:
            self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
            logger.info(
                f"Optimiseur configuré avec {len(parameters)} groupes de paramètres"
            )
        else:
            logger.warning("Aucun paramètre à optimiser")
            self.optimizer = None

    def fine_tune(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        criterion: nn.Module = nn.MSELoss(),
        early_stopping_patience: int = 5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune le modèle sur les données du marché cible.

        Args:
            train_loader: DataLoader pour les données d'entraînement
            val_loader: DataLoader pour les données de validation
            epochs: Nombre d'epochs d'entraînement
            criterion: Fonction de perte à utiliser
            early_stopping_patience: Patience pour l'early stopping
            scheduler_factor: Facteur de réduction du learning rate
            scheduler_patience: Patience pour le scheduler

        Returns:
            Historique d'entraînement avec les pertes train et val
        """
        if self.optimizer is None:
            logger.error("Aucun paramètre à optimiser, impossible de fine-tuner")
            return {"train_loss": [], "val_loss": []}

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        # Scheduler pour ajuster le learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )

        for epoch in range(epochs):
            # Entraînement
            self.base_model.train()
            if self.feature_mapper:
                self.feature_mapper.train()

            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # Appliquer le feature mapping si nécessaire
                if self.feature_mapper:
                    data = self.feature_mapper(data)

                # Forward pass
                output = self.base_model(data)
                loss = criterion(output, target)

                # Backward pass et optimisation
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping à l'epoch {epoch+1}")
                    break

                scheduler.step(val_loss)
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f}")

        # Restaurer le meilleur modèle
        self._load_best_model()

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
        self.base_model.eval()
        if self.feature_mapper:
            self.feature_mapper.eval()

        total_loss = 0.0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Appliquer le feature mapping si nécessaire
                if self.feature_mapper:
                    data = self.feature_mapper(data)

                # Forward pass
                output = self.base_model(data)
                loss = criterion(output, target)

                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fait des prédictions avec le modèle fine-tuné.

        Args:
            data: Données d'entrée

        Returns:
            Prédictions du modèle
        """
        self.base_model.eval()
        if self.feature_mapper:
            self.feature_mapper.eval()

        data = data.to(self.device)

        with torch.no_grad():
            # Appliquer le feature mapping si nécessaire
            if self.feature_mapper:
                data = self.feature_mapper(data)

            # Forward pass
            output = self.base_model(data)

        return output

    def _save_best_model(self) -> None:
        """
        Sauvegarde l'état actuel comme le meilleur modèle.
        """
        self.best_model_state = {
            "base_model": self.base_model.state_dict(),
            "feature_mapper": (
                self.feature_mapper.state_dict() if self.feature_mapper else None
            ),
        }

    def _load_best_model(self) -> None:
        """
        Charge le meilleur modèle sauvegardé.
        """
        if hasattr(self, "best_model_state"):
            self.base_model.load_state_dict(self.best_model_state["base_model"])
            if self.feature_mapper and self.best_model_state["feature_mapper"]:
                self.feature_mapper.load_state_dict(
                    self.best_model_state["feature_mapper"]
                )

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle fine-tuné et le feature mapper.

        Args:
            path: Chemin où sauvegarder le modèle
        """
        save_path = MODELS_DIR / path if not path.startswith("/") else path
        torch.save(
            {
                "base_model_state_dict": self.base_model.state_dict(),
                "feature_mapper_state_dict": (
                    self.feature_mapper.state_dict() if self.feature_mapper else None
                ),
                "fine_tune_layers": self.fine_tune_layers,
            },
            save_path,
        )
        logger.info(f"Modèle fine-tuné sauvegardé: {save_path}")

    def load(self, path: str) -> None:
        """
        Charge un modèle fine-tuné.

        Args:
            path: Chemin du modèle à charger
        """
        load_path = MODELS_DIR / path if not path.startswith("/") else path
        checkpoint = torch.load(load_path, map_location=self.device)
        self.base_model.load_state_dict(checkpoint["base_model_state_dict"])
        if self.feature_mapper and checkpoint["feature_mapper_state_dict"]:
            self.feature_mapper.load_state_dict(checkpoint["feature_mapper_state_dict"])
        self.fine_tune_layers = checkpoint["fine_tune_layers"]
        logger.info(f"Modèle fine-tuné chargé: {load_path}")


class DomainAdaptation:
    """
    Adaptation de domaine entre différents marchés financiers.

    Cette classe implémente des techniques pour réduire l'écart entre les distributions
    de différents marchés, permettant ainsi un meilleur transfert d'apprentissage.
    """

    def __init__(
        self,
        source_model: nn.Module,
        adaptation_type: str = "dann",  # "dann", "coral", "mmd"
        lambda_param: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise le système d'adaptation de domaine.

        Args:
            source_model: Modèle entraîné sur le domaine source
            adaptation_type: Type d'adaptation de domaine à utiliser
            lambda_param: Poids du terme d'adaptation de domaine
            device: Appareil sur lequel exécuter les calculs
        """
        self.source_model = source_model.to(device)
        self.adaptation_type = adaptation_type
        self.lambda_param = lambda_param
        self.device = device

        # Créer le discriminateur de domaine si nécessaire (pour DANN)
        self.domain_discriminator = None
        if adaptation_type == "dann":
            # Détecter la dimension de sortie du modèle source
            output_dim = self._detect_output_dim()
            if output_dim is not None:
                self.domain_discriminator = nn.Sequential(
                    nn.Linear(output_dim, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 1),
                    nn.Sigmoid(),
                ).to(device)
                logger.info(
                    f"Discriminateur de domaine créé avec dimension d'entrée {output_dim}"
                )

        # Initialiser l'optimiseur
        self._setup_optimizer()

        logger.info(
            f"Domain Adaptation initialisée: type={adaptation_type}, lambda={lambda_param}"
        )

    def _detect_output_dim(self) -> Optional[int]:
        """
        Détecte la dimension de sortie du modèle source.

        Returns:
            Dimension de sortie ou None si impossible à détecter
        """
        # Chercher la dernière couche linéaire
        output_dim = None
        for module in self.source_model.modules():
            if isinstance(module, nn.Linear):
                output_dim = module.out_features

        if output_dim is None:
            logger.warning(
                "Impossible de détecter automatiquement la dimension de sortie"
            )

        return output_dim

    def _setup_optimizer(self) -> None:
        """
        Configure les optimiseurs pour l'adaptation de domaine.
        """
        # Optimiseur pour le modèle principal
        self.model_optimizer = torch.optim.Adam(
            self.source_model.parameters(), lr=0.0001
        )

        # Optimiseur pour le discriminateur de domaine si présent
        if self.domain_discriminator:
            self.discriminator_optimizer = torch.optim.Adam(
                self.domain_discriminator.parameters(), lr=0.0001
            )

    def _coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcule la perte CORAL (CORrelation ALignment) entre domaines.

        Args:
            source: Features du domaine source
            target: Features du domaine cible

        Returns:
            Perte CORAL
        """
        d = source.size(1)

        # Centrer les données
        source_centered = source - torch.mean(source, dim=0)
        target_centered = target - torch.mean(target, dim=0)

        # Calculer les matrices de covariance
        source_cov = torch.matmul(source_centered.t(), source_centered) / (
            source.size(0) - 1
        )
        target_cov = torch.matmul(target_centered.t(), target_centered) / (
            target.size(0) - 1
        )

        # Calculer la perte CORAL
        loss = torch.norm(source_cov - target_cov, p="fro")
        return loss / (4 * d * d)

    def _mmd_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcule la perte MMD (Maximum Mean Discrepancy) entre domaines.

        Args:
            source: Features du domaine source
            target: Features du domaine cible

        Returns:
            Perte MMD
        """

        def gaussian_kernel(x, y, sigma=1.0):
            size_x = x.size(0)
            size_y = y.size(0)
            dim = x.size(1)

            x = x.unsqueeze(1)
            y = y.unsqueeze(0)

            tiled_x = x.expand(size_x, size_y, dim)
            tiled_y = y.expand(size_x, size_y, dim)

            kernel_input = torch.pow(tiled_x - tiled_y, 2).sum(2) / (2 * sigma**2)
            return torch.exp(-kernel_input)

        # Utiliser plusieurs valeurs de sigma pour un meilleur résultat
        sigmas = [1.0, 5.0, 10.0]

        xx_ker = 0
        xy_ker = 0
        yy_ker = 0

        for sigma in sigmas:
            xx_ker += gaussian_kernel(source, source, sigma)
            xy_ker += gaussian_kernel(source, target, sigma)
            yy_ker += gaussian_kernel(target, target, sigma)

        return torch.mean(xx_ker) - 2 * torch.mean(xy_ker) + torch.mean(yy_ker)

    def train_step(
        self,
        source_data: torch.Tensor,
        source_labels: torch.Tensor,
        target_data: torch.Tensor,
        task_criterion: nn.Module = nn.MSELoss(),
    ) -> Dict[str, float]:
        """
        Effectue une étape d'entraînement avec adaptation de domaine.

        Args:
            source_data: Données du domaine source
            source_labels: Étiquettes du domaine source
            target_data: Données du domaine cible (sans étiquettes)
            task_criterion: Fonction de perte pour la tâche principale

        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        metrics = {}

        # Déplacer les données vers le bon device
        source_data = source_data.to(self.device)
        source_labels = source_labels.to(self.device)
        target_data = target_data.to(self.device)

        # Créer les étiquettes de domaine (1 pour source, 0 pour cible)
        source_domain_labels = torch.ones(source_data.size(0), 1).to(self.device)
        target_domain_labels = torch.zeros(target_data.size(0), 1).to(self.device)

        # Réinitialiser les gradients
        self.model_optimizer.zero_grad()
        if self.domain_discriminator:
            self.discriminator_optimizer.zero_grad()

        # Forward pass pour les données source
        source_preds = self.source_model(source_data)
        task_loss = task_criterion(source_preds, source_labels)
        metrics["task_loss"] = task_loss.item()

        # Perte d'adaptation de domaine
        domain_loss = 0.0

        if self.adaptation_type == "dann":
            # Domain-Adversarial Neural Network (DANN)
            if self.domain_discriminator:
                # Forward pass pour les données cible
                target_preds = self.source_model(target_data)

                # Prédictions du discriminateur
                source_domain_preds = self.domain_discriminator(source_preds.detach())
                target_domain_preds = self.domain_discriminator(target_preds.detach())

                # Perte du discriminateur
                discriminator_loss = F.binary_cross_entropy(
                    source_domain_preds, source_domain_labels
                ) + F.binary_cross_entropy(target_domain_preds, target_domain_labels)

                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                metrics["discriminator_loss"] = discriminator_loss.item()

                # Réinitialiser les gradients pour le modèle principal
                self.model_optimizer.zero_grad()

                # Forward pass à nouveau (maintenant pour la perte adversariale)
                source_preds = self.source_model(source_data)
                target_preds = self.source_model(target_data)

                # Prédictions du discriminateur
                source_domain_preds = self.domain_discriminator(source_preds)
                target_domain_preds = self.domain_discriminator(target_preds)

                # Perte adversariale (inverser les étiquettes pour confondre le discriminateur)
                domain_loss = F.binary_cross_entropy(
                    source_domain_preds, target_domain_labels
                ) + F.binary_cross_entropy(target_domain_preds, source_domain_labels)

        elif self.adaptation_type == "coral":
            # CORAL (CORrelation ALignment)
            source_features = source_preds
            target_features = self.source_model(target_data)
            domain_loss = self._coral_loss(source_features, target_features)

        elif self.adaptation_type == "mmd":
            # MMD (Maximum Mean Discrepancy)
            source_features = source_preds
            target_features = self.source_model(target_data)
            domain_loss = self._mmd_loss(source_features, target_features)

        # Combiner les pertes
        metrics["domain_loss"] = domain_loss.item()
        total_loss = task_loss + self.lambda_param * domain_loss
        metrics["total_loss"] = total_loss.item()

        # Backward pass et optimisation
        total_loss.backward()
        self.model_optimizer.step()

        return metrics

    def train(
        self,
        source_loader: torch.utils.data.DataLoader,
        target_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        task_criterion: nn.Module = nn.MSELoss(),
    ) -> Dict[str, List[float]]:
        """
        Entraîne le modèle avec adaptation de domaine.

        Args:
            source_loader: DataLoader pour les données source (avec étiquettes)
            target_loader: DataLoader pour les données cible (sans étiquettes)
            val_loader: DataLoader pour la validation (données cible avec étiquettes)
            epochs: Nombre d'epochs d'entraînement
            task_criterion: Fonction de perte pour la tâche principale

        Returns:
            Historique d'entraînement
        """
        history = {"task_loss": [], "domain_loss": [], "total_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Boucle d'entraînement
            self.source_model.train()
            if self.domain_discriminator:
                self.domain_discriminator.train()

            # Créer un itérateur pour les données cible (possiblement plus petites)
            target_iter = iter(target_loader)

            # Métriques moyennes sur l'epoch
            epoch_metrics = {"task_loss": 0.0, "domain_loss": 0.0, "total_loss": 0.0}

            # Parcourir les données source
            for batch_idx, (source_data, source_labels) in enumerate(source_loader):
                try:
                    # Obtenir un batch de données cible
                    target_data, _ = next(target_iter)
                except StopIteration:
                    # Réinitialiser l'itérateur si on a épuisé les données cible
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)

                # Effectuer une étape d'entraînement
                batch_metrics = self.train_step(
                    source_data, source_labels, target_data, task_criterion
                )

                # Mettre à jour les métriques moyennes
                for key, value in batch_metrics.items():
                    epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

            # Calculer les moyennes
            for key in epoch_metrics.keys():
                epoch_metrics[key] /= len(source_loader)
                history[key].append(epoch_metrics[key])

            # Validation si disponible
            if val_loader:
                val_loss = self.evaluate(val_loader, task_criterion)
                history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Task Loss: {epoch_metrics['task_loss']:.6f}, "
                    f"Domain Loss: {epoch_metrics['domain_loss']:.6f}, "
                    f"Total Loss: {epoch_metrics['total_loss']:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Task Loss: {epoch_metrics['task_loss']:.6f}, "
                    f"Domain Loss: {epoch_metrics['domain_loss']:.6f}, "
                    f"Total Loss: {epoch_metrics['total_loss']:.6f}"
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
        self.source_model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.source_model(data)
                loss = criterion(output, target)

                total_loss += loss.item()

        return total_loss / len(data_loader)

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle adapté et le discriminateur.

        Args:
            path: Chemin où sauvegarder le modèle
        """
        save_path = MODELS_DIR / path if not path.startswith("/") else path
        torch.save(
            {
                "model_state_dict": self.source_model.state_dict(),
                "discriminator_state_dict": (
                    self.domain_discriminator.state_dict()
                    if self.domain_discriminator
                    else None
                ),
                "adaptation_type": self.adaptation_type,
                "lambda_param": self.lambda_param,
            },
            save_path,
        )
        logger.info(f"Modèle adapté sauvegardé: {save_path}")

    def load(self, path: str) -> None:
        """
        Charge un modèle adapté.

        Args:
            path: Chemin du modèle à charger
        """
        load_path = MODELS_DIR / path if not path.startswith("/") else path
        checkpoint = torch.load(load_path, map_location=self.device)
        self.source_model.load_state_dict(checkpoint["model_state_dict"])
        if self.domain_discriminator and checkpoint["discriminator_state_dict"]:
            self.domain_discriminator.load_state_dict(
                checkpoint["discriminator_state_dict"]
            )
        self.adaptation_type = checkpoint["adaptation_type"]
        self.lambda_param = checkpoint["lambda_param"]
        logger.info(f"Modèle adapté chargé: {load_path}")
