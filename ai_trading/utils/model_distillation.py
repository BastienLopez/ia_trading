from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader


class ModelDistillation:
    def __init__(self, teacher_model, student_model, device):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.scheduler = None

    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_step(self, inputs, targets):
        # Implementation of train_step method
        pass

    def validate(self, val_loader):
        # Implementation of validate method
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        log_interval: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Entraîne le modèle élève avec distillation.

        Args:
            train_loader: DataLoader pour les données d'entraînement
            val_loader: DataLoader optionnel pour les données de validation
            epochs: Nombre d'époques d'entraînement
            early_stopping_patience: Patience pour l'arrêt anticipé
            log_interval: Fréquence des logs en nombre de batchs
            save_path: Chemin pour sauvegarder le meilleur modèle élève

        Returns:
            Historique des métriques d'entraînement et de validation
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Début de l'entraînement par distillation sur {self.device}")
        logger.info(
            f"Taille de l'enseignant: {self._count_parameters(self.teacher_model):,} paramètres"
        )
        logger.info(
            f"Taille de l'élève: {self._count_parameters(self.student_model):,} paramètres"
        )
        logger.info(
            f"Ratio de compression: {self._count_parameters(self.teacher_model) / self._count_parameters(self.student_model):.2f}x"
        )

        for epoch in range(epochs):
            # Entraînement
            self.student_model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                metrics = self.train_step(inputs, targets)
                epoch_loss += metrics["loss"]
                epoch_accuracy += metrics["accuracy"]

                if batch_idx % log_interval == 0:
                    logger.info(
                        f"Époque {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                        f"Loss: {metrics['loss']:.4f} Accuracy: {metrics['accuracy']:.4f}"
                    )

            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_accuracy = epoch_accuracy / len(train_loader)

            history["train_loss"].append(avg_train_loss)
            history["train_accuracy"].append(avg_train_accuracy)

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])

                if "student_accuracy" in val_metrics:
                    history["val_accuracy"].append(val_metrics["student_accuracy"])
                    logger.info(
                        f"Époque {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, "
                        f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['student_accuracy']:.4f}, "
                        f"Teacher Acc: {val_metrics['teacher_accuracy']:.4f}"
                    )
                else:
                    history["val_accuracy"].append(
                        -val_metrics["student_mse"]
                    )  # Négatif car nous voulons maximiser
                    logger.info(
                        f"Époque {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train MSE: {avg_train_accuracy:.4f}, "
                        f"Val Loss: {val_metrics['val_loss']:.4f}, Val MSE: {val_metrics['student_mse']:.4f}, "
                        f"Teacher MSE: {val_metrics['teacher_mse']:.4f}"
                    )

                # Early stopping
                if early_stopping_patience is not None:
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        patience_counter = 0

                        # Sauvegarder le meilleur modèle
                        if save_path is not None:
                            self.save_student_model(save_path)
                            logger.info(f"Meilleur modèle sauvegardé à {save_path}")
                    else:
                        patience_counter += 1
                        logger.info(
                            f"EarlyStopping: {patience_counter}/{early_stopping_patience}"
                        )

                        if patience_counter >= early_stopping_patience:
                            logger.info(
                                f"Early stopping déclenché à l'époque {epoch+1}"
                            )
                            break
            else:
                logger.info(
                    f"Époque {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}"
                )

            # Mise à jour du scheduler si présent
            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("Entraînement terminé")
        return history

    def save_student_model(self, path: str) -> None:
        """
        Sauvegarde le modèle élève.

        Args:
            path: Chemin où sauvegarder le modèle
        """
        if path is None:
            logger.warning("Aucun chemin fourni pour sauvegarder le modèle élève")
            return

        # Créer le répertoire parent si nécessaire
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self.student_model.state_dict(), path)
        logger.info(f"Modèle élève sauvegardé à {path}")
