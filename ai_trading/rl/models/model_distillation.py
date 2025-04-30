import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_trading.config import MODELS_DIR
from ai_trading.rl.models.temporal_transformer import FinancialTemporalTransformer


class DistillationLoss(nn.Module):
    """
    Fonction de perte pour la distillation de connaissances.
    Combine la perte de prédiction (MSE) avec la distillation de connaissances (KL div).
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        """
        Args:
            alpha: Poids entre la perte dure (MSE) et la perte souple (KL)
            temperature: Température pour le softmax dans la distillation
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        targets: torch.Tensor,
        student_attention: List[torch.Tensor] = None,
        teacher_attention: List[torch.Tensor] = None,
        attention_weight: float = 0.3,
    ) -> torch.Tensor:
        """
        Calcule la perte de distillation combinée.

        Args:
            student_preds: Prédictions du modèle étudiant
            teacher_preds: Prédictions du modèle enseignant
            targets: Cibles réelles
            student_attention: Poids d'attention du modèle étudiant (optionnel)
            teacher_attention: Poids d'attention du modèle enseignant (optionnel)
            attention_weight: Poids pour la perte d'attention (si utilisée)

        Returns:
            Perte totale
        """
        # Perte dure par rapport aux cibles réelles (MSE)
        hard_loss = self.mse_loss(student_preds, targets)

        # Normaliser les prédictions pour la perte KL (les deux modèles prédisent des valeurs continues)
        # Déterminer la dimension appropriée pour le softmax
        softmax_dim = 1 if student_preds.dim() > 1 else 0

        # Ajouter une dimension si nécessaire pour le cas des tenseurs 1D
        if student_preds.dim() == 1:
            student_preds = student_preds.unsqueeze(1)
            teacher_preds = teacher_preds.unsqueeze(1)
            softmax_dim = 1

        # Appliquer softmax avec la dimension appropriée
        s_logits = F.log_softmax(student_preds / self.temperature, dim=softmax_dim)
        t_logits = F.softmax(teacher_preds / self.temperature, dim=softmax_dim)

        # Perte souple de distillation
        soft_loss = self.kl_div_loss(s_logits, t_logits) * (self.temperature**2)

        # Combiner les pertes
        combined_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        # Ajouter la distillation d'attention si les tenseurs d'attention sont fournis
        if student_attention is not None and teacher_attention is not None:
            attention_loss = 0.0

            # Pour chaque couche, calculer la perte de distillation d'attention
            for s_attn, t_attn in zip(student_attention, teacher_attention):
                # Vérifier et ajuster les dimensions des tenseurs d'attention
                if s_attn.dim() != t_attn.dim():
                    # Ajuster les dimensions si nécessaire
                    if s_attn.dim() < t_attn.dim():
                        # Ajouter des dimensions à s_attn
                        while s_attn.dim() < t_attn.dim():
                            s_attn = s_attn.unsqueeze(-1)
                    else:
                        # Ajouter des dimensions à t_attn
                        while t_attn.dim() < s_attn.dim():
                            t_attn = t_attn.unsqueeze(-1)

                # Normaliser les poids d'attention avec softmax
                s_attn_norm = F.log_softmax(s_attn / self.temperature, dim=-1)
                t_attn_norm = F.softmax(t_attn / self.temperature, dim=-1)

                # KL divergence pour chaque tête d'attention
                layer_attn_loss = self.kl_div_loss(s_attn_norm, t_attn_norm) * (
                    self.temperature**2
                )
                attention_loss += layer_attn_loss

            attention_loss /= len(student_attention)  # Moyenne sur toutes les couches
            combined_loss = combined_loss + attention_weight * attention_loss

        return combined_loss


class DistilledFinancialTransformer(nn.Module):
    """
    Transformer financier distillé, plus léger et plus rapide que le modèle enseignant.
    """

    def __init__(
        self,
        teacher_model: FinancialTemporalTransformer,
        reduction_factor: int = 2,
        distill_attention: bool = True,
    ):
        """
        Initialise un modèle étudiant plus léger basé sur un modèle enseignant.

        Args:
            teacher_model: Modèle enseignant à distiller
            reduction_factor: Facteur de réduction pour les dimensions du modèle
            distill_attention: Si True, distille également les mécanismes d'attention
        """
        super().__init__()

        # Récupérer les paramètres du modèle enseignant
        input_dim = teacher_model.input_projection[0].in_features
        d_model = teacher_model.input_projection[0].out_features
        nhead = teacher_model.transformer_blocks[0].self_attn.num_heads
        num_layers = len(teacher_model.transformer_blocks)

        # Réduire les dimensions pour le modèle étudiant
        student_d_model = d_model // reduction_factor
        student_nhead = max(1, nhead // reduction_factor)
        student_num_layers = max(1, num_layers // reduction_factor)
        student_dim_feedforward = (
            teacher_model.transformer_blocks[0].linear1.out_features // reduction_factor
        )

        # Créer un modèle étudiant plus léger
        self.student = FinancialTemporalTransformer(
            input_dim=input_dim,
            d_model=student_d_model,
            nhead=student_nhead,
            num_layers=student_num_layers,
            dim_feedforward=student_dim_feedforward,
            dropout=0.1,  # Généralement plus faible pour l'étudiant
            output_dim=1,
        )

        # Sauvegarder les références pour la distillation
        self.teacher = teacher_model
        self.teacher.eval()  # Mettre le modèle enseignant en mode évaluation
        self.distill_attention = distill_attention

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass du modèle étudiant.

        Args:
            src: Tenseur d'entrée de forme (batch_size, seq_len, input_dim)
            src_mask: Masque d'attention optionnel
            src_key_padding_mask: Masque de padding optionnel

        Returns:
            Tuple contenant:
            - Prédictions du modèle étudiant
            - Poids d'attention du modèle étudiant
        """
        # Vérifier le format de l'entrée et ajuster si nécessaire
        if len(src.shape) == 4:  # [batch, seq_len, autre_dim, features]
            batch_size, seq_len, other_dim, input_dim = src.shape
            # Aplatir les dimensions temporelles
            src = src.reshape(batch_size, seq_len * other_dim, input_dim)
            print(
                f"Redimensionnement de l'entrée: {batch_size}x{seq_len}x{other_dim}x{input_dim} -> {batch_size}x{seq_len * other_dim}x{input_dim}"
            )

        # Prédiction du modèle étudiant
        student_output, student_attention = self.student(
            src, src_mask, src_key_padding_mask
        )

        return student_output, student_attention

    def distill_knowledge(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        alpha: float = 0.5,
        temperature: float = 4.0,
        attention_weight: float = 0.3,
    ) -> Dict[str, float]:
        """
        Entraîne le modèle étudiant par distillation de connaissances.

        Args:
            data: Données d'entrée
            targets: Cibles réelles
            optimizer: Optimiseur pour l'étudiant
            alpha: Poids entre perte dure et perte de distillation
            temperature: Température pour le softmax dans la distillation
            attention_weight: Poids pour la distillation d'attention

        Returns:
            Dictionnaire avec les pertes
        """
        # Passer en mode entraînement
        self.student.train()

        # Générer les prédictions de l'enseignant (sans calcul de gradient)
        with torch.no_grad():
            # Vérifier le format de l'entrée et ajuster si nécessaire pour l'enseignant
            teacher_input = data
            if len(data.shape) == 4:  # [batch, seq_len, autre_dim, features]
                batch_size, seq_len, other_dim, input_dim = data.shape
                # Aplatir les dimensions temporelles
                teacher_input = data.reshape(batch_size, seq_len * other_dim, input_dim)

            teacher_preds, teacher_attention = self.teacher(teacher_input)

            # S'assurer que les poids d'attention de l'enseignant ont le bon format pour la comparaison
            # avec ceux de l'étudiant (qui a moins de couches)
            if self.distill_attention:
                num_student_layers = len(self.student.transformer_blocks)
                teacher_attention = teacher_attention[:num_student_layers]

        # Générer les prédictions de l'étudiant
        student_preds, student_attention = self.student(data)

        # S'assurer que les prédictions ont les mêmes dimensions
        if student_preds.dim() != teacher_preds.dim():
            if student_preds.dim() > teacher_preds.dim():
                teacher_preds = teacher_preds.view_as(student_preds)
            else:
                student_preds = student_preds.view_as(teacher_preds)

        # S'assurer que les cibles ont les mêmes dimensions que les prédictions
        if targets.dim() != student_preds.dim():
            targets = targets.view_as(student_preds)

        # Initialiser la fonction de perte de distillation
        distill_loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)

        # Calculer la perte
        if self.distill_attention:
            loss = distill_loss_fn(
                student_preds=student_preds,
                teacher_preds=teacher_preds,
                targets=targets,
                student_attention=student_attention,
                teacher_attention=teacher_attention,
                attention_weight=attention_weight,
            )
        else:
            loss = distill_loss_fn(
                student_preds=student_preds,
                teacher_preds=teacher_preds,
                targets=targets,
            )

        # Rétropropagation et mise à jour des poids
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculer les pertes individuelles pour le monitoring
        with torch.no_grad():
            mse_loss = F.mse_loss(student_preds, targets).item()
            teacher_mse = F.mse_loss(teacher_preds, targets).item()

        return {
            "total_loss": loss.item(),
            "student_mse": mse_loss,
            "teacher_mse": teacher_mse,
        }


def train_distilled_model(
    teacher_model: FinancialTemporalTransformer,
    train_data: torch.Tensor,
    train_targets: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    val_targets: Optional[torch.Tensor] = None,
    reduction_factor: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    alpha: float = 0.5,
    temperature: float = 4.0,
    attention_weight: float = 0.3,
    patience: int = 5,
    save_dir: Optional[str] = None,
):
    """
    Fonction complète pour entraîner un modèle distillé.

    Args:
        teacher_model: Modèle enseignant pré-entraîné
        train_data: Données d'entraînement
        train_targets: Cibles d'entraînement
        val_data: Données de validation (optionnel)
        val_targets: Cibles de validation (optionnel)
        reduction_factor: Facteur de réduction pour le modèle étudiant
        epochs: Nombre total d'époques
        batch_size: Taille des batchs
        learning_rate: Taux d'apprentissage
        alpha: Poids entre perte dure et perte de distillation
        temperature: Température pour le softmax dans la distillation
        attention_weight: Poids pour la distillation d'attention
        patience: Nombre d'époques sans amélioration avant early stopping
        save_dir: Répertoire pour sauvegarder le modèle et les métriques (optionnel)

    Returns:
        Tuple contenant:
        - Modèle étudiant entraîné
        - Historique d'entraînement
    """
    # Créer le modèle étudiant
    distilled_model = DistilledFinancialTransformer(
        teacher_model=teacher_model,
        reduction_factor=reduction_factor,
        distill_attention=True,
    )

    # Optimiseur
    optimizer = torch.optim.Adam(distilled_model.student.parameters(), lr=learning_rate)

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Historique d'entraînement
    history = {"train_loss": [], "val_loss": [], "student_mse": [], "teacher_mse": []}

    # Boucle d'entraînement
    for epoch in range(epochs):
        epoch_losses = []
        epoch_student_mse = []
        epoch_teacher_mse = []

        # Traitement par batch
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i : i + batch_size]
            batch_targets = train_targets[i : i + batch_size]

            # Distillation sur ce batch
            losses = distilled_model.distill_knowledge(
                data=batch_data,
                targets=batch_targets,
                optimizer=optimizer,
                alpha=alpha,
                temperature=temperature,
                attention_weight=attention_weight,
            )

            epoch_losses.append(losses["total_loss"])
            epoch_student_mse.append(losses["student_mse"])
            epoch_teacher_mse.append(losses["teacher_mse"])

        # Moyennes sur l'époque
        avg_train_loss = np.mean(epoch_losses)
        avg_student_mse = np.mean(epoch_student_mse)
        avg_teacher_mse = np.mean(epoch_teacher_mse)

        history["train_loss"].append(avg_train_loss)
        history["student_mse"].append(avg_student_mse)
        history["teacher_mse"].append(avg_teacher_mse)

        # Validation si données fournies
        if val_data is not None and val_targets is not None:
            distilled_model.student.eval()

            with torch.no_grad():
                val_preds, _ = distilled_model.student(val_data)
                val_loss = F.mse_loss(val_preds, val_targets).item()

                # Calcul de la prédiction enseignant pour référence
                teacher_val_preds, _ = teacher_model(val_data)
                teacher_val_loss = F.mse_loss(teacher_val_preds, val_targets).item()

            history["val_loss"].append(val_loss)

            # Early stopping et sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder l'état du meilleur modèle
                best_model_state = distilled_model.student.state_dict().copy()
            else:
                patience_counter += 1

            # Afficher les métriques
            print(
                f"Époque {epoch+1}/{epochs} - "
                f"Loss: {avg_train_loss:.4f} - "
                f"MSE étudiant: {avg_student_mse:.4f} - "
                f"MSE enseignant: {avg_teacher_mse:.4f} - "
                f"Val MSE étudiant: {val_loss:.4f} - "
                f"Val MSE enseignant: {teacher_val_loss:.4f}"
            )

            # Vérifier early stopping
            if patience_counter >= patience:
                print(f"Early stopping à l'époque {epoch+1}")
                break
        else:
            # Afficher les métriques sans validation
            print(
                f"Époque {epoch+1}/{epochs} - "
                f"Loss: {avg_train_loss:.4f} - "
                f"MSE étudiant: {avg_student_mse:.4f} - "
                f"MSE enseignant: {avg_teacher_mse:.4f}"
            )

    # Restaurer le meilleur modèle si disponible
    if best_model_state is not None:
        distilled_model.student.load_state_dict(best_model_state)

    # Sauvegarder le modèle et l'historique si un répertoire est spécifié
    if save_dir is not None:
        save_model_and_metrics(distilled_model, history, save_dir, reduction_factor)
    elif MODELS_DIR is not None:
        # Utiliser le répertoire par défaut dans info_retour
        default_save_dir = MODELS_DIR / "distilled"
        save_model_and_metrics(
            distilled_model, history, default_save_dir, reduction_factor
        )

    return distilled_model, history


def evaluate_distilled_model(
    distilled_model: DistilledFinancialTransformer,
    teacher_model: FinancialTemporalTransformer,
    test_data: torch.Tensor,
    test_targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Évalue et compare les performances du modèle étudiant et du modèle enseignant.

    Args:
        distilled_model: Modèle étudiant
        teacher_model: Modèle enseignant
        test_data: Données de test
        test_targets: Cibles de test

    Returns:
        Métriques de performance
    """
    distilled_model.student.eval()
    teacher_model.eval()

    with torch.no_grad():
        # Calcul du temps d'inférence
        import time

        # Inférence avec le modèle étudiant - répéter plusieurs fois pour plus de précision
        student_times = []
        for _ in range(5):  # Faire plusieurs passes pour une mesure plus fiable
            start_time = time.time()
            student_preds, _ = distilled_model.student(test_data)
            end_time = time.time()
            student_times.append(end_time - start_time)
        student_time = max(
            sum(student_times) / len(student_times), 1e-6
        )  # Éviter division par zéro

        # Inférence avec le modèle enseignant - répéter plusieurs fois pour plus de précision
        teacher_times = []
        for _ in range(5):  # Faire plusieurs passes pour une mesure plus fiable
            start_time = time.time()
            teacher_preds, _ = teacher_model(test_data)
            end_time = time.time()
            teacher_times.append(end_time - start_time)
        teacher_time = max(
            sum(teacher_times) / len(teacher_times), 1e-6
        )  # Éviter division par zéro

        # Calcul des métriques
        student_mse = F.mse_loss(student_preds, test_targets).item()
        teacher_mse = F.mse_loss(teacher_preds, test_targets).item()

        # Calcul des paramètres
        student_params = sum(p.numel() for p in distilled_model.student.parameters())
        teacher_params = sum(p.numel() for p in teacher_model.parameters())

    # S'assurer que les valeurs sont non nulles pour éviter la division par zéro
    return {
        "student_mse": student_mse,
        "teacher_mse": teacher_mse,
        "student_inference_time": student_time,
        "teacher_inference_time": teacher_time,
        "speed_improvement": teacher_time / student_time,
        "student_parameters": student_params,
        "teacher_parameters": teacher_params,
        "size_reduction": (
            teacher_params / student_params if student_params > 0 else float("inf")
        ),
    }


def save_model_and_metrics(
    distilled_model: DistilledFinancialTransformer,
    history: Dict,
    save_dir: str,
    reduction_factor: int,
):
    """
    Sauvegarde le modèle distillé et les métriques d'entraînement.

    Args:
        distilled_model: Modèle distillé à sauvegarder
        history: Historique d'entraînement
        save_dir: Répertoire de sauvegarde
        reduction_factor: Facteur de réduction utilisé
    """
    # Créer le répertoire si nécessaire
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarder le modèle
    model_path = os.path.join(save_dir, f"distilled_model_r{reduction_factor}.pt")
    torch.save(distilled_model.student.state_dict(), model_path)

    # Sauvegarder l'historique
    history_path = os.path.join(save_dir, f"training_history_r{reduction_factor}.json")

    # Convertir les valeurs pour la sérialisation JSON
    json_history = {}
    for key, values in history.items():
        if isinstance(values, list) and len(values) > 0:
            if isinstance(values[0], (int, float)):
                json_history[key] = values
            else:
                json_history[key] = [float(v) for v in values]

    with open(history_path, "w") as f:
        json.dump(json_history, f, indent=4)

    print(f"Modèle sauvegardé dans {model_path}")
    print(f"Historique sauvegardé dans {history_path}")


def save_distillation_results(results: Dict, filename: str) -> str:
    """
    Sauvegarde les résultats de distillation dans le dossier info_retour.

    Args:
        results: Dictionnaire de résultats à sauvegarder
        filename: Nom du fichier JSON

    Returns:
        Chemin complet du fichier sauvegardé
    """
    # Créer le répertoire pour les résultats de distillation
    distillation_dir = MODELS_DIR / "distilled" / "results"
    os.makedirs(distillation_dir, exist_ok=True)

    # Convertir les résultats pour compatibilité JSON
    json_results = {}

    if isinstance(results, dict) and all(isinstance(k, int) for k in results.keys()):
        # Si c'est un dictionnaire de facteurs de réduction
        for reduction_factor, metrics in results.items():
            json_results[str(reduction_factor)] = {
                k: float(v) for k, v in metrics.items()
            }
    else:
        # Si c'est un dictionnaire simple de métriques
        json_results = {k: float(v) for k, v in results.items()}

    # Chemin complet du fichier
    save_path = distillation_dir / filename

    # Sauvegarder les résultats au format JSON
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=4)

    print(f"Résultats de distillation sauvegardés dans {save_path}")
    return save_path
