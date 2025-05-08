"""
Module d'optimisation spécifique pour RTX 3070.
Implémente des fonctions pour améliorer les performances sur la série RTX 30xx.
"""

import logging
import os
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def setup_rtx_optimization(
    enable_tensor_cores=True,
    enable_mixed_precision=True,
    optimize_memory_allocation=True,
):
    """
    Configure les optimisations recommandées pour RTX 3070.

    Args:
        enable_tensor_cores: Active les Tensor Cores pour les calculs
        enable_mixed_precision: Active la précision mixte (float16/float32)
        optimize_memory_allocation: Configure l'allocation mémoire optimale

    Returns:
        bool: True si toutes les optimisations ont été appliquées, False sinon
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA n'est pas disponible. Les optimisations RTX ne seront pas appliquées."
        )
        return False

    device_name = torch.cuda.get_device_name(0)
    logger.info(f"Configuration des optimisations pour GPU: {device_name}")

    success = True

    # Vérifier si c'est bien une carte RTX série 30xx
    if "RTX 30" not in device_name and "RTX 3070" not in device_name:
        logger.warning(
            f"GPU détecté ({device_name}) n'est pas une RTX 30xx. "
            "Les optimisations pourraient ne pas être idéales."
        )

    # Activer les Tensor Cores
    if enable_tensor_cores:
        try:
            # Activer TF32 (disponible sur Ampere+)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Variable d'environnement pour les Tensor Cores
            os.environ["PYTORCH_CUDA_USE_TENSOR_CORES"] = "1"

            # Optimiser cudnn pour les convolutions
            torch.backends.cudnn.benchmark = True

            logger.info("Tensor Cores activés pour les opérations matricielles")
        except Exception as e:
            logger.error(f"Erreur lors de l'activation des Tensor Cores: {e}")
            success = False

    # Configurer l'allocation mémoire
    if optimize_memory_allocation:
        try:
            # Configurer l'allocation mémoire CUDA
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "max_split_size_mb:128,garbage_collection_threshold:0.8"
            )

            logger.info("Allocation mémoire CUDA optimisée configurée")
        except Exception as e:
            logger.error(
                f"Erreur lors de la configuration de l'allocation mémoire: {e}"
            )
            success = False

    # Retourner le statut
    return success


@contextmanager
def mixed_precision_context():
    """
    Contexte pour l'entraînement en précision mixte avec torch.cuda.amp.
    Utilise l'autocast pour accélérer les calculs et économiser la mémoire.

    Exemple d'utilisation:
        with mixed_precision_context():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA n'est pas disponible. mixed_precision_context fonctionnera en mode passthrough."
        )
        yield
        return

    try:
        with torch.cuda.amp.autocast(enabled=True):
            logger.debug("Contexte de précision mixte activé")
            yield
    except Exception as e:
        logger.error(f"Erreur dans le contexte de précision mixte: {e}")
        yield
    finally:
        logger.debug("Contexte de précision mixte désactivé")


class MixedPrecisionTrainer:
    """
    Classe pour faciliter l'entraînement en précision mixte avec un scaler de gradient.
    Optimise l'utilisation de la mémoire et accélère l'entraînement sur RTX 3070.
    """

    def __init__(self, model, optimizer, initial_scale=2**10):
        """
        Initialise le trainer avec précision mixte.

        Args:
            model: Le modèle PyTorch à entraîner
            optimizer: L'optimiseur à utiliser
            initial_scale: Échelle initiale pour le GradScaler
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler(init_scale=initial_scale)

        # Vérification de la disponibilité CUDA
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning(
                "CUDA n'est pas disponible. MixedPrecisionTrainer utilisera la précision standard."
            )

    def training_step(self, inputs, targets, criterion):
        """
        Effectue une étape d'entraînement avec précision mixte.

        Args:
            inputs: Entrées du modèle
            targets: Cibles pour le calcul de perte
            criterion: Fonction de perte

        Returns:
            loss: La valeur de perte calculée
        """
        # Réinitialiser les gradients
        self.optimizer.zero_grad()

        if self.cuda_available:
            # Utilisation de la précision mixte
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

            # Mise à l'échelle de la perte pour éviter l'underflow
            self.scaler.scale(loss).backward()

            # Décompression et mise à jour
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Calcul standard si CUDA n'est pas disponible
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

        return loss.item()


def optimize_batch_size(model, input_shape, max_batch_size=128, start_batch_size=8):
    """
    Trouve la taille de batch optimale pour la RTX 3070.
    Augmente progressivement la taille jusqu'à atteindre la limite mémoire.

    Args:
        model: Le modèle PyTorch à tester
        input_shape: Forme des données d'entrée (sans la dim batch)
        max_batch_size: Taille de batch maximale à tester
        start_batch_size: Taille de batch de départ

    Returns:
        int: Taille de batch optimale
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA n'est pas disponible. Impossible d'optimiser la taille de batch."
        )
        return start_batch_size

    # Placer le modèle sur GPU
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    # Tester des tailles croissantes
    optimal_batch_size = start_batch_size

    for batch_size in [2**i for i in range(3, 11) if 2**i <= max_batch_size]:
        try:
            # Créer un tensor d'entrée aléatoire
            dummy_input = torch.rand(batch_size, *input_shape, device=device)

            # Nettoyer la mémoire avant le test
            torch.cuda.empty_cache()

            # Exécuter un forward pass
            with torch.no_grad():
                model(dummy_input)

            # Si réussi, mettre à jour la taille optimale
            optimal_batch_size = batch_size
            logger.info(f"Batch size {batch_size} : OK")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.info(f"Batch size {batch_size} : Out of memory")
                break
            else:
                logger.error(f"Erreur inattendue: {e}")
                break

    # Recommander une taille de batch qui est un multiple de 8 (optimal pour Tensor Cores)
    recommended_batch_size = optimal_batch_size
    if recommended_batch_size % 8 != 0:
        # Trouver le multiple de 8 le plus proche inférieur ou égal
        recommended_batch_size = (recommended_batch_size // 8) * 8

    logger.info(
        f"Taille de batch recommandée pour RTX: {recommended_batch_size} (multiple de 8 optimal pour Tensor Cores)"
    )
    return recommended_batch_size


if __name__ == "__main__":
    # Configuration du logger pour les tests directs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Tester les optimisations si ce fichier est exécuté directement
    setup_rtx_optimization()

    if torch.cuda.is_available():
        logger.info(
            f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
        logger.info(f"Architecture CUDA: {torch.cuda.get_device_properties(0).name}")
