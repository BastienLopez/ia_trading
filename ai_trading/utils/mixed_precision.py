"""
Module d'optimisation par entraînement en précision mixte (Mixed Precision Training).

Ce module fournit des utilitaires pour l'entraînement en précision mixte avec torch.cuda.amp,
permettant de réduire la consommation mémoire de moitié et d'accélérer l'entraînement de 1,5x
sur les GPU modernes (Ampere et plus récents).
"""

import contextlib
import logging
from typing import Any, Callable, Dict, Tuple

import torch

logger = logging.getLogger(__name__)


def is_mixed_precision_supported() -> bool:
    """
    Vérifie si la précision mixte est supportée par le matériel actuel.

    Returns:
        bool: True si la précision mixte est prise en charge, False sinon
    """
    if not torch.cuda.is_available():
        return False

    # Obtenez les informations sur le GPU
    device_name = torch.cuda.get_device_name(0)

    # Vérifiez si le GPU est compatible avec les Tensor Cores (Volta, Turing, Ampere ou plus récent)
    # Les GPU RTX 2000, RTX 3000, RTX 4000, A100, etc. sont compatibles
    return any(
        gpu_type in device_name
        for gpu_type in [
            "RTX",
            "V100",
            "A100",
            "A10",
            "A6000",
            "A40",
            "A16",
            "A2",
            "A30",
            "T4",
            "Quadro",
            "Tesla",
            "GTX 16",
            "RTX 30",
            "RTX 40",
        ]
    )


def setup_mixed_precision() -> bool:
    """
    Configure PyTorch pour l'utilisation de la précision mixte.

    Returns:
        bool: True si la configuration a réussi, False sinon
    """
    if not is_mixed_precision_supported():
        logger.warning("La précision mixte n'est pas prise en charge sur ce matériel")
        return False

    try:
        # Activer TF32 pour les opérations matricielles (Ampere+)
        # Cela permet d'accélérer les calculs sans perte significative de précision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Activer le benchmark cudnn pour optimiser les convolutions
        torch.backends.cudnn.benchmark = True

        logger.info("Précision mixte configurée avec succès")
        logger.info("TF32 activé pour les opérations matricielles")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la configuration de la précision mixte: {e}")
        return False


@contextlib.contextmanager
def autocast_context(
    enabled: bool = True, dtype: torch.dtype = torch.float16, cache_enabled: bool = True
):
    """
    Fournit un contexte pour l'autocast (précision mixte automatique).

    Args:
        enabled: Si True, active l'autocast
        dtype: Type de données à utiliser pour la précision réduite (float16 par défaut)
        cache_enabled: Si True, active le cache pour les opérations autocast

    Yields:
        Un contexte dans lequel l'autocast est activé
    """
    if not torch.cuda.is_available():
        # Si CUDA n'est pas disponible, retourner un contexte vide
        yield
        return

    try:
        with torch.cuda.amp.autocast(
            enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        ):
            yield
    except Exception as e:
        logger.error(f"Erreur dans le contexte autocast: {e}")
        raise


class MixedPrecisionWrapper:
    """
    Wrapper pour faciliter l'utilisation de la précision mixte avec n'importe quel modèle PyTorch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        initial_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        """
        Initialise le wrapper pour précision mixte.

        Args:
            model: Le modèle PyTorch à entraîner
            optimizer: L'optimiseur PyTorch à utiliser
            initial_scale: Échelle initiale pour le gradient scaler
            growth_factor: Facteur d'augmentation de l'échelle
            backoff_factor: Facteur de réduction de l'échelle en cas d'overflow
            growth_interval: Intervalle pour augmenter l'échelle
            enabled: Si True, active la précision mixte
        """
        self.model = model
        self.optimizer = optimizer
        self.enabled = enabled and torch.cuda.is_available()

        # Créer un GradScaler pour gérer l'échelle des gradients
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.enabled,
            init_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )

        # Vérifier si la précision mixte est supportée
        if self.enabled and not is_mixed_precision_supported():
            logger.warning(
                "La précision mixte est activée mais peut ne pas être optimale sur ce matériel"
            )

        if self.enabled:
            logger.info("MixedPrecisionWrapper initialisé avec précision mixte activée")
            current_device = torch.cuda.get_device_name(0)
            logger.info(f"GPU détecté: {current_device}")
        else:
            logger.info(
                "MixedPrecisionWrapper initialisé avec précision mixte désactivée"
            )

    def training_step(
        self,
        batch: Any,
        forward_fn: Callable,
        loss_fn: Callable,
        accumulation_steps: int = 1,
    ) -> torch.Tensor:
        """
        Effectue une étape d'entraînement avec précision mixte.

        Args:
            batch: Les données d'entrée pour le modèle
            forward_fn: Fonction qui prend le batch et retourne la sortie du modèle
            loss_fn: Fonction qui prend la sortie et le batch et retourne la perte
            accumulation_steps: Nombre d'étapes pour accumuler les gradients

        Returns:
            La perte calculée
        """
        # Normaliser la perte pour l'accumulation de gradient
        loss_scale = accumulation_steps if accumulation_steps > 1 else 1

        if self.enabled:
            # Forward pass avec precision mixte
            with autocast_context():
                outputs = forward_fn(batch)
                loss = loss_fn(outputs, batch)

                # Normaliser la perte pour l'accumulation de gradient
                if accumulation_steps > 1:
                    loss = loss / loss_scale

            # Backward pass avec scaling
            self.scaler.scale(loss).backward()

            # Mise à jour si c'est la dernière étape d'accumulation ou pas d'accumulation
            if accumulation_steps == 1:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # Mode précision standard
            outputs = forward_fn(batch)
            loss = loss_fn(outputs, batch)

            # Normaliser la perte pour l'accumulation de gradient
            if accumulation_steps > 1:
                loss = loss / loss_scale

            # Backward pass standard
            loss.backward()

            # Mise à jour si c'est la dernière étape d'accumulation ou pas d'accumulation
            if accumulation_steps == 1:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        return loss

    def optimizer_step(self) -> None:
        """
        Effectue une étape d'optimisation après l'accumulation des gradients.
        À appeler après les étapes d'accumulation de gradient.
        """
        if self.enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> Dict:
        """
        Retourne l'état du GradScaler pour la sauvegarde.

        Returns:
            Dict: État du GradScaler
        """
        if self.enabled:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Charge l'état du GradScaler depuis une sauvegarde.

        Args:
            state_dict: État du GradScaler à charger
        """
        if self.enabled and state_dict:
            self.scaler.load_state_dict(state_dict)


def test_mixed_precision_performance(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Teste les performances du modèle avec et sans précision mixte.

    Args:
        model: Le modèle PyTorch à tester
        input_shape: La forme des données d'entrée (sans la dimension de batch)
        batch_size: La taille du batch à utiliser
        iterations: Le nombre d'itérations pour le test

    Returns:
        Dict contenant les temps d'exécution et l'utilisation mémoire
    """
    import gc
    import time

    if not torch.cuda.is_available():
        logger.warning("CUDA n'est pas disponible, test impossible")
        return {
            "fp32_time": 0.0,
            "fp16_time": 0.0,
            "speedup": 0.0,
            "fp32_memory_mb": 0.0,
            "fp16_memory_mb": 0.0,
            "memory_reduction": 0.0,
        }

    device = torch.device("cuda")
    model = model.to(device)
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Nettoyer la mémoire avant les tests
    torch.cuda.empty_cache()
    gc.collect()

    # Test avec précision standard (FP32)
    model.eval()
    start_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    fp32_time = time.time() - start_time
    fp32_memory = (torch.cuda.max_memory_allocated() / (1024**2)) - start_memory

    # Nettoyer la mémoire
    torch.cuda.empty_cache()
    gc.collect()

    # Test avec précision mixte (FP16)
    start_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

    start_time = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    fp16_time = time.time() - start_time
    fp16_memory = (torch.cuda.max_memory_allocated() / (1024**2)) - start_memory

    # Calculer les ratios
    speedup = fp32_time / fp16_time if fp16_time > 0 else 0
    memory_reduction = fp32_memory / fp16_memory if fp16_memory > 0 else 0

    results = {
        "fp32_time": fp32_time,
        "fp16_time": fp16_time,
        "speedup": speedup,
        "fp32_memory_mb": fp32_memory,
        "fp16_memory_mb": fp16_memory,
        "memory_reduction": memory_reduction,
    }

    logger.info(f"Test de performance Mixed Precision:")
    logger.info(f"  - Temps FP32: {fp32_time:.4f}s")
    logger.info(f"  - Temps FP16: {fp16_time:.4f}s")
    logger.info(f"  - Accélération: {speedup:.2f}x")
    logger.info(f"  - Mémoire FP32: {fp32_memory:.2f} MB")
    logger.info(f"  - Mémoire FP16: {fp16_memory:.2f} MB")
    logger.info(f"  - Réduction mémoire: {memory_reduction:.2f}x")

    return results


if __name__ == "__main__":
    # Configuration basique du logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Vérifier si la précision mixte est supportée
    if is_mixed_precision_supported():
        logger.info("La précision mixte est supportée sur ce matériel")
        setup_mixed_precision()
    else:
        logger.info("La précision mixte n'est pas supportée sur ce matériel")
