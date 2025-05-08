"""
Module pour nettoyer la mémoire GPU après l'exécution des tests.
Compatible avec PyTorch et TensorFlow.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def cleanup_gpu_memory():
    """
    Nettoie la mémoire GPU en libérant les ressources de PyTorch et TensorFlow.
    """
    logger.info("Nettoyage de la mémoire GPU...")

    # Collecter les objets garbage
    gc.collect()

    # Nettoyage spécifique à PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
            logger.info("Mémoire PyTorch CUDA libérée")
        else:
            logger.info("PyTorch CUDA n'est pas disponible, aucun nettoyage nécessaire")
    except ImportError:
        logger.info("PyTorch n'est pas installé, ignoré")
    except Exception as e:
        logger.warning(f"Erreur lors du nettoyage de la mémoire PyTorch: {e}")

    # Nettoyage spécifique à TensorFlow
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()

        # Pour TensorFlow 2.x
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu)
            logger.info("Mémoire TensorFlow libérée")
        except:
            # Pour les versions plus anciennes ou si la méthode n'est pas disponible
            logger.info("Nettoyage TensorFlow limité effectué")
    except ImportError:
        logger.info("TensorFlow n'est pas installé, ignoré")
    except Exception as e:
        logger.warning(f"Erreur lors du nettoyage de la mémoire TensorFlow: {e}")

    logger.info("Nettoyage de la mémoire GPU terminé")


if __name__ == "__main__":
    # Configuration du logger pour les tests directs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Exécuter le nettoyage si ce fichier est exécuté directement
    cleanup_gpu_memory()
