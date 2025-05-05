#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module wrapper pour TensorFlow GPU qui fonctionne même sans GPU disponible.
Fournit des fonctionnalités de fallback CPU automatique et une gestion des erreurs
pour assurer la compatibilité sur tous les environnements.
"""

import logging
import os

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Activer la croissance mémoire pour TensorFlow
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Vérifier si TensorFlow est disponible
TF_AVAILABLE = False
TF_GPU_AVAILABLE = False
GPU_DEVICES = []

try:
    import tensorflow as tf

    TF_AVAILABLE = True

    # Vérifier si des GPUs sont disponibles
    gpus = tf.config.list_physical_devices("GPU")
    TF_GPU_AVAILABLE = len(gpus) > 0

    if TF_GPU_AVAILABLE:
        # Enregistrer les infos sur les GPUs
        GPU_DEVICES = gpus

        # Configurer la mémoire GPU
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configuration de la mémoire adaptative pour {gpu.name}")
            except:
                # Certains environnements ne permettent pas cette configuration
                pass

        logger.info(f"TensorFlow GPU disponible: {len(gpus)} GPU(s) détecté(s)")
    else:
        logger.warning("TensorFlow est disponible mais aucun GPU n'est détecté")
except ImportError:
    logger.warning("TensorFlow n'est pas installé")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation de TensorFlow: {str(e)}")


class TFDeviceSelector:
    """
    Classe pour gérer automatiquement le choix des dispositifs TensorFlow (GPU/CPU).
    """

    @staticmethod
    def get_device_strategy():
        """
        Retourne la stratégie optimale pour l'exécution TensorFlow.

        Returns:
            tf.distribute.Strategy: MirroredStrategy pour multi-GPU, OneDeviceStrategy sinon
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow n'est pas disponible")

        try:
            # Si plusieurs GPUs sont disponibles, utiliser MirroredStrategy
            if TF_GPU_AVAILABLE and len(GPU_DEVICES) > 1:
                return tf.distribute.MirroredStrategy()

            # Si un seul GPU est disponible, utiliser OneDeviceStrategy avec GPU
            elif TF_GPU_AVAILABLE:
                return tf.distribute.OneDeviceStrategy(device="/gpu:0")

            # Aucun GPU, utiliser CPU
            else:
                return tf.distribute.OneDeviceStrategy(device="/cpu:0")
        except Exception as e:
            logger.warning(f"Erreur lors de la création de la stratégie: {str(e)}")
            # En cas d'erreur, revenir à la stratégie par défaut
            return tf.distribute.get_strategy()

    @staticmethod
    def get_available_devices():
        """
        Retourne la liste des dispositifs disponibles.

        Returns:
            List[str]: Liste des dispositifs TensorFlow disponibles
        """
        if not TF_AVAILABLE:
            return ["CPU (TensorFlow non disponible)"]

        try:
            return [device.name for device in tf.config.list_logical_devices()]
        except:
            if TF_GPU_AVAILABLE:
                return [f"GPU:{i}" for i in range(len(GPU_DEVICES))] + ["CPU"]
            else:
                return ["CPU"]

    @staticmethod
    def get_recommended_device():
        """
        Retourne le dispositif recommandé pour l'exécution.

        Returns:
            str: Nom du dispositif recommandé
        """
        if not TF_AVAILABLE:
            return "CPU (TensorFlow non disponible)"

        if TF_GPU_AVAILABLE:
            return "/gpu:0"
        else:
            return "/cpu:0"


class GPUMemoryManager:
    """
    Gestionnaire de mémoire GPU pour TensorFlow.
    """

    @staticmethod
    def limit_memory_growth(limit_percent=None):
        """
        Limite la croissance mémoire pour chaque GPU.

        Args:
            limit_percent: Pourcentage de mémoire à utiliser (None = croissance adaptative)
        """
        if not TF_AVAILABLE or not TF_GPU_AVAILABLE:
            logger.warning("Aucun GPU TensorFlow disponible pour limiter la mémoire")
            return

        try:
            for gpu in GPU_DEVICES:
                if limit_percent is None:
                    # Croissance adaptative
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    # Limiter à un pourcentage fixe
                    limit = limit_percent / 100.0
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=limit
                                * tf.config.experimental.get_device_details(gpu)[
                                    "memory_limit"
                                ]
                            )
                        ],
                    )
            logger.info(
                f"Configuration mémoire GPU appliquée: {'adaptative' if limit_percent is None else f'{limit_percent}%'}"
            )
        except Exception as e:
            logger.warning(f"Erreur lors de la limitation mémoire GPU: {str(e)}")

    @staticmethod
    def clear_gpu_memory():
        """
        Libère la mémoire GPU non utilisée.
        """
        if not TF_AVAILABLE:
            return

        try:
            if hasattr(tf.keras.backend, "clear_session"):
                tf.keras.backend.clear_session()

            if TF_GPU_AVAILABLE:
                # Pour les anciens moteurs TF
                if hasattr(tf, "reset_default_graph"):
                    tf.reset_default_graph()

            logger.info("Mémoire GPU libérée")
        except Exception as e:
            logger.warning(f"Erreur lors de la libération mémoire GPU: {str(e)}")


def create_tf_gpu_compatible_model(model_fn, *args, **kwargs):
    """
    Crée un modèle TensorFlow compatible sur tous les environnements.

    Args:
        model_fn: Fonction qui crée le modèle
        *args, **kwargs: Arguments pour model_fn

    Returns:
        Le modèle TensorFlow créé avec la stratégie appropriée
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow n'est pas disponible")

    # Obtenir la stratégie optimale
    strategy = TFDeviceSelector.get_device_strategy()

    # Créer le modèle avec la stratégie
    with strategy.scope():
        model = model_fn(*args, **kwargs)

    # Journaliser les informations
    device_type = "GPU" if TF_GPU_AVAILABLE else "CPU"
    logger.info(
        f"Modèle TensorFlow créé sur {device_type} avec stratégie {type(strategy).__name__}"
    )

    return model


def mixed_precision_tf_config():
    """
    Configure TensorFlow pour utiliser la précision mixte si disponible.

    Returns:
        bool: True si la précision mixte est activée
    """
    if not TF_AVAILABLE:
        return False

    try:
        # Activer la précision mixte si sur GPU
        if TF_GPU_AVAILABLE:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Précision mixte TensorFlow activée (float16)")
            return True
        else:
            logger.info("GPU non disponible, précision mixte non activée")
            return False
    except Exception as e:
        logger.warning(f"Erreur lors de l'activation de la précision mixte: {str(e)}")
        return False


def is_tf_gpu_available():
    """
    Vérifie si TensorFlow avec GPU est disponible.

    Returns:
        Tuple[bool, str]: (Disponibilité, Message détaillé)
    """
    if not TF_AVAILABLE:
        return False, "TensorFlow n'est pas installé"

    if TF_GPU_AVAILABLE:
        gpu_names = [gpu.name for gpu in GPU_DEVICES]
        return (
            True,
            f"TensorFlow GPU disponible: {len(GPU_DEVICES)} GPU(s) détecté(s): {', '.join(gpu_names)}",
        )
    else:
        return False, "TensorFlow est disponible mais aucun GPU n'est détecté"


# Fonction de test
if __name__ == "__main__":
    # Vérifier la disponibilité de TensorFlow GPU
    available, message = is_tf_gpu_available()
    print(f"TensorFlow GPU disponible: {available}")
    print(f"Message: {message}")

    # Afficher les dispositifs disponibles
    if TF_AVAILABLE:
        print(f"Dispositifs disponibles: {TFDeviceSelector.get_available_devices()}")
        print(f"Dispositif recommandé: {TFDeviceSelector.get_recommended_device()}")

        # Créer un petit modèle de test
        def create_model():
            return tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )

        try:
            # Activer la précision mixte
            mixed_precision_tf_config()

            # Créer le modèle avec la stratégie appropriée
            model = create_tf_gpu_compatible_model(create_model)

            # Tester le modèle
            inputs = tf.random.normal([32, 5])
            outputs = model(inputs)

            print(
                f"Modèle créé et testé avec succès sur {TFDeviceSelector.get_recommended_device()}"
            )
            print(f"Forme de sortie: {outputs.shape}")

            # Libérer la mémoire
            GPUMemoryManager.clear_gpu_memory()
        except Exception as e:
            print(f"Erreur lors du test: {str(e)}")
    else:
        print("TensorFlow n'est pas disponible pour les tests")
