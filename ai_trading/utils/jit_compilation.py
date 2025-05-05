#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la compilation JIT des modèles avec TorchScript et XLA.
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Vérifier si PyTorch est disponible
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch n'est pas disponible. Les fonctionnalités TorchScript seront désactivées."
    )

# Vérifier si TensorFlow est disponible pour XLA
try:
    import tensorflow as tf

    TF_AVAILABLE = True

    # Vérifier si XLA est disponible
    try:
        XLA_AVAILABLE = True
    except ImportError:
        XLA_AVAILABLE = False
        warnings.warn("XLA n'est pas disponible dans votre installation TensorFlow.")
except ImportError:
    TF_AVAILABLE = False
    XLA_AVAILABLE = False
    warnings.warn(
        "TensorFlow n'est pas disponible. Les fonctionnalités XLA seront désactivées."
    )


class TorchScriptCompiler:
    """Classe pour compiler des modèles PyTorch en utilisant TorchScript."""

    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialise le compilateur TorchScript.

        Args:
            save_dir: Répertoire pour sauvegarder les modèles compilés
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch est requis pour utiliser TorchScriptCompiler.")

        # Déterminer si nous sommes déjà dans le répertoire ai_trading
        current_dir = os.path.basename(os.getcwd())
        if save_dir:
            self.save_dir = Path(save_dir)
        elif current_dir == "ai_trading":
            # Si nous sommes déjà dans ai_trading, utiliser un chemin relatif
            self.save_dir = Path("info_retour/models/jit")
        else:
            # Sinon, utiliser le chemin complet
            self.save_dir = Path("ai_trading/info_retour/models/jit")
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Les modèles compilés seront sauvegardés dans {self.save_dir}")

    def trace_model(
        self,
        model: "torch.nn.Module",
        example_inputs: Any,
        optimize: bool = True,
        name: Optional[str] = None,
    ) -> "torch.jit.ScriptModule":
        """
        Compile un modèle PyTorch en utilisant torch.jit.trace.

        Args:
            model: Le modèle PyTorch à compiler
            example_inputs: Exemple d'entrées pour le traçage
            optimize: Si True, optimise le modèle tracé
            name: Nom optionnel pour sauvegarder le modèle

        Returns:
            Le modèle compilé avec TorchScript
        """
        # Mettre le modèle en mode évaluation
        model.eval()

        # Compiler le modèle avec torch.jit.trace
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_inputs)

        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)

        # Sauvegarder le modèle si un nom est fourni
        if name:
            model_path = self.save_dir / f"{name}.pt"
            traced_model.save(str(model_path))
            logger.info(f"Modèle compilé sauvegardé dans {model_path}")

        return traced_model

    def script_model(
        self,
        model: "torch.nn.Module",
        optimize: bool = True,
        name: Optional[str] = None,
    ) -> "torch.jit.ScriptModule":
        """
        Compile un modèle PyTorch en utilisant torch.jit.script.

        Args:
            model: Le modèle PyTorch à compiler
            optimize: Si True, optimise le modèle scripté
            name: Nom optionnel pour sauvegarder le modèle

        Returns:
            Le modèle compilé avec TorchScript
        """
        # Mettre le modèle en mode évaluation
        model.eval()

        # Compiler le modèle avec torch.jit.script
        scripted_model = torch.jit.script(model)

        if optimize:
            # Note: optimize_for_inference ne fonctionne que pour les modèles tracés
            scripted_model = torch.jit.freeze(scripted_model)

        # Sauvegarder le modèle si un nom est fourni
        if name:
            model_path = self.save_dir / f"{name}.pt"
            scripted_model.save(str(model_path))
            logger.info(f"Modèle compilé sauvegardé dans {model_path}")

        return scripted_model

    def benchmark_model(
        self,
        model: "torch.nn.Module",
        scripted_model: "torch.jit.ScriptModule",
        input_data: Any,
        num_warmup: int = 10,
        num_iter: int = 100,
    ) -> Dict[str, Any]:
        """
        Compare les performances entre un modèle original et sa version compilée.

        Args:
            model: Le modèle PyTorch original
            scripted_model: Le modèle compilé avec TorchScript
            input_data: Données d'entrée pour le benchmark
            num_warmup: Nombre d'itérations de préchauffage
            num_iter: Nombre d'itérations pour le benchmark

        Returns:
            Dictionnaire contenant les résultats du benchmark
        """
        # Mettre les modèles en mode évaluation
        model.eval()

        # Préchauffement pour le modèle original
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_data)

        # Benchmark du modèle original
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iter):
                _ = model(input_data)
        original_time = time.time() - start_time

        # Préchauffement pour le modèle compilé
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = scripted_model(input_data)

        # Benchmark du modèle compilé
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iter):
                _ = scripted_model(input_data)
        scripted_time = time.time() - start_time

        # Calculer le speedup
        speedup = original_time / scripted_time if scripted_time > 0 else float("inf")

        return {
            "original_time": original_time,
            "scripted_time": scripted_time,
            "original_avg_ms": (original_time / num_iter) * 1000,
            "scripted_avg_ms": (scripted_time / num_iter) * 1000,
            "speedup": speedup,
            "num_iterations": num_iter,
        }

    def load_compiled_model(self, name: str) -> "torch.jit.ScriptModule":
        """
        Charge un modèle compilé précédemment.

        Args:
            name: Nom du modèle à charger

        Returns:
            Le modèle compilé chargé
        """
        model_path = self.save_dir / f"{name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Le modèle compilé {model_path} n'existe pas.")

        return torch.jit.load(str(model_path))


class XLAOptimizer:
    """Classe pour optimiser les modèles TensorFlow avec XLA."""

    def __init__(self):
        """Initialise l'optimiseur XLA."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow est requis pour utiliser XLAOptimizer.")
        if not XLA_AVAILABLE:
            raise ImportError(
                "XLA n'est pas disponible dans votre installation TensorFlow."
            )

    def enable_xla(self):
        """Active XLA pour toutes les opérations TensorFlow."""
        # Configuration de XLA
        tf.config.optimizer.set_jit(True)

        # Vérifier si des GPU sont disponibles
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Activer la mémoire GPU de croissance
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Utiliser XLA sur GPU si disponible
                os.environ["TF_XLA_FLAGS"] = (
                    "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
                )
                logger.info("XLA GPU activé")
            except RuntimeError as e:
                logger.error(f"Erreur lors de l'activation de XLA sur GPU: {e}")
        else:
            # Utiliser XLA sur CPU
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
            logger.info("XLA CPU activé")

    def compile_model(
        self, model: "tf.keras.Model", optimizer: Optional[Any] = None
    ) -> "tf.keras.Model":
        """
        Compile un modèle TensorFlow avec XLA.

        Args:
            model: Le modèle TensorFlow à compiler
            optimizer: Optimiseur à utiliser pour la compilation

        Returns:
            Le modèle compilé avec XLA
        """
        # Si le modèle n'est pas encore compilé, le compiler avec XLA
        if not model._is_compiled:
            if optimizer:
                model.compile(
                    optimizer=optimizer,
                    loss=model.loss if hasattr(model, "loss") else None,
                    metrics=model.metrics if hasattr(model, "metrics") else None,
                    jit_compile=True,  # Activer la compilation JIT (XLA)
                )
            else:
                # Si aucun optimiseur n'est fourni, ne spécifier que jit_compile
                model.compile(jit_compile=True)
        else:
            # Si le modèle est déjà compilé, créer une version avec XLA activé
            config = model.get_config()
            weights = model.get_weights()

            # Créer un nouveau modèle avec la même configuration
            new_model = model.__class__.from_config(config)
            new_model.set_weights(weights)

            # Compiler avec XLA
            new_model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
                jit_compile=True,
            )
            model = new_model

        logger.info("Modèle compilé avec XLA")
        return model

    def benchmark_model(
        self,
        original_model: "tf.keras.Model",
        xla_model: "tf.keras.Model",
        input_data: Any,
        num_warmup: int = 10,
        num_iter: int = 100,
    ) -> Dict[str, Any]:
        """
        Compare les performances entre un modèle original et sa version XLA.

        Args:
            original_model: Le modèle TensorFlow original
            xla_model: Le modèle compilé avec XLA
            input_data: Données d'entrée pour le benchmark
            num_warmup: Nombre d'itérations de préchauffage
            num_iter: Nombre d'itérations pour le benchmark

        Returns:
            Dictionnaire contenant les résultats du benchmark
        """
        # Préchauffement pour le modèle original
        for _ in range(num_warmup):
            _ = original_model(input_data)

        # Benchmark du modèle original
        start_time = time.time()
        for _ in range(num_iter):
            _ = original_model(input_data)
        original_time = time.time() - start_time

        # Préchauffement pour le modèle XLA
        for _ in range(num_warmup):
            _ = xla_model(input_data)

        # Benchmark du modèle XLA
        start_time = time.time()
        for _ in range(num_iter):
            _ = xla_model(input_data)
        xla_time = time.time() - start_time

        # Calculer le speedup
        speedup = original_time / xla_time if xla_time > 0 else float("inf")

        return {
            "original_time": original_time,
            "xla_time": xla_time,
            "original_avg_ms": (original_time / num_iter) * 1000,
            "xla_avg_ms": (xla_time / num_iter) * 1000,
            "speedup": speedup,
            "num_iterations": num_iter,
        }


# Fonction utilitaire pour la compilation JIT facile
def compile_model(
    model: Union["torch.nn.Module", "tf.keras.Model"],
    example_inputs: Optional[Any] = None,
    framework: str = "auto",
    method: str = "auto",
) -> Union["torch.jit.ScriptModule", "tf.keras.Model"]:
    """
    Compile un modèle en utilisant TorchScript ou XLA selon le framework détecté.

    Args:
        model: Le modèle à compiler (PyTorch ou TensorFlow)
        example_inputs: Exemple d'entrées pour le traçage (uniquement pour PyTorch)
        framework: Framework à utiliser ('torch', 'tensorflow' ou 'auto')
        method: Méthode de compilation ('trace', 'script', 'xla' ou 'auto')

    Returns:
        Le modèle compilé
    """
    # Détecter automatiquement le framework
    if framework == "auto":
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            framework = "torch"
        elif TF_AVAILABLE and isinstance(model, tf.keras.Model):
            framework = "tensorflow"
        else:
            raise ValueError("Impossible de détecter automatiquement le framework.")

    # Compiler le modèle selon le framework
    if framework == "torch":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch est requis pour compiler des modèles PyTorch.")

        compiler = TorchScriptCompiler()

        # Choisir la méthode de compilation
        if method == "auto":
            # Si des entrées d'exemple sont fournies, utiliser trace
            if example_inputs is not None:
                method = "trace"
            else:
                method = "script"

        if method == "trace":
            if example_inputs is None:
                raise ValueError("example_inputs est requis pour la méthode 'trace'.")
            return compiler.trace_model(model, example_inputs)
        elif method == "script":
            return compiler.script_model(model)
        else:
            raise ValueError(
                f"Méthode de compilation '{method}' non prise en charge pour PyTorch."
            )

    elif framework == "tensorflow":
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow est requis pour compiler des modèles TensorFlow."
            )
        if not XLA_AVAILABLE:
            raise ImportError(
                "XLA n'est pas disponible dans votre installation TensorFlow."
            )

        optimizer = XLAOptimizer()
        optimizer.enable_xla()
        return optimizer.compile_model(model)

    else:
        raise ValueError(f"Framework '{framework}' non pris en charge.")


def optimize_function(func: Callable) -> Callable:
    """
    Décorateur pour optimiser une fonction PyTorch avec torch.jit.script.

    Args:
        func: Fonction à optimiser

    Returns:
        Fonction optimisée
    """
    if not TORCH_AVAILABLE:
        warnings.warn(
            "PyTorch n'est pas disponible. La fonction ne sera pas optimisée."
        )
        return func

    try:
        return torch.jit.script(func)
    except Exception as e:
        logger.warning(f"Impossible d'optimiser la fonction {func.__name__}: {e}")
        return func


# Configuration globale pour l'utilisation de XLA avec TensorFlow
def enable_tensorflow_xla():
    """Active XLA globalement pour TensorFlow."""
    if not TF_AVAILABLE:
        warnings.warn("TensorFlow n'est pas disponible. XLA ne peut pas être activé.")
        return False

    try:
        # Activer XLA pour toutes les opérations
        tf.config.optimizer.set_jit(True)

        # Configurer les flags XLA
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"

        # Vérifier si des GPU sont disponibles
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Activer XLA sur GPU
            os.environ["TF_XLA_FLAGS"] += " --tf_xla_auto_jit=2"

        logger.info(
            f"XLA activé pour TensorFlow {'avec GPU' if gpus else 'sur CPU uniquement'}"
        )
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'activation de XLA pour TensorFlow: {e}")
        return False
