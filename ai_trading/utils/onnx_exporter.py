#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'exportation de modèles vers le format ONNX.
Permet de convertir des modèles PyTorch et TensorFlow vers ONNX pour
une inférence plus rapide et portable.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Vérification des dépendances nécessaires
try:
    import onnx
    import onnxruntime as ort

    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False
    logging.warning(
        "ONNX n'est pas installé. Exécutez 'pip install onnx onnxruntime' pour l'installer."
    )

try:
    import tensorflow as tf

    HAVE_TF = True
except ImportError:
    HAVE_TF = False
    logging.warning(
        "TensorFlow n'est pas installé. L'export des modèles TF vers ONNX ne sera pas disponible."
    )

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelInfo:
    """Classe pour stocker les informations sur un modèle exporté."""

    def __init__(
        self,
        model_type: str,
        input_shape: Tuple,
        output_shape: Tuple,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Optional[Dict] = None,
    ):
        """
        Initialise les informations du modèle.

        Args:
            model_type: Type du modèle ('pytorch', 'tensorflow', etc.)
            input_shape: Forme des entrées
            output_shape: Forme des sorties
            input_names: Noms des tenseurs d'entrée
            output_names: Noms des tenseurs de sortie
            dynamic_axes: Axes dynamiques pour les entrées/sorties de taille variable
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convertit les informations du modèle en dictionnaire."""
        return {
            "model_type": self.model_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "dynamic_axes": self.dynamic_axes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Crée une instance à partir d'un dictionnaire."""
        return cls(
            model_type=data["model_type"],
            input_shape=tuple(data["input_shape"]),
            output_shape=tuple(data["output_shape"]),
            input_names=data["input_names"],
            output_names=data["output_names"],
            dynamic_axes=data.get("dynamic_axes"),
        )


class ONNXExporter:
    """
    Classe pour exporter des modèles PyTorch et TensorFlow vers le format ONNX.
    """

    def __init__(
        self,
        output_dir: str = "onnx_models",
        device: str = "cpu",
        opset_version: int = 12,
    ):
        """
        Initialise l'exporteur ONNX.

        Args:
            output_dir: Répertoire de sortie pour les modèles ONNX
            device: Appareil pour l'inférence ('cpu' ou 'cuda')
            opset_version: Version de l'opset ONNX à utiliser
        """
        if not HAVE_ONNX:
            raise ImportError(
                "ONNX n'est pas installé. Exécutez 'pip install onnx onnxruntime' pour l'installer."
            )

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.opset_version = opset_version

        logger.info(
            f"ONNXExporter initialisé avec le répertoire de sortie: {output_dir}"
        )
        logger.info(
            f"Utilisation de l'appareil: {self.device}, version d'opset: {opset_version}"
        )

    def export_pytorch_model(
        self,
        model: nn.Module,
        input_shape: Tuple,
        model_name: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict] = None,
        check_model: bool = True,
    ) -> str:
        """
        Exporte un modèle PyTorch vers ONNX.

        Args:
            model: Modèle PyTorch à exporter
            input_shape: Forme des entrées (sans le batch)
            model_name: Nom du modèle exporté
            input_names: Noms des tenseurs d'entrée
            output_names: Noms des tenseurs de sortie
            dynamic_axes: Axes dynamiques pour les entrées/sorties de taille variable
            check_model: Vérifier la validité du modèle ONNX après l'export

        Returns:
            Chemin vers le modèle ONNX exporté
        """
        # S'assurer que le modèle est en mode évaluation
        model.eval()
        model.to(self.device)

        # Créer un exemple d'entrée
        dummy_input = torch.randn(1, *input_shape, device=self.device)

        # Définir les noms des entrées/sorties si non spécifiés
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Préparer les axes dynamiques s'ils sont fournis
        if dynamic_axes is None and len(input_shape) > 1:
            # Par défaut, considérer le premier axe comme dynamique (batch_size)
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"},
            }

        # Chemin du modèle ONNX
        onnx_model_path = os.path.join(self.output_dir, f"{model_name}.onnx")

        # Exporter le modèle
        try:
            logger.info(f"Export du modèle PyTorch vers {onnx_model_path}")

            # Obtenir la forme de sortie en exécutant une inférence
            with torch.no_grad():
                outputs = model(dummy_input)

                # Si le résultat est un tuple, utiliser le premier élément
                if isinstance(outputs, tuple):
                    output_shape = outputs[0].shape[1:]
                    dummy_output = outputs[0]
                else:
                    output_shape = outputs.shape[1:]
                    dummy_output = outputs

            # Exporter le modèle avec torch.onnx
            torch.onnx.export(
                model,
                dummy_input,
                onnx_model_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            # Vérifier le modèle si demandé
            if check_model:
                onnx_model = onnx.load(onnx_model_path)
                onnx.checker.check_model(onnx_model)
                logger.info("Modèle ONNX validé avec succès")

            # Sauvegarder les informations du modèle
            model_info = ModelInfo(
                model_type="pytorch",
                input_shape=input_shape,
                output_shape=(
                    tuple(output_shape)
                    if isinstance(output_shape, torch.Size)
                    else output_shape
                ),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            # Sauvegarder les métadonnées
            self._save_model_info(model_name, model_info)

            logger.info(f"Modèle PyTorch exporté avec succès vers {onnx_model_path}")

            return onnx_model_path

        except Exception as e:
            logger.error(f"Erreur lors de l'export du modèle PyTorch: {str(e)}")
            raise

    def export_tensorflow_model(
        self,
        model: Union[tf.keras.Model, str],
        input_shape: Tuple,
        model_name: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        check_model: bool = True,
    ) -> str:
        """
        Exporte un modèle TensorFlow/Keras vers ONNX.

        Args:
            model: Modèle TensorFlow/Keras ou chemin vers un modèle sauvegardé
            input_shape: Forme des entrées (sans le batch)
            model_name: Nom du modèle exporté
            input_names: Noms des tenseurs d'entrée
            output_names: Noms des tenseurs de sortie
            check_model: Vérifier la validité du modèle ONNX après l'export

        Returns:
            Chemin vers le modèle ONNX exporté
        """
        if not HAVE_TF:
            raise ImportError(
                "TensorFlow n'est pas installé. L'export des modèles TF vers ONNX n'est pas disponible."
            )

        try:
            import tf2onnx
        except ImportError:
            raise ImportError(
                "tf2onnx n'est pas installé. Exécutez 'pip install tf2onnx' pour l'installer."
            )

        # Charger le modèle si c'est un chemin
        if isinstance(model, str):
            model = tf.keras.models.load_model(model)

        # Chemin du modèle ONNX
        onnx_model_path = os.path.join(self.output_dir, f"{model_name}.onnx")

        try:
            logger.info(f"Export du modèle TensorFlow vers {onnx_model_path}")

            # Créer un exemple d'entrée
            spec = (tf.TensorSpec((None, *input_shape), tf.float32),)

            # Convertir le modèle
            onnx_model, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                opset=self.opset_version,
                output_path=onnx_model_path,
            )

            # Vérifier le modèle si demandé
            if check_model:
                onnx.checker.check_model(onnx_model)
                logger.info("Modèle ONNX validé avec succès")

            # Déterminer la forme des sorties
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            outputs = model.predict(dummy_input)

            if isinstance(outputs, list):
                output_shape = outputs[0].shape[1:]
            else:
                output_shape = outputs.shape[1:]

            # Noms des entrées/sorties
            if input_names is None:
                input_names = [inp.name for inp in model.inputs]
            if output_names is None:
                output_names = [out.name for out in model.outputs]

            # Sauvegarder les informations du modèle
            model_info = ModelInfo(
                model_type="tensorflow",
                input_shape=input_shape,
                output_shape=output_shape,
                input_names=input_names,
                output_names=output_names,
            )

            # Sauvegarder les métadonnées
            self._save_model_info(model_name, model_info)

            logger.info(f"Modèle TensorFlow exporté avec succès vers {onnx_model_path}")

            return onnx_model_path

        except Exception as e:
            logger.error(f"Erreur lors de l'export du modèle TensorFlow: {str(e)}")
            raise

    def _save_model_info(self, model_name: str, model_info: ModelInfo) -> None:
        """Sauvegarde les informations du modèle dans un fichier JSON."""
        info_path = os.path.join(self.output_dir, f"{model_name}_info.json")
        with open(info_path, "w") as f:
            json.dump(model_info.to_dict(), f, indent=2)

    def load_model_info(self, model_name: str) -> ModelInfo:
        """Charge les informations d'un modèle depuis un fichier JSON."""
        info_path = os.path.join(self.output_dir, f"{model_name}_info.json")
        with open(info_path, "r") as f:
            data = json.load(f)
        return ModelInfo.from_dict(data)

    def test_onnx_model(
        self,
        model_path: str,
        input_data: Optional[np.ndarray] = None,
        input_shape: Optional[Tuple] = None,
        num_samples: int = 1,
    ) -> Dict[str, Any]:
        """
        Teste un modèle ONNX avec des données aléatoires ou spécifiées.

        Args:
            model_path: Chemin vers le modèle ONNX
            input_data: Données d'entrée (facultatif)
            input_shape: Forme des entrées si input_data n'est pas fourni
            num_samples: Nombre d'échantillons à générer si input_data n'est pas fourni

        Returns:
            Dictionnaire contenant les résultats du test
        """
        # Charger le modèle ONNX
        session = ort.InferenceSession(model_path)

        # Obtenir les noms des entrées
        input_names = [input.name for input in session.get_inputs()]

        # Préparer les données d'entrée
        if input_data is None:
            if input_shape is None:
                raise ValueError("Vous devez fournir input_data ou input_shape")
            # Générer des données aléatoires
            input_data = np.random.randn(num_samples, *input_shape).astype(np.float32)

        # S'assurer que les données sont au format float32
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # Mesurer le temps d'inférence
        import time

        start_time = time.time()

        # Exécuter l'inférence
        inputs = {input_names[0]: input_data}
        outputs = session.run(None, inputs)

        # Calculer le temps d'inférence
        inference_time = (time.time() - start_time) * 1000  # en millisecondes

        return {
            "outputs": outputs,
            "output_shapes": [output.shape for output in outputs],
            "inference_time_ms": inference_time,
            "samples_per_second": num_samples / (inference_time / 1000),
        }

    def compare_pytorch_onnx(
        self,
        pytorch_model: nn.Module,
        onnx_model_path: str,
        input_shape: Tuple,
        num_samples: int = 10,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Compare les sorties d'un modèle PyTorch et de sa version ONNX.

        Args:
            pytorch_model: Modèle PyTorch original
            onnx_model_path: Chemin vers le modèle ONNX
            input_shape: Forme des entrées
            num_samples: Nombre d'échantillons à générer pour le test
            rtol: Tolérance relative pour la comparaison
            atol: Tolérance absolue pour la comparaison

        Returns:
            Dictionnaire contenant les résultats de la comparaison
        """
        # S'assurer que le modèle PyTorch est en mode évaluation
        pytorch_model.eval()
        pytorch_model.to(self.device)

        # Charger le modèle ONNX
        session = ort.InferenceSession(onnx_model_path)
        input_name = session.get_inputs()[0].name

        # Générer des données aléatoires
        np_data = np.random.randn(num_samples, *input_shape).astype(np.float32)
        torch_data = torch.tensor(np_data, device=self.device)

        # Exécuter l'inférence avec PyTorch
        with torch.no_grad():
            torch_outputs = pytorch_model(torch_data)

            # Convertir en numpy
            if isinstance(torch_outputs, tuple):
                torch_outputs = torch_outputs[0]
            torch_outputs = torch_outputs.cpu().numpy()

        # Exécuter l'inférence avec ONNX
        ort_outputs = session.run(None, {input_name: np_data})
        ort_output = ort_outputs[0]

        # Comparer les résultats
        is_close = np.allclose(torch_outputs, ort_output, rtol=rtol, atol=atol)
        max_diff = np.max(np.abs(torch_outputs - ort_output))
        mean_diff = np.mean(np.abs(torch_outputs - ort_output))

        return {
            "is_close": is_close,
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "pytorch_output_shape": torch_outputs.shape,
            "onnx_output_shape": ort_output.shape,
        }


# Exemple d'utilisation
def export_pytorch_policy_model(
    model: nn.Module,
    input_shape: Tuple,
    model_name: str = "policy_model",
    output_dir: str = "onnx_models",
    device: str = "cpu",
) -> str:
    """
    Exporte une fonction de politique PyTorch vers ONNX.

    Args:
        model: Modèle PyTorch à exporter
        input_shape: Forme des entrées (sans le batch)
        model_name: Nom du modèle exporté
        output_dir: Répertoire de sortie
        device: Appareil pour l'inférence ('cpu' ou 'cuda')

    Returns:
        Chemin vers le modèle ONNX exporté
    """
    if not HAVE_ONNX:
        logger.warning("ONNX n'est pas installé. L'export n'est pas disponible.")
        return None

    exporter = ONNXExporter(output_dir=output_dir, device=device)
    return exporter.export_pytorch_model(
        model=model,
        input_shape=input_shape,
        model_name=model_name,
        input_names=["state"],
        output_names=["action"],
        dynamic_axes={"state": {0: "batch_size"}, "action": {0: "batch_size"}},
    )


def export_tensorflow_policy_model(
    model: tf.keras.Model,
    input_shape: Tuple,
    model_name: str = "policy_model_tf",
    output_dir: str = "onnx_models",
) -> str:
    """
    Exporte une fonction de politique TensorFlow vers ONNX.

    Args:
        model: Modèle TensorFlow à exporter
        input_shape: Forme des entrées (sans le batch)
        model_name: Nom du modèle exporté
        output_dir: Répertoire de sortie

    Returns:
        Chemin vers le modèle ONNX exporté
    """
    if not HAVE_ONNX or not HAVE_TF:
        logger.warning(
            "ONNX ou TensorFlow n'est pas installé. L'export n'est pas disponible."
        )
        return None

    exporter = ONNXExporter(output_dir=output_dir)
    return exporter.export_tensorflow_model(
        model=model,
        input_shape=input_shape,
        model_name=model_name,
        input_names=["state"],
        output_names=["action"],
    )
