"""
Module d'optimisation par offloading CPU/GPU.

Ce module fournit des utilitaires pour décharger les modèles entre CPU et GPU
de manière intelligente lorsque la VRAM est limitée, en utilisant Huggingface Accelerate.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

try:
    from accelerate import (
        infer_auto_device_map,
        init_empty_weights,
        load_checkpoint_and_dispatch,
    )

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_accelerate_available() -> bool:
    """
    Vérifie si Accelerate est disponible.

    Returns:
        bool: True si Accelerate est disponible, False sinon
    """
    return ACCELERATE_AVAILABLE


def check_vram_requirements(
    model: torch.nn.Module, input_size: Tuple[int, ...], batch_size: int = 1
) -> Dict[str, float]:
    """
    Estime les besoins en VRAM pour un modèle donné.

    Args:
        model: Le modèle PyTorch à évaluer
        input_size: La taille d'entrée (sans la dimension de batch)
        batch_size: La taille du batch

    Returns:
        Dict contenant les estimations de mémoire en Mo
    """
    # Calculer la taille du modèle (fonctionne avec ou sans CUDA)
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size / (1024 * 1024)
    
    # Estimer l'espace requis pour les activations et les gradients
    # en général, les activations et gradients peuvent prendre 2-4x la taille du modèle
    # mais cela varie beaucoup selon l'architecture
    forward_pass_estimate = model_size_mb * 3
    
    # Espace total requis
    total_required = model_size_mb + forward_pass_estimate
    
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA n'est pas disponible, estimation basée uniquement sur la taille du modèle"
        )
        return {
            "model_size_mb": model_size_mb,
            "forward_pass_mb": forward_pass_estimate,
            "total_required_mb": total_required,
            "available_mb": 0,
            "free_memory_mb": 0,
            "is_sufficient": False,
        }

    # Nettoyer la mémoire avant l'estimation
    torch.cuda.empty_cache()
    gc.collect()

    # Espace disponible
    available = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    free_memory = torch.cuda.memory_reserved(0) / (1024 * 1024)

    results = {
        "model_size_mb": model_size_mb,
        "forward_pass_mb": forward_pass_estimate,
        "total_required_mb": total_required,
        "available_mb": available,
        "free_memory_mb": free_memory,
        "is_sufficient": available > total_required,
    }

    logger.info(
        f"Estimation VRAM: "
        f"Taille modèle: {model_size_mb:.2f} Mo, "
        f"Forward pass: {forward_pass_estimate:.2f} Mo, "
        f"Total requis: {total_required:.2f} Mo, "
        f"Disponible: {available:.2f} Mo"
    )

    return results


def offload_to_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """
    Décharge un modèle sur CPU pour économiser la VRAM.

    Args:
        model: Le modèle à décharger

    Returns:
        Le modèle déchargé sur CPU
    """
    if not torch.cuda.is_available():
        return model

    logger.info("Déchargement du modèle sur CPU")
    return model.cpu()


def setup_model_offloading(model: torch.nn.Module) -> torch.nn.Module:
    """
    Configure le modèle pour l'offloading en utilisant la méthode standard de PyTorch.
    Utile lorsque Accelerate n'est pas disponible.

    Args:
        model: Le modèle à configurer

    Returns:
        Le modèle configuré
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA n'est pas disponible, le modèle restera sur CPU")
        return model

    logger.info("Configuration du modèle pour l'offloading CPU/GPU standard")

    # Stratégie simple: garder les couches importantes sur GPU, le reste sur CPU
    device_map = {}
    critical_layers = set(["last_layers", "output", "classifier", "head", "decoder"])

    for name, module in model.named_children():
        # Déterminer si c'est une couche critique
        is_critical = any(
            critical_name in name.lower() for critical_name in critical_layers
        )

        if is_critical:
            device_map[name] = "cuda"
            module.to("cuda")
            logger.info(f"Module '{name}' sur CUDA")
        else:
            device_map[name] = "cpu"
            module.to("cpu")
            logger.info(f"Module '{name}' sur CPU")

    return model


def setup_accelerate_offloading(
    model_class,
    checkpoint_path: Optional[str] = None,
    max_memory: Optional[Dict[str, Union[int, str]]] = None,
) -> torch.nn.Module:
    """
    Configure le modèle pour l'offloading avec Accelerate.

    Args:
        model_class: La classe du modèle à initialiser
        checkpoint_path: Chemin vers les poids du modèle pré-entraîné (optionnel)
        max_memory: Dictionnaire spécifiant la mémoire maximale par device

    Returns:
        Le modèle configuré avec offloading
    """
    if not is_accelerate_available():
        logger.error(
            "Accelerate n'est pas disponible. Veuillez l'installer: pip install accelerate"
        )
        raise ImportError("Accelerate est requis pour cette fonctionnalité")

    logger.info("Configuration du modèle pour l'offloading avec Accelerate")

    # Si max_memory n'est pas spécifié, utiliser une configuration automatique
    if max_memory is None:
        cuda_memory = None
        if torch.cuda.is_available():
            # Laisser 2GB de mémoire libre pour la sécurité
            free_memory = torch.cuda.get_device_properties(0).total_memory - 2 * (
                1024**3
            )
            cuda_memory = f"{free_memory // (1024 ** 2)}MB"

        max_memory = {"cpu": "16GB", "cuda": cuda_memory if cuda_memory else "0MB"}

    logger.info(f"Configuration mémoire: {max_memory}")

    # Initialiser un modèle vide
    with init_empty_weights():
        model = model_class()

    # Inférer la carte des périphériques automatiquement
    no_split_module_classes = getattr(model, "no_split_modules", None)
    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    logger.info(f"Carte des périphériques inférée: {device_map}")

    # Charger les poids et répartir le modèle
    if checkpoint_path:
        logger.info(f"Chargement des poids depuis {checkpoint_path}")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint_path,
            device_map=device_map,
            no_split_module_classes=no_split_module_classes,
        )
    else:
        # Si pas de checkpoint, initialiser le modèle normalement
        # et le déplacer manuellement selon la carte
        model = model_class()
        for module_name, device in device_map.items():
            try:
                module = get_module_by_name(model, module_name)
                module.to(device)
                logger.info(f"Module '{module_name}' déplacé vers {device}")
            except Exception as e:
                logger.warning(
                    f"Erreur lors du déplacement du module '{module_name}': {e}"
                )

    return model


def get_module_by_name(model, module_name):
    """
    Récupère un module par son nom dans le modèle.

    Args:
        model: Le modèle contenant le module
        module_name: Le nom du module à récupérer

    Returns:
        Le module correspondant
    """
    if not module_name:
        return model

    names = module_name.split(".")
    module = model
    for name in names:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            raise AttributeError(f"Module {name} non trouvé dans {module}")

    return module


class ModelOffloader:
    """
    Classe utilitaire pour gérer l'offloading CPU/GPU de manière intelligente.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        offload_strategy: str = "auto",
        max_memory: Optional[Dict[str, Union[int, str]]] = None,
        keep_in_cpu: Optional[List[str]] = None,
        keep_in_gpu: Optional[List[str]] = None,
    ):
        """
        Initialise le gestionnaire d'offloading.

        Args:
            model: Le modèle à gérer
            offload_strategy: Stratégie d'offloading ('auto', 'accelerate', 'standard', 'none')
            max_memory: Dictionnaire de mémoire max par device pour Accelerate
            keep_in_cpu: Liste des noms de modules à garder sur CPU
            keep_in_gpu: Liste des noms de modules à garder sur GPU
        """
        self.model = model
        self.strategy = offload_strategy
        self.max_memory = max_memory
        self.keep_in_cpu = keep_in_cpu or []
        self.keep_in_gpu = keep_in_gpu or []

        # Vérifier la disponibilité de CUDA
        self.cuda_available = torch.cuda.is_available()

        # Vérifier si Accelerate est disponible
        self.accelerate_available = is_accelerate_available()

        # Détecter la stratégie automatiquement si nécessaire
        if offload_strategy == "auto":
            self._detect_best_strategy()

        # Appliquer la configuration initiale
        self._configure_offloading()

    def _detect_best_strategy(self):
        """Détecte la meilleure stratégie d'offloading en fonction des ressources."""
        if not self.cuda_available:
            self.strategy = "none"
            logger.info("Stratégie détectée: 'none' (CUDA non disponible)")
            return

        # Estimer les besoins en mémoire
        vram_check = check_vram_requirements(self.model, (1, 1), batch_size=1)

        if vram_check["is_sufficient"]:
            # Assez de VRAM pour tout le modèle
            self.strategy = "none"
            logger.info("Stratégie détectée: 'none' (VRAM suffisante)")
        elif self.accelerate_available:
            # Pas assez de VRAM, mais Accelerate est disponible
            self.strategy = "accelerate"
            logger.info("Stratégie détectée: 'accelerate'")
        else:
            # Pas assez de VRAM et pas d'Accelerate
            self.strategy = "standard"
            logger.info("Stratégie détectée: 'standard'")

    def _configure_offloading(self):
        """Configure l'offloading selon la stratégie choisie."""
        if self.strategy == "none":
            if self.cuda_available:
                self.model.to("cuda")
                logger.info("Modèle entièrement chargé sur GPU")
            else:
                logger.info("Modèle maintenu sur CPU (CUDA non disponible)")

        elif self.strategy == "accelerate":
            if not self.accelerate_available:
                logger.error(
                    "Stratégie 'accelerate' sélectionnée mais Accelerate n'est pas installé"
                )
                logger.warning("Utilisation de la stratégie 'standard' à la place")
                self.strategy = "standard"
                self._configure_offloading()
                return

            model_class = self.model.__class__
            # Sauvegarde temporaire des poids
            temp_dir = Path("ai_trading/info_retour/models/temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_path = temp_dir / "temp_weights.pt"
            torch.save(self.model.state_dict(), temp_path)

            # Reconfigurer avec Accelerate
            self.model = setup_accelerate_offloading(
                model_class, checkpoint_path=str(temp_path), max_memory=self.max_memory
            )

            # Supprimer le fichier temporaire
            if temp_path.exists():
                os.remove(temp_path)

            logger.info("Modèle configuré avec Accelerate pour l'offloading CPU/GPU")

        elif self.strategy == "standard":
            self.model = setup_model_offloading(self.model)
            logger.info("Modèle configuré avec l'offloading standard PyTorch")

    def to_eval_mode(self):
        """
        Prépare le modèle pour l'inférence (évaluation).
        Cette méthode peut appliquer des optimisations spécifiques au mode d'évaluation.
        """
        self.model.eval()

        # Si on est en mode standard, on peut optimiser en chargeant les
        # modules critiques sur GPU et le reste sur CPU
        if self.strategy == "standard":
            for name, module in self.model.named_modules():
                # Les modules de sortie sont généralement nécessaires sur GPU
                is_output = any(
                    x in name.lower()
                    for x in ["output", "head", "decoder", "classifier"]
                )

                # Respecter les contraintes explicites
                if name in self.keep_in_cpu:
                    module.to("cpu")
                elif name in self.keep_in_gpu and self.cuda_available:
                    module.to("cuda")
                # Sinon, appliquer la logique par défaut
                elif is_output and self.cuda_available:
                    module.to("cuda")

        logger.info("Modèle préparé pour l'inférence")
        return self

    def to_train_mode(self):
        """
        Prépare le modèle pour l'entraînement.
        """
        self.model.train()

        # Pour l'entraînement, la configuration dépend de la stratégie
        if self.strategy == "standard" and self.cuda_available:
            # En entraînement standard, on essaie de garder sur GPU les modules
            # qui nécessitent beaucoup de calculs de gradient
            for name, module in self.model.named_modules():
                # Les premières couches et les couches finales sont souvent critiques
                is_critical = (
                    any(
                        x in name.lower()
                        for x in ["input", "embed", "output", "decoder"]
                    )
                    or len(name.split(".")) <= 1  # modules de premier niveau
                )

                # Respecter les contraintes explicites
                if name in self.keep_in_cpu:
                    module.to("cpu")
                elif name in self.keep_in_gpu and self.cuda_available:
                    module.to("cuda")
                # Sinon, appliquer la logique par défaut
                elif is_critical and self.cuda_available:
                    module.to("cuda")

        logger.info("Modèle préparé pour l'entraînement")
        return self

    def optimize_memory(self):
        """
        Optimise manuellement la mémoire en libérant les caches.
        """
        # Libérer la mémoire CUDA si disponible
        if self.cuda_available:
            torch.cuda.empty_cache()

        # Collecter les objets inutilisés
        gc.collect()

        logger.info("Mémoire optimisée")

    def __call__(self, *args, **kwargs):
        """
        Appelle le modèle sous-jacent avec les arguments donnés.
        """
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    # Configuration du logger pour les tests
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Vérifier si Accelerate est disponible
    logger.info(f"Accelerate disponible: {is_accelerate_available()}")
