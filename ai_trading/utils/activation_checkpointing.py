"""
Module d'optimisation par Activation Checkpointing.

Ce module fournit des utilitaires pour économiser la mémoire VRAM lors de l'entraînement
de modèles profonds en libérant la mémoire des activations intermédiaires et en les
recalculant pendant la passe arrière (backward).
"""

import functools
import logging
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def checkpoint_sequential(functions, segments, input, **kwargs):
    """
    Divise un modèle séquentiel en segments et applique le checkpointing à chaque segment.

    Version simplifiée de torch.utils.checkpoint.checkpoint_sequential avec plus d'options.

    Args:
        functions: Liste de modules ou fonctions à exécuter séquentiellement
        segments: Nombre de segments à diviser le modèle
        input: Entrée du modèle
        **kwargs: Arguments supplémentaires passés à chaque fonction

    Returns:
        Sortie du modèle séquentiel
    """
    if segments == 0:
        # S'il n'y a pas de segments, on exécute simplement le modèle normalement
        for func in functions:
            input = func(input, **kwargs)
        return input

    # Diviser les fonctions en segments
    segment_size = len(functions) // segments
    # S'assurer que tous les segments sont couverts
    if segment_size * segments < len(functions):
        segment_size += 1

    def run_segment(start_idx, end_idx, inp):
        """Exécute un segment du modèle avec ou sans checkpointing."""
        for idx in range(start_idx, min(end_idx, len(functions))):
            inp = functions[idx](inp, **kwargs)
        return inp

    # Itérer sur les segments
    segment_start = 0
    for segment_idx in range(segments):
        segment_end = min(segment_start + segment_size, len(functions))

        # Utiliser le checkpointing pour ce segment
        input = torch.utils.checkpoint.checkpoint(
            functools.partial(run_segment, segment_start, segment_end),
            input,
            preserve_rng_state=kwargs.get("preserve_rng_state", True),
        )

        segment_start = segment_end

    return input


def apply_activation_checkpointing(model, module_types=None, chunk_size=0):
    """
    Applique l'activation checkpointing à un modèle existant.

    Cette fonction remplace les modules spécifiés ou une séquence de modules
    par des versions avec checkpointing activé.

    Args:
        model: Le modèle PyTorch
        module_types: Types de modules à remplacer (ex: nn.TransformerEncoderLayer)
                      Si None, applique à nn.Sequential
        chunk_size: Nombre de chunks pour découper les séquences
                   Si 0, utilise le comportement par défaut

    Returns:
        Le modèle modifié
    """
    # Si aucun type spécifique n'est demandé, chercher des nn.Sequential
    if module_types is None:
        module_types = (nn.Sequential,)
    elif not isinstance(module_types, (list, tuple)):
        module_types = (module_types,)

    # Parcourir tous les modules
    for name, module in list(model.named_children()):
        # Si c'est un module séquentiel et qu'on veut appliquer le checkpointing
        if isinstance(module, nn.Sequential) and nn.Sequential in module_types:
            num_segments = chunk_size if chunk_size > 0 else min(len(module), 3)
            if len(module) <= 1:
                # Pas besoin de checkpointing pour un seul module
                continue

            # Définir une fonction forward avec checkpointing
            def create_checkpointed_sequential(orig_module, num_segments):
                orig_forward = orig_module.forward

                def checkpointed_forward(self, x, *args, **kwargs):
                    return checkpoint_sequential(
                        list(self.children()), segments=num_segments, input=x, **kwargs
                    )

                # Remplacer temporairement la méthode forward
                orig_module.forward = functools.partial(
                    checkpointed_forward, orig_module
                )
                return orig_module

            # Appliquer le checkpointing
            setattr(model, name, create_checkpointed_sequential(module, num_segments))
            logger.info(
                f"Appliquer activation checkpointing à {name} avec {num_segments} segments"
            )

        # Pour les autres types de modules spécifiés
        elif any(isinstance(module, mt) for mt in module_types if mt != nn.Sequential):
            # Créer une fonction wrapée avec checkpointing
            orig_forward = module.forward

            def checkpointed_module_forward(orig_func, self, *args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    orig_func, *args, **kwargs, preserve_rng_state=True
                )

            # Remplacer la méthode forward
            module.forward = functools.partial(
                checkpointed_module_forward, orig_forward, module
            )
            logger.info(
                f"Appliquer activation checkpointing à {name} (module individuel)"
            )

        # Récursivement appliquer aux sous-modules
        else:
            apply_activation_checkpointing(module, module_types, chunk_size)

    return model


class ActivationCheckpointingModifier:
    """
    Classe pour appliquer ou retirer l'activation checkpointing à un modèle.

    Permet de basculer facilement entre mode normal et mode économie mémoire.
    """

    def __init__(self, module_types=None, chunk_size=0):
        """
        Initialise le modificateur.

        Args:
            module_types: Types de modules à traiter
            chunk_size: Nombre de chunks pour les séquences
        """
        self.module_types = module_types or (nn.Sequential,)
        self.chunk_size = chunk_size
        self.original_forwards = {}  # Stocke les méthodes forward originales

    def apply(self, model):
        """
        Applique l'activation checkpointing au modèle.

        Args:
            model: Le modèle PyTorch

        Returns:
            Le modèle modifié
        """
        # Sauvegarder les méthodes forward originales
        self._save_original_forwards(model)

        # Appliquer le checkpointing
        return apply_activation_checkpointing(model, self.module_types, self.chunk_size)

    def remove(self, model):
        """
        Retire l'activation checkpointing du modèle.

        Args:
            model: Le modèle PyTorch

        Returns:
            Le modèle restauré
        """
        # Restaurer les méthodes forward originales
        self._restore_original_forwards(model)

        return model

    def _save_original_forwards(self, model, prefix=""):
        """Sauvegarde récursivement les méthodes forward originales."""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Vérifier si c'est un module cible
            if (
                isinstance(module, nn.Sequential) and nn.Sequential in self.module_types
            ) or any(
                isinstance(module, mt)
                for mt in self.module_types
                if mt != nn.Sequential
            ):
                # Sauvegarder la méthode forward originale si pas déjà fait
                if full_name not in self.original_forwards:
                    self.original_forwards[full_name] = module.forward

            # Récursivement sauvegarder les sous-modules
            self._save_original_forwards(module, full_name)

    def _restore_original_forwards(self, model, prefix=""):
        """Restaure récursivement les méthodes forward originales."""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Restaurer la méthode forward si elle a été sauvegardée
            if full_name in self.original_forwards:
                module.forward = self.original_forwards[full_name]
                logger.info(f"Restauré la méthode forward originale pour {full_name}")

            # Récursivement restaurer les sous-modules
            self._restore_original_forwards(module, full_name)


class CheckpointedModule(nn.Module):
    """
    Wrapper pour appliquer le checkpointing à n'importe quel module.

    Ce wrapper peut être utilisé pour rendre checkpointé n'importe quel module.
    """

    def __init__(self, module, preserve_rng_state=True):
        """
        Initialise le wrapper.

        Args:
            module: Le module à rendre checkpointé
            preserve_rng_state: Si True, préserve l'état RNG entre les appels
        """
        super().__init__()
        self.module = module
        self.preserve_rng_state = preserve_rng_state

    def forward(self, *args, **kwargs):
        """
        Forward pass avec checkpointing activé.

        Args:
            *args, **kwargs: Arguments passés au module

        Returns:
            Sortie du module
        """
        # Checkpoint le module
        return torch.utils.checkpoint.checkpoint(
            self.module, *args, preserve_rng_state=self.preserve_rng_state, **kwargs
        )


def wrap_checkpointed_modules(model, layer_types, preserve_rng_state=True):
    """
    Remplace certains modules d'un modèle par leurs versions checkpointées.

    Args:
        model: Le modèle à modifier
        layer_types: Liste des types de couches à wrapper
        preserve_rng_state: Si True, préserve l'état RNG entre les appels

    Returns:
        Modèle modifié
    """
    for name, module in list(model.named_children()):
        # Si le module est d'un des types spécifiés
        if any(isinstance(module, layer_type) for layer_type in layer_types):
            # Remplacer par un module checkpointé
            setattr(model, name, CheckpointedModule(module, preserve_rng_state))
            logger.info(f"Module {name} remplacé par sa version checkpointée")
        else:
            # Récursivement appliquer aux sous-modules
            wrap_checkpointed_modules(module, layer_types, preserve_rng_state)

    return model


def analyze_memory_usage(model, input_size, batch_size=1, detailed=False):
    """
    Analyse la consommation mémoire d'un modèle.

    Args:
        model: Le modèle à analyser
        input_size: Taille d'une entrée (sans le batch)
        batch_size: Taille du batch
        detailed: Si True, montre les détails par couche

    Returns:
        Dictionnaire avec les informations de consommation mémoire
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA non disponible, l'analyse de mémoire peut être imprécise")

    # Créer une entrée aléatoire
    input_shape = (batch_size,) + tuple(input_size)
    dummy_input = torch.randn(
        input_shape, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Mesurer la mémoire avant
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Passe avant
    outputs = model(dummy_input)

    # Mesurer la mémoire après la passe avant
    torch.cuda.synchronize()
    mem_after_forward = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )

    # Passe arrière
    if isinstance(outputs, tuple):
        loss = outputs[0].sum()
    else:
        loss = outputs.sum()
    loss.backward()

    # Mesurer la mémoire après la passe arrière
    torch.cuda.synchronize()
    mem_after_backward = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )

    # Libérer la mémoire
    del dummy_input, outputs, loss
    torch.cuda.empty_cache()

    # Calculer les consommations
    forward_memory = (mem_after_forward - mem_before) / (1024 * 1024)  # MB
    backward_memory = (mem_after_backward - mem_after_forward) / (1024 * 1024)  # MB
    total_memory = forward_memory + backward_memory

    result = {
        "forward_memory_mb": forward_memory,
        "backward_memory_mb": backward_memory,
        "total_memory_mb": total_memory,
        "batch_size": batch_size,
        "memory_per_sample_mb": total_memory / batch_size,
    }

    if detailed:
        # Analyse détaillée par couche
        layer_memory = {}

        def hook_fn(name):
            def hook(module, input, output):
                input_size = sum(
                    i.nelement() * i.element_size()
                    for i in input
                    if isinstance(i, torch.Tensor)
                )
                output_size = (
                    output.nelement() * output.element_size()
                    if isinstance(output, torch.Tensor)
                    else sum(
                        o.nelement() * o.element_size()
                        for o in output
                        if isinstance(o, torch.Tensor)
                    )
                )
                layer_memory[name] = (input_size + output_size) / (1024 * 1024)  # MB

            return hook

        # Enregistrer des hooks temporaires
        hooks = []
        for name, module in model.named_modules():
            if name and not any(name.endswith(f".{i}") for i in range(10)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Passe avant pour collecter les infos
        dummy_input = torch.randn(
            input_shape, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model(dummy_input)

        # Supprimer les hooks
        for hook in hooks:
            hook.remove()

        # Ajouter les infos détaillées
        sorted_layers = sorted(layer_memory.items(), key=lambda x: x[1], reverse=True)
        result["layer_memory"] = {name: mem for name, mem in sorted_layers}

    return result


def analyze_checkpointing_savings(
    model, input_size, batch_size=1, layer_types=None, chunk_size=3
):
    """
    Analyse les économies de mémoire potentielles avec l'activation checkpointing.

    Args:
        model: Le modèle à analyser
        input_size: Taille d'une entrée (sans le batch)
        batch_size: Taille du batch
        layer_types: Types de couches à checkpointer
        chunk_size: Nombre de chunks pour les séquences

    Returns:
        Dictionnaire avec la comparaison avant/après checkpointing
    """
    # Cloner le modèle pour ne pas modifier l'original
    import copy

    model_orig = copy.deepcopy(model)
    model_checkpointed = copy.deepcopy(model)

    # Placer sur CUDA si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_orig.to(device)
    model_checkpointed.to(device)

    # Analyser la mémoire du modèle original
    logger.info("Analyse du modèle original...")
    orig_memory = analyze_memory_usage(model_orig, input_size, batch_size)

    # Appliquer le checkpointing
    logger.info("Application de l'activation checkpointing...")
    if layer_types is None:
        layer_types = (nn.Sequential,)
    apply_activation_checkpointing(model_checkpointed, layer_types, chunk_size)

    # Analyser la mémoire du modèle avec checkpointing
    logger.info("Analyse du modèle avec checkpointing...")
    checkpointed_memory = analyze_memory_usage(
        model_checkpointed, input_size, batch_size
    )

    # Calculer les économies
    memory_savings = (
        orig_memory["total_memory_mb"] - checkpointed_memory["total_memory_mb"]
    )
    savings_percent = (
        (memory_savings / orig_memory["total_memory_mb"]) * 100
        if orig_memory["total_memory_mb"] > 0
        else 0
    )

    # Nettoyer
    del model_orig, model_checkpointed
    torch.cuda.empty_cache()

    # Résultats
    return {
        "original_memory_mb": orig_memory["total_memory_mb"],
        "checkpointed_memory_mb": checkpointed_memory["total_memory_mb"],
        "memory_savings_mb": memory_savings,
        "savings_percent": savings_percent,
        "original_details": orig_memory,
        "checkpointed_details": checkpointed_memory,
    }


if __name__ == "__main__":
    # Configuration du logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Créer un modèle de test
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
        ),
        nn.Linear(1000, 10),
    )

    # Analyse rapide
    if torch.cuda.is_available():
        model.cuda()
        results = analyze_checkpointing_savings(model, (1000,), batch_size=32)

        logger.info(f"Mémoire originale : {results['original_memory_mb']:.2f} MB")
        logger.info(
            f"Mémoire avec checkpointing : {results['checkpointed_memory_mb']:.2f} MB"
        )
        logger.info(
            f"Économie mémoire : {results['memory_savings_mb']:.2f} MB ({results['savings_percent']:.1f}%)"
        )
