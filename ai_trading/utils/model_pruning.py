"""
Module d'utilitaires pour l'élagage (pruning) de modèles PyTorch.

Ce module fournit des fonctions et des classes pour réduire la taille des modèles
en supprimant les poids/neurones non essentiels, ce qui permet d'améliorer l'efficacité
sans perte significative de performance.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Union, Callable, Tuple, Any

logger = logging.getLogger(__name__)


def apply_global_unstructured_pruning(
    model: nn.Module, 
    pruning_method: Callable, 
    amount: float = 0.2,
    parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None
) -> nn.Module:
    """
    Applique un élagage global non structuré au modèle.
    
    Args:
        model: Le modèle PyTorch à élaguer
        pruning_method: La méthode d'élagage à utiliser (ex: prune.L1Unstructured)
        amount: Pourcentage de poids à élaguer (0.2 = 20%)
        parameters_to_prune: Liste des paramètres à élaguer, si None, tous les modules Conv et Linear
        
    Returns:
        Le modèle élagué
    """
    # Si aucun paramètre spécifique n'est fourni, on identifie tous les layers Conv et Linear
    if parameters_to_prune is None:
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
    
    # Appliquer l'élagage global
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=amount,
    )
    
    logger.info(f"Élagage global appliqué à {len(parameters_to_prune)} paramètres avec {amount*100:.1f}% de sparsité")
    
    return model


def apply_layerwise_pruning(
    model: nn.Module, 
    method: str = 'l1', 
    amount: float = 0.2,
    exclude_layers: Optional[List[str]] = None
) -> nn.Module:
    """
    Applique un élagage couche par couche au modèle.
    
    Args:
        model: Le modèle PyTorch à élaguer
        method: Méthode d'élagage ('l1', 'random')
        amount: Pourcentage de poids à élaguer par couche
        exclude_layers: Liste des noms de couches à exclure de l'élagage
        
    Returns:
        Le modèle élagué
    """
    exclude_layers = exclude_layers or []
    pruned_layers = 0
    
    # Sélectionner la méthode d'élagage
    if method == 'l1':
        pruning_method = prune.L1Unstructured
    elif method == 'random':
        pruning_method = prune.RandomUnstructured
    else:
        raise ValueError(f"Méthode d'élagage '{method}' non prise en charge. Utilisez 'l1' ou 'random'.")
    
    # Parcourir tous les modules
    for name, module in model.named_modules():
        # Vérifier si le module doit être élagué
        if any(excluded in name for excluded in exclude_layers):
            logger.info(f"Module '{name}' exclu de l'élagage")
            continue
            
        # Appliquer l'élagage aux couches Conv et Linear
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            pruned_layers += 1
    
    logger.info(f"Élagage par couche appliqué à {pruned_layers} couches avec {amount*100:.1f}% de sparsité")
    
    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Rend l'élagage permanent en remplaçant les paramètres masqués.
    
    Args:
        model: Le modèle avec des masques d'élagage temporaires
        
    Returns:
        Le modèle avec un élagage permanent
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                # Le paramètre n'a pas de masque d'élagage
                pass
    
    logger.info("Élagage rendu permanent sur le modèle")
    
    return model


def get_model_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calcule la sparsité (proportion de zéros) du modèle.
    
    Args:
        model: Le modèle PyTorch
        
    Returns:
        Dictionnaire contenant les statistiques de sparsité
    """
    total_params = 0
    zero_params = 0
    layer_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            weight = module.weight
            
            # Calculer le nombre total de paramètres
            layer_total = weight.numel()
            total_params += layer_total
            
            # Calculer le nombre de paramètres à zéro
            layer_zeros = (weight == 0).sum().item()
            zero_params += layer_zeros
            
            # Calculer la sparsité de la couche
            layer_sparsity = layer_zeros / layer_total if layer_total > 0 else 0
            layer_stats[name] = layer_sparsity
    
    # Calculer la sparsité globale
    global_sparsity = zero_params / total_params if total_params > 0 else 0
    
    return {
        "global_sparsity": global_sparsity,
        "total_params": total_params,
        "zero_params": zero_params,
        "layer_sparsity": layer_stats
    }


def iterative_pruning(
    model: nn.Module,
    train_fn: Callable[[nn.Module], float],
    prune_amount_per_iteration: float = 0.1,
    max_iterations: int = 3,
    threshold_accuracy: float = 0.01,
    method: str = 'l1'
) -> nn.Module:
    """
    Effectue un élagage itératif avec réentraînement entre les itérations.
    
    Args:
        model: Le modèle PyTorch à élaguer
        train_fn: Fonction d'entraînement qui prend le modèle et retourne la perte/précision
        prune_amount_per_iteration: Pourcentage à élaguer à chaque itération
        max_iterations: Nombre maximum d'itérations d'élagage
        threshold_accuracy: Perte de précision maximale tolérée
        method: Méthode d'élagage ('l1', 'random')
        
    Returns:
        Le modèle élagué
    """
    # Sélectionner la méthode d'élagage
    if method == 'l1':
        pruning_method = prune.L1Unstructured
    elif method == 'random':
        pruning_method = prune.RandomUnstructured
    else:
        raise ValueError(f"Méthode d'élagage '{method}' non prise en charge. Utilisez 'l1' ou 'random'.")
    
    # Évaluer le modèle initial
    initial_performance = train_fn(model)
    current_performance = initial_performance
    
    logger.info(f"Performance initiale avant élagage: {initial_performance:.4f}")
    
    # Identifier les paramètres à élaguer
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Effectuer l'élagage itératif
    for iteration in range(max_iterations):
        logger.info(f"Itération d'élagage {iteration+1}/{max_iterations}")
        
        # Appliquer l'élagage
        apply_global_unstructured_pruning(
            model, 
            pruning_method=pruning_method, 
            amount=prune_amount_per_iteration,
            parameters_to_prune=parameters_to_prune
        )
        
        # Réentraîner le modèle
        new_performance = train_fn(model)
        
        # Vérifier si la performance est encore acceptable
        if initial_performance - new_performance > threshold_accuracy:
            logger.warning(f"Performance trop dégradée: {new_performance:.4f} vs. initial {initial_performance:.4f}")
            logger.warning(f"Annulation de la dernière itération d'élagage")
            break
        
        current_performance = new_performance
        logger.info(f"Performance après élagage et réentraînement: {current_performance:.4f}")
        
        # Afficher les statistiques de sparsité
        sparsity_stats = get_model_sparsity(model)
        logger.info(f"Sparsité globale: {sparsity_stats['global_sparsity']*100:.2f}%")
    
    # Rendre l'élagage permanent
    make_pruning_permanent(model)
    
    return model


def sensitivity_analysis(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    prune_amounts: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    method: str = 'l1'
) -> Dict[str, Dict[float, float]]:
    """
    Effectue une analyse de sensibilité pour déterminer l'impact de l'élagage sur chaque couche.
    
    Args:
        model: Le modèle PyTorch à analyser
        eval_fn: Fonction d'évaluation qui prend le modèle et retourne la performance
        prune_amounts: Liste des pourcentages d'élagage à tester
        method: Méthode d'élagage ('l1', 'random')
        
    Returns:
        Dictionnaire avec les résultats de sensibilité par couche
    """
    # Sélectionner la méthode d'élagage
    if method == 'l1':
        pruning_method = prune.L1Unstructured
    elif method == 'random':
        pruning_method = prune.RandomUnstructured
    else:
        raise ValueError(f"Méthode d'élagage '{method}' non prise en charge. Utilisez 'l1' ou 'random'.")
    
    # Évaluer la performance initiale
    initial_performance = eval_fn(model)
    logger.info(f"Performance initiale: {initial_performance:.4f}")
    
    # Sauvegarder l'état initial du modèle
    initial_state = {name: param.clone() for name, param in model.state_dict().items()}
    
    # Analyser la sensibilité pour chaque couche
    sensitivity_results = {}
    
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        
        layer_sensitivity = {}
        logger.info(f"Analyse de sensibilité pour la couche: {name}")
        
        for amount in prune_amounts:
            # Restaurer le modèle à son état initial
            model.load_state_dict(initial_state)
            
            # Appliquer l'élagage uniquement à cette couche
            prune.l1_unstructured(module, name='weight', amount=amount)
            
            # Évaluer la performance
            performance = eval_fn(model)
            performance_drop = initial_performance - performance
            
            layer_sensitivity[amount] = performance_drop
            logger.info(f"  Élagage {amount*100:.1f}%: Baisse de performance de {performance_drop:.4f}")
            
            # Supprimer l'élagage pour cette couche
            prune.remove(module, 'weight')
        
        sensitivity_results[name] = layer_sensitivity
    
    # Restaurer le modèle à son état initial
    model.load_state_dict(initial_state)
    
    return sensitivity_results


class PruningScheduler:
    """
    Planificateur d'élagage progressif pendant l'entraînement.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        pruning_method: Callable = prune.L1Unstructured,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        start_epoch: int = 0,
        end_epoch: int = 10,
        prune_frequency: int = 1,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None
    ):
        """
        Initialise le planificateur d'élagage.
        
        Args:
            model: Le modèle PyTorch à élaguer
            pruning_method: La méthode d'élagage à utiliser
            initial_sparsity: Sparsité initiale (0.0 = aucun élagage)
            final_sparsity: Sparsité finale cible
            start_epoch: Époque à laquelle commencer l'élagage
            end_epoch: Époque à laquelle terminer l'élagage
            prune_frequency: Fréquence d'élagage en époques
            parameters_to_prune: Liste des paramètres à élaguer
        """
        self.model = model
        self.pruning_method = pruning_method
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.prune_frequency = prune_frequency
        
        # Si aucun paramètre spécifique n'est fourni, identifier tous les modules Conv et Linear
        if parameters_to_prune is None:
            self.parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    self.parameters_to_prune.append((module, 'weight'))
        else:
            self.parameters_to_prune = parameters_to_prune
        
        # Initialiser l'état
        self.current_sparsity = initial_sparsity
        self.last_prune_epoch = -1
        
        logger.info(f"Planificateur d'élagage initialisé: {initial_sparsity*100:.1f}% → {final_sparsity*100:.1f}%")
        logger.info(f"Époques: {start_epoch} → {end_epoch} (fréquence: {prune_frequency})")
        logger.info(f"Paramètres à élaguer: {len(self.parameters_to_prune)}")
    
    def step(self, epoch: int) -> None:
        """
        Effectue une étape d'élagage si nécessaire.
        
        Args:
            epoch: L'époque actuelle
        """
        # Vérifier si nous devons effectuer un élagage à cette époque
        if (epoch < self.start_epoch or 
            epoch > self.end_epoch or 
            (epoch - self.start_epoch) % self.prune_frequency != 0 or
            epoch == self.last_prune_epoch):
            return
        
        # Calculer la sparsité cible pour cette époque
        if self.end_epoch == self.start_epoch:
            target_sparsity = self.final_sparsity
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            target_sparsity = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
        
        # Calculer la quantité à élaguer pour cette étape
        if self.current_sparsity == 0:
            # Cas spécial pour le premier élagage
            amount = target_sparsity
        else:
            # Calculer la quantité relative pour atteindre la sparsité cible
            amount = 1.0 - (1.0 - target_sparsity) / (1.0 - self.current_sparsity)
        
        # Appliquer l'élagage
        apply_global_unstructured_pruning(
            self.model,
            pruning_method=self.pruning_method,
            amount=amount,
            parameters_to_prune=self.parameters_to_prune
        )
        
        # Mettre à jour l'état
        self.current_sparsity = target_sparsity
        self.last_prune_epoch = epoch
        
        logger.info(f"Époque {epoch}: Élagage appliqué, sparsité actuelle: {self.current_sparsity*100:.2f}%")
    
    def finalize(self) -> None:
        """
        Finalise l'élagage en le rendant permanent.
        """
        make_pruning_permanent(self.model)
        logger.info("Élagage finalisé et rendu permanent")


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)
    
    # Créer un modèle simple pour le test
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Appliquer un élagage simple
    model = apply_layerwise_pruning(model, method='l1', amount=0.3)
    
    # Obtenir les statistiques de sparsité
    sparsity_stats = get_model_sparsity(model)
    logger.info(f"Sparsité globale: {sparsity_stats['global_sparsity']*100:.2f}%")
    logger.info(f"Nombre total de paramètres: {sparsity_stats['total_params']}")
    logger.info(f"Nombre de paramètres à zéro: {sparsity_stats['zero_params']}")
    
    # Rendre l'élagage permanent
    model = make_pruning_permanent(model) 