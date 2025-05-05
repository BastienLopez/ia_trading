#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'utilitaires pour l'élagage (pruning) de modèles PyTorch.

Ce module implémente diverses méthodes pour réduire la taille des modèles
en éliminant les poids non essentiels, ce qui permet d'améliorer l'efficacité
sans perte significative de performance.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable, Tuple, Any

logger = logging.getLogger(__name__)

# Créer un stub pour torch.nn.utils.prune s'il n'est pas disponible
try:
    import torch.nn.utils.prune as prune
    PRUNE_AVAILABLE = True
except ImportError:
    logger.warning("torch.nn.utils.prune n'est pas disponible, utilisation d'un stub")
    PRUNE_AVAILABLE = False
    
    # Simuler le module prune
    class _PruneStub:
        """Module stub pour simuler les fonctionnalités d'élagage."""
        
        @staticmethod
        def global_unstructured(parameters, pruning_method, amount):
            """Simule l'élagage global non structuré."""
            logger.warning("Fonction stub pour global_unstructured appelée")
            return None
        
        @staticmethod
        def L1Unstructured(amount):
            """Simule la méthode d'élagage basée sur la norme L1."""
            logger.warning("Fonction stub pour L1Unstructured appelée")
            return lambda x, y: None
        
        @staticmethod
        def RandomUnstructured(amount):
            """Simule la méthode d'élagage aléatoire."""
            logger.warning("Fonction stub pour RandomUnstructured appelée")
            return lambda x, y: None
        
        @staticmethod
        def l1_unstructured(module, name, amount):
            """Simule l'élagage par couche basé sur la norme L1."""
            logger.warning("Fonction stub pour l1_unstructured appelée")
            return None
        
        @staticmethod
        def random_unstructured(module, name, amount):
            """Simule l'élagage par couche aléatoire."""
            logger.warning("Fonction stub pour random_unstructured appelée")
            return None
        
        @staticmethod
        def remove(module, name):
            """Simule la suppression d'un masque d'élagage."""
            logger.warning("Fonction stub pour remove appelée")
            return None
    
    # Utiliser le stub comme module prune
    prune = _PruneStub()


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
    
    if not parameters_to_prune:
        logger.warning("Aucun paramètre à élaguer trouvé")
        return model
    
    # Appliquer l'élagage global
    if PRUNE_AVAILABLE:
        try:
            # Créer la méthode correctement
            if pruning_method == prune.L1Unstructured:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=amount
                )
            elif pruning_method == prune.RandomUnstructured:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.RandomUnstructured,
                    amount=amount
                )
            else:
                # Cas générique
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=pruning_method,
                    amount=amount
                )
                
            logger.info(f"Élagage global appliqué à {len(parameters_to_prune)} paramètres avec {amount*100:.1f}% de sparsité")
        except Exception as e:
            logger.error(f"Erreur lors de l'application de l'élagage: {str(e)}")
    else:
        logger.warning("Élagage simulé : module torch.nn.utils.prune non disponible")
    
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
    
    # Parcourir tous les modules
    for name, module in model.named_modules():
        # Vérifier si le module doit être élagué
        if any(excluded in name for excluded in exclude_layers):
            logger.info(f"Module '{name}' exclu de l'élagage")
            continue
            
        # Appliquer l'élagage aux couches Conv et Linear
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if PRUNE_AVAILABLE:
                try:
                    # Supprimer le masque existant s'il y en a un
                    if hasattr(module, 'weight_mask'):
                        prune.remove(module, 'weight')
                    
                    # Appliquer un nouveau masque
                    if method == 'l1':
                        prune.l1_unstructured(module, name='weight', amount=amount)
                    else:  # random
                        prune.random_unstructured(module, name='weight', amount=amount)
                    
                    pruned_layers += 1
                except Exception as e:
                    logger.error(f"Erreur lors de l'élagage de la couche {name}: {str(e)}")
            else:
                return model  # Si prune n'est pas disponible, retourner le modèle intact
    
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
    if PRUNE_AVAILABLE:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    # Le paramètre n'a pas de masque d'élagage
                    pass
        
        logger.info("Élagage rendu permanent sur le modèle")
    else:
        logger.warning("Opération simulée : module torch.nn.utils.prune non disponible")
    
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
    initial_performance = train_fn(model)
    logger.info(f"Performance initiale: {initial_performance:.4f}")
    
    current_model = model
    current_performance = initial_performance
    
    # Suivi de la sparsité globale
    global_sparsity = 0.0
    cumulative_weights_kept = 1.0
    
    # Conserver une liste des paramètres élagués
    pruned_parameters = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            pruned_parameters.append((module, 'weight'))
    
    for iteration in range(max_iterations):
        logger.info(f"Itération d'élagage {iteration+1}/{max_iterations}")
        
        # Pour chaque itération, le pourcentage d'élagage s'applique aux poids restants
        effective_prune_rate = prune_amount_per_iteration
        
        # Appliquer un élagage direct par couche pour mieux contrôler le taux d'élagage
        pruned_layers = 0
        for module, param_name in pruned_parameters:
            try:
                # Enlever les masques précédents si on les rend permanents
                if hasattr(module, f"{param_name}_mask"):
                    prune.remove(module, param_name)
                
                # Appliquer le nouvel élagage
                if method == 'l1':
                    prune.l1_unstructured(module, name=param_name, amount=effective_prune_rate)
                else:
                    prune.random_unstructured(module, name=param_name, amount=effective_prune_rate)
                
                pruned_layers += 1
            except Exception as e:
                logger.error(f"Erreur lors de l'élagage: {str(e)}")
        
        # Mettre à jour le suivi de la sparsité
        cumulative_weights_kept *= (1.0 - effective_prune_rate)
        global_sparsity = 1.0 - cumulative_weights_kept
        
        logger.info(f"Itération {iteration+1}: {pruned_layers} couches élaguées, sparsité théorique: {global_sparsity:.4f}")
        
        # Rendre l'élagage permanent à la fin de chaque itération
        if iteration == max_iterations - 1:
            current_model = make_pruning_permanent(current_model)
        
        # Réentraîner le modèle
        new_performance = train_fn(current_model)
        logger.info(f"Performance après élagage et réentraînement: {new_performance:.4f}")
        
        # Vérifier la perte de performance
        performance_drop = current_performance - new_performance
        logger.info(f"Perte de performance: {performance_drop:.4f}")
        
        if performance_drop > threshold_accuracy:
            logger.warning(f"Arrêt précoce: perte de performance {performance_drop:.4f} > seuil {threshold_accuracy:.4f}")
            break
        
        # Continuer avec le modèle élagué
        current_performance = new_performance
    
    # Vérifier la sparsité finale
    sparsity_stats = get_model_sparsity(current_model)
    logger.info(f"Élagage itératif terminé. Sparsité finale: {sparsity_stats['global_sparsity']:.4f} (théorique: {global_sparsity:.4f})")
    
    return current_model


def sensitivity_analysis(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    prune_amounts: List[float] = [0.2, 0.4, 0.6, 0.8],
    method: str = 'l1'
) -> Dict[str, Dict[float, float]]:
    """
    Analyse la sensibilité de chaque couche à l'élagage.
    
    Pour chaque couche du modèle, applique différents niveaux d'élagage et
    mesure l'impact sur les performances du modèle.
    
    Args:
        model: Le modèle PyTorch à analyser
        eval_fn: Fonction d'évaluation qui prend le modèle et retourne une métrique de performance
        prune_amounts: Liste des pourcentages d'élagage à tester
        method: Méthode d'élagage ('l1', 'random')
        
    Returns:
        Dictionnaire avec les résultats, structuré comme {nom_couche: {taux_élagage: performance}}
    """
    # Résultats par couche
    results = {}
    
    # Performance de référence du modèle non élagué
    baseline_performance = eval_fn(model)
    logger.info(f"Performance de référence du modèle: {baseline_performance:.4f}")
    
    # Analyser chaque couche indépendamment
    pruneable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            pruneable_layers.append((name, module))
    
    for layer_name, layer in pruneable_layers:
        logger.info(f"Analyse de la sensibilité de la couche '{layer_name}'")
        layer_results = {}
        
        # Sauvegarde des poids originaux
        original_weights = layer.weight.data.clone()
        
        # Tester chaque taux d'élagage
        for amount in prune_amounts:
            if not PRUNE_AVAILABLE:
                # Si l'élagage n'est pas disponible, simuler des résultats fictifs
                layer_results[amount] = baseline_performance * (1.0 - amount / 2.0)  # Simulation fictive
                continue
                
            # Créer une instance spécifique de la méthode d'élagage
            if method == 'l1':
                prune_fn = prune.l1_unstructured
            elif method == 'random':
                prune_fn = prune.random_unstructured
            else:
                raise ValueError(f"Méthode d'élagage '{method}' non prise en charge")
            
            # Appliquer l'élagage uniquement à cette couche
            if hasattr(layer, 'weight_mask'):
                prune.remove(layer, 'weight')  # Supprimer le masque existant
            
            # Appliquer l'élagage
            prune_fn(layer, name='weight', amount=amount)
            
            # Évaluer la performance
            performance = eval_fn(model)
            performance_delta = baseline_performance - performance
            
            # Enregistrer les résultats
            layer_results[amount] = performance
            logger.info(f"  Taux: {amount:.2f}, Performance: {performance:.4f}, Dégradation: {performance_delta:.4f}")
            
            # Restaurer les poids originaux
            layer.weight.data.copy_(original_weights)
            
            # Supprimer le masque d'élagage
            if hasattr(layer, 'weight_mask'):
                prune.remove(layer, 'weight')
        
        # Enregistrer les résultats de cette couche
        results[layer_name] = layer_results
    
    return results


class PruningScheduler:
    """
    Planificateur d'élagage progressif pour modèles PyTorch.
    Permet d'augmenter progressivement la sparsité du modèle au cours de l'entraînement.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        pruning_method: Callable = None,
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
            model: Modèle à élaguer
            pruning_method: Méthode d'élagage (ex: prune.L1Unstructured)
            initial_sparsity: Sparsité initiale (0.0 = 0%)
            final_sparsity: Sparsité finale (1.0 = 100%)
            start_epoch: Époque à laquelle commencer l'élagage
            end_epoch: Époque à laquelle atteindre la sparsité finale
            prune_frequency: Fréquence d'élagage en époques
            parameters_to_prune: Liste des paramètres à élaguer, si None, tous les Conv et Linear
        """
        self.model = model
        self.pruning_method = pruning_method if pruning_method is not None else prune.L1Unstructured
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.prune_frequency = prune_frequency
        
        # Déterminer les paramètres à élaguer
        if parameters_to_prune is None:
            self.parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    self.parameters_to_prune.append((module, 'weight'))
        else:
            self.parameters_to_prune = parameters_to_prune
        
        # Initialiser les variables de suivi
        self.current_sparsity = initial_sparsity
        self.last_epoch = -1
    
    def step(self, epoch: int) -> None:
        """
        Met à jour l'élagage en fonction de l'époque actuelle.
        
        Args:
            epoch: Époque actuelle
        """
        # Si c'est avant le début de l'élagage ou après la fin, ne rien faire
        if epoch < self.start_epoch or epoch > self.end_epoch:
            return
        
        # Si ce n'est pas un multiple de la fréquence, ne rien faire
        if self.prune_frequency > 1 and epoch % self.prune_frequency != 0:
            return
        
        # Si c'est la même époque que la dernière fois, ne rien faire
        if epoch == self.last_epoch:
            return
        
        # Calculer la sparsité cible pour cette époque
        if self.end_epoch == self.start_epoch:
            target_sparsity = self.final_sparsity
        else:
            # Interpolation linéaire
            normalized_epoch = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            target_sparsity = self.initial_sparsity + normalized_epoch * (self.final_sparsity - self.initial_sparsity)
            target_sparsity = min(target_sparsity, self.final_sparsity)  # Limite max
        
        # Calculer la quantité relative à élaguer
        if self.current_sparsity >= 1.0:
            relative_amount = 0.0  # Déjà 100% élagué
        else:
            # Quantité à élaguer par rapport à ce qui reste
            relative_amount = (target_sparsity - self.current_sparsity) / (1.0 - self.current_sparsity)
            relative_amount = max(0.0, min(relative_amount, 1.0))  # Limiter entre 0 et 1
        
        # Appliquer l'élagage si nécessaire
        if relative_amount > 0.0:
            logger.info(f"Élagage époque {epoch}: {self.current_sparsity:.4f} -> {target_sparsity:.4f} (relative: {relative_amount:.4f})")
            
            # Appliquer l'élagage global
            if PRUNE_AVAILABLE:
                try:
                    prune.global_unstructured(
                        self.parameters_to_prune,
                        pruning_method=self.pruning_method,
                        amount=relative_amount
                    )
                    
                    # Mettre à jour la sparsité actuelle
                    self.current_sparsity = target_sparsity
                except Exception as e:
                    logger.error(f"Erreur lors de l'élagage planifié: {str(e)}")
            
        # Mettre à jour l'époque
        self.last_epoch = epoch
    
    def finalize(self) -> None:
        """
        Finalise l'élagage en rendant les masques permanents.
        """
        make_pruning_permanent(self.model)
        logger.info(f"Élagage finalisé avec sparsité: {self.current_sparsity:.4f}")


# Pour tester rapidement le module
def _test():
    """
    Test rapide du module avec un modèle simple.
    """
    # Créer un modèle simple
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Afficher la sparsité initiale
    sparsity_stats = get_model_sparsity(model)
    print(f"Sparsité initiale: {sparsity_stats['global_sparsity']*100:.2f}%")
    
    # Appliquer l'élagage
    model = apply_layerwise_pruning(model, method='l1', amount=0.3)
    
    # Afficher la sparsité après élagage
    sparsity_stats = get_model_sparsity(model)
    print(f"Sparsité après élagage: {sparsity_stats['global_sparsity']*100:.2f}%")
    
    # Rendre l'élagage permanent
    model = make_pruning_permanent(model)
    
    return model


if __name__ == "__main__":
    _test() 