#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour utiliser torch.compile de manière compatible sur toutes les plateformes.
Fournit une alternative qui fonctionne même sur Windows où torch.compile n'est pas
encore officiellement supporté.
"""

import logging
import platform
from typing import Dict, Any, Optional, Callable, Union, List, Type
import functools

import torch
import torch.nn as nn

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérifier si torch.compile est disponible et si nous sommes sur Windows
IS_WINDOWS = platform.system() == "Windows"
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")

# Mode de compatibilité pour torch.compile
class CrossPlatformCompiler:
    """Classe pour gérer la compilation torch de manière compatible sur toutes les plateformes."""
    
    @staticmethod
    def is_available() -> bool:
        """
        Vérifie si torch.compile est réellement disponible sur cette plateforme.
        
        Returns:
            bool: True si torch.compile est utilisable
        """
        if not TORCH_COMPILE_AVAILABLE:
            return False
        
        if IS_WINDOWS:
            # Sur Windows, torch.compile n'est pas supporté officiellement
            try:
                # Tenter de compiler une fonction triviale pour voir si ça fonctionne quand même
                # (si une future version le supporte ou si l'utilisateur a une configuration spéciale)
                @torch.compile
                def test_func(x):
                    return x + 1
                    
                test_func(torch.tensor([1.0]))
                return True
            except Exception:
                return False
        
        return True
    
    @staticmethod
    def compile(
        model: Union[nn.Module, Callable],
        mode: Optional[str] = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[nn.Module, Callable]:
        """
        Version compatible multiplateforme de torch.compile.
        Sur les plateformes non supportées, renvoie le modèle d'origine sans compilation.
        
        Args:
            model: Module PyTorch ou fonction à compiler
            mode: Mode de compilation ("default", "reduce-overhead", "max-autotune")
            fullgraph: Si True, utilise le graphe complet pour la compilation
            dynamic: Si True, permet les dimensions dynamiques
            options: Options supplémentaires pour le compilateur
            
        Returns:
            Module compilé ou module d'origine si la compilation n'est pas disponible
        """
        if not CrossPlatformCompiler.is_available():
            if IS_WINDOWS:
                logger.warning("torch.compile n'est pas supporté sur Windows. Le modèle original sera utilisé.")
            else:
                logger.warning("torch.compile n'est pas disponible. Le modèle original sera utilisé.")
            return model
        
        try:
            options = options or {}
            compiled_model = torch.compile(
                model, 
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                **options
            )
            logger.info(f"Modèle {type(model).__name__} compilé avec torch.compile()")
            return compiled_model
        except Exception as e:
            logger.warning(f"Erreur lors de la compilation avec torch.compile: {str(e)}")
            logger.warning("Le modèle original sera utilisé.")
            return model

    @staticmethod
    def compile_function(
        func: Callable,
        mode: Optional[str] = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Décorateur pour compiler des fonctions PyTorch.
        
        Args:
            func: Fonction à compiler
            mode: Mode de compilation
            fullgraph: Si True, utilise le graphe complet pour la compilation
            dynamic: Si True, permet les dimensions dynamiques
            options: Options supplémentaires pour le compilateur
            
        Returns:
            Fonction compilée ou fonction d'origine si la compilation n'est pas disponible
        """
        return CrossPlatformCompiler.compile(func, mode, fullgraph, dynamic, options)

# Alias pour une utilisation simple
compile_model = CrossPlatformCompiler.compile
compile_fn = CrossPlatformCompiler.compile_function
is_compile_available = CrossPlatformCompiler.is_available


def create_jit_function_if_compile_not_available(func):
    """
    Décorateur qui utilise JIT si torch.compile n'est pas disponible.
    
    Args:
        func: Fonction à optimiser
        
    Returns:
        Fonction optimisée avec torch.compile ou torch.jit.script
    """
    if is_compile_available():
        return compile_fn(func)
    else:
        try:
            logger.info(f"torch.compile n'est pas disponible, utilisation de torch.jit.script pour {func.__name__}")
            return torch.jit.script(func)
        except Exception as e:
            logger.warning(f"Échec de l'optimisation de {func.__name__}: {str(e)}")
            return func


def create_optimized_model(model_class: Type[nn.Module], *args, **kwargs) -> nn.Module:
    """
    Crée et optimise automatiquement un modèle avec la meilleure méthode disponible.
    
    Args:
        model_class: Classe du modèle à instancier
        *args, **kwargs: Arguments à passer au constructeur du modèle
        
    Returns:
        Modèle optimisé avec torch.compile si disponible, sinon le modèle original
    """
    model = model_class(*args, **kwargs)
    
    if is_compile_available():
        logger.info(f"Optimisation du modèle {model_class.__name__} avec torch.compile")
        return compile_model(model)
    elif IS_WINDOWS:
        logger.warning(f"torch.compile n'est pas supporté sur Windows. {model_class.__name__} non optimisé.")
    else:
        logger.warning(f"torch.compile n'est pas disponible. {model_class.__name__} non optimisé.")
    
    return model


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimise un modèle pour l'inférence en utilisant la meilleure méthode disponible.
    
    Args:
        model: Modèle PyTorch à optimiser
        
    Returns:
        Modèle optimisé
    """
    model.eval()  # Passer en mode évaluation
    
    # Essayer d'abord avec torch.compile
    if is_compile_available():
        try:
            logger.info("Optimisation du modèle pour l'inférence avec torch.compile")
            return compile_model(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"Échec de l'optimisation avec torch.compile: {str(e)}")
    
    # Si torch.compile n'est pas disponible ou a échoué, essayer avec TorchScript
    try:
        logger.info("Optimisation du modèle pour l'inférence avec torch.jit.script")
        # Utiliser une entrée factice pour capturer le graphe de calcul
        example_forward_input = next(model.parameters()).new_zeros(1, *model.input_size) if hasattr(model, "input_size") else None
        
        # Si nous avons une entrée d'exemple, utiliser trace, sinon script
        if example_forward_input is not None:
            return torch.jit.trace(model, example_forward_input)
        else:
            return torch.jit.script(model)
    except Exception as e:
        logger.warning(f"Échec de l'optimisation avec TorchScript: {str(e)}")
    
    logger.warning("Aucune optimisation n'a pu être appliquée au modèle")
    return model


def optimize_function(func):
    """
    Décorateur qui optimise une fonction PyTorch avec la meilleure méthode disponible.
    
    Args:
        func: Fonction à optimiser
        
    Returns:
        Fonction optimisée
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # Essayer d'optimiser la fonction avec compile si disponible
        if is_compile_available():
            try:
                optimized_fn = compile_fn(func)
                return optimized_fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Échec de l'optimisation avec torch.compile: {str(e)}")
        
        # Si compile n'est pas disponible ou a échoué, essayer avec TorchScript
        try:
            optimized_fn = torch.jit.script(func)
            return optimized_fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Échec de l'optimisation avec TorchScript: {str(e)}")
        
        # Si tout échoue, utiliser la fonction originale
        return func(*args, **kwargs)
    
    return wrapped


# Exemples et tests
if __name__ == "__main__":
    # Vérifier la disponibilité de torch.compile
    print(f"Plateforme: {platform.system()}")
    print(f"torch.compile disponible: {is_compile_available()}")
    
    # Exemple de fonction
    @optimize_function
    def example_function(x: torch.Tensor) -> torch.Tensor:
        return x.sin() + x.cos()
    
    # Exemple de modèle
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 1)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # Créer et optimiser le modèle
    model = create_optimized_model(ExampleModel)
    
    # Tester le modèle
    test_input = torch.randn(5, 10)
    output = model(test_input)
    print(f"Forme de sortie: {output.shape}")
    
    # Tester la fonction
    test_input = torch.randn(5)
    output = example_function(test_input)
    print(f"Forme de sortie de la fonction: {output.shape}") 