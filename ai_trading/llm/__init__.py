# -*- coding: utf-8 -*-
"""
Package pour l'analyse de sentiment et l'optimisation de modèles LLM.

Ce package inclut:
- Optimisation des modèles LLM (quantification, pruning, distillation)
- Analyse de sentiment pour les données de marché
"""

__all__ = [
    "ModelOptimizer",
    "QuantizationType",
    "free_gpu_memory",
    "get_memory_info",
    "print_model_info",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from ai_trading.llm import optimization

    return getattr(optimization, name)
