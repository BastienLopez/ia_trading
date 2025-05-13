# -*- coding: utf-8 -*-
"""
Package pour l'analyse de sentiment et l'optimisation de modèles LLM.

Ce package inclut:
- Optimisation des modèles LLM (quantification, pruning, distillation)
- Analyse de sentiment pour les données de marché
"""

from ai_trading.llm.optimization import (
    ModelOptimizer,
    QuantizationType,
    free_gpu_memory,
    get_memory_info,
    print_model_info,
)

__all__ = [
    "ModelOptimizer",
    "QuantizationType",
    "free_gpu_memory",
    "get_memory_info",
    "print_model_info",
]
