"""
Package pour les optimiseurs et schedulers personnalisés.
"""

from ai_trading.optim.optimizers import Adam, RAdam
from ai_trading.optim.schedulers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_one_cycle_schedule,
    get_polynomial_decay_schedule_with_warmup,
)
from ai_trading.optim.critical_operations import (
    use_jit_script,
    fast_matrix_multiply,
    optimize_model_with_compile,
    enable_cudnn_benchmark,
    VectorizedOperations,
    configure_performance_settings,
    benchmark_function,
)
from ai_trading.optim.operation_time_reduction import (
    precalculate_and_cache,
    get_optimal_batch_size,
    PredictionCache,
    get_prediction_cache,
    ParallelOperations,
)

__all__ = [
    # Optimiseurs
    "Adam",
    "RAdam",
    
    # Schedulers
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_one_cycle_schedule",
    
    # Optimisation des opérations critiques
    "use_jit_script",
    "fast_matrix_multiply",
    "optimize_model_with_compile",
    "enable_cudnn_benchmark",
    "VectorizedOperations",
    "configure_performance_settings",
    "benchmark_function",
    
    # Réduction des temps d'opération
    "precalculate_and_cache",
    "get_optimal_batch_size",
    "PredictionCache",
    "get_prediction_cache",
    "ParallelOperations",
]
