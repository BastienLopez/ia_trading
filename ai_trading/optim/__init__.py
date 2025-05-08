"""
Package pour les optimiseurs et schedulers personnalisés.
"""

from ai_trading.optim.critical_operations import (
    VectorizedOperations,
    benchmark_function,
    configure_performance_settings,
    enable_cudnn_benchmark,
    fast_matrix_multiply,
    optimize_model_with_compile,
    use_jit_script,
)
from ai_trading.optim.operation_time_reduction import (
    ParallelOperations,
    PredictionCache,
    get_optimal_batch_size,
    get_prediction_cache,
    precalculate_and_cache,
)
from ai_trading.optim.optimizers import Adam, RAdam
from ai_trading.optim.schedulers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_one_cycle_schedule,
    get_polynomial_decay_schedule_with_warmup,
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
