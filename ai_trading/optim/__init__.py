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

__all__ = [
    "Adam",
    "RAdam",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_one_cycle_schedule",
]
