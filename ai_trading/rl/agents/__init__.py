"""Agents RL importés paresseusement pour isoler les runtimes ML."""

__all__ = ["OptimizedSACAgent"]


def __getattr__(name):
    if name == "OptimizedSACAgent":
        from .sac_agent import OptimizedSACAgent

        return OptimizedSACAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
