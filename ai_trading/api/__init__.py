"""Point d'entree public et stable de l'API Trading."""

from .application import get_app, run

__all__ = ["get_app", "run"]


def __getattr__(name: str):
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
