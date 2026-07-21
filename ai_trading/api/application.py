"""Facade canonique pour l'application FastAPI historique.

Le depot contient encore `ai_trading/api.py` et le package `ai_trading.api`.
Cette facade fournit un chemin d'import stable sans copier ni deplacer le
module historique avant la migration complete de ses appelants.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


LEGACY_API_PATH = Path(__file__).resolve().parents[1] / "api.py"
_LEGACY_MODULE_NAME = "ai_trading._legacy_api_application"


def _load_legacy_module() -> ModuleType:
    cached = sys.modules.get(_LEGACY_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, LEGACY_API_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger l'application historique: {LEGACY_API_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def run() -> None:
    """Demarre l'application historique via le chemin de package canonique."""
    _load_legacy_module().run()


def get_app() -> object:
    """Retourne l'objet FastAPI historique lorsque le serveur en a besoin."""
    return _load_legacy_module().app
