"""
Module de journalisation avancée pour le projet ai_trading.

Ce module fournit une configuration centralisée pour la journalisation,
avec rotation des fichiers, différents niveaux de logs, et formats personnalisés.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

# Chemins par défaut pour les logs
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "info_retour" / "logs"
DEFAULT_LOG_FILE = LOG_DIR / "ai_trading.log"

# S'assurer que le répertoire de logs existe
os.makedirs(LOG_DIR, exist_ok=True)

# Niveaux de logging personnalisés
TRACE = 5  # Niveau plus détaillé que DEBUG
logging.addLevelName(TRACE, "TRACE")

# Formats des logs
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
JSON_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# Configuration par défaut
_default_config = {
    "level": logging.INFO,
    "format": DEFAULT_FORMAT,
    "file_handler": True,
    "console_handler": True,
    "json_handler": False,
    "max_bytes": 10 * 1024 * 1024,  # 10 Mo
    "backup_count": 10,
    "propagate": False,
}


class JSONFormatter(logging.Formatter):
    """
    Formatteur qui produit des messages de log au format JSON.
    """

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Ajouter des informations sur la localisation du code
        if hasattr(record, "filename") and hasattr(record, "lineno"):
            log_data["location"] = f"{record.filename}:{record.lineno}"

        # Ajouter des informations sur l'exception
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Ajouter des données supplémentaires
        if hasattr(record, "data") and isinstance(record.data, dict):
            log_data["data"] = record.data

        return json.dumps(log_data)


def trace(self, message, *args, **kwargs):
    """
    Log un message au niveau TRACE.
    """
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Ajouter la méthode trace à la classe Logger
logging.Logger.trace = trace


def setup_logging(
    name: str = "ai_trading",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: str = DEFAULT_FORMAT,
    json_format: bool = False,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 10,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure et retourne un logger avec les paramètres spécifiés.

    Args:
        name: Nom du logger
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin du fichier de log, ou None pour désactiver
        format_str: Format du log (chaîne de formatage)
        json_format: True pour activer le format JSON
        console: True pour activer la sortie console
        max_bytes: Taille maximale d'un fichier de log avant rotation
        backup_count: Nombre de fichiers de backup à conserver
        propagate: True pour propager les logs aux loggers parents

    Returns:
        Un logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Supprimer les handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Créer les formateurs
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(format_str)

    # Ajouter la sortie fichier si demandée
    if log_file:
        if not isinstance(log_file, (str, Path)):
            log_file = DEFAULT_LOG_FILE

        # Assurer que le répertoire parent existe
        log_path = Path(log_file)
        os.makedirs(log_path.parent, exist_ok=True)

        # Ajouter un handler avec rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Ajouter la sortie console si demandée
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Obtient un logger configuré avec les paramètres spécifiés.

    Args:
        name: Nom du logger
        config: Configuration du logger (optionnel)

    Returns:
        Un logger configuré
    """
    if config is None:
        config = _default_config.copy()

    log_file = config.get("log_file")
    if log_file is None and config.get("file_handler", True):
        # Utiliser un fichier basé sur le nom du module
        module_name = name.split(".")[-1]
        log_file = LOG_DIR / f"{module_name}.log"

    # Configurer le logger
    return setup_logging(
        name=name,
        level=config.get("level", _default_config["level"]),
        log_file=log_file,
        format_str=config.get("format", _default_config["format"]),
        json_format=config.get("json_handler", _default_config["json_handler"]),
        console=config.get("console_handler", _default_config["console_handler"]),
        max_bytes=config.get("max_bytes", _default_config["max_bytes"]),
        backup_count=config.get("backup_count", _default_config["backup_count"]),
        propagate=config.get("propagate", _default_config["propagate"]),
    )


def log_exceptions(
    logger: Optional[logging.Logger] = None, exit_on_error: bool = False
):
    """
    Décorateur pour logger les exceptions levées par une fonction.

    Args:
        logger: Logger à utiliser (None pour en créer un nouveau)
        exit_on_error: Si True, termine le programme en cas d'exception

    Returns:
        Le décorateur
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception dans {func.__name__}: {str(e)}",
                    exc_info=True,
                )
                if exit_on_error:
                    logger.critical("Arrêt du programme suite à une erreur critique")
                    sys.exit(1)
                raise

        return wrapper

    return decorator


def log_execution_time(
    logger: Optional[logging.Logger] = None, level: int = logging.DEBUG
):
    """
    Décorateur pour logger le temps d'exécution d'une fonction.

    Args:
        logger: Logger à utiliser (None pour en créer un nouveau)
        level: Niveau de log pour le message

    Returns:
        Le décorateur
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.log(
                level,
                f"Exécution de {func.__name__} en {execution_time:.4f} secondes",
            )
            return result

        return wrapper

    return decorator


def configure_all_loggers(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure tous les loggers du package ai_trading avec la configuration spécifiée.

    Args:
        config: Configuration à appliquer à tous les loggers
    """
    if config is None:
        config = _default_config.copy()

    # Configurer le logger racine
    root_logger = get_logger("ai_trading", config)

    # Ajouter des handlers au logger racine
    logging.getLogger("ai_trading").handlers = root_logger.handlers
    logging.getLogger("ai_trading").setLevel(
        config.get("level", _default_config["level"])
    )

    # Désactiver la propagation pour les modules externes
    for logger_name in ["requests", "urllib3", "matplotlib"]:
        logging.getLogger(logger_name).propagate = False


def log_function_call(
    log_args: bool = True, log_result: bool = False, level: int = logging.DEBUG
):
    """
    Décorateur pour logger les appels de fonction avec leurs arguments.

    Args:
        log_args: Si True, log les arguments de la fonction
        log_result: Si True, log le résultat de la fonction
        level: Niveau de log pour le message

    Returns:
        Le décorateur
    """

    def decorator(func):
        logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if log_args:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger.log(level, f"Appel {func.__name__}({signature})")

            result = func(*args, **kwargs)

            if log_result:
                logger.log(level, f"Résultat de {func.__name__}: {result!r}")

            return result

        return wrapper

    return decorator


def handle_exceptions(error_logger=None):
    """
    Configure un hook pour capturer les exceptions non gérées.

    Args:
        error_logger: Logger à utiliser pour les erreurs
    """
    if error_logger is None:
        error_logger = get_logger("ai_trading.error")

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Laisser passer les interruptions clavier (Ctrl+C)
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_logger.critical(
            "Exception non gérée:",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = exception_handler


# Configuration par défaut lors de l'importation du module
configure_all_loggers()


# Fonction simplifiée pour la compatibilité avec les imports existants
def setup_logger(name, level=logging.INFO):
    """
    Configure un logger simple pour la compatibilité.

    Args:
        name: Nom du logger
        level: Niveau de log

    Returns:
        Logger configuré
    """
    return setup_logging(
        name=name, level=level, log_file=LOG_DIR / f"{name}.log", console=True
    )
