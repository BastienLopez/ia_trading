"""
Exemple d'utilisation du système de journalisation avancée.

Ce script démontre les différentes fonctionnalités du module de journalisation avancée,
notamment les différents niveaux de logs, les formats personnalisés, la rotation des fichiers,
et l'utilisation des décorateurs.
"""

import os
import sys
import time
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging

from ai_trading.utils.advanced_logging import (
    DETAILED_FORMAT,
    get_logger,
    handle_exceptions,
    log_exceptions,
    log_execution_time,
    log_function_call,
)


# Configuration de différents loggers pour la démonstration
def setup_example_loggers():
    """Configure plusieurs loggers avec différentes configurations pour la démonstration."""

    # Logger standard avec format par défaut
    standard_logger = get_logger("ai_trading.examples.standard")

    # Logger avec format détaillé
    detailed_logger = get_logger(
        "ai_trading.examples.detailed",
        {
            "format": DETAILED_FORMAT,
            "level": logging.DEBUG,
        },
    )

    # Logger JSON
    json_logger = get_logger(
        "ai_trading.examples.json",
        {
            "json_handler": True,
            "level": logging.INFO,
        },
    )

    # Logger fichier uniquement (pas de console)
    file_only_logger = get_logger(
        "ai_trading.examples.file_only",
        {
            "console_handler": False,
            "level": logging.WARNING,
        },
    )

    return {
        "standard": standard_logger,
        "detailed": detailed_logger,
        "json": json_logger,
        "file_only": file_only_logger,
    }


# Fonction qui génère une exception pour la démonstration
@log_exceptions()
def function_with_exception():
    """Fonction qui génère une exception pour démontrer la journalisation des exceptions."""
    print("Génération d'une exception pour démonstration...")
    x = 1 / 0  # Division par zéro
    return x


# Fonction avec mesure du temps d'exécution
@log_execution_time()
def slow_function(delay=1.0):
    """Fonction lente pour démontrer la journalisation du temps d'exécution."""
    print(f"Exécution d'une fonction lente (délai: {delay}s)...")
    time.sleep(delay)
    return delay


# Fonction qui journalise ses arguments et son résultat
@log_function_call(log_args=True, log_result=True)
def calculate_something(a, b, operation="add"):
    """Fonction qui journalise ses arguments et son résultat."""
    print(f"Calcul en cours: {a} {operation} {b}")
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        return None


# Fonction de démonstration pour le niveau TRACE
def demonstrate_trace_level(logger):
    """Démontre l'utilisation du niveau TRACE."""
    logger.trace("Ceci est un message de niveau TRACE (très détaillé)")
    logger.debug("Ceci est un message de niveau DEBUG")
    logger.info("Ceci est un message de niveau INFO")
    logger.warning("Ceci est un message de niveau WARNING")
    logger.error("Ceci est un message de niveau ERROR")
    logger.critical("Ceci est un message de niveau CRITICAL")


# Fonction principale pour la démonstration
def main():
    """Fonction principale qui démontre les différentes fonctionnalités de journalisation."""
    print("Démonstration du système de journalisation avancée")
    print("=" * 80)

    # Configurer le gestionnaire d'exceptions global
    handle_exceptions()

    # Configurer différents loggers
    loggers = setup_example_loggers()

    # Démonstration des différents niveaux de log
    print("\nDémonstration des différents niveaux de log:")
    print("-" * 50)
    for name, logger in loggers.items():
        print(f"\nLogger: {name}")
        demonstrate_trace_level(logger)

    # Démonstration de la journalisation des exceptions
    print("\nDémonstration de la journalisation des exceptions:")
    print("-" * 50)
    try:
        function_with_exception()
    except ZeroDivisionError:
        print("Exception capturée et journalisée")

    # Démonstration de la mesure du temps d'exécution
    print("\nDémonstration de la mesure du temps d'exécution:")
    print("-" * 50)
    slow_function(1.5)

    # Démonstration de la journalisation des arguments et résultats
    print("\nDémonstration de la journalisation des arguments et résultats:")
    print("-" * 50)
    calculate_something(10, 5)
    calculate_something(10, 5, operation="multiply")

    # Informations sur les fichiers de log générés
    log_dir = Path(__file__).parent.parent / "info_retour" / "logs"
    print(f"\nFichiers de log générés dans: {log_dir}")
    if log_dir.exists():
        print("Fichiers de log générés:")
        for log_file in log_dir.glob("*.log"):
            print(f"- {log_file.name} ({log_file.stat().st_size} octets)")

    print("\nDémonstration terminée.")


if __name__ == "__main__":
    main()
