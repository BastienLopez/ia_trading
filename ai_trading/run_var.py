"""
Script principal pour exécuter les tests et exemples de Value-at-Risk (VaR).
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chemin du projet
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(command, cwd=None):
    """
    Exécute une commande shell et retourne le code de sortie.

    Args:
        command (list): Commande à exécuter
        cwd (str, optional): Répertoire de travail

    Returns:
        int: Code de sortie (0 = succès)
    """
    try:
        logger.info(f"Exécution de la commande: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=cwd,
        )

        # Afficher la sortie en temps réel
        for line in process.stdout:
            print(line, end="")

        # Attendre la fin du processus
        process.wait()

        return process.returncode
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la commande: {str(e)}")
        return 1


def run_tests():
    """Exécute les tests unitaires de VaR."""
    logger.info("Exécution des tests unitaires de VaR")

    # Chemin du script de test
    test_script = str(PROJECT_ROOT / "ai_trading" / "tests" / "run_var_tests.py")

    # Exécuter avec l'option tests
    return run_command([sys.executable, test_script, "--mode", "tests"])


def run_examples():
    """Exécute les exemples de VaR."""
    logger.info("Exécution des exemples de VaR")

    # Chemin du script de test
    test_script = str(PROJECT_ROOT / "ai_trading" / "tests" / "run_var_tests.py")

    # Exécuter avec l'option example
    return run_command([sys.executable, test_script, "--mode", "example"])


def run_all():
    """Exécute à la fois les tests et les exemples."""
    logger.info("Exécution de tous les tests et exemples de VaR")

    # Chemin du script de test
    test_script = str(PROJECT_ROOT / "ai_trading" / "tests" / "run_var_tests.py")

    # Exécuter avec l'option all
    return run_command([sys.executable, test_script, "--mode", "all"])


def install_dependencies():
    """Installe les dépendances nécessaires."""
    logger.info("Installation des dépendances pour VaR")

    # Liste des packages requis
    requirements = [
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
    ]

    # Installer les dépendances
    command = [sys.executable, "-m", "pip", "install"] + requirements
    return run_command(command)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Exécution des tests et exemples de Value-at-Risk (VaR)"
    )
    parser.add_argument(
        "action",
        choices=["tests", "examples", "all", "install"],
        help="Action à exécuter: tests, examples, all ou install",
    )

    args = parser.parse_args()

    if args.action == "tests":
        return run_tests()
    elif args.action == "examples":
        return run_examples()
    elif args.action == "all":
        return run_all()
    elif args.action == "install":
        return install_dependencies()
    else:
        logger.error(f"Action inconnue: {args.action}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
