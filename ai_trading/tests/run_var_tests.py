"""
Script pour exécuter les tests de l'implémentation VaR et gestion des risques avancée.
"""

import logging
import os
import sys
import unittest
from datetime import datetime

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_all_tests():
    """Exécute tous les tests liés à la VaR et gestion des risques avancée."""
    logger.info("Démarrage des tests de l'implémentation VaR")

    # Charger les tests
    from ai_trading.tests.test_advanced_risk_manager import TestAdvancedRiskManager

    # Créer la suite de tests
    test_suite = unittest.TestSuite()

    # Ajouter les tests
    test_suite.addTest(unittest.makeSuite(TestAdvancedRiskManager))

    # Exécuter les tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_results = test_runner.run(test_suite)

    # Vérifier les résultats
    if test_results.wasSuccessful():
        logger.info("Tous les tests ont réussi!")
        return 0
    else:
        logger.error(
            f"Échec des tests: {len(test_results.failures)} échecs, {len(test_results.errors)} erreurs"
        )
        return 1


def run_example():
    """Exécute l'exemple de démonstration."""
    logger.info("Démarrage de l'exemple de démonstration VaR")

    try:
        from ai_trading.examples.advanced_risk_management_example import (
            main as run_risk_example,
        )

        run_risk_example()
        logger.info("Exemple de gestion des risques avancée terminé avec succès")

        from ai_trading.examples.trading_with_var import run_demonstration

        run_demonstration()
        logger.info("Démonstration de l'intégration VaR terminée avec succès")

        return 0
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des exemples: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Vérifier les arguments
    import argparse

    parser = argparse.ArgumentParser(description="Tests pour l'implémentation VaR")
    parser.add_argument(
        "--mode",
        choices=["tests", "example", "all"],
        default="all",
        help="Mode d'exécution: tests, example ou all",
    )

    args = parser.parse_args()

    # Afficher l'heure de début
    start_time = datetime.now()
    logger.info(f"Démarrage des tests VaR à {start_time.strftime('%H:%M:%S')}")

    exit_code = 0

    if args.mode in ["tests", "all"]:
        exit_code += run_all_tests()

    if args.mode in ["example", "all"]:
        exit_code += run_example()

    # Afficher l'heure de fin et la durée
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Tests terminés à {end_time.strftime('%H:%M:%S')}")
    logger.info(f"Durée totale: {duration.total_seconds() / 60:.2f} minutes")

    # 0 signifie succès, toute autre valeur est un échec
    sys.exit(1 if exit_code > 0 else 0)
