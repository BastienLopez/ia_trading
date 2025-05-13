"""
Script d'exécution de l'API Trading
Ce fichier est utilisé comme point d'entrée principal pour démarrer l'API
dans l'environnement Docker.
"""

import logging
import os
import sys

# Ajouter le répertoire parent au chemin Python pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_trading.api import run

if __name__ == "__main__":
    # Créer le répertoire de logs si nécessaire
    os.makedirs("ai_trading/info_retour/logs", exist_ok=True)

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("ai_trading/info_retour/logs/api.log", mode="a"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("api_runner")
    logger.info("Démarrage de l'API depuis ai_trading.api.run")

    # Exécuter la fonction principale définie dans api.py
    try:
        run()
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de l'API: {str(e)}")
        sys.exit(1)
