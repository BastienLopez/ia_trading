"""
Script d'exécution du dashboard interactif d'analyse post-trade.

Ce script permet de lancer l'application dashboard.
"""

import argparse
import os

from ai_trading.dashboard.app import app


def parse_args():
    """
    Parse les arguments de ligne de commande.

    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(
        description="Dashboard d'analyse post-trade pour le trading crypto basé sur l'IA"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host à utiliser (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port", type=int, default=8050, help="Port à utiliser (default: 8050)"
    )

    parser.add_argument("--debug", action="store_true", help="Activer le mode debug")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Chemin vers les données (par défaut: data/dashboard)",
    )

    return parser.parse_args()


def main():
    """
    Fonction principale pour lancer le dashboard.
    """
    args = parse_args()

    # Afficher les informations de lancement
    print(f"Démarrage du dashboard sur http://{args.host}:{args.port}/")
    print(f"Mode debug: {'activé' if args.debug else 'désactivé'}")

    if args.data_path:
        print(f"Utilisation des données depuis: {args.data_path}")
        os.environ["DASHBOARD_DATA_PATH"] = args.data_path

    # Lancer le serveur Dash
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
