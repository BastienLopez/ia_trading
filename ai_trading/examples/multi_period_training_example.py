#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os

import tensorflow as tf

from ai_trading.rl.multi_period_trainer import MultiPeriodTrainer

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Exemple d'entraînement multi-période pour le trading RL"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Symbole de la crypto-monnaie (ex: BTC, ETH)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Nombre de jours de données historiques à collecter",
    )
    parser.add_argument(
        "--periods",
        type=str,
        default="1440,240,60",
        help="Périodes en minutes, séparées par des virgules (ex: 1440,240,60 pour 1j,4h,1h)",
    )
    parser.add_argument(
        "--use_gru",
        action="store_true",
        help="Utiliser une architecture GRU pour l'agent",
    )
    parser.add_argument(
        "--no_curriculum",
        dest="use_curriculum",
        action="store_false",
        help="Ne pas utiliser l'approche curriculum learning",
    )
    parser.add_argument(
        "--no_sentiment",
        dest="include_sentiment",
        action="store_false",
        help="Ne pas inclure les données de sentiment",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Nombre d'époques par période"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Nombre d'épisodes par époque"
    )
    parser.add_argument(
        "--initial_balance",
        type=float,
        default=10000.0,
        help="Solde initial pour l'environnement de trading",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="trained_models",
        help="Répertoire pour sauvegarder les modèles",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=10,
        help="Longueur de séquence pour les modèles GRU",
    )
    parser.add_argument(
        "--reward_scaling",
        type=float,
        default=1.0,
        help="Facteur d'échelle pour les récompenses",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
        help="Type d'espace d'action (continuous ou discrete)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Évaluer l'agent final après l'entraînement"
    )
    parser.add_argument(
        "--test_days",
        type=int,
        default=10,
        help="Nombre de jours pour les données de test",
    )

    parser.set_defaults(use_curriculum=True, include_sentiment=True)

    return parser.parse_args()


def main():
    """Fonction principale."""
    args = parse_args()

    # Convertir la chaîne de périodes en liste d'entiers
    periods = [int(p) for p in args.periods.split(",")]

    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(args.save_dir, exist_ok=True)

    # Configurer TensorFlow pour limiter l'utilisation de la mémoire GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Utilisation de {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.error(f"Erreur lors de la configuration GPU: {e}")
    else:
        logger.info("Aucun GPU détecté, utilisation du CPU")

    # Créer le MultiPeriodTrainer
    trainer = MultiPeriodTrainer(
        symbol=args.symbol,
        days=args.days,
        periods=periods,
        agent_type="sac",
        use_gru=args.use_gru,
        initial_balance=args.initial_balance,
        save_dir=args.save_dir,
        use_curriculum=args.use_curriculum,
        epochs_per_period=args.epochs,
        episodes_per_epoch=args.episodes,
        include_sentiment=args.include_sentiment,
        sequence_length=args.sequence_length,
        reward_scaling=args.reward_scaling,
        action_type=args.action_type,
    )

    # Exécuter l'entraînement
    results = trainer.run_curriculum()

    # Tracer les graphiques des résultats
    trainer.plot_results(results)

    # Évaluer l'agent final
    if args.test:
        test_results = trainer.evaluate_final_agent(
            test_days=args.test_days, visualize=True
        )

        logger.info("\nRésultats de l'évaluation finale:")
        logger.info(f"Récompense totale: {test_results['reward']:.4f}")
        logger.info(
            f"Valeur finale du portefeuille: ${test_results['portfolio_value']:.2f}"
        )
        logger.info(f"Rendement: {test_results['return_pct']:.2f}%")


if __name__ == "__main__":
    main()
