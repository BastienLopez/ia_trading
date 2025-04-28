import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

# Importer les modules corrects
from ai_trading.rl.trading_system import RLTradingSystem

# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai_trading")

INFO_RETOUR_DIR = Path(__file__).parent / "info_retour"
INFO_RETOUR_DIR.mkdir(exist_ok=True)


def main():
    """
    Fonction principale pour entraîner l'agent RL.
    """
    logger.info("Démarrage de l'entraînement de l'agent RL")

    try:
        # Créer une instance du système de trading RL avec une configuration par défaut
        trading_system = RLTradingSystem(
            config={
                "env_name": "trading",
                "agent_type": "dqn",
                "window_size": 20,
                "initial_balance": 10000,
            }
        )

        # Intégrer les données
        data = trading_system.integrate_data(
            symbol="BTC/USDT",
            start_date="2022-01-01",
            end_date="2023-01-01",
            timeframe="1d",
        )

        # Créer l'environnement si non existant
        if not hasattr(trading_system, "_env"):
            env = trading_system.create_environment(
                data=data, initial_balance=10000, window_size=20
            )
        else:
            env = trading_system._env

        # Créer l'agent si non existant
        if not hasattr(trading_system, "_agent"):
            agent = trading_system.create_agent(
                state_size=env.observation_space.shape[0],
                action_size=(
                    env.action_space.n
                    if hasattr(env.action_space, "n")
                    else env.action_space.shape[0]
                ),
                batch_size=32,
                learning_rate=0.001,
            )
        else:
            agent = trading_system._agent

        # Entraîner l'agent
        trading_system.train(
            episodes=50,
            batch_size=32,
            save_path=str(INFO_RETOUR_DIR / "models" / "dqn_agent"),
        )

        # Évaluer l'agent
        results = trading_system.evaluate(episodes=10)

        logger.info(f"Résultats de l'évaluation: {results}")

    except Exception as e:
        logger.error(f"Une erreur s'est produite: {str(e)}")
        raise


if __name__ == "__main__":
    main()
