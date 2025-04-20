import os
import sys

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


def main():
    """
    Fonction principale pour entraîner l'agent RL.
    """
    logger.info("Démarrage de l'entraînement de l'agent RL")

    # Créer une instance du système de trading RL
    trading_system = RLTradingSystem()

    # Intégrer les données
    data = trading_system.integrate_data(
        symbol="BTC/USDT",
        start_date="2022-01-01",
        end_date="2023-01-01",
        timeframe="1d",
    )

    # Créer l'environnement
    env = trading_system.create_environment(
        data=data, initial_balance=10000, window_size=20
    )

    # Créer l'agent
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

    # Entraîner l'agent
    trading_system.train(
        agent=agent, env=env, episodes=50, batch_size=32, save_path="models/dqn_agent"
    )

    # Évaluer l'agent
    results = trading_system.evaluate(agent=agent, env=env, episodes=10)

    logger.info(f"Résultats de l'évaluation: {results}")


if __name__ == "__main__":
    main()
