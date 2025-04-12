import sys
import os
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl_agent import TradingEnvironment
from ai_trading.rl import DQNAgent

logger = logging.getLogger(__name__)

def run_training_example():
    """
    Exemple d'entraînement d'un agent RL sur des données de crypto-monnaie.
    """
    print("Démarrage de l'exemple d'entraînement RL...")

    # Créer l'intégrateur de données
    integrator = RLDataIntegrator()

    # Définir la période
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )  # 1 an de données

    # Collecter et prétraiter les données
    print(
        f"Collecte et prétraitement des données pour BTC du {start_date} au {end_date}..."
    )
    market_data = integrator.collect_market_data(
        symbol="BTC", start_date=start_date, end_date=end_date, interval="1d"
    )
    if market_data is None:
        logger.warning("Génération de données de marché synthétiques")
        market_data = integrator.generate_synthetic_market_data(start_date, end_date)
    preprocessed_market_data = integrator.preprocess_market_data(market_data)

    sentiment_data = integrator.collect_sentiment_data(
        symbol="BTC", start_date=start_date, end_date=end_date
    )
    if sentiment_data is None:
        logger.warning("Génération de données de sentiment synthétiques")
        sentiment_data = integrator.generate_synthetic_sentiment_data(start_date, end_date)

    # Intégrer les données pour l'apprentissage par renforcement
    print("Intégration des données pour l'apprentissage par renforcement...")
    train_data, test_data = integrator.integrate_data(
        market_data=preprocessed_market_data,
        sentiment_data=sentiment_data,
        lookback_window=10,
        test_split=0.2,
    )

    print(f"Données d'entraînement: {len(train_data)} points")
    print(f"Données de test: {len(test_data)} points")

    # Créer l'environnement de trading
    env = TradingEnvironment(
        data_source=train_data,
        initial_balance=10000,
        lookback_window=10
    )

    # Créer l'agent DQN
    agent = DQNAgent(env)

    # Entraîner l'agent
    agent.train(episodes=100)

    # Évaluer l'agent sur les données de test
    test_env = TradingEnvironment(
        data_source=test_data,
        initial_balance=10000,
        lookback_window=10
    )

    # Évaluation
    total_reward = 0
    state = test_env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = test_env.step(action)
        total_reward += reward
        state = next_state

    print(f"Récompense totale sur les données de test: {total_reward:.2f}")

    print("\nExemple d'entraînement terminé!")

    if args.plot:
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_history)
        plt.title('Performance du portefeuille')
        plt.savefig('reports/plots/portfolio_performance.png')

    if args.plot_dir:
        plt.figure(figsize=(18, 9))
        
        # Graphique de performance du portefeuille
        plt.subplot(2, 2, 1)
        plt.plot(portfolio_history)
        plt.title('Performance du portefeuille')
        
        # Graphique des récompenses
        plt.subplot(2, 2, 2)
        plt.plot(rewards_history)
        plt.title('Historique des récompenses')
        
        # Graphique de l'exploration vs exploitation
        plt.subplot(2, 2, 3)
        plt.plot(exploration_rates)
        plt.title('Taux d\'exploration')
        
        plt.tight_layout()
        plt.savefig(f"{args.plot_dir}/rl_training_metrics.png")
        plt.close()


def train_agent():
    env = TradingEnvironment(...)
    agent = DQNAgent(env)
    agent.train(episodes=1000)


if __name__ == "__main__":
    run_training_example()
    train_agent()
