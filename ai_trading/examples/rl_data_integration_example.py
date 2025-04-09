import os
import sys
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl_agent import RLTradingSystem


def run_data_integration_example():
    """
    Exemple d'intégration des données pour l'apprentissage par renforcement.
    """
    print("Démarrage de l'exemple d'intégration des données pour RL...")

    # Créer l'intégrateur de données
    integrator = RLDataIntegrator()

    # Définir la période
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )  # 1 an de données

    # Collecter les données de marché
    print(f"Collecte des données de marché pour BTC du {start_date} au {end_date}...")
    market_data = integrator.collect_market_data(
        symbol="BTC", start_date=start_date, end_date=end_date, interval="1d"
    )

    # Prétraiter les données de marché
    print("Prétraitement des données de marché...")
    preprocessed_market_data = integrator.preprocess_market_data(market_data)

    # Collecter les données de sentiment
    print(
        f"Collecte des données de sentiment pour BTC du {start_date} au {end_date}..."
    )
    sentiment_data = integrator.collect_sentiment_data(
        symbol="BTC", start_date=start_date, end_date=end_date
    )

    # Visualiser les données intégrées
    print("Visualisation des données intégrées...")
    integrator.visualize_integrated_data(
        market_data=preprocessed_market_data,
        sentiment_data=sentiment_data,
        save_path="data_integration_example",
    )

    # Intégrer les données pour l'apprentissage par renforcement
    print("Intégration des données pour l'apprentissage par renforcement...")
    train_data, test_data = integrator.integrate_data(
        market_data=preprocessed_market_data,
        sentiment_data=sentiment_data,
        window_size=10,
        test_split=0.2,
    )

    print(f"Données d'entraînement: {len(train_data)} points")
    print(f"Données de test: {len(test_data)} points")

    # Afficher les premières lignes des données d'entraînement
    print("\nAperçu des données d'entraînement:")
    print(train_data.head())

    # Créer le système de trading RL
    rl_system = RLTradingSystem()

    # Créer l'environnement de trading avec les données d'entraînement
    print("\nCréation de l'environnement de trading...")
    env = rl_system.create_environment(
        data=train_data, initial_balance=10000, transaction_fee=0.001, window_size=10
    )

    # Tester une stratégie aléatoire
    print("\nTest d'une stratégie aléatoire...")
    results = rl_system.test_random_strategy(num_episodes=3)

    # Afficher les résultats
    for i, result in enumerate(results):
        print(f"\nÉpisode {i+1}:")
        print(f"  Valeur finale du portefeuille: ${result['final_value']:.2f}")
        print(f"  Rendement: {result['returns']*100:.2f}%")
        print(f"  Récompense moyenne: {result['avg_reward']:.4f}")

    print("\nExemple d'intégration des données terminé!")


if __name__ == "__main__":
    run_data_integration_example()
