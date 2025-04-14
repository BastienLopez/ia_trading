import os
import sys
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl_agent import TradingEnvironment, RiskManager


class SimpleTradingSystem:
    """Une classe simple pour remplacer RLTradingSystem qui n'existe pas."""
    
    def __init__(self):
        self.env = None
        self.results = []
        
    def create_environment(self, data, initial_balance, transaction_fee, window_size):
        """Crée un environnement de trading avec les données fournies."""
        # Créer un objet data_source simple pour fournir à TradingEnvironment
        data_source = type('DataSource', (), {
            'historical_data': [{'close': price} for price in data['close'].values]
        })()
        
        # Créer un portfolio mocké qui a une méthode current_exposure qui peut prendre un argument
        mock_portfolio = type('Portfolio', (), {
            'current_exposure': lambda self=None: 0.0
        })()
        
        self.env = TradingEnvironment(
            initial_balance=initial_balance,
            data_source=data_source,
            risk_params={'max_exposure': 0.2, 'max_leverage': 1.5}
        )
        
        # Remplacer le portfolio par défaut par notre version mockée
        self.env.portfolio = mock_portfolio
        
        return self.env
    
    def test_random_strategy(self, num_episodes=3):
        """Teste une stratégie aléatoire dans l'environnement."""
        import random
        
        if not self.env:
            raise ValueError("L'environnement doit être créé avant de tester une stratégie")
        
        self.results = []
        
        for episode in range(num_episodes):
            # Réinitialisation de l'environnement pour un nouvel épisode
            self.env.current_step = 0
            initial_balance = 10000
            current_balance = initial_balance
            
            # Simulation de trading avec des actions aléatoires
            action_types = ['buy', 'sell', 'hold']
            total_reward = 0
            num_steps = min(100, len(self.env.prices))  # Limiter le nombre d'étapes
            
            for step in range(num_steps):
                # Générer une action aléatoire
                action_type = random.choice(action_types)
                amount = random.uniform(0.1, 1.0) if action_type != 'hold' else 0
                
                # Exécuter l'action
                action = (action_type, amount)
                _, reward, done, _ = self.env.step(action)
                
                total_reward += reward
                
                # Simuler un changement de solde basé sur l'action
                if action_type == 'buy':
                    current_balance -= amount * 100  # Simuler un achat
                elif action_type == 'sell':
                    current_balance += amount * 100  # Simuler une vente
                
                if done:
                    break
            
            # Calculer les résultats de l'épisode
            final_value = current_balance
            returns = (final_value - initial_balance) / initial_balance
            avg_reward = total_reward / num_steps
            
            self.results.append({
                'final_value': final_value,
                'returns': returns,
                'avg_reward': avg_reward
            })
            
        return self.results


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
    try:
        # Essayer d'abord avec les paramètres originaux
        integrator.visualize_integrated_data(
            market_data=preprocessed_market_data,
            sentiment_data=sentiment_data,
            save_path="data_integration_example",
        )
    except TypeError as e:
        print(f"Avertissement: {e}")
        # Tenter une approche alternative - vérifier la signature de la méthode
        print("Tentative avec une signature alternative...")
        try:
            # Essayer sans paramètres nommés
            integrator.visualize_integrated_data(
                preprocessed_market_data,
                sentiment_data,
                "data_integration_example"
            )
        except Exception as e:
            print(f"Échec de la visualisation: {e}")
            print("Poursuite de l'exécution sans visualisation...")

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

    # Créer le système de trading RL (utiliser notre implémentation simple)
    rl_system = SimpleTradingSystem()

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
