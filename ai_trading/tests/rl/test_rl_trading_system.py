import os
import sys
import tempfile
import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Désactiver l'interface graphique pour les tests

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.trading_system import RLTradingSystem


class TestRLTradingSystem(unittest.TestCase):
    """Tests pour le système complet de trading RL."""

    def setUp(self):
        """Prépare le système et les données pour les tests."""
        # Créer un intégrateur de données
        self.integrator = RLDataIntegrator()

        # Générer des données synthétiques
        self.test_data = self.integrator.generate_synthetic_data(
            n_samples=100, trend="bullish", volatility=0.02, with_sentiment=True
        )

        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()

        # Créer le système de trading
        self.system = RLTradingSystem()

    def test_create_environment(self):
        """Teste la création de l'environnement."""
        # Créer l'environnement
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )

        # Vérifier que l'environnement est créé
        self.assertIsNotNone(env)
        self.assertEqual(env.initial_balance, 10000)
        self.assertEqual(env.transaction_fee, 0.001)
        self.assertEqual(env.window_size, 10)

    def test_create_agent(self):
        """Teste la création de l'agent."""
        # Créer l'environnement d'abord
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )

        # Créer l'agent
        agent = self.system.create_agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000,
        )

        # Vérifier que l'agent est créé
        self.assertIsNotNone(agent)
        self.assertEqual(agent.state_size, env.observation_space.shape[0])
        self.assertEqual(agent.action_size, env.action_space.n)

    def test_train(self):
        """Teste l'entraînement du système."""
        # Créer l'environnement
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )

        # Obtenir la taille de l'état à partir de l'environnement
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # Créer l'agent avec la bonne taille d'état
        agent = self.system.create_agent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000,
        )

        # Entraîner le système
        self.system.train(
            agent=agent,
            episodes=2,
            batch_size=32,
            save_path=None,
        )

        # Vérifier que l'agent a été entraîné
        self.assertIsNotNone(agent)

    def test_evaluate(self):
        """Teste l'évaluation du système."""
        # Créer l'environnement
        env = self.system.create_environment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )

        # Réinitialiser l'environnement pour obtenir un état réel
        initial_state, _ = env.reset()

        # Obtenir la taille de l'état à partir de l'état réel
        state_size = initial_state.shape[0]
        action_size = env.action_space.n

        # Créer l'agent avec la taille d'état correcte
        agent = self.system.create_agent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=0.0,  # Pas d'exploration pour l'évaluation
            epsilon_decay=0.995,
            epsilon_min=0.01,
            batch_size=32,
            memory_size=1000,
        )

        # Créer un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Évaluer le système
            results = self.system.evaluate(
                agent=agent,
                env=env,
                num_episodes=1,
            )

            # Vérifier que les résultats sont retournés
            self.assertIsInstance(results, dict)

            # Vérifier que les résultats contiennent les bonnes clés
            expected_keys = [
                "final_value",
                "returns",
                "sharpe_ratio",
                "max_drawdown",
                "portfolio_history",
                "actions",
                "rewards",
            ]

            for key in expected_keys:
                self.assertIn(key, results)

    def test_integrate_data(self):
        """Teste l'intégration des données."""
        # Créer des données de marché et de sentiment
        market_data = pd.DataFrame(
            {
                "open": np.random.random(100) * 100 + 100,
                "high": np.random.random(100) * 100 + 110,
                "low": np.random.random(100) * 100 + 90,
                "close": np.random.random(100) * 100 + 105,
                "volume": np.random.random(100) * 1000 + 1000,
            }
        )

        sentiment_data = pd.DataFrame({"compound_score": np.random.uniform(-1, 1, 100)})

        # Intégrer les données
        train_data, test_data = self.system.integrate_data(
            market_data=market_data, sentiment_data=sentiment_data, split_ratio=0.8
        )

        # Vérifier que les données sont divisées correctement
        expected_train_size = int(len(market_data) * 0.8)
        expected_test_size = len(market_data) - expected_train_size

        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), expected_test_size)

    def test_test_random_strategy(self):
        """Teste la stratégie aléatoire."""
        # Créer l'environnement s'il n'existe pas déjà
        if not hasattr(self.system, "env") or self.system.env is None:
            self.system.env = self.system.create_environment(
                data=self.test_data,
                initial_balance=10000,
                transaction_fee=0.001,
                window_size=10,
            )

        # Tester la stratégie aléatoire
        results = self.system.test_random_strategy(num_episodes=2)

        # Vérifier que les résultats sont retournés
        self.assertIsInstance(results, dict)

        # Vérifier que les résultats contiennent les bonnes clés
        expected_keys = [
            "average_reward",
            "max_reward",
            "min_reward",
            "total_return",
            "final_portfolio_value",
        ]

        for key in expected_keys:
            self.assertIn(key, results)


if __name__ == "__main__":
    unittest.main()
