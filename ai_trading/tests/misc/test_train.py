import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import matplotlib

# Ajouter le répertoire parent au path pour importer les modules du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.train import TrainingMonitor, train_agent

matplotlib.use("Agg")  # Utiliser le backend non-interactif


class TestTrain(unittest.TestCase):
    """Tests pour la boucle d'entraînement."""

    def setUp(self):
        """Prépare l'environnement, l'agent et les données pour les tests."""
        # Créer un intégrateur de données
        integrator = RLDataIntegrator()

        # Générer des données synthétiques
        self.test_data = integrator.generate_synthetic_data(
            n_samples=100, trend="bullish", volatility=0.02, with_sentiment=True
        )

        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()

        # Créer l'environnement
        self.env = TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )

        # Créer l'agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.agent = DQNAgent(
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

    def _create_test_environment(self):
        """Crée un environnement de test."""
        return TradingEnvironment(
            df=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
        )

    def _create_test_agent(self, env):
        """Crée un agent de test."""
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        return DQNAgent(
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

    def test_training_monitor(self):
        """Teste la classe TrainingMonitor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingMonitor(
                initial_balance=10000,
                save_dir=temp_dir,
                plot_interval=1,  # Forcer la mise à jour à chaque épisode
            )

            # Simuler des mises à jour avec des épisodes différents
            for i in range(3):
                monitor.update(
                    episode=i,
                    total_reward=50.0 + i * 10,
                    portfolio_value=10500.0 + i * 500,
                    epsilon=0.5 - i * 0.1,
                    avg_loss=0.1 + i * 0.05,
                    elapsed_time=10.0 + i * 5,
                )

            # Ajouter une pause pour la génération des graphiques
            time.sleep(1)

            monitor.save_plots()

            # Vérifier que les fichiers ont été générés (avec timestamp)
            files_in_dir = os.listdir(temp_dir)

            # Vérifier que chaque type de graphique a été généré
            self.assertTrue(
                any(
                    f.startswith("rewards_") and f.endswith(".png")
                    for f in files_in_dir
                ),
                f"Fichier rewards_*.png non trouvé. Fichiers présents: {files_in_dir}",
            )

            self.assertTrue(
                any(
                    f.startswith("portfolio_") and f.endswith(".png")
                    for f in files_in_dir
                ),
                f"Fichier portfolio_*.png non trouvé. Fichiers présents: {files_in_dir}",
            )

            self.assertTrue(
                any(
                    f.startswith("returns_") and f.endswith(".png")
                    for f in files_in_dir
                ),
                f"Fichier returns_*.png non trouvé. Fichiers présents: {files_in_dir}",
            )

    def test_train_agent(self):
        """Teste la fonction d'entraînement de l'agent."""
        # Créer un répertoire temporaire pour les modèles
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer l'environnement
            env = self._create_test_environment()

            # Créer l'agent avec la taille d'état correcte
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = DQNAgent(
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

            # Entraîner l'agent
            history = train_agent(
                agent=agent,
                env=env,
                episodes=5,
                batch_size=32,
                save_path=os.path.join(temp_dir, "test_model"),
                visualize=True,
                checkpoint_interval=2,
                early_stopping=None,
                max_steps_per_episode=None,
                use_tensorboard=False,
            )

            # Vérifier que l'historique est retourné
            self.assertIsInstance(history, dict)

            # Vérifier que l'historique contient les bonnes clés
            expected_keys = [
                "episode_rewards",
                "episode_portfolio_values",
                "episode_returns",
            ]
            for key in expected_keys:
                self.assertIn(key, history)
                self.assertEqual(len(history[key]), 5)

            # Vérifier que le modèle final est sauvegardé
            final_model_path = os.path.join(temp_dir, "test_model_final.h5")
            if os.path.exists(final_model_path):
                self.assertTrue(True)
            else:
                # Vérifier le chemin alternatif
                alt_path = os.path.join(temp_dir, "test_model_final.h5.weights.h5")
                if os.path.exists(alt_path):
                    self.assertTrue(True)
                else:
                    self.assertTrue(
                        False, f"Aucun modèle trouvé à {final_model_path} ou {alt_path}"
                    )

    def test_early_stopping(self):
        """Teste la fonctionnalité d'arrêt anticipé."""
        # Créer un environnement et un agent
        env = self._create_test_environment()
        agent = self._create_test_agent(env)

        # Entraîner l'agent avec arrêt anticipé
        history = train_agent(
            agent=agent,
            env=env,
            episodes=20,
            batch_size=32,
            save_path=os.path.join(self.temp_dir, "test_model"),
            visualize=False,  # Désactiver les visualisations pour éviter les problèmes de tkinter
            early_stopping={"patience": 2, "min_delta": 0.0, "metric": "reward"},
        )

        # Vérifier que l'entraînement s'est arrêté avant le nombre maximum d'épisodes
        self.assertLess(
            len(history["episode_rewards"]),
            20,
            "L'entraînement aurait dû s'arrêter avant d'atteindre le nombre maximum d'épisodes",
        )


if __name__ == "__main__":
    unittest.main()
