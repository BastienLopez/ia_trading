import os
import sys
import unittest

import numpy as np
import pandas as pd
import matplotlib

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.data_integration import RLDataIntegrator

matplotlib.use('Agg')  # Utiliser le backend non-interactif


class TestDataIntegration(unittest.TestCase):
    """Tests pour l'intégration des données."""

    def setUp(self):
        """Prépare les données pour les tests."""
        # Créer des données de marché synthétiques
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Créer une tendance haussière simple
        prices = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)

        # Créer un DataFrame avec les données de marché
        self.market_data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 10, 100),
                "low": prices - np.random.uniform(0, 10, 100),
                "close": prices + np.random.normal(0, 3, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        # Créer un DataFrame avec les données de sentiment
        self.sentiment_data = pd.DataFrame(
            {
                "compound_score": np.random.uniform(-1, 1, 100),
                "positive_score": np.random.uniform(0, 1, 100),
                "negative_score": np.random.uniform(0, 1, 100),
                "neutral_score": np.random.uniform(0, 1, 100),
            },
            index=dates,
        )

        # Créer l'intégrateur de données
        self.integrator = RLDataIntegrator()

    def test_preprocess_market_data(self):
        """Teste le prétraitement des données de marché."""
        # Générer des données synthétiques
        data = self.integrator.generate_synthetic_data(n_samples=100)
        
        # Prétraiter les données
        processed_data = self.integrator.preprocess_market_data(data)
        
        # Vérifier que les données sont prétraitées
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(processed_data), 100)
        
        # Vérifier que les colonnes nécessaires sont présentes
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, processed_data.columns)
        
        # Vérifier qu'il n'y a pas de valeurs manquantes
        self.assertTrue(processed_data[required_columns].notna().all().all())

    def test_integrate_data(self):
        """Teste l'intégration des données de marché et de sentiment."""
        # Prétraiter les données de marché
        preprocessed_market_data = self.integrator.preprocess_market_data(
            self.market_data
        )

        # Intégrer les données
        train_data, test_data = self.integrator.integrate_data(
            market_data=preprocessed_market_data,
            sentiment_data=self.sentiment_data,
            window_size=10,
            test_split=0.2,
        )

        # Vérifier que les données sont divisées correctement
        expected_train_size = int(len(preprocessed_market_data) * 0.8)
        expected_test_size = len(preprocessed_market_data) - expected_train_size

        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), expected_test_size)

        # Vérifier que les données de sentiment sont intégrées
        self.assertIn("sentiment_compound_score", train_data.columns)

        # Vérifier que les données sont prêtes pour l'RL
        self.assertTrue(train_data.isna().sum().sum() == 0)
        self.assertTrue(test_data.isna().sum().sum() == 0)

    def test_generate_synthetic_data(self):
        """Teste la génération de données synthétiques."""
        # Générer des données synthétiques
        synthetic_data = self.integrator.generate_synthetic_data(
            n_samples=100, trend="bullish", volatility=0.02, with_sentiment=True
        )

        # Vérifier que les données ont la bonne taille
        self.assertEqual(len(synthetic_data), 100)

        # Vérifier que les colonnes nécessaires sont présentes
        expected_columns = ["open", "high", "low", "close", "volume", "compound_score"]
        for col in expected_columns:
            self.assertIn(col, synthetic_data.columns)

        # Vérifier que les prix sont cohérents (high >= open, close, low)
        self.assertTrue(all(synthetic_data["high"] >= synthetic_data["open"]))
        self.assertTrue(all(synthetic_data["high"] >= synthetic_data["close"]))
        self.assertTrue(all(synthetic_data["high"] >= synthetic_data["low"]))

        # Vérifier que les prix bas sont cohérents (low <= open, close)
        self.assertTrue(all(synthetic_data["low"] <= synthetic_data["open"]))
        self.assertTrue(all(synthetic_data["low"] <= synthetic_data["close"]))

    def test_visualize_integrated_data(self):
        """Teste la visualisation des données intégrées."""
        # Prétraiter les données de marché
        preprocessed_market_data = self.integrator.preprocess_market_data(
            self.market_data
        )

        # Créer un répertoire temporaire pour les visualisations
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualiser les données
            files = self.integrator.visualize_integrated_data(
                data=preprocessed_market_data,
                save_dir=temp_dir
            )

            # Vérifier que les fichiers de visualisation sont créés
            for file in files:
                self.assertTrue(os.path.exists(os.path.join(temp_dir, file)))


if __name__ == "__main__":
    unittest.main()
