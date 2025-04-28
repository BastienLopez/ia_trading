import os
import sys
import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Utilisation du backend non-interactif 'Agg' pour les tests
import logging

# Configurer le logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.sac_agent import SACAgent
from ai_trading.utils.temporal_cross_validation import TemporalCrossValidator


class TestTemporalCrossValidation(unittest.TestCase):
    """Tests pour la validation croisée temporelle."""

    def setUp(self):
        """Initialisation avant chaque test."""
        # Créer un DataFrame synthétique de prix pour les tests
        self.df = self._generate_test_data()

    def _generate_test_data(self):
        """Génère des données synthétiques pour les tests."""
        np.random.seed(42)  # Pour la reproductibilité
        n_points = 300

        # Générer une tendance
        t = np.linspace(0, 1, n_points)
        trend = 100 + 20 * np.sin(2 * np.pi * t) + t * 10

        # Ajouter du bruit
        noise = np.random.normal(0, 2, n_points)
        price = trend + noise

        # Créer un dataframe
        dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
        df = pd.DataFrame(
            {
                "open": price,
                "high": price * np.random.uniform(1.0, 1.02, n_points),
                "low": price * np.random.uniform(0.98, 1.0, n_points),
                "close": price * np.random.uniform(0.99, 1.01, n_points),
                "volume": np.random.uniform(1000, 5000, n_points),
            },
            index=dates,
        )

        return df

    def test_split(self):
        """Teste la génération des splits."""
        validator = TemporalCrossValidator(
            data=self.df, n_splits=5, initial_train_size=0.7, gap=10
        )

        splits = validator.split()

        # Vérifier le nombre de splits
        self.assertEqual(len(splits), 5)

        # Calculer les tailles attendues
        total_size = len(self.df)
        initial_train_end = int(total_size * 0.7)
        val_size = (total_size - initial_train_end) // 5

        for i, (train_data, test_data) in enumerate(splits):
            # Calculer les indices attendus
            train_end = initial_train_end + i * val_size
            val_start = train_end + 10  # gap
            val_end = min(val_start + val_size, total_size)

            # Vérifier les tailles
            self.assertEqual(len(train_data), train_end)
            self.assertEqual(len(test_data), val_end - val_start)

            # Vérifier que les indices sont consécutifs avec un gap de 10
            self.assertEqual(test_data.index[0], self.df.index[val_start])
            self.assertEqual(train_data.index[-1], self.df.index[train_end - 1])

    def test_evaluate(self):
        """Teste l'évaluation d'un agent sur les splits."""
        validator = TemporalCrossValidator(
            data=self.df, n_splits=3, initial_train_size=0.7, gap=10
        )

        # Créer un agent de test
        agent = SACAgent(
            state_dim=10,
            action_dim=1,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
        )

        # Évaluer l'agent
        results = validator.evaluate(
            model_fn=lambda train_data: agent,
            eval_fn=lambda model, test_data: {
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.1,
                "profit_pct": 0.05,
            },
        )

        # Vérifier que toutes les métriques sont présentes
        self.assertIn("train_metrics", results)
        self.assertIn("val_metrics", results)
        self.assertIn("fold_sizes", results)

        # Vérifier que nous avons des résultats pour chaque split
        self.assertEqual(len(results["train_metrics"]), 3)
        self.assertEqual(len(results["val_metrics"]), 3)
        self.assertEqual(len(results["fold_sizes"]), 3)


if __name__ == "__main__":
    unittest.main()
