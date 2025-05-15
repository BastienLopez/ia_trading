import unittest

import numpy as np
import pandas as pd

from ai_trading.validation.temporal_cross_validator import TemporalCrossValidator


class TestTemporalCrossValidator(unittest.TestCase):
    def setUp(self):
        # Créer un DataFrame de test avec 100 lignes
        self.df = pd.DataFrame(
            {"close": np.random.random(100), "volume": np.random.random(100)}
        )

    def test_split_basic(self):
        """Test basique de la méthode split"""
        validator = TemporalCrossValidator(n_splits=3, train_ratio=0.6, gap_ratio=0.2)
        splits = list(validator.split(self.df))

        # Vérifier le nombre de splits
        self.assertEqual(len(splits), 3)

        # Vérifier la structure des splits
        for train_idx, test_idx in splits:
            # Vérifier que les indices sont des listes
            self.assertIsInstance(train_idx, list)
            self.assertIsInstance(test_idx, list)

            # Vérifier que les indices sont dans l'ordre
            self.assertTrue(
                all(train_idx[i] <= train_idx[i + 1] for i in range(len(train_idx) - 1))
            )
            self.assertTrue(
                all(test_idx[i] <= test_idx[i + 1] for i in range(len(test_idx) - 1))
            )

            # Vérifier qu'il n'y a pas de chevauchement entre train et test
            self.assertEqual(len(set(train_idx).intersection(test_idx)), 0)

            # Vérifier les tailles relatives
            self.assertAlmostEqual(len(train_idx) / len(self.df), 0.6, delta=0.1)

    def test_split_edge_cases(self):
        """Test des cas limites"""
        # Test avec un seul split
        validator = TemporalCrossValidator(n_splits=1, train_ratio=0.6, gap_ratio=0.2)
        splits = list(validator.split(self.df))
        self.assertEqual(len(splits), 1)

        # Test avec des ratios extrêmes
        validator = TemporalCrossValidator(n_splits=2, train_ratio=0.8, gap_ratio=0.1)
        splits = list(validator.split(self.df))
        self.assertTrue(len(splits) > 0)


if __name__ == "__main__":
    unittest.main()
