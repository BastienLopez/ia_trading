import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Utilisation du backend non-interactif 'Agg' pour les tests
import matplotlib.pyplot as plt
import logging

# Configurer le logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.utils.temporal_cross_validation import TemporalCrossValidator, WalkForwardOptimizer

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
        dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
        df = pd.DataFrame({
            'open': price,
            'high': price * np.random.uniform(1.0, 1.02, n_points),
            'low': price * np.random.uniform(0.98, 1.0, n_points),
            'close': price * np.random.uniform(0.99, 1.01, n_points),
            'volume': np.random.uniform(1000, 5000, n_points)
        }, index=dates)
        
        return df
    
    def test_temporal_cross_validator_initialization(self):
        """Teste l'initialisation du validateur croisé temporel."""
        cv = TemporalCrossValidator(
            data=self.df,
            n_splits=5,
            initial_train_size=0.6,
            gap=1
        )
        
        # Vérifier que les attributs sont correctement initialisés
        self.assertEqual(cv.n_splits, 5)
        self.assertEqual(cv.initial_train_size, 0.6)
        self.assertEqual(cv.gap, 1)
        self.assertEqual(cv.initial_train_end, int(len(self.df) * 0.6))
        
        # Vérifier que stride est correctement calculé
        expected_val_size = (len(self.df) - cv.initial_train_end) // 5
        self.assertEqual(cv.stride, expected_val_size)
    
    def test_temporal_cross_validator_split(self):
        """Teste la génération des divisions temporelles."""
        cv = TemporalCrossValidator(
            data=self.df,
            n_splits=3,
            initial_train_size=0.5,
            gap=2
        )
        
        splits = cv.split()
        
        # Vérifier le nombre de divisions
        self.assertEqual(len(splits), 3)
        
        # Vérifier que chaque division est un tuple (train, val)
        for train, val in splits:
            self.assertIsInstance(train, pd.DataFrame)
            self.assertIsInstance(val, pd.DataFrame)
            
            # Vérifier que les périodes d'entraînement et de validation ne se chevauchent pas
            train_end = train.index[-1]
            val_start = val.index[0]
            self.assertTrue(train_end < val_start)
            
            # Vérifier le gap
            days_diff = (val_start - train_end).days
            self.assertGreaterEqual(days_diff, 1)  # Au moins 1 jour de gap
    
    def test_temporal_cross_validator_evaluate(self):
        """Teste l'évaluation avec la validation croisée temporelle."""
        cv = TemporalCrossValidator(
            data=self.df,
            n_splits=3,
            initial_train_size=0.5,
            gap=1
        )
        
        # Définir une fonction de modèle simple
        def simple_model_fn(train_data):
            # Un modèle très simple qui prédit la dernière valeur observée
            return {"last_close": train_data["close"].iloc[-1]}
        
        # Définir une fonction d'évaluation
        def simple_eval_fn(model, data):
            # Calcule une erreur simple (moyenne des différences carrées)
            last_close = model["last_close"]
            actual_closes = data["close"].values
            return np.mean((actual_closes - last_close)**2)
        
        # Évaluer le modèle
        results = cv.evaluate(simple_model_fn, simple_eval_fn)
        
        # Vérifier les résultats
        self.assertIn("train_metrics", results)
        self.assertIn("val_metrics", results)
        self.assertIn("fold_sizes", results)
        
        # Vérifier que nous avons 3 métriques (une pour chaque division)
        self.assertEqual(len(results["train_metrics"]), 3)
        self.assertEqual(len(results["val_metrics"]), 3)
        self.assertEqual(len(results["fold_sizes"]), 3)
        
        # Vérifier que les métriques sont des nombres
        for train_metric in results["train_metrics"]:
            self.assertIsInstance(train_metric, (int, float))
            
        for val_metric in results["val_metrics"]:
            self.assertIsInstance(val_metric, (int, float))
    
    def test_walk_forward_optimizer_initialization(self):
        """Teste l'initialisation de l'optimiseur walk-forward."""
        param_grid = {
            "window_size": [5, 10, 20],
            "alpha": [0.01, 0.1]
        }
        
        optimizer = WalkForwardOptimizer(
            data=self.df,
            param_grid=param_grid,
            n_splits=3,
            initial_train_size=0.6,
            gap=1,
            evaluation_metric="profit"
        )
        
        # Vérifier que les attributs sont correctement initialisés
        self.assertEqual(optimizer.n_splits, 3)
        self.assertEqual(optimizer.initial_train_size, 0.6)
        self.assertEqual(optimizer.gap, 1)
        self.assertEqual(optimizer.evaluation_metric, "profit")
        
        # Vérifier que toutes les combinaisons d'hyperparamètres sont générées
        self.assertEqual(len(optimizer.param_combinations), 6)  # 3 * 2 = 6 combinaisons
        
        # Vérifier que les combinaisons sont correctes
        expected_combinations = [
            {"window_size": 5, "alpha": 0.01},
            {"window_size": 5, "alpha": 0.1},
            {"window_size": 10, "alpha": 0.01},
            {"window_size": 10, "alpha": 0.1},
            {"window_size": 20, "alpha": 0.01},
            {"window_size": 20, "alpha": 0.1}
        ]
        
        # Vérifier que toutes les combinaisons attendues sont présentes
        for expected_combo in expected_combinations:
            self.assertIn(expected_combo, optimizer.param_combinations)
    
    def test_walk_forward_optimizer_optimize(self):
        """Teste l'optimisation des hyperparamètres avec validation croisée temporelle."""
        param_grid = {
            "window_size": [5, 10],
            "alpha": [0.1, 0.5]
        }
        
        optimizer = WalkForwardOptimizer(
            data=self.df,
            param_grid=param_grid,
            n_splits=2,
            initial_train_size=0.6,
            gap=1,
            evaluation_metric="profit"
        )
        
        # Définir une fonction simple pour créer un modèle avec des hyperparamètres
        def model_fn(train_data, params):
            # Un modèle très simple qui calcule une moyenne mobile
            window_size = params["window_size"]
            return {
                "window_size": window_size,
                "alpha": params["alpha"],
                "last_ma": train_data["close"].rolling(window=window_size).mean().iloc[-1]
            }
        
        # Définir une fonction d'évaluation qui retourne un profit fictif
        def eval_fn(model, data):
            # Utilise le modèle pour générer un profit fictif
            alpha = model["alpha"]
            last_ma = model["last_ma"]
            
            # Un profit fictif basé sur l'écart par rapport à la moyenne mobile
            deviations = data["close"] - last_ma
            profit = (deviations * alpha).sum()
            
            return profit
        
        # Optimiser les hyperparamètres
        optimization_results = optimizer.optimize(model_fn, eval_fn)
        
        # Vérifier les résultats
        self.assertIn("best_params", optimization_results)
        self.assertIn("best_metric", optimization_results)
        self.assertIn("all_results", optimization_results)
        
        # Vérifier que nous avons 4 résultats (un pour chaque combinaison d'hyperparamètres)
        self.assertEqual(len(optimization_results["all_results"]), 4)
        
        # Vérifier que les meilleurs paramètres font partie des combinaisons testées
        best_params = optimization_results["best_params"]
        self.assertIn(best_params, optimizer.param_combinations)
        
        # Vérifier que la meilleure métrique est un nombre
        self.assertIsInstance(optimization_results["best_metric"], (int, float))
        
        # Vérifier que les résultats sont triés par ordre décroissant de métrique
        all_results = optimization_results["all_results"]
        for i in range(len(all_results) - 1):
            self.assertGreaterEqual(all_results[i]["avg_val_metric"], all_results[i+1]["avg_val_metric"])

    def test_visualization(self):
        """Teste la visualisation des divisions temporelles."""
        # Ce test n'effectue pas vraiment d'assertions mais génère une visualisation
        # qui peut être inspectée manuellement
        cv = TemporalCrossValidator(
            data=self.df,
            n_splits=3,
            initial_train_size=0.5,
            gap=5
        )
        
        splits = cv.split()
        
        # Créer une figure pour visualiser les divisions
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df["close"], label="Données complètes", color="gray", alpha=0.5)
        
        colors = ["blue", "green", "red", "purple", "orange"]
        for i, (train, val) in enumerate(splits):
            plt.plot(train.index, train["close"], label=f"Train {i+1}", color=colors[i % len(colors)])
            plt.plot(val.index, val["close"], label=f"Val {i+1}", color=colors[i % len(colors)], linestyle="--")
        
        plt.title("Visualisation des divisions temporelles")
        plt.xlabel("Date")
        plt.ylabel("Prix de clôture")
        plt.legend()
        
        # Sauvegarder la figure pour inspection manuelle
        # plt.savefig("temporal_cross_validation.png")
        plt.close()
        
        # Pas d'assertions ici, ce test est principalement pour la visualisation


if __name__ == '__main__':
    unittest.main() 