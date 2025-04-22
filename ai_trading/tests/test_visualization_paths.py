"""
Tests unitaires pour vérifier le bon fonctionnement de la sauvegarde des visualisations.
"""

import os
import sys
import unittest

# Configurer matplotlib pour utiliser un backend non-interactif
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # IMPORTANT: Ceci doit être fait avant d'importer pyplot

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.llm.sentiment_analysis import EnhancedNewsAnalyzer, NewsAnalyzer
from ai_trading.config import VISUALIZATION_DIR, SENTIMENT_CACHE_DIR, INFO_RETOUR_DIR


class TestVisualizationPaths(unittest.TestCase):
    """Tests pour vérifier que les visualisations sont sauvegardées au bon endroit."""

    def setUp(self):
        """Configuration avant chaque test."""
        # Créer des données de test
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D")
        sentiment_scores = np.random.normal(0.6, 0.2, 10)

        self.test_data = []
        for i, date in enumerate(dates):
            self.test_data.append(
                {
                    "title": f"Test title {i+1}",
                    "body": f"Test body {i+1}",
                    "published_at": date.isoformat(),
                    "sentiment_score": sentiment_scores[i],
                    "global_sentiment": {
                        "label": (
                            "positive" if sentiment_scores[i] > 0.5 else "negative"
                        ),
                        "score": sentiment_scores[i],
                    },
                }
            )

        self.df = pd.DataFrame(self.test_data)

        # Liste des fichiers à nettoyer après les tests
        self.files_to_cleanup = []

        self.examples_visualization_dir = INFO_RETOUR_DIR / "test" / "visualizations"
        os.makedirs(self.examples_visualization_dir, exist_ok=True)

    def tearDown(self):
        """Nettoyage après chaque test."""
        # Supprimer les fichiers créés pendant les tests
        for file_path in self.files_to_cleanup:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_news_analyzer_visualization_path(self):
        """Teste si NewsAnalyzer sauvegarde les visualisations dans le bon dossier."""
        analyzer = NewsAnalyzer()
        filename = "test_news_analyzer_viz.png"

        # Générer la visualisation
        analyzer.plot_trends(self.df, filename)

        # Chemin où le fichier devrait être sauvegardé
        expected_path = os.path.join(self.sentiment_visualization_dir, filename)
        self.files_to_cleanup.append(expected_path)

        # Vérifier que le fichier existe
        self.assertTrue(
            os.path.exists(expected_path), f"Le fichier {expected_path} n'existe pas"
        )

        # Vérifier que c'est bien une image
        self.assertTrue(os.path.getsize(expected_path) > 0, "Le fichier image est vide")

    def test_enhanced_news_analyzer_visualization_path(self):
        """Teste si EnhancedNewsAnalyzer sauvegarde les visualisations dans le bon dossier."""
        analyzer = EnhancedNewsAnalyzer()
        filename = "test_enhanced_news_analyzer_viz.png"

        # Générer la visualisation
        analyzer.plot_trends(self.df, filename)

        # Chemin où le fichier devrait être sauvegardé
        expected_path = os.path.join(self.sentiment_visualization_dir, filename)
        self.files_to_cleanup.append(expected_path)

        # Vérifier que le fichier existe
        self.assertTrue(
            os.path.exists(expected_path), f"Le fichier {expected_path} n'existe pas"
        )

        # Vérifier que c'est bien une image
        self.assertTrue(os.path.getsize(expected_path) > 0, "Le fichier image est vide")

    def test_custom_visualization_path(self):
        """Teste la sauvegarde d'une visualisation personnalisée dans le dossier examples."""
        import matplotlib.pyplot as plt

        # Nom du fichier
        filename = "test_custom_viz.png"
        output_path = os.path.join(self.examples_visualization_dir, filename)
        self.files_to_cleanup.append(output_path)

        # Créer une visualisation simple
        plt.figure(figsize=(10, 5))
        plt.plot(range(10), np.random.rand(10))
        plt.title("Test Visualization")
        plt.savefig(output_path)
        plt.close()

        # Vérifier que le fichier existe
        self.assertTrue(
            os.path.exists(output_path), f"Le fichier {output_path} n'existe pas"
        )

        # Vérifier que c'est bien une image
        self.assertTrue(os.path.getsize(output_path) > 0, "Le fichier image est vide")

    def test_no_visualization_in_root(self):
        """Vérifie qu'aucune visualisation n'est sauvegardée à la racine du projet."""
        analyzer = NewsAnalyzer()
        filename = "test_root_viz.png"

        # Générer la visualisation
        analyzer.plot_trends(self.df, filename)

        # Chemin à la racine du projet
        root_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            filename,
        )

        # Vérifier que le fichier n'existe pas à la racine
        self.assertFalse(
            os.path.exists(root_path),
            f"Le fichier {root_path} existe à la racine du projet",
        )

        # Mais il devrait exister dans le dossier visualizations/sentiment
        expected_path = os.path.join(self.sentiment_visualization_dir, filename)
        self.files_to_cleanup.append(expected_path)
        self.assertTrue(
            os.path.exists(expected_path), f"Le fichier {expected_path} n'existe pas"
        )


if __name__ == "__main__":
    unittest.main()
