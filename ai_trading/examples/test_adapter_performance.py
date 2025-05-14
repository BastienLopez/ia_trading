#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour évaluer les performances du RealTimeAdapter optimisé.

Ce script génère des données de marché synthétiques et mesure les performances
du traitement par lots et des optimisations implémentées.
"""

import os
import sys
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

# Ajouter le répertoire racine au chemin d'importation Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Importer les modules nécessaires
from ai_trading.llm.predictions.real_time_adapter import RealTimeAdapter


# Classe simple de prédiction pour les tests
class SimplePredictorForTest:
    """Prédicteur simple pour les tests de performance"""

    def __init__(self, name="TestPredictor"):
        self.name = name
        self.predict_count = 0

    def predict(self, market_state):
        """Simule une prédiction avec un délai configurable"""
        # Simuler un certain délai de calcul
        time.sleep(0.01)
        self.predict_count += 1

        # Générer une prédiction simple
        price = market_state.get("price", 0)
        return {
            "prediction": price * (1 + 0.01 * np.random.randn()),
            "confidence": 0.7 + 0.2 * np.random.random(),
            "direction": "up" if np.random.random() > 0.5 else "down",
            "timestamp": datetime.now(),
        }


def generate_market_data(num_points=1000, with_outliers=True, with_trends=True):
    """
    Génère des données de marché synthétiques pour tester les performances.

    Args:
        num_points: Nombre de points à générer
        with_outliers: Si True, ajoute des valeurs aberrantes
        with_trends: Si True, génère des tendances dans les données

    Returns:
        List: Liste de tuples (prix, timestamp)
    """
    # Prix de base et tendance initiale
    base_price = 50000.0
    trend = 0.0001
    volatility = 0.001

    # Générer des timestamps espacés régulièrement
    now = datetime.now()
    timestamps = [now - timedelta(seconds=i * 5) for i in range(num_points)]
    timestamps.reverse()  # Mettre en ordre chronologique

    # Générer les prix avec marche aléatoire et tendance
    prices = [base_price]
    for i in range(1, num_points):
        # Changer la tendance périodiquement
        if with_trends and i % 200 == 0:
            trend = np.random.normal(0, 0.0005)

        # Calculer le nouveau prix
        price_change = prices[-1] * (trend + np.random.normal(0, volatility))
        new_price = prices[-1] + price_change
        prices.append(new_price)

    # Ajouter des outliers
    if with_outliers:
        # Ajouter environ 5% de valeurs aberrantes
        outlier_indices = np.random.choice(
            num_points, size=int(num_points * 0.05), replace=False
        )
        for idx in outlier_indices:
            # Multiplier par un facteur aléatoire entre 0.5 et 1.5
            outlier_factor = 0.5 + np.random.random() * 1.0
            prices[idx] = prices[idx] * outlier_factor

    # Créer la liste finale
    market_data = [(prices[i], timestamps[i]) for i in range(num_points)]
    return market_data


def prediction_callback(prediction):
    """Callback pour les mises à jour de prédiction"""
    pass  # Ne rien faire pour éviter de ralentir les tests


def run_performance_test(data_size=1000, batch_sizes=[10, 50, 100, 250]):
    """
    Exécute un test de performance avec différentes tailles de lots.

    Args:
        data_size: Nombre total de points de données à traiter
        batch_sizes: Liste des tailles de lots à tester

    Returns:
        Dict: Résultats des performances
    """
    print(f"Test de performance avec {data_size} points de données")

    # Générer les données de test une seule fois
    test_data = generate_market_data(data_size)

    results = {}

    for batch_size in batch_sizes:
        print(f"\nTest avec taille de lot = {batch_size}")

        # Créer un nouveau prédicteur et adaptateur pour chaque test
        predictor = SimplePredictorForTest(f"Batch{batch_size}")

        # Initialiser l'adaptateur en mode backtest pour éviter les threads actifs
        adapter = RealTimeAdapter(
            prediction_model=predictor,
            update_frequency=5.0,
            backtest_mode=True,
            cache_size=128,
        )

        # Configurer les callbacks
        adapter.set_callback("update", prediction_callback)

        # Mesurer le temps de démarrage
        start_time = time.time()

        # Ajouter les données par lots
        for i in range(0, data_size, batch_size):
            batch_end = min(i + batch_size, data_size)
            batch = test_data[i:batch_end]

            # Convertir les données au format attendu par l'adaptateur
            for price, timestamp in batch:
                adapter.add_data({"price": price}, timestamp)

            # Traiter les données explicitement (car nous sommes en mode backtest)
            adapter._process_new_data()

        # Calculer le temps total
        total_time = time.time() - start_time

        # Récupérer les statistiques de performance
        perf_stats = adapter.get_performance_stats()

        # Stocker les résultats
        results[batch_size] = {
            "total_time": total_time,
            "avg_processing_time": perf_stats.get("avg_processing_time", 0),
            "total_data_processed": perf_stats.get("total_data_processed", 0),
            "total_outliers_removed": perf_stats.get("total_outliers_removed", 0),
            "avg_compression_ratio": perf_stats.get("avg_compression_ratio", 0),
            "predictions_generated": predictor.predict_count,
        }

        print(f"  Temps total: {total_time:.4f}s")
        print(
            f"  Temps moyen de traitement par lot: {perf_stats.get('avg_processing_time', 0):.6f}s"
        )
        print(
            f"  Valeurs aberrantes filtrées: {perf_stats.get('total_outliers_removed', 0)}"
        )

    return results


def plot_performance_results(results):
    """
    Affiche un graphique des résultats de performance.

    Args:
        results: Dictionnaire des résultats par taille de lot
    """
    batch_sizes = list(results.keys())
    total_times = [results[b]["total_time"] for b in batch_sizes]
    avg_times = [results[b]["avg_processing_time"] for b in batch_sizes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Premier graphique: temps total
    ax1.plot(batch_sizes, total_times, "o-", color="blue")
    ax1.set_title("Temps total de traitement par taille de lot")
    ax1.set_xlabel("Taille du lot")
    ax1.set_ylabel("Temps total (s)")
    ax1.grid(True)

    # Deuxième graphique: temps moyen de traitement par lot
    ax2.plot(batch_sizes, avg_times, "o-", color="green")
    ax2.set_title("Temps moyen de traitement par lot")
    ax2.set_xlabel("Taille du lot")
    ax2.set_ylabel("Temps moyen (s)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("performance_results.png")
    print("Graphique des résultats enregistré sous 'performance_results.png'")


def main():
    """Fonction principale"""
    print("=== Test de performance de l'adaptateur en temps réel ===\n")

    # Exécuter les tests avec différentes tailles de lots
    results = run_performance_test(data_size=2000, batch_sizes=[10, 50, 100, 200, 500])

    # Afficher le récapitulatif des résultats
    print("\n=== Récapitulatif des résultats ===")
    for batch_size, stats in results.items():
        print(f"\nTaille de lot: {batch_size}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    # Tracer les résultats
    try:
        plot_performance_results(results)
    except Exception as e:
        print(f"Erreur lors de la création du graphique: {e}")


if __name__ == "__main__":
    main()
