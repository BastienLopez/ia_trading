"""
Visualisation des résultats de benchmarks.

Ce module contient des fonctions pour visualiser les résultats des benchmarks
des fonctionnalités transversales.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_trading.utils.advanced_logging import get_logger

# Logger pour ce module
logger = get_logger("ai_trading.tests.performance.visualization")


def load_benchmark_results(filepath):
    """
    Charge les résultats d'un benchmark depuis un fichier JSON.

    Args:
        filepath: Chemin vers le fichier JSON

    Returns:
        dict: Données de benchmark
    """
    logger.info(f"Chargement des résultats depuis {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def plot_benchmark_comparison(
    results1, results2, output_file=None, title1="Avant", title2="Après"
):
    """
    Compare visuellement deux séries de résultats de benchmarks.

    Args:
        results1: Premier ensemble de résultats (baseline)
        results2: Deuxième ensemble de résultats (comparaison)
        output_file: Fichier de sortie pour le graphique
        title1: Titre pour le premier ensemble
        title2: Titre pour le deuxième ensemble
    """
    if isinstance(results1, str):
        results1 = load_benchmark_results(results1)
    if isinstance(results2, str):
        results2 = load_benchmark_results(results2)

    # Créer une figure avec 3 sous-graphiques (un par catégorie)
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    categories = ["logging", "metrics", "checkpoints"]
    titles = ["Journalisation", "Métriques", "Checkpoints"]

    for i, (category, title) in enumerate(zip(categories, titles)):
        # Créer des dictionnaires pour faciliter la comparaison
        r1 = {r["name"]: r["average_ms"] for r in results1["results"][category]}
        r2 = {r["name"]: r["average_ms"] for r in results2["results"][category]}

        # Trouver les tests communs
        common_tests = sorted(set(r1.keys()) & set(r2.keys()))

        # Créer les données pour le graphique
        x = np.arange(len(common_tests))
        width = 0.35

        # Obtenir les valeurs pour les barres
        y1 = [r1[test] for test in common_tests]
        y2 = [r2[test] for test in common_tests]

        # Calculer le pourcentage de différence
        diff_pct = [
            (y2[j] - y1[j]) / y1[j] * 100 if y1[j] > 0 else 0 for j in range(len(y1))
        ]

        # Tracer les barres
        axs[i].bar(x - width / 2, y1, width, label=title1, color="royalblue")
        axs[i].bar(x + width / 2, y2, width, label=title2, color="orangered")

        # Ajouter les étiquettes et les pourcentages
        for j, (val1, val2, pct) in enumerate(zip(y1, y2, diff_pct)):
            # Mettre le texte au-dessus de la barre la plus haute
            ypos = max(val1, val2) + 1
            color = "green" if pct <= 0 else "red"
            axs[i].text(
                x[j],
                ypos,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                color=color,
                fontweight="bold",
            )

        # Configurer le graphique
        axs[i].set_xlabel("Test")
        axs[i].set_ylabel("Temps moyen (ms)")
        axs[i].set_title(f"Benchmark de {title}")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(common_tests, rotation=45, ha="right")
        axs[i].grid(True, linestyle="--", alpha=0.7)
        axs[i].legend()

    # Ajuster la mise en page
    plt.tight_layout()

    # Ajouter un titre général
    benchmark_date1 = datetime.fromtimestamp(results1["timestamp"]).strftime(
        "%Y-%m-%d %H:%M"
    )
    benchmark_date2 = datetime.fromtimestamp(results2["timestamp"]).strftime(
        "%Y-%m-%d %H:%M"
    )
    fig.suptitle(
        f"Comparaison des benchmarks\n{title1} ({benchmark_date1}) vs {title2} ({benchmark_date2})",
        fontsize=16,
    )
    plt.subplots_adjust(top=0.95)

    # Sauvegarder si demandé
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Graphique de comparaison sauvegardé dans {output_file}")

    return fig


def visualize_benchmark_results(results, output_file=None):
    """
    Visualise les résultats d'un benchmark.

    Args:
        results: Résultats du benchmark (fichier ou dictionnaire)
        output_file: Fichier de sortie pour le graphique
    """
    # Charger les résultats si c'est un chemin de fichier
    if isinstance(results, (str, Path)):
        results = load_benchmark_results(results)

    # Créer 3 sous-graphiques
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 1. Graphique pour la journalisation
    logging_results = results["results"]["logging"]
    names = [result["name"] for result in logging_results]
    avg_times = [result["average_ms"] for result in logging_results]
    min_times = [result["min_ms"] for result in logging_results]
    max_times = [result["max_ms"] for result in logging_results]

    x = np.arange(len(names))
    width = 0.6

    ax1.bar(x, avg_times, width, label="Temps moyen", color="skyblue")
    ax1.errorbar(
        x,
        avg_times,
        yerr=[
            [avg - min for avg, min in zip(avg_times, min_times)],
            [max - avg for avg, max in zip(avg_times, max_times)],
        ],
        fmt="o",
        color="black",
        capsize=5,
    )

    ax1.set_xlabel("Test")
    ax1.set_ylabel("Temps (ms)")
    ax1.set_title("Benchmarks de journalisation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # 2. Graphique pour les métriques
    metrics_results = results["results"]["metrics"]
    names = [result["name"] for result in metrics_results]
    avg_times = [result["average_ms"] for result in metrics_results]
    min_times = [result["min_ms"] for result in metrics_results]
    max_times = [result["max_ms"] for result in metrics_results]

    x = np.arange(len(names))

    ax2.bar(x, avg_times, width, label="Temps moyen", color="lightgreen")
    ax2.errorbar(
        x,
        avg_times,
        yerr=[
            [avg - min for avg, min in zip(avg_times, min_times)],
            [max - avg for avg, max in zip(avg_times, max_times)],
        ],
        fmt="o",
        color="black",
        capsize=5,
    )

    ax2.set_xlabel("Test")
    ax2.set_ylabel("Temps (ms)")
    ax2.set_title("Benchmarks de collecte de métriques")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # 3. Graphique pour les checkpoints
    checkpoint_results = results["results"]["checkpoints"]
    names = [result["name"] for result in checkpoint_results]
    avg_times = [result["average_ms"] for result in checkpoint_results]
    min_times = [result["min_ms"] for result in checkpoint_results]
    max_times = [result["max_ms"] for result in checkpoint_results]

    x = np.arange(len(names))

    ax3.bar(x, avg_times, width, label="Temps moyen", color="salmon")
    ax3.errorbar(
        x,
        avg_times,
        yerr=[
            [avg - min for avg, min in zip(avg_times, min_times)],
            [max - avg for avg, max in zip(avg_times, max_times)],
        ],
        fmt="o",
        color="black",
        capsize=5,
    )

    ax3.set_xlabel("Test")
    ax3.set_ylabel("Temps (ms)")
    ax3.set_title("Benchmarks de gestion des checkpoints")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Ajouter des informations générales
    benchmark_date = datetime.fromtimestamp(results["timestamp"]).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    duration = results["duration_seconds"]
    plt.figtext(
        0.5,
        0.01,
        f"Benchmark exécuté le {benchmark_date} (durée: {duration:.2f}s)",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "lavender", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout()

    # Sauvegarder le graphique si demandé
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Graphique sauvegardé dans {output_file}")

    return fig


def create_performance_report(results, output_file):
    """
    Crée un rapport HTML de performance à partir des résultats de benchmark.

    Args:
        results: Résultats du benchmark (fichier ou dictionnaire)
        output_file: Fichier de sortie pour le rapport HTML
    """
    # Charger les résultats si c'est un chemin de fichier
    if isinstance(results, (str, Path)):
        results = load_benchmark_results(results)

    # Créer un graphique et le sauvegarder comme image temporaire
    temp_img = os.path.join(os.path.dirname(output_file), "temp_benchmark.png")
    visualize_benchmark_results(results, temp_img)

    # Compiler les informations
    system_info = results["system_info"]
    benchmark_date = datetime.fromtimestamp(results["timestamp"]).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    duration = results["duration_seconds"]

    # Créer le contenu HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport de performance - {benchmark_date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .graph {{ margin: 20px 0; text-align: center; }}
            .graph img {{ max-width: 100%; }}
            .info-box {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rapport de performance des fonctionnalités transversales</h1>
            
            <div class="info-box">
                <h3>Informations générales</h3>
                <p><strong>Date du benchmark:</strong> {benchmark_date}</p>
                <p><strong>Durée totale:</strong> {duration:.2f} secondes</p>
                <p><strong>Python:</strong> {system_info.get('python_version', 'Non disponible')}</p>
                <p><strong>PyTorch:</strong> {system_info.get('torch_version', 'Non disponible')}</p>
                <p><strong>NumPy:</strong> {system_info.get('numpy_version', 'Non disponible')}</p>
            </div>
            
            <div class="graph">
                <h2>Visualisation des résultats</h2>
                <img src="temp_benchmark.png" alt="Graphique des benchmarks">
            </div>
            
            <h2>Résultats détaillés</h2>
    """

    # Ajouter les tableaux pour chaque catégorie
    categories = [
        ("Journalisation", "logging"),
        ("Métriques", "metrics"),
        ("Checkpoints", "checkpoints"),
    ]

    for title, key in categories:
        html_content += f"""
            <h3>{title}</h3>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Temps moyen (ms)</th>
                    <th>Temps min (ms)</th>
                    <th>Temps max (ms)</th>
                    <th>Nombre d'exécutions</th>
                </tr>
        """

        for result in results["results"][key]:
            html_content += f"""
                <tr>
                    <td>{result['name']}</td>
                    <td>{result['average_ms']:.3f}</td>
                    <td>{result['min_ms']:.3f}</td>
                    <td>{result['max_ms']:.3f}</td>
                    <td>{result['total_runs']}</td>
                </tr>
            """

        html_content += "</table>"

    html_content += """
        </div>
    </body>
    </html>
    """

    # Écrire le fichier HTML
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Rapport de performance créé dans {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualisation des résultats de benchmarks"
    )
    parser.add_argument(
        "input", help="Fichier JSON contenant les résultats du benchmark"
    )
    parser.add_argument(
        "--output", "-o", help="Fichier de sortie pour le graphique (PNG)"
    )
    parser.add_argument("--report", "-r", help="Générer un rapport HTML")
    parser.add_argument("--compare", "-c", help="Fichier JSON de comparaison")

    args = parser.parse_args()

    if args.compare:
        # Mode comparaison
        output = args.output or f"benchmark_comparison_{int(time.time())}.png"
        plot_benchmark_comparison(args.input, args.compare, output)
    elif args.report:
        # Générer un rapport HTML
        create_performance_report(args.input, args.report)
    else:
        # Visualisation simple
        output = args.output or f"benchmark_results_{int(time.time())}.png"
        visualize_benchmark_results(args.input, output)
