"""
Module de profilage pour les prédictions de marché.

Ce module implémente des outils de profilage pour identifier et optimiser
les goulots d'étranglement dans les modules de prédiction.
"""

import cProfile
import functools
import gc
import io
import json
import os
import pstats
import time
import tracemalloc
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt

# Configuration du logger
from ai_trading.utils import setup_logger

logger = setup_logger("performance_profiler")


class PerformanceProfiler:
    """
    Profileur de performance pour les opérations de prédiction de marché.

    Permet d'identifier les goulots d'étranglement dans le code et de suivre
    l'évolution des performances au fil du temps.
    """

    def __init__(
        self, output_dir: Optional[str] = None, enable_memory_tracking: bool = True
    ):
        """
        Initialise le profileur de performance.

        Args:
            output_dir: Répertoire de sortie pour les rapports (None = désactivé)
            enable_memory_tracking: Activer le suivi de la mémoire
        """
        self.output_dir = output_dir
        self.enable_memory_tracking = enable_memory_tracking

        # Création du répertoire de sortie si nécessaire
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Historique des mesures
        self.performance_history = []
        self.memory_snapshots = {}

        # État du traceur de mémoire
        self.memory_tracker_started = False

        logger.info(
            f"PerformanceProfiler initialisé, suivi mémoire: {'activé' if enable_memory_tracking else 'désactivé'}"
        )

    def profile_function(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Profile une fonction et renvoie son résultat et les statistiques.

        Args:
            func: Fonction à profiler
            *args, **kwargs: Arguments à passer à la fonction

        Returns:
            Tuple contenant le résultat de la fonction et les statistiques de profilage
        """
        # Démarrage du profilage
        profiler = cProfile.Profile()
        profiler.enable()

        # Mesure du temps d'exécution
        start_time = time.time()

        # Suivi de la mémoire
        if self.enable_memory_tracking and not self.memory_tracker_started:
            tracemalloc.start()
            self.memory_tracker_started = True

        # Capture de l'utilisation mémoire avant
        if self.enable_memory_tracking:
            gc.collect()
            memory_before = tracemalloc.get_traced_memory()[0]

        # Exécution de la fonction
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            exception = str(e)
            logger.error(f"Erreur lors du profilage de {func.__name__}: {e}")

        # Mesure du temps d'exécution
        execution_time = time.time() - start_time

        # Capture de l'utilisation mémoire après
        if self.enable_memory_tracking:
            gc.collect()
            memory_after = tracemalloc.get_traced_memory()[0]
            memory_diff = memory_after - memory_before
        else:
            memory_diff = None

        # Arrêt du profilage
        profiler.disable()

        # Analyse des résultats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 des fonctions les plus coûteuses

        # Extraction des statistiques importantes
        profile_result = s.getvalue()

        # Analyse des lignes les plus coûteuses
        try:
            # Extraction des lignes de statistiques
            stats_lines = profile_result.strip().split("\n")
            # Ignorer les lignes d'en-tête
            data_lines = [
                line
                for line in stats_lines
                if len(line.strip()) > 0
                and not line.startswith(("ncalls", "Ordered by"))
            ]
            # Extraire les 5 fonctions les plus coûteuses
            top_functions = []
            for i, line in enumerate(data_lines[:5]):
                parts = line.strip().split()
                if len(parts) >= 6:
                    # Format typique: ncalls tottime percall cumtime percall filename:lineno(function)
                    func_info = " ".join(parts[5:])
                    time_info = float(parts[3])  # cumtime
                    top_functions.append(
                        {
                            "function": func_info,
                            "cumulative_time": time_info,
                            "percentage": (
                                time_info / execution_time * 100
                                if execution_time > 0
                                else 0
                            ),
                        }
                    )
        except Exception as e:
            top_functions = []
            logger.error(f"Erreur lors de l'analyse des résultats de profilage: {e}")

        # Compilation des statistiques
        stats = {
            "function_name": func.__name__,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "memory_diff_bytes": memory_diff,
            "top_functions": top_functions,
        }

        if not success:
            stats["exception"] = exception

        # Sauvegarde des statistiques
        self.performance_history.append(stats)

        # Sauvegarde du rapport détaillé si un répertoire est spécifié
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.output_dir, f"profile_{func.__name__}_{timestamp}.txt"
            )

            with open(report_path, "w") as f:
                f.write(profile_result)

            # Sauvegarde des statistiques en JSON
            stats_path = os.path.join(
                self.output_dir, f"stats_{func.__name__}_{timestamp}.json"
            )
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Rapport de profilage sauvegardé dans {report_path}")

        return result, stats

    def take_memory_snapshot(self, label: str) -> Dict[str, Any]:
        """
        Prend un instantané de l'utilisation mémoire actuelle.

        Args:
            label: Étiquette pour identifier l'instantané

        Returns:
            Dictionnaire des statistiques mémoire
        """
        if not self.enable_memory_tracking:
            return {"error": "Suivi mémoire désactivé"}

        if not self.memory_tracker_started:
            tracemalloc.start()
            self.memory_tracker_started = True

        # Force la collecte des objets inutilisés
        gc.collect()

        # Capture du snapshot
        snapshot = tracemalloc.take_snapshot()

        # Statistiques générales
        current, peak = tracemalloc.get_traced_memory()

        # Analyse des statistiques par fichier
        stats = snapshot.statistics("filename")

        # Extraction des 10 plus grands consommateurs de mémoire
        top_stats = [
            {
                "file": stat.traceback[0].filename,
                "line": stat.traceback[0].lineno,
                "size": stat.size,
                "count": stat.count,
            }
            for stat in stats[:10]
        ]

        # Compilation des résultats
        memory_stats = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "current_bytes": current,
            "peak_bytes": peak,
            "top_allocations": top_stats,
        }

        # Sauvegarde de l'instantané
        self.memory_snapshots[label] = memory_stats

        # Sauvegarde du rapport détaillé si un répertoire est spécifié
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_path = os.path.join(
                self.output_dir, f"memory_{label}_{timestamp}.json"
            )

            with open(stats_path, "w") as f:
                json.dump(memory_stats, f, indent=2)

            logger.info(f"Instantané mémoire '{label}' sauvegardé dans {stats_path}")

        return memory_stats

    def compare_memory_snapshots(self, label1: str, label2: str) -> Dict[str, Any]:
        """
        Compare deux instantanés mémoire.

        Args:
            label1: Étiquette du premier instantané
            label2: Étiquette du deuxième instantané

        Returns:
            Dictionnaire de comparaison
        """
        if not all(label in self.memory_snapshots for label in [label1, label2]):
            missing = [
                label
                for label in [label1, label2]
                if label not in self.memory_snapshots
            ]
            return {"error": f"Instantanés manquants: {', '.join(missing)}"}

        snapshot1 = self.memory_snapshots[label1]
        snapshot2 = self.memory_snapshots[label2]

        # Différence de mémoire
        memory_diff = snapshot2["current_bytes"] - snapshot1["current_bytes"]

        # Compilation des résultats
        comparison = {
            "label1": label1,
            "label2": label2,
            "timestamp": datetime.now().isoformat(),
            "memory_diff_bytes": memory_diff,
            "memory_diff_percentage": (
                (memory_diff / snapshot1["current_bytes"]) * 100
                if snapshot1["current_bytes"] > 0
                else 0
            ),
            "snapshot1": snapshot1,
            "snapshot2": snapshot2,
        }

        # Sauvegarde du rapport si un répertoire est spécifié
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.output_dir,
                f"memory_comparison_{label1}_vs_{label2}_{timestamp}.json",
            )

            with open(report_path, "w") as f:
                json.dump(comparison, f, indent=2)

            logger.info(f"Comparaison mémoire sauvegardée dans {report_path}")

        return comparison

    def generate_performance_report(
        self, function_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère un rapport de performance basé sur l'historique.

        Args:
            function_name: Nom de la fonction à analyser (None = toutes)

        Returns:
            Dictionnaire du rapport
        """
        if not self.performance_history:
            return {"error": "Aucune donnée de performance disponible"}

        # Filtrage par fonction si spécifié
        if function_name:
            history = [
                item
                for item in self.performance_history
                if item["function_name"] == function_name
            ]
            if not history:
                return {"error": f"Aucune donnée pour la fonction {function_name}"}
        else:
            history = self.performance_history

        # Regroupement par fonction
        functions = {}
        for item in history:
            func_name = item["function_name"]
            if func_name not in functions:
                functions[func_name] = []
            functions[func_name].append(item)

        # Analyse par fonction
        function_stats = {}
        for func_name, items in functions.items():
            # Temps d'exécution
            execution_times = [
                item["execution_time"] for item in items if item["success"]
            ]

            if not execution_times:
                function_stats[func_name] = {"error": "Aucune exécution réussie"}
                continue

            # Utilisation mémoire
            memory_diffs = [
                item["memory_diff_bytes"]
                for item in items
                if item["success"] and item["memory_diff_bytes"] is not None
            ]

            # Statistiques
            function_stats[func_name] = {
                "calls": len(items),
                "successful_calls": len(execution_times),
                "execution_time": {
                    "min": min(execution_times),
                    "max": max(execution_times),
                    "mean": sum(execution_times) / len(execution_times),
                    "latest": execution_times[-1] if execution_times else None,
                },
            }

            if memory_diffs:
                function_stats[func_name]["memory_usage"] = {
                    "min": min(memory_diffs),
                    "max": max(memory_diffs),
                    "mean": sum(memory_diffs) / len(memory_diffs),
                    "latest": memory_diffs[-1] if memory_diffs else None,
                }

            # Extraction des goulots d'étranglement
            bottlenecks = []
            for item in items:
                if "top_functions" in item and item["top_functions"]:
                    for func_info in item["top_functions"]:
                        bottlenecks.append(func_info)

            if bottlenecks:
                # Regroupement des bottlenecks par fonction
                grouped_bottlenecks = {}
                for bottleneck in bottlenecks:
                    func = bottleneck["function"]
                    if func not in grouped_bottlenecks:
                        grouped_bottlenecks[func] = []
                    grouped_bottlenecks[func].append(bottleneck["percentage"])

                # Calcul des moyennes
                top_bottlenecks = []
                for func, percentages in grouped_bottlenecks.items():
                    top_bottlenecks.append(
                        {
                            "function": func,
                            "average_percentage": sum(percentages) / len(percentages),
                            "appearances": len(percentages),
                        }
                    )

                # Tri par pourcentage moyen
                top_bottlenecks.sort(
                    key=lambda x: x["average_percentage"], reverse=True
                )

                function_stats[func_name]["bottlenecks"] = top_bottlenecks[:5]  # Top 5

        # Compilation du rapport global
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_profiles": len(history),
            "function_details": function_stats,
        }

        # Sauvegarde du rapport si un répertoire est spécifié
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = function_name or "all"
            report_path = os.path.join(
                self.output_dir, f"performance_report_{fname}_{timestamp}.json"
            )

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Génération de graphiques
            self._generate_performance_charts(report, function_name)

            logger.info(f"Rapport de performance sauvegardé dans {report_path}")

        return report

    def _generate_performance_charts(
        self, report: Dict[str, Any], function_name: Optional[str] = None
    ) -> None:
        """
        Génère des graphiques de performance.

        Args:
            report: Rapport de performance
            function_name: Nom de la fonction spécifique
        """
        if not self.output_dir:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = function_name or "all"

            # Extraction des données
            functions = []
            execution_times = []
            memory_usages = []

            for func_name, stats in report["function_details"].items():
                if "error" in stats:
                    continue

                functions.append(func_name)
                execution_times.append(stats["execution_time"]["mean"])

                if "memory_usage" in stats:
                    # Conversion en MB pour lisibilité
                    memory_usages.append(stats["memory_usage"]["mean"] / (1024 * 1024))
                else:
                    memory_usages.append(0)

            if not functions:
                return

            # Création de la figure et des sous-graphiques
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

            # Graphique des temps d'exécution
            bars1 = ax1.bar(range(len(functions)), execution_times, color="skyblue")
            ax1.set_xlabel("Fonctions")
            ax1.set_ylabel("Temps moyen (secondes)")
            ax1.set_title("Temps d'exécution moyen par fonction")
            ax1.set_xticks(range(len(functions)))
            ax1.set_xticklabels(functions, rotation=45, ha="right")

            # Ajout des valeurs sur les barres
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.annotate(
                    f"{height:.3f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points de décalage vertical
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

            # Graphique de l'utilisation mémoire
            bars2 = ax2.bar(range(len(functions)), memory_usages, color="lightgreen")
            ax2.set_xlabel("Fonctions")
            ax2.set_ylabel("Utilisation mémoire moyenne (MB)")
            ax2.set_title("Utilisation mémoire moyenne par fonction")
            ax2.set_xticks(range(len(functions)))
            ax2.set_xticklabels(functions, rotation=45, ha="right")

            # Ajout des valeurs sur les barres
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.annotate(
                    f"{height:.2f} MB",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points de décalage vertical
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()

            # Sauvegarde du graphique
            chart_path = os.path.join(
                self.output_dir, f"performance_chart_{fname}_{timestamp}.png"
            )
            plt.savefig(chart_path)
            plt.close()

            logger.info(f"Graphique de performance sauvegardé dans {chart_path}")

            # Création d'un graphique pour les goulots d'étranglement
            for func_name, stats in report["function_details"].items():
                if "error" in stats or "bottlenecks" not in stats:
                    continue

                bottlenecks = stats["bottlenecks"]
                if not bottlenecks:
                    continue

                # Extraction des noms de fonction simplifiés et pourcentages
                bottleneck_names = []
                bottleneck_percentages = []

                for bottleneck in bottlenecks:
                    # Simplification du nom de fonction pour l'affichage
                    name = bottleneck["function"]
                    if "(" in name:
                        name = name.split("(")[1].split(")")[
                            0
                        ]  # Extraction du nom entre parenthèses

                    # Troncature si trop long
                    if len(name) > 30:
                        name = name[:27] + "..."

                    bottleneck_names.append(name)
                    bottleneck_percentages.append(bottleneck["average_percentage"])

                # Création du graphique
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    range(len(bottleneck_names)), bottleneck_percentages, color="salmon"
                )
                plt.xlabel("Fonctions")
                plt.ylabel("Pourcentage du temps d'exécution (%)")
                plt.title(f"Goulots d'étranglement pour {func_name}")
                plt.xticks(
                    range(len(bottleneck_names)),
                    bottleneck_names,
                    rotation=45,
                    ha="right",
                )

                # Ajout des valeurs sur les barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.annotate(
                        f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points de décalage vertical
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

                plt.tight_layout()

                # Sauvegarde du graphique
                bottleneck_chart_path = os.path.join(
                    self.output_dir, f"bottleneck_chart_{func_name}_{timestamp}.png"
                )
                plt.savefig(bottleneck_chart_path)
                plt.close()

                logger.info(
                    f"Graphique des goulots d'étranglement pour {func_name} sauvegardé dans {bottleneck_chart_path}"
                )

        except Exception as e:
            logger.error(f"Erreur lors de la génération des graphiques: {e}")

    def stop_tracking(self) -> None:
        """
        Arrête le suivi de la mémoire.
        """
        if self.memory_tracker_started:
            tracemalloc.stop()
            self.memory_tracker_started = False
            logger.info("Suivi mémoire arrêté")

    def clear_history(self) -> None:
        """
        Efface l'historique des mesures.
        """
        self.performance_history = []
        self.memory_snapshots = {}
        logger.info("Historique des mesures effacé")


def profile(output_dir: Optional[str] = None, enable_memory_tracking: bool = True):
    """
    Décorateur pour profiler une fonction.

    Args:
        output_dir: Répertoire de sortie pour les rapports (None = désactivé)
        enable_memory_tracking: Activer le suivi de la mémoire

    Returns:
        Fonction décorée
    """
    profiler = PerformanceProfiler(
        output_dir=output_dir, enable_memory_tracking=enable_memory_tracking
    )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, _ = profiler.profile_function(func, *args, **kwargs)
            return result

        # Ajout d'une référence au profilage
        wrapper.profiler = profiler
        wrapper.get_profile_report = lambda: profiler.generate_performance_report(
            func.__name__
        )

        return wrapper

    return decorator
