"""
Module d'analyse de sensibilité pour évaluer la robustesse des stratégies de trading
en faisant varier les paramètres clés.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import json
import os
import seaborn as sns

from ai_trading.ml.backtesting.backtest_engine import BacktestEngine, BacktestResult

# Configuration du logging
logger = logging.getLogger(__name__)

@dataclass
class ParameterRange:
    """Définition d'une plage de paramètres à tester."""
    name: str
    values: List[Union[float, int, str, bool]]
    display_name: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name.replace("_", " ").title()

@dataclass
class SensitivityResult:
    """Résultat d'une analyse de sensibilité."""
    parameter_ranges: List[ParameterRange]
    results: List[Tuple[Dict, BacktestResult]]
    metrics: pd.DataFrame
    best_parameters: Dict
    worst_parameters: Dict
    most_sensitive_parameters: List[str]
    metadata: Dict = field(default_factory=dict)
    
    def _serialize_value(self, value):
        """
        Sérialise une valeur pour la sauvegarde JSON.
        Gère les types non standards comme numpy arrays, timestamps, et objets complexes.
        
        Args:
            value: Valeur à sérialiser
            
        Returns:
            Valeur sérialisable
        """
        if value is None:
            return None
            
        # Cas spéciaux pour les types numpy
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
            
        # Cas pandas
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, pd.DataFrame):
            # Pour les DataFrames, ne garder que les colonnes les plus importantes
            # et limiter le nombre de lignes pour réduire la taille
            if len(value) > 1000:
                sample = value.sample(n=1000, random_state=42)
                return {
                    "type": "dataframe_sample",
                    "data": sample.to_dict(orient="split"),
                    "original_shape": value.shape
                }
            return value.to_dict(orient="split")
        elif isinstance(value, pd.Series):
            if len(value) > 1000:
                sample = value.sample(n=1000, random_state=42)
                return {
                    "type": "series_sample",
                    "data": sample.to_dict(),
                    "original_shape": (len(value),)
                }
            return value.to_dict()
            
        # Types de base Python
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
            
        # Pour les objets non sérialisables, essayer de les convertir en string
        else:
            try:
                return str(value)
            except:
                return f"<Non-serializable object of type {type(value).__name__}>"
    
    def save(self, filepath: str):
        """
        Sauvegarde les résultats de l'analyse de sensibilité.
        
        Args:
            filepath: Chemin du fichier pour la sauvegarde
        """
        # Créer un dictionnaire avec les données à sauvegarder
        data = {
            "parameter_ranges": [
                {
                    "name": p.name,
                    "values": self._serialize_value(p.values),
                    "display_name": p.display_name,
                    "description": p.description
                } for p in self.parameter_ranges
            ],
            "best_parameters": self._serialize_value(self.best_parameters),
            "worst_parameters": self._serialize_value(self.worst_parameters),
            "most_sensitive_parameters": self._serialize_value(self.most_sensitive_parameters),
            "metadata": self._serialize_value(self.metadata)
        }
        
        # Si les métriques ne sont pas vides, les sauvegarder aussi
        # Note: on évite de sauvegarder tous les résultats car ils peuvent être très volumineux
        if not self.metrics.empty:
            # Pour réduire la taille, on ne garde que les colonnes essentielles
            essential_metrics = list(self.best_parameters.keys()) + list(self.metadata.get("target_metrics", []))
            if self.metadata.get("target_metric"):
                essential_metrics.append(self.metadata["target_metric"])
                
            # Éliminer les doublons
            essential_metrics = list(set(essential_metrics))
            
            # S'assurer que toutes les colonnes existent
            metrics_columns = list(self.metrics.columns)
            essential_metrics = [col for col in essential_metrics if col in metrics_columns]
            
            # Si on n'a pas assez de colonnes essentielles, garder les N premières colonnes
            if len(essential_metrics) < min(5, len(metrics_columns)):
                essential_metrics = metrics_columns[:min(10, len(metrics_columns))]
                
            # Sauvegarder un échantillon du DataFrame si grand
            metrics_df = self.metrics
            if len(metrics_df) > 1000:
                metrics_df = metrics_df.sample(n=1000, random_state=42)
                
            # Convertir en dict efficacement
            data["metrics"] = {
                "columns": list(metrics_df.columns),
                "data": metrics_df.to_dict(orient="records"),
                "original_shape": self.metrics.shape
            }
        
        # Ajouter un indicateur de version pour les chargements futurs
        data["_version"] = "1.1"
        
        # Sauvegarder avec gestion des erreurs
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Utiliser un format compact pour réduire la taille du fichier
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=None, separators=(",", ":"))
                
            logger.info(f"Résultats d'analyse de sensibilité sauvegardés dans {filepath}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            return False
    
    @classmethod
    def load(cls, filepath: str):
        """
        Charge les résultats d'une analyse de sensibilité.
        
        Args:
            filepath: Chemin du fichier à charger
            
        Returns:
            Instance de SensitivityResult
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Vérifier la version
            version = data.get("_version", "1.0")
            
            # Reconstruire les parameter_ranges
            parameter_ranges = []
            for p_data in data["parameter_ranges"]:
                parameter_ranges.append(ParameterRange(
                    name=p_data["name"],
                    values=p_data["values"],
                    display_name=p_data.get("display_name", p_data["name"]),
                    description=p_data.get("description", "")
                ))
                
            # Reconstruire les métriques si présentes
            metrics = pd.DataFrame()
            if "metrics" in data:
                try:
                    if isinstance(data["metrics"], dict) and "data" in data["metrics"]:
                        # Format nouveau (v1.1+)
                        metrics = pd.DataFrame(data["metrics"]["data"])
                    else:
                        # Ancien format
                        metrics = pd.DataFrame(data["metrics"])
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement des métriques: {str(e)}")
            
            # Créer l'instance
            result = cls(
                parameter_ranges=parameter_ranges,
                results=[],  # On ne sauvegarde pas les résultats complets
                metrics=metrics,
                best_parameters=data["best_parameters"],
                worst_parameters=data["worst_parameters"],
                most_sensitive_parameters=data["most_sensitive_parameters"],
                metadata=data["metadata"]
            )
            
            logger.info(f"Résultats d'analyse de sensibilité chargés depuis {filepath}")
            return result
        except Exception as e:
            logger.error(f"Erreur lors du chargement des résultats: {str(e)}")
            return None
    
    def plot_sensitivity(self, metric: str = "total_return", top_n: int = 3):
        """
        Visualise l'impact des paramètres les plus sensibles sur une métrique.
        
        Args:
            metric: Métrique à analyser
            top_n: Nombre de paramètres les plus sensibles à afficher
        """
        if not self.metrics.empty:
            # Vérifier si la métrique existe
            if metric not in self.metrics.columns:
                logger.warning(f"Métrique {metric} non trouvée dans les résultats")
                return None
            
            # Conversion en numpy pour performance avec protection contre les valeurs non valides
            metric_values = np.array(self.metrics[metric])
            
            # Filtrer les valeurs non-finies
            valid_metrics = np.isfinite(metric_values)
            if not np.any(valid_metrics):
                logger.warning(f"Aucune valeur valide pour la métrique {metric}")
                return None
                
            # Obtenir les paramètres les plus sensibles par leur valeur de sensibilité
            if isinstance(self.most_sensitive_parameters, dict):
                sorted_params = sorted(self.most_sensitive_parameters.items(), key=lambda x: -x[1])
                most_sensitive = [p[0] for p in sorted_params[:top_n]]
            else:
                # Si most_sensitive_parameters est une liste
                most_sensitive = self.most_sensitive_parameters[:min(top_n, len(self.most_sensitive_parameters))]
            
            if not most_sensitive:
                logger.warning("Aucun paramètre sensible identifié")
                return None
                
            # Créer une figure plus grande pour améliorer la visibilité
            fig, axes = plt.subplots(1, min(top_n, len(most_sensitive)), figsize=(15, 5), sharey=True)
            if top_n == 1 and len(most_sensitive) > 0:
                axes = [axes]  # Convertir en liste pour l'indexation uniforme
                
            for i, param in enumerate(most_sensitive[:top_n]):
                if i >= len(axes):
                    break
                    
                # Éviter de traiter des colonnes manquantes
                if param not in self.metrics.columns:
                    logger.warning(f"Paramètre {param} non trouvé dans les données")
                    continue
                
                # Obtenir les valeurs uniques du paramètre
                param_values = np.array(self.metrics[param])
                
                # Gérer plus efficacement les paramètres à valeurs continues
                unique_count = len(np.unique(param_values))
                if unique_count > 15:  # Binning pour les paramètres avec trop de valeurs uniques
                    # Créer des bins pour plus de clarté visuelle
                    bins = np.linspace(np.min(param_values), np.max(param_values), 10)
                    binned_values = []
                    bin_centers = []
                    binned_metrics = []
                    
                    for j in range(len(bins)-1):
                        bin_start, bin_end = bins[j], bins[j+1]
                        bin_center = (bin_start + bin_end) / 2
                        
                        # Créer un masque pour les valeurs dans ce bin
                        bin_mask = (param_values >= bin_start) & (param_values < bin_end) & valid_metrics
                        bin_metrics = metric_values[bin_mask]
                        
                        if len(bin_metrics) > 0:
                            binned_values.append((bin_start, bin_end))
                            bin_centers.append(bin_center)
                            binned_metrics.append(bin_metrics)
                    
                    # Préparer les données pour le graphique
                    x_values = bin_centers
                    y_values = [np.median(metrics) for metrics in binned_metrics]
                    y_errors = [(np.percentile(metrics, 75) - np.percentile(metrics, 25))/2 
                               for metrics in binned_metrics if len(metrics) > 1]
                    
                    # Ajuster pour les bins sans suffisamment de données
                    if len(y_errors) < len(y_values):
                        y_errors = y_errors + [0] * (len(y_values) - len(y_errors))
                else:
                    # Pour les paramètres avec peu de valeurs uniques, utiliser l'approche originale
                    x_values = []
                    y_values = []
                    y_errors = []
                    
                    for val in np.sort(np.unique(param_values)):
                        # Filtrer pour cette valeur de paramètre
                        param_mask = (self.metrics[param] == val).values
                        subset_metrics = metric_values[param_mask & valid_metrics]
                        
                        if len(subset_metrics) > 0:
                            # Calculer des statistiques robustes
                            # Utiliser la médiane au lieu de la moyenne pour être moins sensible aux valeurs extrêmes
                            median_val = np.median(subset_metrics)
                            
                            # Calculer les quartiles pour une mesure robuste de la dispersion
                            if len(subset_metrics) > 1:
                                q1 = np.percentile(subset_metrics, 25)
                                q3 = np.percentile(subset_metrics, 75)
                                iqr = q3 - q1
                            else:
                                iqr = 0
                            
                            x_values.append(val)
                            y_values.append(median_val)
                            y_errors.append(iqr / 2)  # Moitié de l'IQR comme barre d'erreur
                
                if len(x_values) > 0:
                    # Créer le graphique avec les barres d'erreur
                    axes[i].errorbar(x_values, y_values, yerr=y_errors, marker='o', linestyle='-', capsize=5)
                    
                    # Améliorer l'apparence du graphique
                    display_name = next((p.display_name for p in self.parameter_ranges if p.name == param), param)
                    axes[i].set_xlabel(display_name)
                    
                    # Afficher la valeur de sensibilité si disponible
                    sensitivity_val = self.most_sensitive_parameters.get(param, 0) if isinstance(self.most_sensitive_parameters, dict) else 0
                    axes[i].set_title(f"Sensibilité: {sensitivity_val:.2f}")
                    
                    # Optimiser l'affichage des ticks pour les paramètres à valeurs nombreuses
                    if len(x_values) > 10:
                        # Réduire le nombre d'étiquettes sur l'axe x pour éviter l'encombrement
                        step = max(1, len(x_values) // 8)
                        axes[i].set_xticks(x_values[::step])
                        axes[i].set_xticklabels([f"{x:.2g}" for x in x_values[::step]], rotation=45)
                    
                    # Ajouter une grille pour faciliter la lecture
                    axes[i].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            metric_display_name = metric.replace('_', ' ').title()
            plt.suptitle(f"Impact des paramètres sur {metric_display_name}", fontsize=16)
            plt.subplots_adjust(top=0.85)
            
            return fig
        return None
    
    def plot_heatmap(self, param1: str, param2: str, metric: str = "total_return"):
        """
        Crée un heatmap pour visualiser l'interaction entre deux paramètres.
        
        Args:
            param1: Premier paramètre
            param2: Second paramètre
            metric: Métrique à visualiser
        """
        if not self.metrics.empty and param1 in self.metrics.columns and param2 in self.metrics.columns:
            try:
                # Extraire les données nécessaires
                df = self.metrics[[param1, param2, metric]].copy()
                
                # Filtrer les valeurs non-finies pour éviter les problèmes
                df = df[np.isfinite(df[metric])]
                
                if df.empty:
                    logger.warning(f"Aucune valeur valide pour la métrique {metric}")
                    return None
                
                # Optimisation pour grandes données: échantillonner si le DataFrame est trop grand
                if len(df) > 10000:
                    df = df.sample(n=10000, random_state=42)
                    logger.info(f"Échantillonnage à 10000 points pour le heatmap (sur {len(self.metrics)} points)")
                
                # Déterminer les valeurs uniques pour chaque paramètre
                unique_param1 = np.sort(df[param1].unique())
                unique_param2 = np.sort(df[param2].unique())
                
                # Si trop de valeurs uniques, simplifier en utilisant des bins
                max_bins = 20  # Limite pour éviter des heatmaps trop détaillées
                
                if len(unique_param1) > max_bins:
                    # Utiliser des quantiles pour une meilleure répartition
                    quantiles = np.linspace(0, 1, max_bins)
                    bins1 = np.quantile(df[param1], quantiles)
                    # S'assurer que les bins sont uniques
                    bins1 = np.unique(bins1)
                    if len(bins1) <= 1:
                        # Fallback si les quantiles ne fonctionnent pas bien
                        bins1 = np.linspace(df[param1].min(), df[param1].max(), max_bins)
                    
                    labels1 = [(bins1[i] + bins1[i+1])/2 for i in range(len(bins1)-1)]
                    df['binned_' + param1] = pd.cut(df[param1], bins=bins1, labels=labels1)
                    param1_key = 'binned_' + param1
                else:
                    param1_key = param1
                
                if len(unique_param2) > max_bins:
                    # Utiliser des quantiles pour une meilleure répartition
                    quantiles = np.linspace(0, 1, max_bins)
                    bins2 = np.quantile(df[param2], quantiles)
                    # S'assurer que les bins sont uniques
                    bins2 = np.unique(bins2)
                    if len(bins2) <= 1:
                        # Fallback si les quantiles ne fonctionnent pas bien
                        bins2 = np.linspace(df[param2].min(), df[param2].max(), max_bins)
                        
                    labels2 = [(bins2[i] + bins2[i+1])/2 for i in range(len(bins2)-1)]
                    df['binned_' + param2] = pd.cut(df[param2], bins=bins2, labels=labels2)
                    param2_key = 'binned_' + param2
                else:
                    param2_key = param2
                
                # Créer la matrice du heatmap avec gestion des NaN
                # Utiliser la médiane pour être plus robuste aux outliers
                pivot_table = df.pivot_table(
                    values=metric, 
                    index=param1_key, 
                    columns=param2_key, 
                    aggfunc=np.median
                )
                
                # Gérer les valeurs manquantes plus intelligemment
                # Utiliser une interpolation pour les NaN isolés
                if pivot_table.isna().sum().sum() > 0 and pivot_table.isna().sum().sum() < pivot_table.size * 0.3:
                    # Interpolation pour les NaN isolés
                    pivot_table = pivot_table.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
                
                # Si encore des NaN, les remplacer par la valeur minimale
                pivot_table = pivot_table.fillna(pivot_table.min().min())
                
                # Créer la figure
                plt.figure(figsize=(12, 8))
                
                # Déterminer la palette de couleurs en fonction de la métrique
                # Pour certaines métriques, des valeurs plus petites sont meilleures (comme drawdown)
                if metric in ["max_drawdown", "volatility"]:
                    cmap = "YlOrRd_r"  # Palette inversée
                else:
                    cmap = "YlGnBu"
                
                # Créer le heatmap avec une colorbar plus informative
                ax = sns.heatmap(
                    pivot_table,
                    annot=False,  # Trop de valeurs pour les annotations
                    fmt=".2f",
                    cmap=cmap,
                    linewidths=0.5,
                    cbar_kws={"label": metric, "shrink": 0.8}
                )
                
                # Améliorer l'apparence
                # Obtenir les noms d'affichage des paramètres s'ils existent
                param1_display = next((p.display_name for p in self.parameter_ranges if p.name == param1), param1)
                param2_display = next((p.display_name for p in self.parameter_ranges if p.name == param2), param2)
                
                plt.title(f"Impact de {param1_display} et {param2_display} sur {metric}", fontsize=14)
                plt.tight_layout()
                
                # Optimiser les ticks avec rotation pour éviter l'encombrement
                if len(pivot_table.index) > 10:
                    step = max(1, len(pivot_table.index) // 8)
                    ax.set_yticks(np.arange(len(pivot_table.index))[::step] + 0.5)
                    ax.set_yticklabels([f"{x:.3g}" for x in pivot_table.index[::step]], rotation=0)
                
                if len(pivot_table.columns) > 10:
                    step = max(1, len(pivot_table.columns) // 8)
                    ax.set_xticks(np.arange(len(pivot_table.columns))[::step] + 0.5)
                    ax.set_xticklabels([f"{x:.3g}" for x in pivot_table.columns[::step]], rotation=45)
                
                return plt.gcf()
            except Exception as e:
                logger.error(f"Erreur lors de la création du heatmap: {str(e)}")
                return None
        return None
    
    def plot_parallel_coordinates(self, top_n: int = 20, metric: str = "total_return"):
        """
        Crée un graphique de coordonnées parallèles pour visualiser les meilleures combinaisons.
        
        Args:
            top_n: Nombre de meilleures combinaisons à afficher
            metric: Métrique à utiliser pour le classement
        """
        if self.metrics.empty or len(self.metrics) < top_n:
            logger.warning("Pas assez de données pour générer un graphique de coordonnées parallèles")
            return None
            
        # Sélectionner les top_n meilleures combinaisons
        top_results = self.metrics.sort_values(metric, ascending=False).head(top_n)
        
        # Sélectionner uniquement les colonnes de paramètres et la métrique
        param_columns = [p.name for p in self.parameter_ranges]
        plot_data = top_results[param_columns + [metric]]
        
        # Normaliser les données pour le graphique
        normalized_data = plot_data.copy()
        for col in plot_data.columns:
            if col in param_columns:
                # Convertir les valeurs de paramètres en catégories numériques
                unique_values = sorted(plot_data[col].unique())
                value_map = {val: i for i, val in enumerate(unique_values)}
                normalized_data[col] = plot_data[col].map(value_map)
            
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Obtenir les coordonnées pour chaque axe
        x = list(range(len(plot_data.columns)))
        
        # Tracer chaque combinaison de paramètres
        for i, (_, row) in enumerate(normalized_data.iterrows()):
            y = row.values
            ax.plot(x, y, marker='o', linestyle='-', alpha=0.7, 
                   label=f"Rank {i+1}: {plot_data[metric].iloc[i]:.2f}")
            
        # Configurer les axes
        ax.set_xticks(x)
        ax.set_xticklabels([col.replace("_", " ").title() for col in plot_data.columns], rotation=45)
        
        # Ajouter des ticks personnalisés pour chaque paramètre
        for i, col in enumerate(plot_data.columns):
            if col in param_columns:
                unique_values = sorted(plot_data[col].unique())
                value_map = {i: val for i, val in enumerate(unique_values)}
                
                # Créer un axe secondaire pour ce paramètre
                ax2 = ax.twinx()
                ax2.spines['right'].set_position(('axes', i/(len(plot_data.columns)-1)))
                ax2.set_yticks(range(len(value_map)))
                ax2.set_yticklabels([str(value_map[j]) for j in range(len(value_map))])
                ax2.set_ylim(ax.get_ylim())
        
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Top {top_n} Combinaisons de Paramètres par {metric.replace('_', ' ').title()}")
        
        plt.tight_layout()
        return fig

class SensitivityAnalyzer:
    """
    Classe pour l'analyse de sensibilité des stratégies de trading
    en faisant varier les paramètres clés.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur de sensibilité.
        
        Args:
            config: Configuration pour l'analyse
        """
        self.config = config or {}
        
        # Paramètres par défaut
        self.default_params = {
            "parallel": {
                "enabled": True,
                "max_workers": 4  # Nombre de processus parallèles
            },
            "metrics": [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate"
            ],
            "optimization_target": "sharpe_ratio"  # Métrique à optimiser
        }
        
        # Fusionner avec la configuration fournie
        for key, default_value in self.default_params.items():
            if key in self.config:
                if isinstance(default_value, dict) and isinstance(self.config[key], dict):
                    for sub_key, sub_value in default_value.items():
                        if sub_key not in self.config[key]:
                            self.config[key][sub_key] = sub_value
                else:
                    continue  # Conserver la valeur fournie
            else:
                self.config[key] = default_value
    
    @staticmethod
    def _run_backtest(params_values, param_names, market_data, strategy_fn, base_parameters, backtest_config=None):
        """
        Exécute un backtest avec un ensemble de paramètres.
        Cette méthode est définie statiquement pour permettre la sérialisation avec pickle.
        
        Args:
            params_values: Valeurs des paramètres pour cette combinaison
            param_names: Noms des paramètres correspondants
            market_data: Données de marché
            strategy_fn: Fonction de stratégie
            base_parameters: Paramètres de base
            backtest_config: Configuration optionnelle du backtest
            
        Returns:
            Tuple (params, result) ou None en cas d'erreur
        """
        try:
            # Créer le dictionnaire de paramètres pour cette combinaison
            params = base_parameters.copy()
            for i, name in enumerate(param_names):
                params[name] = params_values[i]
                
            # Créer un moteur de backtest
            engine = BacktestEngine(backtest_config)
            
            # Exécuter le backtest
            result = engine.run(market_data, strategy_fn, params)
            return params, result
        except Exception as e:
            logger.error(f"Erreur lors du backtest avec paramètres {params_values}: {str(e)}")
            return None
    
    def analyze(self, 
              market_data: Dict[str, pd.DataFrame], 
              strategy_fn: Callable, 
              parameter_ranges: List[ParameterRange],
              base_parameters: Dict = None) -> SensitivityResult:
        """
        Effectue une analyse de sensibilité pour une stratégie.
        
        Args:
            market_data: Dictionnaire des données de marché par symbole
            strategy_fn: Fonction de stratégie qui génère des signaux
            parameter_ranges: Liste des plages de paramètres à tester
            base_parameters: Paramètres de base (non variables) pour la stratégie
            
        Returns:
            Résultats de l'analyse de sensibilité
        """
        if not parameter_ranges:
            logger.error("Aucune plage de paramètres fournie pour l'analyse")
            return None
            
        base_parameters = base_parameters or {}
        
        # Créer toutes les combinaisons de paramètres à tester
        param_names = [p.name for p in parameter_ranges]
        param_values = [p.values for p in parameter_ranges]
        
        # Calculer le nombre total de combinaisons sans les générer toutes immédiatement
        # pour économiser de la mémoire
        total_combinations = np.prod([len(values) for values in param_values])
        
        logger.info(f"Analyse de sensibilité avec {total_combinations} combinaisons de paramètres")
        
        # Configuration pour le suivi de progression
        progress_interval = max(1, int(total_combinations / 100))  # Afficher la progression tous les 1%
        completed = 0
        start_time = pd.Timestamp.now()
        
        # Exécuter les backtests
        results = []
        error_count = 0
        
        # Optimisation de la gestion des données
        # Prétraiter les données de marché pour éviter les copies multiples
        # Vérifier si les données de marché sont déjà optimisées pour numpy
        optimized_market_data = {}
        for symbol, df in market_data.items():
            # Convertir en numpy pour des calculs plus rapides si nécessaire
            if isinstance(df, pd.DataFrame):
                # Garder l'index pour la correspondance temporelle
                optimized_market_data[symbol] = {
                    'data': df.copy(),
                    'index': df.index,
                    'columns': df.columns
                }
            else:
                optimized_market_data[symbol] = df
        
        # Récupérer la config du backtest pour la passer aux processus
        backtest_config = self.config.get("backtest_config", None)
        
        if self.config["parallel"]["enabled"] and total_combinations > 1:
            # Exécution parallèle avec gestion des erreurs améliorée
            max_workers = min(self.config["parallel"]["max_workers"], int(total_combinations))
            logger.info(f"Exécution parallèle avec {max_workers} workers")
            
            # Préparer les arguments pour le processus parallèle
            # Pour le parallélisme, nous devons passer les arguments explicitement
            # car les fonctions locales ne peuvent pas être sérialisées
            task_args = []
            for combo in itertools.product(*param_values):
                task_args.append((combo, param_names, market_data, strategy_fn, base_parameters, backtest_config))
            
            # Utiliser ProcessPoolExecutor si le nombre de combinaisons est très grand
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Soumettre les tâches avec les arguments extraits
                futures = [executor.submit(SensitivityAnalyzer._run_backtest, *args) for args in task_args]
                
                # Traiter les résultats au fur et à mesure
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        completed += 1
                        
                        if result:
                            results.append(result)
                        else:
                            error_count += 1
                        
                        # Afficher la progression
                        if completed % progress_interval == 0 or completed == total_combinations:
                            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                            progress_pct = (completed / total_combinations) * 100
                            est_total = (elapsed / completed) * total_combinations if completed > 0 else 0
                            remaining = est_total - elapsed
                            
                            logger.info(f"Progression: {progress_pct:.1f}% ({completed}/{total_combinations}) "
                                        f"- Temps restant estimé: {remaining/60:.1f} min "
                                        f"- Erreurs: {error_count}")
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Erreur lors du traitement du résultat: {str(e)}")
        else:
            # Exécution séquentielle optimisée
            logger.info("Exécution séquentielle")
            
            # Boucle optimisée pour les combinaisons de paramètres
            for combo_idx, combo in enumerate(itertools.product(*param_values)):
                result = SensitivityAnalyzer._run_backtest(combo, param_names, market_data, strategy_fn, base_parameters, backtest_config)
                completed += 1
                
                if result:
                    results.append(result)
                else:
                    error_count += 1
                
                # Afficher la progression
                if completed % progress_interval == 0 or completed == total_combinations:
                    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                    progress_pct = (completed / total_combinations) * 100
                    est_total = (elapsed / completed) * total_combinations if completed > 0 else 0
                    remaining = est_total - elapsed
                    
                    logger.info(f"Progression: {progress_pct:.1f}% ({completed}/{total_combinations}) "
                                f"- Temps restant estimé: {remaining/60:.1f} min "
                                f"- Erreurs: {error_count}")
                    
                # Libérer la mémoire si nécessaire
                if completed % 1000 == 0:
                    import gc
                    gc.collect()
        
        # Journaliser les résultats finaux
        total_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Analyse terminée en {total_time/60:.1f} minutes - "
                    f"{len(results)}/{total_combinations} combinaisons réussies, {error_count} erreurs")
        
        if not results:
            logger.error("Aucun résultat valide obtenu, impossible de continuer l'analyse")
            return None
        
        # Compiler les métriques de tous les résultats avec numpy pour de meilleures performances
        # Pré-allouer les structures pour les métriques
        metrics_list = []
        
        # Métriques à extraire
        target_metrics = set(self.config["metrics"] + ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor", "avg_trade"])
        
        for params, result in results:
            metrics = {}
            # Ajouter les paramètres
            for name, value in params.items():
                metrics[name] = value
                
            # Ajouter les métriques du backtest
            metrics["total_return"] = float(result.total_pnl) if hasattr(result, 'total_pnl') else 0.0
            
            # Extraire les statistiques souhaitées, en gérant les cas où elles sont manquantes
            for metric in target_metrics:
                if metric in ["total_return", "win_rate"]:  # Déjà ajoutés ou calculés différemment
                    if metric == "win_rate" and hasattr(result, 'win_rate'):
                        metrics["win_rate"] = float(result.win_rate)
                    continue
                
                # Chercher dans les statistiques
                if hasattr(result, 'statistics') and metric in result.statistics:
                    try:
                        # Convertir en float pour éviter les problèmes de sérialisation
                        metrics[metric] = float(result.statistics[metric])
                    except (ValueError, TypeError):
                        # Ignorer les conversions qui échouent
                        metrics[metric] = 0.0
                else:
                    # Valeur par défaut si la métrique n'est pas disponible
                    metrics[metric] = 0.0
                    
            metrics_list.append(metrics)
        
        # Créer un DataFrame des métriques de manière efficace
        metrics_df = pd.DataFrame(metrics_list)
        
        # Identifier les paramètres les plus sensibles
        target_metric = self.config["optimization_target"]
        sensitive_params = self._identify_sensitive_parameters(metrics_df, param_names, target_metric)
        
        # Trouver les meilleurs et pires paramètres
        if not metrics_df.empty:
            try:
                # Utiliser des méthodes de sélection robustes
                # Pour les métriques où plus grand est meilleur (ex: return, sharpe)
                if target_metric in ["total_return", "sharpe_ratio", "win_rate", "profit_factor"]:
                    # Filtrer les valeurs non-finies
                    valid_metrics = metrics_df[np.isfinite(metrics_df[target_metric])]
                    if not valid_metrics.empty:
                        best_idx = valid_metrics[target_metric].idxmax()
                        worst_idx = valid_metrics[target_metric].idxmin()
                    else:
                        # Pas de valeurs valides
                        best_idx = worst_idx = 0 if len(metrics_df) > 0 else None
                # Pour les métriques où plus petit est meilleur (ex: drawdown)
                else:
                    # Filtrer les valeurs non-finies
                    valid_metrics = metrics_df[np.isfinite(metrics_df[target_metric])]
                    if not valid_metrics.empty:
                        best_idx = valid_metrics[target_metric].idxmin()
                        worst_idx = valid_metrics[target_metric].idxmax()
                    else:
                        # Pas de valeurs valides
                        best_idx = worst_idx = 0 if len(metrics_df) > 0 else None
                
                if best_idx is not None and worst_idx is not None:
                    best_params = {name: metrics_df.iloc[best_idx][name] for name in param_names}
                    worst_params = {name: metrics_df.iloc[worst_idx][name] for name in param_names}
                else:
                    best_params = worst_params = {}
            except Exception as e:
                logger.error(f"Erreur lors de la recherche des meilleurs/pires paramètres: {str(e)}")
                best_params = worst_params = {}
        else:
            best_params = worst_params = {}
        
        # Créer et retourner le résultat
        return SensitivityResult(
            parameter_ranges=parameter_ranges,
            results=results[:10] if len(results) > 10 else results,  # Limiter pour la mémoire
            metrics=metrics_df,
            best_parameters=best_params,
            worst_parameters=worst_params,
            most_sensitive_parameters=sensitive_params,
            metadata={
                "total_combinations": int(total_combinations),
                "completed_combinations": len(results),
                "errors": error_count,
                "execution_time_seconds": total_time,
                "target_metric": target_metric
            }
        )
    
    def _identify_sensitive_parameters(self, metrics: pd.DataFrame, param_names: List[str], target_metric: str) -> Dict[str, float]:
        """
        Identifie les paramètres les plus sensibles.
        
        Args:
            metrics: DataFrame contenant les métriques du backtest
            param_names: Liste des noms des paramètres
            target_metric: Nom de la métrique cible
            
        Returns:
            Dictionnaire avec les sensibilités par paramètre
        """
        if metrics.empty or target_metric not in metrics.columns:
            return {}
            
        # Utiliser numpy pour des calculs vectorisés plus efficaces
        sensitivities = {}
        
        for param in param_names:
            try:
                # Vérifier si le paramètre a assez de valeurs différentes
                unique_values = metrics[param].unique()
                if len(unique_values) <= 1:
                    sensitivities[param] = 0.0
                    continue
                
                # Convertir en tableaux numpy pour de meilleures performances
                # Utiliser float32 pour économiser de la mémoire tout en maintenant la précision
                param_values = np.array(metrics[param], dtype=np.float32)
                metric_values = np.array(metrics[target_metric], dtype=np.float32)
                
                # Nettoyer les valeurs non finies (NaN, inf)
                mask = np.isfinite(param_values) & np.isfinite(metric_values)
                if not np.any(mask) or np.sum(mask) < 2:
                    sensitivities[param] = 0.0
                    continue
                    
                param_values = param_values[mask]
                metric_values = metric_values[mask]
                
                # Protection contre les valeurs extrêmes et outliers
                # Utiliser des percentiles robustes au lieu de min/max
                param_low = np.percentile(param_values, 1)
                param_high = np.percentile(param_values, 99)
                metric_low = np.percentile(metric_values, 1)
                metric_high = np.percentile(metric_values, 99)
                
                # Écrêter les valeurs extrêmes pour éviter les problèmes numériques
                param_values = np.clip(param_values, param_low, param_high)
                metric_values = np.clip(metric_values, metric_low, metric_high)
                
                # Normalisation robuste avec protection contre la division par zéro
                param_range = np.percentile(param_values, 95) - np.percentile(param_values, 5)
                metric_range = np.percentile(metric_values, 95) - np.percentile(metric_values, 5)
                
                # Éviter les divisions par zéro ou les valeurs trop petites
                epsilon = 1e-8
                if param_range < epsilon or metric_range < epsilon:
                    sensitivities[param] = 0.0
                    continue
                
                # Calculer les valeurs centrales de façon robuste
                param_center = np.median(param_values)
                metric_center = np.median(metric_values)
                
                # Normalisation avec protection contre la division par zéro
                param_norm = (param_values - param_center) / (param_range + epsilon)
                metric_norm = (metric_values - metric_center) / (metric_range + epsilon)
                
                # Calcul de la corrélation de façon plus robuste
                try:
                    # Utiliser np.corrcoef qui est plus stable numériquement
                    corr_matrix = np.corrcoef(param_norm, metric_norm)
                    if corr_matrix.shape == (2, 2):
                        corr = np.abs(corr_matrix[0, 1])
                        if np.isnan(corr):
                            corr = 0.0
                    else:
                        corr = 0.0
                except:
                    # En cas d'erreur, mettre la corrélation à zéro
                    corr = 0.0
                
                # Calculer la variabilité de la métrique pour différentes valeurs du paramètre
                # Utiliser une approche plus robuste aux valeurs extrêmes
                var_sensitivity = 0.0
                try:
                    # Agréger par bins si trop de valeurs uniques pour éviter le surapprentissage
                    if len(np.unique(param_values)) > 20:
                        # Créer des bins pour regrouper les valeurs proches
                        bins = np.linspace(param_values.min(), param_values.max(), 20)
                        bin_indices = np.digitize(param_values, bins)
                        
                        # Calculer les variances par bin
                        variances = []
                        for i in range(1, len(bins) + 1):
                            bin_metrics = metric_values[bin_indices == i]
                            if len(bin_metrics) > 1:
                                # Utiliser la médiane comme mesure de tendance centrale robuste
                                # et l'IQR (écart interquartile) comme mesure de dispersion
                                median_val = np.median(bin_metrics)
                                q75 = np.percentile(bin_metrics, 75)
                                q25 = np.percentile(bin_metrics, 25)
                                if np.abs(median_val) > epsilon:
                                    variances.append((q75 - q25) / (np.abs(median_val) + epsilon))
                                else:
                                    variances.append(q75 - q25)
                    else:
                        # Si peu de valeurs, utiliser les valeurs originales
                        unique_param_values = np.unique(param_values)
                        variances = []
                        
                        for val in unique_param_values:
                            group_metrics = metric_values[np.abs(param_values - val) < epsilon]
                            if len(group_metrics) > 1:
                                median_val = np.median(group_metrics)
                                q75 = np.percentile(group_metrics, 75)
                                q25 = np.percentile(group_metrics, 25)
                                if np.abs(median_val) > epsilon:
                                    variances.append((q75 - q25) / (np.abs(median_val) + epsilon))
                                else:
                                    variances.append(q75 - q25)
                    
                    # Calculer une mesure de sensibilité basée sur la variance
                    if variances:
                        # Nettoyer les valeurs NaN ou infinies
                        clean_variances = np.array([v for v in variances if np.isfinite(v)])
                        if len(clean_variances) > 0:
                            # Limiter l'influence des valeurs extrêmes
                            capped_variances = np.clip(
                                clean_variances, 
                                0, 
                                np.percentile(clean_variances, 95) if len(clean_variances) > 5 else float('inf')
                            )
                            # Inverser car une variance plus faible indique une plus grande sensibilité
                            var_sensitivity = 1.0 - np.mean(capped_variances) / (np.max(capped_variances) + epsilon)
                            # S'assurer que la sensibilité est entre 0 et 1
                            var_sensitivity = max(0.0, min(1.0, var_sensitivity))
                except Exception as e:
                    logger.debug(f"Erreur lors du calcul de la sensibilité par variance pour {param}: {str(e)}")
                    var_sensitivity = 0.0
                
                # Combiner les deux métriques de sensibilité (corrélation et variance)
                # Donner plus de poids à la corrélation
                sensitivity = (0.7 * corr) + (0.3 * var_sensitivity)
                
                # Garantir une valeur entre 0 et 1
                sensitivity = max(0.0, min(1.0, sensitivity))
                sensitivities[param] = sensitivity
                
            except Exception as e:
                logger.warning(f"Erreur lors du calcul de la sensibilité pour {param}: {str(e)}")
                sensitivities[param] = 0.0
                
        return sensitivities 