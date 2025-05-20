"""
Module pour l'analyse des performances des prédictions.
"""

import os
import json
import psutil
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from functools import wraps

def profile(output_dir=None):
    """Décorateur pour profiler une fonction."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Création du profiler
            profiler = PerformanceProfiler()
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Démarrage du profilage
            profiler.start_profiling()
            
            try:
                # Exécution de la fonction
                result = func(*args, **kwargs)
                
                # Enregistrement des métriques
                profiler.record_metrics()
                
                # Sauvegarde des métriques si un répertoire est spécifié
                if output_dir:
                    metrics_file = os.path.join(output_dir, f"{func.__name__}_{int(time.time())}.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(profiler.get_summary(), f)
                
                return result
                
            finally:
                # Enregistrement final des métriques
                profiler.record_metrics()
            
        return wrapper
    return decorator

class PerformanceProfiler:
    """Classe pour profiler les performances d'exécution."""
    
    def __init__(self):
        """Initialise le profileur."""
        self.start_time = None
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory_usage': [],
            'execution_time': []
        }

    def start_profiling(self):
        """Démarre le profilage."""
        self.start_time = time.time()
        self.metrics = {key: [] for key in self.metrics}

    def record_metrics(self):
        """Enregistre les métriques actuelles."""
        if self.start_time is None:
            raise RuntimeError("Le profilage n'a pas été démarré")

        # CPU et mémoire
        process = psutil.Process(os.getpid())
        self.metrics['cpu_usage'].append(process.cpu_percent())
        self.metrics['memory_usage'].append(process.memory_info().rss / 1024 / 1024)  # MB

        # GPU si disponible
        if torch.cuda.is_available():
            self.metrics['gpu_memory_usage'].append(
                torch.cuda.memory_allocated() / 1024 / 1024  # MB
            )
        else:
            self.metrics['gpu_memory_usage'].append(0)

        # Temps d'exécution
        self.metrics['execution_time'].append(time.time() - self.start_time)

    def get_summary(self):
        """Retourne un résumé des métriques."""
        return {
            'total_time': self.metrics['execution_time'][-1],
            'max_memory': max(self.metrics['memory_usage']),
            'max_gpu_memory': max(self.metrics['gpu_memory_usage']),
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage'])
        }

class PerformanceProfile:
    """Classe pour analyser les performances des prédictions."""
    
    def __init__(self):
        """Initialise le profil de performance."""
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        self.metrics = {}

    def add_prediction(self, actual, prediction, timestamp):
        """Ajoute une prédiction et sa valeur réelle."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)

    def calculate_metrics(self):
        """Calcule les métriques de performance."""
        if not self.predictions:
            raise ValueError("Aucune donnée disponible pour calculer les métriques")

        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)

        # Calcul des métriques
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)

        # Précision directionnelle
        pred_dir = np.sign(np.diff(predictions))
        actual_dir = np.sign(np.diff(actuals))
        direction_accuracy = np.mean(pred_dir == actual_dir)

        # Corrélation
        correlation, _ = pearsonr(predictions, actuals)

        self.metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'correlation': correlation
        }

        return self.metrics

    def plot_performance(self):
        """Génère un graphique des performances."""
        if not self.predictions:
            raise ValueError("Aucune donnée disponible pour le graphique")

        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.actuals, label='Réel', color='blue')
        plt.plot(self.timestamps, self.predictions, label='Prédiction', color='red')
        plt.title('Performance des Prédictions')
        plt.xlabel('Temps')
        plt.ylabel('Valeur')
        plt.legend()
        plt.grid(True)
        return plt.gcf()

    def save_metrics(self, filepath):
        """Sauvegarde les métriques dans un fichier JSON."""
        if not self.metrics:
            self.calculate_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f)

    def load_metrics(self, filepath):
        """Charge les métriques depuis un fichier JSON."""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f) 