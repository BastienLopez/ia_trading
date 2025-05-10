"""
Module d'optimisation bayésienne des hyperparamètres.

Ce module implémente :
- Recherche efficace d'hyperparamètres avec l'optimisation bayésienne
- Adaptation dynamique des distributions de recherche
- Prise en compte des incertitudes dans l'exploration
- Optimisation multi-objectifs
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.rl.hyperparameter_optimizer import HyperparameterOptimizer
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment


# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BayesianOptimizer(HyperparameterOptimizer):
    """
    Optimiseur bayésien d'hyperparamètres pour les agents d'apprentissage par renforcement.
    
    Étend HyperparameterOptimizer avec des méthodes d'optimisation bayésienne pour
    une recherche plus efficace des hyperparamètres optimaux.
    """
    
    def __init__(
        self,
        env_creator,
        agent_class,
        param_space,
        n_episodes=50,
        max_steps=None,
        eval_episodes=10,
        metrics=["total_reward", "sharpe_ratio", "max_drawdown", "win_rate"],
        save_dir=None,
        n_jobs=1,
        verbose=1,
        n_initial_points=5,
        n_iterations=15,
        exploration_weight=0.01,
        gp_noise=0.01,
        multi_objective_weights=None,
    ):
        """
        Initialise l'optimiseur bayésien d'hyperparamètres.
        
        Args:
            env_creator: Fonction qui crée et retourne un environnement de trading
            agent_class: Classe de l'agent à optimiser (ex: SACAgent)
            param_space: Dictionnaire avec les hyperparamètres à optimiser et leurs intervalles
                         Format: {"param_name": (min_value, max_value)}
            n_episodes: Nombre d'épisodes pour l'entraînement de chaque configuration
            max_steps: Nombre maximal d'étapes par épisode (None = pas de limite)
            eval_episodes: Nombre d'épisodes pour l'évaluation de chaque configuration
            metrics: Liste des métriques à calculer pour l'évaluation
            save_dir: Répertoire pour sauvegarder les résultats
            n_jobs: Nombre de processus parallèles (1 = séquentiel)
            verbose: Niveau de détail des logs (0 = silencieux, 1 = normal, 2 = détaillé)
            n_initial_points: Nombre de points initiaux à échantillonner aléatoirement
            n_iterations: Nombre d'itérations d'optimisation bayésienne
            exploration_weight: Poids d'exploration dans l'acquisition (plus élevé = plus d'exploration)
            gp_noise: Niveau de bruit pour le processus gaussien
            multi_objective_weights: Dictionnaire des poids pour l'optimisation multi-objectifs
                                    Format: {"metric_name": weight}
        """
        # Convertir param_space en param_grid pour l'initialisation de la classe parente
        param_grid = {}
        for param, bounds in param_space.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # Pour les paramètres numériques, on crée une liste avec une seule valeur (sera ignorée)
                param_grid[param] = [bounds[0]]
            else:
                # Pour les paramètres catégoriels, on utilise directement la liste
                param_grid[param] = bounds
                
        # Initialiser la classe parente
        super().__init__(
            env_creator=env_creator,
            agent_class=agent_class,
            param_grid=param_grid,
            n_episodes=n_episodes,
            max_steps=max_steps,
            eval_episodes=eval_episodes,
            metrics=metrics,
            save_dir=save_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        
        # Paramètres spécifiques à l'optimisation bayésienne
        self.param_space = param_space
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.gp_noise = gp_noise
        
        # Paramètres pour l'optimisation multi-objectifs
        self.multi_objective_weights = multi_objective_weights or {}
        
        # Modèle de processus gaussien
        self.gp_model = None
        
        # Historique des points évalués
        self.X_samples = []  # Points dans l'espace des paramètres
        self.y_samples = []  # Scores correspondants
        self.metrics_history = []  # Historique des métriques
        
        logger.info(f"BayesianOptimizer initialisé avec {len(param_space)} paramètres à optimiser")
        logger.info(f"Stratégie: {n_initial_points} points initiaux + {n_iterations} itérations")
    
    def _param_dict_to_vector(self, param_dict):
        """
        Convertit un dictionnaire de paramètres en vecteur numérique.
        
        Args:
            param_dict: Dictionnaire des paramètres
            
        Returns:
            np.ndarray: Vecteur de paramètres normalisés
        """
        x = []
        for param, bounds in self.param_space.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # Paramètre numérique: normaliser entre 0 et 1
                min_val, max_val = bounds
                val = param_dict[param]
                normalized_val = (val - min_val) / (max_val - min_val)
                x.append(normalized_val)
            else:
                # Paramètre catégoriel: one-hot encoding
                categories = bounds
                val = param_dict[param]
                for category in categories:
                    x.append(1.0 if val == category else 0.0)
        
        return np.array(x).reshape(1, -1)
    
    def _vector_to_param_dict(self, x):
        """
        Convertit un vecteur numérique en dictionnaire de paramètres.
        
        Args:
            x: Vecteur de paramètres normalisés
            
        Returns:
            dict: Dictionnaire des paramètres
        """
        param_dict = {}
        idx = 0
        
        for param, bounds in self.param_space.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # Paramètre numérique: dénormaliser
                min_val, max_val = bounds
                normalized_val = x[idx]
                val = min_val + normalized_val * (max_val - min_val)
                
                # Arrondir à l'entier si nécessaire
                if isinstance(min_val, int) and isinstance(max_val, int):
                    val = int(round(val))
                
                param_dict[param] = val
                idx += 1
            else:
                # Paramètre catégoriel: décodage one-hot
                categories = bounds
                cat_values = x[idx:idx+len(categories)]
                cat_idx = np.argmax(cat_values)
                param_dict[param] = categories[cat_idx]
                idx += len(categories)
        
        return param_dict
    
    def _sample_random_params(self):
        """
        Échantillonne des paramètres aléatoires dans l'espace de recherche.
        
        Returns:
            dict: Dictionnaire des paramètres
        """
        params = {}
        for param, bounds in self.param_space.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # Paramètre numérique
                min_val, max_val = bounds
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Entier
                    val = np.random.randint(min_val, max_val + 1)
                else:
                    # Flottant
                    val = np.random.uniform(min_val, max_val)
                params[param] = val
            else:
                # Paramètre catégoriel
                params[param] = np.random.choice(bounds)
        
        return params
    
    def _expected_improvement(self, x, xi=0.01):
        """
        Calcule l'amélioration espérée pour un point donné.
        
        Args:
            x: Point à évaluer
            xi: Paramètre d'exploration
            
        Returns:
            float: Valeur de l'amélioration espérée
        """
        # Prédiction du modèle
        mu, sigma = self.gp_model.predict(x.reshape(1, -1), return_std=True)
        
        # Meilleur score observé jusqu'à présent
        y_best = np.max(self.y_samples)
        
        # Amélioration par rapport au meilleur score
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma if sigma > 0 else 0
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 0.0] = 0.0
        
        return ei[0]
    
    def _negative_ei(self, x):
        """
        Fonction objectif négative pour la minimisation.
        
        Args:
            x: Point à évaluer
            
        Returns:
            float: Valeur négative de l'amélioration espérée
        """
        return -self._expected_improvement(x, xi=self.exploration_weight)
    
    def _find_next_point(self):
        """
        Trouve le prochain point à évaluer en maximisant l'amélioration espérée.
        
        Returns:
            dict: Dictionnaire des paramètres optimaux
        """
        # Dimension de l'espace de recherche
        dims = 0
        for bounds in self.param_space.values():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                dims += 1
            else:
                dims += len(bounds)  # Catégoriel: one-hot encoding
        
        # Échantillonnage aléatoire de points de départ pour l'optimisation
        x_tries = np.random.uniform(0, 1, size=(100, dims))
        
        # Évaluer l'amélioration espérée pour chaque point
        ei_values = [-self._negative_ei(x) for x in x_tries]
        
        # Sélectionner les meilleurs points comme points de départ pour l'optimisation locale
        x_seeds = x_tries[np.argsort(ei_values)[-5:]]
        
        # Optimisation locale à partir de chaque point de départ
        best_x = None
        best_ei = -np.inf
        
        for x_seed in x_seeds:
            # Optimisation avec contraintes [0, 1] pour chaque dimension
            bounds = [(0, 1) for _ in range(dims)]
            result = minimize(
                self._negative_ei,
                x_seed,
                bounds=bounds,
                method="L-BFGS-B"
            )
            
            if result.success and -result.fun > best_ei:
                best_ei = -result.fun
                best_x = result.x
        
        # Si l'optimisation échoue, utiliser le meilleur point d'échantillonnage
        if best_x is None:
            best_idx = np.argmax(ei_values)
            best_x = x_tries[best_idx]
        
        # Convertir le vecteur en dictionnaire de paramètres
        return self._vector_to_param_dict(best_x)
    
    def _fit_gp_model(self):
        """
        Ajuste le modèle de processus gaussien aux données observées.
        """
        # Convertir les listes en tableaux numpy
        X = np.vstack(self.X_samples)
        y = np.array(self.y_samples).reshape(-1, 1)
        
        # Créer et ajuster le modèle GP avec un noyau Matérn
        kernel = Matern(nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.gp_noise,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.gp_model.fit(X, y)
    
    def bayesian_optimization(self):
        """
        Effectue l'optimisation bayésienne des hyperparamètres.
        
        Returns:
            dict: Meilleurs hyperparamètres trouvés
            float: Score de la meilleure configuration
        """
        logger.info(f"Démarrage de l'optimisation bayésienne avec {self.n_initial_points} points initiaux")
        
        # Phase 1: Échantillonnage initial aléatoire
        for i in range(self.n_initial_points):
            params = self._sample_random_params()
            logger.info(f"Évaluation du point initial {i+1}/{self.n_initial_points}: {params}")
            
            score, _, metrics = self._evaluate_params(params, i, self.n_initial_points + self.n_iterations)
            
            # Stocker les résultats
            self.X_samples.append(self._param_dict_to_vector(params))
            self.y_samples.append(score)
            self.metrics_history.append(metrics)
            
            # Mettre à jour le meilleur résultat
            self._update_best(score, params, metrics)
        
        # Phase 2: Optimisation bayésienne
        for i in range(self.n_iterations):
            # Ajuster le modèle GP aux données observées
            self._fit_gp_model()
            
            # Trouver le prochain point à évaluer
            next_params = self._find_next_point()
            logger.info(f"Itération {i+1}/{self.n_iterations}: Évaluation de {next_params}")
            
            # Évaluer le point
            score, _, metrics = self._evaluate_params(
                next_params, 
                i + self.n_initial_points, 
                self.n_initial_points + self.n_iterations
            )
            
            # Stocker les résultats
            self.X_samples.append(self._param_dict_to_vector(next_params))
            self.y_samples.append(score)
            self.metrics_history.append(metrics)
            
            # Mettre à jour le meilleur résultat
            self._update_best(score, next_params, metrics)
            
            # Afficher la progression
            logger.info(f"Meilleur score jusqu'à présent: {self.best_score} avec {self.best_params}")
        
        # Sauvegarder les résultats finaux
        self._save_results()
        self._generate_convergence_plot()
        
        logger.info(f"Optimisation bayésienne terminée. Meilleur score: {self.best_score}")
        logger.info(f"Meilleurs paramètres: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def _calculate_score(self, metrics):
        """
        Calcule un score global à partir des métriques d'évaluation.
        Surcharge la méthode de la classe parente pour supporter l'optimisation multi-objectifs.
        
        Args:
            metrics: Dictionnaire des métriques
            
        Returns:
            float: Score global
        """
        # Si des poids spécifiques sont fournis pour l'optimisation multi-objectifs
        if self.multi_objective_weights:
            score = 0.0
            for metric, weight in self.multi_objective_weights.items():
                if metric in metrics:
                    # Inverser le signe pour les métriques à minimiser
                    if metric == "max_drawdown":
                        score -= metrics[metric] * weight
                    else:
                        score += metrics[metric] * weight
            return score
        
        # Sinon, utiliser la méthode de la classe parente
        return super()._calculate_score(metrics)
    
    def _generate_convergence_plot(self):
        """
        Génère un graphique de convergence de l'optimisation bayésienne.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracer l'évolution du meilleur score
        best_so_far = np.maximum.accumulate(self.y_samples)
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, 'b-', marker='o', label='Meilleur score')
        ax.plot(range(1, len(self.y_samples) + 1), self.y_samples, 'r--', alpha=0.5, label='Score courant')
        
        # Marquer la transition entre l'échantillonnage initial et l'optimisation bayésienne
        ax.axvline(x=self.n_initial_points + 0.5, color='k', linestyle='--', alpha=0.5)
        ax.text(self.n_initial_points + 0.5, min(self.y_samples), 'Début BO', 
                rotation=90, verticalalignment='bottom')
        
        ax.set_xlabel('Itération')
        ax.set_ylabel('Score')
        ax.set_title('Convergence de l\'optimisation bayésienne')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sauvegarder le graphique
        plot_path = os.path.join(self.save_dir, f"convergence_plot_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Graphique de convergence sauvegardé: {plot_path}")
    
    def _save_results(self):
        """
        Sauvegarde les résultats de l'optimisation bayésienne.
        Surcharge la méthode de la classe parente pour inclure des informations spécifiques.
        """
        # Appeler la méthode de la classe parente
        super()._save_results()
        
        # Sauvegarder des informations supplémentaires spécifiques à l'optimisation bayésienne
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convertir l'historique en DataFrame
        results_data = []
        for i, (x_vec, score, metrics) in enumerate(zip(self.X_samples, self.y_samples, self.metrics_history)):
            params = self._vector_to_param_dict(x_vec[0])
            entry = {
                "iteration": i + 1,
                "score": score,
                **params,
                **metrics
            }
            results_data.append(entry)
        
        df = pd.DataFrame(results_data)
        
        # Sauvegarder en CSV
        csv_path = os.path.join(self.save_dir, f"bayesian_optimization_history_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Sauvegarder en JSON avec des métadonnées supplémentaires
        bo_results = {
            "timestamp": timestamp,
            "best_params": self.best_params,
            "best_score": float(self.best_score),
            "n_initial_points": self.n_initial_points,
            "n_iterations": self.n_iterations,
            "exploration_weight": self.exploration_weight,
            "gp_noise": self.gp_noise,
            "multi_objective_weights": self.multi_objective_weights,
            "history": df.to_dict(orient="records")
        }
        
        json_path = os.path.join(self.save_dir, f"bayesian_optimization_results_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(bo_results, f, indent=2)
        
        logger.info(f"Historique d'optimisation bayésienne sauvegardé: {csv_path}")
        logger.info(f"Résultats d'optimisation bayésienne sauvegardés: {json_path}")


def optimize_sac_agent_bayesian(
    train_data,
    param_space=None,
    n_episodes=50,
    eval_episodes=10,
    save_dir=None,
    n_jobs=1,
    n_initial_points=5,
    n_iterations=15,
    exploration_weight=0.01,
    multi_objective_weights=None,
):
    """
    Optimise les hyperparamètres d'un agent SAC avec l'optimisation bayésienne.
    
    Args:
        train_data: DataFrame des données d'entraînement
        param_space: Dictionnaire des espaces de paramètres à optimiser
        n_episodes: Nombre d'épisodes pour l'entraînement
        eval_episodes: Nombre d'épisodes pour l'évaluation
        save_dir: Répertoire pour sauvegarder les résultats
        n_jobs: Nombre de processus parallèles
        n_initial_points: Nombre de points initiaux à échantillonner aléatoirement
        n_iterations: Nombre d'itérations d'optimisation bayésienne
        exploration_weight: Poids d'exploration dans l'acquisition
        multi_objective_weights: Dictionnaire des poids pour l'optimisation multi-objectifs
        
    Returns:
        dict: Meilleurs hyperparamètres trouvés
    """
    # Espace de paramètres par défaut si non spécifié
    if param_space is None:
        param_space = {
            "actor_learning_rate": (1e-5, 1e-3),
            "critic_learning_rate": (1e-5, 1e-3),
            "batch_size": (32, 256),
            "hidden_size": (64, 512),
            "entropy_regularization": (0.0, 0.2),
            "grad_clip_value": (0.5, 5.0),
        }
    
    # Répertoire de sauvegarde par défaut
    if save_dir is None:
        save_dir = INFO_RETOUR_DIR / "bayesian_opt" / "sac"
        os.makedirs(save_dir, exist_ok=True)
    
    # Fonction pour créer l'environnement
    def create_env():
        return TradingEnvironment(
            df=train_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            reward_function="simple",
            action_type="continuous",
        )
    
    # Créer et exécuter l'optimiseur bayésien
    optimizer = BayesianOptimizer(
        env_creator=create_env,
        agent_class=SACAgent,
        param_space=param_space,
        n_episodes=n_episodes,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        n_jobs=n_jobs,
        n_initial_points=n_initial_points,
        n_iterations=n_iterations,
        exploration_weight=exploration_weight,
        multi_objective_weights=multi_objective_weights,
    )
    
    best_params, _ = optimizer.bayesian_optimization()
    return best_params


def optimize_gru_sac_agent_bayesian(
    train_data,
    param_space=None,
    n_episodes=50,
    eval_episodes=10,
    save_dir=None,
    n_jobs=1,
    n_initial_points=5,
    n_iterations=15,
    exploration_weight=0.01,
    multi_objective_weights=None,
):
    """
    Optimise les hyperparamètres d'un agent SAC avec GRU en utilisant l'optimisation bayésienne.
    
    Args:
        train_data: DataFrame des données d'entraînement
        param_space: Dictionnaire des espaces de paramètres à optimiser
        n_episodes: Nombre d'épisodes pour l'entraînement
        eval_episodes: Nombre d'épisodes pour l'évaluation
        save_dir: Répertoire pour sauvegarder les résultats
        n_jobs: Nombre de processus parallèles
        n_initial_points: Nombre de points initiaux à échantillonner aléatoirement
        n_iterations: Nombre d'itérations d'optimisation bayésienne
        exploration_weight: Poids d'exploration dans l'acquisition
        multi_objective_weights: Dictionnaire des poids pour l'optimisation multi-objectifs
        
    Returns:
        dict: Meilleurs hyperparamètres trouvés
    """
    # Espace de paramètres par défaut si non spécifié
    if param_space is None:
        param_space = {
            "actor_learning_rate": (1e-5, 1e-3),
            "critic_learning_rate": (1e-5, 1e-3),
            "batch_size": (32, 256),
            "hidden_size": (64, 512),
            "entropy_regularization": (0.0, 0.2),
            "grad_clip_value": (0.5, 5.0),
            "use_gru": [True],  # Paramètre fixe
            "sequence_length": (3, 20),
            "gru_units": (16, 256),
        }
    
    # Répertoire de sauvegarde par défaut
    if save_dir is None:
        save_dir = INFO_RETOUR_DIR / "bayesian_opt" / "gru_sac"
        os.makedirs(save_dir, exist_ok=True)
    
    # Fonction pour créer l'environnement
    def create_env():
        return TradingEnvironment(
            df=train_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            reward_function="simple",
            action_type="continuous",
        )
    
    # Créer et exécuter l'optimiseur bayésien
    optimizer = BayesianOptimizer(
        env_creator=create_env,
        agent_class=SACAgent,  # SACAgent avec use_gru=True
        param_space=param_space,
        n_episodes=n_episodes,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        n_jobs=n_jobs,
        n_initial_points=n_initial_points,
        n_iterations=n_iterations,
        exploration_weight=exploration_weight,
        multi_objective_weights=multi_objective_weights,
    )
    
    best_params, _ = optimizer.bayesian_optimization()
    return best_params 