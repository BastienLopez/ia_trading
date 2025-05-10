import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import Optimizer as SkOptimizer
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

from ai_trading.rl.hyperparameter_optimizer import HyperparameterOptimizer
from ai_trading.config import INFO_RETOUR_DIR

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
    
    Utilise l'optimisation bayésienne pour efficacement explorer l'espace des hyperparamètres
    en construisant un modèle probabiliste (processus gaussien) qui prédit les performances
    en fonction des hyperparamètres.
    """
    
    def __init__(
        self,
        env_creator: Callable,
        agent_class,
        param_space: Dict,
        n_episodes: int = 50,
        max_steps: Optional[int] = None,
        eval_episodes: int = 10,
        metrics: List[str] = ["total_reward", "sharpe_ratio", "max_drawdown", "win_rate"],
        save_dir = None,
        n_jobs: int = 1,
        verbose: int = 1,
        n_initial_points: int = 10,
        n_calls: int = 50,
        acq_func: str = "EI",
        random_state: int = 42
    ):
        """
        Initialise l'optimiseur bayésien d'hyperparamètres.
        
        Args:
            env_creator: Fonction qui crée et retourne un environnement de trading
            agent_class: Classe de l'agent à optimiser (ex: SACAgent)
            param_space: Dictionnaire avec les hyperparamètres à optimiser et leurs espaces
            n_episodes: Nombre d'épisodes pour l'entraînement de chaque configuration
            max_steps: Nombre maximal d'étapes par épisode (None = pas de limite)
            eval_episodes: Nombre d'épisodes pour l'évaluation de chaque configuration
            metrics: Liste des métriques à calculer pour l'évaluation
            save_dir: Répertoire pour sauvegarder les résultats
            n_jobs: Nombre de processus parallèles (1 = séquentiel)
            verbose: Niveau de détail des logs (0 = silencieux, 1 = normal, 2 = détaillé)
            n_initial_points: Nombre de points initiaux pour l'exploration aléatoire
            n_calls: Nombre total d'appels à la fonction objectif
            acq_func: Fonction d'acquisition ('EI', 'LCB', 'PI', 'gp_hedge')
            random_state: Graine aléatoire pour la reproductibilité
        """
        # Ne pas appeler le constructeur parent, car param_space est différent de param_grid
        # Mais initialiser les attributs communs
        self.env_creator = env_creator
        self.agent_class = agent_class
        self.param_space = param_space
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.eval_episodes = eval_episodes
        self.metrics = metrics
        
        # Utiliser INFO_RETOUR_DIR/bayesian_opt par défaut si save_dir n'est pas spécifié
        if save_dir is None:
            self.save_dir = INFO_RETOUR_DIR / "bayesian_opt"
        else:
            self.save_dir = save_dir
            
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Paramètres spécifiques à l'optimisation bayésienne
        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.acq_func = acq_func
        self.random_state = random_state
        
        # Créer le répertoire de sauvegarde s'il n'existe pas
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Historique des résultats
        self.results = []
        self.best_params = None
        self.best_score = float("-inf")
        
        # Convertir le dictionnaire param_space en liste de dimensions pour skopt
        self.dimensions = []
        self.param_names = []
        
        for param_name, space_def in self.param_space.items():
            self.param_names.append(param_name)
            
            if isinstance(space_def, tuple) and len(space_def) == 3:
                # Format (low, high, type)
                low, high, param_type = space_def
                
                if param_type == 'real':
                    self.dimensions.append(Real(low, high, name=param_name))
                elif param_type == 'int':
                    self.dimensions.append(Integer(low, high, name=param_name))
            elif isinstance(space_def, list):
                # Liste de choix catégoriels
                self.dimensions.append(Categorical(space_def, name=param_name))
            else:
                raise ValueError(f"Format de param_space invalide pour {param_name}")
        
        logger.info(f"BayesianOptimizer initialisé avec {len(self.dimensions)} dimensions")
    
    def bayesian_search(self) -> Tuple[Dict, float]:
        """
        Effectue une recherche bayésienne sur les hyperparamètres.
        
        Returns:
            tuple: (meilleurs hyperparamètres, meilleur score)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Démarrage de l'optimisation bayésienne avec {self.n_calls} appels")
        
        # Créer l'optimiseur bayésien
        optimizer = SkOptimizer(
            dimensions=self.dimensions,
            base_estimator="GP",  # Processus gaussien
            acq_func=self.acq_func,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state
        )
        
        # Définir la fonction objectif pour skopt
        @use_named_args(self.dimensions)
        def objective_function(**params):
            # Convertir les valeurs des hyperparamètres au format approprié
            processed_params = {}
            for param_name, value in params.items():
                # S'assurer que les entiers sont bien des entiers
                for dim in self.dimensions:
                    if dim.name == param_name and isinstance(dim, Integer):
                        value = int(value)
                processed_params[param_name] = value
            
            if self.verbose > 0:
                logger.info(f"Évaluation des paramètres: {processed_params}")
            
            # Créer et évaluer l'agent avec ces paramètres
            score, metrics = self._evaluate_single_config(processed_params)
            
            # Mettre à jour le meilleur résultat
            self._update_best(score, processed_params, metrics)
            
            # Stocker les résultats
            self.results.append({
                "params": processed_params,
                "score": score,
                "metrics": metrics
            })
            
            # Retourner le score négatif car skopt minimise par défaut
            return -score
        
        # Lancer l'optimisation
        result = optimizer.run(objective_function, n_iter=self.n_calls)
        
        # Sauvegarder les résultats
        self._save_results(optimizer, timestamp)
        
        logger.info(f"Optimisation bayésienne terminée. Meilleur score: {self.best_score}")
        logger.info(f"Meilleurs paramètres: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def _evaluate_single_config(self, params):
        """
        Évalue une configuration d'hyperparamètres.
        
        Args:
            params: Dictionnaire des hyperparamètres à évaluer
            
        Returns:
            tuple: (score global, dictionnaire des métriques)
        """
        # Créer l'environnement
        env = self.env_creator()
        
        # Extraire les paramètres pour l'agent
        agent_params = params.copy()
        
        # Créer l'agent avec les paramètres
        agent = self.agent_class(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
            **agent_params
        )
        
        # Entraîner l'agent
        self._train_agent(env, agent, params, self.n_episodes, self.max_steps)
        
        # Évaluer l'agent
        metrics = self._evaluate_agent(env, agent, self.eval_episodes)
        
        # Calculer le score global
        score = self._calculate_score(metrics)
        
        return score, metrics
    
    def _save_results(self, optimizer, timestamp):
        """
        Sauvegarde les résultats de l'optimisation.
        
        Args:
            optimizer: L'optimiseur skopt
            timestamp: Horodatage pour les noms de fichiers
        """
        # Sauvegarder les résultats bruts
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.save_dir, f"bayesian_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Sauvegarder les meilleurs paramètres
        best_path = os.path.join(self.save_dir, f"bayesian_best_params_{timestamp}.csv")
        pd.DataFrame([self.best_params]).to_csv(best_path, index=False)
        
        # Générer et sauvegarder les visualisations
        self._generate_plots(optimizer, timestamp)
        
        logger.info(f"Résultats sauvegardés dans {self.save_dir}")
    
    def _generate_plots(self, optimizer, timestamp):
        """
        Génère des visualisations de l'optimisation bayésienne.
        
        Args:
            optimizer: L'optimiseur skopt
            timestamp: Horodatage pour les noms de fichiers
        """
        try:
            # Tracé de convergence
            plt.figure(figsize=(10, 6))
            plot_convergence(optimizer.Xi, optimizer.yi)
            plt.title("Convergence de l'optimisation bayésienne")
            convergence_path = os.path.join(self.save_dir, f"bayesian_convergence_{timestamp}.png")
            plt.savefig(convergence_path)
            plt.close()
            
            # Tracé des objectifs pour chaque dimension
            for i, dim_name in enumerate(self.param_names):
                try:
                    plt.figure(figsize=(10, 6))
                    plot_objective(optimizer.Xi, optimizer.yi, dimensions=self.dimensions, 
                                   n_points=100, n_samples=150, dimension_names=self.param_names,
                                   plot_dims=[i])
                    plt.title(f"Impact de {dim_name} sur la performance")
                    objective_path = os.path.join(self.save_dir, f"bayesian_objective_{dim_name}_{timestamp}.png")
                    plt.savefig(objective_path)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Erreur lors de la génération du tracé pour {dim_name}: {e}")
            
            # Tracé des performances au fil des itérations
            plt.figure(figsize=(10, 6))
            iterations = range(1, len(self.results) + 1)
            scores = [-result["score"] for result in self.results]  # Inverser car on minimise
            plt.plot(iterations, scores, marker='o', linestyle='-')
            plt.axhline(y=-self.best_score, color='r', linestyle='--', label=f'Meilleur score: {self.best_score:.4f}')
            plt.xlabel("Itération")
            plt.ylabel("Score")
            plt.title("Progression de l'optimisation bayésienne")
            plt.legend()
            plt.grid(True)
            progress_path = os.path.join(self.save_dir, f"bayesian_progress_{timestamp}.png")
            plt.savefig(progress_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations: {e}")


def optimize_agent_bayesian(
    train_data,
    agent_class,
    param_space=None,
    n_episodes=50,
    eval_episodes=10,
    n_calls=30,
    save_dir=None,
    n_jobs=1,
):
    """
    Fonction pratique pour optimiser un agent avec l'optimisation bayésienne.
    
    Args:
        train_data: Données d'entraînement
        agent_class: Classe de l'agent à optimiser
        param_space: Espace des hyperparamètres à optimiser (par défaut, utilise un espace prédéfini)
        n_episodes: Nombre d'épisodes d'entraînement
        eval_episodes: Nombre d'épisodes d'évaluation
        n_calls: Nombre d'appels à la fonction objectif
        save_dir: Répertoire de sauvegarde
        n_jobs: Nombre de processus parallèles
        
    Returns:
        dict: Meilleurs hyperparamètres trouvés
    """
    # Définir un espace de paramètres par défaut si non spécifié
    if param_space is None:
        param_space = {
            "learning_rate": (1e-5, 1e-2, "real"),
            "gamma": (0.9, 0.999, "real"),
            "tau": (0.001, 0.1, "real"),
            "batch_size": (32, 256, "int"),
            "buffer_size": (10000, 1000000, "int"),
            "hidden_size": (64, 512, "int"),
            "alpha": (0.1, 5.0, "real"),
        }
    
    # Fonction de création de l'environnement
    def create_env():
        from ai_trading.rl.trading_environment import TradingEnvironment
        env = TradingEnvironment(
            df=train_data,
            window_size=50,
            initial_balance=10000,
            commission=0.001,
            use_risk_manager=True,
        )
        return env
    
    # Créer l'optimiseur bayésien
    optimizer = BayesianOptimizer(
        env_creator=create_env,
        agent_class=agent_class,
        param_space=param_space,
        n_episodes=n_episodes,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        n_jobs=n_jobs,
        n_calls=n_calls,
    )
    
    # Lancer l'optimisation
    best_params, _ = optimizer.bayesian_search()
    
    return best_params 