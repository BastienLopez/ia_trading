import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class TemporalCrossValidator:
    """
    Validation croisée temporelle spécialement conçue pour les séries temporelles financières.
    Contrairement à la validation croisée traditionnelle, cette approche respecte l'ordre chronologique
    des données pour éviter le look-ahead bias.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        initial_train_size: float = 0.5,
        gap: int = 0,
        stride: Optional[int] = None
    ):
        """
        Initialise le validateur croisé temporel.
        
        Args:
            data: DataFrame contenant les données temporelles
            n_splits: Nombre de divisions à créer
            initial_train_size: Proportion des données à utiliser pour l'ensemble d'entraînement initial
            gap: Nombre de périodes à exclure entre l'entraînement et la validation
            stride: Pas entre chaque division (par défaut: taille de l'ensemble de validation)
        """
        self.data = data
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.gap = gap
        
        total_size = len(data)
        self.initial_train_end = int(total_size * initial_train_size)
        
        # Calculer la taille de chaque ensemble de validation
        val_size = (total_size - self.initial_train_end) // n_splits
        if val_size <= 0:
            raise ValueError("Pas assez de données pour le nombre de divisions demandé")
        
        # Si stride n'est pas spécifié, utiliser la taille de l'ensemble de validation
        self.stride = stride if stride is not None else val_size
    
    def split(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Génère les divisions temporelles des données.
        
        Returns:
            Liste de tuples (ensemble d'entraînement, ensemble de validation)
        """
        splits = []
        total_size = len(self.data)
        
        val_size = (total_size - self.initial_train_end) // self.n_splits
        
        for i in range(self.n_splits):
            # Calculer les indices pour cette division
            train_end = self.initial_train_end + i * self.stride
            val_start = train_end + self.gap
            val_end = min(val_start + val_size, total_size)
            
            # S'assurer que nous ne dépassons pas la taille des données
            if val_start >= total_size or val_end > total_size:
                break
                
            # Extraire les ensembles d'entraînement et de validation
            train_data = self.data.iloc[:train_end]
            val_data = self.data.iloc[val_start:val_end]
            
            splits.append((train_data, val_data))
            
        return splits
    
    def evaluate(
        self, 
        model_fn: Callable[[pd.DataFrame], Any], 
        eval_fn: Callable[[Any, pd.DataFrame], float]
    ) -> Dict[str, List[float]]:
        """
        Évalue un modèle en utilisant la validation croisée temporelle.
        
        Args:
            model_fn: Fonction qui prend des données d'entraînement et renvoie un modèle entraîné
            eval_fn: Fonction qui prend un modèle et des données de validation et renvoie une métrique
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation pour chaque division
        """
        results = {"train_metrics": [], "val_metrics": [], "fold_sizes": []}
        
        for i, (train_data, val_data) in enumerate(self.split()):
            logger.info(f"Division {i+1}/{self.n_splits} - Entraînement: {len(train_data)} exemples, "
                       f"Validation: {len(val_data)} exemples")
            
            # Entraîner le modèle
            model = model_fn(train_data)
            
            # Évaluer sur l'ensemble d'entraînement
            train_metric = eval_fn(model, train_data)
            results["train_metrics"].append(train_metric)
            
            # Évaluer sur l'ensemble de validation
            val_metric = eval_fn(model, val_data)
            results["val_metrics"].append(val_metric)
            
            # Stocker les tailles des ensembles
            results["fold_sizes"].append((len(train_data), len(val_data)))
            
        return results

class WalkForwardOptimizer:
    """
    Optimisation des hyperparamètres basée sur la validation croisée temporelle.
    Utilise une approche de marche en avant pour tester différentes combinaisons
    d'hyperparamètres sur des fenêtres temporelles successives.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        n_splits: int = 5,
        initial_train_size: float = 0.5,
        gap: int = 0,
        evaluation_metric: str = "profit"
    ):
        """
        Initialise l'optimiseur Walk-Forward.
        
        Args:
            data: DataFrame contenant les données temporelles
            param_grid: Dictionnaire des hyperparamètres à optimiser
            n_splits: Nombre de divisions à créer
            initial_train_size: Proportion des données à utiliser pour l'ensemble d'entraînement initial
            gap: Nombre de périodes à exclure entre l'entraînement et la validation
            evaluation_metric: Métrique à optimiser ('profit', 'sharpe', etc.)
        """
        self.data = data
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.gap = gap
        self.evaluation_metric = evaluation_metric
        
        # Générer toutes les combinaisons d'hyperparamètres
        self.param_combinations = self._generate_param_combinations()
        
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        Génère toutes les combinaisons possibles d'hyperparamètres.
        
        Returns:
            Liste de dictionnaires d'hyperparamètres
        """
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        param_combinations = []
        self._recursive_grid_search(param_keys, param_values, 0, {}, param_combinations)
        
        return param_combinations
    
    def _recursive_grid_search(
        self, 
        param_keys: List[str], 
        param_values: List[List[Any]], 
        idx: int, 
        current_combo: Dict[str, Any], 
        result: List[Dict[str, Any]]
    ):
        """
        Fonction récursive pour générer toutes les combinaisons d'hyperparamètres.
        
        Args:
            param_keys: Liste des noms d'hyperparamètres
            param_values: Liste des valeurs possibles pour chaque hyperparamètre
            idx: Index actuel dans la liste des hyperparamètres
            current_combo: Dictionnaire de la combinaison actuelle
            result: Liste pour stocker toutes les combinaisons
        """
        if idx == len(param_keys):
            result.append(current_combo.copy())
            return
        
        key = param_keys[idx]
        for value in param_values[idx]:
            current_combo[key] = value
            self._recursive_grid_search(param_keys, param_values, idx + 1, current_combo, result)
    
    def optimize(
        self,
        model_fn: Callable[[pd.DataFrame, Dict[str, Any]], Any],
        eval_fn: Callable[[Any, pd.DataFrame], float]
    ) -> Dict[str, Any]:
        """
        Optimise les hyperparamètres en utilisant la validation croisée temporelle.
        
        Args:
            model_fn: Fonction qui prend des données d'entraînement et des hyperparamètres et renvoie un modèle
            eval_fn: Fonction qui prend un modèle et des données de validation et renvoie une métrique
            
        Returns:
            Dictionnaire contenant les meilleurs hyperparamètres et les résultats
        """
        best_params = None
        best_metric = float('-inf') if self.evaluation_metric != 'error' else float('inf')
        all_results = []
        
        # Pour chaque combinaison d'hyperparamètres
        for params in self.param_combinations:
            logger.info(f"Évaluation des paramètres: {params}")
            
            # Créer un validateur temporel
            cv = TemporalCrossValidator(
                data=self.data,
                n_splits=self.n_splits,
                initial_train_size=self.initial_train_size,
                gap=self.gap
            )
            
            # Définir une fonction de modèle partielle avec les hyperparamètres actuels
            def model_with_params(train_data):
                return model_fn(train_data, params)
            
            # Évaluer avec la validation croisée temporelle
            results = cv.evaluate(model_with_params, eval_fn)
            
            # Calculer la métrique moyenne sur les divisions
            avg_val_metric = np.mean(results["val_metrics"])
            
            # Enregistrer les résultats
            param_results = {
                "params": params,
                "avg_val_metric": avg_val_metric,
                "train_metrics": results["train_metrics"],
                "val_metrics": results["val_metrics"]
            }
            all_results.append(param_results)
            
            # Mettre à jour les meilleurs paramètres si nécessaire
            is_better = False
            if self.evaluation_metric == 'error':
                is_better = avg_val_metric < best_metric
            else:
                is_better = avg_val_metric > best_metric
                
            if is_better:
                best_metric = avg_val_metric
                best_params = params
        
        # Trier les résultats
        if self.evaluation_metric == 'error':
            all_results.sort(key=lambda x: x["avg_val_metric"])
        else:
            all_results.sort(key=lambda x: x["avg_val_metric"], reverse=True)
        
        return {
            "best_params": best_params,
            "best_metric": best_metric,
            "all_results": all_results
        } 