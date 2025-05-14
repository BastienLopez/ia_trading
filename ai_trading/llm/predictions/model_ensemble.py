"""
Module d'ensemble de modèles pour combiner plusieurs prédicteurs
et améliorer la robustesse des prédictions.

Ce module permet de combiner les résultats de plusieurs modèles 
avec différentes stratégies de fusion (vote majoritaire, moyenne pondérée, 
ou basée sur la confiance) et d'enrichir les prédictions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Union, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from collections import Counter

# Configuration du logger
from ai_trading.utils import setup_logger
logger = setup_logger("model_ensemble")

class ModelEnsemble:
    """
    Classe pour combiner plusieurs modèles de prédiction et fusionner leurs résultats.
    
    Cette classe permet d'améliorer la robustesse des prédictions en:
    - Combinant les résultats de plusieurs modèles
    - Utilisant différentes stratégies de fusion (majorité, pondérée, basée sur la confiance)
    - Enrichissant les informations des prédictions
    - Ajustant dynamiquement les poids des modèles selon les performances
    
    Attributs:
        models (List[Dict]): Liste de dictionnaires, chacun contenant un modèle et ses métadonnées
        fusion_strategy (str): Stratégie de fusion des prédictions ('majority', 'weighted', 'confidence')
        adjust_weights (bool): Indique si les poids doivent être ajustés automatiquement
        min_consensus_ratio (float): Ratio minimum de consensus pour valider une prédiction
    """
    
    def __init__(self, 
                fusion_strategy: str = 'weighted', 
                adjust_weights: bool = True,
                min_consensus_ratio: float = 0.6):
        """
        Initialise un ensemble de modèles avec une stratégie de fusion.
        
        Args:
            fusion_strategy (str): Stratégie de fusion ('majority', 'weighted', 'confidence')
            adjust_weights (bool): Ajustement automatique des poids basé sur les performances
            min_consensus_ratio (float): Ratio minimum de consensus pour valider une prédiction
        """
        self.models = []
        self.fusion_strategy = fusion_strategy
        self.adjust_weights = adjust_weights
        self.min_consensus_ratio = min_consensus_ratio
        
        # Validation des paramètres
        valid_strategies = ['majority', 'weighted', 'confidence']
        if fusion_strategy not in valid_strategies:
            raise ValueError(f"Stratégie de fusion non valide. Choisissez parmi: {valid_strategies}")
        
        if not 0 <= min_consensus_ratio <= 1:
            raise ValueError("Le ratio de consensus minimum doit être entre 0 et 1")
        
        logger.info(f"Ensemble de modèles initialisé avec stratégie '{fusion_strategy}'")
    
    def add_model(self, 
                 model: Any, 
                 name: str, 
                 weight: float = 1.0, 
                 predict_method: str = 'predict',
                 confidence_method: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Ajoute un modèle à l'ensemble.
        
        Args:
            model (Any): Instance du modèle
            name (str): Nom unique du modèle
            weight (float): Poids initial du modèle dans l'ensemble
            predict_method (str): Nom de la méthode de prédiction à appeler
            confidence_method (Optional[str]): Nom de la méthode retournant la confiance de prédiction
            metadata (Optional[Dict]): Métadonnées supplémentaires du modèle
        """
        if any(m['name'] == name for m in self.models):
            raise ValueError(f"Un modèle avec le nom '{name}' existe déjà dans l'ensemble")
        
        if weight <= 0:
            raise ValueError("Le poids du modèle doit être positif")
        
        model_info = {
            'model': model,
            'name': name,
            'weight': weight,
            'predict_method': predict_method,
            'confidence_method': confidence_method,
            'metadata': metadata or {},
            'performance_metrics': {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'confidence_calibration': None
            }
        }
        
        self.models.append(model_info)
        logger.info(f"Modèle '{name}' ajouté à l'ensemble avec poids {weight}")
        
    def remove_model(self, name: str) -> bool:
        """
        Supprime un modèle de l'ensemble.
        
        Args:
            name (str): Nom du modèle à supprimer
            
        Returns:
            bool: True si le modèle a été supprimé, False sinon
        """
        initial_count = len(self.models)
        self.models = [m for m in self.models if m['name'] != name]
        
        if len(self.models) < initial_count:
            logger.info(f"Modèle '{name}' supprimé de l'ensemble")
            return True
        else:
            logger.warning(f"Modèle '{name}' non trouvé dans l'ensemble")
            return False
    
    def _get_model_prediction(self, model_info: Dict[str, Any], X: Any) -> Tuple[Any, Optional[float]]:
        """
        Obtient la prédiction et la confiance d'un modèle.
        
        Args:
            model_info (Dict): Informations sur le modèle
            X (Any): Données d'entrée pour la prédiction
            
        Returns:
            Tuple: (prédiction, confiance)
        """
        model = model_info['model']
        predict_method = getattr(model, model_info['predict_method'])
        
        prediction = predict_method(X)
        
        confidence = None
        if model_info['confidence_method']:
            try:
                confidence_method = getattr(model, model_info['confidence_method'])
                confidence = confidence_method(X)
            except (AttributeError, Exception) as e:
                logger.warning(f"Erreur lors de l'obtention de la confiance pour '{model_info['name']}': {e}")
        
        return prediction, confidence
    
    def predict(self, X: Any) -> Dict[str, Any]:
        """
        Effectue une prédiction en combinant les résultats de tous les modèles.
        
        Args:
            X (Any): Données d'entrée pour la prédiction
            
        Returns:
            Dict: Résultat fusionné avec métadonnées
        """
        if not self.models:
            raise ValueError("Aucun modèle n'a été ajouté à l'ensemble")
        
        # Collecter les prédictions de chaque modèle
        predictions = []
        confidences = []
        weights = []
        
        for model_info in self.models:
            prediction, confidence = self._get_model_prediction(model_info, X)
            predictions.append(prediction)
            confidences.append(confidence if confidence is not None else 1.0)
            weights.append(model_info['weight'])
        
        # Fusionner les prédictions selon la stratégie choisie
        result = self._fusion_predictions(predictions, confidences, weights)
        
        return result
    
    def _fusion_predictions(self, 
                           predictions: List[Any], 
                           confidences: List[float], 
                           weights: List[float]) -> Dict[str, Any]:
        """
        Fusionne les prédictions selon la stratégie choisie.
        
        Args:
            predictions (List): Liste des prédictions
            confidences (List): Liste des confiances associées
            weights (List): Liste des poids des modèles
            
        Returns:
            Dict: Résultat fusionné avec métadonnées
        """
        # Convertir les éventuels tableaux numpy en valeurs Python standard
        processed_predictions = []
        for pred in predictions:
            # Cas des tableaux numpy
            if hasattr(pred, 'shape') and hasattr(pred, 'dtype'):
                # Si c'est un tableau avec un seul élément, extraire la valeur
                if np.size(pred) == 1:
                    processed_predictions.append(pred.item())
                else:
                    # Sinon, conserver le tableau
                    processed_predictions.append(pred)
            else:
                # Valeur non-numpy, la conserver telle quelle
                processed_predictions.append(pred)
                
        # Cas spécial pour les prédictions catégorielles (classification)
        first_pred = processed_predictions[0]
        if isinstance(first_pred, (str, int, bool)) or (hasattr(first_pred, '__len__') and len(first_pred) == 1):
            return self._fusion_categorical(processed_predictions, confidences, weights)
        
        # Cas des prédictions numériques (régression)
        elif isinstance(first_pred, (float, int, np.float32, np.float64)) or (
                hasattr(first_pred, 'dtype') and np.issubdtype(first_pred.dtype, np.number)):
            return self._fusion_numerical(processed_predictions, confidences, weights)
        
        # Cas des prédictions vectorielles ou autres
        else:
            return self._fusion_complex(processed_predictions, confidences, weights)
    
    def _fusion_categorical(self, 
                          predictions: List[Any], 
                          confidences: List[float], 
                          weights: List[float]) -> Dict[str, Any]:
        """
        Fusionne des prédictions catégorielles.
        
        Args:
            predictions (List): Liste des prédictions catégorielles
            confidences (List): Liste des confiances associées
            weights (List): Liste des poids des modèles
            
        Returns:
            Dict: Résultat fusionné avec métadonnées
        """
        # Stratégie de vote majoritaire
        if self.fusion_strategy == 'majority':
            counter = Counter(predictions)
            final_prediction = counter.most_common(1)[0][0]
            votes = dict(counter)
            
            # Calcul du ratio de consensus
            max_votes = counter.most_common(1)[0][1]
            consensus_ratio = max_votes / len(predictions)
            
            # Déterminer si le consensus est suffisant
            is_consensus_sufficient = consensus_ratio >= self.min_consensus_ratio
            
            return {
                'prediction': final_prediction,
                'consensus_ratio': consensus_ratio,
                'is_consensus_sufficient': is_consensus_sufficient,
                'votes': votes,
                'raw_predictions': predictions,
                'fusion_strategy': 'majority'
            }
        
        # Stratégie pondérée
        elif self.fusion_strategy == 'weighted':
            weighted_votes = {}
            
            for pred, weight in zip(predictions, weights):
                if pred in weighted_votes:
                    weighted_votes[pred] += weight
                else:
                    weighted_votes[pred] = weight
            
            final_prediction = max(weighted_votes, key=weighted_votes.get)
            total_weight = sum(weights)
            consensus_ratio = weighted_votes[final_prediction] / total_weight
            
            return {
                'prediction': final_prediction,
                'consensus_ratio': consensus_ratio,
                'is_consensus_sufficient': consensus_ratio >= self.min_consensus_ratio,
                'weighted_votes': weighted_votes,
                'raw_predictions': predictions,
                'fusion_strategy': 'weighted'
            }
        
        # Stratégie basée sur la confiance
        elif self.fusion_strategy == 'confidence':
            confidence_votes = {}
            
            for pred, conf, weight in zip(predictions, confidences, weights):
                effective_weight = conf * weight
                if pred in confidence_votes:
                    confidence_votes[pred] += effective_weight
                else:
                    confidence_votes[pred] = effective_weight
            
            final_prediction = max(confidence_votes, key=confidence_votes.get)
            total_confidence_weight = sum(conf * weight for conf, weight in zip(confidences, weights))
            consensus_ratio = confidence_votes[final_prediction] / total_confidence_weight
            
            return {
                'prediction': final_prediction,
                'consensus_ratio': consensus_ratio,
                'is_consensus_sufficient': consensus_ratio >= self.min_consensus_ratio,
                'confidence_weighted_votes': confidence_votes,
                'raw_predictions': predictions,
                'raw_confidences': confidences,
                'fusion_strategy': 'confidence'
            }
    
    def _fusion_numerical(self, 
                        predictions: List[float], 
                        confidences: List[float], 
                        weights: List[float]) -> Dict[str, Any]:
        """
        Fusionne des prédictions numériques (régression).
        
        Args:
            predictions (List): Liste des prédictions numériques
            confidences (List): Liste des confiances associées
            weights (List): Liste des poids des modèles
            
        Returns:
            Dict: Résultat fusionné avec métadonnées
        """
        # Convertir les prédictions en valeurs numériques Python standard si nécessaire
        predictions_std = []
        for p in predictions:
            if hasattr(p, 'dtype'):
                if np.size(p) == 1:  # Tableau avec un seul élément
                    predictions_std.append(float(p.item()))
                else:
                    # Pour les tableaux de taille > 1, utiliser la moyenne
                    predictions_std.append(float(np.mean(p)))
            else:
                predictions_std.append(float(p) if isinstance(p, (int, float)) else p)
        
        # Créer un tableau numpy pour les calculs
        predictions_array = np.array(predictions_std, dtype=float)
        
        # Stratégie simple (moyenne)
        if self.fusion_strategy == 'majority':
            final_prediction = float(np.mean(predictions_array))
            
            return {
                'prediction': final_prediction,
                'std_dev': float(np.std(predictions_array)),
                'min': float(np.min(predictions_array)),
                'max': float(np.max(predictions_array)),
                'raw_predictions': predictions,
                'fusion_strategy': 'majority'
            }
        
        # Stratégie pondérée
        elif self.fusion_strategy == 'weighted':
            weights_array = np.array(weights)
            final_prediction = float(np.average(predictions_array, weights=weights_array))
            
            return {
                'prediction': final_prediction,
                'weighted_std_dev': float(np.sqrt(np.average((predictions_array - final_prediction)**2, weights=weights_array))),
                'min': float(np.min(predictions_array)),
                'max': float(np.max(predictions_array)),
                'raw_predictions': predictions,
                'weights': weights,
                'fusion_strategy': 'weighted'
            }
        
        # Stratégie basée sur la confiance
        elif self.fusion_strategy == 'confidence':
            confidences_array = np.array(confidences)
            weights_array = np.array(weights)
            combined_weights = confidences_array * weights_array
            
            final_prediction = float(np.average(predictions_array, weights=combined_weights))
            
            return {
                'prediction': final_prediction,
                'confidence_weighted_std_dev': float(np.sqrt(np.average((predictions_array - final_prediction)**2, weights=combined_weights))),
                'min': float(np.min(predictions_array)),
                'max': float(np.max(predictions_array)),
                'raw_predictions': predictions,
                'raw_confidences': confidences,
                'fusion_strategy': 'confidence'
            }
    
    def _fusion_complex(self, 
                      predictions: List[Any], 
                      confidences: List[float], 
                      weights: List[float]) -> Dict[str, Any]:
        """
        Fusionne des prédictions complexes (vecteurs, matrices, etc.)
        
        Args:
            predictions (List): Liste des prédictions complexes
            confidences (List): Liste des confiances associées
            weights (List): Liste des poids des modèles
            
        Returns:
            Dict: Résultat fusionné avec métadonnées
        """
        # Adaptation basée sur le type des prédictions
        # Par défaut, retourne la prédiction du modèle le plus confiant
        if self.fusion_strategy == 'confidence':
            combined_weights = [c * w for c, w in zip(confidences, weights)]
            max_idx = np.argmax(combined_weights)
            
            return {
                'prediction': predictions[max_idx],
                'selected_model_idx': max_idx,
                'selected_model_name': self.models[max_idx]['name'],
                'confidence': confidences[max_idx],
                'weight': weights[max_idx],
                'raw_predictions': predictions,
                'fusion_strategy': 'confidence'
            }
        
        # Pour les autres stratégies, utiliser le modèle le plus pondéré
        else:
            max_idx = np.argmax(weights)
            
            return {
                'prediction': predictions[max_idx],
                'selected_model_idx': max_idx,
                'selected_model_name': self.models[max_idx]['name'],
                'weight': weights[max_idx],
                'raw_predictions': predictions,
                'fusion_strategy': self.fusion_strategy
            }
    
    def evaluate(self, X: Any, y_true: Any) -> Dict[str, Dict[str, float]]:
        """
        Évalue les performances de chaque modèle et de l'ensemble.
        
        Args:
            X (Any): Données d'entrée
            y_true (Any): Valeurs cibles réelles
            
        Returns:
            Dict: Métriques de performance pour chaque modèle et l'ensemble
        """
        metrics = {}
        
        # Évaluer chaque modèle individuellement
        for i, model_info in enumerate(self.models):
            model_name = model_info['name']
            prediction, _ = self._get_model_prediction(model_info, X)
            
            try:
                # Pour la classification
                if isinstance(y_true[0], (str, int, bool)) or len(np.unique(y_true)) < 10:
                    # S'assurer que prediction est un tableau de la même forme que y_true
                    if np.isscalar(prediction):
                        prediction_array = np.full_like(y_true, prediction)
                    else:
                        prediction_array = np.array(prediction)
                        
                    accuracy = accuracy_score(y_true, prediction_array)
                    precision = precision_score(y_true, prediction_array, average='weighted', zero_division=0)
                    recall = recall_score(y_true, prediction_array, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, prediction_array, average='weighted', zero_division=0)
                    
                    metrics[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    
                    # Mettre à jour les métriques dans l'objet model_info
                    self.models[i]['performance_metrics'].update(metrics[model_name])
                
                # Pour la régression
                else:
                    # S'assurer que prediction est un tableau de la même forme que y_true
                    if np.isscalar(prediction):
                        prediction_array = np.full_like(y_true, prediction)
                    else:
                        prediction_array = np.array(prediction)
                        
                    mse = np.mean((y_true - prediction_array) ** 2)
                    mae = np.mean(np.abs(y_true - prediction_array))
                    
                    metrics[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': np.sqrt(mse)
                    }
                    
                    # Mettre à jour les métriques dans l'objet model_info
                    self.models[i]['performance_metrics'].update(metrics[model_name])
                
            except Exception as e:
                logger.warning(f"Erreur lors de l'évaluation du modèle '{model_name}': {e}")
                metrics[model_name] = {'error': str(e)}
        
        # Évaluer l'ensemble
        try:
            ensemble_result = self.predict(X)
            ensemble_prediction = ensemble_result['prediction']
            
            # S'assurer que ensemble_prediction est un tableau de la même forme que y_true
            if np.isscalar(ensemble_prediction):
                ensemble_prediction_array = np.full_like(y_true, ensemble_prediction)
            else:
                ensemble_prediction_array = np.array(ensemble_prediction)
            
            # Pour la classification
            if isinstance(y_true[0], (str, int, bool)) or len(np.unique(y_true)) < 10:
                accuracy = accuracy_score(y_true, ensemble_prediction_array)
                precision = precision_score(y_true, ensemble_prediction_array, average='weighted', zero_division=0)
                recall = recall_score(y_true, ensemble_prediction_array, average='weighted', zero_division=0)
                f1 = f1_score(y_true, ensemble_prediction_array, average='weighted', zero_division=0)
                
                metrics['ensemble'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            
            # Pour la régression
            else:
                mse = np.mean((y_true - ensemble_prediction_array) ** 2)
                mae = np.mean(np.abs(y_true - ensemble_prediction_array))
                
                metrics['ensemble'] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'évaluation de l'ensemble: {e}")
            metrics['ensemble'] = {'error': str(e)}
        
        # Ajuster les poids si nécessaire
        if self.adjust_weights and len(self.models) > 1:
            self._adjust_weights(metrics)
        
        return metrics
    
    def _adjust_weights(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Ajuste les poids des modèles en fonction de leurs performances.
        
        Args:
            metrics (Dict): Métriques de performance pour chaque modèle
        """
        # Déterminer la métrique utilisée pour l'ajustement
        if 'accuracy' in metrics[self.models[0]['name']]:
            metric_name = 'accuracy'  # Classification
        elif 'mse' in metrics[self.models[0]['name']]:
            metric_name = 'mse'  # Régression (plus petit = meilleur)
            # Inverser pour que plus grand = meilleur
            for model_name in metrics:
                if model_name != 'ensemble' and metric_name in metrics[model_name]:
                    metrics[model_name][metric_name] = 1 / (metrics[model_name][metric_name] + 1e-10)
        else:
            logger.warning("Impossible d'ajuster les poids: métriques non compatibles")
            return
        
        # Calculer la somme des métriques
        total_metric = sum(
            metrics[model_info['name']][metric_name] 
            for model_info in self.models 
            if metric_name in metrics[model_info['name']]
        )
        
        if total_metric <= 0:
            logger.warning("Impossible d'ajuster les poids: somme des métriques <= 0")
            return
        
        # Ajuster les poids
        for i, model_info in enumerate(self.models):
            model_name = model_info['name']
            if model_name in metrics and metric_name in metrics[model_name]:
                # Nouveau poids = proportion de la métrique
                new_weight = metrics[model_name][metric_name] / total_metric
                old_weight = self.models[i]['weight']
                
                # Mise à jour du poids (avec lissage)
                self.models[i]['weight'] = 0.7 * new_weight + 0.3 * old_weight
                
                logger.info(f"Poids du modèle '{model_name}' ajusté de {old_weight:.4f} à {self.models[i]['weight']:.4f}")
    
    def get_models_info(self) -> List[Dict[str, Any]]:
        """
        Retourne les informations sur tous les modèles de l'ensemble.
        
        Returns:
            List[Dict]: Informations sur chaque modèle
        """
        return [
            {
                'name': m['name'],
                'weight': m['weight'],
                'performance_metrics': m['performance_metrics'],
                'metadata': m['metadata']
            }
            for m in self.models
        ] 