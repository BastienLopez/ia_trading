import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ai_trading.rl.models.multitask_learning_model import MultitaskLearningModel


logger = logging.getLogger(__name__)


class FinancialMultitaskDataset(Dataset):
    """
    Dataset pour l'entraînement multi-tâches sur données financières.
    Prépare les données d'entrée et les cibles pour chaque tâche.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 50,
        prediction_horizons: List[int] = [1, 5, 10, 20],
        asset_columns: Optional[List[str]] = None,
        num_trend_classes: int = 3,
        trend_threshold: float = 0.005,  # ±0.5% pour définir une tendance
    ):
        """
        Initialise le dataset.
        
        Args:
            data: DataFrame avec les données financières
            window_size: Taille de la fenêtre d'observation
            prediction_horizons: Horizons de prédiction en pas de temps
            asset_columns: Colonnes contenant les données pour chaque actif
            num_trend_classes: Nombre de classes de tendance
            trend_threshold: Seuil pour définir une tendance
        """
        self.data = data
        self.window_size = window_size
        self.prediction_horizons = prediction_horizons
        self.num_trend_classes = num_trend_classes
        self.trend_threshold = trend_threshold
        
        # Identifier les colonnes pour chaque actif
        if asset_columns is None:
            # Par défaut, utiliser toutes les colonnes numériques
            self.asset_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.asset_columns = asset_columns
            
        # Calculer le nombre d'échantillons valides
        max_horizon = max(prediction_horizons)
        self.valid_indices = range(window_size, len(data) - max_horizon)
        
        # Calculer les tendances pour chaque horizon
        self._calculate_trends()
        
        # Préparer les poids d'allocation optimaux simulés (pour démonstration)
        self._prepare_optimal_allocations()
        
        # Préparer les paramètres de gestion des risques simulés
        self._prepare_risk_parameters()
        
    def _calculate_trends(self):
        """Calcule les tendances pour chaque horizon."""
        # Utiliser le prix de clôture pour calculer les tendances
        close_prices = self.data['close'].values if 'close' in self.data.columns else self.data.iloc[:, 0].values
        
        self.trends = {}
        
        for horizon in self.prediction_horizons:
            trends = np.zeros(len(self.data) - horizon, dtype=int)
            
            for i in range(len(trends)):
                pct_change = (close_prices[i + horizon] - close_prices[i]) / close_prices[i]
                
                # Classifier la tendance: 0 = baissier, 1 = neutre, 2 = haussier
                if pct_change < -self.trend_threshold:
                    trends[i] = 0  # Baissier
                elif pct_change > self.trend_threshold:
                    trends[i] = 2  # Haussier
                else:
                    trends[i] = 1  # Neutre
                    
            self.trends[horizon] = trends
            
    def _prepare_optimal_allocations(self):
        """
        Prépare des allocations optimales simulées pour l'entraînement.
        Dans un cas réel, cela serait calculé à partir de données historiques.
        """
        # Pour la démonstration, nous simulons des allocations optimales
        num_assets = len(self.asset_columns)
        self.optimal_allocations = np.zeros((len(self.data), num_assets))
        
        # Simuler des allocations qui évoluent lentement
        for i in range(len(self.data)):
            # Allouer proportionnellement aux rendements récents simulés
            allocation = np.random.dirichlet(np.ones(num_assets) * 3)
            self.optimal_allocations[i] = allocation
    
    def _prepare_risk_parameters(self):
        """
        Prépare des paramètres de gestion des risques simulés.
        """
        self.risk_parameters = {
            'stop_loss': np.zeros(len(self.data)),
            'take_profit': np.zeros(len(self.data)),
            'position_size': np.zeros(len(self.data)),
            'risk_score': np.zeros(len(self.data)),
        }
        
        # Volatilité comme proxy pour les paramètres de risque
        if 'close' in self.data.columns:
            prices = self.data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Calculer une volatilité roulante
            for i in range(1, len(self.data)):
                start_idx = max(0, i - 20)
                volatility = np.std(returns[start_idx:i]) if i > start_idx else 0.01
                
                # Paramètres basés sur la volatilité
                self.risk_parameters['stop_loss'][i] = min(0.2, volatility * 2)  # 2x volatilité, max 20%
                self.risk_parameters['take_profit'][i] = min(0.5, volatility * 4)  # 4x volatilité, max 50%
                self.risk_parameters['position_size'][i] = max(0.1, 1.0 - volatility * 5)  # Inversement proportionnel
                self.risk_parameters['risk_score'][i] = min(0.9, volatility * 10)  # Proportionnel, max 90%
        else:
            # Valeurs par défaut si pas de prix de clôture
            self.risk_parameters['stop_loss'] = np.random.uniform(0.02, 0.1, len(self.data))
            self.risk_parameters['take_profit'] = np.random.uniform(0.1, 0.3, len(self.data))
            self.risk_parameters['position_size'] = np.random.uniform(0.3, 0.8, len(self.data))
            self.risk_parameters['risk_score'] = np.random.uniform(0.1, 0.6, len(self.data))
    
    def __len__(self):
        """Retourne le nombre d'échantillons valides."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Retourne un échantillon pour l'entraînement multi-tâches.
        
        Args:
            idx: Index de l'échantillon
            
        Returns:
            Tuple (entrée, cibles) où cibles est un dictionnaire
            pour chaque tâche
        """
        # Index réel dans les données
        real_idx = self.valid_indices[idx]
        
        # Fenêtre d'observation
        window_data = self.data.iloc[real_idx - self.window_size:real_idx].select_dtypes(include=[np.number]).values
        
        # Normaliser l'entrée (simple standardisation)
        window_mean = np.mean(window_data, axis=0, keepdims=True)
        window_std = np.std(window_data, axis=0, keepdims=True) + 1e-8
        window_normalized = (window_data - window_mean) / window_std
        
        # Préparer les cibles pour chaque tâche
        targets = {}
        
        # 1. Prédiction de prix et volumes
        price_volume_targets = {}
        for h in self.prediction_horizons:
            future_idx = real_idx + h
            if future_idx < len(self.data):
                future_data = self.data.iloc[future_idx]
                
                # OHLC (normaliser en utilisant la même normalisation que l'entrée)
                price_targets = []
                for col in ['open', 'high', 'low', 'close']:
                    if col in future_data:
                        price_targets.append(future_data[col])
                    else:
                        # Utiliser une valeur par défaut si colonne manquante
                        price_targets.append(0.0)
                
                price_targets = np.array(price_targets)
                price_targets = (price_targets - window_mean[0, :4]) / window_std[0, :4]
                
                # Volume
                volume_target = future_data['volume'] if 'volume' in future_data else 0.0
                volume_target = (volume_target - window_mean[0, 4]) / window_std[0, 4] if window_data.shape[1] > 4 else volume_target
                
                price_volume_targets[f"h{h}"] = {
                    'price': price_targets,
                    'volume': np.array([volume_target]),
                }
        
        targets['price_prediction'] = price_volume_targets
        
        # 2. Classification de tendances
        trend_targets = {}
        for h in self.prediction_horizons:
            if real_idx < len(self.trends[h]):
                trend_targets[f"h{h}"] = self.trends[h][real_idx]
        
        targets['trend_classification'] = trend_targets
        
        # 3. Optimisation de portefeuille
        targets['portfolio_optimization'] = self.optimal_allocations[real_idx]
        
        # 4. Gestion des risques
        risk_targets = {}
        for param_name, param_values in self.risk_parameters.items():
            risk_targets[param_name] = param_values[real_idx]
        
        targets['risk_management'] = risk_targets
        
        # Convertir en tenseurs PyTorch
        input_tensor = torch.FloatTensor(window_normalized)
        
        # Convertir les cibles en tenseurs
        tensor_targets = {}
        
        # Prix et volumes
        price_volume_tensor = {}
        for h, h_targets in targets['price_prediction'].items():
            price_volume_tensor[h] = {
                'price': torch.FloatTensor(h_targets['price']),
                'volume': torch.FloatTensor(h_targets['volume']),
            }
        tensor_targets['price_prediction'] = price_volume_tensor
        
        # Classification de tendances
        trend_tensor = {}
        for h, trend in targets['trend_classification'].items():
            trend_tensor[h] = torch.LongTensor([trend])
        tensor_targets['trend_classification'] = trend_tensor
        
        # Optimisation de portefeuille
        tensor_targets['portfolio_optimization'] = torch.FloatTensor(targets['portfolio_optimization'])
        
        # Gestion des risques
        risk_tensor = {}
        for param_name, param_value in targets['risk_management'].items():
            risk_tensor[param_name] = torch.FloatTensor([param_value])
        tensor_targets['risk_management'] = risk_tensor
        
        return input_tensor, tensor_targets


class MultitaskTrainer:
    """
    Entraîneur pour le modèle d'apprentissage multi-tâches.
    """
    
    def __init__(
        self, 
        model: MultitaskLearningModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Dict = None,
        lr_scheduler_class: Optional = None,
        lr_scheduler_kwargs: Dict = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'models',
    ):
        """
        Initialise l'entraîneur.
        
        Args:
            model: Modèle multi-tâches à entraîner
            train_loader: DataLoader pour les données d'entraînement
            val_loader: DataLoader pour les données de validation (optionnel)
            optimizer_class: Classe d'optimiseur à utiliser
            optimizer_kwargs: Arguments pour l'optimiseur
            lr_scheduler_class: Classe de scheduler de taux d'apprentissage (optionnel)
            lr_scheduler_kwargs: Arguments pour le scheduler
            device: Périphérique sur lequel entraîner ('cpu' ou 'cuda')
            save_dir: Répertoire où sauvegarder les modèles
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Déplacer le modèle sur le périphérique
        self.model = self.model.to(self.device)
        
        # Initialiser l'optimiseur
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.001}
        self.optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # Initialiser le scheduler (optionnel)
        self.scheduler = None
        if lr_scheduler_class is not None:
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = {}
            self.scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_kwargs)
            
        # Historiques d'entraînement
        self.train_losses = []
        self.val_losses = []
        self.task_train_losses = {
            'price_prediction': [],
            'trend_classification': [],
            'portfolio_optimization': [],
            'risk_management': [],
        }
        self.task_val_losses = {
            'price_prediction': [],
            'trend_classification': [],
            'portfolio_optimization': [],
            'risk_management': [],
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Entraîne le modèle pendant une époque.
        
        Returns:
            Dictionnaire des pertes
        """
        self.model.train()
        total_loss = 0.0
        task_losses = {
            'price_prediction': 0.0,
            'trend_classification': 0.0,
            'portfolio_optimization': 0.0,
            'risk_management': 0.0,
        }
        num_batches = 0
        
        for inputs, targets in self.train_loader:
            # Déplacer les données sur le périphérique
            inputs = inputs.to(self.device)
            
            # Déplacer les cibles sur le périphérique (structure complexe)
            tensor_targets = {}
            
            # Prix et volumes
            price_prediction_targets = {}
            for h, h_targets in targets['price_prediction'].items():
                price_prediction_targets[h] = {
                    'price': h_targets['price'].to(self.device),
                    'volume': h_targets['volume'].to(self.device),
                }
            tensor_targets['price_prediction'] = price_prediction_targets
            
            # Classification de tendances
            trend_targets = {}
            for h, h_target in targets['trend_classification'].items():
                trend_targets[h] = h_target.squeeze().to(self.device)
            tensor_targets['trend_classification'] = trend_targets
            
            # Optimisation de portefeuille
            tensor_targets['portfolio_optimization'] = targets['portfolio_optimization'].to(self.device)
            
            # Gestion des risques
            risk_targets = {}
            for param_name, param_value in targets['risk_management'].items():
                risk_targets[param_name] = param_value.to(self.device)
            tensor_targets['risk_management'] = risk_targets
            
            # Remise à zéro des gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculer les pertes
            loss, batch_task_losses = self.model.compute_combined_loss(
                outputs, 
                tensor_targets
            )
            
            # Backward pass et optimisation
            loss.backward()
            self.optimizer.step()
            
            # Accumulation des pertes
            total_loss += loss.item()
            for task_name, task_loss in batch_task_losses.items():
                task_losses[task_name] += task_loss.item()
                
            num_batches += 1
            
        # Calculer les moyennes
        epoch_loss = total_loss / num_batches
        for task_name in task_losses:
            task_losses[task_name] = task_losses[task_name] / num_batches
            
        # Mettre à jour les historiques
        self.train_losses.append(epoch_loss)
        for task_name, task_loss in task_losses.items():
            self.task_train_losses[task_name].append(task_loss)
            
        return {'total': epoch_loss, **task_losses}
    
    def validate(self) -> Dict[str, float]:
        """
        Valide le modèle sur l'ensemble de validation.
        
        Returns:
            Dictionnaire des pertes de validation
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        task_losses = {
            'price_prediction': 0.0,
            'trend_classification': 0.0,
            'portfolio_optimization': 0.0,
            'risk_management': 0.0,
        }
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Déplacer les données sur le périphérique
                inputs = inputs.to(self.device)
                
                # Déplacer les cibles sur le périphérique (structure complexe)
                tensor_targets = {}
                
                # Prix et volumes
                price_prediction_targets = {}
                for h, h_targets in targets['price_prediction'].items():
                    price_prediction_targets[h] = {
                        'price': h_targets['price'].to(self.device),
                        'volume': h_targets['volume'].to(self.device),
                    }
                tensor_targets['price_prediction'] = price_prediction_targets
                
                # Classification de tendances
                trend_targets = {}
                for h, h_target in targets['trend_classification'].items():
                    trend_targets[h] = h_target.squeeze().to(self.device)
                tensor_targets['trend_classification'] = trend_targets
                
                # Optimisation de portefeuille
                tensor_targets['portfolio_optimization'] = targets['portfolio_optimization'].to(self.device)
                
                # Gestion des risques
                risk_targets = {}
                for param_name, param_value in targets['risk_management'].items():
                    risk_targets[param_name] = param_value.to(self.device)
                tensor_targets['risk_management'] = risk_targets
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculer les pertes
                loss, batch_task_losses = self.model.compute_combined_loss(
                    outputs, 
                    tensor_targets
                )
                
                # Accumulation des pertes
                total_loss += loss.item()
                for task_name, task_loss in batch_task_losses.items():
                    task_losses[task_name] += task_loss.item()
                    
                num_batches += 1
                
        # Calculer les moyennes
        epoch_loss = total_loss / num_batches
        for task_name in task_losses:
            task_losses[task_name] = task_losses[task_name] / num_batches
            
        # Mettre à jour les historiques
        self.val_losses.append(epoch_loss)
        for task_name, task_loss in task_losses.items():
            self.task_val_losses[task_name].append(task_loss)
            
        return {'total': epoch_loss, **task_losses}
    
    def train(
        self, 
        num_epochs: int, 
        early_stopping: bool = True,
        patience: int = 5,
        model_name: str = 'multitask_model',
    ) -> Dict:
        """
        Entraîne le modèle pendant un nombre spécifié d'époques.
        
        Args:
            num_epochs: Nombre d'époques d'entraînement
            early_stopping: Si True, arrête l'entraînement quand la validation n'améliore plus
            patience: Nombre d'époques à attendre pour l'amélioration de la validation
            model_name: Nom de base pour le fichier de modèle sauvegardé
            
        Returns:
            Historiques d'entraînement
        """
        logger.info(f"Début de l'entraînement pour {num_epochs} époques")
        
        best_val_loss = float('inf')
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            # Entraînement
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate() if self.val_loader else {}
            
            # Mise à jour du scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses.get('total', train_losses['total']))
                else:
                    self.scheduler.step()
            
            # Logging
            log_message = f"Époque {epoch+1}/{num_epochs} - "
            log_message += f"Perte d'entraînement: {train_losses['total']:.4f} - "
            
            for task_name, task_loss in train_losses.items():
                if task_name != 'total':
                    log_message += f"{task_name}: {task_loss:.4f} "
                    
            if val_losses:
                log_message += f" | Perte de validation: {val_losses['total']:.4f}"
                
            logger.info(log_message)
            
            # Sauvegarde du meilleur modèle
            if val_losses and val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                no_improve_count = 0
                
                # Sauvegarder le modèle
                self.save_model(f"{model_name}_best.pt")
                logger.info(f"Nouveau meilleur modèle sauvegardé avec perte de validation {best_val_loss:.4f}")
            elif val_losses:
                no_improve_count += 1
                
            # Early stopping
            if early_stopping and no_improve_count >= patience:
                logger.info(f"Early stopping après {epoch+1} époques")
                break
                
        # Sauvegarder le modèle final
        self.save_model(f"{model_name}_final.pt")
        logger.info("Entraînement terminé")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'task_train_losses': self.task_train_losses,
            'task_val_losses': self.task_val_losses,
        }
    
    def save_model(self, filename: str):
        """
        Sauvegarde le modèle.
        
        Args:
            filename: Nom du fichier pour sauvegarder le modèle
        """
        save_path = self.save_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'task_train_losses': self.task_train_losses,
            'task_val_losses': self.task_val_losses,
        }, save_path)
    
    def load_model(self, filename: str):
        """
        Charge un modèle sauvegardé.
        
        Args:
            filename: Nom du fichier de modèle à charger
        """
        load_path = self.save_dir / filename
        
        if not load_path.exists():
            logger.error(f"Le fichier {load_path} n'existe pas")
            return False
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.task_train_losses = checkpoint.get('task_train_losses', {})
        self.task_val_losses = checkpoint.get('task_val_losses', {})
        
        logger.info(f"Modèle chargé depuis {load_path}")
        return True 