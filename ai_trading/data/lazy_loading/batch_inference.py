#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour l'optimisation des inférences par lots (batch inference).
Permet de traiter plusieurs prédictions en même temps pour économiser CPU/GPU.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchInferenceOptimizer:
    """
    Classe pour optimiser les inférences par lots avec les modèles PyTorch.
    Permet des prédictions efficaces sur de grands ensembles de données.
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_half_precision: bool = False,
        use_torch_compile: bool = False,
        optimize_for_inference: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        timeout: int = 0,
    ):
        """
        Initialise l'optimiseur d'inférence par lots.
        
        Args:
            model: Modèle PyTorch à utiliser pour l'inférence.
            batch_size: Taille des lots pour l'inférence.
            device: Périphérique sur lequel exécuter l'inférence ('cpu', 'cuda', 'cuda:0', etc.).
            use_half_precision: Si True, utilise la précision FP16 pour l'inférence.
            use_torch_compile: Si True, compile le modèle avec torch.compile pour accélérer l'inférence.
            optimize_for_inference: Si True, optimise le modèle pour l'inférence (fusion des couches, etc.).
            num_workers: Nombre de workers pour le DataLoader.
            pin_memory: Si True, utilise la mémoire paginée pour un transfert plus rapide vers le GPU.
            prefetch_factor: Nombre de batchs à précharger par worker.
            persistent_workers: Si True, garde les workers en vie entre les itérations.
            timeout: Délai d'attente pour le DataLoader (0 = pas de limite).
        """
        self.original_model = model
        self.batch_size = batch_size
        self.device = device
        self.use_half_precision = use_half_precision
        self.use_torch_compile = use_torch_compile
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.timeout = timeout
        
        # Optimiser le modèle pour l'inférence
        self.model = self._prepare_model(model, optimize_for_inference)
        
    def _prepare_model(self, model: nn.Module, optimize_for_inference: bool) -> nn.Module:
        """
        Prépare le modèle pour une inférence optimisée.
        
        Args:
            model: Modèle à optimiser.
            optimize_for_inference: Si True, applique des optimisations spécifiques.
            
        Returns:
            Modèle optimisé.
        """
        # Copier le modèle pour ne pas modifier l'original
        inference_model = model
        
        # Mettre le modèle en mode évaluation
        inference_model.eval()
        
        # Déplacer le modèle sur le périphérique cible
        inference_model = inference_model.to(self.device)
        
        # Utiliser la demi-précision si demandé
        if self.use_half_precision:
            inference_model = inference_model.half()
        
        # Optimiser pour l'inférence
        if optimize_for_inference:
            # Fusionner les couches BatchNorm avec les couches Conv si possible
            for module in inference_model.modules():
                if hasattr(module, 'fuse_model'):
                    module.fuse_model()
            
            # Compiler avec TorchScript
            try:
                # Créer un exemple d'entrée
                sample_shape = self._get_sample_input_shape(inference_model)
                
                if sample_shape is not None:
                    sample_input = torch.rand(*sample_shape, device=self.device)
                    if self.use_half_precision:
                        sample_input = sample_input.half()
                    
                    # Tracer le modèle pour l'optimisation
                    with torch.no_grad():
                        traced_model = torch.jit.trace(inference_model, sample_input)
                    
                    # Optimiser le modèle tracé
                    inference_model = torch.jit.optimize_for_inference(traced_model)
                else:
                    logger.warning("Impossible de déterminer la forme d'entrée du modèle pour l'optimisation")
            except Exception as e:
                logger.warning(f"Erreur lors de l'optimisation du modèle avec TorchScript: {e}")
        
        # Compiler avec torch.compile si demandé et disponible
        if self.use_torch_compile and hasattr(torch, 'compile'):
            try:
                inference_model = torch.compile(inference_model, mode="reduce-overhead")
                logger.info("Modèle compilé avec torch.compile")
            except Exception as e:
                logger.warning(f"Erreur lors de la compilation du modèle: {e}")
        
        return inference_model
    
    def _get_sample_input_shape(self, model: nn.Module) -> Optional[Tuple[int, ...]]:
        """
        Tente de déterminer la forme d'entrée du modèle.
        
        Args:
            model: Modèle à analyser.
            
        Returns:
            Forme estimée de l'entrée ou None si impossible à déterminer.
        """
        # Vérifier si le modèle a un attribut pour la forme d'entrée
        if hasattr(model, 'input_shape'):
            shape = getattr(model, 'input_shape')
            return (1, *shape) if isinstance(shape, tuple) else (1, shape)
        
        # Essayer de déduire à partir des paramètres
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                return (1, module.in_channels, 100)  # Estimation pour séquences temporelles
            elif isinstance(module, nn.Conv2d):
                return (1, module.in_channels, 32, 32)  # Estimation pour images
            elif isinstance(module, nn.Linear) and module == list(model.modules())[1]:
                return (1, module.in_features)  # Premier module linéaire
        
        # Impossible de déterminer
        return None
    
    def predict(
        self,
        data: Union[torch.Tensor, np.ndarray, List, Dataset],
        return_numpy: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Effectue des prédictions par lots sur les données.
        
        Args:
            data: Données pour l'inférence (tenseur, tableau, liste ou Dataset).
            return_numpy: Si True, retourne les résultats sous forme de tableau NumPy.
            progress_callback: Fonction de rappel pour suivre la progression (reçoit current, total).
            
        Returns:
            Prédictions du modèle.
        """
        # Convertir les données en dataset PyTorch
        if isinstance(data, Dataset):
            dataset = data
        elif isinstance(data, torch.Tensor):
            dataset = TensorDataset(data)
        elif isinstance(data, np.ndarray):
            tensor_data = torch.from_numpy(data.astype(np.float32))
            dataset = TensorDataset(tensor_data)
        elif isinstance(data, list):
            if all(isinstance(x, torch.Tensor) for x in data):
                dataset = TensorDataset(*data)
            else:
                # Convertir en tenseur si ce sont des tableaux ou des listes
                tensor_data = torch.tensor(data, dtype=torch.float32)
                dataset = TensorDataset(tensor_data)
        else:
            raise ValueError(f"Type de données non pris en charge: {type(data)}")
        
        # Créer un DataLoader optimisé
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            timeout=self.timeout
        )
        
        # Effectuer les prédictions
        all_predictions = []
        start_time = time.time()
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                # Extraire les features (premier élément du tuple)
                if isinstance(batch_data, torch.Tensor):
                    features = batch_data
                else:
                    features = batch_data[0]
                
                # Transférer sur le périphérique cible
                features = features.to(self.device)
                
                # Convertir en demi-précision si nécessaire
                if self.use_half_precision:
                    features = features.half()
                
                # Faire l'inférence
                batch_predictions = self.model(features)
                
                # Transférer les prédictions sur CPU et convertir en float32
                batch_predictions = batch_predictions.cpu().float()
                
                # Ajouter aux prédictions
                all_predictions.append(batch_predictions)
                
                # Mettre à jour la progression si une fonction de rappel est fournie
                if progress_callback:
                    progress_callback(i + 1, total_batches)
        
        # Concaténer toutes les prédictions
        predictions = torch.cat(all_predictions, dim=0)
        
        # Convertir en NumPy si demandé
        if return_numpy:
            predictions = predictions.numpy()
        
        # Logging des performances
        elapsed_time = time.time() - start_time
        samples_per_second = len(dataset) / elapsed_time
        logger.info(f"Inférence complétée: {len(dataset)} échantillons en {elapsed_time:.2f}s ({samples_per_second:.2f} échantillons/s)")
        
        return predictions
    
    def predict_proba(
        self,
        data: Union[torch.Tensor, np.ndarray, List, Dataset],
        return_numpy: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Effectue des prédictions de probabilité par lots.
        Applique softmax (pour classification) ou sigmoid (pour prédiction binaire).
        
        Args:
            data: Données pour l'inférence.
            return_numpy: Si True, retourne les résultats sous forme de tableau NumPy.
            progress_callback: Fonction de rappel pour suivre la progression.
            
        Returns:
            Probabilités prédites.
        """
        raw_predictions = self.predict(data, return_numpy=False, progress_callback=progress_callback)
        
        # Déterminer le type de problème (classification ou régression)
        if raw_predictions.dim() > 1 and raw_predictions.size(1) > 1:
            # Classification multi-classes, appliquer softmax
            probas = torch.softmax(raw_predictions, dim=1)
        else:
            # Binaire ou régression, appliquer sigmoid
            probas = torch.sigmoid(raw_predictions)
        
        # Convertir en NumPy si demandé
        if return_numpy:
            probas = probas.numpy()
        
        return probas
    
    def predict_classes(
        self,
        data: Union[torch.Tensor, np.ndarray, List, Dataset],
        return_numpy: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Prédit les classes pour des données de classification.
        
        Args:
            data: Données pour l'inférence.
            return_numpy: Si True, retourne les résultats sous forme de tableau NumPy.
            progress_callback: Fonction de rappel pour suivre la progression.
            
        Returns:
            Classes prédites.
        """
        raw_predictions = self.predict(data, return_numpy=False, progress_callback=progress_callback)
        
        # Obtenir l'indice de la classe avec la plus haute probabilité
        if raw_predictions.dim() > 1 and raw_predictions.size(1) > 1:
            # Classification multi-classes
            predicted_classes = torch.argmax(raw_predictions, dim=1)
        else:
            # Classification binaire
            predicted_classes = (raw_predictions > 0.5).long().squeeze()
        
        # Convertir en NumPy si demandé
        if return_numpy:
            predicted_classes = predicted_classes.numpy()
        
        return predicted_classes
    
    def batch_predict_generator(
        self,
        data_generator: Any,
        batch_transform: Optional[Callable] = None,
        device_transfer: bool = True,
    ) -> Any:
        """
        Effectue des prédictions sur un générateur de données en utilisant des lots.
        Utile pour les flux de données ou les générateurs personnalisés.
        
        Args:
            data_generator: Générateur ou itérateur de données.
            batch_transform: Fonction de transformation à appliquer à chaque lot.
            device_transfer: Si True, transfère les données sur le périphérique cible.
            
        Yields:
            Prédictions par lots.
        """
        with torch.no_grad():
            current_batch = []
            
            for data_item in data_generator:
                # Ajouter l'élément au lot courant
                current_batch.append(data_item)
                
                # Traiter le lot quand il atteint la taille souhaitée
                if len(current_batch) >= self.batch_size:
                    # Convertir le lot en tenseur
                    if batch_transform:
                        batch = batch_transform(current_batch)
                    else:
                        batch = torch.stack(current_batch) if isinstance(current_batch[0], torch.Tensor) else torch.tensor(current_batch)
                    
                    # Transférer sur le périphérique cible si demandé
                    if device_transfer:
                        batch = batch.to(self.device)
                    
                    # Appliquer la demi-précision si nécessaire
                    if self.use_half_precision:
                        batch = batch.half()
                    
                    # Faire l'inférence
                    predictions = self.model(batch)
                    
                    # Retourner les prédictions
                    yield predictions.cpu()
                    
                    # Réinitialiser le lot
                    current_batch = []
            
            # Traiter le dernier lot s'il reste des éléments
            if current_batch:
                # Convertir le lot en tenseur
                if batch_transform:
                    batch = batch_transform(current_batch)
                else:
                    batch = torch.stack(current_batch) if isinstance(current_batch[0], torch.Tensor) else torch.tensor(current_batch)
                
                # Transférer sur le périphérique cible si demandé
                if device_transfer:
                    batch = batch.to(self.device)
                
                # Appliquer la demi-précision si nécessaire
                if self.use_half_precision:
                    batch = batch.half()
                
                # Faire l'inférence
                predictions = self.model(batch)
                
                # Retourner les prédictions
                yield predictions.cpu()


def batch_inference(
    model: nn.Module,
    data: Union[torch.Tensor, np.ndarray, List, Dataset],
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_half_precision: bool = False,
    return_numpy: bool = True,
    optimize_for_inference: bool = True,
    num_workers: int = 4,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Fonction utilitaire pour effectuer une inférence par lots.
    
    Args:
        model: Modèle PyTorch.
        data: Données pour l'inférence.
        batch_size: Taille des lots.
        device: Périphérique pour l'inférence.
        use_half_precision: Si True, utilise la précision FP16.
        return_numpy: Si True, retourne un tableau NumPy.
        optimize_for_inference: Si True, optimise le modèle pour l'inférence.
        num_workers: Nombre de workers pour le DataLoader.
        
    Returns:
        Prédictions du modèle.
    """
    # Créer l'optimiseur d'inférence
    optimizer = BatchInferenceOptimizer(
        model=model,
        batch_size=batch_size,
        device=device,
        use_half_precision=use_half_precision,
        optimize_for_inference=optimize_for_inference,
        num_workers=num_workers
    )
    
    # Effectuer les prédictions
    return optimizer.predict(data, return_numpy=return_numpy) 