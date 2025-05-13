"""
Module de détection de patterns chartistes basé sur des réseaux de neurones convolutifs.
"""
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import os


class ChartPatternType(Enum):
    """Types de patterns chartistes reconnus par le détecteur."""
    HEAD_AND_SHOULDERS = "tête_et_épaules"
    INVERSE_HEAD_AND_SHOULDERS = "tête_et_épaules_inversé"
    DOUBLE_TOP = "double_sommet"
    DOUBLE_BOTTOM = "double_plancher"
    TRIPLE_TOP = "triple_sommet"
    TRIPLE_BOTTOM = "triple_plancher"
    ASCENDING_TRIANGLE = "triangle_ascendant"
    DESCENDING_TRIANGLE = "triangle_descendant"
    SYMMETRICAL_TRIANGLE = "triangle_symétrique"
    RISING_WEDGE = "coin_montant"
    FALLING_WEDGE = "coin_descendant"
    RECTANGLE = "rectangle"
    CUP_AND_HANDLE = "tasse_et_anse"
    INVERSE_CUP_AND_HANDLE = "tasse_et_anse_inversée"
    FLAG = "drapeau"
    PENNANT = "fanion"


@dataclass
class PatternInstance:
    """Représente une instance de pattern détecté."""
    pattern_type: ChartPatternType
    start_idx: int
    end_idx: int
    confidence: float
    breakout_target: Optional[float] = None
    stop_loss: Optional[float] = None
    
    @property
    def duration(self) -> int:
        """Retourne la durée du pattern en nombre de points."""
        return self.end_idx - self.start_idx + 1
    
    def to_dict(self) -> Dict:
        """Convertit l'instance en dictionnaire."""
        return {
            "pattern_type": self.pattern_type.value,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "confidence": self.confidence,
            "duration": self.duration,
            "breakout_target": self.breakout_target,
            "stop_loss": self.stop_loss
        }


class CNNPatternDetector:
    """
    Détecteur de patterns chartistes basé sur un réseau de neurones convolutif.
    
    Cette classe permet:
    1. L'entraînement d'un modèle CNN à partir de données labellisées
    2. La détection de patterns dans de nouvelles données
    3. Le calcul de cibles de prix et de stops basés sur les patterns détectés
    """
    
    def __init__(self, window_size: int = 128, confidence_threshold: float = 0.7):
        """
        Initialise le détecteur de patterns CNN.
        
        Args:
            window_size: Taille de la fenêtre d'analyse (default: 128)
            confidence_threshold: Seuil de confiance pour la détection (default: 0.7)
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.pattern_types = list(ChartPatternType)
        
    def build_model(self, input_shape: Tuple[int, int] = None):
        """
        Construit l'architecture du modèle CNN.
        
        Args:
            input_shape: Forme des données d'entrée (default: (window_size, 5))
                          5 représente OHLCV ou d'autres caractéristiques
        """
        if input_shape is None:
            input_shape = (self.window_size, 5)
            
        # Vérifier si la fenêtre est assez grande pour les opérations de pooling
        if input_shape[0] < 8:
            # Pour les petites fenêtres de test, utiliser un modèle simplifié
            model = models.Sequential([
                layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                             input_shape=input_shape),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(len(self.pattern_types) + 1, activation='softmax')  # +1 pour "pas de pattern"
            ])
        else:
            # Définir l'architecture du modèle CNN
            model = models.Sequential([
                # Première couche convolutive
                layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                             input_shape=input_shape),
                layers.MaxPooling1D(pool_size=2),
                layers.BatchNormalization(),
                
                # Deuxième couche convolutive
                layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                layers.MaxPooling1D(pool_size=2),
                layers.BatchNormalization(),
                
                # Troisième couche convolutive
                layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
                layers.MaxPooling1D(pool_size=2),
                layers.BatchNormalization(),
                
                # Aplatissement et couches denses
                layers.Flatten(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(len(self.pattern_types) + 1, activation='softmax')  # +1 pour "pas de pattern"
            ])
        
        # Compiler le modèle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prétraite les données pour l'entrée du modèle.
        
        Args:
            data: DataFrame contenant au moins les colonnes OHLCV
            
        Returns:
            Tableau numpy normalisé prêt pour l'entrée du modèle
        """
        # Vérifier que les colonnes OHLCV existent
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"La colonne {col} est requise dans les données")
        
        # Extraire les caractéristiques
        features = data[required_columns].values
        
        # Normalisation des prix par le prix de clôture initial de chaque fenêtre
        price_cols = features[:, 0:4]  # open, high, low, close
        for i in range(0, len(features), self.window_size):
            end_idx = min(i + self.window_size, len(features))
            if i < end_idx:
                ref_price = price_cols[i, 3]  # close price du premier point
                if ref_price != 0:  # éviter division par zéro
                    price_cols[i:end_idx, 0:4] = price_cols[i:end_idx, 0:4] / ref_price
        
        # Normalisation du volume
        volume = features[:, 4:5]
        # Log-transformation pour gérer les grandes valeurs
        volume = np.log1p(volume)
        # Normalisation min-max par fenêtre
        for i in range(0, len(volume), self.window_size):
            end_idx = min(i + self.window_size, len(volume))
            if i < end_idx:
                window_vol = volume[i:end_idx]
                min_vol, max_vol = np.min(window_vol), np.max(window_vol)
                if max_vol > min_vol:  # éviter division par zéro
                    volume[i:end_idx] = (window_vol - min_vol) / (max_vol - min_vol)
        
        # Recombiner prix et volume
        features[:, 0:4] = price_cols
        features[:, 4:5] = volume
        
        return features
    
    def create_sliding_windows(self, features: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Crée des fenêtres glissantes à partir des données.
        
        Args:
            features: Tableau de caractéristiques prétraitées
            stride: Pas entre les fenêtres successives (default: 1)
            
        Returns:
            Tableau de fenêtres glissantes
        """
        if len(features) < self.window_size:
            raise ValueError(f"Données insuffisantes: {len(features)} points, {self.window_size} requis")
            
        # Calculer le nombre de fenêtres
        n_windows = max(1, (len(features) - self.window_size) // stride + 1)
        
        # Initialiser le tableau de sortie
        windows = np.zeros((n_windows, self.window_size, features.shape[1]))
        
        # Remplir les fenêtres
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + self.window_size
            windows[i] = features[start_idx:end_idx]
            
        return windows
    
    def train(self, train_data: np.ndarray, train_labels: np.ndarray, 
              val_data: Optional[np.ndarray] = None, 
              val_labels: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32, 
              save_path: Optional[str] = None):
        """
        Entraîne le modèle de détection de patterns.
        
        Args:
            train_data: Données d'entraînement sous forme de fenêtres
            train_labels: Étiquettes one-hot de chaque fenêtre
            val_data: Données de validation (default: None)
            val_labels: Étiquettes de validation (default: None)
            epochs: Nombre d'époques d'entraînement (default: 50)
            batch_size: Taille des lots (default: 32)
            save_path: Chemin pour sauvegarder le modèle (default: None)
        """
        # S'assurer que le modèle existe
        if self.model is None:
            input_shape = (self.window_size, train_data.shape[2])
            self.build_model(input_shape)
            
        # Préparer les données de validation
        validation_data = None
        if val_data is not None and val_labels is not None:
            validation_data = (val_data, val_labels)
            
        # Entraîner le modèle
        history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # Sauvegarder le modèle si un chemin est spécifié
        if save_path:
            self.model.save(save_path)
            
        return history
        
    def load_model(self, model_path: str):
        """
        Charge un modèle préentraîné.
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        self.model = tf.keras.models.load_model(model_path)
        
    def detect_patterns(self, data: pd.DataFrame, stride: int = 5) -> List[PatternInstance]:
        """
        Détecte les patterns dans une série temporelle.
        
        Args:
            data: DataFrame contenant au moins les colonnes OHLCV
            stride: Pas entre les fenêtres successives (default: 5)
            
        Returns:
            Liste des instances de patterns détectés
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas initialisé. Utilisez build_model() ou load_model()")
            
        # Prétraiter les données
        features = self.preprocess_data(data)
        
        # Créer des fenêtres glissantes
        windows = self.create_sliding_windows(features, stride)
        
        # Prédire les patterns
        predictions = self.model.predict(windows)
        
        # Traiter les prédictions
        results = []
        for i, pred in enumerate(predictions):
            # Ignorer la classe "pas de pattern" (dernière)
            pattern_probs = pred[:-1]
            max_prob = np.max(pattern_probs)
            
            # Vérifier si la confiance est suffisante
            if max_prob >= self.confidence_threshold:
                pattern_idx = np.argmax(pattern_probs)
                pattern_type = self.pattern_types[pattern_idx]
                
                # Calculer les indices de début et de fin dans les données originales
                start_idx = i * stride
                end_idx = start_idx + self.window_size - 1
                
                # Créer l'instance de pattern
                pattern = PatternInstance(
                    pattern_type=pattern_type,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    confidence=max_prob
                )
                
                # Ajouter les cibles de trading
                self._add_trading_targets(pattern, data.iloc[start_idx:end_idx+1])
                
                results.append(pattern)
                
        # Fusionner les patterns similaires qui se chevauchent
        results = self._merge_overlapping_patterns(results)
        
        return results
    
    def _add_trading_targets(self, pattern: PatternInstance, pattern_data: pd.DataFrame):
        """
        Ajoute des cibles de trading et stops basés sur le type de pattern.
        
        Args:
            pattern: Instance du pattern détecté
            pattern_data: Données du pattern (sous-ensemble du DataFrame)
        """
        # Obtenir les prix pertinents
        high = pattern_data['high'].max()
        low = pattern_data['low'].min()
        close = pattern_data['close'].iloc[-1]
        height = high - low
        
        # Définir les cibles en fonction du type de pattern
        if pattern.pattern_type in [ChartPatternType.HEAD_AND_SHOULDERS, 
                                   ChartPatternType.TRIPLE_TOP,
                                   ChartPatternType.DOUBLE_TOP,
                                   ChartPatternType.RISING_WEDGE,
                                   ChartPatternType.RECTANGLE]:
            # Patterns baissiers
            pattern.breakout_target = close - height
            pattern.stop_loss = high + 0.1 * height
            
        elif pattern.pattern_type in [ChartPatternType.INVERSE_HEAD_AND_SHOULDERS,
                                     ChartPatternType.TRIPLE_BOTTOM,
                                     ChartPatternType.DOUBLE_BOTTOM,
                                     ChartPatternType.FALLING_WEDGE,
                                     ChartPatternType.CUP_AND_HANDLE]:
            # Patterns haussiers
            pattern.breakout_target = close + height
            pattern.stop_loss = low - 0.1 * height
            
        elif pattern.pattern_type == ChartPatternType.ASCENDING_TRIANGLE:
            # Triangle ascendant (haussier)
            pattern.breakout_target = close + 0.7 * height
            pattern.stop_loss = close - 0.3 * height
            
        elif pattern.pattern_type == ChartPatternType.DESCENDING_TRIANGLE:
            # Triangle descendant (baissier)
            pattern.breakout_target = close - 0.7 * height
            pattern.stop_loss = close + 0.3 * height
            
        elif pattern.pattern_type == ChartPatternType.SYMMETRICAL_TRIANGLE:
            # Triangle symétrique (direction incertaine)
            # On ne définit pas de targets car la direction est incertaine
            pass
            
        elif pattern.pattern_type in [ChartPatternType.FLAG, ChartPatternType.PENNANT]:
            # Drapeaux et fanions (continuation)
            # Détecter la tendance précédente
            first_half = pattern_data.iloc[:len(pattern_data)//2]
            trend = 1 if first_half['close'].iloc[-1] > first_half['close'].iloc[0] else -1
            
            pattern.breakout_target = close + trend * height
            pattern.stop_loss = close - trend * 0.5 * height
    
    def _merge_overlapping_patterns(self, patterns: List[PatternInstance]) -> List[PatternInstance]:
        """
        Fusionne les patterns similaires qui se chevauchent.
        
        Args:
            patterns: Liste des patterns détectés
            
        Returns:
            Liste des patterns après fusion
        """
        if not patterns:
            return []
            
        # Trier les patterns par indice de début
        sorted_patterns = sorted(patterns, key=lambda p: p.start_idx)
        
        # Liste pour stocker les patterns fusionnés
        merged = [sorted_patterns[0]]
        
        for current in sorted_patterns[1:]:
            previous = merged[-1]
            
            # Vérifier si les patterns se chevauchent et sont du même type
            if (current.start_idx <= previous.end_idx and 
                current.pattern_type == previous.pattern_type):
                
                # Calculer les durées
                prev_duration = previous.end_idx - previous.start_idx + 1
                curr_duration = current.end_idx - current.start_idx + 1
                
                # Calculer la confiance pondérée par la durée
                weighted_confidence = (previous.confidence * prev_duration + 
                                      current.confidence * curr_duration) / (prev_duration + curr_duration)
                
                # Étendre le pattern précédent
                previous.end_idx = max(previous.end_idx, current.end_idx)
                
                # Mettre à jour la confiance
                previous.confidence = weighted_confidence
                
                # Conserver les cibles les plus récentes
                previous.breakout_target = current.breakout_target
                previous.stop_loss = current.stop_loss
                
            else:
                # Ajouter un nouveau pattern à la liste
                merged.append(current)
                
        return merged
    
    def visualize_patterns(self, data: pd.DataFrame, patterns: List[PatternInstance], 
                          save_path: Optional[str] = None):
        """
        Visualise les patterns détectés sur un graphique.
        
        Args:
            data: DataFrame avec les données OHLCV
            patterns: Liste des patterns détectés
            save_path: Dossier pour sauvegarder les visualisations (default: None)
        """
        if not patterns:
            print("Aucun pattern à visualiser")
            return
            
        # Créer un dossier pour les visualisations si nécessaire
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Couleurs pour différents types de patterns
        pattern_colors = {
            ChartPatternType.HEAD_AND_SHOULDERS: 'red',
            ChartPatternType.INVERSE_HEAD_AND_SHOULDERS: 'green',
            ChartPatternType.DOUBLE_TOP: 'red',
            ChartPatternType.DOUBLE_BOTTOM: 'green',
            ChartPatternType.TRIPLE_TOP: 'red',
            ChartPatternType.TRIPLE_BOTTOM: 'green',
            ChartPatternType.ASCENDING_TRIANGLE: 'green',
            ChartPatternType.DESCENDING_TRIANGLE: 'red',
            ChartPatternType.SYMMETRICAL_TRIANGLE: 'blue',
            ChartPatternType.RISING_WEDGE: 'red',
            ChartPatternType.FALLING_WEDGE: 'green',
            ChartPatternType.RECTANGLE: 'blue',
            ChartPatternType.CUP_AND_HANDLE: 'green',
            ChartPatternType.INVERSE_CUP_AND_HANDLE: 'red',
            ChartPatternType.FLAG: 'purple',
            ChartPatternType.PENNANT: 'orange'
        }
        
        # Visualiser chaque pattern
        for i, pattern in enumerate(patterns):
            start_idx = max(0, pattern.start_idx - 20)  # Contexte avant le pattern
            end_idx = min(len(data) - 1, pattern.end_idx + 20)  # Contexte après le pattern
            
            # Données pour la visualisation
            plot_data = data.iloc[start_idx:end_idx+1].copy()
            
            # Créer la figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Tracer les prix
            ax.plot(plot_data['close'], label='Prix de clôture', color='black', alpha=0.7)
            
            # Mettre en évidence le pattern
            pattern_slice = slice(pattern.start_idx - start_idx, pattern.end_idx - start_idx + 1)
            ax.plot(plot_data.index[pattern_slice], 
                   plot_data['close'].iloc[pattern_slice], 
                   color=pattern_colors.get(pattern.pattern_type, 'blue'),
                   linewidth=2,
                   label=f"{pattern.pattern_type.value} (conf. {pattern.confidence:.2f})")
            
            # Ajouter la zone ombragée pour le pattern
            ax.axvspan(plot_data.index[pattern_slice.start], 
                      plot_data.index[pattern_slice.stop-1], 
                      alpha=0.2, 
                      color=pattern_colors.get(pattern.pattern_type, 'blue'))
            
            # Ajouter les cibles de trading si disponibles
            last_price = plot_data['close'].iloc[-1]
            if pattern.breakout_target:
                ax.axhline(y=pattern.breakout_target, color='green', linestyle='--', alpha=0.7, 
                          label=f"Cible: {pattern.breakout_target:.2f}")
                
            if pattern.stop_loss:
                ax.axhline(y=pattern.stop_loss, color='red', linestyle='--', alpha=0.7, 
                          label=f"Stop: {pattern.stop_loss:.2f}")
            
            # Configuration du graphique
            ax.set_title(f"Pattern détecté: {pattern.pattern_type.value}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Prix")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ajouter du texte descriptif
            ax.text(0.02, 0.02, 
                   f"Durée: {pattern.duration} points\nConfiance: {pattern.confidence:.2f}", 
                   transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            
            # Sauvegarder ou afficher
            if save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pattern_name = pattern.pattern_type.value.replace(" ", "_")
                filename = f"{save_path}/pattern_{pattern_name}_{timestamp}_{i}.png"
                plt.savefig(filename, dpi=300)
                plt.close()
            else:
                plt.show()
                
    def export_to_json(self, patterns: List[PatternInstance], filepath: str):
        """
        Exporte les patterns détectés au format JSON.
        
        Args:
            patterns: Liste des patterns détectés
            filepath: Chemin du fichier de sortie
        """
        import json
        
        # Convertir les patterns en dictionnaires
        patterns_dict = [p.to_dict() for p in patterns]
        
        # Écrire dans le fichier
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patterns_dict, f, ensure_ascii=False, indent=2) 