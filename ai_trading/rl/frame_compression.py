import numpy as np
import torch
import cv2
import zlib
import pickle
from typing import List, Tuple, Dict, Any, Union, Optional

class FrameCompressor:
    """
    Compresse les états historiques en RL pour réduire l'utilisation de la mémoire
    et accélérer le chargement/stockage des données d'expérience.
    """
    
    def __init__(
        self, 
        compression_level: int = 5,
        frame_stack_size: int = 4,
        resize_dim: Optional[Tuple[int, int]] = None,
        use_grayscale: bool = False,
        quantize: bool = False,
        use_delta_encoding: bool = False
    ):
        """
        Initialise le compresseur de frames.
        
        Args:
            compression_level: Niveau de compression zlib (0-9)
            frame_stack_size: Nombre de frames à empiler
            resize_dim: Dimensions pour redimensionner les frames (width, height)
            use_grayscale: Convertir en niveaux de gris pour réduire les dimensions
            quantize: Quantifier les valeurs pour réduire la précision
            use_delta_encoding: Utiliser l'encodage delta (différences entre frames)
        """
        self.compression_level = compression_level
        self.frame_stack_size = frame_stack_size
        self.resize_dim = resize_dim
        self.use_grayscale = use_grayscale
        self.quantize = quantize
        self.use_delta_encoding = use_delta_encoding
        self.last_frames = []
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prétraite une frame pour réduire sa taille.
        
        Args:
            frame: Frame d'entrée (numpy array)
            
        Returns:
            Frame prétraitée
        """
        # Conversion en niveau de gris si demandé
        if self.use_grayscale and frame.ndim > 2:
            if frame.shape[-1] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.shape[-1] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        
        # Redimensionnement si demandé
        if self.resize_dim is not None:
            frame = cv2.resize(frame, self.resize_dim)
        
        # Quantification si demandée
        if self.quantize:
            # Quantification 8-bit (0-255)
            if frame.dtype != np.uint8:
                min_val, max_val = frame.min(), frame.max()
                if min_val < max_val:  # Éviter division par zéro
                    frame = np.uint8(255 * (frame - min_val) / (max_val - min_val))
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
        
        return frame
    
    def compress_frame(self, frame: np.ndarray) -> bytes:
        """
        Compresse une frame en utilisant zlib.
        
        Args:
            frame: Frame prétraitée
            
        Returns:
            Données compressées
        """
        return zlib.compress(frame.tobytes(), self.compression_level)
    
    def decompress_frame(self, compressed_data: bytes, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """
        Décompresse des données en une frame.
        
        Args:
            compressed_data: Données compressées
            shape: Forme de la frame décompressée
            dtype: Type de données de la frame
            
        Returns:
            Frame décompressée
        """
        decompressed = zlib.decompress(compressed_data)
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
    
    def stack_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Empile plusieurs frames en un seul état.
        
        Args:
            frames: Liste des frames à empiler
            
        Returns:
            Frames empilées
        """
        # Assurer que nous avons le bon nombre de frames
        while len(frames) < self.frame_stack_size:
            # Dupliquer la première frame si pas assez
            frames.insert(0, frames[0].copy() if len(frames) > 0 else np.zeros_like(frames[0]))
            
        # Sélectionner seulement les dernières frames si trop nombreuses
        if len(frames) > self.frame_stack_size:
            frames = frames[-self.frame_stack_size:]
            
        # Empiler selon l'axe approprié
        if frames[0].ndim == 2:  # Grayscale (H, W)
            return np.stack(frames, axis=0)  # Devient (N, H, W)
        elif frames[0].ndim == 3:  # Color (H, W, C)
            return np.concatenate(frames, axis=2)  # Devient (H, W, N*C)
            
        return np.array(frames)  # Cas par défaut
    
    def process_state(self, state: np.ndarray, update_stack: bool = True) -> Union[np.ndarray, bytes]:
        """
        Traite un état complet (prétraitement, compression, empilement).
        
        Args:
            state: État d'entrée
            update_stack: Mettre à jour la stack interne
            
        Returns:
            État traité (compressé ou empilé)
        """
        # Prétraiter la frame
        processed = self.preprocess_frame(state)
        
        # Utiliser l'encodage delta si activé
        if self.use_delta_encoding and self.last_frames:
            base_frame = self.last_frames[-1]
            processed = processed.astype(np.int16) - base_frame.astype(np.int16)
            
        # Mettre à jour la pile de frames
        if update_stack:
            self.last_frames.append(processed.copy())
            if len(self.last_frames) > self.frame_stack_size:
                self.last_frames.pop(0)
        
        # Si empilement demandé
        if self.frame_stack_size > 1:
            return self.stack_frames(list(self.last_frames))
        
        return processed
    
    def compress_state(self, state: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compresse un état entier avec métadonnées.
        
        Args:
            state: État à compresser
            
        Returns:
            Tuple (données compressées, métadonnées)
        """
        processed = self.process_state(state, update_stack=True)
        
        metadata = {
            "shape": processed.shape,
            "dtype": str(processed.dtype),
            "frame_stack_size": self.frame_stack_size,
            "compressed": True
        }
        
        compressed = self.compress_frame(processed)
        return compressed, metadata
    
    def decompress_state(self, compressed_data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Décompresse un état avec ses métadonnées.
        
        Args:
            compressed_data: Données compressées
            metadata: Métadonnées nécessaires pour la décompression
            
        Returns:
            État décompressé
        """
        shape = metadata["shape"]
        dtype = np.dtype(metadata["dtype"])
        
        return self.decompress_frame(compressed_data, shape, dtype)
    
    def reset(self):
        """
        Réinitialise l'état interne du compresseur.
        """
        self.last_frames = []

class FrameStackWrapper:
    """
    Wrapper pour l'empilement de frames compatible avec les environnements RL.
    Peut être utilisé comme un wrapper gym ou directement.
    """
    
    def __init__(
        self,
        env=None,
        n_frames: int = 4,
        compress: bool = False,
        compression_level: int = 5
    ):
        """
        Initialise le wrapper d'empilement de frames.
        
        Args:
            env: Environnement à wrapper (optionnel)
            n_frames: Nombre de frames à empiler
            compress: Utiliser la compression
            compression_level: Niveau de compression (0-9)
        """
        self.env = env
        self.n_frames = n_frames
        self.frames = []
        self.compress = compress
        self.compression_level = compression_level
        
        # Initialiser le compresseur si nécessaire
        if self.compress:
            self.compressor = FrameCompressor(
                compression_level=compression_level,
                frame_stack_size=n_frames
            )
    
    def reset(self, **kwargs):
        """
        Réinitialise l'environnement et la pile de frames.
        
        Returns:
            État initial empilé
        """
        if self.env is not None:
            observation = self.env.reset(**kwargs)
        else:
            observation = kwargs.get('observation', None)
            if observation is None:
                raise ValueError("Si env est None, observation doit être fournie")
        
        self.frames = [observation] * self.n_frames
        
        if self.compress:
            self.compressor.reset()
            return self.compressor.process_state(observation)
        
        return self._get_observation()
    
    def step(self, action):
        """
        Fait un pas dans l'environnement et met à jour la pile de frames.
        
        Args:
            action: Action à prendre
            
        Returns:
            (observation empilée, reward, done, info)
        """
        if self.env is None:
            raise ValueError("Cette méthode ne peut être utilisée que si env est fourni")
            
        observation, reward, done, info = self.env.step(action)
        
        self.frames.pop(0)
        self.frames.append(observation)
        
        if self.compress:
            return self.compressor.process_state(observation), reward, done, info
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """
        Renvoit l'observation actuelle (pile de frames).
        
        Returns:
            Observation empilée
        """
        # Empiler les frames selon leur dimension
        if isinstance(self.frames[0], (np.ndarray, torch.Tensor)):
            if hasattr(self.frames[0], 'shape') and len(self.frames[0].shape) == 3:
                # Images (H, W, C) -> (H, W, n_frames*C)
                return np.concatenate(self.frames, axis=2)
            else:
                # Autres arrays -> premier axe
                return np.stack(self.frames, axis=0)
        
        # Pour d'autres types de données
        return np.array(self.frames)
    
    def add_frame(self, frame):
        """
        Ajoute manuellement une frame à la pile.
        
        Args:
            frame: Nouvelle frame à ajouter
            
        Returns:
            Nouvelle observation empilée
        """
        if len(self.frames) >= self.n_frames:
            self.frames.pop(0)
        
        self.frames.append(frame)
        
        if self.compress:
            return self.compressor.process_state(frame)
        
        return self._get_observation() 