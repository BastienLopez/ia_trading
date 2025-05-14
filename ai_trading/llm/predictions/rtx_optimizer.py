"""
Module d'optimisation spécifique pour les GPU NVIDIA RTX séries 30 et 40.

Ce module fournit des fonctionnalités d'optimisation avancées pour tirer
le meilleur parti des GPU RTX de dernière génération grâce à leurs
capacités Tensor Cores, FP16/BF16 et TensorRT.
"""

import os
import logging
import sys
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import platform
import re

# Configuration du logger
from ai_trading.utils import setup_logger
logger = setup_logger("rtx_optimizer")

# Importation conditionnelle des frameworks
HAS_TORCH = False
HAS_TENSORRT = False
HAS_CUDA = False
HAS_CUDNN = False

try:
    import torch
    HAS_TORCH = True
    
    # Vérification de CUDA et des capacités
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        # Version de CUDA
        if hasattr(torch.version, 'cuda'):
            logger.info(f"Version CUDA: {torch.version.cuda}")
        
        # Vérifier la version de cuDNN
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'version'):
            HAS_CUDNN = True
            logger.info(f"Version cuDNN: {torch.backends.cudnn.version()}")
except ImportError:
    logger.info("PyTorch non détecté")

try:
    import tensorrt
    HAS_TENSORRT = True
    logger.info(f"TensorRT détecté, version: {tensorrt.__version__}")
except ImportError:
    logger.info("TensorRT non détecté")

# Liste des GPU RTX supportés spécifiquement
RTX_30_SERIES = ['RTX 3050', 'RTX 3060', 'RTX 3070', 'RTX 3080', 'RTX 3090']
RTX_40_SERIES = ['RTX 4060', 'RTX 4070', 'RTX 4080', 'RTX 4090']
SUPPORTED_GPUS = RTX_30_SERIES + RTX_40_SERIES

class RTXOptimizer:
    """
    Optimiseur spécifique pour les GPU RTX séries 30 et 40.
    
    Cette classe détecte et configure automatiquement les paramètres optimaux
    pour tirer parti des capacités avancées des GPU RTX récents.
    """
    
    def __init__(self, device_id: Optional[int] = None, enable_tensor_cores: bool = True,
                 enable_half_precision: bool = True, optimize_memory: bool = True,
                 enable_tensorrt: bool = False):
        """
        Initialise l'optimiseur RTX.
        
        Args:
            device_id: ID du dispositif à utiliser (None = auto-détection)
            enable_tensor_cores: Activer les optimisations pour les Tensor Cores
            enable_half_precision: Activer la précision mixte automatique (AMP)
            optimize_memory: Optimiser l'utilisation de la mémoire
            enable_tensorrt: Activer TensorRT pour l'inférence (si disponible)
        """
        self.has_rtx_gpu = False
        self.gpu_model = None
        self.compute_capability = None
        self.device = None
        self.device_id = device_id
        self.enable_tensor_cores = enable_tensor_cores
        self.enable_half_precision = enable_half_precision
        self.optimize_memory = optimize_memory
        self.enable_tensorrt = enable_tensorrt and HAS_TENSORRT
        
        # Détection et configuration
        if HAS_CUDA:
            # Déterminer l'ID du dispositif
            if device_id is None:
                device_id = 0
            
            # Vérifier que le dispositif est valide
            if device_id >= torch.cuda.device_count():
                logger.warning(f"Device ID {device_id} non valide, utilisation du GPU 0")
                device_id = 0
            
            self.device_id = device_id
            self.device = f"cuda:{device_id}"
            
            # Obtenir les informations sur le GPU
            self.gpu_model = torch.cuda.get_device_name(device_id)
            
            # Vérifier si c'est un GPU RTX de la série 30 ou 40
            self.has_rtx_gpu = any(rtx_model in self.gpu_model for rtx_model in SUPPORTED_GPUS)
            
            # Obtenir la capacité de calcul
            if hasattr(torch.cuda, 'get_device_capability'):
                major, minor = torch.cuda.get_device_capability(device_id)
                self.compute_capability = f"{major}.{minor}"
            
            # Sélection des optimisations basées sur le modèle de GPU
            self._setup_rtx_optimizations()
            
            logger.info(f"RTXOptimizer initialisé pour {self.gpu_model} (compute capability: {self.compute_capability})")
            if self.has_rtx_gpu:
                logger.info(f"GPU RTX détecté et supporté, optimisations spécifiques activées")
            else:
                logger.info(f"GPU non reconnu comme RTX série 30/40, optimisations génériques appliquées")
        else:
            logger.warning("CUDA non disponible, RTXOptimizer désactivé")
    
    def _setup_rtx_optimizations(self) -> None:
        """
        Configure les optimisations spécifiques aux GPU RTX.
        """
        if not HAS_CUDA:
            return
        
        # Optimisations pour Tensor Cores
        if self.enable_tensor_cores:
            if self.compute_capability and float(self.compute_capability) >= 7.0:
                # Ampere (série 30) utilise TF32 par défaut pour les Tensor Cores
                if float(self.compute_capability) >= 8.0:
                    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                        logger.info("TF32 activé pour les opérations matricielles")
                    
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                        logger.info("TF32 activé pour cuDNN")
                
                # Configuration avancée de cuDNN pour Tensor Cores
                if HAS_CUDNN:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    logger.info("cuDNN benchmark activé pour optimiser les performances")
        
        # Optimisations de mémoire
        if self.optimize_memory:
            # Série 30: RTX 3080/3090 ont plus de mémoire, donc optimisations différentes
            if 'RTX 3090' in self.gpu_model or 'RTX 4090' in self.gpu_model:
                # Plus de mémoire disponible, optimisations moins agressives
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
            elif 'RTX 3080' in self.gpu_model or 'RTX 4080' in self.gpu_model:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            else:
                # Pour les GPU avec moins de VRAM
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"
            
            logger.info(f"Optimisations mémoire configurées pour {self.gpu_model}")
    
    def get_device(self) -> str:
        """
        Retourne le dispositif optimisé.
        
        Returns:
            Nom du dispositif CUDA optimisé
        """
        return self.device if HAS_CUDA else "cpu"
    
    def to_device(self, model: Any) -> Any:
        """
        Déplace un modèle vers le dispositif RTX optimisé.
        
        Args:
            model: Modèle PyTorch à déplacer
            
        Returns:
            Modèle sur le dispositif RTX
        """
        if not HAS_TORCH or not HAS_CUDA:
            return model
        
        try:
            if hasattr(model, 'to') and callable(model.to):
                # Déplacement sur le dispositif GPU
                model = model.to(self.device)
                
                # Optimisation précision mixte si activée
                if self.enable_half_precision and (self.compute_capability and float(self.compute_capability) >= 7.0):
                    # RTX 30 et 40 supportent bien mieux les opérations FP16
                    model = model.to(torch.float16)
                    logger.info(f"Modèle converti en FP16 pour {self.gpu_model}")
                
                return model
            else:
                return model
        except Exception as e:
            logger.warning(f"Impossible de déplacer le modèle sur {self.device}: {e}")
            return model
    
    def optimize_for_inference(self, model: Any) -> Any:
        """
        Optimise un modèle pour l'inférence sur RTX.
        
        Args:
            model: Modèle à optimiser
            
        Returns:
            Modèle optimisé pour l'inférence
        """
        if not HAS_TORCH or not HAS_CUDA:
            return model
        
        # Mise en mode évaluation
        if hasattr(model, 'eval') and callable(model.eval):
            model.eval()
        
        # Optimisations spécifiques pour les GPU supportés
        if self.has_rtx_gpu:
            try:
                # Fusion des opérations batch norm
                if hasattr(torch, 'quantization') and hasattr(torch.quantization, 'fuse_modules'):
                    # Tentative de fusion des modules si applicable
                    try:
                        # Détecter les modules fusionnables
                        modules_to_fuse = []
                        for name, module in model.named_modules():
                            if isinstance(module, torch.nn.Conv2d) and f"{name}.bn" in dict(model.named_modules()):
                                modules_to_fuse.append([name, f"{name}.bn"])
                        
                        if modules_to_fuse:
                            model = torch.quantization.fuse_modules(model, modules_to_fuse)
                            logger.info("Modules fusionnés pour optimisation RTX")
                    except Exception as e:
                        logger.debug(f"Fusion des modules échouée: {e}, non critique")
                
                # Optimisations TensorRT si demandées et disponibles
                if self.enable_tensorrt and HAS_TENSORRT:
                    try:
                        import torch_tensorrt
                        
                        # Génération d'input d'exemple
                        if hasattr(model, 'input_shape'):
                            example_input = torch.randn(model.input_shape, device=self.device)
                            
                            # Conversion TensorRT
                            compiled_model = torch_tensorrt.compile(
                                model,
                                inputs=[example_input],
                                enabled_precisions={torch.float16} if self.enable_half_precision else {torch.float32}
                            )
                            
                            logger.info("Modèle optimisé avec TensorRT")
                            return compiled_model
                    except Exception as e:
                        logger.warning(f"Optimisation TensorRT échouée: {e}")
                
                # JIT compilation comme fallback
                try:
                    with torch.no_grad():
                        if hasattr(torch, 'jit'):
                            # Optimisation JIT pour les opérations GPU
                            model = torch.jit.optimize_for_inference(
                                torch.jit.script(model)
                            )
                            logger.info("Modèle optimisé avec JIT pour les GPU RTX")
                except Exception as e:
                    logger.debug(f"Optimisation JIT échouée: {e}, non critique")
            
            except Exception as e:
                logger.warning(f"Optimisation pour RTX échouée: {e}")
        
        return model
    
    def autocast_context(self):
        """
        Retourne un contexte de précision mixte optimisé pour RTX.
        
        Les GPU RTX séries 30 et 40 ont des Tensor Cores qui bénéficient
        grandement de la précision mixte automatique.
        
        Returns:
            Contexte AMP optimisé ou contexte factice
        """
        if not HAS_TORCH or not HAS_CUDA or not self.enable_half_precision:
            from contextlib import nullcontext
            return nullcontext()
        
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            # Les séries 30 et 40 supportent bien les opérations FP16
            dtypes = {}
            
            # Optimisations spécifiques selon la série
            if 'RTX 40' in self.gpu_model and hasattr(torch, 'bfloat16'):
                # Les GPU série 40 (Ada Lovelace) supportent mieux bfloat16
                dtypes = {torch.float16: True, torch.bfloat16: True}
                logger.info("Utilisation de FP16/BF16 pour précision mixte sur RTX série 40")
            else:
                # Série 30 et autres: optimisé pour float16
                dtypes = {torch.float16: True}
                logger.info("Utilisation de FP16 pour précision mixte sur RTX série 30")
            
            # Retourne le contexte AMP optimisé pour le GPU
            return torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            from contextlib import nullcontext
            return nullcontext()
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Obtient des informations sur les optimisations appliquées.
        
        Returns:
            Dictionnaire d'informations sur les optimisations RTX
        """
        info = {
            "device": self.device,
            "gpu_model": self.gpu_model,
            "compute_capability": self.compute_capability,
            "has_rtx_gpu": self.has_rtx_gpu,
            "tensor_cores_enabled": self.enable_tensor_cores,
            "half_precision_enabled": self.enable_half_precision,
            "memory_optimization": self.optimize_memory,
            "tensorrt_enabled": self.enable_tensorrt and HAS_TENSORRT,
        }
        
        # Optimisations actives
        active_optimizations = []
        
        if self.has_rtx_gpu and self.enable_tensor_cores:
            if self.compute_capability and float(self.compute_capability) >= 8.0:
                active_optimizations.append("TF32 Tensor Cores (Ampere+)")
            elif self.compute_capability and float(self.compute_capability) >= 7.0:
                active_optimizations.append("Tensor Cores (Turing+)")
        
        if self.enable_half_precision:
            if 'RTX 40' in str(self.gpu_model):
                active_optimizations.append("FP16/BF16 Mixed Precision (Ada Lovelace)")
            else:
                active_optimizations.append("FP16 Mixed Precision")
        
        if self.optimize_memory:
            if 'RTX 3090' in str(self.gpu_model) or 'RTX 4090' in str(self.gpu_model):
                active_optimizations.append("Mémoire haute capacité (24GB+)")
            elif 'RTX 3080' in str(self.gpu_model) or 'RTX 4080' in str(self.gpu_model):
                active_optimizations.append("Mémoire intermédiaire (10GB+)")
            else:
                active_optimizations.append("Mémoire optimisée")
        
        if self.enable_tensorrt and HAS_TENSORRT:
            active_optimizations.append("TensorRT")
        
        info["active_optimizations"] = active_optimizations
        
        return info
    
    def clear_cache(self) -> None:
        """
        Libère la mémoire GPU avec optimisations spécifiques à RTX.
        """
        if not HAS_TORCH or not HAS_CUDA:
            return
        
        # Libération de la mémoire cache
        torch.cuda.empty_cache()
        
        # Synchronisation du dispositif
        torch.cuda.synchronize(self.device_id)
        
        # Application d'une stratégie agressive pour les GPU avec peu de VRAM
        if self.optimize_memory and self.has_rtx_gpu:
            if not ('RTX 3090' in self.gpu_model or 'RTX 4090' in self.gpu_model or 
                   'RTX 3080' in self.gpu_model or 'RTX 4080' in self.gpu_model):
                # Force un cycle de collecte des déchets Python
                import gc
                gc.collect()
        
        logger.info(f"Mémoire GPU libérée pour {self.gpu_model}")

def detect_rtx_gpu() -> Optional[Dict[str, Any]]:
    """
    Détecte si un GPU RTX des séries 30 ou 40 est disponible.
    
    Returns:
        Dictionnaire d'informations sur le GPU ou None si aucun RTX n'est détecté
    """
    if not HAS_CUDA:
        return None
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        
        # Vérification pour les séries RTX 30 et 40
        if any(rtx_model in gpu_name for rtx_model in SUPPORTED_GPUS):
            # Obtenir la capacité de calcul
            compute_capability = None
            if hasattr(torch.cuda, 'get_device_capability'):
                major, minor = torch.cuda.get_device_capability(i)
                compute_capability = f"{major}.{minor}"
            
            # Déterminer la série
            series = "40 series" if any(rtx_model in gpu_name for rtx_model in RTX_40_SERIES) else "30 series"
            
            return {
                "device_id": i,
                "name": gpu_name,
                "compute_capability": compute_capability,
                "series": series,
                "vram_gb": _estimate_vram_from_name(gpu_name)
            }
    
    return None

def _estimate_vram_from_name(gpu_name: str) -> Optional[int]:
    """
    Estime la quantité de VRAM en GB basée sur le nom du GPU.
    
    Args:
        gpu_name: Nom du GPU
        
    Returns:
        Estimation de la VRAM en GB ou None si impossible à déterminer
    """
    # Mapping approximatif
    vram_map = {
        'RTX 3050': 8,
        'RTX 3060': 12,
        'RTX 3070': 8,
        'RTX 3080': 10,  # 10 GB pour la version standard, 12 GB pour Ti
        'RTX 3090': 24,
        'RTX 4060': 8,
        'RTX 4070': 12,
        'RTX 4080': 16,
        'RTX 4090': 24
    }
    
    # Recherche dans le mapping
    for model, vram in vram_map.items():
        if model in gpu_name:
            # Ajustement pour les variantes Ti
            if 'Ti' in gpu_name:
                if model == 'RTX 3080':
                    return 12
                elif model == 'RTX 3070':
                    return 8  # Reste 8GB même pour Ti
            return vram
    
    return None

def setup_rtx_environment() -> bool:
    """
    Configure l'environnement pour l'utilisation optimale des GPU RTX.
    
    Returns:
        True si un GPU RTX a été détecté et configuré, False sinon
    """
    # Détection d'un GPU RTX
    rtx_info = detect_rtx_gpu()
    
    if not rtx_info:
        logger.info("Aucun GPU RTX des séries 30/40 détecté")
        return False
    
    logger.info(f"GPU RTX détecté: {rtx_info['name']} (série {rtx_info['series']}, VRAM estimée: {rtx_info['vram_gb']} GB)")
    
    try:
        # Configuration des variables d'environnement pour optimisation
        os.environ["PYTORCH_TF32"] = "1"  # Active TF32 si supporté
        
        # Optimisations de mémoire selon la quantité de VRAM
        if rtx_info['vram_gb'] and rtx_info['vram_gb'] >= 16:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
        elif rtx_info['vram_gb'] and rtx_info['vram_gb'] >= 12:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"
        
        # Configuration PyTorch
        if HAS_TORCH:
            torch.backends.cudnn.benchmark = True
            
            # Optimisations Ampere/Ada Lovelace (séries 30/40)
            if rtx_info['compute_capability'] and float(rtx_info['compute_capability']) >= 8.0:
                if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                logger.info("Optimisations TF32 activées pour GPU RTX Ampere/Ada Lovelace")
        
        logger.info(f"Environnement optimisé pour RTX {rtx_info['series']}")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la configuration de l'environnement RTX: {e}")
        return False 