"""
Module pour l'optimisation des modèles de langage (LLM).

Ce module fournit des outils pour:
- Quantification INT8/INT4 des modèles pour accélérer l'inférence
- Pruning des modèles pour réduire leur taille sans perdre en précision
- Distillation de modèles pour créer des versions plus légères et rapides
"""

import gc
import logging
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationType(str, Enum):
    """Types de quantification disponibles."""
    
    INT8 = "int8"
    INT4 = "int4"
    DYNAMIC_INT8 = "dynamic_int8"
    NONE = "none"


class ModelOptimizer:
    """
    Classe pour optimiser les modèles de langage (LLM) afin de réduire 
    leur taille et améliorer leur efficacité.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialise l'optimiseur de modèles.
        
        Args:
            cache_dir: Répertoire de cache pour les modèles
            device: Appareil sur lequel exécuter le modèle ('cpu', 'cuda', etc.)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vérifier si CUDA est disponible
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available and device == "cuda":
            logger.warning("CUDA demandé mais non disponible. Utilisation du CPU à la place.")
            self.device = "cpu"
        
        if self.cuda_available:
            logger.info(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        
        logger.info(f"Initialisation de l'optimiseur de modèles (device={self.device})")
    
    def load_model(
        self,
        model_name_or_path: str,
        model_type: str = "causal_lm",
        quantization: QuantizationType = QuantizationType.NONE,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Charge un modèle avec éventuellement de la quantification.
        
        Args:
            model_name_or_path: Nom ou chemin du modèle
            model_type: Type de modèle ('causal_lm' ou 'sequence_classification')
            quantization: Type de quantification à appliquer
            
        Returns:
            Tuple contenant le modèle et le tokenizer
        """
        logger.info(f"Chargement du modèle {model_name_or_path}")
        
        # Charger la configuration et le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=self.cache_dir,
            use_fast=True,
        )
        
        # Préparer les paramètres de quantification
        if quantization != QuantizationType.NONE:
            # Vérifier si bitsandbytes est installé pour la quantification
            try:
                import bitsandbytes as bnb
                logger.info(f"Bitsandbytes version {bnb.__version__} trouvé pour la quantification")
            except ImportError:
                logger.warning("La bibliothèque bitsandbytes n'est pas installée. Désactivation de la quantification.")
                quantization = QuantizationType.NONE
        
        # Paramètres de chargement du modèle
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        # Appliquer la quantification
        if quantization == QuantizationType.INT8:
            try:
                model_kwargs["load_in_8bit"] = True
                logger.info("Chargement du modèle avec quantification INT8")
            except Exception as e:
                logger.error(f"Erreur lors de la quantification INT8: {e}")
                model_kwargs.pop("load_in_8bit", None)
        elif quantization == QuantizationType.INT4:
            try:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                model_kwargs["bnb_4bit_use_double_quant"] = True
                logger.info("Chargement du modèle avec quantification INT4")
            except Exception as e:
                logger.error(f"Erreur lors de la quantification INT4: {e}")
                model_kwargs.pop("load_in_4bit", None)
        elif quantization == QuantizationType.DYNAMIC_INT8:
            # La quantification dynamique sera appliquée après le chargement
            logger.info("La quantification dynamique INT8 sera appliquée après le chargement")
        
        # Charger le modèle selon son type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
        elif model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
        
        # Appliquer la quantification dynamique au besoin
        if quantization == QuantizationType.DYNAMIC_INT8:
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Quantification dynamique INT8 appliquée")
            except Exception as e:
                logger.error(f"Erreur lors de la quantification dynamique: {e}")
        
        # Déplacer le modèle sur le bon appareil si non déjà fait
        if self.device != "cuda" or "device_map" not in model_kwargs:
            model = model.to(self.device)
        
        # Activer le mode évaluation
        model.eval()
        
        return model, tokenizer
    
    def prune_model(
        self,
        model: PreTrainedModel,
        sparsity: float = 0.3,
        method: str = "magnitude",
        save_path: Optional[Union[str, Path]] = None,
    ) -> PreTrainedModel:
        """
        Élague (prune) un modèle pour réduire sa taille sans perdre en précision.
        
        Args:
            model: Modèle à élaguer
            sparsity: Proportion des poids à élaguer (0.0 à 1.0)
            method: Méthode d'élagage ('magnitude', 'random', 'structured')
            save_path: Chemin pour sauvegarder le modèle élagué
            
        Returns:
            Modèle élagué
        """
        logger.info(f"Élagage du modèle avec sparsité {sparsity:.2f}")
        
        # Vérifier si PyTorch prune est disponible
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            logger.error("PyTorch utils.prune non disponible. Impossible d'élaguer le modèle.")
            return model
        
        # Sauvegarder les valeurs initiales de paramètres pour comparer plus tard
        original_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Identifier les modules à élaguer (on se concentre sur les couches Linear)
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_to_prune.append((module, "weight"))
        
        logger.info(f"Élagage de {len(modules_to_prune)} modules")
        
        # Appliquer l'élagage selon la méthode choisie
        if method == "magnitude":
            # Élagage basé sur la magnitude (supprime les poids les plus faibles)
            for module, param_name in modules_to_prune:
                prune.l1_unstructured(module, name=param_name, amount=sparsity)
        elif method == "random":
            # Élagage aléatoire
            for module, param_name in modules_to_prune:
                prune.random_unstructured(module, name=param_name, amount=sparsity)
        elif method == "structured":
            # Élagage structuré (supprime des neurones entiers)
            for module, param_name in modules_to_prune:
                prune.ln_structured(module, name=param_name, amount=sparsity, n=2, dim=0)
        else:
            logger.warning(f"Méthode d'élagage inconnue: {method}. Utilisation de 'magnitude'")
            for module, param_name in modules_to_prune:
                prune.l1_unstructured(module, name=param_name, amount=sparsity)
        
        # Rendre l'élagage permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        # Calculer la taille après élagage
        pruned_size = sum(
            torch.sum(p != 0).item() for p in model.parameters() if p.requires_grad
        )
        compression_ratio = original_size / (pruned_size + 1e-8)  # Éviter division par zéro
        
        logger.info(f"Taille originale: {original_size:,} paramètres")
        logger.info(f"Taille après élagage: {pruned_size:,} paramètres")
        logger.info(f"Ratio de compression: {compression_ratio:.2f}x")
        
        # Sauvegarder le modèle élagué si demandé
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            logger.info(f"Modèle élagué sauvegardé à {save_path}")
        
        return model
    
    def distill_model(
        self,
        teacher_model: PreTrainedModel,
        teacher_tokenizer: PreTrainedTokenizer,
        student_model_name: str,
        training_data: Union[str, List[str]],
        output_dir: Union[str, Path],
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Distille un modèle enseignant vers un modèle élève plus petit.
        
        Args:
            teacher_model: Modèle enseignant
            teacher_tokenizer: Tokenizer du modèle enseignant
            student_model_name: Nom du modèle élève à initialiser
            training_data: Données d'entraînement (texte brut ou liste de textes)
            output_dir: Répertoire de sortie pour le modèle distillé
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
            learning_rate: Taux d'apprentissage
            
        Returns:
            Tuple contenant le modèle élève et son tokenizer
        """
        logger.info(f"Distillation du modèle {teacher_model.config._name_or_path} vers {student_model_name}")
        
        # Vérifier si transformers et datasets sont disponibles
        try:
            from transformers import Trainer, TrainingArguments
            from transformers import DataCollatorForLanguageModeling
        except ImportError:
            logger.error("Bibliothèques datasets ou transformers non disponibles. Impossible de distiller le modèle.")
            return teacher_model, teacher_tokenizer
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le modèle élève
        try:
            student_config = AutoConfig.from_pretrained(student_model_name)
            student_config.hidden_dropout_prob = 0.1
            student_config.attention_probs_dropout_prob = 0.1
            
            # Réduire la taille si possible
            original_hidden_size = student_config.hidden_size
            if hasattr(student_config, "hidden_size") and student_config.hidden_size > 512:
                reduction_factor = 2
                student_config.hidden_size = student_config.hidden_size // reduction_factor
                if hasattr(student_config, "intermediate_size"):
                    student_config.intermediate_size = student_config.intermediate_size // reduction_factor
                logger.info(f"Taille du modèle élève réduite: {original_hidden_size} → {student_config.hidden_size}")
            
            # Créer le modèle élève et utiliser le tokenizer de l'enseignant
            student_model = AutoModelForCausalLM.from_config(student_config)
            student_tokenizer = teacher_tokenizer
            
            logger.info(f"Modèle élève initialisé: {student_model.config.hidden_size} dimensions cachées")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle élève: {e}")
            return teacher_model, teacher_tokenizer
        
        # Préparer les données d'entraînement
        if isinstance(training_data, str):
            # Si c'est un chemin de fichier
            if os.path.exists(training_data):
                with open(training_data, 'r', encoding='utf-8') as f:
                    texts = f.readlines()
            else:
                # Considérer comme un texte brut
                texts = [training_data]
        else:
            texts = training_data
        
        # Créer un dataset à partir des textes
        dataset = Dataset.from_dict({
            "text": texts
        })
        
        # Tokeniser les données
        def tokenize_function(examples):
            return teacher_tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Créer un collator pour le langage modélisé
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=teacher_tokenizer,
            mlm=False
        )
        
        # Définir les arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=100,
            save_steps=1000,
            save_total_limit=2,
            fp16=self.device == "cuda",
        )
        
        # Définir la fonction de perte de distillation
        def compute_loss(model, inputs, teacher_model, alpha=0.5, temperature=2.0):
            # Obtenir les logits du modèle élève
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits
            
            # Obtenir les logits du modèle enseignant (sans calcul de gradient)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Perte de distillation
            loss_function = torch.nn.KLDivLoss(reduction="batchmean")
            distillation_loss = loss_function(
                torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
                torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
            ) * (temperature ** 2)
            
            # Perte standard
            standard_loss = student_outputs.loss
            
            # Combiner les deux pertes
            loss = alpha * distillation_loss + (1 - alpha) * standard_loss
            return loss
        
        # Créer un entraîneur personnalisé pour la distillation
        class DistillationTrainer(Trainer):
            def __init__(self, *args, teacher_model=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
            
            def compute_loss(self, model, inputs, return_outputs=False):
                loss = compute_loss(model, inputs, self.teacher_model)
                return (loss, None) if return_outputs else loss
        
        # Initialiser l'entraîneur
        trainer = DistillationTrainer(
            model=student_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            teacher_model=teacher_model
        )
        
        # Entraîner le modèle
        try:
            logger.info("Début de la distillation...")
            trainer.train()
            logger.info("Distillation terminée")
            
            # Sauvegarder le modèle final
            student_model.save_pretrained(output_dir)
            student_tokenizer.save_pretrained(output_dir)
            logger.info(f"Modèle distillé sauvegardé à {output_dir}")
            
            return student_model, student_tokenizer
        except Exception as e:
            logger.error(f"Erreur lors de la distillation: {e}")
            return teacher_model, teacher_tokenizer
    
    def benchmark_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        input_texts: List[str],
        batch_size: int = 1,
        num_runs: int = 10,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Évalue la performance d'un modèle.
        
        Args:
            model: Modèle à évaluer
            tokenizer: Tokenizer associé au modèle
            input_texts: Textes d'entrée pour le benchmark
            batch_size: Taille du batch
            num_runs: Nombre d'exécutions pour moyenner les résultats
            max_length: Longueur maximale de séquence
            
        Returns:
            Dictionnaire des métriques de performance
        """
        logger.info(f"Benchmark du modèle ({num_runs} exécutions)")
        
        import time
        
        # S'assurer que le modèle est en mode évaluation
        model.eval()
        
        # Préparer les données d'entrée
        all_tokens = tokenizer(
            input_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Mesurer l'empreinte mémoire
        memory_before = torch.cuda.memory_allocated() if self.device == "cuda" else None
        
        # Mesurer le temps d'inférence
        inference_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                
                # Inférence par batch
                if batch_size > 1:
                    num_samples = len(input_texts)
                    for i in range(0, num_samples, batch_size):
                        batch_end = min(i + batch_size, num_samples)
                        batch_input = {
                            k: v[i:batch_end] for k, v in all_tokens.items()
                        }
                        model(**batch_input)
                else:
                    # Inférence simple échantillon par échantillon
                    for i in range(len(input_texts)):
                        single_input = {
                            k: v[i:i+1] for k, v in all_tokens.items()
                        }
                        model(**single_input)
                
                inference_times.append(time.time() - start_time)
        
        # Calculer l'utilisation mémoire
        memory_after = torch.cuda.memory_allocated() if self.device == "cuda" else None
        memory_used = (memory_after - memory_before) if memory_before is not None else None
        
        # Calculer les statistiques
        avg_time = sum(inference_times) / len(inference_times)
        avg_time_per_sample = avg_time / len(input_texts)
        throughput = len(input_texts) / avg_time
        
        # Calculer la taille du modèle en Mo
        model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Résultats du benchmark
        results = {
            "model_name": model.config._name_or_path,
            "average_inference_time": avg_time,
            "average_time_per_sample": avg_time_per_sample,
            "throughput_samples_per_second": throughput,
            "model_size_mb": model_size_mb,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": self.device,
        }
        
        if memory_used is not None:
            results["memory_used_bytes"] = memory_used
            results["memory_used_mb"] = memory_used / (1024 * 1024)
        
        # Afficher un résumé
        logger.info(f"Résultats du benchmark:")
        logger.info(f"  Taille du modèle: {model_size_mb:.2f} Mo")
        logger.info(f"  Temps d'inférence moyen: {avg_time:.4f} secondes")
        logger.info(f"  Temps moyen par échantillon: {avg_time_per_sample:.4f} secondes")
        logger.info(f"  Débit: {throughput:.2f} échantillons/seconde")
        
        return results


# Fonctions utilitaires
def get_memory_info():
    """Récupère des informations sur l'utilisation de la mémoire."""
    memory_info = {}
    
    # CUDA
    if torch.cuda.is_available():
        memory_info["cuda"] = {}
        for i in range(torch.cuda.device_count()):
            device_info = {}
            device_info["total"] = torch.cuda.get_device_properties(i).total_memory
            device_info["allocated"] = torch.cuda.memory_allocated(i)
            device_info["cached"] = torch.cuda.memory_reserved(i)
            device_info["free"] = device_info["total"] - device_info["allocated"]
            memory_info["cuda"][f"device_{i}"] = device_info
    
    # CPU (avec psutil si disponible)
    try:
        import psutil
        memory_info["cpu"] = {}
        memory_info["cpu"]["total"] = psutil.virtual_memory().total
        memory_info["cpu"]["available"] = psutil.virtual_memory().available
        memory_info["cpu"]["used"] = psutil.virtual_memory().used
        memory_info["cpu"]["percent"] = psutil.virtual_memory().percent
    except ImportError:
        pass
    
    return memory_info


def free_gpu_memory():
    """Libère la mémoire GPU."""
    if torch.cuda.is_available():
        # Libération de la mémoire CUDA
        torch.cuda.empty_cache()
        # Collecte des déchets Python
        gc.collect()
        
        logger.info("Mémoire GPU libérée")
        return True
    
    return False


def print_model_info(model: PreTrainedModel):
    """Affiche des informations sur le modèle."""
    model_info = {
        "name": model.config._name_or_path,
        "type": model.__class__.__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "size_MB": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
    }
    
    # Ajouter des informations spécifiques au modèle
    if hasattr(model.config, "hidden_size"):
        model_info["hidden_size"] = model.config.hidden_size
    if hasattr(model.config, "num_hidden_layers"):
        model_info["num_layers"] = model.config.num_hidden_layers
    if hasattr(model.config, "num_attention_heads"):
        model_info["num_heads"] = model.config.num_attention_heads
    
    logger.info(f"Informations sur le modèle:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    return model_info 