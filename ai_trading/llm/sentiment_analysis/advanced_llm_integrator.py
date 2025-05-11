"""
Module d'intégration de modèles LLM avancés pour l'analyse de sentiment crypto.
Ce module permet d'utiliser des modèles plus récents et spécialisés pour l'analyse
du sentiment dans le domaine des cryptomonnaies.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
# Importation conditionnelle de peft pour gérer les problèmes de compatibilité
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except (ImportError, AttributeError):
    PEFT_AVAILABLE = False
    # Classes factices pour éviter les erreurs
    class PeftModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("La bibliothèque PEFT n'est pas disponible ou incompatible")
    
    class PeftConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("La bibliothèque PEFT n'est pas disponible ou incompatible")

import numpy as np
from .sentiment_tools import SentimentCache, text_hash

logger = logging.getLogger(__name__)

class AdvancedLLMIntegrator:
    """
    Intégrateur de modèles LLM avancés pour l'analyse de sentiment crypto.
    
    Cette classe permet d'utiliser des modèles de langage plus récents et
    spécialisés pour l'analyse du sentiment dans le domaine des cryptomonnaies.
    """
    
    def __init__(self, 
                 cache_dir: str = "ai_trading/info_retour/llm_cache",
                 use_gpu: bool = torch.cuda.is_available(),
                 quantize: bool = True):
        """
        Initialise l'intégrateur de modèles LLM avancés.
        
        Args:
            cache_dir: Répertoire de cache pour les modèles et résultats
            use_gpu: Utiliser le GPU si disponible
            quantize: Quantifier les modèles pour réduire l'empreinte mémoire
        """
        self.cache = SentimentCache(cache_dir)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.quantize = quantize
        self.models = {}
        self.tokenizers = {}
        
        # Configuration pour la quantification 8-bit/4-bit
        self.quantization_config = None
        if self.quantize:
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        logger.info(f"AdvancedLLMIntegrator initialisé sur {self.device}")
    
    def load_model(self, 
                  model_name: str, 
                  model_type: str = "sentiment",
                  use_peft: bool = False,
                  peft_model_id: Optional[str] = None) -> None:
        """
        Charge un modèle LLM.
        
        Args:
            model_name: Nom du modèle à charger (HuggingFace model ID)
            model_type: Type de modèle ('sentiment', 'generation', 'qa')
            use_peft: Utiliser un modèle fine-tuné avec PEFT
            peft_model_id: ID du modèle PEFT si use_peft=True
        """
        try:
            # Chargement du tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            
            # Chargement du modèle selon son type
            if model_type == "sentiment":
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    quantization_config=self.quantization_config,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:  # modèle génératif par défaut
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=self.quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            # Application du modèle PEFT si demandé
            if use_peft and peft_model_id and PEFT_AVAILABLE:
                model = PeftModel.from_pretrained(base_model, peft_model_id)
                logger.info(f"Modèle PEFT {peft_model_id} appliqué sur {model_name}")
            else:
                model = base_model
            
            # Stockage du modèle
            self.models[model_name] = model
            logger.info(f"Modèle {model_name} chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            raise
    
    def analyze_sentiment(self, 
                         text: str, 
                         model_name: Optional[str] = None,
                         crypto_context: Optional[Dict] = None) -> Dict:
        """
        Analyse le sentiment d'un texte avec un modèle spécialisé crypto.
        
        Args:
            text: Texte à analyser
            model_name: Nom du modèle à utiliser (si None, utilise le modèle par défaut)
            crypto_context: Contexte crypto pour améliorer l'analyse
            
        Returns:
            Dict: Résultats de l'analyse de sentiment
        """
        # Utilisation du cache si disponible
        cache_key = f"sentiment_{text_hash(text)}_{model_name or 'default'}"
        cached_result = self.cache.load(cache_key)
        if cached_result:
            return cached_result
        
        # Sélection du modèle
        if not model_name:
            model_name = "ElKulako/cryptobert"  # Modèle par défaut spécialisé crypto
        
        # Chargement du modèle s'il n'est pas déjà chargé
        if model_name not in self.models:
            self.load_model(model_name, model_type="sentiment")
        
        # Préparation du texte avec contexte crypto si disponible
        prepared_text = self._prepare_text_with_context(text, crypto_context)
        
        # Analyse du sentiment
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Création du pipeline d'analyse
        sentiment_pipeline = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True
        )
        
        # Analyse
        results = sentiment_pipeline(prepared_text)
        
        # Post-traitement des résultats
        processed_results = self._process_sentiment_results(results, model_name)
        
        # Mise en cache des résultats
        self.cache.save(cache_key, processed_results)
        
        return processed_results
    
    def generate_crypto_analysis(self, 
                               prompt: str, 
                               model_name: Optional[str] = None,
                               max_length: int = 256) -> str:
        """
        Génère une analyse crypto basée sur un prompt.
        
        Args:
            prompt: Texte d'entrée pour la génération
            model_name: Nom du modèle à utiliser (si None, utilise le modèle par défaut)
            max_length: Longueur maximale de la génération
            
        Returns:
            str: Texte généré
        """
        # Sélection du modèle
        if not model_name:
            model_name = "ybelkada/falcon-7b-sharded-bf16"  # Modèle par défaut
        
        # Chargement du modèle s'il n'est pas déjà chargé
        if model_name not in self.models:
            self.load_model(model_name, model_type="generation")
        
        # Création du pipeline de génération
        generation_pipeline = pipeline(
            "text-generation",
            model=self.models[model_name],
            tokenizer=self.tokenizers[model_name],
            device=0 if self.device == "cuda" else -1,
            max_length=max_length
        )
        
        # Génération
        result = generation_pipeline(prompt)[0]["generated_text"]
        
        # Extraction du texte généré (sans le prompt)
        generated_text = result[len(prompt):].strip()
        
        return generated_text
    
    def fine_tune_for_crypto(self, 
                           model_name: str,
                           training_data: List[Dict],
                           output_dir: str = "ai_trading/info_retour/fine_tuned_models",
                           epochs: int = 3) -> str:
        """
        Fine-tune un modèle pour l'analyse de sentiment crypto.
        
        Args:
            model_name: Nom du modèle à fine-tuner
            training_data: Données d'entraînement
            output_dir: Répertoire de sortie pour le modèle fine-tuné
            epochs: Nombre d'époques d'entraînement
            
        Returns:
            str: Chemin vers le modèle fine-tuné
        """
        if not PEFT_AVAILABLE:
            raise ImportError("La bibliothèque PEFT n'est pas disponible ou incompatible")
            
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Création du dataset
        dataset = Dataset.from_list(training_data)
        
        # Chargement du modèle s'il n'est pas déjà chargé
        if model_name not in self.models:
            self.load_model(model_name, model_type="sentiment")
        
        # Configuration PEFT (Parameter-Efficient Fine-Tuning)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        
        # Application de la configuration PEFT
        peft_model = get_peft_model(self.models[model_name], peft_config)
        
        # Configuration de l'entraînement
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            push_to_hub=False,
        )
        
        # Préparation de la fonction de tokenization
        tokenizer = self.tokenizers[model_name]
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        
        # Tokenization du dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Création du Trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        # Entraînement
        trainer.train()
        
        # Sauvegarde du modèle
        output_path = f"{output_dir}/{model_name.split('/')[-1]}-crypto-finetuned"
        trainer.save_model(output_path)
        
        return output_path
    
    def optimize_prompts(self, 
                       base_prompt: str,
                       test_cases: List[Dict],
                       model_name: Optional[str] = None) -> str:
        """
        Optimise les prompts pour l'analyse crypto.
        
        Args:
            base_prompt: Prompt de base à optimiser
            test_cases: Cas de test pour évaluer les prompts
            model_name: Nom du modèle à utiliser
            
        Returns:
            str: Prompt optimisé
        """
        if not model_name:
            model_name = "gpt2"  # Modèle par défaut pour l'optimisation
        
        # Chargement du modèle s'il n'est pas déjà chargé
        if model_name not in self.models:
            self.load_model(model_name, model_type="generation")
        
        # Variantes de prompts à tester
        prompt_variants = [
            base_prompt,
            f"En tant qu'expert en cryptomonnaies, {base_prompt}",
            f"Analyse de sentiment crypto: {base_prompt}",
            f"Contexte: marché des cryptomonnaies. {base_prompt}"
        ]
        
        best_score = 0
        best_prompt = base_prompt
        
        # Test de chaque variante
        for prompt in prompt_variants:
            score = 0
            
            for test_case in test_cases:
                input_text = test_case["input"]
                expected_output = test_case["expected"]
                
                full_prompt = f"{prompt} {input_text}"
                generated = self.generate_crypto_analysis(full_prompt, model_name)
                
                # Calcul simple de similarité
                if expected_output.lower() in generated.lower():
                    score += 1
            
            # Mise à jour du meilleur prompt
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        return best_prompt
    
    def _prepare_text_with_context(self, text: str, context: Optional[Dict] = None) -> str:
        """
        Prépare le texte avec le contexte crypto pour améliorer l'analyse.
        
        Args:
            text: Texte à analyser
            context: Contexte crypto (prix, tendances, etc.)
            
        Returns:
            str: Texte préparé avec contexte
        """
        if not context:
            return text
        
        # Construction du contexte
        context_str = "Contexte: "
        
        if "price" in context:
            context_str += f"Prix: {context['price']}. "
        
        if "trend" in context:
            context_str += f"Tendance: {context['trend']}. "
        
        if "market_cap" in context:
            context_str += f"Capitalisation: {context['market_cap']}. "
        
        if "volume" in context:
            context_str += f"Volume: {context['volume']}. "
        
        # Combinaison du contexte et du texte
        return f"{context_str}\n\n{text}"
    
    def _process_sentiment_results(self, results: List, model_name: str) -> Dict:
        """
        Traite les résultats bruts de l'analyse de sentiment.
        
        Args:
            results: Résultats bruts de l'analyse
            model_name: Nom du modèle utilisé
            
        Returns:
            Dict: Résultats traités
        """
        # Structure de base des résultats
        processed = {
            "model": model_name,
            "sentiment": {},
            "overall_sentiment": 0,
            "confidence": 0
        }
        
        # Traitement selon le modèle
        if "cryptobert" in model_name.lower():
            # Traitement spécifique pour CryptoBERT
            for item in results[0]:
                label = item["label"].lower()
                score = item["score"]
                
                # Conversion des labels en sentiments
                if "positive" in label or "bullish" in label:
                    processed["sentiment"]["positive"] = score
                    processed["overall_sentiment"] += score
                elif "negative" in label or "bearish" in label:
                    processed["sentiment"]["negative"] = score
                    processed["overall_sentiment"] -= score
                else:
                    processed["sentiment"]["neutral"] = score
            
            # Normalisation du sentiment global entre -1 et 1
            processed["overall_sentiment"] = np.clip(processed["overall_sentiment"], -1, 1)
            
            # Calcul de la confiance (max des scores)
            processed["confidence"] = max([s for s in processed["sentiment"].values()])
            
        else:
            # Traitement générique pour les autres modèles
            for item in results[0]:
                label = item["label"].lower()
                score = item["score"]
                
                processed["sentiment"][label] = score
                
                # Calcul du sentiment global pondéré
                if "1" in label or "positive" in label:
                    processed["overall_sentiment"] += score
                elif "0" in label or "negative" in label:
                    processed["overall_sentiment"] -= score
            
            # Normalisation
            processed["overall_sentiment"] = np.clip(processed["overall_sentiment"], -1, 1)
            processed["confidence"] = max([s for s in processed["sentiment"].values()])
        
        return processed


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation de l'intégrateur
    llm_integrator = AdvancedLLMIntegrator()
    
    # Analyse de sentiment avec CryptoBERT
    result = llm_integrator.analyze_sentiment(
        "Bitcoin pourrait atteindre de nouveaux sommets cette année grâce à l'adoption institutionnelle.",
        model_name="ElKulako/cryptobert"
    )
    
    print(f"Sentiment: {result['overall_sentiment']}, Confiance: {result['confidence']}") 