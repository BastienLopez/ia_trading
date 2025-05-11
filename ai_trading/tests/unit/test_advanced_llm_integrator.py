import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ai_trading.llm.sentiment_analysis.advanced_llm_integrator import AdvancedLLMIntegrator

# Patch pour simuler la disponibilité de PEFT
patch('ai_trading.llm.sentiment_analysis.advanced_llm_integrator.PEFT_AVAILABLE', True).start()

class TestAdvancedLLMIntegrator(unittest.TestCase):
    
    def setUp(self):
        # Création d'un mock pour éviter de charger les modèles réels
        self.patcher = patch('ai_trading.llm.sentiment_analysis.advanced_llm_integrator.AutoModelForSequenceClassification')
        self.mock_model = self.patcher.start()
        
        self.tokenizer_patcher = patch('ai_trading.llm.sentiment_analysis.advanced_llm_integrator.AutoTokenizer')
        self.mock_tokenizer = self.tokenizer_patcher.start()
        
        self.pipeline_patcher = patch('ai_trading.llm.sentiment_analysis.advanced_llm_integrator.pipeline')
        self.mock_pipeline = self.pipeline_patcher.start()
        
        # Configuration des mocks pour simuler les résultats
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [[
            {"label": "POSITIVE", "score": 0.8},
            {"label": "NEGATIVE", "score": 0.1},
            {"label": "NEUTRAL", "score": 0.1}
        ]]
        self.mock_pipeline.return_value = mock_pipeline_instance
        
        # Création de l'intégrateur avec GPU désactivé
        self.integrator = AdvancedLLMIntegrator(use_gpu=False, quantize=False)
        
        # Mock du cache
        self.integrator.cache = MagicMock()
        self.integrator.cache.load.return_value = None
        
    def tearDown(self):
        self.patcher.stop()
        self.tokenizer_patcher.stop()
        self.pipeline_patcher.stop()
    
    def test_initialization(self):
        """Teste l'initialisation correcte de l'intégrateur."""
        self.assertEqual(self.integrator.device, "cpu")
        self.assertEqual(self.integrator.quantize, False)
        self.assertEqual(self.integrator.models, {})
        self.assertEqual(self.integrator.tokenizers, {})
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_detection(self, mock_cuda):
        """Teste la détection du GPU."""
        integrator = AdvancedLLMIntegrator(use_gpu=True)
        self.assertEqual(integrator.device, "cuda")
    
    def test_load_model(self):
        """Teste le chargement d'un modèle."""
        self.integrator.load_model("test/model")
        
        # Vérification que le tokenizer a été chargé
        self.mock_tokenizer.from_pretrained.assert_called_once_with("test/model")
        
        # Vérification que le modèle a été chargé
        self.mock_model.from_pretrained.assert_called_once()
        
        # Vérification que le modèle est stocké
        self.assertIn("test/model", self.integrator.models)
        self.assertIn("test/model", self.integrator.tokenizers)
    
    def test_analyze_sentiment(self):
        """Teste l'analyse de sentiment."""
        # Configuration du mock pour le modèle
        self.integrator.models = {"ElKulako/cryptobert": MagicMock()}
        self.integrator.tokenizers = {"ElKulako/cryptobert": MagicMock()}
        
        # Analyse du sentiment
        result = self.integrator.analyze_sentiment(
            "Bitcoin est en hausse aujourd'hui."
        )
        
        # Vérification des résultats
        self.assertIn("model", result)
        self.assertIn("sentiment", result)
        self.assertIn("overall_sentiment", result)
        self.assertIn("confidence", result)
        
        # Vérification que le résultat est mis en cache
        self.integrator.cache.save.assert_called_once()
    
    def test_prepare_text_with_context(self):
        """Teste la préparation du texte avec contexte."""
        text = "Bitcoin est en hausse."
        context = {
            "price": 50000,
            "trend": "bullish",
            "market_cap": "1T",
            "volume": "10B"
        }
        
        prepared = self.integrator._prepare_text_with_context(text, context)
        
        # Vérification que le contexte est inclus
        self.assertIn("Prix: 50000", prepared)
        self.assertIn("Tendance: bullish", prepared)
        self.assertIn("Capitalisation: 1T", prepared)
        self.assertIn("Volume: 10B", prepared)
        self.assertIn(text, prepared)
    
    def test_process_sentiment_results_cryptobert(self):
        """Teste le traitement des résultats pour CryptoBERT."""
        results = [[
            {"label": "POSITIVE", "score": 0.7},
            {"label": "NEGATIVE", "score": 0.2},
            {"label": "NEUTRAL", "score": 0.1}
        ]]
        
        processed = self.integrator._process_sentiment_results(results, "ElKulako/cryptobert")
        
        self.assertEqual(processed["model"], "ElKulako/cryptobert")
        self.assertIn("positive", processed["sentiment"])
        self.assertEqual(processed["sentiment"]["positive"], 0.7)
        self.assertGreater(processed["overall_sentiment"], 0)
        self.assertEqual(processed["confidence"], 0.7)
    
    def test_process_sentiment_results_generic(self):
        """Teste le traitement des résultats pour un modèle générique."""
        results = [[
            {"label": "1", "score": 0.6},
            {"label": "0", "score": 0.4}
        ]]
        
        processed = self.integrator._process_sentiment_results(results, "generic/model")
        
        self.assertEqual(processed["model"], "generic/model")
        self.assertIn("1", processed["sentiment"])
        self.assertEqual(processed["sentiment"]["1"], 0.6)
        self.assertGreater(processed["overall_sentiment"], 0)
        self.assertEqual(processed["confidence"], 0.6)
    
    @patch('ai_trading.llm.sentiment_analysis.advanced_llm_integrator.AutoModelForCausalLM')
    def test_generate_crypto_analysis(self, mock_causal_model):
        """Teste la génération d'analyse crypto."""
        # Configuration des mocks
        self.integrator.models = {"test/model": MagicMock()}
        self.integrator.tokenizers = {"test/model": MagicMock()}
        
        # Mock du pipeline de génération
        mock_generation = MagicMock()
        mock_generation.return_value = [{"generated_text": "Prompt Texte généré"}]
        self.mock_pipeline.return_value = mock_generation
        
        # Génération d'analyse
        result = self.integrator.generate_crypto_analysis(
            "Prompt",
            model_name="test/model"
        )
        
        # Vérification du résultat
        self.assertEqual(result, "Texte généré")
        
        # Vérification que le pipeline a été créé correctement
        self.mock_pipeline.assert_called_with(
            "text-generation",
            model=self.integrator.models["test/model"],
            tokenizer=self.integrator.tokenizers["test/model"],
            device=-1,
            max_length=256
        )
    
    def test_optimize_prompts(self):
        """Teste l'optimisation des prompts."""
        # Configuration des mocks
        self.integrator.generate_crypto_analysis = MagicMock()
        self.integrator.generate_crypto_analysis.side_effect = lambda prompt, model_name: "positif" if "expert" in prompt else "négatif"
        
        # Mock de la méthode load_model pour éviter l'appel à Hugging Face
        self.integrator.load_model = MagicMock()
        
        # Cas de test
        test_cases = [
            {"input": "BTC", "expected": "positif"}
        ]
        
        # Optimisation des prompts
        result = self.integrator.optimize_prompts(
            "Analyse",
            test_cases,
            model_name="test/model"
        )
        
        # Vérification que le meilleur prompt a été sélectionné
        self.assertEqual(result, "En tant qu'expert en cryptomonnaies, Analyse")

if __name__ == '__main__':
    unittest.main() 