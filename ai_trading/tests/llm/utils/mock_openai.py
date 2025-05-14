"""
Mock pour OpenAI à utiliser dans les tests.

Ce module fournit des mocks pour les classes et fonctions d'OpenAI,
ce qui permet de tester le code qui dépend d'OpenAI sans avoir besoin
d'une clé API valide ou d'une connexion Internet.
"""

import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

class MockOpenAI:
    """Mock pour la classe OpenAI."""
    
    def __init__(self, api_key=None):
        """Initialise le mock avec un objet chat qui contient un objet completions."""
        self.api_key = api_key or "mock-api-key"
        self.chat = self.Chat()
    
    class Chat:
        """Mock pour le chat d'OpenAI."""
        
        def __init__(self):
            """Initialise avec un objet completions."""
            self.completions = self.Completions()
        
        class Completions:
            """Mock pour les completions de chat d'OpenAI."""
            
            @staticmethod
            def create(model, messages, temperature=0, max_tokens=None, response_format=None, **kwargs):
                """
                Simule l'appel à l'API de chat completion d'OpenAI.
                
                Args:
                    model: Nom du modèle à utiliser
                    messages: Liste des messages de la conversation
                    temperature: Paramètre de température pour la génération
                    max_tokens: Nombre maximum de tokens à générer
                    response_format: Format de réponse attendu (ex: {"type": "json_object"})
                    **kwargs: Arguments supplémentaires
                
                Returns:
                    Un objet de réponse simulée
                """
                # Introduit un délai pour simuler l'appel API
                time.sleep(0.1)
                
                # Formatage d'une réponse générique basée sur le contenu du message
                last_message = messages[-1]['content'] if messages else ""
                
                # Si le message mentionne des crypto-monnaies spécifiques, adapte la réponse
                response_content = {}
                if "BTC" in last_message or "Bitcoin" in last_message:
                    response_content = {
                        "direction": "bullish",
                        "confidence": 0.85,
                        "analysis": "The asset shows strong bullish indicators based on recent price movements and volume.",
                        "key_factors": ["Price trend", "Volume increase", "Positive sentiment"],
                        "contradictions": [],
                        "volatility": "medium"
                    }
                elif "ETH" in last_message or "Ethereum" in last_message:
                    response_content = {
                        "direction": "bearish",
                        "confidence": 0.65,
                        "analysis": "The asset is showing bearish signals primarily due to technical indicators and market sentiment.",
                        "key_factors": ["Price decline", "Negative news", "Technical indicators"],
                        "contradictions": ["Some institutional buying"],
                        "volatility": "high"
                    }
                else:
                    response_content = {
                        "direction": "neutral",
                        "confidence": 0.40,
                        "analysis": "The market is showing mixed signals with no clear direction at this time.",
                        "key_factors": ["Mixed signals", "Sideways movement"],
                        "contradictions": ["Conflicting indicators"],
                        "volatility": "low"
                    }
                
                # Gestion du format de réponse JSON
                if response_format and response_format.get("type") == "json_object":
                    content = json.dumps(response_content)
                else:
                    # Format texte avec du JSON inclus
                    content = json.dumps(response_content)
                
                # Crée un objet de message simulé avec les attributs requis
                message = type('Message', (), {
                    'content': content,
                    'role': 'assistant',
                    'function_call': None
                })
                
                # Crée un objet de choix simulé avec les attributs requis
                choice = type('Choice', (), {
                    'message': message,
                    'index': 0,
                    'finish_reason': 'stop'
                })
                
                # Crée un objet de réponse simulé avec tous les attributs requis
                return type('Response', (), {
                    'choices': [choice],
                    'created': int(time.time()),
                    'id': f"mock-{model}-{int(time.time())}",
                    'model': model,
                    'object': 'chat.completion',
                    'usage': type('Usage', (), {
                        'prompt_tokens': 100,
                        'completion_tokens': 150,
                        'total_tokens': 250
                    })
                })

# Pour les tests qui importent directement des modules OpenAI
Completion = MockOpenAI().Chat.Completions 