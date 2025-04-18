import pytest
from unittest.mock import patch
from ai_trading.tests.mocks import MockNewsAnalyzer, MockSocialAnalyzer

# Définir les patches à appliquer globalement à tous les tests
@pytest.fixture(autouse=True, scope="session")
def patch_sentiment_analyzers():
    """
    Remplace les classes d'analyse de sentiment par des mocks pour tous les tests,
    évitant ainsi les erreurs d'accès mémoire lors du chargement des modèles HuggingFace.
    """
    with patch('ai_trading.llm.sentiment_analysis.news_analyzer.NewsAnalyzer', MockNewsAnalyzer), \
         patch('ai_trading.llm.sentiment_analysis.social_analyzer.SocialAnalyzer', MockSocialAnalyzer):
        yield 