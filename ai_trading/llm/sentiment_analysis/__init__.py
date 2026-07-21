# -*- coding: utf-8 -*-
"""Module d'analyse de sentiment pour les données financières et crypto."""

__all__ = ["NewsAnalyzer", "EnhancedNewsAnalyzer", "SocialAnalyzer", "SentimentPipeline"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    if name == "SocialAnalyzer":
        from ai_trading.llm.sentiment_analysis.social_analyzer import SocialAnalyzer

        return SocialAnalyzer
    if name == "SentimentPipeline":
        from ai_trading.llm.sentiment_analysis.sentiment_pipeline import SentimentPipeline

        return SentimentPipeline
    from ai_trading.llm.sentiment_analysis.news_analyzer import (
        EnhancedNewsAnalyzer,
        NewsAnalyzer,
    )

    return {"NewsAnalyzer": NewsAnalyzer, "EnhancedNewsAnalyzer": EnhancedNewsAnalyzer}[name]
