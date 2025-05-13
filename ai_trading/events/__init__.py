"""
Package de détection et d'analyse d'événements critiques pour les marchés de crypto-monnaies.

Ce package contient des modules pour:
- Détecter les annonces réglementaires importantes
- Identifier les évolutions fondamentales (acquisitions, partenariats)
- Analyser les changements de narratif de marché
"""

from ai_trading.events.critical_events import (
    CriticalEventDetector, 
    EventType, 
    EventSeverity,
    preprocess_news,
    filter_news_by_relevance
) 