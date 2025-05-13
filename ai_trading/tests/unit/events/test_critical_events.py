"""
Tests unitaires pour le module de détection d'événements critiques.
"""

import unittest
from datetime import datetime, timedelta

from ai_trading.events.critical_events import (
    CriticalEventDetector,
    EventSeverity,
    EventType,
    filter_news_by_relevance,
    preprocess_news,
)


def create_test_news():
    """Crée des données d'actualités de test."""
    return [
        {
            "title": "La SEC approuve de nouvelles régulations pour les crypto-monnaies",
            "content": "La Securities and Exchange Commission a annoncé aujourd'hui de nouvelles directives de conformité pour les plateformes d'échange de crypto-monnaies, imposant des exigences plus strictes en matière de reporting et de transparence.",
            "date": datetime.now().isoformat(),
            "source": "CryptoNews",
        },
        {
            "title": "Bitcoin atteint un nouveau record historique",
            "content": "Le prix du Bitcoin a dépassé son précédent record historique, atteignant 70 000 dollars pour la première fois de son histoire dans un contexte de forte demande institutionnelle.",
            "date": (datetime.now() - timedelta(days=1)).isoformat(),
            "source": "CoinDesk",
        },
        {
            "title": "MegaCorp annonce un partenariat stratégique avec Ethereum",
            "content": "MegaCorp, le géant de la technologie, a annoncé un partenariat stratégique avec la Fondation Ethereum pour développer des solutions d'entreprise basées sur la blockchain Ethereum. Cette collaboration marque une étape importante dans l'adoption des technologies blockchain par les grandes entreprises.",
            "date": (datetime.now() - timedelta(days=2)).isoformat(),
            "source": "TechCrunch",
        },
        {
            "title": "Changement de paradigme : L'IA révolutionne le trading de crypto-monnaies",
            "content": "Une nouvelle ère s'ouvre pour le trading de crypto-monnaies avec l'intégration de technologies d'intelligence artificielle avancées. Cette transformation fondamentale change la manière dont les traders analysent les marchés et prennent des décisions d'investissement.",
            "date": (datetime.now() - timedelta(days=3)).isoformat(),
            "source": "AI Finance",
        },
        {
            "title": "Une tendance émergente : La finance décentralisée gagne du terrain",
            "content": "Le secteur de la finance décentralisée (DeFi) connaît une croissance sans précédent, marquant un tournant majeur dans l'écosystème crypto. Cette réorientation vers des services financiers non-custodial représente un changement fondamental dans l'industrie.",
            "date": (datetime.now() - timedelta(days=4)).isoformat(),
            "source": "DeFi Pulse",
        },
    ]


class TestCriticalEventDetector(unittest.TestCase):
    """Tests pour la classe CriticalEventDetector."""

    def setUp(self):
        """Initialise le détecteur et les données de test."""
        self.detector = CriticalEventDetector()
        self.test_news = create_test_news()

    def test_initialization(self):
        """Teste l'initialisation du détecteur."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.regulatory_keywords)
        self.assertIsNotNone(self.detector.fundamental_keywords)
        self.assertIsNotNone(self.detector.narrative_patterns)

    def test_detect_regulatory_events(self):
        """Teste la détection d'événements réglementaires."""
        # Abaisser le seuil pour les tests
        events = self.detector.detect_regulatory_events(self.test_news, threshold=0.05)

        # Vérifier qu'au moins un événement réglementaire est détecté
        self.assertGreaterEqual(len(events), 1)

        # Vérifier que le premier événement est bien de type réglementaire
        self.assertEqual(events[0]["type"], EventType.REGULATORY.value)

        # Vérifier que les mots-clés détectés sont présents
        self.assertTrue(
            any(
                term in events[0]["matched_terms"]
                for term in ["SEC", "régulation", "conformité"]
            )
        )

    def test_detect_fundamental_events(self):
        """Teste la détection d'événements fondamentaux."""
        # Abaisser le seuil pour les tests
        events = self.detector.detect_fundamental_events(self.test_news, threshold=0.05)

        # Vérifier qu'au moins un événement fondamental est détecté
        self.assertGreaterEqual(len(events), 1)

        # Vérifier que le premier événement est bien de type fondamental
        self.assertEqual(events[0]["type"], EventType.FUNDAMENTAL.value)

        # Vérifier que les mots-clés détectés sont présents
        self.assertTrue(
            any(
                term in events[0]["matched_terms"]
                for term in ["partenariat", "stratégique"]
            )
        )

        # Vérifier que le champ entities existe, sans vérifier son contenu spécifique
        self.assertIn("entities", events[0])

    def test_detect_narrative_changes(self):
        """Teste la détection de changements de narratif."""
        # Créer des données avec plusieurs mentions de narrative
        narrative_news = []
        patterns = [
            "changement de paradigme",
            "nouvelle ère",
            "transformation fondamentale",
        ]

        for i in range(5):
            for pattern in patterns:
                news = {
                    "title": f"Article sur {pattern}",
                    "content": f"Cet article parle du {pattern} dans le secteur des crypto-monnaies.",
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                    "source": f"Source{i}",
                }
                narrative_news.append(news)

        events = self.detector.detect_narrative_changes(narrative_news, min_mentions=2)

        # Vérifier qu'au moins un changement de narratif est détecté
        self.assertGreaterEqual(len(events), 1)

        # Vérifier que le premier événement est bien de type narratif
        self.assertEqual(events[0]["type"], EventType.NARRATIVE.value)

        # Vérifier que le pattern est présent et que la fréquence est correcte
        self.assertIn(events[0]["pattern"], patterns)
        self.assertGreaterEqual(events[0]["frequency"], 2)

    def test_analyze_all_events(self):
        """Teste l'analyse de tous les types d'événements."""
        # Abaisser les seuils pour les tests
        all_events = self.detector.analyze_all_events(
            self.test_news,
            regulatory_threshold=0.05,
            fundamental_threshold=0.05,
            narrative_min_mentions=1,
        )

        # Vérifier que tous les types d'événements sont présents
        self.assertIn(EventType.REGULATORY.value, all_events)
        self.assertIn(EventType.FUNDAMENTAL.value, all_events)
        self.assertIn(EventType.NARRATIVE.value, all_events)

        # Vérifier qu'au moins un événement réglementaire est détecté
        self.assertGreaterEqual(len(all_events[EventType.REGULATORY.value]), 1)

    def test_get_alert_recommendations(self):
        """Teste la génération de recommandations d'alerte."""
        # Abaisser les seuils pour les tests
        all_events = self.detector.analyze_all_events(
            self.test_news,
            regulatory_threshold=0.05,
            fundamental_threshold=0.05,
            narrative_min_mentions=1,
        )

        # Forcer la sévérité à CRITICAL pour tous les événements détectés
        for event_type in all_events:
            for event in all_events[event_type]:
                event["severity"] = EventSeverity.CRITICAL.name

        alerts = self.detector.get_alert_recommendations(all_events)

        # Vérifier qu'au moins une alerte est générée
        self.assertGreaterEqual(len(alerts), 1)

        # Vérifier le format de l'alerte
        self.assertIn("event_type", alerts[0])
        self.assertIn("severity", alerts[0])
        self.assertIn("message", alerts[0])
        self.assertIn("timestamp", alerts[0])


class TestUtilityFunctions(unittest.TestCase):
    """Tests pour les fonctions utilitaires."""

    def test_preprocess_news(self):
        """Teste le prétraitement des actualités."""
        # Actualités avec données manquantes
        incomplete_news = [
            {"title": "Test", "content": None},
            {"title": None, "content": "Test content"},
            {},
        ]

        processed = preprocess_news(incomplete_news)

        # Vérifier que tous les champs sont présents et normalisés
        for news in processed:
            self.assertIn("title", news)
            self.assertIn("content", news)
            self.assertIn("date", news)
            self.assertIn("source", news)

    def test_filter_news_by_relevance(self):
        """Teste le filtrage des actualités par pertinence."""
        test_news = create_test_news()

        # Filtrer les actualités liées à Bitcoin
        bitcoin_keywords = ["Bitcoin", "BTC"]
        filtered = filter_news_by_relevance(test_news, bitcoin_keywords)

        # Vérifier que les actualités Bitcoin sont bien filtrées
        self.assertGreaterEqual(len(filtered), 1)
        self.assertIn("Bitcoin", filtered[0]["title"])

        # Vérifier que chaque actualité a un score de pertinence
        for news in filtered:
            self.assertIn("relevance_score", news)
            self.assertGreaterEqual(news["relevance_score"], 0.1)


if __name__ == "__main__":
    unittest.main()
