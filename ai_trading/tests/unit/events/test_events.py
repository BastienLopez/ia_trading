"""
Script de test simple pour vérifier la fonctionnalité du module de détection d'événements critiques.
"""

from datetime import datetime, timedelta

from ai_trading.events.critical_events import CriticalEventDetector


def main():
    # Créer des données de test
    test_news = [
        {
            "title": "La SEC approuve de nouvelles régulations pour les crypto-monnaies",
            "content": "La Securities and Exchange Commission a annoncé aujourd'hui de nouvelles directives de conformité pour les plateformes d'échange de crypto-monnaies.",
            "date": datetime.now().isoformat(),
            "source": "CryptoNews",
        },
        {
            "title": "MegaCorp annonce un partenariat stratégique avec Ethereum",
            "content": "MegaCorp a annoncé un partenariat stratégique avec la Fondation Ethereum pour développer des solutions blockchain.",
            "date": (datetime.now() - timedelta(days=2)).isoformat(),
            "source": "TechCrunch",
        },
    ]

    # Initialiser le détecteur
    detector = CriticalEventDetector()

    # Tester la détection d'événements réglementaires
    regulatory = detector.detect_regulatory_events(test_news, threshold=0.3)
    print(f"Événements réglementaires détectés: {len(regulatory)}")
    if regulatory:
        print(f"  Premier événement: {regulatory[0]['title']}")
        print(f"  Mots-clés: {regulatory[0]['matched_terms']}")

    # Tester la détection d'événements fondamentaux
    fundamental = detector.detect_fundamental_events(test_news, threshold=0.3)
    print(f"Événements fondamentaux détectés: {len(fundamental)}")
    if fundamental:
        print(f"  Premier événement: {fundamental[0]['title']}")
        print(f"  Mots-clés: {fundamental[0]['matched_terms']}")
        print(f"  Entités: {fundamental[0]['entities']}")

    print("Test terminé avec succès!")


if __name__ == "__main__":
    main()
