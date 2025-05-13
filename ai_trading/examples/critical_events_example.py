"""
Exemple d'utilisation du module de détection d'événements critiques.
"""

from datetime import datetime, timedelta

# import matplotlib.pyplot as plt  # Commenté pour éviter les erreurs tk
from tabulate import tabulate

from ai_trading.events.critical_events import (
    CriticalEventDetector,
    EventType,
    filter_news_by_relevance,
    preprocess_news,
)


def create_example_news():
    """Crée un ensemble d'actualités pour la démonstration."""
    return [
        {
            "title": "La SEC approuve un ETF Bitcoin spot",
            "content": "La Securities and Exchange Commission des États-Unis a enfin approuvé le premier ETF Bitcoin spot, marquant un tournant dans la régulation des crypto-monnaies. Cette décision pourrait ouvrir la voie à une adoption institutionnelle plus large.",
            "date": datetime.now().isoformat(),
            "source": "CryptoNews",
        },
        {
            "title": "La Chine durcit sa position sur les crypto-monnaies",
            "content": "Les autorités chinoises ont annoncé de nouvelles restrictions sur les activités liées aux crypto-monnaies, interdisant aux institutions financières de faciliter les transactions. Cette interdiction renforce la position déjà restrictive du pays.",
            "date": (datetime.now() - timedelta(days=1)).isoformat(),
            "source": "Global Finance",
        },
        {
            "title": "L'Union Européenne adopte le règlement MiCA",
            "content": "Le Parlement européen a adopté le règlement sur les marchés de crypto-actifs (MiCA), établissant un cadre réglementaire complet pour l'industrie des crypto-monnaies dans l'UE. Les nouvelles règles visent à protéger les investisseurs tout en favorisant l'innovation.",
            "date": (datetime.now() - timedelta(days=2)).isoformat(),
            "source": "EU News",
        },
        {
            "title": "Apple acquiert une startup spécialisée en blockchain",
            "content": "Apple a annoncé l'acquisition de BlockTech, une startup spécialisée dans les solutions de blockchain pour les paiements mobiles. Cette acquisition stratégique pourrait permettre à Apple d'intégrer des fonctionnalités blockchain dans ses futurs services financiers.",
            "date": (datetime.now() - timedelta(days=3)).isoformat(),
            "source": "TechCrunch",
        },
        {
            "title": "Microsoft et Ethereum Foundation lancent un partenariat",
            "content": "Microsoft a annoncé un partenariat stratégique avec la Fondation Ethereum pour développer des solutions d'entreprise basées sur la blockchain. Cette collaboration vise à améliorer l'évolutivité et la sécurité de la blockchain Ethereum pour les applications d'entreprise.",
            "date": (datetime.now() - timedelta(days=4)).isoformat(),
            "source": "Microsoft Blog",
        },
        {
            "title": "Facebook révèle son nouveau projet de crypto-monnaie",
            "content": "Meta (anciennement Facebook) a dévoilé un nouveau projet de crypto-monnaie, redessiné après l'échec de Libra/Diem. Ce nouveau projet intégrera les technologies NFT et metaverse, marquant une évolution de la stratégie blockchain de l'entreprise.",
            "date": (datetime.now() - timedelta(days=5)).isoformat(),
            "source": "Meta Newsroom",
        },
        {
            "title": "Une nouvelle ère s'ouvre pour la finance décentralisée",
            "content": "Les experts s'accordent à dire qu'une nouvelle ère s'ouvre pour la finance décentralisée (DeFi) avec l'arrivée de protocoles plus matures et régulés. Ce changement de paradigme pourrait transformer radicalement le secteur financier traditionnel.",
            "date": (datetime.now() - timedelta(days=2)).isoformat(),
            "source": "DeFi Pulse",
        },
        {
            "title": "Changement fondamental dans le narratif des NFTs",
            "content": "Le marché des NFTs connaît un changement fondamental de narratif, passant de l'art numérique spéculatif à des cas d'utilisation plus pratiques dans les domaines de l'identité, des documents légaux et des droits numériques. Cette transformation redéfinit l'utilité des tokens non-fongibles.",
            "date": (datetime.now() - timedelta(days=3)).isoformat(),
            "source": "NFT Insight",
        },
        {
            "title": "Le narratif 'Bitcoin comme réserve de valeur' gagne du terrain",
            "content": "Un changement de paradigme s'opère dans la perception du Bitcoin, de plus en plus considéré comme une 'réserve de valeur numérique' plutôt que comme un moyen de paiement. Cette évolution du narratif est renforcée par l'adoption croissante par les investisseurs institutionnels.",
            "date": (datetime.now() - timedelta(days=4)).isoformat(),
            "source": "Institutional Investor",
        },
        {
            "title": "Les marchés crypto entrent dans une nouvelle phase d'adoption",
            "content": "Selon plusieurs analyses, le marché des crypto-monnaies entre dans une nouvelle phase d'adoption, marquée par la maturité des infrastructures et l'arrivée d'acteurs institutionnels. Cette transformation représente un tournant majeur pour l'industrie.",
            "date": (datetime.now() - timedelta(days=5)).isoformat(),
            "source": "Crypto Analysis",
        },
    ]


def display_events(events, title):
    """Affiche les événements détectés dans un format lisible."""
    print(f"\n{'-'*80}")
    print(f"{title.upper()}")
    print(f"{'-'*80}")

    if not events:
        print("Aucun événement détecté.")
        return

    table_data = []
    for event in events:
        # Adapter l'affichage selon le type d'événement
        if event["type"] == EventType.REGULATORY.value:
            event_info = {
                "Type": "RÉGLEMENTAIRE",
                "Sévérité": event["severity"],
                "Titre": event["title"],
                "Source": event["source"],
                "Date": (
                    event["date"].split("T")[0]
                    if "T" in event["date"]
                    else event["date"]
                ),
                "Mots-clés": ", ".join(event["matched_terms"][:3]),
            }
        elif event["type"] == EventType.FUNDAMENTAL.value:
            event_info = {
                "Type": "FONDAMENTAL",
                "Sévérité": event["severity"],
                "Titre": event["title"],
                "Source": event["source"],
                "Date": (
                    event["date"].split("T")[0]
                    if "T" in event["date"]
                    else event["date"]
                ),
                "Entités": ", ".join(event["entities"][:3]),
            }
        elif event["type"] == EventType.NARRATIVE.value:
            event_info = {
                "Type": "NARRATIF",
                "Sévérité": event["severity"],
                "Pattern": event["pattern"],
                "Fréquence": event["frequency"],
                "Sources": ", ".join(event["sources"][:3]),
                "Dernière mention": (
                    event["latest_mention"].split("T")[0]
                    if "T" in event["latest_mention"]
                    else event["latest_mention"]
                ),
            }
        else:
            event_info = {
                "Type": "INCONNU",
                "Sévérité": event.get("severity", "?"),
                "Détails": str(event),
            }

        table_data.append(event_info)

    # Afficher le tableau
    if table_data:
        print(tabulate(table_data, headers="keys", tablefmt="pretty"))


def display_alerts(alerts):
    """Affiche les alertes générées."""
    print(f"\n{'-'*80}")
    print("ALERTES GÉNÉRÉES")
    print(f"{'-'*80}")

    if not alerts:
        print("Aucune alerte générée.")
        return

    for i, alert in enumerate(alerts, 1):
        print(f"ALERTE #{i}: [{alert['severity']}] {alert['message']}")
        print(
            f"  Horodatage: {alert['timestamp'].split('T')[0] if 'T' in alert['timestamp'] else alert['timestamp']}"
        )
        print()


def plot_event_distribution(all_events):
    """Trace un graphique de la distribution des événements par type et sévérité."""
    print(
        "\nLa génération de graphique a été désactivée en raison de problèmes de dépendances."
    )
    print(
        "Un graphique aurait montré la distribution des événements par type et sévérité."
    )

    # Préparer les données pour le graphique (pour référence)
    event_counts = {
        "REGULATORY": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0},
        "FUNDAMENTAL": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0},
        "NARRATIVE": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0},
    }

    # Compter les événements par type et sévérité
    for event_type, events in all_events.items():
        for event in events:
            severity = event.get("severity", "INFO")
            if event_type in event_counts and severity in event_counts[event_type]:
                event_counts[event_type][severity] += 1

    # Afficher un résumé textuel à la place
    print("\nRésumé de la distribution des événements:")
    for event_type, severities in event_counts.items():
        print(f"  {event_type}:")
        for severity, count in severities.items():
            if count > 0:
                print(f"    - {severity}: {count}")


def main():
    """Fonction principale de démonstration."""
    print("\n" + "=" * 80)
    print("DÉMONSTRATION DU DÉTECTEUR D'ÉVÉNEMENTS CRITIQUES")
    print("=" * 80)

    # Créer des données d'actualités
    news_data = create_example_news()
    print(f"Nombre d'actualités chargées: {len(news_data)}")

    # Prétraiter les actualités
    preprocessed_news = preprocess_news(news_data)

    # Filtrer les actualités pertinentes pour Bitcoin
    bitcoin_keywords = ["Bitcoin", "BTC"]
    bitcoin_news = filter_news_by_relevance(
        preprocessed_news, bitcoin_keywords, threshold=0.05
    )
    print(f"Nombre d'actualités pertinentes pour Bitcoin: {len(bitcoin_news)}")

    # Initialiser le détecteur d'événements
    detector = CriticalEventDetector()

    # Analyser tous les événements
    print("\nAnalyse des événements critiques en cours...")
    all_events = detector.analyze_all_events(
        preprocessed_news,
        regulatory_threshold=0.4,  # Seuil abaissé pour la démo
        fundamental_threshold=0.4,  # Seuil abaissé pour la démo
        narrative_min_mentions=1,  # Minimum abaissé pour la démo
    )

    # Afficher les événements réglementaires
    display_events(all_events[EventType.REGULATORY.value], "Événements Réglementaires")

    # Afficher les événements fondamentaux
    display_events(all_events[EventType.FUNDAMENTAL.value], "Événements Fondamentaux")

    # Afficher les changements de narratif
    display_events(all_events[EventType.NARRATIVE.value], "Changements de Narratif")

    # Générer des alertes
    alerts = detector.get_alert_recommendations(all_events)
    display_alerts(alerts)

    # Tracer un graphique de la distribution des événements
    plot_event_distribution(all_events)

    print("\n" + "=" * 80)
    print("DÉMONSTRATION TERMINÉE")
    print("=" * 80)


if __name__ == "__main__":
    main()
