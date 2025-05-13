"""
Test basique du module de détection d'événements critiques.
"""

from datetime import datetime, timedelta

from ai_trading.events.critical_events import (
    CriticalEventDetector,
    EventType,
    preprocess_news,
)

# Création de données de test avec des mots-clés plus explicites
nouvelles_test = [
    {
        "title": "La SEC impose des sanctions sévères contre une plateforme d'échange",
        "content": "La Securities and Exchange Commission a annoncé aujourd'hui des sanctions et amendes contre BitExchange pour non-conformité aux régulations en vigueur. Cette interdiction pourrait entraîner des restrictions importantes pour d'autres plateformes.",
        "date": datetime.now().isoformat(),
        "source": "CryptoNews",
    },
    {
        "title": "CryptoFirm annonce un partenariat stratégique majeur avec Visa",
        "content": "La société CryptoFirm a officialisé aujourd'hui un partenariat stratégique avec Visa pour développer des solutions de paiement basées sur la blockchain. Cette alliance stratégique marque une étape importante dans l'adoption de la technologie blockchain.",
        "date": (datetime.now() - timedelta(days=1)).isoformat(),
        "source": "BusinessCrypto",
    },
    {
        "title": "Une nouvelle ère pour les NFTs : au-delà de l'art numérique",
        "content": "Les experts s'accordent à dire qu'un changement de paradigme s'opère dans l'utilisation des NFTs, désormais adoptés pour la certification de documents et l'identité numérique. Cette nouvelle ère transforme radicalement le marché.",
        "date": (datetime.now() - timedelta(days=2)).isoformat(),
        "source": "TechInsight",
    },
    {
        "title": "Une nouvelle ère s'ouvre pour la finance décentralisée",
        "content": "Un changement de paradigme se profile dans le monde de la finance décentralisée. Cette transformation fondamentale pourrait redéfinir complètement le secteur financier traditionnel.",
        "date": (datetime.now() - timedelta(days=3)).isoformat(),
        "source": "DeFi Pulse",
    },
]

# Prétraitement des nouvelles
nouvelles_preprocessees = preprocess_news(nouvelles_test)

# Initialisation du détecteur
detecteur = CriticalEventDetector()
print("Détecteur initialisé avec succès!")

# Détection des différents types d'événements avec des seuils très bas
print("\n--- DÉTECTION D'ÉVÉNEMENTS RÉGLEMENTAIRES ---")
evenements_reglementaires = detecteur.detect_regulatory_events(
    nouvelles_preprocessees, threshold=0.05
)
print(f"Événements réglementaires détectés: {len(evenements_reglementaires)}")
if evenements_reglementaires:
    print(f"  Titre: {evenements_reglementaires[0]['title']}")
    print(f"  Mots-clés: {evenements_reglementaires[0]['matched_terms']}")
    print(f"  Sévérité: {evenements_reglementaires[0]['severity']}")

print("\n--- DÉTECTION D'ÉVÉNEMENTS FONDAMENTAUX ---")
evenements_fondamentaux = detecteur.detect_fundamental_events(
    nouvelles_preprocessees, threshold=0.05
)
print(f"Événements fondamentaux détectés: {len(evenements_fondamentaux)}")
if evenements_fondamentaux:
    print(f"  Titre: {evenements_fondamentaux[0]['title']}")
    print(f"  Mots-clés: {evenements_fondamentaux[0]['matched_terms']}")
    print(f"  Entités: {evenements_fondamentaux[0]['entities']}")
    print(f"  Sévérité: {evenements_fondamentaux[0]['severity']}")

print("\n--- DÉTECTION DE CHANGEMENTS DE NARRATIF ---")
changements_narratif = detecteur.detect_narrative_changes(
    nouvelles_preprocessees, min_mentions=1
)
print(f"Changements de narratif détectés: {len(changements_narratif)}")
if changements_narratif:
    print(f"  Pattern: {changements_narratif[0]['pattern']}")
    print(f"  Fréquence: {changements_narratif[0]['frequency']}")
    print(f"  Sévérité: {changements_narratif[0]['severity']}")

# Analyse complète avec des seuils très bas
print("\n--- ANALYSE COMPLÈTE DES ÉVÉNEMENTS ---")
tous_evenements = {
    EventType.REGULATORY.value: evenements_reglementaires,
    EventType.FUNDAMENTAL.value: evenements_fondamentaux,
    EventType.NARRATIVE.value: changements_narratif,
}

# Pour forcer la génération d'alertes, on met une sévérité élevée
for event_type in tous_evenements:
    for event in tous_evenements[event_type]:
        event["severity"] = "HIGH"  # On utilise HIGH au lieu de CRITICAL

# Génération d'alertes
print("\n--- GÉNÉRATION D'ALERTES ---")
alertes = detecteur.get_alert_recommendations(tous_evenements)
print(f"Alertes générées: {len(alertes)}")
if alertes:
    for i, alerte in enumerate(alertes, 1):
        print(f"  Alerte #{i}: {alerte['message']}")

print("\nTest terminé avec succès!")
