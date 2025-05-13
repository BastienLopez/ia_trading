"""
Module pour la détection d'événements critiques sur les marchés de crypto-monnaies.

Ce module permet :
1. L'identification d'annonces réglementaires importantes
2. La détection d'évolutions fondamentales (acquisitions, partenariats)
3. L'analyse des changements de narratif de marché
"""

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types d'événements critiques."""

    REGULATORY = "regulatory"  # Réglementaire
    FUNDAMENTAL = "fundamental"  # Fondamental (partenariats, acquisitions)
    NARRATIVE = "narrative"  # Changement de narratif
    UNKNOWN = "unknown"  # Type inconnu


class EventSeverity(Enum):
    """Niveau de sévérité des événements."""

    CRITICAL = 5  # Impact potentiel très élevé
    HIGH = 4  # Impact potentiel élevé
    MEDIUM = 3  # Impact potentiel modéré
    LOW = 2  # Impact potentiel faible
    INFO = 1  # Informatif seulement


class CriticalEventDetector:
    """
    Détecteur d'événements critiques pour les marchés de crypto-monnaies.

    Cette classe permet d'analyser les flux d'actualités, communiqués et données de marché
    pour identifier et alerter sur des événements susceptibles d'avoir un impact significatif.
    """

    def __init__(self, regulatory_keywords: Optional[Dict[str, float]] = None):
        """
        Initialise le détecteur d'événements critiques.

        Args:
            regulatory_keywords: Dictionnaire de mots-clés réglementaires avec leur poids
        """
        # Dictionnaire de mots-clés pour les événements réglementaires
        self.regulatory_keywords = regulatory_keywords or {
            "SEC": 0.8,
            "régulation": 0.7,
            "interdiction": 0.9,
            "sanction": 0.8,
            "conformité": 0.6,
            "légal": 0.5,
            "amende": 0.7,
            "restriction": 0.75,
            "autorisation": 0.65,
            "législation": 0.7,
            "directive": 0.6,
            "enforcement": 0.85,
            "compliance": 0.65,
            "ban": 0.9,
            "licence": 0.6,
            "approbation": 0.6,
            "investigation": 0.75,
            "procès": 0.8,
            "CFTC": 0.8,
            "FinCEN": 0.75,
            "FSA": 0.75,
            "BaFin": 0.75,
            "AMF": 0.75,
            "FINMA": 0.75,
            "blanchiment": 0.85,
            "KYC": 0.6,
            "AML": 0.6,
            "gel des actifs": 0.8,
            "taxe": 0.7,
            "fiscalité": 0.7,
        }

        # Mots-clés pour les événements fondamentaux
        self.fundamental_keywords = {
            "partenariat": 0.7,
            "acquisition": 0.8,
            "fusion": 0.8,
            "rachat": 0.75,
            "investissement": 0.6,
            "levée de fonds": 0.7,
            "collaboration": 0.65,
            "alliance": 0.7,
            "contrat": 0.6,
            "joint venture": 0.75,
            "développement": 0.5,
            "lancement": 0.6,
            "déploiement": 0.6,
            "adoption": 0.7,
            "intégration": 0.65,
            "entreprise": 0.5,
            "corporatif": 0.5,
            "commercial": 0.5,
            "innovation": 0.6,
            "expansion": 0.6,
            "disruption": 0.7,
            "technologie": 0.5,
        }

        # Phrases clés pour détecter les changements de narratif
        self.narrative_patterns = [
            r"changement de paradigme",
            r"nouvelle ère",
            r"révolution",
            r"transformation",
            r"évolution du marché",
            r"changement fondamental",
            r"tendance émergente",
            r"pivot",
            r"réorientation",
            r"redéfinition",
            r"reconception",
            r"réinvention",
            r"changement structurel",
            r"inflexion",
            r"tournant majeur",
            r"changement de direction",
        ]

        # Historique des événements détectés (pour éviter les doublons)
        self.detected_events: List[Dict] = []

        # Période minimale entre deux événements similaires (en heures)
        self.min_event_interval = 24

        logger.info("Détecteur d'événements critiques initialisé")

    def detect_regulatory_events(
        self, news_data: List[Dict], threshold: float = 0.6
    ) -> List[Dict]:
        """
        Identifie les annonces réglementaires importantes.

        Args:
            news_data: Liste de dictionnaires contenant les actualités
                      (doit contenir 'title', 'content', 'date', 'source')
            threshold: Seuil de score pour considérer un événement comme réglementaire

        Returns:
            Liste des événements réglementaires détectés
        """
        regulatory_events = []

        for news in news_data:
            title = news.get("title", "").lower()
            content = news.get("content", "").lower()

            # Calculer un score basé sur les mots-clés réglementaires
            score = 0
            matched_terms = []

            # Vérifier le titre (poids plus important)
            for keyword, weight in self.regulatory_keywords.items():
                if keyword.lower() in title:
                    score += weight * 1.5
                    matched_terms.append(keyword)
                elif keyword.lower() in content:
                    score += weight
                    matched_terms.append(keyword)

            # Normaliser le score
            max_possible_score = sum(w * 1.5 for w in self.regulatory_keywords.values())
            normalized_score = (
                score / max_possible_score if max_possible_score > 0 else 0
            )

            # Si le score dépasse le seuil, considérer comme événement réglementaire
            if normalized_score >= threshold:
                event = {
                    "type": EventType.REGULATORY.value,
                    "title": news.get("title", ""),
                    "content": news.get("content", ""),
                    "date": news.get("date", datetime.now().isoformat()),
                    "source": news.get("source", "unknown"),
                    "score": normalized_score,
                    "matched_terms": list(set(matched_terms)),
                    "severity": self._determine_severity(normalized_score),
                    "detected_at": datetime.now().isoformat(),
                }

                # Ne pas ajouter de doublons
                if not self._is_duplicate_event(event):
                    regulatory_events.append(event)
                    self.detected_events.append(event)

        logger.info(
            f"Détection d'événements réglementaires: {len(regulatory_events)} trouvés"
        )
        return regulatory_events

    def detect_fundamental_events(
        self, news_data: List[Dict], threshold: float = 0.55
    ) -> List[Dict]:
        """
        Détecte les évolutions fondamentales comme les acquisitions, partenariats.

        Args:
            news_data: Liste de dictionnaires contenant les actualités
            threshold: Seuil de score pour considérer un événement comme fondamental

        Returns:
            Liste des événements fondamentaux détectés
        """
        fundamental_events = []

        for news in news_data:
            title = news.get("title", "").lower()
            content = news.get("content", "").lower()

            # Calculer un score basé sur les mots-clés fondamentaux
            score = 0
            matched_terms = []

            # Vérifier le titre (poids plus important)
            for keyword, weight in self.fundamental_keywords.items():
                if keyword.lower() in title:
                    score += weight * 1.5
                    matched_terms.append(keyword)
                elif keyword.lower() in content:
                    score += weight
                    matched_terms.append(keyword)

            # Normaliser le score
            max_possible_score = sum(
                w * 1.5 for w in self.fundamental_keywords.values()
            )
            normalized_score = (
                score / max_possible_score if max_possible_score > 0 else 0
            )

            # Si le score dépasse le seuil, considérer comme événement fondamental
            if normalized_score >= threshold:
                # Déterminer les entités impliquées
                entities = self._extract_entities(title + " " + content)

                event = {
                    "type": EventType.FUNDAMENTAL.value,
                    "title": news.get("title", ""),
                    "content": news.get("content", ""),
                    "date": news.get("date", datetime.now().isoformat()),
                    "source": news.get("source", "unknown"),
                    "score": normalized_score,
                    "matched_terms": list(set(matched_terms)),
                    "entities": entities,
                    "severity": self._determine_severity(normalized_score),
                    "detected_at": datetime.now().isoformat(),
                }

                # Ne pas ajouter de doublons
                if not self._is_duplicate_event(event):
                    fundamental_events.append(event)
                    self.detected_events.append(event)

        logger.info(
            f"Détection d'événements fondamentaux: {len(fundamental_events)} trouvés"
        )
        return fundamental_events

    def detect_narrative_changes(
        self, news_data: List[Dict], timeframe_days: int = 7, min_mentions: int = 3
    ) -> List[Dict]:
        """
        Détecte les changements de narratif du marché.

        Args:
            news_data: Liste de dictionnaires contenant les actualités
            timeframe_days: Période en jours pour analyser les tendances
            min_mentions: Nombre minimum de mentions pour considérer un changement de narratif

        Returns:
            Liste des changements de narratif détectés
        """
        narrative_events = []

        # Regrouper les actualités par date
        news_by_date = {}
        for news in news_data:
            date_str = news.get("date", datetime.now().isoformat())
            try:
                date = datetime.fromisoformat(date_str).date()
            except (ValueError, TypeError):
                # Fallback si le format de date est incorrect
                date = datetime.now().date()

            if date not in news_by_date:
                news_by_date[date] = []
            news_by_date[date].append(news)

        # Analyser les patterns narratifs sur la période
        pattern_mentions = {}

        for date, news_list in news_by_date.items():
            # Combiner tous les textes d'actualités du jour
            all_text = " ".join(
                [f"{n.get('title', '')} {n.get('content', '')}" for n in news_list]
            ).lower()

            # Chercher les patterns de changement de narratif
            for pattern in self.narrative_patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                if matches:
                    if pattern not in pattern_mentions:
                        pattern_mentions[pattern] = []
                    pattern_mentions[pattern].append((date, len(matches), news_list))

        # Identifier les patterns avec suffisamment de mentions
        for pattern, mentions in pattern_mentions.items():
            if len(mentions) >= min_mentions:
                # Calculer un score basé sur la fréquence et la récence
                frequency_score = min(len(mentions) / min_mentions, 1.0)

                # Trouver le premier et dernier jour de mention
                dates = [m[0] for m in mentions]
                first_date = min(dates)
                last_date = max(dates)

                # Si le changement de narratif est récent (dans les derniers jours)
                if (datetime.now().date() - last_date).days <= timeframe_days:
                    # Collecter les sources qui mentionnent ce narratif
                    sources = set()
                    all_news_with_pattern = []
                    for _, _, news_list in mentions:
                        for news in news_list:
                            source = news.get("source", "unknown")
                            sources.add(source)
                            all_news_with_pattern.append(news)

                    # Le score final prend en compte la diversité des sources
                    diversity_score = min(
                        len(sources) / 3, 1.0
                    )  # Normaliser à 3 sources différentes
                    final_score = (frequency_score * 0.6) + (diversity_score * 0.4)

                    # Créer l'événement
                    event = {
                        "type": EventType.NARRATIVE.value,
                        "pattern": pattern,
                        "first_mention": first_date.isoformat(),
                        "latest_mention": last_date.isoformat(),
                        "frequency": len(mentions),
                        "sources": list(sources),
                        "score": final_score,
                        "severity": self._determine_severity(final_score),
                        "example_news": (
                            all_news_with_pattern[0] if all_news_with_pattern else None
                        ),
                        "detected_at": datetime.now().isoformat(),
                    }

                    # Ne pas ajouter de doublons
                    if not self._is_duplicate_event(event):
                        narrative_events.append(event)
                        self.detected_events.append(event)

        logger.info(
            f"Détection de changements de narratif: {len(narrative_events)} trouvés"
        )
        return narrative_events

    def analyze_all_events(
        self,
        news_data: List[Dict],
        regulatory_threshold: float = 0.6,
        fundamental_threshold: float = 0.55,
        narrative_min_mentions: int = 3,
    ) -> Dict[str, List[Dict]]:
        """
        Analyse tous les types d'événements critiques.

        Args:
            news_data: Liste de dictionnaires contenant les actualités
            regulatory_threshold: Seuil pour les événements réglementaires
            fundamental_threshold: Seuil pour les événements fondamentaux
            narrative_min_mentions: Mentions minimales pour les changements de narratif

        Returns:
            Dictionnaire avec tous les événements regroupés par type
        """
        regulatory = self.detect_regulatory_events(news_data, regulatory_threshold)
        fundamental = self.detect_fundamental_events(news_data, fundamental_threshold)
        narrative = self.detect_narrative_changes(
            news_data, min_mentions=narrative_min_mentions
        )

        # Agréger tous les événements
        all_events = {
            EventType.REGULATORY.value: regulatory,
            EventType.FUNDAMENTAL.value: fundamental,
            EventType.NARRATIVE.value: narrative,
        }

        # Trier les événements par sévérité et date
        for event_type, events in all_events.items():
            all_events[event_type] = sorted(
                events,
                key=lambda x: (
                    EventSeverity[x["severity"]].value if "severity" in x else 0,
                    x.get("date", ""),
                ),
                reverse=True,
            )

        return all_events

    def get_alert_recommendations(self, events: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Génère des recommandations d'alerte basées sur les événements détectés.

        Args:
            events: Dictionnaire d'événements par type

        Returns:
            Liste de recommandations d'alerte
        """
        alerts = []

        # Considérer tous les événements critiques et high
        critical_events = []
        for event_type, event_list in events.items():
            for event in event_list:
                if event.get("severity") in ["CRITICAL", "HIGH"]:
                    critical_events.append(event)

        # Trier par sévérité
        critical_events = sorted(
            critical_events,
            key=lambda x: EventSeverity[x["severity"]].value,
            reverse=True,
        )

        # Générer des alertes pour les événements les plus critiques
        for event in critical_events:
            alert = {
                "event_type": event["type"],
                "severity": event["severity"],
                "message": self._generate_alert_message(event),
                "timestamp": datetime.now().isoformat(),
                "event_details": event,
            }
            alerts.append(alert)

        return alerts

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extrait les entités (entreprises, projets, etc.) du texte.
        Méthode simple basée sur les majuscules.

        Args:
            text: Texte à analyser

        Returns:
            Liste des entités détectées
        """
        # Version simple: chercher les mots avec majuscules
        # Pour une version plus sophistiquée, utiliser un modèle NER (Named Entity Recognition)
        words = text.split()
        entities = []

        # Chercher des séquences de mots commençant par une majuscule
        current_entity = []

        for word in words:
            # Nettoyage du mot
            clean_word = re.sub(r"[^\w\s]", "", word)

            if clean_word and clean_word[0].isupper():
                current_entity.append(clean_word)
            elif current_entity:
                if len(current_entity) > 0:
                    entities.append(" ".join(current_entity))
                current_entity = []

        # Ajouter la dernière entité
        if current_entity:
            entities.append(" ".join(current_entity))

        # Filtrer les entités (longueur minimale, pas de mots communs)
        common_words = {
            "le",
            "la",
            "les",
            "des",
            "du",
            "au",
            "aux",
            "un",
            "une",
            "et",
            "ou",
            "a",
        }
        filtered_entities = [
            e for e in entities if len(e) > 2 and e.lower() not in common_words
        ]

        return list(set(filtered_entities))

    def _is_duplicate_event(self, event: Dict) -> bool:
        """
        Vérifie si un événement est un doublon.

        Args:
            event: Événement à vérifier

        Returns:
            True si l'événement est un doublon, False sinon
        """
        # Pour les événements réglementaires et fondamentaux, vérifier le titre
        if event["type"] in [EventType.REGULATORY.value, EventType.FUNDAMENTAL.value]:
            event_title = event.get("title", "").lower()

            for past_event in self.detected_events:
                if past_event["type"] == event["type"]:
                    past_title = past_event.get("title", "").lower()

                    # Si les titres sont très similaires
                    if self._similarity(event_title, past_title) > 0.8:
                        # Vérifier si l'événement est récent
                        try:
                            past_date = datetime.fromisoformat(
                                past_event.get("detected_at", "")
                            )
                            hours_diff = (
                                datetime.now() - past_date
                            ).total_seconds() / 3600

                            if hours_diff < self.min_event_interval:
                                return True
                        except (ValueError, TypeError):
                            pass

        # Pour les changements de narratif, vérifier le pattern
        elif event["type"] == EventType.NARRATIVE.value:
            event_pattern = event.get("pattern", "").lower()

            for past_event in self.detected_events:
                if past_event["type"] == EventType.NARRATIVE.value:
                    if past_event.get("pattern", "").lower() == event_pattern:
                        # Vérifier si l'événement est récent
                        try:
                            past_latest = datetime.fromisoformat(
                                past_event.get("latest_mention", "")
                            )
                            event_latest = datetime.fromisoformat(
                                event.get("latest_mention", "")
                            )

                            if event_latest <= past_latest:
                                return True
                        except (ValueError, TypeError):
                            pass

        return False

    def _similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité entre deux textes (méthode simple).

        Args:
            text1: Premier texte
            text2: Deuxième texte

        Returns:
            Score de similarité entre 0 et 1
        """
        # Méthode simple: proportion de mots communs
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0

        common = words1.intersection(words2)
        return len(common) / max(len(words1), len(words2))

    def _determine_severity(self, score: float) -> str:
        """
        Détermine la sévérité d'un événement basée sur son score.

        Args:
            score: Score de l'événement (0 à 1)

        Returns:
            Niveau de sévérité
        """
        if score >= 0.85:
            return EventSeverity.CRITICAL.name
        elif score >= 0.7:
            return EventSeverity.HIGH.name
        elif score >= 0.5:
            return EventSeverity.MEDIUM.name
        elif score >= 0.3:
            return EventSeverity.LOW.name
        else:
            return EventSeverity.INFO.name

    def _generate_alert_message(self, event: Dict) -> str:
        """
        Génère un message d'alerte pour un événement.

        Args:
            event: Événement à traiter

        Returns:
            Message d'alerte
        """
        if event["type"] == EventType.REGULATORY.value:
            terms = ", ".join(event.get("matched_terms", [])[:3])
            return f"ALERTE {event['severity']}: Événement réglementaire détecté - {event.get('title', '')}. Mots-clés: {terms}"

        elif event["type"] == EventType.FUNDAMENTAL.value:
            entities = ", ".join(event.get("entities", [])[:3])
            return f"ALERTE {event['severity']}: Évolution fondamentale - {event.get('title', '')}. Entités: {entities}"

        elif event["type"] == EventType.NARRATIVE.value:
            return f"ALERTE {event['severity']}: Changement de narratif détecté - '{event.get('pattern', '')}'. Mentionné {event.get('frequency', 0)} fois."

        else:
            return f"ALERTE {event['severity']}: Événement critique détecté."


# Fonctions utilitaires


def preprocess_news(news_data: List[Dict]) -> List[Dict]:
    """
    Prétraite les données d'actualités pour l'analyse.

    Args:
        news_data: Liste de dictionnaires d'actualités

    Returns:
        Liste de dictionnaires d'actualités prétraitées
    """
    processed_data = []

    for news in news_data:
        processed = news.copy()

        # Normaliser les champs
        if "title" not in processed or not processed["title"]:
            processed["title"] = ""

        if "content" not in processed or not processed["content"]:
            processed["content"] = ""

        if "date" not in processed:
            processed["date"] = datetime.now().isoformat()
        elif not isinstance(processed["date"], str):
            try:
                processed["date"] = processed["date"].isoformat()
            except:
                processed["date"] = datetime.now().isoformat()

        if "source" not in processed:
            processed["source"] = "unknown"

        processed_data.append(processed)

    return processed_data


def filter_news_by_relevance(
    news_data: List[Dict], asset_keywords: List[str], threshold: float = 0.1
) -> List[Dict]:
    """
    Filtre les actualités par pertinence pour un actif spécifique.

    Args:
        news_data: Liste de dictionnaires d'actualités
        asset_keywords: Liste de mots-clés liés à l'actif
        threshold: Seuil de score pour considérer une actualité comme pertinente

    Returns:
        Liste d'actualités filtrées
    """
    relevant_news = []

    for news in news_data:
        title = news.get("title", "").lower()
        content = news.get("content", "").lower()
        full_text = title + " " + content

        # Calculer un score de pertinence
        score = 0
        for keyword in asset_keywords:
            if keyword.lower() in title:
                score += 0.5  # Poids plus important pour les mentions dans le titre
            if keyword.lower() in content:
                score += 0.2

        # Normaliser le score
        max_possible_score = 0.5 * len(asset_keywords) + 0.2 * len(asset_keywords)
        relevance_score = score / max_possible_score if max_possible_score > 0 else 0

        # Ajouter le score de pertinence
        news["relevance_score"] = relevance_score

        # Filtrer par seuil de pertinence
        if relevance_score >= threshold:
            relevant_news.append(news)

    # Trier par pertinence
    return sorted(
        relevant_news, key=lambda x: x.get("relevance_score", 0), reverse=True
    )
