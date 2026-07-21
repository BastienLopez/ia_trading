# Suivi de reprise des phases du README

Ce document suit les phases historiques du `README.md`. Il est distinct des rapports
P0/P1/P2 de fondation technique: un `PASS` de fondation ne valide pas une phase
historique du README.

## Phase 1 - Collecte et prétraitement

Statut: **EN COURS**.

Sous-lots repris et validés:

- `ai_trading/utils/enhanced_data_collector.py`: appels HTTP resilients, timeout
  injectable, retry teste, arguments de fusion corriges, provenance `source` ajoutee.
- `ai_trading/utils/enhanced_preprocessor.py`: pas de backfill futur, capitalisations
  en `float32`, normalisation train/validation/test explicite, NLP optionnel sans
  telechargement reseau au demarrage.
- `ai_trading/utils/enhanced_cache.py`: compatibilite Redis Cluster moderne et fallback
  Redis standard.
- `ai_trading/utils/blockchain_data_collector.py`: conversion correcte des timestamps
  Unix Ethereum.

Validations executees:

- Collecteur historique: 10 tests OK.
- Contrats Phase 1 collecte/pretraitement: 8 tests OK.
- Preprocesseur historique: 10 tests OK.
- Cache ameliore: 11 tests OK.
- Collecteur blockchain synchrone: 12 tests OK.
- Resilience HTTP: 24 tests OK.

Validation Docker actuelle:

- 75 tests P1 OK, dont 11 tests du collecteur blockchain asynchrone.

Points restants avant `PASS` P1:

- Implémenter LRU et compression applicatifs dans le cache distribué.
- Faire appliquer les priorités aux requêtes async, pas seulement à l'ordre de
  création des tâches.
- Relier le fallback multi-source de `ResilientRequester` au collecteur de prix.

## Phase 2 - Analyse de sentiment

Statut: **EN COURS**.

Sous-lots repris et validés statiquement:

- `ai_trading/llm/sentiment_analysis/social_analyzer.py`: le score continu du
  modèle est conservé, le rapport social utilise le contrat de rapport existant
  et accepte les données Reddit sans hashtags.
- `ai_trading/tests/llm/test_social_analyzer.py`: ajout du contrat Reddit pour
  la génération de rapport social.
- `ai_trading/llm/sentiment_analysis/sentiment_pipeline.py`: orchestration
  unique sentiment, crédibilité, propagation et contexte marché.

Validations exécutées:

- Compilation des modules et du test social: OK.
- 43 tests P2 OK : news/social, fake news, utilitaires, intégration RL et contexte.
- Pipeline unifié: 1 test OK.

Bloquants restants avant `PASS` Phase 2:

- Ajouter un dataset crypto versionné et un artefact de fine-tuning évalué avant
  de déclarer l'objectif « fine-tuning sur données crypto » terminé.

## Phases 3 et 4

Statut: **NON DEMARREES POUR LA REPRISE HISTORIQUE**. Elles ne doivent pas etre
declarees validees sur la seule base des rapports de fondation technique.
