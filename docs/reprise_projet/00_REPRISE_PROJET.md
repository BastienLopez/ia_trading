# Reprise du projet AI Trading

## Statut reel au 2026-07-21

Le projet existant est conserve. Il contient deja les phases fonctionnelles de collecte, analyse LLM, RL, risque, backtesting, API et interfaces decrites dans le README. La mission P0 a P4 est une remise a niveau progressive : reduire les doublons, clarifier les proprietaires, stabiliser les contrats et valider le comportement existant. Ce n'est pas une reconstruction.

Pendant cet audit, aucun fichier de code, configuration, dependance ou test n'a ete modifie. Aucun test, reseau, Docker, entrainement, backtest ou ordre de trading n'a ete execute.

## Regle de conduite

1. Conserver par defaut le comportement fonctionnel existant.
2. Regrouper uniquement apres avoir trace les appelants, les tests et les points d'entree.
3. Migrer un appelant a la fois avec une couche de compatibilite courte si necessaire.
4. Ne supprimer un fichier que lorsque son absence d'utilisation est prouvee, que son remplacement est valide et qu'une decision explicite le confirme.
5. Ne jamais melanger securisation, migration metier, optimisation et suppression dans le meme lot.

## Resultat de l'audit statique

- 609 fichiers non ignores repertories par `rg --files`, dont 504 fichiers Python et 166 fichiers de tests detectes. L'inventaire physique CSV inclut aussi les artefacts ignores presents localement.
- 5 056 fonctions et 683 classes detectees par AST.
- 0 erreur de syntaxe Python apres lecture UTF-8 avec gestion du BOM.
- Les risques principaux sont les chemins concurrents et les doublons, pas l'absence de fonctionnalites : API `ai_trading/api.py` et package `ai_trading/api/`, trois `risk_manager.py`, deux `data_processor.py`, deux `technical_indicators.py`, deux environnements multi-actifs, deux agents DQN/SAC et plusieurs exemples/tests miroirs.

## Lecture dans l'ordre

1. [Architecture actuelle](01_ARCHITECTURE_ACTUELLE.md)
2. [Etat des phases historiques 1 a 4](02_ETAT_PHASES_1_A_4.md)
3. [Audit de dette et priorites](03_AUDIT_CODE_ET_DETTE_TECHNIQUE.md)
4. [Registre de regroupement](04_AUDIT_REGROUPEMENT.md)
5. [Strategie de validation](05_STRATEGIE_TESTS.md)
6. [Plan P0 a P4](13_PLAN_EXECUTION_DETAILLE_P0_A_P4.md)
7. [Rapport P0](14_RAPPORT_EXECUTION_P0.md), [rapport P1](15_RAPPORT_EXECUTION_P1.md) et [rapport P2](16_RAPPORT_EXECUTION_P2.md)

Les CSV `06`, `11` et `12` sont des inventaires statiques. Ils decrivent l'existant et ne constituent pas une demande de suppression.

## Etat des gates

| Gate | Etat | Condition de sortie |
| --- | --- | --- |
| P0 - baseline et securite | PASS | configuration runtime testee, defaults locaux et Compose valide |
| P1 - contrats data et validation | PASS | qualite OHLCV, folds progressifs et indicateurs historiques valides |
| P2 - consolidation progressive | PASS | facade API migree sans suppression, health check FastAPI valide |
| P3 - mesure et qualite | NON DEMARRE | tests classes et performances mesurees avant optimisation |
| P4 - exploitation paper | NON DEMARRE | fonctionnement observable et procedure de reprise documentee |
