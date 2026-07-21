# Plan d'action P0 a P4

| Phase | Objectif | Changement attendu | Gate |
| --- | --- | --- | --- |
| P0 | etablir le baseline sans casser | inventaire des points d'entree, config partagee documentee, comportements de reference | aucun comportement existant perdu |
| P1 | stabiliser data/features/validation | contrats et tests de parite autour des modules actifs | pipeline reproductible sur fixture |
| P2 | consolider par domaine | facades, adaptateurs et migration de quelques appelants a la fois | doublons classes avec proprietaire clair |
| P3 | mesurer et fiabiliser | classification des tests, dependances separees, profils avant/apres | qualite et ressources mesurees |
| P4 | operer en paper de facon lisible | runbook, audit, health, dashboard/API coherents | aucune execution live implicite |

Le detail executable, les fichiers candidats, les validations et les prompts sont dans `13_PLAN_EXECUTION_DETAILLE_P0_A_P4.md`.
