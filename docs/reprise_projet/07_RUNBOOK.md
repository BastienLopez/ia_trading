# Runbook de reprise

## Avant tout lot P0-P4

1. Lire `AGENTS.md`, `PROJECT_MEMORY.md` s'il existe, et `13_PLAN_EXECUTION_DETAILLE_P0_A_P4.md`.
2. Executer `git status --short`; ne jamais ecraser un changement present.
3. Interroger Codebase Memory. S'il est indisponible ou pointe un autre depot, l'indiquer et utiliser des recherches ciblees.
4. Etablir une liste finie de fichiers et de tests impactes.
5. Capturer le comportement de reference sur fixture avant tout regroupement.

## Pendant le lot

- Une responsabilite, un contrat et un petit groupe d'appelants.
- Aucune suppression sans matrice de migration approuvee.
- Aucun test global, Docker, GPU, reseau, entrainement ou backtest lourd sans autorisation explicite.
- Consigner dans le rapport de phase les decisions, les compatibilites temporaires et les preuves.

## Fin du lot

1. Executer uniquement les checks annonces.
2. Relire le diff et verifier les imports sortants/entrants.
3. Mettre a jour le rapport de phase avec le statut reel `PASS`, `PARTIAL` ou `FAIL`.
4. Demander validation avant la phase suivante.
