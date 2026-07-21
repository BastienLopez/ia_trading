# Strategie de tests et validation

## Regle

L'audit n'a execute aucun test. Chaque phase introduit ou deplace seulement les tests lies a son lot, apres lecture de leurs imports et de leurs effets de bord.

## Pyramide cible

| Niveau | But | Entree | Interdits par defaut |
| --- | --- | --- | --- |
| Statique | syntaxe, imports, doublons, secrets | code source | reseau, Docker, GPU |
| Unitaire pur | contrats, features, risque, fees | fixtures locales minuscules | exchange, DB, timeouts longs |
| Parite | conserver une sortie lors d'un regroupement | fixture + implementation ancienne/nouvelle | donnees live |
| Integration locale | pipeline data -> decision -> paper/backtest | snapshot versionne | ordre prive, service externe |
| Integration opt-in | lecture exchange ou interface | environnement explicitement autorise | CI par defaut |
| Performance | CPU/RAM/temps avant-apres | dataset fige | optimisation sans baseline |

## Classification a faire avant P1

Les 166 fichiers de tests detectes doivent etre etiquetes `unit`, `contract`, `integration`, `slow`, `gpu`, `network` ou `example`. Un test sans etiquette ni fixture controlee ne sera pas lance en lot global.

## Invariants de non-regression

- meme schema de donnees et memes timestamps pour une migration data;
- memes decisions de risque pour les memes entrees;
- memes fills/frais attendus pour une migration execution;
- aucune lecture future dans les features ou splits temporels;
- aucun ordre exchange pendant les tests normaux.
