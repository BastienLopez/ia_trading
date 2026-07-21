# Plan d'execution detaille P0 a P4

Ce plan remet a niveau le projet existant. Chaque phase est independante, revue avant la suivante, et limitee a un petit lot. Le mot `regrouper` signifie facade, contrat, adaptateur et migration progressive. Il ne signifie jamais suppression massive.

## Regles non negociables

1. Lire `AGENTS.md`, `PROJECT_MEMORY.md` s'il existe, le README et ce document.
2. Utiliser Codebase Memory pour tracer les appels; si indisponible, l'indiquer et faire des recherches ciblees.
3. Inspecter `git status --short` avant/apres. Preserver tout changement utilisateur.
4. Avant de modifier un module, lister ses imports entrants, ses imports sortants, ses scripts, ses exemples et ses tests.
5. Avant de fusionner, capturer une parite de sortie sur une fixture locale.
6. Avant de supprimer, obtenir les quatre preuves : aucun appelant, remplacement valide, test/fixture de parite, accord explicite.
7. Ne pas lancer test global, Docker, GPU, reseau, entrainement, backtest lourd ou execution live sans accord explicite.
8. Arreter a la gate de phase et fournir le rapport factuel avant de continuer.

## P0 - Baseline, securite et cartographie executable

### Objectif

Rendre le comportement actuel lisible et non ambigu, sans le refactorer. P0 ne change pas les algorithmes, ne deplace pas les modules et ne retire pas de fonctionnalite.

### Fichiers candidats

| Sujet | Fichiers a analyser en premier | Livrable |
| --- | --- | --- |
| Points d'entree | `ai_trading/api.py`, `ai_trading/api/run.py`, `ai_trading/dashboard/run.py`, `web_app/app.py`, `docker/docker-compose.yml` | tableau commande -> module charge -> interface -> statut |
| Configuration | `ai_trading/config.py`, `ai_trading/config/trading_pairs.py`, `web_app/config.py`, `.env.example`, `.gitignore` | matrice de variables, valeurs par defaut et proprietaire |
| Execution | `execution/`, `orders/`, `risk/`, `rl/risk_manager.py` | carte ordre -> risque -> fill -> persistance |
| Donnees | `data/`, `data_processor.py`, collecteurs `utils/` | carte source -> schema -> stockage -> consommateur |
| Tests | `ai_trading/tests/`, `web_app/tests/`, tests dans modules | etiquetage sans execution |

### Changements autorises

- Corriger une configuration manifestement dangereuse ou une documentation fausse, avec diff minimal.
- Ajouter un inventaire, une matrice de compatibilite ou un test de configuration pur.
- Ajouter des commentaires/deprecations sans modifier la semantique.

### Changements interdits

- Renommer `api.py` ou le package `api/` avant tracer les imports.
- Modifier les agents, modeles, collecteurs, strategies ou dashboards.
- Retirer `.env`, dependances, exemples, tests ou dossiers entiers.

### Validation P0

1. `git status` proprement interprete.
2. Aucun secret affiche dans le diff ou les logs.
3. Carte complete des points d'entree, contrats de configuration et chemins de risque.
4. Liste des candidats P1/P2 avec appelants connus.

### Prompt P0

```text
Execute uniquement P0 du plan docs/reprise_projet/13_PLAN_EXECUTION_DETAILLE_P0_A_P4.md.
Le projet fonctionne deja : preserve chaque comportement et ne supprime aucun module, test, exemple, dependance ou dossier.

Lis AGENTS.md, PROJECT_MEMORY.md s'il existe et README.md. Utilise Codebase Memory en premier; si indisponible, explique le fallback et trace les imports de facon ciblee.

Cartographie les points d'entree FastAPI/API package/Dash/Flask/Docker, les configurations, les chemins data, le risque et l'execution. Documente les collisions api.py/api package et les doublons sans les corriger encore. Ajoute seulement une preuve statique ou un test pur de configuration si ses imports sont connus et sans effet de bord.

Ne lance ni test global, ni Docker, ni reseau, ni GPU, ni entrainement, ni backtest, ni execution exchange. Termine par les fichiers lus/modifies, la matrice des appelants, les risques et la gate P0.
```

## P1 - Contrats data, features et validation de parite

### Objectif

Faire coexister les modules data existants derriere des contrats stables. Le but est de mesurer les equivalences, pas de remplacer les collecteurs, indicateurs ou datasets.

### Lots recommandes

| Lot | Fichiers candidats | Action precise | Preuve |
| --- | --- | --- | --- |
| P1.1 contrats marche | `data/market_data.py`, `data/order_flow.py`, `data_processor.py`, `utils/orderbook_collector.py` | definir ou documenter `Candle`, `Trade`, `OrderBook`, horodatage, source et schema | fixture locale + validation de schema |
| P1.2 features | `data/technical_indicators.py`, `indicators/*`, `rl/technical_indicators.py`, `utils/technical_analyzer.py` | comparer les sorties par famille; choisir une facade, pas supprimer les implementations | test de parite sur OHLCV fige |
| P1.3 validation | `validation/*`, `rl/temporal_cross_validation.py`, `ml/backtesting/*` | expliciter split temporel, frais, spread, slippage et metriques | test anti-leakage local |
| P1.4 universe | `config/trading_pairs.py`, collecteurs actifs | documenter selection d'actifs et filtres existants | fixture de marche + resultat attendu |

### Regles P1

- Chaque contrat indique source, exchange, symbole, timeframe, UTC, version et fraicheur.
- Le resultat existant sert de reference avant extraction vers une facade.
- Les donnees synthetiques, si elles existent, sont etiquetees dans les fixtures; elles ne sont pas renommees ou retirees pendant P1.
- Le premier changement cible un seul contrat et au plus deux appelants.

### Validation P1

1. Fixture locale minuscule versionnee.
2. Comparaison ancien/nouveau pour le contrat concerne.
3. Test de tri, doublons, trous et fuite temporelle.
4. Aucun appel reseau ou ordre prive.

### Prompt P1

```text
Execute uniquement P1 apres validation P0. Ne remplace ni ne supprime les modules data existants.

Trace les appelants de data/market_data.py, data/order_flow.py, data_processor.py, les indicateurs data/indicators/rl/utils et les validateurs/backtests. Choisis un seul sous-lot P1.1 a P1.4. Cree une facade ou un contrat compatible et migre au plus deux appelants avec adaptateurs temporaires.

Avant modification, produis une fixture locale et une comparaison de sorties. Apres modification, lance seulement les tests purs de schema/parite/anti-leakage annonces. Ne lance ni reseau, ni Docker, ni entrainement, ni backtest lourd. Documente les differences fonctionnelles trouvees et conserve les implmentations non migrees.
```

## P2 - Consolidation progressive des responsabilites

### Objectif

Retirer l'ambiguite des chemins concurrents tout en preservant les implementations utilisees. P2 n'autorise pas une purge de sous-systemes.

### Ordre de travail

| Lot | Fichiers candidats | Resultat attendu |
| --- | --- | --- |
| P2.1 API/interface | `ai_trading/api.py`, `ai_trading/api/`, appels Docker, dashboard, web_app | un point d'entree declare et les autres surfaces compatibles/documentees |
| P2.2 risque | `risk/`, `execution/risk_manager.py`, `rl/risk_manager.py` | contrat commun de decision et adaptateurs de chaque moteur |
| P2.3 ordre/execution | `orders/`, `execution/`, `ml/backtesting/execution_model.py` | schema commun ordre/fill/frais; aucun changement de strategie |
| P2.4 RL | `rl/agents/`, `rl/*agent.py`, runners, buffers | roles separes, reexports reduits, variantes preservees |
| P2.5 services/utilitaires | `utils/`, `optim/`, LLM, reporting | proprietaire de chaque utilitaire partage; imports directs remplaces graduellement |

### Procedure obligatoire par lot

1. Construire la matrice `fichier actuel -> role -> appelants -> test -> facade cible`.
2. Definir le contrat public cible et son test de contrat.
3. Adapter un premier appelant; comparer sortie/effets de bord.
4. Adapter le second appelant seulement si le premier est valide.
5. Marquer l'ancien chemin deprecated, avec date/condition de retrait.
6. Ne proposer une suppression qu'au lot suivant, avec preuves, jamais dans la meme modification.

### Validation P2

- Aucun import ajoute vers une couche plus haute ou cycle nouveau.
- API, risque et ordre ont un proprietaire documente.
- Les adaptateurs conservent les signatures requises par les appelants non migres.
- Les tests de parite/contrat cibles passent.

### Prompt P2

```text
Execute uniquement P2 apres P0 et P1. Le projet est fonctionnel et ne doit pas etre reconstruit.

Choisis un seul lot P2.1 a P2.5. Avant tout edit, trace les imports entrants/sortants, scripts, tests et exemples du candidat. Cree une matrice de migration et un contrat de parite. Introduis une facade ou un adaptateur, puis migre au maximum deux appelants. Garde les chemins historiques avec une compatibilite documentee.

Il est interdit de supprimer un dossier, une famille de tests, une interface, une dependance ou un framework pendant P2 sans une demande explicite et les quatre preuves du plan. Ne lance ni test global, ni Docker, ni GPU, ni reseau, ni entrainement. Termine avec le diff, les appels migres, les compatibilites restantes et la gate P2.
```

## P3 - Qualite, dependances et optimisation mesuree

### Objectif

Rendre la base consolidee rapide et reproductible sans casser les environnements existants.

### Actions

| Lot | Fichiers candidats | Action |
| --- | --- | --- |
| P3.1 tests | tous les repertoires `tests`, tests dans packages, `examples` | etiqueter et deplacer seulement les tests deja couverts par parite |
| P3.2 dependances | `requirements.txt`, `setup.py`, Docker | table `import runtime -> paquet`; extraire base/dev/gpu sans supprimer les options |
| P3.3 performance | pipeline P1 retenu, `optim/`, utilitaires caches | profiler fixture figee avant/apres; ne retenir que gain mesure |
| P3.4 hygiene | BOM, artefacts, docs | corrections mecanique isolees et non semantiques |

### Validation P3

- tests classes, commandes explicites et durees connues;
- installation de base documentee sans retirer l'environnement historique;
- metriques CPU/RAM/temps comparables pour chaque optimisation;
- pas de gain affirme sans baseline.

### Prompt P3

```text
Execute uniquement P3 apres P2. Conserve les dependances et options actuelles tant que leurs imports n'ont pas ete inventories.

Classe les tests, construis la table import -> dependance et profile un seul pipeline/fixture deja valide. Propose une separation base/dev/gpu reversible, avec fichiers de compatibilite si necessaire. N'optimise pas avant mesure et ne lance aucun GPU, Docker, reseau ou test global. Fournis les metriques avant/apres, les dependances impactees et les risques de migration.
```

## P4 - Exploitation locale et paper trading observable

### Objectif

Faire fonctionner les surfaces existantes de facon lisible, auditable et sure. P4 n'ajoute pas de trading live.

### Actions

| Sujet | Fichiers candidats | Action |
| --- | --- | --- |
| Orchestration | scripts API/dashboard/Docker existants | declarer une commande principale et les commandes de compatibilite |
| Observabilite | loggers, performance tracker, dashboards | schema commun d'evenement, correlation de run et health |
| Paper/backtest | execution, orders, risk, backtest | rendre visibles decision, frais, fill, refus et etat |
| Operations | README, Docker, runbook | preconditions, demarrage, arret, reprise, incident |

### Validation P4

1. Simulation locale de decision/refus avec fixture.
2. Verification que chaque interface est lecture seule ou identifie clairement ses actions.
3. Health et logs correlables a un run.
4. Procedure arret/reprise testee dans un lot explicitement approuve.

### Prompt P4

```text
Execute uniquement P4 apres P3. Preserve les API, dashboards et scripts existants; commence par tracer leurs utilisateurs et leurs commandes.

Rends observable le parcours paper/backtest deja valide : configuration, data, decision, risque, ordre/fill, logs et health. Unifie les metriques par contrat et documente une commande principale sans supprimer les anciennes commandes. Ne soumets aucun ordre live, ne demarre aucun processus permanent, Docker ou reseau sans validation explicite.

Termine par le runbook, les checks locaux proposes, les limites et la gate P4.
```

## Gate avant P5

P5 est autorisee seulement si P0-P4 ont chacune un rapport `PASS` ou `PARTIAL` explicite, avec comportements conserves, migrations restantes, tests executes et risques connus. Une gate incomplete renvoie au plus petit lot de la phase concernee; elle ne justifie pas une reecriture globale.
