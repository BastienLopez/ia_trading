# Reprise du projet ia_trading

> État arrêté le 24 juillet 2026. Ce document est le point de départ après
> une pause : il sépare ce qui fonctionne techniquement de ce qui est prouvé
> économiquement. Il ne constitue pas une autorisation de trading réel.

## Résumé en une minute

Le pipeline P1 + P2 + P3 + P4 s'exécute dans Docker `ai_api`, génère des
artefacts auditables et ne présente pas de fuite temporelle connue dans le
chemin P4 corrigé. En revanche, **aucun candidat P3 n'est validé comme stable
ou rentable** sur la campagne étudiée. P4 fonctionne comme feature de recherche
et de supervision, mais n'améliore pas P3 de manière robuste.

Ne pas remplacer le candidat historique par un réglage choisi sur une seule
fenêtre. Ne pas activer de trading réel, d'allocation réelle ou d'ordre réel.

## Références à conserver

Racine de la campagne reproductible :

```text
ai_trading/info_retour/p3_p4_protocol_w4_8_20260724_211029/
```

| Élément | Emplacement | Rôle |
|---|---|---|
| Snapshot P1 figé | `market_snapshot/` | Même OHLCV BTC/ETH pour tous les arms ; hash contrôlé. |
| Baseline P3 | `p3_only/20260724T191040Z/` | Candidat historique, sans P4. |
| P3 + P4 corrigé | `p3_p4_corrected/20260724T200443Z/` | Même snapshot, sentiment P2 et features P4. |
| Ablation blend=0 | `p3_blend0_ablation/20260724T205840Z/` | Même run, seul `signal_action_blend` forcé à 0. |
| Verdict P3/P4 | `p3_vs_p4_comparison.json` | Comparaison pré-enregistrée, pas de sélection a posteriori. |

Un ancien dossier `p3_p4/` a été interrompu car il utilisait encore une
calibration scikit-learn non strictement temporelle. Il est conservé comme
trace de diagnostic et **ne doit pas être utilisé**.

## Ce qui fonctionne réellement

### P1 — Marché

- Le walk-forward récupère BTC/USDT et ETH/USDT, puis les aligne sur une grille
  temporelle commune.
- Un snapshot Parquet immuable peut être exporté et relu. Son hash est vérifié
  avant une comparaison ; P3 seul et P3+P4 ne peuvent donc pas être comparés
  sur deux téléchargements légèrement différents.
- XAU/USD reste limité au timeframe `1d` avec la source Yahoo actuelle. Ne pas
  le mélanger à un run intraday BTC/ETH.

### P2 — Sentiment

- Les observations P2 requises par P4 sont horodatées, avec actif, timeframe,
  score, qualité et source.
- Le run P4 réel a utilisé
  `ai_trading/info_retour/data/sentiment/observations.parquet` pour BTC et ETH
  en `1d`.
- Cela prouve la lecture et l'alignement causal des données P2 ; cela **ne
  prouve pas** que le sentiment est prédictif.

### P3 — RL, exécution simulée et garde-fous

- `run_real_multi_asset_walk_forward` entraîne et évalue les agents sur chaque
  seed/fenêtre, écrit actions, fills, positions, ledger, métriques et courbes.
- Les frais, slippage, positions short, stop-loss et ledger sont inclus dans la
  simulation ; aucune exécution réelle n'est branchée.
- Les critères d'éligibilité rejettent un candidat instable entre seeds, un
  ledger déficitaire ou une diversité d'actions insuffisante.

### P4 — Features de prédiction et audit

- Les modèles P4 sont entraînés uniquement sur le passé et calibrés sur le
  segment temporel postérieur. Une calibration invalide provoque une abstention,
  pas un repli sur un modèle brut.
- P4 a généré dans le run corrigé : 1 760 prédictions, toutes marquées
  `calibrated=true`, et 930 abstentions.
- Les artefacts P4 comprennent `p4_predictions.parquet`,
  `p4_training_reports.json`, `p4_quality_metrics.json`,
  `p4_features_*.parquet` et `p4_feature_manifest.json`.
- Les métriques Brier, ECE, coverage, largeur d'intervalle et taux
  d'abstention sont persistés. BTC a accepté 10 réentraînements et rejeté 2
  calibrations ; ETH a accepté 8 réentraînements et rejeté 4. Ces rejets sont
  attendus quand l'historique causal ne suffit pas.

### Tests exécutés

Runtime de référence : Docker `ai_api`.

```text
47 passed in 114.66s
```

Le lot couvre `prediction_model`, P4/P1, features walk-forward, protocole de
comparaison P3/P4, masques d'actions, risque short et walk-forward RL. Un test
vert prouve le comportement testé ; il ne prouve pas la rentabilité.

## Résultats économiques : aucun candidat promu

Protocole utilisé : fenêtres chronologiques **4 à 8**, 3 seeds (`73 211 997`),
frais identiques, validation uniquement, 100 bougies d'évaluation par fenêtre.
Les données OHLCV sont exactement les mêmes grâce au snapshot.

| Fenêtre | P3 seul | Buy & Hold | P3 + P4 | Delta P4 - P3 |
|---:|---:|---:|---:|---:|
| 4 | +17,10 % | +25,71 % | +16,58 % | -0,52 pt |
| 5 | +26,00 % | +40,43 % | +24,19 % | -1,81 pt |
| 6 | -17,43 % | -4,20 % | -16,55 % | +0,88 pt |
| 7 | +20,49 % | +42,43 % | +18,21 % | -2,28 pt |
| 8 | -15,58 % | -30,48 % | -15,49 % | +0,09 pt |

Verdict automatique P3+P4 : **rejeté**.

- P4 n'est meilleur que sur 2/5 fenêtres.
- Delta médian P4 - P3 : **-0,52 point**.
- P3+P4 reste sous Buy & Hold en médiane.
- Au moins une fenêtre a un profit factor inférieur ou égal à 1 et une
  expectancy négative.

L'ablation `signal_action_blend=0` est également rejetée : elle réduit la
perte sur la fenêtre 6 (-0,95 % contre -17,43 %), mais dégrade les fenêtres 4,
5, 7 et 8. Sa performance médiane est négative. Elle ne doit pas remplacer le
candidat de référence.

## Pourquoi aucun candidat n'est stable

Ce diagnostic est une **inférence à partir des artefacts**, pas une certitude
sur le marché futur.

1. Les performances changent fortement selon la fenêtre et le seed. Plusieurs
   fenêtres ont des seeds tous sous Buy & Hold ; les règles d'éligibilité les
   rejettent donc.
2. Les phases baissières restent mal gérées : les fenêtres 6 et 8 ont un
   profit factor faible et une expectancy négative malgré une protection
   partielle sur la fenêtre 8.
3. L'agent devient parfois trop directionnel. Exemple : P3 seul sur la
   fenêtre 4 est majoritairement long ; l'ablation blend=0 est encore plus
   concentrée. Un bon rendement ponctuel ne compense pas ce défaut.
4. Une fenêtre de 100 bougies est utile pour détecter des défauts, mais elle
   reste courte pour conclure à une supériorité économique durable. C'est la
   raison des seeds multiples et des cinq fenêtres, pas une raison pour choisir
   seulement les fenêtres positives.
5. P4 est correctement calculé, mais les métriques de calibration ne suffisent
   pas à créer un alpha. Il faut démontrer son gain net dans le portefeuille,
   ce que la comparaison ne montre pas.

## État des candidats

- `p3-bf72563778749e59` : baseline historique préservée ; non promue par la
  campagne actuelle.
- `ai_trading/configs/p3_p4_research_candidates.json` : manifeste de recherche
  séparé. Il contient baseline, RL sans blend, blend faible/short conservateur
  et long-only. Il ne modifie pas le lock historique.
- Ne pas modifier le lock ou le README pour déclarer un vainqueur. Une variante
  doit d'abord gagner selon le protocole complet ci-dessous.

## Architecture utile pour reprendre

| Zone | Fichiers de départ | Responsabilité |
|---|---|---|
| Orchestrateur | `ai_trading/scripts/run_real_multi_asset_walk_forward.py` | Données, snapshots, fenêtres, candidats, training RL, artefacts. |
| Environnement RL | `ai_trading/rl/multi_asset_trading_environment.py` | Positions, shorts, frais, risque, projection d'actions. |
| Contrat P4 | `ai_trading/llm/predictions/prediction_contract.py` | Horodatage, provenance, alignement marché/sentiment. |
| Modèle P4 | `ai_trading/llm/predictions/prediction_model.py` | Features, calibration temporelle, probabilités. |
| Calibration | `ai_trading/llm/predictions/uncertainty_calibration.py` | Brier, ECE, coverage, intervalles et validation OOS. |
| Injection P4 | `ai_trading/llm/predictions/walk_forward_features.py` | Retrain causal, abstentions, artefacts et features RL. |
| Comparaison | `ai_trading/scripts/compare_p3_p4_validation.py` | Refuse une comparaison dont les données, fenêtres, seeds ou frais divergent. |
| Protocole | `ai_trading/configs/p3_p4_validation_protocol.json` | Fenêtres 4–8, candidat, seeds, frais, critères de promotion. |

## Comment reprendre proprement

### 1. Ne rien effacer et vérifier l'état local

```powershell
git status --short
git diff --check
docker compose -f docker\docker-compose.yml ps
```

Le worktree contient des modifications P4/P3 non committées. Les préserver et
définir explicitement le périmètre d'un futur commit ; ne pas inclure les
artefacts de `ai_trading/info_retour/` s'ils sont générés et volumineux.

### 2. Revalider le socle avant toute nouvelle recherche

```powershell
docker compose -f docker\docker-compose.yml exec -T ai_api pytest -q `
  ai_trading/tests/llm/test_prediction_model.py `
  ai_trading/tests/llm/test_p4_p1_features.py `
  ai_trading/tests/integration/test_p4_walk_forward_features.py `
  ai_trading/tests/integration/test_p3_p4_comparison_protocol.py `
  ai_trading/tests/rl/test_multi_asset_short_audit.py
```

Si ce lot ne passe pas, corriger le périmètre concerné avant de relancer des
campagnes longues.

### 3. Rechercher P3 avant de réactiver P4 comme feature de décision

1. Garder le même snapshot P1, les mêmes frais, seeds et fenêtres pour chaque
   candidat d'une campagne.
2. Exécuter le manifeste `p3_p4_research_candidates.json` **sans**
   `--candidate-id` afin d'évaluer toute la grille pré-enregistrée en
   validation. Ne pas ajouter une variante après avoir vu une fenêtre gagnante.
3. Exiger pour chaque candidat : stabilité des 3 seeds, pas de diversité
   d'actions en échec, profit factor supérieur à 1, expectancy positive,
   rendement médian au moins égal à Buy & Hold et comportement acceptable dans
   les régimes baissiers.
4. Seulement après cette sélection de validation, réserver une nouvelle période
   chronologique jamais testée et lancer `--evaluation-mode final-test` une
   seule fois. Le registre OOS empêche la réutilisation silencieuse d'un
   holdout.

### 4. Réévaluer P4 séparément

Après un P3 acceptable, rejouer exactement le même protocole P3 seul / P3+P4 :

- même snapshot, même candidat, mêmes fenêtres et seeds ;
- vérifier `p4_feature_manifest.json`, `p4_predictions.parquet` et
  `p4_quality_metrics.json` ;
- exiger un gain médian positif, assez de fenêtres gagnantes, un portefeuille
  au moins au niveau de Buy & Hold et un ledger sain ;
- sinon garder P4 pour le diagnostic/dashboard uniquement.

## Règles à ne pas oublier

- Docker `ai_api` est le runtime de référence ; un test host seul n'est pas une
  validation.
- Aucune donnée future ne doit entrer dans P2/P4 ou dans les features RL.
- Une prédiction calibrée n'est pas une prédiction rentable.
- Buy & Hold est un benchmark net à conserver ; ne pas le retirer après une
  mauvaise comparaison.
- Ne jamais choisir une fenêtre après avoir vu son résultat.
- Aucun ordre réel, allocation réelle ou promesse de surperformance tant qu'un
  candidat n'a pas franchi validation multi-fenêtres puis holdout inédit.

## Décision de reprise recommandée

La prochaine tâche utile n'est pas une nouvelle feature. C'est une campagne
P3 pré-enregistrée de candidats, suivie d'un seul holdout inédit pour le seul
candidat qui satisfait les critères. Tant que cette preuve n'existe pas, le
projet doit être considéré comme une plateforme de recherche et de simulation,
pas comme un système de trading prêt à déployer.
