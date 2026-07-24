## Résumé exécutif

Phase 4 n’est pas prête à être intégrée ni à alimenter P3.

Blocages principaux :

- `MarketPredictor` produit des OHLCV fictifs et utilise un `MockLLMClient`, pas un LLM ni la collecte P1 réelle.
- `PredictionModel` appelle six méthodes absentes (`_prepare_data`, `_ensemble_predict`, `_save_models`, `_fetch_recent_data`, `_combine_predictions`, `_get_ml_prediction`) : entraînement et prédiction hybride sont cassés.
- Les types de confiance divergent : le prédicteur retourne `0.75`, tandis que multi-horizons et calibrateur attendent `"low"|"medium"|"high"`.
- Il n’existe aucun branchement P4 vers l’API, le dashboard ou P3. L’API Docker ne publie que `/predict` RL.
- Les incertitudes/calibrations sont simulées, non calibrées sur données hors échantillon ; la validation croisée emploie `KFold(shuffle=True)`, donc non temporelle.
- TensorRT n’est pas installé dans `ai_api`. Les tests TensorRT/RTX passent surtout avec mocks.
- Aucun artefact P4 durable réel (prédiction, calibration, rapport, visualisation, cache workspace) n’est présent.
- P3 reste verrouillée sur `p3-bf72563778749e59`, mais économiquement provisoire : aucun signal P4 ne devra autoriser du trading réel.

Aucun fichier n’a été modifié. Le seul changement du worktree est `README.md`, préexistant et intact.

## État des exigences README P4

| Exigence | État réel | Preuve | Risque | Action requise |
|---|---|---|---|---|
| Prédiction marché + sentiment | cassé | `market_predictor.py:_fetch_market_data` fabrique 30 jours de données ; `MockLLMClient` est utilisé | sorties non réelles | connecter P1/P2 et un fournisseur LLM injectable |
| Fusion technique + sentiment | cassé | `PredictionModel.predict()` appelle des méthodes absentes ; `_preprocess_data()` fait un `concat` sans date si dates absentes | look-ahead/décalage temporel/types incohérents | contrat horodaté causal unique |
| Multi-horizons minutes/heures/jours | partiel/cassé | horizons déclarés dans `MultiHorizonPredictor`; hybride appelle `PredictionModel.predict(current_data, timeframe)` avec une signature incompatible | fallback LLM fictif, aucun entraînement par horizon | corriger contrat et persistance par horizon |
| Adaptation dynamique | partiel | adaptation limitée au `RealTimeMarketMonitor` (EMA/seuils volatils) | non reliée aux prédictions ni à P3 | définir régimes et actions de dégradation |
| Ensemble | partiel, déconnecté | `ModelEnsemble` passe les tests numériques/catégoriels, mais n’est utilisé que par `RealTimeAdapter`, pas `PredictionModel` | stratégie confiance non calibrée | choisir un unique ensemble et le brancher |
| Stratégie confiance | cassé | valeur numérique `0.75` contre catégories attendues par `MultiHorizonPredictor`/`UncertaintyCalibrator` | confiance perdue ou `"unknown"` | type numérique borné unique + calibration OOS |
| Intervalles/distributions | partiel, simulé | bootstrap/bruit gaussien synthétique dans `UncertaintyCalibrator` | faux sentiment de précision | calibration sur prédictions/labels historiques |
| Calibration | absent | méthodes Platt/isotonic explicitement simplifiées ; `CalibratedClassifierCV` importé mais non utilisé | métriques non exploitables | calibration temporelle hors échantillon + Brier/ECE |
| Validation croisée | cassé | `perform_cross_validation()` utilise `KFold(... shuffle=True)` | fuite temporelle | `TimeSeriesSplit`/walk-forward purgé |
| Outliers | partiel | Z-score/IQR sur direction catégorielle convertie en -1/0/1 | détection statistiquement faible | anomalie sur prix, volume, spreads, sentiment et qualité source |
| SHAP/LIME | partiel/non validé | imports réels et gestion d’erreur, mais tests mockent SHAP/LIME | coût et compatibilité inconnus | chargement lazy, budget temps/mémoire, tests sur modèle entraîné |
| Visualisations/rapports | partiel | fonctions Plotly/Matplotlib existent ; PDF simulé dans `PredictionExplainer` et `reporting.py` | chemins sans fichier produit | rendu réel + vérification d’artefact |
| Temps réel | partiel | queue/thread/cache locaux ; 13 tests verts | pas de connecteur de flux, reconnexion, backpressure ou persistance | adaptateur de flux, limites de queue, reprise et métriques |
| Cache/invalidation | partiel | `CacheManager` RLock/LRU/TTL/disk ; clé `asset:timeframe` ignore données et timestamp | prédiction périmée malgré nouvelles données | clé de fraîcheur, invalidation évènementielle, test concurrence |
| GPU RTX 30/40 | partiel | Docker : RTX 3070, CUDA 13, capability 8.6 ; `rtx_optimizer.py` détecte RTX 30/40 | pas de benchmark réel modèle P4 | benchmark CPU/GPU, fallback vérifié, métriques VRAM |
| TensorRT | absent | smoke Docker : `TensorRT non détecté` | promesse README non fondée | capability explicite, pas d’activation sans runtime/benchmark |
| Quantification | absent pour P4 | `PredictionModel` ne quantifie pas ; tests verts visent `utils.model_quantization`, avec fallbacks simulés | pas de gain mémoire P4 prouvé | ne quantifier qu’un modèle P4 exportable, mesurer taille/latence |
| Volatilité/anomalies | partiel | IQR/médiane, alertes de volatilité/prix/volume | pas de détection de manipulation ni intégration prédiction | scénarios stress réalistes et règles de dégradation |
| Correction `test_predict_numerical`/`categorical` | implémenté localement | Docker : `2 passed in 14.38s` | couvre seulement `ModelEnsemble` isolé | conserver, élargir aux entrées P4 réelles |
| Tests P4 historiques | cassé | `test_predictions.py`: `1 failed, 1 passed`; `_get_ml_prediction` absent | fausse couverture P4 | réparer/remplacer le test et l’implémentation |

## Fichiers critiques et dépendances

- `market_predictor.py`
  - Dépend de P2 (`NewsAnalyzer`, `SocialAnalyzer`), mais les interroge avec une chaîne de texte, pas avec des observations sentiment horodatées.
  - N’utilise pas P1 : `_fetch_market_data()` est explicitement fictive.
  - Ne dépend pas de P3 ; son cache, son LLM et ses données ne sont pas utilisables comme signal RL.

- `prediction_model.py`
  - Vise une fusion LLM/ML, ensemble parallèle, cache et RTX.
  - Le chemin fonctionnel est incomplet à cause des méthodes absentes.
  - Aucune connexion P1/P2/P3 effective.

- `multi_horizon_predictor.py`
  - Déclare 11 horizons de `15m` à `30d`.
  - Dépend de fonctions de génération mock situées dans `test_predictions.py` pour ses données : dépendance de production vers test.
  - Génère des signaux `strong_buy`/`strong_sell`, sans garde P3 ni lien réel à P3.

- `uncertainty_calibration.py`, `prediction_explainer.py`, `visualization.py`, `reporting.py`
  - Outils auxiliaires existants, mais sans pipeline P4 entraîné et sans artefacts vérifiés.

- `real_time_adapter.py`, `cache_manager.py`, `model_ensemble.py`, `rtx_optimizer.py`
  - Composants locaux isolés ; pas de route API, de tâche de streaming ni de dashboard les utilisant.

- `api.py`
  - API strictement RL. `/predict` charge un modèle RL et retourne une confiance codée en dur à `0.8`.
  - OpenAPI Docker : `/health`, `/`, `/predict`, `/train`, `/backtest`, `/api/ema_metrics`; aucun endpoint P4.

- `configs/p3_locked_candidate.json`
  - Lock P3 vérifié : `p3-bf72563778749e59`.
  - Aucune intégration P4→P3 ; c’est préférable tant que P4 reste non validée.

## Tests Docker exécutés

Runtime de référence : `docker compose -f docker/docker-compose.yml exec -T ai_api ...`

| Commande ciblée | Résultat |
|---|---|
| Ensemble numérique + catégoriel | `2 passed in 14.38s` |
| `test_market_predictor.py` | `9 passed in 11.57s` ; couvre le mock, pas une prédiction réelle |
| multi-horizons + calibration + explainer | `17 passed in 32.58s` ; majorité avec données/mocks synthétiques |
| `test_real_time_adapter.py` | `13 passed in 11.44s` |
| optimisations/cache + RTX | `22 passed in 14.62s` ; RTX/TensorRT largement mockés |
| `test_performance_analysis.py` | `9 passed in 10.98s` ; pas de mesure de performance P4 réelle |
| quantification | `10 passed in 8.65s` ; module utilitaire hors P4, avec fallbacks simulés |
| intégration P1+P2+P3 | `1 passed in 14.01s` ; P4 n’est pas inclus |
| `test_predictions.py` | `1 failed, 1 passed in 13.27s` : `_get_ml_prediction` absent |
| `test_uncertainty.py` | erreur de collecte : import `from uncertainty_calibration` invalide |
| groupe P4 initial | timeout à 124 s ; processus résiduel arrêté proprement puis tests isolés exécutés |

Preuve GPU Docker : PyTorch `2.13.0+cu130`, CUDA disponible, RTX 3070, capability `(8,6)`. TensorRT est absent.

Smoke contrôlé P4 dans Docker : cache mémoire/disque et RTX ont été observés, mais la sortie reste celle du mock LLM (`bullish`, confiance `0.75`, `factors=[]`). Le démarrage charge des modèles sentiment et effectue des requêtes Hugging Face ; la latence LLM réelle n’a donc pas été mesurée.

## Doublons ou regroupements à envisager

| Élément | Risque | Proposition |
|---|---|---|
| `ModelEnsemble` vs ensemble interne de `PredictionModel` | élevé | définir une seule implémentation d’ensemble, après contrat commun |
| `PredictionExplainer.generate_report` vs `PredictionReporter` | moyen | unifier le rendu HTML/PDF et le répertoire d’artefacts |
| décorateur cache de `RealTimeAdapter` vs `CacheManager` | moyen | conserver un seul mécanisme thread-safe avec métriques |
| `test_predictions.py` dans le code P4 vs suite `tests/llm/` | élevé | migrer les fixtures/tests valides, puis retirer le test legacy seulement après remplacement |
| données mock importées depuis un fichier de test par `MultiHorizonPredictor` | élevé | déplacer les fixtures vers `tests/fixtures`, aucune dépendance production→test |
| `MockLLMClient` dans le module de production | élevé | injecter un client test dans les tests, jamais en chemin runtime |
| PDF simulés dans deux modules | faible à moyen | ne fusionner qu’après génération réelle d’artefacts |

Aucune suppression ou fusion n’est proposée avant validation.

## Plan d’action proposé

### P0 — bloquant

| Action | Fichiers | Modification | Test/critère d’acceptation |
|---|---|---|---|
| Contrat P4 causal | `market_predictor.py`, `prediction_model.py`, `multi_horizon_predictor.py`, nouvelles fixtures | schéma unique : actif, horizon, `as_of`, OHLCV, sentiment, fraîcheur, provenance, confiance numérique | rejet explicite des données non horodatées/futures ; test sans look-ahead |
| Connecter P1/P2 réellement | mêmes fichiers, collecteur/P2 ciblés | remplacer OHLCV fictifs et requêtes sentiment textuelles par adaptateurs P1/P2 horodatés | fixture P1+P2 réelle/rejouable ; vérification du cutoff temporel |
| Réparer le cœur hybride | `prediction_model.py`, `parallel_processor.py` si nécessaire | implémenter ou retirer proprement les appels aux méthodes absentes ; persistance chargement/sauvegarde cohérente | entraînement puis prédiction Docker sur fixture ; aucun `AttributeError` |
| Unifier ensemble/confiance | `prediction_model.py`, `model_ensemble.py`, `multi_horizon_predictor.py` | une stratégie unique, probabilités normalisées, abstention si faible consensus | tests numérique, catégoriel, probabiliste, disagreement et abstention |
| Calibration valide | `uncertainty_calibration.py` | remplacer `KFold(shuffle=True)` et calibrations simulées par walk-forward + calibration OOS | Brier/ECE/coverage calculés sur jeu temporel figé |
| Cache correct | `cache_manager.py`, `market_predictor.py`, `real_time_adapter.py` | clé incluant snapshot/fraîcheur, invalidation sur nouvelles données, tests concurrence | aucune réutilisation après changement de bougie/sentiment ; pas de race |
| Réparer la suite P4 | tests P4 ciblés | corriger imports et tests legacy, sans mocks cachant les chemins critiques | collecte verte complète, aucun skip masquant une erreur |

### P1 — important

| Action | Fichiers | Modification | Test/critère |
|---|---|---|---|
| Temps réel robuste | `real_time_adapter.py` | queue bornée, backpressure, reprise flux, timeouts, état dégradé | perte/reconnexion/fort débit dans Docker |
| SHAP/LIME utile | `prediction_explainer.py`, visualisations/tests | imports lazy, garde modèle compatible, budget temps/mémoire, sorties fichiers réelles | SHAP/LIME réels sur modèle P4 entraîné, artefacts ouverts |
| GPU/TensorRT honnête | `rtx_optimizer.py`, tests | capability reportée, CPU fallback, TensorRT seulement si réellement installé | benchmark reproductible CPU/GPU et TensorRT si disponible |
| Anomalies/volatilité | `real_time_adapter.py`, calibrateur/tests | scénarios sauts, volume, outliers sentiment et dégradation de confiance | stress test haute volatilité avec artefacts et seuils vérifiés |
| API/dashboard P4 en lecture seule | `api.py`, dashboard/tests | endpoint P4 séparé, jamais d’ordre de trading ; observabilité cache/latence | OpenAPI + appel Docker + schéma stable |

### P2 — optimisation après preuves

| Action | Fichiers | Modification | Test/critère |
|---|---|---|---|
| Quantification P4 | modèle P4 exportable, `rtx_optimizer.py` | INT8/FP16 uniquement après compatibilité | taille, VRAM, latence et dérive de sortie mesurées |
| Mémoire/profilage | cache, temps réel, profiling | budgets mémoire, nettoyage cycle de vie, métriques persistées | charge longue durée sans croissance non bornée |
| Regroupement des doublons | modules/tests cités | migration progressive, aucune suppression préalable | couverture migrée et importeurs mis à jour |

## Validation finale après implémentation

1. Tests P4 unitaires ciblés dans `ai_api`, y compris échecs LLM, données manquantes, cache, concurrence, calibration, GPU/CPU, SHAP/LIME et haute volatilité.
2. Test intégré Docker P1+P2+P3+P4 avec fixtures horodatées et vérification explicite que chaque signal P4 était disponible avant la bougie RL.
3. Exécuter P3 avec le lock `p3-bf72563778749e59` sans modifier ses paramètres et sans trading réel.
4. Contrôler les artefacts : données source/version, sentiment, prédictions, probabilités, intervalles, métriques de calibration, logs, cache, visualisations et rapports réellement écrits.
5. Produire un rapport séparant strictement : code présent, tests verts, intégration validée, performance mesurée et qualité prédictive hors échantillon.
6. Ne pas qualifier une prédiction de bonne, ni P3 de supérieure au Buy & Hold, sans métriques temporelles hors échantillon reproductibles.

J’attends votre validation explicite de ce plan avant toute modification.