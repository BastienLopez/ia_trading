# Etat des phases historiques 1 a 4

Ce document reprend le README comme declaration fonctionnelle. Il ne remplace pas une preuve de test, mais il interdit de traiter ces modules comme inexistants.

| Phase historique | Capacites annoncees dans le README | Zones code associees | Statut de reprise |
| --- | --- | --- | --- |
| 1 - donnees | collecte multi-source, preprocessing, on-chain, cache, retries | `data/`, `utils/*collector*`, `data_processor.py` | a conserver et cartographier |
| 2 - LLM/sentiment | news, social, fake news, contexte, reporting | `llm/sentiment_analysis/` | a isoler derriere un contrat de donnees |
| 3 - RL/risque | DQN, SAC/PPO, features, risk management, multi-actifs, validation temporelle | `rl/`, `risk/`, `execution/`, `models/` | a consolider sans changer la logique valide |
| 4 - prediction | prediction marche, calibration, ensemble, performance | `llm/predictions/`, `models/` | a relier aux donnees et metriques canoniques |

## Interpretation correcte

- Les phases historiques sont des briques deja implementees ou explorees, pas des modules a effacer.
- P0 a P4 sont des phases de reprise technique, independantes de cette numerotation historique.
- Une fonctionnalite n'est classee obsolete qu'apres preuve de non-utilisation et accord explicite, jamais seulement parce qu'elle est complexe ou ancienne.
