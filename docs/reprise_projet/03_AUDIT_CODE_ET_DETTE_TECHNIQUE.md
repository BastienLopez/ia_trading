# Audit code et dette technique

## Constats verifies statiquement

| Priorite | Constat | Risque | Reponse de reprise |
| --- | --- | --- | --- |
| P0 | `ai_trading/api.py` et `ai_trading/api/` coexistent | resolution d'import et point d'entree ambigus | tracer les imports et la commande lancee; retenir un chemin canonique avec compatibilite temporaire |
| P0 | configuration repartie entre core et web | defaults divergents entre execution et interface | documenter les valeurs partagees puis introduire un adaptateur de configuration, sans rename massif |
| P1 | quatre implementations apparentes d'indicateurs/features | sorties divergentes ou fuite temporelle | ecrire des tests de parite sur fixtures avant choisir le proprietaire canonique |
| P1 | trois familles de risque/execution | regles ou frais incoherents | definir le schema ordre/fill/risque et adapter les implementations existantes |
| P2 | agents, runners et buffers RL multiples | difficultes de maintenance et tests en double | distinguer agent, environnement, runner et utilitaire; garder les variantes ayant des consommateurs |
| P2 | tests, examples et sorties sont melanges aux packages | collecte de tests imprevisible, artefacts versionnes | reclasser par lots, avec import et test conserves |
| P3 | runtime, dev, GPU et experimentations sont melanges dans `requirements.txt` | installation lente et fragile | separer les groupes seulement apres analyse des imports runtime |
| P4 | FastAPI, Dash et Flask coexistent | interfaces et operations dupliquees | definir les cas d'usage puis migrer la surface secondaire vers lecture seule ou compatibilite |

## Doublons a examiner, pas a supprimer d'avance

- `risk_manager.py` : `risk/`, `execution/`, `rl/`.
- `data_processor.py` : racine et `rl/`.
- `technical_indicators.py` : `data/` et `rl/`.
- `dqn_agent.py`, `sac_agent.py`, `trading_environment.py` : racine RL et sous-packages agents/environments.
- `adaptive_normalization.py`, `model_distillation.py`, `temporal_cross_validation.py`, `parallel_processor.py`.

## Regle de decision

Pour chaque candidat : relever son contrat public, ses appelants, ses tests, les effets de bord et le comportement de reference. Ensuite seulement choisir `conserver`, `extraire`, `adapter`, `deprecier` ou `supprimer`. Le statut initial de tous les candidats est `A ANALYSER`.
