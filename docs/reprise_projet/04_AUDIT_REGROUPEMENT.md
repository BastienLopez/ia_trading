# Audit de regroupement

## Principe

Regrouper signifie donner une porte d'entree et un contrat communs. Cela ne signifie pas fusionner des milliers de lignes ni supprimer des variantes utiles.

## Matrice de consolidation

| Domaine | Sources actuelles | Cible de regroupement | Methode | Suppression initiale |
| --- | --- | --- | --- | --- |
| Configuration | core, paires, web | `ai_trading.config` comme facade publique | facade + adaptateurs de lecture | interdite |
| Donnees | collectors, processor, datasets, orderbook | contrats `MarketData`, `Candle`, `OrderBook` | tests de parite puis adaptateurs | interdite |
| Features | data, indicators, rl, utils | catalogue de features documente | comparer valeurs sur fixture | interdite |
| Risque | risk, execution, rl | contrats `RiskDecision` et `RiskLimits` | adapter chaque moteur | interdite |
| Ordres | orders, execution, backtest | contrats `OrderRequest`, `Fill`, `PortfolioState` | mapping explicite | interdite |
| RL | agents, runners, environments, buffers | sous-packages explicites | enlever les reexports flous, pas les algorithmes | apres migration prouvee |
| Interfaces | FastAPI, API package, Dash, Flask | contrats API et lecture dashboard | choisir par cas d'usage | apres migration prouvee |
| Dependances | runtime/dev/GPU/experiments | groupes de dependances | inventaire import -> paquet | aucune avant validation |

## Definition d'un lot de regroupement

1. Un seul domaine et un seul contrat cible.
2. Au plus un ou deux appelants migrent dans le lot.
3. Les sorties de reference sont capturees sur une fixture locale avant changement.
4. Les adaptateurs deprecated restent tant que des appelants existent.
5. Le lot produit une matrice `ancien -> facade/contrat -> appelants migres -> tests`.

## Interdictions

- Pas de suppression recursive de dossiers.
- Pas de remplacement de framework sans mesure de compatibilite.
- Pas de fusion de fichiers uniquement a cause de leur nom proche.
- Pas de nettoyage de dependances avant connaitre les imports utilises au runtime.
