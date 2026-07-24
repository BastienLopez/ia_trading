# AI Trading - Système de Trading Crypto basé sur l'IA

Ce module contient l'implémentation d'un système de trading de cryptomonnaies utilisant l'intelligence artificielle, l'apprentissage par renforcement et l'analyse de sentiment.

## État du Projet

### Phase 1: Collecte et Prétraitement des Données ✅

> Validée dans Docker le 22/07/2026 : 75 tests ciblés verts. Ils couvrent les
> collecteurs, le prétraitement, le cache LRU compressé à deux niveaux, la
> blockchain synchrone/asynchrone et la résilience HTTP multi-source.
- ✅ 1.1 Implémentation de `utils/enhanced_data_collector.py`
  - Connexion à plusieurs APIs de cryptomonnaies (CoinGecko, CoinCap, CryptoCompare)
  - Collecte des données de prix, volumes et capitalisation
  - Récupération de l'indice Fear & Greed
  - Scraping des actualités et réseaux sociaux
- ✅ 1.2 Implémentation de `utils/enhanced_preprocessor.py`
  - Nettoyage des données multi-sources
  - Normalisation avancée
  - Création de features techniques étendues
  - Prétraitement des données textuelles
- ✅ **Intégration de données on-chain (blockchain)**
  - Analyse des transactions sur la blockchain
  - Métriques DeFi (TVL, volumes de pools)
  - Données de staking et de gouvernance
  - Flux de capitaux entre protocoles
- ✅ **Optimisation de la parallélisation des collectes**
  - Collecte asynchrone multi-sources
  - Gestion des rate limits
  - Priorisation des sources
  - Cache distribué
- ✅ **Mise en cache intelligente des données**
  - Stratégie LRU (Least Recently Used)
  - Préchargement des données fréquentes
  - Compression des données historiques
  - Gestion de la cohérence
- ✅ **Gestion avancée des timeouts et retries**
  - Backoff exponentiel
  - Circuit breaker pattern
  - Fallback sur sources alternatives
  - Monitoring des performances

-----------------------------------------------------------------------------------------------------------------------

### Phase 2: Analyse de Sentiment (LLM) ✅

> Validée dans Docker le 22/07/2026 : 47 tests ciblés verts pour le pipeline
> unifié sentiment + crédibilité + propagation + contexte. BERTweet a été adapté sur 23 301 articles crypto DLT,
> en CUDA/FP16 (18 640 entraînement, 4 661 validation temporelle). Le manifeste
> versionné rapporte une évaluation isolée sur 512 articles : accuracy 57,03 %
> et macro-F1 56,82 %. Cette métrique est une preuve d'exécution, pas une
> garantie de performance de trading.
> L'artefact et son manifeste sont dans `ai_trading/info_retour/models/` (ignorés
> de Git) ; le corpus CC-BY-NC-4.0 est réservé à l'usage personnel non commercial.
- ✅ 2.1 Implémentation de `llm/sentiment_analysis/news_analyzer.py`
  - Analyse des actualités crypto
  - Extraction des entités et sentiments
  - Version améliorée avec visualisations et rapports
- ✅ 2.2 Implémentation de `llm/sentiment_analysis/social_analyzer.py`
  - Analyse des tweets et posts Reddit
  - Détection des tendances et sentiments
- ✅ **Détection de fake news**
  - Vérification des sources
  - Analyse de la propagation
  - Détection des bots
  - Score de crédibilité
- ✅ **Analyse de sentiment contextuelle**
  - Compréhension du contexte marché
  - Analyse des relations entre entités
  - Détection des sarcasmes/ironies
  - Analyse temporelle des sentiments
- ✅ **Intégration de modèles LLM plus récents**
  - Modèles spécialisés crypto
  - Fine-tuning sur données crypto
  - Optimisation des prompts

-----------------------------------------------------------------------------------------------------------------------

### Phase 3: Développement de l'Agent d'Apprentissage par Renforcement ✅

> Le cœur P3 est opérationnel dans Docker avec CUDA et n'accepte plus de
> données de marché synthétiques implicitement. Le protocole réel impose à
> présent un ledger FIFO, des masques d'actions, une validation anti-politique
> mono-action et un walk-forward train/validation/test strictement temporel.
> Le socle P3 est verrouillé de façon provisoire par décision produit dans
> `ai_trading/configs/p3_locked_candidate.json`. La porte de mise en production
> reste fermée : aucun résultat court ne vaut preuve de surperformance.

#### 3.1 Fondations RL ✅

- ✅ Environnement Gymnasium mono-actif, actions discrètes partielles et continues.
- ✅ État enrichi : OHLCV, MACD, stochastique, momentum, OBV, RSI, Bollinger,
  EMA, ATR, Volume Profile, ADX, pivots, Ichimoku et sentiment sans look-ahead.
- ✅ Risque : stop-loss/take-profit ATR, trailing stop, exposition et taille de
  position ajustées à la volatilité.
- ✅ Récompenses : rendement, Sharpe, pénalités de turnover/frais et drawdown.
- ✅ Agents : DQN priorisé avec Double/Dueling/UCB/Noisy Linear/n-step ; SAC et
  PPO continus avec clipping de gradient et régularisation d'entropie.
- ✅ Curriculum, GRU/LSTM/Transformers et recherche par grille.

#### 3.2 Robustesse et multi-actifs ✅

- ✅ Slippage, impact, délais d'exécution, carnet d'ordres et flux d'ordres.
- ✅ Sortino, diversification, contraintes de corrélation et allocation.
- ✅ Environnement vectoriel multi-actifs ; BTC, ETH, or (proxy `GC=F` pour
  XAU/USD), AAPL et NVDA sont collectés auprès de sources publiques réelles.
- ✅ Validation temporelle, sélection de caractéristiques, architectures CNN,
  attention, distillation et tests de robustesse.

#### 3.3 Composants avancés ✅

- ✅ VaR, allocation adaptative, gestion de risque multi-niveaux et ordres
  limite/SL/TP/taille de position.
- ✅ PPO/SAC, apprentissage inverse, transfert inter-marchés et expérience
  distribuée.
- ✅ Allocation multi-facteurs, arbitrage/pairs trading, multitâche,
  visualisations Dash/3D/post-mortem et optimisation bayésienne.

#### Validation réelle obligatoire avant la Phase 4

- ✅ Ledger FIFO `trades.csv` : entrées/sorties, quantité, frais, PnL net,
  durée, raison et win/loss ; win-rate uniquement sur trades clôturés.
- ✅ DQN, PPO et SAC reçoivent les masques des actions impossibles ; une suite
  de tests refuse une politique mono-action.
- ✅ Runner `run_real_market_walk_forward` : BTC, ETH et or (`GC=F`, proxy
  Yahoo Finance de XAU/USD), plusieurs fenêtres disjointes, frais/slippage,
  sélection sur validation et un seul test figé par modèle retenu.
- ✅ Porte P4 calculée : rendement net vs Buy & Hold, drawdown, profit factor,
  nombre minimal de trades, Sharpe/Sortino et diversité/stabilité.
- ⚠️ La validation économique reste à obtenir : le lock P3 permet de poursuivre
  le développement de P4/P5, mais `overall_phase4_gate.passed` n'est pas `true`
  sur les trois actifs. Aucune mise en production ou allocation réelle n'est
  autorisée sur cette base. Voir `p3_lock.md`.

-----------------------------------------------------------------------------------------------------------------------

### Phase 4: Prédictions de Marché (LLM) ⏳
- ✅ 4.1 Implémentation de `llm/predictions/market_predictor.py`
  - Génération de prédictions basées sur les données de marché et le sentiment
- ✅ 4.2 Implémentation de `llm/predictions/prediction_model.py`
  - Modèle de prédiction combinant données techniques et sentiment
- ✅ **Prédictions multi-horizons**
  - Court terme (minutes)
  - Moyen terme (heures)
  - Long terme (jours)
  - Adaptation dynamique
- ✅ **Interprétabilité des prédictions**
  - SHAP values
  - LIME explanations
  - Visualisations interactives
  - Rapports détaillés
- ✅ **Calibration des incertitudes**
  - Intervalles de confiance
  - Distributions de probabilité
  - Gestion des outliers
  - Validation croisée
- ✅ **Ensemble de modèles**
  - Combinaison de plusieurs modèles
  - Fusion de prédictions
  - Enrichissement des informations
- ✅ **Adaptation en temps réel**
  - Mise à jour des prédictions en temps réel
  - Réaction aux changements de marché
  - Intégration de nouvelles données
- ✅ **Correction des tests unitaires** :
  - Résoudre les problèmes dans test_predict_numerical et test_predict_categorical
  - Améliorer la compatibilité avec les types de retour numpy
- ✅ **Amélioration de la robustesse** :
  - Optimiser la détection des changements significatifs du marché
  - Améliorer la gestion des données aberrantes
  - Gestion des cas d'erreur et récupération
  - Tolérance aux erreurs de flux de données
- ✅ **Performance et optimisation** :
  - Optimiser le traitement des données en temps réel pour réduire la latence
  - Implémenter un système de mise en cache pour les calculs intensifs
  - Parallélisation des opérations d'ensemble de modèles
  - Profilage et optimisation des goulots d'étranglement
- ✅ **Points d'amélioration** :
  - Implémenter une stratégie "confidence" robuste pour l'ensemble de modèles
  - Ajouter des tests de stress pour les scénarios de haute volatilité
  - Développer une visualisation en temps réel des indicateurs de marché
  - Optimiser la consommation mémoire pour les historiques de prix volumineux
  - Étendre les capacités de détection d'anomalies pour identifier les manipulations de marché
- ✅ Priorisation GPU du projet si disponible
- ✅ Optimisation supplémentaire pour des GPU spécifiques (comme les RTX séries 30 et 40)
 - Intégration avec d'autres frameworks comme TensorRT pour des performances encore meilleures
 - Quantification plus avancée des modèles pour réduire l'empreinte mémoire

-----------------------------------------------------------------------------------------------------------------------

### Phase 5: Signaux de Trading (ML) ⏳
- ⏳ 5.1 Implémentation de `ml/trading_signals/signal_generator.py`
  - Génération de signaux d'achat/vente
- ⏳ 5.2 Implémentation de `ml/trading_signals/ml_model.py`
  - Modèle ML pour la prédiction des signaux
- [ ] **Signaux multi-timeframes**
  - Analyse technique multi-échelles
  - Confirmation croisée
  - Filtrage des faux signaux
  - Priorisation des signaux
- [ ] **Backtesting avancé**
  - Simulation réaliste
  - Gestion des slippages
  - Coûts de transaction
  - Stress testing
- [ ] **Gestion du biais de survivance**
  - Évaluation des performances
  - Détection des tendances
  - Réduction des erreurs
  - Amélioration des prédictions

-----------------------------------------------------------------------------------------------------------------------

### Phase 6: Intégration et API ⏳
- ⏳ 6.1 Développement de l'API
  - Endpoints pour les prédictions
  - Intégration des différents composants
  - Documentation de l'API
- ⏳ 6.2 Tests d'intégration
  - Tests de bout en bout
  - Validation des performances
- [ ] **API GraphQL**
  - Requêtes flexibles
  - Réduction du trafic
  - Documentation auto-générée
  - Versioning intelligent
- [ ] **Monitoring temps réel**
  - Métriques de performance
  - Alertes configurables
  - Dashboards interactifs
  - Logs structurés

### Phase 7: Optimisation @ai_trading\documentation\opti.md ⏳
- ⏳ 7.1 Optimisation Projet 
  - Optimisation mémoire CPU / RAM
  - Optimisation CPU
  - Optimisation GPU
  - Optimisation sur l'architecture IA elle-même
  - Optimisation sur la gestion RL (Reinforcement Learning)
  - Optimisation générale du projet
  - Outils/méthodes qui peuvent aider
  - Optimisation des performances et fluidité
- ⏳ 7.2 Format
  - autoflake --in-place --remove-all-unused-imports --recursive ai_trading/
	- isort ai_trading/
	- black ai_trading/
- [ ] **Quantification des modèles**
  - INT8/FP16
  - Pruning avancé
  - Distillation
  - Optimisation pour mobile
- [ ] **Pipeline de données optimisé**
  - Streaming efficace
  - Compression adaptative
  - Cache multi-niveaux
  - Gestion de la mémoire

### Phase 8: Interface Web et Déploiement ⏳
- ⏳ 8.1 Développement de l'interface web
  - Dashboard de trading
  - Visualisations interactives
  - Gestion des utilisateurs
- ⏳ 8.2 Intégration plateforme
  - Connexion aux exchanges (mode réel/paper)
  - Implémentation des API de trading
  - Système de journalisation des transactions
- ⏳ 8.3 Déploiement production
  - Configuration cloud
  - Mise en place CI/CD
  - Monitoring de performance
- [ ] **Interface mobile**
  - Application native
  - Notifications push
  - Mode hors-ligne
  - Synchronisation
- [ ] **Scaling automatique**
  - Auto-scaling horizontal
  - Load balancing
  - Gestion des pics
  - Optimisation des coûts

### Phase 9: Integration de données réelles
  - [ ] Intégration tt les modules dans un pipeline complet d'entraînement 
  - [ ] Documentation d'utilisation de la pipeline
  - [ ] Utilisation de données réelles
  - [ ] Intégration avec les APIs d'exchanges


### Phase 10: Integration plateforme réelles
  - [ ] Connexion avec exchanges
  - [ ] Mode paper trading
  - [ ] Système d'alerte et monitoring
  
## 🧩 **Tâches transversales**

  ### 1. Documentation
  - [ ] Doc du code avec exemples
  - [ ] Tutoriels
  - [ ] Documentation des API

  ### 2. Tests
  - [ ] Tests unitaires
  - [ ] Tests d'intégration
  - [ ] Tests de performance

  ### 3. Optimisation
  - [ ] Optimiser les performances de l'environnement
  - [ ] Réduction mémoire
  - [ ] Parallélisation des calculs

  ### 4. Déploiement
  - [ ] Dockerisation
  - [ ] Journalisation avancée
  - [ ] Sauvegarde/reprise

  ### 5. Stratégies d'apprentissage avancées
  - [ ] Imitation learning
  - [ ] Transfer learning
  - [ ] Meta-learning
  - [ ] Reward shaping dynamique
  - [ ] Parallelization avec vectorized environments

  ### 6. Documentation
  - [ ] Doc du code avec exemples
  - [ ] Tutoriels
  - [ ] Documentation des API

  ### 7. Tests
  - [ ] Tests unitaires
  - [ ] Tests d'intégration
  - [ ] Tests de performance

  ### 8. Optimisation
  - [ ] Optimiser les performances de l'environnement
  - [ ] Réduction mémoire
  - [ ] Parallélisation des calculs

  ### 9. Déploiement
  - [ ] Dockerisation
  - [ ] Journalisation avancée
  - [ ] Sauvegarde/reprise

  ### 10. Stratégies d'apprentissage avancées
  - [ ] Imitation learning
  - [ ] Transfer learning
  - [ ] Meta-learning
  - [ ] Reward shaping dynamique
  - [ ] Parallelization avec vectorized environments

### Analyse technique
- Intégration complète dans `data_integration.py`
- Visualisations dans les rapports d'analyse
