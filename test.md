# Tests et sorties observables — Phases 1 à 3

Ce document regroupe les commandes à lancer depuis la racine du dépôt. Elles
s'exécutent dans Docker, une seule à la fois. Ne pas lancer deux entraînements
GPU ou deux tests RL lourds simultanément.

```powershell
Set-Location C:\Users\UTILISATEUR\Documents\GitHub\ia_trading
docker compose -f docker/docker-compose.yml ps
```

Les services `ai_api`, `db` et `redis` doivent être `healthy` avant de lancer
les tests. CUDA peut être vérifié avec :

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Phase 1 — collecte, cache et prétraitement

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m pytest ai_trading/utils/tests/test_resilient_requester.py ai_trading/utils/tests/test_enhanced_cache.py ai_trading/utils/tests/test_enhanced_blockchain_collector.py ai_trading/tests/data/test_enhanced_collector.py ai_trading/tests/data/test_enhanced_collector_contract.py ai_trading/tests/data/test_enhanced_preprocessor.py ai_trading/tests/data/test_enhanced_preprocessor_contract.py -q -ra
```

Attendu : **75 tests verts**. Ce lot ne produit pas de graphique métier ; il
vérifie les contrats de données, le fallback multi-source, les timeouts, le
cache LRU/compressé et la collecte blockchain.

## Phase 2 — sentiment, crédibilité et contexte

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m pytest ai_trading/tests/unit/test_fake_news_detector.py ai_trading/tests/unit/test_contextual_analyzer.py ai_trading/tests/llm/test_enhanced_news_analyzer.py ai_trading/tests/llm/test_crypto_finetuning.py ai_trading/tests/llm/test_social_analyzer.py ai_trading/tests/llm/test_sentiment_pipeline.py ai_trading/tests/llm/test_sentiment_integration.py ai_trading/tests/llm/sentiment_analysis/test_sentiment_utils.py -q -ra
```

Attendu : **47 tests verts**. Le fine-tuning BERTweet est une base de
sentiment ; son accuracy de 57,03 % / macro-F1 de 56,82 % ne doit pas être
interprétée comme un signal de trading rentable.

## Phase 3 — tests de contrats RL

### Environnement, indicateurs et risques

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m pytest ai_trading/tests/ml/test_technical_indicators.py ai_trading/tests/ml/test_indicators.py ai_trading/tests/rl/test_trading_environment.py ai_trading/tests/rl/test_multi_asset_trading_env.py -q -ra
```

Attendu : **49 tests verts**. Vérifie MACD, RSI, ATR, ADX, Ichimoku, Volume
Profile, état sans fuite temporelle, frais, slippage, délais et actions
multi-actifs.

### Agents GPU

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m pytest ai_trading/tests/rl/agents/test_dqn_agent.py ai_trading/tests/rl/test_sac_agent.py ai_trading/tests/rl/test_n_step_sac_agent.py ai_trading/tests/rl/test_noisy_sac_agent.py ai_trading/tests/rl/test_ppo_agent_gpu.py ai_trading/tests/rl/test_rl_trading_system.py -q -ra
```

Attendu : **34 tests verts**. Vérifie DQN priorisé/Double/Dueling/Noisy/n-step,
SAC, PPO, entraînement et refus des données de marché implicites synthétiques.

### Masques, ledger et protocole walk-forward

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m pytest ai_trading/tests/rl/test_action_masks.py ai_trading/tests/rl/test_trade_ledger.py ai_trading/tests/rl/test_real_market_report.py ai_trading/tests/rl/test_walk_forward.py -q -ra
```

Attendu : tests verts sur les masques DQN/PPO/SAC, le backup DQN sans action
impossible, le ledger FIFO, le refus d'une politique mono-action et l'interdiction
de sélectionner un candidat après avoir vu son test hors-échantillon.

### Risque, exécution, allocation et optimisation

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m pytest ai_trading/tests/risk/test_advanced_risk_manager.py ai_trading/tests/risk/test_risk_manager.py ai_trading/tests/unit/execution/test_risk_manager.py ai_trading/tests/execution/test_market_constraints.py ai_trading/tests/execution/test_order_system.py ai_trading/tests/execution/test_order_flow.py ai_trading/tests/misc/test_complete_allocation_system.py ai_trading/tests/optimization/test_hyperparameter_optimizer.py ai_trading/tests/optimization/test_bayesian_optimizer.py ai_trading/tests/optimization/test_bayesian_optimization.py ai_trading/tests/optimization/test_bayesian_optimizer_real_gpu.py -q -ra
```

Attendu : **90 tests verts**. Vérifie ATR/stop/take-profit/VaR, contraintes
d'exécution, allocation, ordre et optimisation bayésienne.

### Contrat de bout en bout P1 → P2 → P3

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m pytest ai_trading/tests/integration/test_p1_p2_p3_pipeline.py ai_trading/tests/rl/test_real_market_report.py -q -ra
```

Attendu : **5 tests verts**. Ce lot valide les contrats et les fichiers d’un
rapport ; il ne remplace pas une évaluation réelle du marché.

## Rapport RL sur données de marché réelles

### BTC/USDT

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m ai_trading.scripts.run_real_market_rl_report --symbol BTC/USDT --exchange binance --days 365 --timeframe 4h --episodes 20 --max-training-steps 20000 --train-every 4 --max-optimization-steps 256 --seed 42
```

### ETH/USDT

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m ai_trading.scripts.run_real_market_rl_report --symbol ETH/USDT --exchange binance --days 365 --timeframe 4h --episodes 20 --max-training-steps 20000 --train-every 4 --max-optimization-steps 256 --seed 42
```

Chaque lancement crée un nouveau dossier sous :

`ai_trading/info_retour/real_market_reports/<SYMBOL>/<TIMESTAMP>/`

La normalisation adaptative est désactivée par défaut dans ces rapports : les
indicateurs gardent leur normalisation causale propre, mais le run est environ
48 fois plus rapide par bougie. Ajouter `--adaptive-normalization` seulement
pour comparer explicitement cette variante.

Fichiers à lire :

- `ohlcv_raw.csv` : bougies publiques réellement téléchargées ;
- `equity_curve.csv` : valeur agent/benchmark, récompense, action et exécution ;
- `metrics.json` : période, source, frais, seed, nombre de pas et métriques ;
- `trades.csv` : ledger FIFO des trades clôturés (PnL net, frais, durée, motif) ;
- `equity_comparison.png` : courbe agent RL contre Buy & Hold ;
- `rewards_and_actions.png` : récompense par bougie et ordres réellement exécutés.

## Lire correctement les graphiques

Un rapport est **invalide pour la validation de performance** si :

- `executed_trade_count` vaut `0` ;
- la récompense est plate à zéro ;
- la valeur de l’agent reste plate alors que le benchmark bouge ;
- les bougies, dates, frais, seed ou nombre de pas ne sont pas présents dans
  `metrics.json` ;
- les données ne sont pas hors-échantillon ou proviennent d’un générateur
  synthétique.

Un objectif de **80 % de win-rate seul est insuffisant**. La validation doit
retenir simultanément : rendement net après frais, rendement contre Buy & Hold,
max drawdown, Sharpe/Sortino, turnover, nombre de trades et stabilité sur
plusieurs fenêtres chronologiques et plusieurs actifs. Une stratégie avec 80 %
de petits gains et une seule grosse perte peut rester perdante.

## Walk-forward réel multi-actifs, porte P4

Le runner n'utilise aucune donnée synthétique. Il évalue chaque fenêtre une fois :
les candidats DQN/PPO/SAC sont comparés sur validation, puis le retenu est
ré-entraîné sur train+validation et évalué une seule fois sur le test figé.

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m ai_trading.scripts.run_real_market_walk_forward --assets BTC/USDT ETH/USDT XAU/USD --agent-types dqn ppo sac --timeframe 1d --days 2200 --train-candles 600 --validation-candles 180 --test-candles 180 --step-candles 180 --max-windows 3 --min-windows 3 --episodes 10 --max-training-steps 6000 --max-optimization-steps 512 --seed 42
```

Lire `ai_trading/info_retour/walk_forward_reports/<TIMESTAMP>/walk_forward_summary.json`.
P4 est autorisée uniquement si `overall_phase4_gate.passed` vaut `true`; ce
fichier donne aussi chaque cause exacte d'échec, sans rejouer ni sélectionner le
test a posteriori.

## Test Phase 3 isolé par processus

Si le lot RL complet charge des familles Torch/Triton incompatibles dans le
même interpréteur, utiliser le lanceur séquentiel :

```powershell
docker compose -f docker/docker-compose.yml exec -T ai_api python3 -m ai_trading.scripts.run_phase3_tests
```

Il ne parallélise rien et arrête correctement son test enfant si le processus
principal est interrompu.
