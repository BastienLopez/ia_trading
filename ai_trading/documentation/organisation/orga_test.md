# Organisation des Tests

Ce document définit l'organisation des tests du projet AI Trading en les catégorisant par domaine fonctionnel.

## Structure des dossiers

La nouvelle organisation des tests suit la structure suivante :

```
ai_trading/
└── tests/
    ├── unit/               # Tests unitaires
    │   ├── llm/            # Tests unitaires pour les modules LLM
    │   ├── ml/             # Tests unitaires pour les modules ML
    │   ├── rl/             # Tests unitaires pour les agents RL
    │   ├── utils/          # Tests unitaires pour les utilitaires
    │   ├── indicators/     # Tests unitaires pour les indicateurs techniques
    │   └── ...
    ├── integration/        # Tests d'intégration
    │   ├── llm/            # Tests d'intégration pour les modules LLM
    │   ├── ml/             # Tests d'intégration pour les modules ML
    │   ├── rl/             # Tests d'intégration pour les agents RL
    │   └── ...
    ├── performance/        # Tests de performance
    │   ├── backtesting/    # Tests de performance pour le backtesting
    │   ├── optimization/   # Tests de performance pour l'optimisation
    │   └── ...
    ├── ml/                 # Tests pour les modèles ML
    │   ├── backtesting/    # Tests pour le backtesting ML
    │   └── ...
    ├── rl/                 # Tests pour les agents RL
    ├── llm/                # Tests pour les modules LLM
    ├── risk/               # Tests pour la gestion des risques
    ├── execution/          # Tests pour l'exécution des ordres
    ├── optimization/       # Tests pour l'optimisation
    ├── data/               # Tests pour la collecte et le prétraitement des données
    ├── misc/               # Tests divers
    ├── conftest.py         # Configuration commune pour pytest
    └── mocks.py            # Objets mock pour les tests
```

## Classification des tests existants

Les tests existants ont été organisés dans les catégories suivantes :

### 1. Modules LLM (tests/llm/)

- test_enhanced_news_analyzer.py
- test_social_analyzer.py
- test_sentiment_integration.py
- test_contextual_analyzer.py
- test_fake_news_detector.py
- test_market_predictor.py
- test_prediction_explainer.py
- test_uncertainty_calibration.py
- test_multi_horizon_predictor.py
- test_model_ensemble.py
- test_real_time_adapter.py

### 2. Modules ML (tests/ml/)

- test_trading_signals.py
- test_model_quantization.py
- test_temporal_cross_validator.py
- test_feature_selector.py
- test_optimized_dataset.py
- test_financial_dataset.py
- test_adaptive_normalization.py
- test_indicators.py
- test_technical_indicators.py
- test_multi_timeframe_signals.py
- test_statistical_tests.py

#### ML - Backtesting (tests/ml/backtesting/)

- test_advanced_backtesting.py
- test_temporal_cross_validation.py
- test_market_robustness.py
- test_survivorship_bias.py

### 3. Modules RL (tests/rl/)

- test_dqn_agent.py
- test_sac_agent.py
- test_noisy_sac_agent.py
- test_n_step_sac_agent.py
- test_gru_sac_agent.py
- test_transformer_sac_agent.py
- test_transformer_hybrid.py
- test_noisy_linear.py
- test_rewards.py
- test_trading_environment.py
- test_curriculum_learning.py
- test_entropy_regularization.py
- test_multi_asset_trading.py
- test_multi_asset_trading_env.py
- test_inverse_rl.py
- test_meta_learning.py
- test_gru_curriculum.py
- test_enhanced_prioritized_replay.py
- test_n_step_replay_buffer.py
- test_disk_replay_buffer.py
- test_distributed_experience.py
- test_policy_lag.py
- test_hybrid_adaptation.py
- test_multi_period_trainer.py
- test_rl_trading_system.py

### 4. Optimisation (tests/optimization/)

- test_hyperparameter_optimizer.py
- test_threading_optimizer.py
- test_bayesian_optimizer.py
- test_system_optimizer.py
- test_ray_rllib_optimizer.py
- test_bayesian_optimization.py
- test_deepspeed_optimizer.py
- test_gradient_accumulation.py
- test_hogwild.py

### 5. Risk Management (tests/risk/)

- test_risk_manager.py
- test_advanced_risk_manager.py
- test_risk_management.py
- test_atr_stop_loss.py
- test_var_tests.py

### 6. Exécution et Ordres (tests/execution/)

- test_execution.py
- test_order_system.py
- test_order_flow.py
- test_smart_routing.py
- test_market_constraints.py

### 7. Performance et Optimisation (tests/performance/)

- test_benchmark.py
- test_profiling.py
- test_critical_operations.py
- test_model_offloading.py
- test_efficient_checkpointing.py
- test_intel_optimizations.py
- test_rtx_optimization.py
- test_compressed_storage.py
- test_variable_batch.py
- test_jit_compilation.py
- test_mixed_precision.py
- test_lazy_loading.py
- test_frame_compression.py
- test_distributed_training.py
- test_operation_time_reduction.py
- test_model_pruning.py
- test_state_cache.py

### 8. Collecte et Prétraitement des Données (tests/data/)

- test_enhanced_collector.py
- test_enhanced_preprocessor.py
- test_alternative_data_collector.py
- test_orderbook_collector.py
- test_data_integration.py

### 9. Tests Divers (tests/misc/)

- test_train.py
- test_evaluation.py
- test_visualization_paths.py
- test_complete_allocation_system.py

## Nouveaux tests créés

### ML - Backtesting

- **test_survivorship_bias.py** : Tests pour le module de gestion du biais de survivance
  - Correction du biais de survivance
  - Détection des régimes de marché
  - Validation croisée adaptée aux séries temporelles
  - Méthodes de bootstrap pour l'analyse de robustesse
  - Tests statistiques

- **test_advanced_backtesting.py** : Tests pour le backtesting avancé
  - Simulation réaliste avec modèle d'exécution d'ordres
  - Gestion fine des slippages basée sur la liquidité historique
  - Modélisation précise des coûts de transaction
  - Stress testing avec scénarios de crise personnalisés
  - Analyse de sensibilité aux paramètres clés

### ML - Trading Signals

- **test_trading_signals.py** : Tests pour le générateur de signaux de trading
  - Génération de signaux d'achat/vente basés sur des indicateurs techniques
  - Filtrage des signaux selon leur qualité et fiabilité
  - Système de scoring pour hiérarchiser les signaux
  - Intégration avec les prédictions LLM

- **test_multi_timeframe_signals.py** : Tests pour les signaux multi-timeframes
  - Analyse technique multi-échelles (1m, 5m, 15m, 1h, 4h, 1j)
  - Système de confirmation croisée entre timeframes
  - Filtrage intelligent des faux signaux basé sur la volatilité
  - Priorisation des signaux selon leur cohérence multi-temporelle
  - Détection de divergences significatives entre timeframes

## État de la migration

La migration est terminée. Tous les fichiers de test ont été déplacés dans leurs dossiers respectifs et de nouveaux tests ont été créés pour les modules récemment développés.

## Avantages de cette organisation

- **Meilleure lisibilité** : Trouver rapidement les tests pertinents
- **Organisation modulaire** : Cohérence avec la structure du code source
- **Maintenance simplifiée** : Tests regroupés par fonctionnalité
- **Exécution ciblée** : Possibilité d'exécuter des tests par domaine 

## Exécution des tests

Pour exécuter tous les tests :
```
pytest
```

Pour exécuter les tests d'un module spécifique :
```
pytest tests/ml/
pytest tests/rl/
pytest tests/llm/
```

Pour exécuter les tests d'un sous-module spécifique :
```
pytest tests/ml/backtesting/
```

Pour exécuter un test spécifique :
```
pytest tests/ml/backtesting/test_survivorship_bias.py
``` 