# Optimisation check 
```bash	
python -m ai_trading.optim.check_all_optimizations --check-all-opti
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dashboard 
```bash	
python -m ai_trading.dashboard.run
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test RL PASSED OR FAILED en ligne
```bash	
python -m pytest ai_trading/tests/ -v -rs 

python -m pytest ai_trading/utils/tests/ -v -rs 

python -m pytest ai_trading/rl/tests/ -v -rs 
```
# Stop si erreurs 
```bash	
python -m pytest ai_trading/tests/ -v -x
```
# Skip test long
```bash	
python -m pytest ai_trading/tests/ -v --skip-slow
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test all indicateurs PASSED OR FAILED en ligne
```bash
python -m pytest ai_trading/tests/test_technical_indicators.py -v
```

# Test optimiseur d'hyperparamètres
```bash
python -m ai_trading.tests.test_hyperparameter_optimizer -v
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exemples disponibles

### 1. Environnement de Trading avec Actions Aléatoires
```bash
python -m ai_trading.examples.test_trading_env
```

### 2. Analyse de Sentiment Améliorée
```bash
python -m ai_trading.examples.enhanced_sentiment_analysis_example --coins bitcoin --days 7 --plot
```

### 3. Pipeline de Données Amélioré
```bash
python -m ai_trading.examples.enhanced_data_pipeline --symbol ETH --days 30 --interval 1d --output ethereum_data.csv
```

### 4. Intégration de Données pour l'Apprentissage par Renforcement
```bash
python -m ai_trading.examples.rl_data_integration_example
```

### 5. Exemple d'Entraînement par Renforcement
```bash
python -m ai_trading.examples.rl_training_example --episodes 100 --symbol ETH --model sac --save
```

### 6. Exemple avancé de fonctions de récompense
```bash
python -m ai_trading.examples.advanced_rewards_example
```

### 7. Optimisation d'Hyperparamètres pour agents de Trading
```bash
python -m ai_trading.examples.hyperparameter_optimization_example --episodes 20 --symbol BTC --agent sac --save
```

### 8. Optimisation d'Hyperparamètres pour agents GRU
```bash
python -m ai_trading.examples.hyperparameter_optimization_example --episodes 20 --symbol ETH --agent gru_sac --save
```

## Diagnostics et débogage

1. **Erreur avec `get_crypto_news(coin=coin)`**:
   - Message d'erreur: `TypeError: EnhancedDataCollector.get_crypto_news() got an unexpected keyword argument 'coin'`
   - Solution: Retirez le paramètre `coin` et utilisez simplement `get_crypto_news(limit=10)`

2. **Erreur avec `integrate_data(lookback_window=10)`**:
   - Message d'erreur: `TypeError: RLDataIntegrator.integrate_data() got an unexpected keyword argument 'lookback_window'`
   - Solution: Retirez le paramètre `lookback_window` de l'appel

3. **Erreur avec `fetch_crypto_data` ou `fetch_crypto_news`**:
   - Solution: Utilisez `get_merged_price_data` et `get_crypto_news` à la place

4. **Erreur avec la conversion de données NumPy vers TensorFlow dans l'optimisation d'hyperparamètres**:
   - Message d'erreur: `ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float)`
   - Solution: Vérifiez que les données d'entrée pour les réseaux de neurones sont bien formatées (utilisez `np.asarray()` ou `tf.convert_to_tensor()`)

5. **Attribut manquant `portfolio_value` dans TradingEnvironment**:
   - Message d'erreur: `AttributeError: 'TradingEnvironment' object has no attribute 'portfolio_value'`
   - Solution: Assurez-vous que l'environnement initialise correctement la valeur du portefeuille dans sa méthode `reset()`

6. **Avertissements concernant des méthodes dépréciées**:
   - Problème: Utilisation de `df.fillna(method='bfill')` qui est déprécié
   - Solution: Remplacez par `df.bfill()` pour le backfill ou `df.ffill()` pour le forward fill

## Test des exemples d'optimisation

Ce document décrit comment tester les différents exemples d'optimisation implémentés dans le projet.

### Optimisation des threads

Pour tester l'optimisation des threads:

```bash
python -m ai_trading.utils.threading_optimizer --test
```

Cela affichera les paramètres optimaux pour votre système.

### Optimisation du système

Pour appliquer les optimisations système:

```bash
python -m ai_trading.utils.system_optimizer --apply
```

Pour vérifier le statut des optimisations:

```bash
python -m ai_trading.utils.system_optimizer --status
```

### Optimisation des performances de PyTorch

Pour tester les optimisations spécifiques à PyTorch:

```bash
python -m ai_trading.examples.rtx_optimization_example
```

### Optimisation de la mémoire

Pour tester l'optimisation de la mémoire:

```bash
python -m ai_trading.examples.model_offloading_example
```

### Optimisation des checkpoints

Pour tester l'optimisation des checkpoints:

```bash
python -m ai_trading.examples.efficient_checkpointing_example
```

### Optimisation des hyperparamètres (Recherche par grille)

Pour tester l'optimisation des hyperparamètres avec recherche par grille:

```bash
python -m ai_trading.examples.hyperparameter_optimization_example --episodes 20 --symbol BTC --agent sac --save
```

Pour l'agent GRU-SAC:

```bash
python -m ai_trading.examples.hyperparameter_optimization_example --episodes 20 --symbol ETH --agent gru_sac --save
```

### Optimisation bayésienne des hyperparamètres

Pour tester l'optimisation bayésienne des hyperparamètres:

```bash
python -m ai_trading.examples.bayesian_optimization_example --episodes 20 --symbol BTC --agent sac --save
```

Options avancées:

```bash
# Optimisation bayésienne avec plus d'exploration
python -m ai_trading.examples.bayesian_optimization_example --agent sac --exploration 0.05 --initial-points 10 --iterations 20

# Optimisation bayésienne multi-objectifs
python -m ai_trading.examples.bayesian_optimization_example --agent gru_sac --multi-objective --save
```

### Optimisation de la précision mixte

Pour tester l'entraînement en précision mixte:

```bash
python -m ai_trading.examples.mixed_precision_example
```

### Optimisation de la quantization des modèles

Pour tester la quantization des modèles:

```bash
python -m ai_trading.examples.model_quantization_example
```

### Optimisation de la compilation JIT

Pour tester la compilation JIT:

```bash
python -m ai_trading.examples.jit_compilation_example
```

### Optimisation du multiprocessing

Pour tester l'optimisation du multiprocessing:

```bash
python -m ai_trading.examples.multiprocessing_optimization_example
```

### Optimisation avec Ray Tune

Pour tester l'optimisation avec Ray Tune:

```bash
python -m ai_trading.examples.ray_tune_example
```

### Optimisation de l'accumulation de gradients

Pour tester l'optimisation de l'accumulation de gradients:

```bash
python -m ai_trading.examples.gradient_accumulation_example
```

### Tests de performance

Pour exécuter les tests de performance:

```bash
python -m ai_trading.examples.performance_test_example
```

### Profiling

Pour exécuter le profiling:

```bash
python -m ai_trading.examples.profiling_example
```
