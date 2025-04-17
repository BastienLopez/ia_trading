# Test globaux IA:

python -m ai_trading.tests.test_enhanced_collector
python -m ai_trading.tests.test_enhanced_preprocessor
python -m ai_trading.tests.test_enhanced_news_analyzer
python -m ai_trading.tests.test_social_analyzer
python -m ai_trading.tests.test_sentiment_integration

# Test + warning

pytest ai_trading/tests/test_enhanced_collector.py -v
pytest ai_trading/tests/test_enhanced_preprocessor.py -v
pytest ai_trading/tests/test_enhanced_news_analyzer.py -v
pytest ai_trading/tests/test_social_analyzer.py -v
pytest ai_trading/tests/test_sentiment_integration.py -v

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test RL PASSED OR FAILED en ligne
```bash	
python -m pytest ai_trading/tests/ -v
```

# Exécuter tous les tests RL
```bash
python -m unittest discover -s ai_trading/tests
```

# Test RL3.1.1+ PASSED OR FAILED en ligne
```bash 
python -m pytest ai_trading/tests/test_trading_environment.py -v
```

# Test all indicateurs PASSED OR FAILED en ligne
```bash
python -m pytest ai_trading/tests/test_technical_indicators.py -v
```

# Test optimiseur d'hyperparamètres
```bash
python -m tests.test_hyperparameter_optimizer
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exemples disponibles

### 1. Environnement de Trading avec Actions Aléatoires
```bash
python ai_trading/examples/test_trading_env.py --episodes 10 --visualize
```

### 2. Analyse de Sentiment Améliorée
```bash
python ai_trading/examples/enhanced_sentiment_analysis_example.py --coins bitcoin ethereum --days 14 --plot
```

### 3. Pipeline de Données Amélioré
```bash
python ai_trading/examples/enhanced_data_pipeline.py --symbol ETH --days 60 --interval 1d --output ethereum_data.csv
```

### 4. Intégration de Données pour l'Apprentissage par Renforcement
```bash
python ai_trading/examples/rl_data_integration_example.py
```

### 5. Exemple d'Entraînement par Renforcement
```bash
python ai_trading/examples/rl_training_example.py --episodes 500 --symbol ETH --model ppo --save
```

### 6. Test de Stop Loss basé sur ATR
```bash
python ai_trading/examples/test_atr_stop_loss.py --symbol ETH --period 21 --multiplier 2.5 --visualize
```

### 7. Optimisation d'Hyperparamètres pour agents de Trading
```bash
python ai_trading/examples/hyperparameter_optimization_example.py --episodes 50 --symbol BTC --agent sac --save
```

### 8. Optimisation d'Hyperparamètres pour agents GRU
```bash
python ai_trading/examples/hyperparameter_optimization_example.py --episodes 50 --symbol ETH --agent gru_sac --save
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
