# Tests des Indicateurs Techniques

# Test all indicateurs PASSED OR FAILED en ligne
```bash
python -m pytest ai_trading/tests/test_technical_indicators.py -v
```
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_ema -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_momentum -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_atr -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_obv -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_mfi -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_rsi -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_cci -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_fibonacci_levels -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_pivots -v
python -m pytest ai_trading/tests/test_data_integration.py::TestDataIntegration::test_technical_indicators_integration -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_performance -v

python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_get_all_indicators -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_volume_average -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_stochastic -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_adx -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_bollinger_bands -v
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_macd -v

## 3. Tests Fonctionnels

### Visualisation des Indicateurs
```bash
python ai_trading/examples/visualize_indicators.py
```
- Générer des graphiques pour chaque indicateur
- Vérifier visuellement la cohérence des calculs
- Comparer avec des outils de trading standard

### Stratégies Basiques avec Indicateurs
```bash
python ai_trading/examples/basic_indicator_strategies.py
```
- Tester des stratégies simples basées sur les indicateurs (ex: croisement de MACD)
- Évaluer la performance de ces stratégies sur des données historiques
- Comparer avec des stratégies de référence

# Test d'integration
```bash
   python ai_trading/examples/test_trading_env.py
```

## 4. Commandes de Test Complètes

### Exécuter les tests avec couverture de code
```bash
python -m pytest ai_trading/tests/test_technical_indicators.py --cov=ai_trading.rl.technical_indicators -v
```

### Exécuter les tests de performance
```bash
python -m pytest ai_trading/tests/test_technical_indicators.py::TestTechnicalIndicators::test_performance -v
``` 