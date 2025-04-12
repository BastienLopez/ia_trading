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

# Exemples 
```bash 
python ai_trading/examples/enhanced_data_pipeline.py
python ai_trading/examples/enhanced_sentiment_analysis_example.py
python ai_trading/examples/rl_data_integration_example.py
python ai_trading/examples/rl_training_example.py
python ai_trading/examples/test_atr_stop_loss.py
python ai_trading/examples/test_trading_env.py
```