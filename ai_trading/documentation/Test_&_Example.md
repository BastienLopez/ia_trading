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
# Tests du système de trading RL
python -m unittest ai_trading/tests/test_trading_environment.py
python -m unittest ai_trading.tests.test_trading_environment.TestTradingEnvironment.test_step_buy
python -m unittest ai_trading/tests/test_dqn_agent.py
python -m unittest ai_trading.tests.test_dqn_agent.TestDQNAgent.test_replay
python -m unittest ai_trading.tests.test_data_integration.TestDataIntegration.test_generate_synthetic_data

python -m unittest ai_trading.tests.test_evaluation.TestEvaluation.test_performance_metrics
python -m unittest ai_trading.tests.test_train.TestTrain.test_early_stopping
python -m unittest ai_trading/tests/test_data_integration.py
python -m unittest ai_trading/tests/test_evaluation.py

python -m unittest ai_trading/tests/test_train.py
python -m unittest ai_trading/tests/test_rl_trading_system.py
python -m unittest ai_trading.tests.test_rl_trading_system.TestRLTradingSystem.test_train

# Exécuter tous les tests RL
python -m unittest discover -s ai_trading/tests

# Test PASSED OR FAILED en ligne
python -m pytest ai_trading/tests/ -v

# Tests RL + warning
pytest ai_trading/tests/test_trading_environment.py -v
pytest ai_trading/tests/test_dqn_agent.py -v
pytest ai_trading/tests/test_data_integration.py -v
pytest ai_trading/tests/test_train.py -v
pytest ai_trading/tests/test_evaluation.py -v
pytest ai_trading/tests/test_rl_trading_system.py -v
