# Test simple:

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


# Formatage et vérification du code

black ai_trading/                                                # Formatage du code
autoflake --in-place --remove-all-unused-imports --recursive ai_trading/  # Suppression des imports inutilisés
isort ai_trading/                                               # Organisation des imports
python fix_fstrings_simple.py ai_trading                        # Correction des f-strings sans placeholders
flake8 ai_trading/                                              # Vérification finale du style
