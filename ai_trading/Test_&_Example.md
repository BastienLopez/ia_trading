### Tests
# Exécution des tests pour le collecteur de données
```bash
python -m ai_trading.tests.test_enhanced_collector
```

# Exécution des tests pour le préprocesseur
```bash
python -m ai_trading.tests.test_enhanced_preprocessor
```

# Exécution des tests pour l'analyseur de sentiment amélioré
```bash
python -m ai_trading.tests.test_enhanced_news_analyzer
```

```bash
python -m ai_trading.tests.test_social_analyzer
```

### Exemples
```bash
python -m ai_trading.examples.social_sentiment_example
```

## Tests Unitaires et d'Intégration

**Tests Principaux :**
- ✅ Analyse de sentiment basique sur un article unique
- ✅ Traitement par lots de 50+ articles avec cache
- 🐛 Test de résilience avec données corrompues
- ✅ Vérification de la colonne `sentiment_score` générée
- 🧪 Test d'intégration complète avec rapport PDF

**Scénarios de Test :**
```python
# Test de charge extrême
def test_charge_max():
    analyzer = EnhancedNewsAnalyzer()
    fake_news = [{"title": "Test", "body": "Content"}] * 1000
    df = analyzer.analyze_news(fake_news)
    assert len(df) == 1000
```

## Exemples d'Utilisation

**Analyse Simple :**
```python
from ai_trading.llm.sentiment_analysis import EnhancedNewsAnalyzer

analyzer = EnhancedNewsAnalyzer()
news = [{"title": "Bitcoin ATH", "body": "Bitcoin reaches new high..."}]
df = analyzer.analyze_news(news)
print(df[['title', 'sentiment_score']])
```

**Génération de Rapport :**
```python
report = analyzer.generate_report(df)
print(f"Sentiment moyen : {report['average_sentiment']:.2f}")
print(f"Article plus positif : {report['most_positive_article']['title']}")
```

**Visualisation :**
```python
analyzer.plot_trends(df, "trends.png")
```

**Configuration Avancée :**
```python
custom_analyzer = EnhancedNewsAnalyzer(
    sentiment_model="cardiffnlp/twitter-roberta-base-sentiment",
    cache_dir="custom_cache/",
    use_gpu=True
)
``` 