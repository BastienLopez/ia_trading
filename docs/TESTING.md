# Guide de Test

Ce document décrit comment exécuter et maintenir les tests du projet.

## Structure des Tests

```
tests/
├── ai_trading/           # Tests du module de trading
│   ├── test_data.py     # Tests du traitement des données
│   ├── test_models.py   # Tests des modèles d'IA
│   └── test_api.py      # Tests de l'API
└── conftest.py          # Configuration des tests
```

## Exécution des Tests

### Tous les Tests

```bash
python -m pytest
```

### Tests avec Couverture

```bash
python -m pytest --cov=ai_trading --cov-report=term-missing
```

### Tests Spécifiques

Pour exécuter les tests d'un module particulier :

```bash
# Tests du module de trading
python -m pytest tests/ai_trading/

# Tests de l'API
python -m pytest tests/ai_trading/test_api.py
```

## Configuration des Tests

Les tests utilisent des mocks pour les services externes (API de change, base de données).

### Configuration de l'Environnement

Les tests nécessitent un fichier `.env` avec les variables suivantes :

```env
API_KEY=test_key
API_SECRET=test_secret
DATABASE_URL=postgresql://user:password@localhost:5432/test_db
LOG_LEVEL=DEBUG
```

## Bonnes Pratiques

1. **Tests Unitaires**
   - Chaque fonction doit avoir son test
   - Utiliser des mocks pour les dépendances externes
   - Tester les cas d'erreur

2. **Tests d'Intégration**
   - Tester les interactions entre les modules
   - Utiliser une base de données de test
   - Nettoyer les données après les tests

3. **Tests de Performance**
   - Mesurer le temps d'exécution
   - Vérifier l'utilisation de la mémoire
   - Tester avec de grandes quantités de données

## Maintenance des Tests

- Mettre à jour les tests lors de l'ajout de nouvelles fonctionnalités
- Vérifier la couverture de code régulièrement
- Documenter les cas d'utilisation complexes 