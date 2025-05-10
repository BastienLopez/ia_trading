# Fonctionnalités Transversales pour AI Trading

Ce répertoire contient les fonctionnalités transversales pour le module ai_trading du projet de trading crypto.

## Journalisation Avancée

Le module de journalisation avancée (`advanced_logging.py`) offre les fonctionnalités suivantes :

- Configuration centralisée des loggers avec différents niveaux et formats
- Rotation automatique des fichiers de logs
- Niveau de log personnalisé TRACE (plus détaillé que DEBUG)
- Formats de logs variés (texte standard, détaillé, JSON)
- Décorateurs pour la journalisation des exceptions et des temps d'exécution
- Capture des exceptions non gérées

### Utilisation

```python
from ai_trading.utils.advanced_logging import get_logger, log_exceptions, log_execution_time

# Obtenir un logger configuré
logger = get_logger("mon_module")

# Utiliser le logger
logger.debug("Message de débogage")
logger.info("Information importante")

# Utiliser les décorateurs
@log_exceptions(logger)
def fonction_avec_gestion_exceptions():
    # Code qui peut lever des exceptions
    pass

@log_execution_time(logger)
def fonction_avec_mesure_temps():
    # Code dont on veut mesurer le temps d'exécution
    pass
```

## Suivi des Performances

Le module de suivi des performances (`performance_logger.py`) permet de :

- Collecter des métriques système en temps réel (CPU, mémoire, GPU)
- Suivre les temps d'exécution des différentes parties du code
- Générer des rapports de performance
- Sauvegarder les métriques pour analyse ultérieure

### Utilisation

```python
from ai_trading.utils.performance_logger import start_metrics_collection, stop_metrics_collection, get_performance_tracker

# Démarrer la collecte des métriques système
collector = start_metrics_collection(interval=5.0)  # Toutes les 5 secondes

# Tracker pour mesurer des performances spécifiques
tracker = get_performance_tracker("mon_tracker")

# Mesurer le temps d'une opération
tracker.start("operation_1")
# ... code à mesurer ...
tracker.stop("operation_1")

# Arrêter la collecte des métriques
stop_metrics_collection()
```

## Gestion des Checkpoints

Le module de gestion des checkpoints (`checkpoint_manager.py`) permet de :

- Sauvegarder et charger l'état des modèles et des sessions d'entraînement
- Gérer les métadonnées associées (performances, configurations, timestamps)
- Effectuer une rotation automatique des checkpoints
- Compresser les fichiers pour économiser de l'espace disque

### Utilisation

```python
from ai_trading.utils.checkpoint_manager import get_checkpoint_manager, CheckpointType

# Obtenir le gestionnaire de checkpoints
checkpoint_manager = get_checkpoint_manager()

# Sauvegarder un modèle
model_id = checkpoint_manager.save_model(
    model=mon_modele,
    name="modele_trading",
    description="Modèle de trading après 100 époques",
    metrics={"accuracy": 0.85, "loss": 0.32}
)

# Sauvegarder l'état d'une session
session_id = checkpoint_manager.save_checkpoint(
    obj=etat_session,
    type=CheckpointType.SESSION,
    prefix="session_trading",
    description="État de la session après interruption"
)

# Charger un modèle
checkpoint_manager.load_model(model_id, mon_modele)

# Charger un checkpoint
etat_session = checkpoint_manager.load_checkpoint(session_id)
```

## Tests de Performance

Le module de tests de performance (`tests/test_performance/`) permet d'évaluer les performances des fonctionnalités transversales :

- Tests unitaires mesurant l'impact des fonctionnalités (`test_units.py`)
- Benchmarks précis pour comparer les implémentations (`benchmark.py`)
- Outils de visualisation des résultats (`visualization.py`)
- Utilitaires de profilage pour l'analyse détaillée (`profile_utils.py`)

### Utilisation

```python
# Exécuter les tests unitaires de performance
python -m unittest ai_trading.tests.test_performance.test_units

# Exécuter les benchmarks
python -m ai_trading.tests.test_performance.benchmark --output results.json

# Visualiser les résultats
python -m ai_trading.tests.test_performance.visualization results.json --output graph.png

# Générer un rapport HTML
python -m ai_trading.tests.test_performance.visualization results.json --report report.html

# Comparer deux benchmarks
python -m ai_trading.tests.test_performance.visualization benchmark1.json --compare benchmark2.json
```

## Exemples

Des exemples d'utilisation sont disponibles dans le répertoire `ai_trading/examples/` :

- `advanced_logging_example.py` : Démontre l'utilisation de la journalisation avancée
- `performance_metrics_example.py` : Montre comment collecter et analyser des métriques de performance
- `checkpoint_example.py` : Illustre la gestion des checkpoints
- `training_with_checkpoints.py` : Combine toutes les fonctionnalités dans un scénario d'entraînement
- `performance_test_example.py` : Démontre l'utilisation des tests de performance 