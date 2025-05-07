# Guide de Compatibilité Cross-Plateforme

Ce document présente les bonnes pratiques et les considérations importantes pour assurer que le projet AI Trading fonctionne correctement sur différentes plateformes (Windows, Linux, et dans des conteneurs Docker).

## Table des matières

1. [Principes généraux](#principes-généraux)
2. [Gestion des chemins de fichiers](#gestion-des-chemins-de-fichiers)
3. [Dépendances et environnement](#dépendances-et-environnement)
4. [Compatibilité TensorFlow](#compatibilité-tensorflow)
5. [Gestion GPU/CPU](#gestion-gpucpu)
6. [Tests et robustesse](#tests-et-robustesse)
7. [Wrappers de compatibilité](#wrappers-de-compatibilité)

## Principes généraux

- **Compatibilité par défaut** : Le code doit fonctionner sur toutes les plateformes sans modification.
- **Détection dynamique** : Détecter les fonctionnalités disponibles plutôt que de coder en dur les différences.
- **Dégradation gracieuse** : Si une fonctionnalité n'est pas disponible, fournir une alternative ou un fallback.
- **Transparence** : Journaliser clairement quand une fonctionnalité spécifique à une plateforme est utilisée.

## Gestion des chemins de fichiers

### Utiliser `pathlib.Path`

Toujours utiliser `pathlib.Path` pour la gestion des chemins de fichiers :

```python
from pathlib import Path

# Bon exemple
data_dir = Path("ai_trading") / "data" / "market"
log_file = data_dir / "logs" / "trading.log"

# Éviter
bad_path = "ai_trading\\data\\market"  # Spécifique à Windows
```

### Utiliser `os.path.join` si nécessaire

Dans les cas où `pathlib` n'est pas disponible :

```python
import os

# Alternative acceptable
data_dir = os.path.join("ai_trading", "data", "market")
```

### Chemins absolus vs relatifs

- Préférer les chemins relatifs par rapport au répertoire du projet
- Utiliser `__file__` pour les chemins relatifs au fichier courant

```python
import os
from pathlib import Path

# Chemin relatif au fichier courant
current_dir = Path(__file__).parent
project_root = current_dir.parent
```

## Dépendances et environnement

### Versions de packages compatibles

- Spécifier les versions minimales compatibles avec toutes les plateformes
- Fournir des alternatives quand une dépendance n'est pas disponible

### Installation conditionnelle

Utiliser des scripts d'installation spécifiques à la plateforme :

- `setup_env.bat` pour Windows
- `setup_env.sh` pour Linux
- `Dockerfile` pour Docker

### Variables d'environnement

Utiliser les variables d'environnement pour la configuration :

```python
import os

# Charger une configuration spécifique à la plateforme
debug_mode = os.environ.get("AI_TRADING_DEBUG", "0").lower() in ("1", "true", "yes")
run_slow_tests = os.environ.get("RUN_SLOW_TESTS", "0").lower() in ("1", "true", "yes")
```

## Compatibilité TensorFlow

### Différences d'API entre versions

Comme nous l'avons vu avec `register_keras_serializable`, certaines API changent entre les versions. Utiliser des wrappers de compatibilité :

```python
def get_register_keras_serializable(package=None):
    """Retourne le décorateur register_keras_serializable compatible avec la version TF actuelle."""
    if hasattr(tf.keras, 'saving') and hasattr(tf.keras.saving, 'register_keras_serializable'):
        # Versions plus récentes de TensorFlow
        if package:
            return tf.keras.saving.register_keras_serializable(package=package)
        else:
            return tf.keras.saving.register_keras_serializable()
    elif hasattr(tf.keras.utils, 'register_keras_serializable'):
        # Versions plus anciennes de TensorFlow
        if package:
            return tf.keras.utils.register_keras_serializable(package=package)
        else:
            return tf.keras.utils.register_keras_serializable()
    else:
        # Fallback
        logger.warning("register_keras_serializable n'est pas disponible.")
        def identity(cls):
            return cls
        return identity
```

### Initialisation TensorFlow

Initialiser TensorFlow de manière robuste :

```python
def initialize_tensorflow():
    """Initialise TensorFlow avec des paramètres adaptés à la plateforme."""
    import tensorflow as tf
    
    # Limiter l'utilisation de mémoire GPU
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU(s) détectés: {len(gpus)}")
        else:
            logger.info("Aucun GPU détecté, utilisation du CPU uniquement")
    except Exception as e:
        logger.warning(f"Erreur lors de la configuration GPU: {e}")
```

## Gestion GPU/CPU

### Détection du matériel disponible

```python
def get_available_devices():
    """Détecte les périphériques disponibles pour le calcul."""
    import tensorflow as tf
    
    devices = []
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            devices.append(f"GPU: {gpu.name}")
        
        cpus = tf.config.list_physical_devices('CPU')
        for cpu in cpus:
            devices.append(f"CPU: {cpu.name}")
    except Exception as e:
        logger.warning(f"Erreur lors de la détection des périphériques: {e}")
        devices.append("CPU: Fallback")
    
    return devices
```

### Fallback CPU

```python
def train_with_fallback(model, data, use_gpu=True):
    """Entraîne un modèle avec fallback sur CPU si nécessaire."""
    import tensorflow as tf
    
    original_device = None
    
    if use_gpu:
        try:
            # Tenter d'utiliser le GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Sauvegarder la stratégie originale
                original_device = tf.config.get_visible_devices()
                # Utiliser le GPU
                logger.info("Utilisation du GPU pour l'entraînement")
            else:
                logger.info("Aucun GPU disponible, utilisation du CPU")
        except Exception as e:
            logger.warning(f"Erreur lors de la configuration GPU: {e}")
            logger.info("Fallback sur CPU")
    
    try:
        # Entraîner le modèle
        history = model.fit(data)
        return history
    finally:
        # Restaurer la configuration originale si nécessaire
        if original_device is not None:
            try:
                tf.config.set_visible_devices(original_device)
            except Exception:
                pass
```

## Tests et robustesse

### Tests conditionnels

Utiliser des décorateurs pour ignorer les tests sur des fonctionnalités non disponibles :

```python
import unittest
import pytest

def is_gpu_available():
    """Vérifie si un GPU est disponible."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except:
        return False

@pytest.mark.skipif(not is_gpu_available(), reason="Nécessite un GPU")
def test_gpu_optimization():
    # Test qui nécessite un GPU
    pass
```

### Tests robustes

Utiliser des assertions souples pour les tests impliquant des ressources spécifiques à la plateforme :

```python
def test_model_performance():
    # Au lieu de vérifier une valeur exacte
    # assert model.evaluate(data) == 0.95
    
    # Utiliser une assertion plus souple
    performance = model.evaluate(data)
    assert performance > 0.9, f"Performance insuffisante: {performance}"
```

## Wrappers de compatibilité

### Fonctions de compatibilité

Implémenter des wrappers pour les fonctionnalités spécifiques à la plateforme :

```python
def get_memory_usage():
    """Retourne l'utilisation mémoire d'une manière compatible cross-plateforme."""
    import os
    
    try:
        # Approche spécifique à Linux
        if os.name == 'posix':
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Approche spécifique à Windows
        elif os.name == 'nt':
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        
        # Fallback générique
        else:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
    except Exception:
        return None
```

### Classes avec détection de plateforme

```python
class PlatformOptimizer:
    """Optimiseur avec des stratégies spécifiques à la plateforme."""
    
    def __init__(self):
        import platform
        self.system = platform.system().lower()
        self.is_windows = self.system == 'windows'
        self.is_linux = self.system == 'linux'
        
    def optimize_for_platform(self, model):
        """Applique des optimisations spécifiques à la plateforme."""
        if self.is_windows:
            # Optimisations spécifiques à Windows
            return self._optimize_for_windows(model)
        elif self.is_linux:
            # Optimisations spécifiques à Linux
            return self._optimize_for_linux(model)
        else:
            # Optimisations génériques
            return self._optimize_generic(model)
            
    def _optimize_for_windows(self, model):
        # Implémentation pour Windows
        pass
        
    def _optimize_for_linux(self, model):
        # Implémentation pour Linux
        pass
        
    def _optimize_generic(self, model):
        # Implémentation générique
        pass
```

## Conclusion

En suivant ces principes et techniques, vous pouvez assurer que le code du projet AI Trading fonctionne efficacement sur toutes les plateformes supportées. L'objectif est de maximiser la compatibilité tout en tirant parti des fonctionnalités spécifiques à chaque plateforme lorsqu'elles sont disponibles. 